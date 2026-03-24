import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def sinusoidal_time_emb(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    ang = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 64, out: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out),
            nn.SiLU(),
            nn.Linear(out, out),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_time_emb(t, self.dim))


class ResBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.05):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


class CondFlowNet(nn.Module):
    """
    v_theta(x_t, cond, t) -> velocity for scalar target.
    x_t:  (B,1)
    cond: (B,l)
    t:    (B,)
    """
    def __init__(
        self,
        cond_dim: int,
        hidden: int = 256,
        depth: int = 4,
        t_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.t_embed = TimeEmbed(dim=t_dim, out=64)

        in_dim = cond_dim + 1 + 64
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            ResBlock(hidden=hidden, dropout=dropout) for _ in range(depth)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_embed(t)
        h = self.in_proj(torch.cat([x_t, cond, te], dim=1))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


def fm_batch_loss(model: nn.Module, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """
    Linear path:
      x_t = (1-t) x0 + t x1,  x0 ~ N(0,1)
      v_target = x1 - x0
    """
    B = x1.shape[0]
    t = torch.rand((B,), device=x1.device).clamp(1e-4, 1.0 - 1e-4)
    x0 = torch.randn_like(x1)
    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    v_target = x1 - x0
    v_pred = model(x_t, cond, t)
    return torch.mean((v_pred - v_target) ** 2)


def fm_batch_loss_pnl(
    model: nn.Module,
    x1: torch.Tensor,
    cond: torch.Tensor,
    x1_real: torch.Tensor,
    lambda_pnl: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flow matching loss + differentiable PnL auxiliary loss.

    The PnL term pushes the model to produce directionally biased samples,
    fixing the pu~0.5 problem where flow matching faithfully learns a
    symmetric distribution but produces tiny position sizes.

    x1:      normalized target return (B,1)
    cond:    normalized condition (B,l)
    x1_real: denormalized target return (B,1) for PnL computation

    Returns: (total_loss, fm_loss, pnl_loss)
    """
    B = x1.shape[0]
    t = torch.rand((B,), device=x1.device).clamp(1e-4, 1.0 - 1e-4)
    x0 = torch.randn_like(x1)
    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    v_target = x1 - x0
    v_pred = model(x_t, cond, t)

    # Standard velocity MSE
    loss_fm = torch.mean((v_pred - v_target) ** 2)

    # PnL auxiliary: estimate x1 from current (x_t, v_pred)
    # x1_hat = x_t + (1-t) * v_pred  (first-order extrapolation to t=1)
    x1_hat_n = x_t + (1.0 - t)[:, None] * v_pred

    # Linear PnL proxy: x1_hat * x1_real (no tanh, no saturation)
    # Gradient always flows: d(loss)/d(v_pred) = -(1-t) * x1_real
    # Rewards directional correctness weighted by actual return magnitude.
    # Unlike tanh(100*x) which saturates and kills gradients, this has
    # uniform gradient flow everywhere.
    pnl_proxy = x1_hat_n * x1_real
    loss_pnl = -torch.mean(pnl_proxy)  # negative because we maximize PnL

    total = loss_fm + lambda_pnl * loss_pnl
    return total, loss_fm, loss_pnl


@torch.no_grad()
def fm_sample_from_x0(model: nn.Module, x0: torch.Tensor, cond: torch.Tensor, n_steps: int = 40) -> torch.Tensor:
    """
    Solve dx/dt = v_theta(x, cond, t) from t=0 -> 1 with Euler.
    We evaluate t at the middle of each step for a slightly better integrator.
    """
    B = cond.shape[0]
    x = x0
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), (k + 0.5) * dt, device=cond.device, dtype=torch.float32)
        v = model(x, cond, t)
        x = x + dt * v
    return x


class FlowGenAdapter(nn.Module):
    """
    Drop-in replacement for the FinGAN generator in Evaluation2.

    forward(noise, condition, h_0, c_0) -> (1,B,1)

    Uses:
    - per-lag condition normalization
    - separate target normalization
    """
    def __init__(
        self,
        flow_model: nn.Module,
        mu_cond: torch.Tensor,
        sd_cond: torch.Tensor,
        mu_x: float,
        sd_x: float,
        ode_steps: int = 40,
        pred: int = 1,
    ):
        super().__init__()
        self.flow = flow_model
        self.ode_steps = int(ode_steps)
        self.pred = int(pred)
        if self.pred != 1:
            raise ValueError("This adapter currently supports pred=1 only.")

        mu_cond = torch.as_tensor(mu_cond, dtype=torch.float32).view(1, -1)
        sd_cond = torch.as_tensor(sd_cond, dtype=torch.float32).view(1, -1)

        self.register_buffer("mu_cond", mu_cond)
        self.register_buffer("sd_cond", sd_cond)

        self.mu_x = float(mu_x)
        self.sd_x = float(sd_x) + 1e-8

    def forward(self, noise, condition, h_0=None, c_0=None):
        if condition.dim() != 3 or noise.dim() != 3:
            raise ValueError("Expected condition and noise shapes (1,B,*).")

        cond = condition.squeeze(0)                         # (B,l)
        cond_n = (cond - self.mu_cond) / (self.sd_cond + 1e-8)

        x0 = noise[:, :, 0:1].squeeze(0)                    # (B,1)
        x1_n = fm_sample_from_x0(self.flow, x0, cond_n, n_steps=self.ode_steps)
        x1 = x1_n * self.sd_x + self.mu_x

        return x1.unsqueeze(0)                              # (1,B,1)