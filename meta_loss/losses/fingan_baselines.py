"""
FinGAN 8 baseline loss combinations, adapted for flow matching.

Each loss function follows the interface:
    def loss_fn(v_pred, v_target, x_pred, x_real, condition, epoch) -> Tensor

Where:
    v_pred:    (B,1) predicted velocity field
    v_target:  (B,1) true velocity = x1 - x0
    x_pred:    (B,1) one-step prediction = x0 + v_pred (approx generated sample)
    x_real:    (B,1) true return (normalized)
    condition: (B,l) lookback window (normalized)
    epoch:     int, current training epoch

All losses include velocity_loss (flow matching base) plus FinGAN trading terms.
"""
import torch
import torch.nn.functional as F


def _velocity_loss(v_pred, v_target):
    return F.mse_loss(v_pred, v_target)


def _differentiable_pnl(x_pred, x_real):
    """Smooth PnL: tanh(100*pred) * real, averaged over batch."""
    return torch.mean(torch.tanh(100.0 * x_pred) * x_real)


def _mse_term(x_pred, x_real):
    return F.mse_loss(x_pred, x_real)


def _sharpe_term(x_pred, x_real):
    """Differentiable Sharpe ratio within mini-batch."""
    pnl = torch.tanh(100.0 * x_pred) * x_real
    mu = pnl.mean()
    sd = pnl.std(unbiased=False) + 1e-8
    return mu / sd


def _std_term(x_pred, x_real):
    """Std of differentiable PnL within mini-batch (denominator of SR)."""
    pnl = torch.tanh(100.0 * x_pred) * x_real
    return pnl.std(unbiased=False)


# ──────────────────────────────────────────────────────────────
# 8 FinGAN loss combinations
# Weights default to 1.0; use gradient_norm_balancing() to calibrate
# ──────────────────────────────────────────────────────────────

def fingan_pnl(v_pred, v_target, x_pred, x_real, condition, epoch,
               alpha=1.0):
    """Combo 1: velocity + PnL*"""
    return _velocity_loss(v_pred, v_target) - alpha * _differentiable_pnl(x_pred, x_real)


def fingan_pnl_std(v_pred, v_target, x_pred, x_real, condition, epoch,
                   alpha=1.0, delta=1.0):
    """Combo 2: velocity + PnL* + STD"""
    return (_velocity_loss(v_pred, v_target)
            - alpha * _differentiable_pnl(x_pred, x_real)
            + delta * _std_term(x_pred, x_real))


def fingan_pnl_mse(v_pred, v_target, x_pred, x_real, condition, epoch,
                   alpha=1.0, beta=1.0):
    """Combo 3: velocity + PnL* + MSE"""
    return (_velocity_loss(v_pred, v_target)
            - alpha * _differentiable_pnl(x_pred, x_real)
            + beta * _mse_term(x_pred, x_real))


def fingan_pnl_sr(v_pred, v_target, x_pred, x_real, condition, epoch,
                  alpha=1.0, gamma=1.0):
    """Combo 4: velocity + PnL* + SR*"""
    return (_velocity_loss(v_pred, v_target)
            - alpha * _differentiable_pnl(x_pred, x_real)
            - gamma * _sharpe_term(x_pred, x_real))


def fingan_pnl_mse_std(v_pred, v_target, x_pred, x_real, condition, epoch,
                       alpha=1.0, beta=1.0, delta=1.0):
    """Combo 5: velocity + PnL* + MSE + STD"""
    return (_velocity_loss(v_pred, v_target)
            - alpha * _differentiable_pnl(x_pred, x_real)
            + beta * _mse_term(x_pred, x_real)
            + delta * _std_term(x_pred, x_real))


def fingan_pnl_mse_sr(v_pred, v_target, x_pred, x_real, condition, epoch,
                      alpha=1.0, beta=1.0, gamma=1.0):
    """Combo 6: velocity + PnL* + MSE + SR*"""
    return (_velocity_loss(v_pred, v_target)
            - alpha * _differentiable_pnl(x_pred, x_real)
            + beta * _mse_term(x_pred, x_real)
            - gamma * _sharpe_term(x_pred, x_real))


def fingan_sr(v_pred, v_target, x_pred, x_real, condition, epoch,
              gamma=1.0):
    """Combo 7: velocity + SR*"""
    return _velocity_loss(v_pred, v_target) - gamma * _sharpe_term(x_pred, x_real)


def fingan_sr_mse(v_pred, v_target, x_pred, x_real, condition, epoch,
                  beta=1.0, gamma=1.0):
    """Combo 8: velocity + SR* + MSE"""
    return (_velocity_loss(v_pred, v_target)
            - gamma * _sharpe_term(x_pred, x_real)
            + beta * _mse_term(x_pred, x_real))


def velocity_only(v_pred, v_target, x_pred, x_real, condition, epoch):
    """Baseline 0: pure flow matching (current FlowFMPlus default)."""
    return _velocity_loss(v_pred, v_target)


# ──────────────────────────────────────────────────────────────
# Registry of all baselines
# ──────────────────────────────────────────────────────────────

FINGAN_BASELINES = {
    "velocity_only":     velocity_only,
    "fingan_pnl":        fingan_pnl,
    "fingan_pnl_std":    fingan_pnl_std,
    "fingan_pnl_mse":    fingan_pnl_mse,
    "fingan_pnl_sr":     fingan_pnl_sr,
    "fingan_pnl_mse_std": fingan_pnl_mse_std,
    "fingan_pnl_mse_sr": fingan_pnl_mse_sr,
    "fingan_sr":         fingan_sr,
    "fingan_sr_mse":     fingan_sr_mse,
}


# ──────────────────────────────────────────────────────────────
# Baseline code strings in FinGAN 3-arg interface: loss_fn(gen_out, real, tanh_temp)
# These can be fed directly to LLM as seed examples and into SE-Agent init pool.
# "velocity_only" = no financial loss (return 0), so FinGAN trains with BCE only.
# ──────────────────────────────────────────────────────────────

FINGAN_BASELINE_CODES = {
    "baseline_bce_only": (
        "import torch\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # No financial loss; FinGAN trains with BCE only.\n"
        "    return torch.tensor(0.0, device=gen_out.device, requires_grad=False)\n"
    ),
    "baseline_pnl": (
        "import torch\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize PnL: tanh-smoothed position * realized return\n"
        "    pnl = torch.mean(torch.tanh(tanh_temp * gen_out) * real)\n"
        "    return -pnl\n"
    ),
    "baseline_pnl_std": (
        "import torch\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize PnL + penalize PnL volatility\n"
        "    pnl = torch.tanh(tanh_temp * gen_out) * real\n"
        "    return -torch.mean(pnl) + torch.std(pnl, unbiased=False)\n"
    ),
    "baseline_pnl_mse": (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize PnL + prediction accuracy\n"
        "    pnl = torch.mean(torch.tanh(tanh_temp * gen_out) * real)\n"
        "    mse = F.mse_loss(gen_out, real)\n"
        "    return -pnl + mse\n"
    ),
    "baseline_pnl_sr": (
        "import torch\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize PnL + Sharpe ratio\n"
        "    pnl = torch.tanh(tanh_temp * gen_out) * real\n"
        "    sharpe = pnl.mean() / (pnl.std(unbiased=False) + 1e-8)\n"
        "    return -torch.mean(pnl) - sharpe\n"
    ),
    "baseline_pnl_mse_std": (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize PnL + prediction accuracy + penalize volatility\n"
        "    pnl = torch.tanh(tanh_temp * gen_out) * real\n"
        "    mse = F.mse_loss(gen_out, real)\n"
        "    return -torch.mean(pnl) + mse + torch.std(pnl, unbiased=False)\n"
    ),
    "baseline_pnl_mse_sr": (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize PnL + prediction accuracy + Sharpe\n"
        "    pnl = torch.tanh(tanh_temp * gen_out) * real\n"
        "    sharpe = pnl.mean() / (pnl.std(unbiased=False) + 1e-8)\n"
        "    mse = F.mse_loss(gen_out, real)\n"
        "    return -torch.mean(pnl) + mse - sharpe\n"
    ),
    "baseline_sr": (
        "import torch\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize Sharpe ratio only\n"
        "    pnl = torch.tanh(tanh_temp * gen_out) * real\n"
        "    sharpe = pnl.mean() / (pnl.std(unbiased=False) + 1e-8)\n"
        "    return -sharpe\n"
    ),
    "baseline_sr_mse": (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def loss_fn(gen_out, real, tanh_temp):\n"
        "    # Maximize Sharpe + prediction accuracy\n"
        "    pnl = torch.tanh(tanh_temp * gen_out) * real\n"
        "    sharpe = pnl.mean() / (pnl.std(unbiased=False) + 1e-8)\n"
        "    mse = F.mse_loss(gen_out, real)\n"
        "    return -sharpe + mse\n"
    ),
}
