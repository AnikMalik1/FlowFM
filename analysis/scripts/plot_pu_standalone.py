"""Standalone pu distribution analysis - no FinGAN.py import."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Model definition (copied from flow_adapter.py) ──────────────────────
def sinusoidal_time_emb(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half-1,1))
    ang = t[:,None].float() * freqs[None,:]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

class TimeEmbed(nn.Module):
    def __init__(self, dim=64, out=64):
        super().__init__(); self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out), nn.SiLU(), nn.Linear(out, out), nn.SiLU())
    def forward(self, t): return self.mlp(sinusoidal_time_emb(t, self.dim))

class ResBlock(nn.Module):
    def __init__(self, hidden, dropout=0.05):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
    def forward(self, x): return self.norm(x + self.ff(x))

class CondFlowNet(nn.Module):
    def __init__(self, cond_dim, hidden=256, depth=4, t_dim=64, dropout=0.05):
        super().__init__()
        self.t_embed = TimeEmbed(dim=t_dim, out=64)
        self.in_proj = nn.Sequential(nn.Linear(cond_dim + 1 + 64, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x_t, cond, t):
        h = self.in_proj(torch.cat([x_t, cond, self.t_embed(t)], dim=1))
        for blk in self.blocks: h = blk(h)
        return self.out(h)

@torch.no_grad()
def fm_sample(model, x0, cond, n_steps=40):
    B = cond.shape[0]; x = x0; dt = 1.0/n_steps
    for k in range(n_steps):
        t = torch.full((B,), (k+0.5)*dt, device=cond.device)
        x = x + dt * model(x, cond, t)
    return x

# ── Data loading (manual, no FinGAN import) ─────────────────────────────
DATA = "/projects/s5e/quant/fingan/colabFinGAN/data"
CKPT = "/projects/s5e/quant/fingan/colabFinGAN/Fin-GAN/TrainedModels"
OUT = "/projects/s5e/quant/fingan"

print("Loading data...")
df_s = pd.read_csv(f"{DATA}/AMZN.csv")
df_e = pd.read_csv(f"{DATA}/XLY.csv")  # AMZN's sector ETF

df_s["date_dt"] = pd.to_datetime(df_s["date"])
df_e["date_dt"] = pd.to_datetime(df_e["date"])
df_s = df_s[df_s["date_dt"] < "2022-01-01"].reset_index(drop=True)
df_e = df_e[df_e["date_dt"] < "2022-01-01"].reset_index(drop=True)

mg = pd.merge(df_s[["date_dt","AdjClose","AdjOpen"]], df_e[["date_dt","AdjClose","AdjOpen"]], on="date_dt", suffixes=("_s","_e"))
N = len(mg)
print(f"  Merged dates: {N}")

s_log = np.zeros(2*N); e_log = np.zeros(2*N)
for i in range(N):
    s_log[2*i] = np.log(mg["AdjOpen_s"].iloc[i])
    s_log[2*i+1] = np.log(mg["AdjClose_s"].iloc[i])
    e_log[2*i] = np.log(mg["AdjOpen_e"].iloc[i])
    e_log[2*i+1] = np.log(mg["AdjClose_e"].iloc[i])

excess = np.clip(np.diff(s_log) - np.diff(e_log), -0.15, 0.15)
print(f"  Excess returns: {len(excess)}")

# Sliding windows
l = 10
windows = np.array([list(excess[i:i+l]) + [excess[i+l]] for i in range(len(excess)-l)])
Nw = len(windows)
N_tr = int(0.8*Nw); N_vl = int(0.1*Nw)
train = torch.tensor(windows[:N_tr], dtype=torch.float32)
test = torch.tensor(windows[N_tr+N_vl:], dtype=torch.float32)
print(f"  Train: {train.shape}, Test: {test.shape}")

# Normalization
mu_cond = train[:,:l].mean(dim=0)
sd_cond = train[:,:l].std(dim=0, unbiased=False) + 1e-8
mu_x = float(train[:,l].mean())
sd_x = float(train[:,l].std(unbiased=False)) + 1e-8
print(f"  mu_x={mu_x:.6f}, sd_x={sd_x:.6f}")

# ── Load model ──────────────────────────────────────────────────────────
print("Loading model...")
model = CondFlowNet(cond_dim=l, hidden=256, depth=4, t_dim=64, dropout=0.05)
model.load_state_dict(torch.load(f"{CKPT}/AMZN_FlowFMPlus_best.pt", map_location="cpu", weights_only=True))
model.eval()
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Generate samples ────────────────────────────────────────────────────
nsamp = 1000
T = test.shape[0]
cond_test = test[:,:l]
real_test = test[:,-1].numpy()

print(f"Generating {nsamp} samples for {T} test windows on CPU...")
# Normalize conditions
cond_norm = (cond_test - mu_cond) / sd_cond

all_pu = np.zeros(T)
all_samples = np.zeros((T, nsamp))

# Process in batches to avoid memory issues
BATCH = 200
for start in range(0, T, BATCH):
    end = min(start + BATCH, T)
    B = end - start
    cond_b = cond_norm[start:end]  # (B, 10)

    samples_b = torch.empty((B, nsamp))
    for s in range(0, nsamp, 8):
        cur = min(8, nsamp - s)
        cond_rep = cond_b.repeat_interleave(cur, dim=0)  # (B*cur, 10)
        x0 = torch.randn((B*cur, 1))
        x1_n = fm_sample(model, x0, cond_rep, n_steps=40)
        x1 = x1_n * sd_x + mu_x  # denormalize
        samples_b[:, s:s+cur] = x1.squeeze(-1).view(B, cur)

    pu_b = (samples_b >= 0).float().mean(dim=1).numpy()
    all_pu[start:end] = pu_b
    all_samples[start:end] = samples_b.numpy()

    if (start // BATCH) % 2 == 0:
        print(f"  Processed {end}/{T} windows")

position = 2 * all_pu - 1

print(f"\n{'='*50}")
print(f"FlowFMPlus pu statistics (AMZN, {T} test windows)")
print(f"{'='*50}")
print(f"  mean(pu)       = {all_pu.mean():.4f}")
print(f"  std(pu)        = {all_pu.std():.4f}")
print(f"  min(pu)        = {all_pu.min():.4f}")
print(f"  max(pu)        = {all_pu.max():.4f}")
print(f"  median(pu)     = {np.median(all_pu):.4f}")
print(f"  pct pu>0.7     = {(all_pu > 0.7).mean()*100:.1f}%")
print(f"  pct pu<0.3     = {(all_pu < 0.3).mean()*100:.1f}%")
print(f"  pct 0.4<pu<0.6 = {((all_pu > 0.4) & (all_pu < 0.6)).mean()*100:.1f}%")
print(f"\n  mean(|pos|)    = {np.abs(position).mean():.4f}")
print(f"  mean(pos)      = {position.mean():.4f}")

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(all_pu, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(0.5, color="red", ls="--", lw=2, label="pu=0.5 (no signal)")
ax.axvline(all_pu.mean(), color="orange", ls="-", lw=2, label=f"mean={all_pu.mean():.3f}")
ax.set_xlabel("pu = P(sample >= 0)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("FlowFMPlus: pu distribution (AMZN test)", fontsize=13)
ax.legend(fontsize=10)

ax = axes[1]
ax.hist(position, bins=50, color="coral", edgecolor="white", alpha=0.8)
ax.axvline(0, color="red", ls="--", lw=2)
ax.set_xlabel("Position = 2*pu - 1", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Position size distribution", fontsize=13)

ax = axes[2]
ax.plot(all_pu, color="steelblue", alpha=0.5, lw=0.5)
ax.axhline(0.5, color="red", ls="--", lw=1)
ax.fill_between(range(len(all_pu)), 0.4, 0.6, color="red", alpha=0.1, label="low-signal zone")
ax.set_xlabel("Test window index", fontsize=12)
ax.set_ylabel("pu", fontsize=12)
ax.set_title("pu over time", fontsize=13)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/pu_distribution_flowfmplus_amzn.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT}/pu_distribution_flowfmplus_amzn.png")

# Sample distributions for individual windows
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
for i, idx in enumerate([0, T//4, T//2, 3*T//4, T-1, T//3]):
    ax = axes2[i//3][i%3]
    s = all_samples[idx]
    ax.hist(s, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.axvline(s.mean(), color="orange", ls="-", lw=1.5, label=f"mean={s.mean():.5f}")
    ax.set_title(f"Window {idx}: pu={all_pu[idx]:.3f}, real={real_test[idx]:.5f}", fontsize=11)
    ax.legend(fontsize=9)

plt.suptitle("FlowFMPlus: MC sample distributions for individual test windows (AMZN)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/sample_distributions_flowfmplus_amzn.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}/sample_distributions_flowfmplus_amzn.png")
