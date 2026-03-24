"""
GPU script: Compare pu distributions between FinGAN (best-by-val) and FlowFMPlus.
Generates 1000 MC samples per test window for both models.
"""
import sys, os, math, time, csv
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Data Loading ─────────────────────────────────────────────────────
DATA = "/projects/s5e/quant/fingan/FlowFM/data/"
FINGAN_CKPT = "/projects/s5e/quant/fingan/FlowFM/FinGAN checkpoints/"
FLOW_CKPT = "/projects/s5e/quant/fingan/FlowFM/FlowFMPlus_Aux_Checkpoints/"
FLOW_CKPT_OLD = "/projects/s5e/quant/fingan/colabFinGAN_deprecated/Fin-GAN/TrainedModels/"
ETF_LIST = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
OUT = "/projects/s5e/quant/fingan"

import pandas as pd

def etf_find(etflistloc, stock):
    data = pd.read_csv(etflistloc)
    return np.array(data['ticker_y'][data['ticker_x'] == stock])[0]

def load_excess_returns(dataloc, stock, etf):
    s_df = pd.read_csv(dataloc + stock + ".csv")
    e_df = pd.read_csv(dataloc + etf + ".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    smp = dates_dt < pd.Timestamp("2022-01-01")
    s_df = s_df[smp]; e_df = e_df[smp]
    s_log = np.zeros(2 * len(s_df))
    e_log = np.zeros(2 * len(s_df))
    for i in range(len(s_df)):
        s_log[2*i] = np.log(s_df['AdjOpen'].iloc[i])
        s_log[2*i+1] = np.log(s_df['AdjClose'].iloc[i])
        e_log[2*i] = np.log(e_df['AdjOpen'].iloc[i])
        e_log[2*i+1] = np.log(e_df['AdjClose'].iloc[i])
    s_ret = np.clip(np.diff(s_log), -0.15, 0.15)
    e_ret = np.clip(np.diff(e_log), -0.15, 0.15)
    return s_ret - e_ret

def make_windows(data, l=10, h=1):
    n = int((len(data) - l - 1) / h) + 1
    windows = np.zeros((n, l + 1))
    for i in range(n):
        windows[i] = data[i*h : i*h + l + 1]
    return windows

TICKER = "AMZN"
etf = etf_find(ETF_LIST, TICKER)
excess = load_excess_returns(DATA, TICKER, etf)
windows = make_windows(excess, l=10, h=1)
N = len(windows)
N_tr = int(0.8 * N); N_vl = int(0.1 * N)
train_np = windows[:N_tr]
val_np = windows[N_tr:N_tr+N_vl]
test_np = windows[N_tr+N_vl:]
train_t = torch.tensor(train_np, dtype=torch.float32)
val_t = torch.tensor(val_np, dtype=torch.float32)
test_t = torch.tensor(test_np, dtype=torch.float32)
print(f"Train: {train_t.shape}, Val: {val_t.shape}, Test: {test_t.shape}")

# Normalization stats from training set
cond_train = train_t[:, :10]
x_train = train_t[:, 10]
train_mean = cond_train.mean(dim=0)  # per-lag mean for Generator normalization
train_std = cond_train.std(dim=0, unbiased=False) + 1e-8

# For FinGAN: Generator uses SCALAR mean/std for both condition normalization
# and output denormalization. Must be scalar so output_dim=1 works.
all_train_cond = train_t[:, :10]
gen_mean = float(all_train_cond.mean())   # scalar
gen_std = float(all_train_cond.std(unbiased=False)) + 1e-8  # scalar

# For FlowFMPlus normalization
mu_cond_flow = all_train_cond.mean(dim=0)
sd_cond_flow = all_train_cond.std(dim=0, unbiased=False) + 1e-8
mu_x_flow = float(x_train.mean())
sd_x_flow = float(x_train.std(unbiased=False)) + 1e-8

print(f"gen_mean: {gen_mean:.6f}, gen_std: {gen_std:.6f}")
print(f"mu_x_flow={mu_x_flow:.6f}, sd_x_flow={sd_x_flow:.6f}")

# ── 2. FinGAN Generator ────────────────────────────────────────────────
def combine_vectors(x, y, dim=2):
    return torch.cat((x.float(), y.float()), dim=dim)

class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, hidden_dim, output_dim, mean, std):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.mean = mean
        self.std = std
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=hidden_dim, num_layers=1, dropout=0)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.linear1 = nn.Linear(hidden_dim + noise_dim, hidden_dim + noise_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_dim + noise_dim, output_dim)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.ReLU()

    def forward(self, noise, condition, h_0, c_0):
        condition = (condition - self.mean) / self.std
        out, (h_n, c_n) = self.lstm(condition, (h_0, c_0))
        out = combine_vectors(noise.float(), h_n.float(), dim=-1)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = out * self.std + self.mean
        return out

# ── 3. FlowFMPlus Model ────────────────────────────────────────────────
def sinusoidal_time_emb(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half-1, 1))
    ang = t[:, None].float() * freqs[None, :]
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

# ── 4. Load Models ──────────────────────────────────────────────────────
l, z_dim, hid_g = 10, 8, 8
nsamp = 1000

# Find best-by-val FinGAN variant
loss_types = ["ForGAN", "MSE", "PnL", "PnLMSE", "PnLMSESR", "PnLMSESTD", "PnLSR", "PnLSTD", "SR", "SRMSE"]
prefix = f"{TICKER}-Fin-GAN-100-epochs-0.0001-lrd-0.0001-lrg-"

print("\n=== Evaluating FinGAN loss variants on validation set ===")
best_val_sr = -999
best_loss_type = None
best_gen = None

for lt in loss_types:
    ckpt_path = os.path.join(FINGAN_CKPT, f"{prefix}{lt}_generator_checkpoint.pth")
    if not os.path.exists(ckpt_path):
        print(f"  {lt}: checkpoint not found, skipping")
        continue

    gen = Generator(z_dim, l, hid_g, 1, gen_mean, gen_std).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['g_state_dict'])
    gen.eval()

    # Quick val SR computation
    T_val = val_t.shape[0]
    cond_val = val_t[:, :l].unsqueeze(0).to(device)
    h0 = torch.zeros(1, T_val, hid_g, device=device)
    c0 = torch.zeros(1, T_val, hid_g, device=device)
    real_val = val_t[:, -1].numpy()

    with torch.no_grad():
        samples_val = torch.empty(T_val, nsamp, device=device)
        for s in range(nsamp):
            noise = torch.randn(1, T_val, z_dim, device=device)
            fake = gen(noise, cond_val, h0, c0)
            samples_val[:, s] = fake.squeeze()

    pu_val = (samples_val >= 0).float().mean(dim=1).cpu().numpy()
    pnl_half = 10000 * (2 * pu_val - 1) * real_val
    pnl_daily = pnl_half[::2][:len(pnl_half)//2] + pnl_half[1::2][:len(pnl_half)//2]
    val_sr = np.mean(pnl_daily) / (np.std(pnl_daily) + 1e-12) * np.sqrt(252)

    print(f"  {lt:12s}: val SR = {val_sr:+.3f}, mean pu = {pu_val.mean():.3f}")

    if val_sr > best_val_sr:
        best_val_sr = val_sr
        best_loss_type = lt
        best_gen = gen

print(f"\n  Best: {best_loss_type} (val SR = {best_val_sr:.3f})")

# Load FlowFMPlus
flow_model = CondFlowNet(cond_dim=l, hidden=256, depth=4, t_dim=64, dropout=0.05).to(device)

# Try Aux checkpoint first, then old
flow_ckpt = os.path.join(FLOW_CKPT, f"{TICKER}_FlowFMPlus_Aux_best.pt")
if not os.path.exists(flow_ckpt):
    flow_ckpt = os.path.join(FLOW_CKPT_OLD, f"{TICKER}_FlowFMPlus_best.pt")
flow_model.load_state_dict(torch.load(flow_ckpt, map_location=device, weights_only=True))
flow_model.eval()
print(f"Flow model loaded from {flow_ckpt}")

# ── 5. Generate test pu for both models ─────────────────────────────────
T = test_t.shape[0]
real_test = test_t[:, -1].numpy()

print(f"\nGenerating {nsamp} samples for {T} test windows...")

# FinGAN
t0 = time.time()
cond_test_gan = test_t[:, :l].unsqueeze(0).to(device)
h0 = torch.zeros(1, T, hid_g, device=device)
c0 = torch.zeros(1, T, hid_g, device=device)

with torch.no_grad():
    samples_gan = torch.empty(T, nsamp, device=device)
    for s in range(nsamp):
        noise = torch.randn(1, T, z_dim, device=device)
        fake = best_gen(noise, cond_test_gan, h0, c0)
        samples_gan[:, s] = fake.squeeze()
t1 = time.time()
print(f"  FinGAN ({best_loss_type}): {t1-t0:.1f}s")

pu_gan = (samples_gan >= 0).float().mean(dim=1).cpu().numpy()
pos_gan = 2 * pu_gan - 1

# FlowFMPlus
t0 = time.time()
cond_test_flow = ((test_t[:, :l] - mu_cond_flow) / sd_cond_flow).to(device)

with torch.no_grad():
    # Batch all: T * nsamp vectors through ODE
    # Process in chunks to avoid GPU OOM
    CHUNK = 100  # samples per chunk
    samples_flow = torch.empty(T, nsamp, device=device)
    done = 0
    while done < nsamp:
        cur = min(CHUNK, nsamp - done)
        cond_rep = cond_test_flow.repeat(cur, 1)  # (T*cur, 10)
        x = torch.randn(T * cur, 1, device=device)
        dt = 1.0 / 40
        for k in range(40):
            tv = torch.full((T * cur,), (k + 0.5) * dt, device=device)
            x = x + dt * flow_model(x, cond_rep, tv)
        x1 = x * sd_x_flow + mu_x_flow
        samples_flow[:, done:done+cur] = x1.squeeze(-1).view(cur, T).T
        done += cur
t1 = time.time()
print(f"  FlowFMPlus: {t1-t0:.1f}s")

pu_flow = (samples_flow >= 0).float().mean(dim=1).cpu().numpy()
pos_flow = 2 * pu_flow - 1

# ── 6. Statistics ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Metric':<25} {'FinGAN':>12} {'FlowFMPlus':>12}")
print(f"{'='*60}")
print(f"{'mean(pu)':<25} {pu_gan.mean():>12.4f} {pu_flow.mean():>12.4f}")
print(f"{'std(pu)':<25} {pu_gan.std():>12.4f} {pu_flow.std():>12.4f}")
print(f"{'min(pu)':<25} {pu_gan.min():>12.4f} {pu_flow.min():>12.4f}")
print(f"{'max(pu)':<25} {pu_gan.max():>12.4f} {pu_flow.max():>12.4f}")
print(f"{'median(pu)':<25} {np.median(pu_gan):>12.4f} {np.median(pu_flow):>12.4f}")
print(f"{'pct pu>0.7':<25} {(pu_gan>0.7).mean()*100:>11.1f}% {(pu_flow>0.7).mean()*100:>11.1f}%")
print(f"{'pct pu<0.3':<25} {(pu_gan<0.3).mean()*100:>11.1f}% {(pu_flow<0.3).mean()*100:>11.1f}%")
print(f"{'pct 0.4<pu<0.6':<25} {((pu_gan>0.4)&(pu_gan<0.6)).mean()*100:>11.1f}% {((pu_flow>0.4)&(pu_flow<0.6)).mean()*100:>11.1f}%")
print(f"{'mean|position|':<25} {np.abs(pos_gan).mean():>12.4f} {np.abs(pos_flow).mean():>12.4f}")
print(f"{'='*60}")

# Compute PnL for comparison
pnl_half_gan = 10000 * pos_gan * real_test
pnl_half_flow = 10000 * pos_flow * real_test
pnl_daily_gan = pnl_half_gan[::2][:len(pnl_half_gan)//2] + pnl_half_gan[1::2][:len(pnl_half_gan)//2]
pnl_daily_flow = pnl_half_flow[::2][:len(pnl_half_flow)//2] + pnl_half_flow[1::2][:len(pnl_half_flow)//2]
sr_gan = np.mean(pnl_daily_gan) / (np.std(pnl_daily_gan) + 1e-12) * np.sqrt(252)
sr_flow = np.mean(pnl_daily_flow) / (np.std(pnl_daily_flow) + 1e-12) * np.sqrt(252)
cum_gan = np.sum(pnl_daily_gan)
cum_flow = np.sum(pnl_daily_flow)

print(f"\n{'Metric':<25} {'FinGAN':>12} {'FlowFMPlus':>12}")
print(f"{'test SR (annualized)':<25} {sr_gan:>12.3f} {sr_flow:>12.3f}")
print(f"{'cumulative PnL (bp)':<25} {cum_gan:>12.1f} {cum_flow:>12.1f}")
print(f"{'mean daily PnL (bp)':<25} {np.mean(pnl_daily_gan):>12.3f} {np.mean(pnl_daily_flow):>12.3f}")
print(f"{'std daily PnL (bp)':<25} {np.std(pnl_daily_gan):>12.3f} {np.std(pnl_daily_flow):>12.3f}")

# ── 7. Plots ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Row 1: pu distributions
ax = axes[0, 0]
ax.hist(pu_gan, bins=50, color="royalblue", edgecolor="white", alpha=0.7, label=f"FinGAN ({best_loss_type})")
ax.hist(pu_flow, bins=50, color="coral", edgecolor="white", alpha=0.7, label="FlowFMPlus")
ax.axvline(0.5, color="black", ls="--", lw=2)
ax.set_xlabel("pu = P(sample >= 0)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("pu distribution comparison", fontsize=13)
ax.legend(fontsize=10)

ax = axes[0, 1]
ax.hist(pu_gan, bins=50, color="royalblue", edgecolor="white", alpha=0.8)
ax.axvline(0.5, color="red", ls="--", lw=2)
ax.axvline(pu_gan.mean(), color="orange", ls="-", lw=2, label=f"mean={pu_gan.mean():.3f}")
ax.set_xlabel("pu", fontsize=12)
ax.set_title(f"FinGAN ({best_loss_type}) pu", fontsize=13)
ax.legend()

ax = axes[0, 2]
ax.hist(pu_flow, bins=50, color="coral", edgecolor="white", alpha=0.8)
ax.axvline(0.5, color="red", ls="--", lw=2)
ax.axvline(pu_flow.mean(), color="orange", ls="-", lw=2, label=f"mean={pu_flow.mean():.3f}")
ax.set_xlabel("pu", fontsize=12)
ax.set_title("FlowFMPlus pu", fontsize=13)
ax.legend()

# Row 2: position, pu over time, cumulative PnL
ax = axes[1, 0]
ax.hist(pos_gan, bins=50, color="royalblue", edgecolor="white", alpha=0.7, label=f"FinGAN")
ax.hist(pos_flow, bins=50, color="coral", edgecolor="white", alpha=0.7, label="FlowFMPlus")
ax.axvline(0, color="black", ls="--", lw=2)
ax.set_xlabel("Position = 2*pu - 1", fontsize=12)
ax.set_title("Position size comparison", fontsize=13)
ax.legend()

ax = axes[1, 1]
ax.plot(pu_gan, color="royalblue", alpha=0.4, lw=0.5, label="FinGAN")
ax.plot(pu_flow, color="coral", alpha=0.4, lw=0.5, label="FlowFMPlus")
ax.axhline(0.5, color="black", ls="--", lw=1)
ax.set_xlabel("Test window index", fontsize=12)
ax.set_ylabel("pu", fontsize=12)
ax.set_title("pu over time", fontsize=13)
ax.legend()

ax = axes[1, 2]
ax.plot(np.cumsum(pnl_daily_gan), color="royalblue", lw=1.5, label=f"FinGAN (SR={sr_gan:.2f})")
ax.plot(np.cumsum(pnl_daily_flow), color="coral", lw=1.5, label=f"FlowFMPlus (SR={sr_flow:.2f})")
ax.set_xlabel("Trading day", fontsize=12)
ax.set_ylabel("Cumulative PnL (bp)", fontsize=12)
ax.set_title("Cumulative PnL comparison (AMZN)", fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle(f"FinGAN vs FlowFMPlus: pu Distribution Analysis (AMZN, {T} test windows, {nsamp} MC samples)", fontsize=15)
plt.tight_layout()
out_path = f"{OUT}/pu_comparison_fingan_vs_flow_amzn.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# Sample distributions for 4 windows
fig2, axes2 = plt.subplots(2, 4, figsize=(24, 10))
indices = [0, T//4, T//2, 3*T//4]

for i, idx in enumerate(indices):
    # FinGAN
    ax = axes2[0, i]
    s = samples_gan[idx].cpu().numpy()
    ax.hist(s, bins=40, color="royalblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.axvline(s.mean(), color="orange", ls="-", lw=1.5)
    ax.set_title(f"FinGAN w{idx}: pu={pu_gan[idx]:.3f}", fontsize=11)
    if i == 0: ax.set_ylabel("FinGAN", fontsize=12)

    # FlowFMPlus
    ax = axes2[1, i]
    s = samples_flow[idx].cpu().numpy()
    ax.hist(s, bins=40, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.axvline(s.mean(), color="orange", ls="-", lw=1.5)
    ax.set_title(f"Flow w{idx}: pu={pu_flow[idx]:.3f}", fontsize=11)
    if i == 0: ax.set_ylabel("FlowFMPlus", fontsize=12)

plt.suptitle("MC sample distributions: FinGAN (top) vs FlowFMPlus (bottom)", fontsize=14)
plt.tight_layout()
out_path2 = f"{OUT}/sample_comparison_fingan_vs_flow_amzn.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
