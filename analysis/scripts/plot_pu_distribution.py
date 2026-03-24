"""
Plot pu distribution for FlowFMPlus on AMZN test set.
pu = fraction of 1000 MC samples >= 0 for each test conditioning window.
CPU-only, no GPU needed.
"""
import sys
sys.path.insert(0, "/projects/s5e/quant/fingan/colabFinGAN")

import torch
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_adapter import CondFlowNet, FlowGenAdapter

# ── 1. Data loading (replicate FinGAN.excessreturns logic) ──────────────
def load_excess_returns(dataloc, ticker, etf):
    df_stock = pd.read_csv(f"{dataloc}/{ticker}.csv")
    df_etf = pd.read_csv(f"{dataloc}/{etf}.csv")

    df_stock["date_dt"] = pd.to_datetime(df_stock["date"])
    df_etf["date_dt"] = pd.to_datetime(df_etf["date"])

    df_stock = df_stock[df_stock["date_dt"] < "2022-01-01"].reset_index(drop=True)
    df_etf = df_etf[df_etf["date_dt"] < "2022-01-01"].reset_index(drop=True)

    # Merge on date
    merged = pd.merge(df_stock, df_etf, on="date_dt", suffixes=("_s", "_e"))

    # Interleaved log prices: open, close, open, close, ...
    N = len(merged)
    s_log = np.zeros(2 * N)
    e_log = np.zeros(2 * N)
    for i in range(N):
        s_log[2*i]   = np.log(merged["AdjOpen_s"].iloc[i] if "AdjOpen_s" in merged.columns else merged["AdjClose_s"].iloc[i])
        s_log[2*i+1] = np.log(merged["AdjClose_s"].iloc[i])
        e_log[2*i]   = np.log(merged["AdjOpen_e"].iloc[i] if "AdjOpen_e" in merged.columns else merged["AdjClose_e"].iloc[i])
        e_log[2*i+1] = np.log(merged["AdjClose_e"].iloc[i])

    # Returns = diff of log prices
    s_ret = np.diff(s_log)
    e_ret = np.diff(e_log)

    # Clip
    s_ret = np.clip(s_ret, -0.15, 0.15)
    e_ret = np.clip(e_ret, -0.15, 0.15)

    # Excess returns
    excess = s_ret - e_ret
    return excess

def make_windows(data, l=10, pred=1, h=1):
    """Create sliding windows: each row = [cond_0, ..., cond_{l-1}, target]"""
    windows = []
    for i in range(0, len(data) - l - pred + 1, h):
        row = list(data[i:i+l]) + [data[i+l]]
        windows.append(row)
    return np.array(windows)

# ── 2. Main ─────────────────────────────────────────────────────────────
DATA_DIR = "/projects/s5e/quant/fingan/colabFinGAN/data"
CKPT_DIR = "/projects/s5e/quant/fingan/colabFinGAN/Fin-GAN/TrainedModels"
OUT_DIR = "/projects/s5e/quant/fingan"

# Check if AdjOpen exists
df_test = pd.read_csv(f"{DATA_DIR}/AMZN.csv", nrows=2)
has_adjopen = "AdjOpen" in df_test.columns
print(f"Has AdjOpen column: {has_adjopen}")

# Load data using FinGAN's exact logic
# We need to replicate exactly, so let's use FinGAN.py directly
from FinGAN import split_train_val_test, excessreturns, ETF_find

etflistloc = f"{DATA_DIR}/stocks-etfs-list.csv" if not pd.io.common.file_exists(f"/projects/s5e/quant/fingan/colabFinGAN/stocks-etfs-list.csv") else "/projects/s5e/quant/fingan/colabFinGAN/stocks-etfs-list.csv"

# Use FinGAN's split function directly
l, pred, h, freq = 10, 1, 1, 2
tr, vl = 0.8, 0.1

try:
    train, val, test, N_tr, N_vl = split_train_val_test(
        DATA_DIR, etflistloc, "AMZN", h, l, pred, freq, tr, vl
    )
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
except Exception as e:
    print(f"split_train_val_test failed: {e}")
    print("Falling back to manual loading...")

    excess = load_excess_returns(DATA_DIR, "AMZN", "XLY")
    all_windows = make_windows(excess, l=10, pred=1, h=1)
    N = len(all_windows)
    N_tr = int(0.8 * N)
    N_vl = int(0.1 * N)
    train = all_windows[:N_tr]
    val = all_windows[N_tr:N_tr+N_vl]
    test = all_windows[N_tr+N_vl:]
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

# Convert to tensors
train_t = torch.tensor(train, dtype=torch.float32)
test_t = torch.tensor(test, dtype=torch.float32)

# ── 3. Normalization (from training set) ────────────────────────────────
cond_train = train_t[:, :l]
x_train = train_t[:, l:l+1]

mu_cond = cond_train.mean(dim=0)
sd_cond = cond_train.std(dim=0, unbiased=False) + 1e-8
mu_x = float(x_train.mean().item())
sd_x = float(x_train.std(unbiased=False).item()) + 1e-8

print(f"mu_x={mu_x:.6f}, sd_x={sd_x:.6f}")
print(f"mu_cond range: [{mu_cond.min():.6f}, {mu_cond.max():.6f}]")

# ── 4. Load model ───────────────────────────────────────────────────────
device = torch.device("cpu")

flow_model = CondFlowNet(cond_dim=l, hidden=256, depth=4, t_dim=64, dropout=0.05)
ckpt_path = f"{CKPT_DIR}/AMZN_FlowFMPlus_best.pt"
state = torch.load(ckpt_path, map_location=device, weights_only=True)
flow_model.load_state_dict(state, strict=True)
flow_model.eval()

gen = FlowGenAdapter(flow_model, mu_cond, sd_cond, mu_x, sd_x, ode_steps=40, pred=1)
gen.eval()

print(f"Model loaded from {ckpt_path}")
print(f"Model params: {sum(p.numel() for p in flow_model.parameters()):,}")

# ── 5. Generate samples and compute pu ──────────────────────────────────
nsamp = 1000
chunk = 8
z_dim = 8
hid_g = 8

cond_test = test_t[:, :l]
real_test = test_t[:, -1].numpy()
T = cond_test.shape[0]

print(f"Test windows: {T}, generating {nsamp} samples each on CPU...")

samples = torch.empty((T, nsamp))
done = 0

with torch.no_grad():
    while done < nsamp:
        cur = min(chunk, nsamp - done)
        eff_batch = T * cur

        cond_rep = cond_test.repeat_interleave(cur, dim=0)  # (T*cur, 10)
        cond_in = cond_rep.unsqueeze(0)  # (1, T*cur, 10)
        noise = torch.randn((1, eff_batch, z_dim))
        h0 = torch.zeros((1, eff_batch, hid_g))
        c0 = torch.zeros((1, eff_batch, hid_g))

        fake = gen(noise, cond_in, h0, c0)  # (1, T*cur, 1)
        out = fake.reshape(-1).view(T, cur)
        samples[:, done:done+cur] = out
        done += cur

        if done % 100 == 0 or done == nsamp:
            print(f"  Generated {done}/{nsamp} samples")

pu = (samples >= 0).float().mean(dim=1).numpy()  # (T,)
position = 2 * pu - 1

print(f"\n=== FlowFMPlus pu statistics (AMZN) ===")
print(f"  mean(pu)     = {pu.mean():.4f}")
print(f"  std(pu)      = {pu.std():.4f}")
print(f"  min(pu)      = {pu.min():.4f}")
print(f"  max(pu)      = {pu.max():.4f}")
print(f"  median(pu)   = {np.median(pu):.4f}")
print(f"  pct pu>0.7   = {(pu > 0.7).mean()*100:.1f}%")
print(f"  pct pu<0.3   = {(pu < 0.3).mean()*100:.1f}%")
print(f"  pct 0.4<pu<0.6 = {((pu > 0.4) & (pu < 0.6)).mean()*100:.1f}%")
print(f"\n=== Position (2*pu-1) statistics ===")
print(f"  mean(pos)    = {position.mean():.4f}")
print(f"  std(pos)     = {position.std():.4f}")
print(f"  mean(|pos|)  = {np.abs(position).mean():.4f}")

# ── 6. Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) pu histogram
ax = axes[0]
ax.hist(pu, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="pu=0.5 (no signal)")
ax.axvline(pu.mean(), color="orange", linestyle="-", linewidth=2, label=f"mean={pu.mean():.3f}")
ax.set_xlabel("pu = P(sample >= 0)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("FlowFMPlus: pu distribution (AMZN test set)", fontsize=13)
ax.legend(fontsize=10)

# (b) position histogram
ax = axes[1]
ax.hist(position, bins=50, color="coral", edgecolor="white", alpha=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="pos=0 (no trade)")
ax.set_xlabel("Position = 2*pu - 1", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("FlowFMPlus: position size distribution", fontsize=13)
ax.legend(fontsize=10)

# (c) pu over time
ax = axes[2]
ax.plot(pu, color="steelblue", alpha=0.5, linewidth=0.5)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
ax.fill_between(range(len(pu)), 0.4, 0.6, color="red", alpha=0.1, label="low-signal zone")
ax.set_xlabel("Test window index", fontsize=12)
ax.set_ylabel("pu", fontsize=12)
ax.set_title("FlowFMPlus: pu over time", fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
out_path = f"{OUT_DIR}/pu_distribution_flowfmplus_amzn.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")

# ── 7. Sample distribution for a few test windows ───────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
indices = [0, T//4, T//2, 3*T//4, T-1, T//3]

for i, idx in enumerate(indices):
    ax = axes2[i//3][i%3]
    s = samples[idx].numpy()
    ax.hist(s, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.axvline(s.mean(), color="orange", linestyle="-", linewidth=1.5, label=f"mean={s.mean():.5f}")
    ax.set_title(f"Window {idx}: pu={pu[idx]:.3f}, real={real_test[idx]:.5f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlabel("Generated return sample")

plt.suptitle("FlowFMPlus: MC sample distributions for individual test windows (AMZN)", fontsize=14)
plt.tight_layout()
out_path2 = f"{OUT_DIR}/sample_distributions_flowfmplus_amzn.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out_path2}")
