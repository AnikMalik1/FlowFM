"""Complete fair comparison: FinGAN vs FlowFM-Composite vs FlowFM-2Phase.
Reports ALL metrics: SR, PnL, cumPnL, drawdown, RMSE, MAE, Corr.
Portfolio-level (Anik slides format) + per-ticker."""
import pandas as pd, numpy as np, os

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]

# BCE → ForGAN mapping for FinGAN PnL files
FINGAN_LOSS_MAP = {"BCE": "ForGAN"}


def load_best_by_val(results_csv):
    df = pd.read_csv(results_csv)
    return df.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False]).groupby("ticker").head(1)


def portfolio_from_pnls(pnl_dir, best_df, prefix="FinGAN", loss_map=None):
    """Load per-ticker PnL CSVs for best-by-val variants, compute portfolio metrics."""
    per_ticker = {}
    for _, r in best_df.iterrows():
        t = r["ticker"]
        lt = r["type"]
        # Apply naming map
        pnl_lt = (loss_map or {}).get(lt, lt)
        f = os.path.join(pnl_dir, f"{t}-{prefix}-{pnl_lt}.csv")
        if not os.path.exists(f):
            f = os.path.join(pnl_dir, f"{t}-{pnl_lt}.csv")
        if not os.path.exists(f):
            continue
        s = pd.read_csv(f, header=None).iloc[:, -1].astype(float).values
        per_ticker[t] = s

    if not per_ticker:
        return None, {}

    min_len = min(len(s) for s in per_ticker.values())
    mat = np.vstack([s[:min_len] for s in per_ticker.values()])
    port = mat.sum(axis=0)

    mean_d = float(np.mean(port))
    std_d = float(np.std(port))
    cum = np.cumsum(port)
    dd = cum - np.maximum.accumulate(cum)

    portfolio = {
        "tickers": len(per_ticker),
        "days": min_len,
        "mean_daily_pnl": mean_d,
        "std_daily_pnl": std_d,
        "portfolio_sr": (mean_d / (std_d + 1e-12)) * np.sqrt(252),
        "cum_pnl": float(cum[-1]),
        "max_drawdown": float(dd.min()),
    }
    return portfolio, per_ticker


# Load results
methods = {}

# FinGAN
fg_best = load_best_by_val("/projects/s5e/quant/fingan/FlowFM/Fin-GAN-4gpu/Results/fingan_full_results.csv")
fg_port, fg_pnls = portfolio_from_pnls(
    "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-4gpu/PnLs/",
    fg_best, prefix="FinGAN", loss_map=FINGAN_LOSS_MAP)
methods["FinGAN"] = {"best": fg_best, "port": fg_port, "pnls": fg_pnls}

# FlowFM-Composite (v3)
fc_path = "/projects/s5e/quant/fingan/FlowFM/FlowFM-10loss-medium/Results/flowfm_10loss_full.csv"
if os.path.exists(fc_path):
    fc_best = load_best_by_val(fc_path)
    methods["FM-Composite"] = {"best": fc_best, "port": None, "pnls": {}}

# FlowFM-2Phase
fp_path = "/projects/s5e/quant/fingan/FlowFM/FlowFM-2phase-medium/Results/flowfm_2phase_full.csv"
if os.path.exists(fp_path):
    fp_best = load_best_by_val(fp_path)
    fp_port, fp_pnls = portfolio_from_pnls(
        "/projects/s5e/quant/fingan/FlowFM/FlowFM-2phase-medium/PnLs/",
        fp_best, prefix="", loss_map={})
    methods["FM-2Phase"] = {"best": fp_best, "port": fp_port, "pnls": fp_pnls}

# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════

method_names = list(methods.keys())

# 1. Per-ticker SR metrics
print("=" * 90)
print("1. PER-TICKER SHARPE RATIO")
print("=" * 90)
header = f"{'Metric':<35}" + "".join(f"{n:>15}" for n in method_names)
print(header)
print("-" * 90)

for label, func in [
    ("Mean Test SR", lambda b: b["SR_w scaled"].mean()),
    ("Median Test SR", lambda b: b["SR_w scaled"].median()),
    ("Std Test SR", lambda b: b["SR_w scaled"].std()),
    ("Min Test SR", lambda b: b["SR_w scaled"].min()),
    ("Max Test SR", lambda b: b["SR_w scaled"].max()),
    ("Mean Val SR", lambda b: b["SR_w scaled val"].mean()),
    ("Positive Test SR (>0)", lambda b: int((b["SR_w scaled"] > 0).sum())),
    ("Strong Test SR (>0.5)", lambda b: int((b["SR_w scaled"] > 0.5).sum())),
    ("Catastrophic SR (<-0.5)", lambda b: int((b["SR_w scaled"] < -0.5).sum())),
    ("Val-Test Gap (mean)", lambda b: (b["SR_w scaled val"] - b["SR_w scaled"]).mean()),
]:
    vals = []
    for n in method_names:
        b = methods[n]["best"]
        try:
            v = func(b)
            vals.append(f"{v:>+15.3f}" if isinstance(v, float) else f"{v:>12}/31  ")
        except:
            vals.append(f"{'N/A':>15}")
    print(f"{label:<35}" + "".join(vals))

# 2. Per-ticker PnL/RMSE/MAE (from results CSV)
print()
print("=" * 90)
print("2. PER-TICKER PnL, RMSE, MAE, CORR")
print("=" * 90)

for metric, col in [("Mean PnL (bp/day)", "PnL_w"), ("Mean RMSE", "RMSE"),
                     ("Mean MAE", "MAE"), ("Mean Corr", "Corr")]:
    vals = []
    for n in method_names:
        b = methods[n]["best"]
        if col in b.columns:
            vals.append(f"{b[col].mean():>+15.4f}")
        else:
            vals.append(f"{'N/A':>15}")
    print(f"{metric:<35}" + "".join(vals))

# 3. Portfolio-level metrics (Anik slides format)
print()
print("=" * 90)
print("3. PORTFOLIO-LEVEL (equal-weight, sum across tickers)")
print("=" * 90)

for metric in ["tickers", "days", "mean_daily_pnl", "std_daily_pnl",
               "portfolio_sr", "cum_pnl", "max_drawdown"]:
    vals = []
    for n in method_names:
        p = methods[n].get("port")
        if p and metric in p:
            v = p[metric]
            if isinstance(v, int):
                vals.append(f"{v:>15}")
            else:
                vals.append(f"{v:>+15.2f}")
        else:
            vals.append(f"{'N/A':>15}")
    print(f"{metric:<35}" + "".join(vals))

# Anik's reference numbers
print()
print("  Anik slides (reference):  SR=1.800, PnL=2.23 bp/day, CumPnL=1227.45 bp, DD=-242.39 bp")

# 4. Per-ticker detail
print()
print("=" * 120)
print("4. PER-TICKER DETAIL")
print("=" * 120)
print(f"{'Ticker':<6} | {'':^30} FinGAN {'':^30} | {'':^25} FM-2Phase")
print(f"{'':6s} | {'loss':>12} {'valSR':>7} {'testSR':>7} {'PnL':>7} {'RMSE':>7} | {'loss':>15} {'valSR':>7} {'testSR':>7} {'PnL':>7} {'RMSE':>7}")
print("-" * 120)

for t in TICKERS:
    parts = []
    for n in ["FinGAN", "FM-2Phase"]:
        if n not in methods:
            parts.append(f"{'N/A':>15} {'':>7} {'':>7} {'':>7} {'':>7}")
            continue
        b = methods[n]["best"]
        row = b[b["ticker"] == t]
        if len(row) == 0:
            parts.append(f"{'N/A':>15} {'':>7} {'':>7} {'':>7} {'':>7}")
            continue
        r = row.iloc[0]
        lt = str(r["type"])[:15]
        vsr = r.get("SR_w scaled val", 0)
        tsr = r.get("SR_w scaled", 0)
        pnl = r.get("PnL_w", 0) if "PnL_w" in r.index else 0
        rmse = r.get("RMSE", 0) if "RMSE" in r.index else 0
        parts.append(f"{lt:>15} {vsr:>+7.3f} {tsr:>+7.3f} {pnl:>+7.2f} {rmse:>7.4f}")
    print(f"{t:<6} | {parts[0]} | {parts[1] if len(parts)>1 else 'N/A'}")

# 5. Loss variant distribution
print()
print("=" * 90)
print("5. BEST LOSS VARIANT DISTRIBUTION")
print("=" * 90)
for n in method_names:
    b = methods[n]["best"]
    dist = dict(b["type"].value_counts())
    print(f"  {n}: {dist}")
