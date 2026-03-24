"""4-GPU parallel FinGAN via subprocess (avoids CUDA context conflicts)."""
import os, sys, time, subprocess, pandas as pd

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]

loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-4gpu/"
dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
script = "/projects/s5e/quant/fingan/FlowFM_repo/train_one_ticker.py"

n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))
print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}", flush=True)

for d in ["TrainedModels", "Plots", "Results"]:
    os.makedirs(os.path.join(loc, d), exist_ok=True)

# Distribute tickers across GPUs
assignments = [[] for _ in range(n_gpus)]
for i, t in enumerate(TICKERS):
    assignments[i % n_gpus].append(t)

for g in range(n_gpus):
    print(f"  GPU {g}: {assignments[g]}", flush=True)

t0 = time.time()

# Launch one subprocess per GPU, each processes its assigned tickers sequentially
procs = []
for gpu_id in range(n_gpus):
    if not assignments[gpu_id]:
        continue

    tickers_str = " ".join(assignments[gpu_id])
    # Shell script: loop over tickers, each with CUDA_VISIBLE_DEVICES set
    cmd = f"""
export CUDA_VISIBLE_DEVICES={gpu_id}
export PATH="/projects/s5e/quant/miniforge3/bin:$PATH"
cd /projects/s5e/quant/fingan/FlowFM_repo
for TICKER in {tickers_str}; do
    python3 {script} --ticker $TICKER --loc {loc} --dataloc {dataloc} --etflistloc {etflistloc} --gpu {gpu_id} --batch_size 10000
done
"""
    p = subprocess.Popen(["bash", "-c", cmd], stdout=sys.stdout, stderr=sys.stderr)
    procs.append(p)
    print(f"  Launched GPU {gpu_id} subprocess (PID {p.pid})", flush=True)

# Wait for all
for p in procs:
    p.wait()

total = time.time() - t0
print(f"\nAll subprocesses done in {total:.0f}s", flush=True)

# Aggregate results
all_rows = []
for ticker in TICKERS:
    csv = os.path.join(loc, "Results", f"{ticker}_results.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        all_rows.append(df)
    else:
        print(f"  WARNING: {ticker} results missing", flush=True)

if all_rows:
    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(os.path.join(loc, "Results", "fingan_full_results.csv"), index=False)

    best = results.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False]).groupby("ticker").head(1)
    best.to_csv(os.path.join(loc, "Results", "fingan_best_by_valSR.csv"), index=False)

    print(f"\nBest per ticker:", flush=True)
    for _, r in best.iterrows():
        print(f"  {r['ticker']:5s} | {r['type']:12s} | val={r['SR_w scaled val']:+.3f} test={r['SR_w scaled']:+.3f}", flush=True)
