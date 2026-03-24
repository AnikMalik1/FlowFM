"""Optimized FinGAN training: 4 GPUs parallel, all 31 tickers × 10 loss variants."""
import os, sys, time, numpy as np, torch, torch.optim as optim, torch.nn as nn
import torch.multiprocessing as mp
import pandas as pd

assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff(); plt.show = lambda *a, **k: None

sys.path.insert(0, "/projects/s5e/quant/fingan/colabFinGAN_deprecated")
sys.path.insert(0, "/projects/s5e/quant/fingan/FlowFM_repo")

import FinGAN

dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-fast/"

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]


def train_ticker(gpu_id, ticker, ti):
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
    torch.manual_seed(42 + ti)

    t0 = time.time()

    # Use FinGAN's original combo training (10 loss variants)
    try:
        df, corr = FinGAN.FinGAN_combos(
            ticker=ticker,
            loc=loc,
            modelsloc=os.path.join(loc, "TrainedModels") + "/",
            plotsloc=os.path.join(loc, "Plots") + "/",
            dataloc=dataloc,
            etflistloc=etflistloc,
            vl_later=True,
            lrg=1e-4, lrd=1e-4,
            n_epochs=100,
            ngrad=25,
            h=1, l=10, pred=1,
            ngpu=1,
            tanh_coeff=100,
            tr=0.8, vl=0.1,
            z_dim=8, hid_d=8, hid_g=8,
            checkpoint_epoch=20,
            batch_size=100,
            diter=1,
            plot=False,
            freq=2,
        )
        elapsed = time.time() - t0

        # Save per-ticker results
        csv_path = os.path.join(loc, "Results", f"{ticker}_results.csv")
        df.to_csv(csv_path, index=False)

        # Find best-by-val
        best = df.sort_values("SR_w scaled val", ascending=False).iloc[0]
        print(f"[GPU{gpu_id}] {ticker} done {elapsed:.0f}s | "
              f"best={best['type']} val_SR={best['SR_w scaled val']:.3f} "
              f"test_SR={best['SR_w scaled']:.3f}", flush=True)

        return df.to_dict(orient="records")

    except Exception as e:
        print(f"[GPU{gpu_id}] {ticker} FAILED: {e}", flush=True)
        return []


def worker(gpu_id, assignments, result_dict):
    rows = []
    for ti, ticker in assignments:
        r = train_ticker(gpu_id, ticker, ti)
        rows.extend(r)
    result_dict[gpu_id] = rows


if __name__ == "__main__":
    mp.set_start_method("spawn")

    for d in ["PnLs", "Results", "Plots", "TrainedModels"]:
        os.makedirs(os.path.join(loc, d), exist_ok=True)

    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}", flush=True)
    print(f"Total models: {len(TICKERS)} × 10 loss variants = {len(TICKERS)*10}", flush=True)

    assignments = [[] for _ in range(n_gpus)]
    for i, t in enumerate(TICKERS):
        assignments[i % n_gpus].append((i, t))

    for g in range(n_gpus):
        tks = [t for _, t in assignments[g]]
        print(f"  GPU {g} ({len(tks)} tickers): {tks}", flush=True)

    manager = mp.Manager()
    result_dict = manager.dict()

    t0 = time.time()
    procs = []
    for g in range(n_gpus):
        if not assignments[g]:
            continue
        p = mp.Process(target=worker, args=(g, assignments[g], result_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Aggregate results
    all_rows = []
    for g in range(n_gpus):
        if g in result_dict:
            all_rows.extend(result_dict[g])

    if all_rows:
        results = pd.DataFrame(all_rows)
        results.to_csv(os.path.join(loc, "Results", "fingan_full_results.csv"), index=False)

        best = (
            results.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False])
                   .groupby("ticker").head(1)
        )
        best.to_csv(os.path.join(loc, "Results", "fingan_best_by_valSR.csv"), index=False)

        print(f"\n{'='*70}", flush=True)
        print(f"All {len(TICKERS)} tickers done in {time.time()-t0:.0f}s", flush=True)
        print(f"Results: {os.path.join(loc, 'Results', 'fingan_full_results.csv')}", flush=True)
        print(f"\nBest-by-ticker:", flush=True)
        for _, row in best.iterrows():
            print(f"  {row['ticker']:5s} | {row['type']:12s} | "
                  f"val_SR={row['SR_w scaled val']:+.3f} | "
                  f"test_SR={row['SR_w scaled']:+.3f} | "
                  f"PnL={row['PnL_w']:.2f}", flush=True)
    else:
        print("No results collected!", flush=True)
