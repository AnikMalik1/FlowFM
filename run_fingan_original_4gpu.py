"""4-GPU parallel wrapper around original FinGAN_combos. Zero code rewrite."""
import os, sys, time
import torch.multiprocessing as mp
import pandas as pd

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff(); plt.show = lambda *a, **k: None

sys.path.insert(0, "/projects/s5e/quant/fingan/FlowFM_repo")

dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-fast-bsz/"

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]


def train_ticker(gpu_id, ticker, ti):
    import torch
    import FinGAN

    t0 = time.time()
    try:
        # BSZ=10000 > max N (~8846) → 1 batch/epoch, ~3x faster training
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
            batch_size=10000,
            diter=1,
            plot=False,
            freq=2,
        )

        elapsed = time.time() - t0
        best = df.sort_values("SR_w scaled val", ascending=False).iloc[0]
        print(f"[GPU{gpu_id}] {ticker} done {elapsed:.0f}s | "
              f"best={best['type']} val_SR={best['SR_w scaled val']:+.3f} "
              f"test_SR={best['SR_w scaled']:+.3f} "
              f"PnL={best['PnL_w']:.3f}", flush=True)

        plt.close("all")
        return df.to_dict(orient="records")

    except Exception as e:
        print(f"[GPU{gpu_id}] {ticker} FAILED ({time.time()-t0:.0f}s): {e}", flush=True)
        import traceback; traceback.print_exc()
        return []


def worker(gpu_id, assignments, result_dict):
    rows = []
    for ti, ticker in assignments:
        rows.extend(train_ticker(gpu_id, ticker, ti))
    result_dict[gpu_id] = rows


if __name__ == "__main__":
    for d in ["PnLs", "Results", "Plots", "TrainedModels"]:
        os.makedirs(os.path.join(loc, d), exist_ok=True)

    print(f"Tickers: {len(TICKERS)}, sequential on cuda:0", flush=True)

    all_rows = []
    t0 = time.time()
    for i, ticker in enumerate(TICKERS):
        print(f"\n===== {ticker} ({i+1}/{len(TICKERS)}) =====", flush=True)
        rows = train_ticker(0, ticker, i)
        all_rows.extend(rows)

        # Save partial results after each ticker
        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(loc, "Results", "fingan_full_results_partial.csv"), index=False)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(loc, "Results", "fingan_full_results.csv"), index=False)

        best = df.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False]).groupby("ticker").head(1)
        best.to_csv(os.path.join(loc, "Results", "fingan_best_by_valSR.csv"), index=False)

        print(f"\nAll done in {time.time()-t0:.0f}s ({len(all_rows)} models)", flush=True)
        print(f"\nBest per ticker:", flush=True)
        for _, r in best.iterrows():
            print(f"  {r['ticker']:5s} | {r['type']:12s} | val={r['SR_w scaled val']:+.3f} "
                  f"test={r['SR_w scaled']:+.3f} PnL={r['PnL_w']:.3f}", flush=True)
