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
loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-orig/"

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]


def train_ticker(gpu_id, ticker, ti):
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch/FinGAN
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    assert torch.cuda.is_available(), f"GPU {gpu_id} not available"
    torch.backends.cuda.matmul.allow_tf32 = True

    import FinGAN

    t0 = time.time()
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
    mp.set_start_method("spawn")

    for d in ["PnLs", "Results", "Plots", "TrainedModels"]:
        os.makedirs(os.path.join(loc, d), exist_ok=True)

    n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))
    print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}", flush=True)

    assignments = [[] for _ in range(n_gpus)]
    for i, t in enumerate(TICKERS):
        assignments[i % n_gpus].append((i, t))

    for g in range(n_gpus):
        print(f"  GPU {g}: {[t for _,t in assignments[g]]}", flush=True)

    manager = mp.Manager()
    result_dict = manager.dict()

    t0 = time.time()
    procs = []
    for g in range(n_gpus):
        if assignments[g]:
            p = mp.Process(target=worker, args=(g, assignments[g], result_dict))
            p.start()
            procs.append(p)

    for p in procs:
        p.join()

    all_rows = []
    for g in range(n_gpus):
        if g in result_dict:
            all_rows.extend(result_dict[g])

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
