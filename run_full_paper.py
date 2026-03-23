import os
import pandas as pd
from tqdm import tqdm

# --- Prevent plot windows / blocking (does NOT affect training results) ---
import matplotlib
matplotlib.use("Agg")  # render to files only, no GUI
import matplotlib.pyplot as plt
plt.ioff()
plt.show = lambda *a, **k: None  # ignore plt.show() calls

import FinGAN

BASE = os.path.dirname(os.path.abspath(__file__))
dataloc = os.path.join(BASE, "data") + os.sep
etflistloc = os.path.join(BASE, "stocks-etfs-list.csv")

out_root  = os.path.join(BASE, "Fin-GAN") + os.sep
modelsloc = os.path.join(out_root, "TrainedModels") + os.sep
plotsloc  = os.path.join(out_root, "Plots") + os.sep
resultsloc= os.path.join(out_root, "Results") + os.sep
pnlsloc   = os.path.join(out_root, "PnLs") + os.sep

for p in [modelsloc, plotsloc, resultsloc, pnlsloc]:
    os.makedirs(p, exist_ok=True)

# Paper stock tickers (Table 1)
stocks = [
    "AMZN","HD","NKE",
    "CL","EL","KO","PEP",
    "APA","OXY",
    "WFC","GS","BLK",
    "PFE","HUM",
    "FDX","GD",
    "IBM","TER",
    "ECL","IP",
    "DTE","WEC",
]

# Paper sector ETFs (raw-return datasets)
etfs = ["XLY","XLP","XLE","XLF","XLV","XLI","XKL","XLB","XLU"]

tickers = stocks + etfs

all_rows = []
partial_path = os.path.join(resultsloc, "paper_full_results_partial.csv")

for t in tqdm(tickers, desc="Tickers completed", unit="ticker"):
    df, corr = FinGAN.FinGAN_combos(
        ticker=t,
        loc=out_root,
        modelsloc=modelsloc,
        plotsloc=plotsloc,
        dataloc=dataloc,
        etflistloc=etflistloc,
        vl_later=True,
        lrg=1e-4, lrd=1e-4,
        n_epochs=100,
        ngrad=25,
        h=1, l=10, pred=1,
        ngpu=0,              # CPU; set 1 only if your torch has CUDA
        tanh_coeff=100,
        tr=0.8, vl=0.1,
        z_dim=8, hid_d=8, hid_g=8,
        checkpoint_epoch=20,
        batch_size=100,
        diter=1,
        plot=False,
        freq=2
    )

    all_rows.append(df)

    # Close any figures created internally (prevents memory growth)
    plt.close("all")

    # Write partial results after each ticker (resume-safe)
    results_partial = pd.concat(all_rows, ignore_index=True)
    results_partial.to_csv(partial_path, index=False)

# Final outputs
results = pd.concat(all_rows, ignore_index=True)
results.to_csv(os.path.join(resultsloc, "paper_full_results.csv"), index=False)

best = (
    results.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False])
           .groupby("ticker")
           .head(1)
)
best.to_csv(os.path.join(resultsloc, "paper_best_by_valSR.csv"), index=False)

print("\nSaved:")
print(" -", os.path.join(resultsloc, "paper_full_results.csv"))
print(" -", os.path.join(resultsloc, "paper_best_by_valSR.csv"))
print(" -", partial_path)
print("\nBest-by-ticker preview:")
print(best[["ticker","type","SR_w scaled val","SR_w scaled","PnL_w val","PnL_w","RMSE","MAE"]])