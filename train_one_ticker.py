"""Train one ticker on one GPU. Called as subprocess."""
import os, sys, time, argparse

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff(); plt.show = lambda *a, **k: None

sys.path.insert(0, "/projects/s5e/quant/fingan/FlowFM_repo")

import torch
assert torch.cuda.is_available(), "CUDA not available"
import FinGAN

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", required=True)
parser.add_argument("--loc", required=True)
parser.add_argument("--dataloc", required=True)
parser.add_argument("--etflistloc", required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=10000)
args = parser.parse_args()

for d in ["TrainedModels", "Plots", "Results", "PnLs"]:
    os.makedirs(os.path.join(args.loc, d), exist_ok=True)

t0 = time.time()
df, corr = FinGAN.FinGAN_combos(
    ticker=args.ticker,
    loc=args.loc,
    modelsloc=os.path.join(args.loc, "TrainedModels") + "/",
    plotsloc=os.path.join(args.loc, "Plots") + "/",
    dataloc=args.dataloc,
    etflistloc=args.etflistloc,
    vl_later=True,
    lrg=1e-4, lrd=1e-4,
    n_epochs=100, ngrad=25,
    h=1, l=10, pred=1, ngpu=1,
    tanh_coeff=100, tr=0.8, vl=0.1,
    z_dim=8, hid_d=8, hid_g=8,
    checkpoint_epoch=20,
    batch_size=args.batch_size,
    diter=1, plot=False, freq=2,
)

elapsed = time.time() - t0
csv_path = os.path.join(args.loc, "Results", f"{args.ticker}_results.csv")
df.to_csv(csv_path, index=False)

best = df.sort_values("SR_w scaled val", ascending=False).iloc[0]
print(f"[GPU{args.gpu}] {args.ticker} done {elapsed:.0f}s | "
      f"best={best['type']} val_SR={best['SR_w scaled val']:+.3f} "
      f"test_SR={best['SR_w scaled']:+.3f}", flush=True)
