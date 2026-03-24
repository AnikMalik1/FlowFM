"""
Two-phase FlowFMPlus v2: FM pretrain → financial fine-tune (partial unfreeze).

Phase 1: Full model trained with velocity MSE (learn distribution shape)
Phase 2: Unfreeze last ResBlock + output head, fine-tune with financial loss (500 ep)

v1 → v2 changes:
  - Partial unfreeze: blocks[-1] + out (was: out only)
  - Phase 2 epochs: 500 (was: 100)
  - Goal: push pu to extremes → higher raw PnL while keeping high SR
"""
import os, sys, time, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.multiprocessing as mp
import pandas as pd

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff(); plt.show = lambda *a, **k: None

sys.path.insert(0, "/projects/s5e/quant/fingan/colabFinGAN_deprecated")
sys.path.insert(0, "/projects/s5e/quant/fingan/FlowFM_repo")

import FinGAN
from flow_adapter_aux import CondFlowNet, fm_batch_loss, FlowGenAdapter, EMA
from run_flowfm_aux_31 import load_data_for_ticker, make_model
from eval_gpu import Evaluation2_gpu

_cfg_name = os.environ.get("FLOWFM_CONFIG", "medium")
MODEL_TAG = f"FlowFM_2phase_{_cfg_name}"
dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = f"/projects/s5e/quant/fingan/FlowFM/FlowFM-2phase-v2-{_cfg_name}/"

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]

MODEL_CONFIGS = {
    "small":  {"hidden": 32, "depth": 2},
    "medium": {"hidden": 64, "depth": 2},
    "large":  {"hidden": 64, "depth": 4},
}
ACTIVE_CONFIG = os.environ.get("FLOWFM_CONFIG", "medium")

# ── Financial losses (Phase 2 only, no FM base) ─────────────────────────
def fin_pnl(x1_hat, x1_real):
    ft = torch.tanh(1.0 * x1_hat.squeeze(-1))
    return -torch.mean(ft * x1_real.squeeze(-1))

def fin_sr(x1_hat, x1_real):
    ft = torch.tanh(1.0 * x1_hat.squeeze(-1))
    pnl_vec = ft * x1_real.squeeze(-1)
    return -(torch.mean(pnl_vec) / (torch.std(pnl_vec) + 1e-8))

def fin_pnlstd(x1_hat, x1_real):
    ft = torch.tanh(1.0 * x1_hat.squeeze(-1))
    pnl_vec = ft * x1_real.squeeze(-1)
    return -torch.mean(pnl_vec) + torch.std(pnl_vec)

def fin_pnlsr(x1_hat, x1_real):
    return fin_pnl(x1_hat, x1_real) + fin_sr(x1_hat, x1_real)

def fin_mse(x1_hat, x1_real):
    return torch.mean((x1_hat - x1_real) ** 2)

def fin_pnlmse(x1_hat, x1_real):
    return fin_pnl(x1_hat, x1_real) + fin_mse(x1_hat, x1_real)

def fin_pnlmsesr(x1_hat, x1_real):
    return fin_pnl(x1_hat, x1_real) + fin_mse(x1_hat, x1_real) + fin_sr(x1_hat, x1_real)

def fin_pnlmsestd(x1_hat, x1_real):
    return fin_pnl(x1_hat, x1_real) + fin_mse(x1_hat, x1_real) + fin_pnlstd(x1_hat, x1_real)

def fin_srmse(x1_hat, x1_real):
    return fin_sr(x1_hat, x1_real) + fin_mse(x1_hat, x1_real)

FIN_LOSSES = {
    "PnL": fin_pnl, "SR": fin_sr, "PnLSTD": fin_pnlstd,
    "PnLSR": fin_pnlsr, "MSE": fin_mse, "PnLMSE": fin_pnlmse,
    "PnLMSESR": fin_pnlmsesr, "PnLMSESTD": fin_pnlmsestd, "SRMSE": fin_srmse,
}

# ── Fast validation ─────────────────────────────────────────────────────
@torch.no_grad()
def fast_val_sr(model, val_data, l, mu_cond, sd_cond, mu_x, sd_x, dev, nsamp=1000, ode_steps=40):
    model.eval()
    T = val_data.shape[0]
    cond = (val_data[:, :l] - mu_cond) / sd_cond
    real = val_data[:, -1]
    cond_rep = cond.repeat(nsamp, 1)
    x = torch.randn(T * nsamp, 1, device=dev)
    dt = 1.0 / ode_steps
    for k in range(ode_steps):
        t = torch.full((T * nsamp,), (k + 0.5) * dt, device=dev)
        x = x + dt * model(x, cond_rep, t)
    x = x * sd_x + mu_x
    samples = x.squeeze(-1).view(nsamp, T).T
    pu = (samples >= 0).float().mean(dim=1)
    pnl_ws = 10000.0 * (2.0 * pu - 1.0) * real
    Td = T // 2
    pnl_wd = pnl_ws[:2*Td].reshape(Td, 2).sum(dim=1)
    mu_pnl = pnl_wd.mean().item()
    sd_pnl = pnl_wd.std(unbiased=False).item() + 1e-12
    return (mu_pnl / sd_pnl) * np.sqrt(252.0)


def train_one(gpu_id, ticker, ti):
    assert torch.cuda.is_available()
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")

    cfg = MODEL_CONFIGS[ACTIVE_CONFIG]
    l, tr, vl = 10, 0.8, 0.1
    hidden, depth, dropout = cfg["hidden"], cfg["depth"], 0.05

    # Phase 1 config
    p1_epochs = 100
    p1_lr = 3e-4

    # Phase 2 config (v2: longer + partial unfreeze)
    p2_epochs = 500
    p2_lr = 1e-4  # lower LR for fine-tuning

    train_np, val_np, test_np = load_data_for_ticker(ticker, dataloc, etflistloc, tr, vl, 1, l, 1)
    train = torch.from_numpy(train_np).float().to(dev)
    val = torch.from_numpy(val_np).float().to(dev)
    test = torch.from_numpy(test_np).float().to(dev)

    mu_cond = train[:, :l].mean(0)
    sd_cond = train[:, :l].std(0, unbiased=False) + 1e-8
    mu_x = float(train[:, l].mean())
    sd_x = float(train[:, l].std(unbiased=False)) + 1e-8

    cond_train = (train[:, :l] - mu_cond) / sd_cond
    x1_train = (train[:, l:l+1] - mu_x) / sd_x
    x1_real_train = train[:, l:l+1]
    N = cond_train.shape[0]

    results = []
    t0_ticker = time.time()

    # ═══ Phase 1: FM pretrain (shared across all financial losses) ═══
    torch.manual_seed(1000 + ti)
    np.random.seed(1000 + ti)

    model = make_model(l, hidden, depth, dropout, dev)
    opt = optim.AdamW(model.parameters(), lr=p1_lr, weight_decay=0.01, betas=(0.9, 0.95))
    ema = EMA(model, decay=0.999)

    warmup = max(10, p1_epochs // 50)
    for ep in range(1, p1_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=dev)
        c = cond_train[perm] + 0.02 * torch.randn(N, l, device=dev)
        x = x1_train[perm]

        mult = (ep / warmup) if ep < warmup else 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / max(1, p1_epochs - warmup)))
        for pg in opt.param_groups:
            pg["lr"] = p1_lr * mult

        opt.zero_grad(set_to_none=True)
        loss = fm_batch_loss(model, x, c)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

    # Save Phase 1 checkpoint (shared base for all Phase 2 variants)
    p1_state = {k: v.detach().clone() for k, v in ema.shadow.items()}

    # Evaluate FM_only (Phase 1 only, no Phase 2) with full Evaluation2
    model.load_state_dict(p1_state)
    model.eval()
    gen = FlowGenAdapter(model, mu_cond, sd_cond, mu_x, sd_x, 40, 1).to(dev)

    df_p1, PnL_p1, _, _, _, _, _, _ = Evaluation2_gpu(
        ticker=ticker, freq=2, gen=gen, test_data=test, val_data=val,
        h=1, l=l, pred=1, hid_d=8, hid_g=8, z_dim=8,
        lrg=1e-4, lrd=1e-4, n_epochs=100,
        losstype="FM_only", sr_val=0, device=dev,
        plotsloc=os.path.join(loc, "Plots") + "/",
        f_name=f"{ticker}-FM_only", plot=False, nsamp=1000,
    )
    pd.DataFrame(PnL_p1).to_csv(os.path.join(loc, "PnLs", f"{ticker}-FM_only.csv"), index=False, header=False)

    results.append({
        "ticker": ticker, "type": "FM_only (P1)",
        "SR_w scaled val": float(df_p1["SR_w scaled val"].iloc[0]),
        "SR_w scaled": float(df_p1["SR_w scaled"].iloc[0]),
        "PnL_w": float(df_p1["PnL_w"].iloc[0]),
        "RMSE": float(df_p1["RMSE"].iloc[0]),
        "MAE": float(df_p1["MAE"].iloc[0]),
        "Corr": float(df_p1["Corr"].iloc[0]),
    })

    # ═══ Phase 2: Fine-tune output head with financial loss ═══
    for fin_name, fin_fn in FIN_LOSSES.items():
        # Restore from Phase 1
        model.load_state_dict(p1_state)

        # Partial unfreeze: last ResBlock + output head (v2)
        n_blocks = len(model.blocks)
        last_block_prefix = f"blocks.{n_blocks - 1}."
        for name, param in model.named_parameters():
            if "out" in name or name.startswith(last_block_prefix):
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        # Phase 2 optimizer (only trainable params)
        opt2 = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=p2_lr, weight_decay=0.01
        )

        best_val_sr = -1e9
        best_state = None
        evals_since_best = 0

        p2_warmup = max(10, p2_epochs // 50)
        for ep in range(1, p2_epochs + 1):
            model.train()
            perm = torch.randperm(N, device=dev)
            c = cond_train[perm]
            x = x1_train[perm]
            x_real = x1_real_train[perm]

            # Cosine LR schedule for Phase 2
            if ep < p2_warmup:
                lr_mult = ep / p2_warmup
            else:
                lr_mult = 0.5 * (1.0 + np.cos(np.pi * (ep - p2_warmup) / max(1, p2_epochs - p2_warmup)))
            for pg in opt2.param_groups:
                pg["lr"] = p2_lr * lr_mult

            # Forward through full model (frozen layers + trainable layers)
            B = x.shape[0]
            t = torch.rand((B,), device=dev).clamp(1e-4, 1.0 - 1e-4)
            x0 = torch.randn_like(x)
            x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x
            v_pred = model(x_t, c, t)

            # Estimated x1 in real space
            x1_hat = (x_t + (1.0 - t)[:, None] * v_pred) * sd_x + mu_x

            # Pure financial loss (gradients flow through unfrozen layers)
            opt2.zero_grad(set_to_none=True)
            loss = fin_fn(x1_hat, x_real)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

            if ep % 10 == 0:
                val_sr = fast_val_sr(model, val, l, mu_cond, sd_cond, mu_x, sd_x, dev)
                if val_sr > best_val_sr:
                    best_val_sr = val_sr
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    evals_since_best = 0
                else:
                    evals_since_best += 1
                    if evals_since_best >= 15:  # wider patience for 500 epochs
                        break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Full test eval with PnL (same as FinGAN Evaluation2)
        model.load_state_dict(best_state)
        model.eval()
        gen = FlowGenAdapter(model, mu_cond, sd_cond, mu_x, sd_x, 40, 1).to(dev)

        loss_tag = f"P2_{fin_name}"
        df_eval, PnL_test, PnL_even, PnL_odd, means_gen, reals_test, _, _ = Evaluation2_gpu(
            ticker=ticker, freq=2, gen=gen, test_data=test, val_data=val,
            h=1, l=l, pred=1, hid_d=8, hid_g=8, z_dim=8,
            lrg=1e-4, lrd=1e-4, n_epochs=200,
            losstype=loss_tag, sr_val=0, device=dev,
            plotsloc=os.path.join(loc, "Plots") + "/",
            f_name=f"{ticker}-{loss_tag}", plot=False, nsamp=1000,
        )

        # Save PnL CSV
        pnl_path = os.path.join(loc, "PnLs", f"{ticker}-{loss_tag}.csv")
        pd.DataFrame(PnL_test).to_csv(pnl_path, index=False, header=False)

        # Extract metrics from Evaluation2 output
        test_sr = float(df_eval["SR_w scaled"].iloc[0]) if "SR_w scaled" in df_eval.columns else 0.0
        test_pnl = float(df_eval["PnL_w"].iloc[0]) if "PnL_w" in df_eval.columns else 0.0
        test_sr_val = float(df_eval["SR_w scaled val"].iloc[0]) if "SR_w scaled val" in df_eval.columns else best_val_sr
        test_rmse = float(df_eval["RMSE"].iloc[0]) if "RMSE" in df_eval.columns else 0.0
        test_mae = float(df_eval["MAE"].iloc[0]) if "MAE" in df_eval.columns else 0.0
        test_corr = float(df_eval["Corr"].iloc[0]) if "Corr" in df_eval.columns else 0.0

        results.append({
            "ticker": ticker, "type": loss_tag,
            "SR_w scaled val": test_sr_val, "SR_w scaled": test_sr,
            "PnL_w": test_pnl, "RMSE": test_rmse, "MAE": test_mae, "Corr": test_corr,
            "trainable_params": trainable, "total_params": total,
        })

    elapsed = time.time() - t0_ticker
    best = max(results, key=lambda r: r["SR_w scaled val"])
    print(f"[GPU{gpu_id}] {ticker} done {elapsed:.0f}s | "
          f"best={best['type']} val_SR={best['SR_w scaled val']:+.3f} "
          f"test_SR={best['SR_w scaled']:+.3f}", flush=True)

    return results


def worker(gpu_id, assignments, result_dict):
    rows = []
    for ti, ticker in assignments:
        rows.extend(train_one(gpu_id, ticker, ti))
    result_dict[gpu_id] = rows


if __name__ == "__main__":
    mp.set_start_method("spawn")

    for d in ["TrainedModels", "Results", "PnLs", "Plots"]:
        os.makedirs(os.path.join(loc, d), exist_ok=True)

    n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))
    cfg = MODEL_CONFIGS[ACTIVE_CONFIG]
    print(f"Config: {ACTIVE_CONFIG} (H={cfg['hidden']}, D={cfg['depth']})", flush=True)
    print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}", flush=True)
    print(f"Phase 1: 100 ep FM pretrain | Phase 2: 500 ep financial fine-tune (last block + out)", flush=True)
    print(f"Financial losses: {len(FIN_LOSSES)} + FM_only = {len(FIN_LOSSES)+1} variants", flush=True)

    assignments = [[] for _ in range(n_gpus)]
    for i, t in enumerate(TICKERS):
        assignments[i % n_gpus].append((i, t))

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
        df.to_csv(os.path.join(loc, "Results", "flowfm_2phase_full.csv"), index=False)

        best = df.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False]).groupby("ticker").head(1)
        best.to_csv(os.path.join(loc, "Results", "flowfm_2phase_best.csv"), index=False)

        total = time.time() - t0
        print(f"\nAll done in {total:.0f}s ({len(all_rows)} models)", flush=True)
        print(f"\nBest per ticker:", flush=True)
        for _, r in best.iterrows():
            print(f"  {r['ticker']:5s} | {r['type']:15s} | val={r['SR_w scaled val']:+.3f} test={r['SR_w scaled']:+.3f}", flush=True)
