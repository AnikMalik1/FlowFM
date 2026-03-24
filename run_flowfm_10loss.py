"""FlowFMPlus with 10 loss variants (fair comparison to FinGAN). 4-GPU, BSZ=N."""
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

_cfg_name = os.environ.get("FLOWFM_CONFIG", "medium")
MODEL_TAG = f"FlowFM_10loss_{_cfg_name}"
dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = f"/projects/s5e/quant/fingan/FlowFM/FlowFM-10loss-{_cfg_name}/"

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]

# ── 10 Loss Functions (same as FinGAN, adapted for flow matching) ────────
# All losses take: v_pred (B,1), v_target (B,1), x1_hat (B,1), x1_real (B,1)
# x1_hat = x_t + (1-t)*v_pred  (estimated sample in normalized space)
# x1_real = denormalized target return

def loss_fm_only(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    """Pure flow matching (equivalent to ForGAN baseline = only distribution learning)."""
    return torch.mean((v_pred - v_target) ** 2)

def loss_mse(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    """Velocity MSE + prediction MSE."""
    fm = torch.mean((v_pred - v_target) ** 2)
    mse = torch.mean((x1_hat - x1_real) ** 2)
    return fm + mse

def loss_pnl(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    """Velocity MSE + PnL."""
    fm = torch.mean((v_pred - v_target) ** 2)
    ft = torch.tanh(tanh_c * x1_hat.squeeze(-1))
    pnl = -torch.mean(ft * x1_real.squeeze(-1))
    return fm + 1000.0 * pnl

def loss_sr(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    """Velocity MSE + Sharpe proxy."""
    fm = torch.mean((v_pred - v_target) ** 2)
    ft = torch.tanh(tanh_c * x1_hat.squeeze(-1))
    pnl_vec = ft * x1_real.squeeze(-1)
    sr = -(torch.mean(pnl_vec) / (torch.std(pnl_vec) + 1e-8))
    return fm + 1000.0 * sr

def loss_pnlmse(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    return loss_pnl(v_pred, v_target, x1_hat, x1_real, tanh_c) + \
           torch.mean((x1_hat - x1_real) ** 2)

def loss_pnlsr(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    fm = torch.mean((v_pred - v_target) ** 2)
    ft = torch.tanh(tanh_c * x1_hat.squeeze(-1))
    r = x1_real.squeeze(-1)
    pnl = -torch.mean(ft * r)
    pnl_vec = ft * r
    sr = -(torch.mean(pnl_vec) / (torch.std(pnl_vec) + 1e-8))
    return fm + 1000.0 * (pnl + sr)

def loss_pnlstd(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    fm = torch.mean((v_pred - v_target) ** 2)
    ft = torch.tanh(tanh_c * x1_hat.squeeze(-1))
    pnl_vec = ft * x1_real.squeeze(-1)
    return fm + 1000.0 * (-torch.mean(pnl_vec) + torch.std(pnl_vec))

def loss_pnlmsesr(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    return loss_pnlmse(v_pred, v_target, x1_hat, x1_real, tanh_c) + \
           loss_sr(v_pred, v_target, x1_hat, x1_real, tanh_c)

def loss_pnlmsestd(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    return loss_pnlmse(v_pred, v_target, x1_hat, x1_real, tanh_c) + \
           loss_pnlstd(v_pred, v_target, x1_hat, x1_real, tanh_c)

def loss_srmse(v_pred, v_target, x1_hat, x1_real, tanh_c=1.0):
    return loss_sr(v_pred, v_target, x1_hat, x1_real, tanh_c) + \
           torch.mean((x1_hat - x1_real) ** 2)

LOSS_FUNCTIONS = {
    "FM_only": loss_fm_only, "MSE": loss_mse, "PnL": loss_pnl,
    "PnLMSE": loss_pnlmse, "PnLMSESR": loss_pnlmsesr,
    "PnLMSESTD": loss_pnlmsestd, "PnLSR": loss_pnlsr,
    "PnLSTD": loss_pnlstd, "SR": loss_sr, "SRMSE": loss_srmse,
}

# ── Fast validation SR ──────────────────────────────────────────────────
@torch.no_grad()
def fast_val_sr(model, val_data, l, mu_cond, sd_cond, mu_x, sd_x, dev, nsamp=64, ode_steps=10):
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


MODEL_CONFIGS = {
    "small":  {"hidden": 32, "depth": 2},   # 16K params
    "medium": {"hidden": 64, "depth": 2},   # 25K params (recommended)
    "large":  {"hidden": 64, "depth": 4},   # 51K params
}

# Select via env var: FLOWFM_CONFIG=small|medium|large (default: medium)
ACTIVE_CONFIG = os.environ.get("FLOWFM_CONFIG", "medium")


def train_one(gpu_id, ticker, ti):
    assert torch.cuda.is_available()
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")

    cfg = MODEL_CONFIGS[ACTIVE_CONFIG]
    l, h, pred, tr, vl = 10, 1, 1, 0.8, 0.1
    max_epochs, lr = 200, 3e-4
    hidden, depth, dropout = cfg["hidden"], cfg["depth"], 0.05
    ema_decay, cond_noise_std = 0.999, 0.02
    eval_every, patience_evals = 10, 8

    train_np, val_np, test_np = load_data_for_ticker(ticker, dataloc, etflistloc, tr, vl, h, l, pred)
    train = torch.from_numpy(train_np).float().to(dev)
    val = torch.from_numpy(val_np).float().to(dev)
    test = torch.from_numpy(test_np).float().to(dev)

    mu_cond = train[:, :l].mean(0)
    sd_cond = train[:, :l].std(0, unbiased=False) + 1e-8
    mu_x = float(train[:, l].mean())
    sd_x = float(train[:, l].std(unbiased=False)) + 1e-8

    cond_train = (train[:, :l] - mu_cond) / sd_cond
    x1_train = (train[:, l:l+1] - mu_x) / sd_x
    x1_real_train = train[:, l:l+1]  # denormalized
    N = cond_train.shape[0]
    total_steps = max_epochs
    warmup_steps = max(10, total_steps // 50)

    results = []
    t0_ticker = time.time()

    for loss_name, loss_fn in LOSS_FUNCTIONS.items():
        torch.manual_seed(1000 + ti)
        np.random.seed(1000 + ti)

        model = make_model(l, hidden, depth, dropout, dev)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        ema = EMA(model, decay=ema_decay)

        best_val_sr, best_state, evals_since_best = -1e9, None, 0
        best_path = os.path.join(loc, "TrainedModels", f"{ticker}_{loss_name}_best.pt")

        for ep in range(1, max_epochs + 1):
            model.train()
            perm = torch.randperm(N, device=dev)
            c = cond_train[perm]
            x = x1_train[perm]
            x_real = x1_real_train[perm]

            if cond_noise_std > 0:
                c = c + cond_noise_std * torch.randn_like(c)

            # LR schedule
            if ep < warmup_steps:
                mult = ep / warmup_steps
            else:
                mult = 0.5 * (1.0 + np.cos(np.pi * (ep - warmup_steps) / max(1, total_steps - warmup_steps)))
            for pg in opt.param_groups:
                pg["lr"] = lr * mult

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Standard FM forward
                B = x.shape[0]
                t = torch.rand((B,), device=dev).clamp(1e-4, 1.0 - 1e-4)
                x0 = torch.randn_like(x)
                x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x
                v_target = x - x0
                v_pred = model(x_t, c, t)

                # Estimated x1 for PnL computation
                x1_hat = (x_t + (1.0 - t)[:, None] * v_pred) * sd_x + mu_x

                loss = loss_fn(v_pred, v_target, x1_hat, x_real)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            if ep % eval_every == 0:
                eval_state = {k: v.clone() for k, v in ema.shadow.items()}
                orig_state = model.state_dict()
                model.load_state_dict(eval_state)
                val_sr = fast_val_sr(model, val, l, mu_cond, sd_cond, mu_x, sd_x, dev)
                model.load_state_dict(orig_state)
                model.train()

                if val_sr > best_val_sr:
                    best_val_sr = val_sr
                    best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
                    torch.save(best_state, best_path)
                    evals_since_best = 0
                else:
                    evals_since_best += 1
                    if evals_since_best >= patience_evals:
                        break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
            torch.save(best_state, best_path)

        # Test eval
        model.load_state_dict(torch.load(best_path, map_location=dev))
        test_sr = fast_val_sr(model, test, l, mu_cond, sd_cond, mu_x, sd_x, dev, nsamp=256, ode_steps=10)

        results.append({
            "ticker": ticker, "type": loss_name,
            "SR_w scaled val": best_val_sr,
            "SR_w scaled": test_sr,
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

    for d in ["TrainedModels", "Results"]:
        os.makedirs(os.path.join(loc, d), exist_ok=True)

    n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))
    cfg = MODEL_CONFIGS[ACTIVE_CONFIG]
    print(f"Config: {ACTIVE_CONFIG} (H={cfg['hidden']}, D={cfg['depth']})", flush=True)
    print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}, Loss variants: {len(LOSS_FUNCTIONS)}", flush=True)
    print(f"Total models: {len(TICKERS) * len(LOSS_FUNCTIONS)}", flush=True)

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
        df.to_csv(os.path.join(loc, "Results", "flowfm_10loss_full.csv"), index=False)

        best = df.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False]).groupby("ticker").head(1)
        best.to_csv(os.path.join(loc, "Results", "flowfm_10loss_best.csv"), index=False)

        total = time.time() - t0
        print(f"\nAll done in {total:.0f}s ({len(all_rows)} models)", flush=True)
        print(f"\nBest per ticker:", flush=True)
        for _, r in best.iterrows():
            print(f"  {r['ticker']:5s} | {r['type']:12s} | val={r['SR_w scaled val']:+.3f} test={r['SR_w scaled']:+.3f}", flush=True)
