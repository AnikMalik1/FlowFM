"""Optimized PnL training: 4 GPUs, BSZ=N, fast val (fewer ODE steps + samples)."""
import os, sys, time, numpy as np, torch, torch.optim as optim
import torch.multiprocessing as mp

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff(); plt.show = lambda *a, **k: None

sys.path.insert(0, "/projects/s5e/quant/fingan/colabFinGAN_deprecated")
sys.path.insert(0, "/projects/s5e/quant/fingan/FlowFM_repo")

import FinGAN
from flow_adapter_aux import CondFlowNet, fm_batch_loss, fm_batch_loss_pnl, FlowGenAdapter, EMA
from run_flowfm_aux_31 import load_data_for_ticker, make_model

MODEL_TAG = "FlowFMPlus_PnL_v7"
dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-PnL/"

for d in ["PnLs", "Results", "Plots", "TrainedModels"]:
    os.makedirs(os.path.join(loc, d), exist_ok=True)


@torch.no_grad()
def fast_val_sr(model, val_data, l, mu_cond, sd_cond, mu_x, sd_x, dev,
                nsamp=64, ode_steps=10):
    """Ultra-fast validation SR: fewer samples + fewer ODE steps."""
    model.eval()
    T = val_data.shape[0]
    cond = (val_data[:, :l] - mu_cond) / sd_cond
    real = val_data[:, -1]

    # Generate all samples in one batch: (T * nsamp, 1)
    cond_rep = cond.repeat(nsamp, 1)  # (T*nsamp, l)
    x = torch.randn(T * nsamp, 1, device=dev)
    dt = 1.0 / ode_steps
    for k in range(ode_steps):
        t = torch.full((T * nsamp,), (k + 0.5) * dt, device=dev)
        x = x + dt * model(x, cond_rep, t)
    x = x * sd_x + mu_x  # denormalize

    samples = x.squeeze(-1).view(nsamp, T).T  # (T, nsamp)
    pu = (samples >= 0).float().mean(dim=1)
    pnl_ws = 10000.0 * (2.0 * pu - 1.0) * real
    Td = T // 2
    pnl_wd = pnl_ws[:2*Td].reshape(Td, 2).sum(dim=1)
    mu = pnl_wd.mean().item()
    sd = pnl_wd.std(unbiased=False).item() + 1e-12
    return (mu / sd) * np.sqrt(252.0)


def train_one(gpu_id, ticker, ti):
    assert torch.cuda.is_available(), f"CUDA not available on GPU {gpu_id}"
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
    torch.manual_seed(1000 + ti)
    np.random.seed(1000 + ti)

    l, h, pred, tr, vl = 10, 1, 1, 0.8, 0.1
    max_epochs, lr = 200, 3e-4
    hidden, depth, dropout = 256, 4, 0.05
    ema_decay, cond_noise_std = 0.999, 0.02
    eval_every, patience_evals = 10, 8
    lambda_pnl, pnl_warmup, pnl_ramp = 1000.0, 50, 20

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
    N = cond_train.shape[0]

    model = make_model(l, hidden, depth, dropout, dev)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    ema = EMA(model, decay=ema_decay)

    total_steps = max_epochs
    warmup_steps = max(10, total_steps // 50)

    best_val_sr, best_state, evals_since_best = -1e9, None, 0
    best_path = os.path.join(loc, "TrainedModels", f"{ticker}_{MODEL_TAG}_best.pt")
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=dev)
        c = cond_train[perm]
        x = x1_train[perm]

        if cond_noise_std > 0:
            c = c + cond_noise_std * torch.randn_like(c)

        if ep < warmup_steps:
            mult = ep / warmup_steps
        else:
            mult = 0.5 * (1.0 + np.cos(np.pi * (ep - warmup_steps) / max(1, total_steps - warmup_steps)))
        for pg in opt.param_groups:
            pg["lr"] = lr * mult

        if ep <= pnl_warmup:
            lam_eff = 0.0
        else:
            lam_eff = lambda_pnl * min(1.0, (ep - pnl_warmup) / max(1, pnl_ramp))

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if lam_eff > 0:
                x_real = x * sd_x + mu_x
                loss, l_fm, l_pnl = fm_batch_loss_pnl(model, x, c, x_real, lambda_pnl=lam_eff)
            else:
                loss = fm_batch_loss(model, x, c)
                l_fm, l_pnl = loss, torch.tensor(0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if ep % 10 == 0:
            tag = f"+PnL(λ={lam_eff:.0f})" if lam_eff > 0 else "FM"
            print(f"[GPU{gpu_id}] {ticker} ep{ep:3d} | FM={l_fm.item():.4f} PnL={l_pnl.item():.6f} | {tag}", flush=True)

        # Fast validation: 64 samples, 10 ODE steps (vs 256 samples, 40 steps before)
        if ep % eval_every == 0:
            eval_state = {k: v.clone() for k, v in ema.shadow.items()}
            orig_state = model.state_dict()
            model.load_state_dict(eval_state)
            val_sr = fast_val_sr(model, val, l, mu_cond, sd_cond, mu_x, sd_x, dev,
                                 nsamp=64, ode_steps=10)
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
                    print(f"[GPU{gpu_id}] {ticker} early stop ep{ep}", flush=True)
                    break

    elapsed = time.time() - t0
    print(f"[GPU{gpu_id}] {ticker} done {elapsed:.1f}s | best SR: {best_val_sr:.4f}", flush=True)

    # Fast pu check: 256 samples (not 1000), 10 ODE steps (not 40)
    model.load_state_dict(torch.load(best_path, map_location=dev))
    model.eval()
    T = test.shape[0]
    cond_test = (test[:, :l] - mu_cond) / sd_cond

    nsamp_test = 256
    cond_rep = cond_test.repeat(nsamp_test, 1)
    with torch.no_grad():
        x = torch.randn(T * nsamp_test, 1, device=dev)
        dt = 1.0 / 10
        for k in range(10):
            t = torch.full((T * nsamp_test,), (k + 0.5) * dt, device=dev)
            x = x + dt * model(x, cond_rep, t)
        x = x * sd_x + mu_x

    samples = x.squeeze(-1).view(nsamp_test, T).T
    pu = (samples >= 0).float().mean(dim=1).cpu().numpy()
    pos = 2 * pu - 1
    print(f"[GPU{gpu_id}] {ticker} pu: mean={pu.mean():.4f} std={pu.std():.4f} "
          f"mid={((pu>0.4)&(pu<0.6)).mean()*100:.0f}% "
          f"|pos|={np.abs(pos).mean():.4f}", flush=True)


def worker(gpu_id, assignments):
    for ti, ticker in assignments:
        train_one(gpu_id, ticker, ti)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    TICKERS = [
        "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
        "WFC","GS","BLK","PFE","HUM","FDX","GD",
        "IBM","TER","ECL","IP","DTE","WEC",
        "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
    ]

    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}", flush=True)

    assignments = [[] for _ in range(n_gpus)]
    for i, t in enumerate(TICKERS):
        assignments[i % n_gpus].append((i, t))

    for g in range(n_gpus):
        tks = [t for _, t in assignments[g]]
        print(f"  GPU {g}: {tks}", flush=True)

    t0 = time.time()
    procs = []
    for g in range(n_gpus):
        if not assignments[g]:
            continue
        p = mp.Process(target=worker, args=(g, assignments[g]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print(f"\nAll {len(TICKERS)} tickers done in {time.time()-t0:.0f}s.", flush=True)
