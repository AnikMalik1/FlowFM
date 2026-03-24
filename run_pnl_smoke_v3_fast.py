"""Fast PnL smoke test: 4 GPUs parallel, BSZ=N, tanh(1.0) + lambda=1000."""
import os, sys, time, numpy as np, torch, torch.optim as optim
import torch.multiprocessing as mp

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff(); plt.show = lambda *a, **k: None

sys.path.insert(0, "/projects/s5e/quant/fingan/colabFinGAN_deprecated")
sys.path.insert(0, "/projects/s5e/quant/fingan/FlowFM_repo")

import FinGAN
from flow_adapter_aux import CondFlowNet, fm_batch_loss, fm_batch_loss_pnl, FlowGenAdapter, EMA
from run_flowfm_aux_31 import load_data_for_ticker, make_model, fast_val_sr_eval2_logic

MODEL_TAG = "FlowFMPlus_PnL_v6"
dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-PnL/"

for d in ["PnLs", "Results", "Plots", "TrainedModels"]:
    os.makedirs(os.path.join(loc, d), exist_ok=True)


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
    hid_g, z_dim = 8, 8
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

    total_steps = max_epochs  # BSZ=N → 1 batch/epoch
    warmup_steps = max(10, total_steps // 50)

    best_val_sr, best_state, evals_since_best = -1e9, None, 0
    best_path = os.path.join(loc, "TrainedModels", f"{ticker}_{MODEL_TAG}_best.pt")
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=dev)
        c = cond_train[perm]
        x = x1_train[perm]

        # Noise augmentation
        if cond_noise_std > 0:
            c = c + cond_noise_std * torch.randn_like(c)

        # LR schedule
        if ep < warmup_steps:
            mult = ep / warmup_steps
        else:
            mult = 0.5 * (1.0 + np.cos(np.pi * (ep - warmup_steps) / max(1, total_steps - warmup_steps)))
        for pg in opt.param_groups:
            pg["lr"] = lr * mult

        # Curriculum lambda
        if ep <= pnl_warmup:
            lam_eff = 0.0
        else:
            lam_eff = lambda_pnl * min(1.0, (ep - pnl_warmup) / max(1, pnl_ramp))

        opt.zero_grad(set_to_none=True)

        # SINGLE BATCH = all training data (BSZ=N, ~9K samples)
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

        if ep % eval_every == 0:
            eval_model = make_model(l, hidden, depth, dropout, dev)
            eval_model.load_state_dict(ema.shadow, strict=True)
            eval_model.eval()
            gen = FlowGenAdapter(eval_model, mu_cond, sd_cond, mu_x, sd_x, 40, 1).to(dev)
            val_sr = fast_val_sr_eval2_logic(gen, val, l, z_dim, hid_g, 256, 8, dev)

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
    print(f"[GPU{gpu_id}] {ticker} done {elapsed:.0f}s | best SR: {best_val_sr:.4f}", flush=True)

    # pu check
    eval_model = make_model(l, hidden, depth, dropout, dev)
    eval_model.load_state_dict(torch.load(best_path, map_location=dev), strict=True)
    eval_model.eval()
    gen = FlowGenAdapter(eval_model, mu_cond, sd_cond, mu_x, sd_x, 40, 1).to(dev)

    T = test.shape[0]
    samples = torch.empty(T, 1000, device=dev)
    with torch.no_grad():
        for s in range(0, 1000, 8):
            cur = min(8, 1000 - s)
            noise = torch.randn(1, T * cur, z_dim, device=dev)
            cond_rep = test[:, :l].repeat_interleave(cur, dim=0).unsqueeze(0)
            h0 = torch.zeros(1, T * cur, hid_g, device=dev)
            c0 = torch.zeros(1, T * cur, hid_g, device=dev)
            out = gen(noise, cond_rep, h0, c0).reshape(-1).view(T, cur)
            samples[:, s:s+cur] = out

    pu = (samples >= 0).float().mean(dim=1).cpu().numpy()
    pos = 2 * pu - 1
    print(f"[GPU{gpu_id}] {ticker} TEST pu: mean={pu.mean():.4f} std={pu.std():.4f} "
          f"min={pu.min():.3f} max={pu.max():.3f} "
          f"pct_mid={((pu>0.4)&(pu<0.6)).mean()*100:.1f}% "
          f"mean|pos|={np.abs(pos).mean():.4f}", flush=True)


def worker(gpu_id, assignments):
    for ti, ticker in assignments:
        train_one(gpu_id, ticker, ti)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    TICKERS = ["AMZN", "APA", "BLK", "PFE", "HD", "NKE", "KO", "GS"]
    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}", flush=True)

    # Distribute tickers across GPUs
    assignments = [[] for _ in range(n_gpus)]
    for i, t in enumerate(TICKERS):
        assignments[i % n_gpus].append((i, t))

    for g in range(n_gpus):
        tks = [t for _, t in assignments[g]]
        print(f"  GPU {g}: {tks}", flush=True)

    procs = []
    for g in range(n_gpus):
        if not assignments[g]:
            continue
        p = mp.Process(target=worker, args=(g, assignments[g]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("\nAll done.", flush=True)
