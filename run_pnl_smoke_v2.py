"""Smoke test for PnL auxiliary loss (linear proxy). Run via srun on GPU node."""
import os, sys, time, numpy as np, torch, torch.optim as optim

# CUDA assertion
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE - fix env setup"
device = torch.device("cuda")
print(f"GPU: {device} - {torch.cuda.get_device_name(0)}")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Patch matplotlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff(); plt.show = lambda *a, **k: None

sys.path.insert(0, "/projects/s5e/quant/fingan/colabFinGAN_deprecated")
sys.path.insert(0, "/projects/s5e/quant/fingan/FlowFM_repo")

import FinGAN
from flow_adapter_aux import CondFlowNet, fm_batch_loss, fm_batch_loss_pnl, FlowGenAdapter, EMA
from run_flowfm_aux_31 import load_data_for_ticker, make_model, fast_val_sr_eval2_logic

TICKERS = ["AMZN", "APA"]
MODEL_TAG = "FlowFMPlus_PnL_linear"
dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-PnL/"

for d in ["PnLs", "Results", "Plots", "TrainedModels"]:
    os.makedirs(os.path.join(loc, d), exist_ok=True)

# Config
l, pred, h = 10, 1, 1
tr, vl = 0.8, 0.1
max_epochs = 200
batch_size = 4096
lr = 3e-4
ode_steps = 40
hidden, depth, dropout = 256, 4, 0.05
ema_decay = 0.999
cond_noise_std = 0.02
eval_every = 10
patience_evals = 8
hid_g, z_dim = 8, 8

# PnL config (ODE-through PnL, like FinGAN's direct sample loss)
lambda_pnl = 1000.0        # FM~1.0, PnL~0.001 → need 1000x to balance
pnl_warmup_epochs = 50
pnl_ramp_epochs = 20

for ti, ticker in enumerate(TICKERS):
    print(f"\n===== {ticker} ({ti+1}/{len(TICKERS)}) =====", flush=True)
    torch.manual_seed(1000 + ti)
    np.random.seed(1000 + ti)

    train_np, val_np, test_np = load_data_for_ticker(ticker, dataloc, etflistloc, tr, vl, h, l, pred)
    train = torch.from_numpy(train_np).float().to(device)
    val = torch.from_numpy(val_np).float().to(device)
    test = torch.from_numpy(test_np).float().to(device)

    mu_cond = train[:, :l].mean(dim=0)
    sd_cond = train[:, :l].std(dim=0, unbiased=False) + 1e-8
    mu_x = float(train[:, l].mean())
    sd_x = float(train[:, l].std(unbiased=False)) + 1e-8

    cond_train = (train[:, :l] - mu_cond) / sd_cond
    x1_train = (train[:, l:l+1] - mu_x) / sd_x

    model = make_model(l, hidden, depth, dropout, device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    ema = EMA(model, decay=ema_decay)

    N = cond_train.shape[0]
    total_steps = int(np.ceil(N / batch_size)) * max_epochs
    warmup_steps = max(100, total_steps // 50)

    def lr_mult(step):
        if step < warmup_steps: return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    global_step = 0
    best_val_sr = -1e9
    best_state = None
    evals_since_best = 0
    best_path = os.path.join(loc, "TrainedModels", f"{ticker}_{MODEL_TAG}_best.pt")
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        c = cond_train[perm]
        x = x1_train[perm]
        ep_fm, ep_pnl = [], []

        # Curriculum ramp
        if ep <= pnl_warmup_epochs:
            lambda_eff = 0.0
        else:
            lambda_eff = lambda_pnl * min(1.0, (ep - pnl_warmup_epochs) / max(1, pnl_ramp_epochs))

        for i in range(0, N, batch_size):
            c_b = c[i:i+batch_size]
            x_b = x[i:i+batch_size]
            if cond_noise_std > 0:
                c_b = c_b + cond_noise_std * torch.randn_like(c_b)

            mult = lr_mult(global_step)
            for pg in opt.param_groups: pg["lr"] = lr * mult
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if lambda_eff > 0:
                    x_b_real = x_b * sd_x + mu_x
                    loss, l_fm, l_pnl = fm_batch_loss_pnl(model, x_b, c_b, x_b_real, lambda_pnl=lambda_eff)
                else:
                    loss = fm_batch_loss(model, x_b, c_b)
                    l_fm, l_pnl = loss, torch.tensor(0.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            global_step += 1
            ep_fm.append(float(l_fm.item()))
            ep_pnl.append(float(l_pnl.item()))

        if ep % 10 == 0:
            tag = f"+PnL(λ={lambda_eff:.0f})" if lambda_eff > 0 else "FM only"
            print(f"{ticker} | ep {ep:3d} | FM={np.mean(ep_fm):.5f} PnL={np.mean(ep_pnl):.6f} | {tag}", flush=True)

        if ep % eval_every == 0:
            eval_model = make_model(l, hidden, depth, dropout, device)
            eval_model.load_state_dict(ema.shadow, strict=True)
            eval_model.eval()
            gen = FlowGenAdapter(eval_model, mu_cond, sd_cond, mu_x, sd_x, ode_steps, 1).to(device)
            val_sr = fast_val_sr_eval2_logic(gen, val, l, z_dim, hid_g, 256, 8, device)
            print(f"{ticker} | [val] ep {ep:3d} | SR={val_sr:.4f}", flush=True)

            if val_sr > best_val_sr:
                best_val_sr = val_sr
                best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
                torch.save(best_state, best_path)
                evals_since_best = 0
                print(f"{ticker} | NEW BEST SR: {best_val_sr:.4f}", flush=True)
            else:
                evals_since_best += 1
                if evals_since_best >= patience_evals:
                    print(f"{ticker} | early stopping", flush=True)
                    break

    elapsed = time.time() - t0
    print(f"{ticker} | done in {elapsed:.0f}s | best val SR: {best_val_sr:.4f}", flush=True)

    # pu check
    eval_model = make_model(l, hidden, depth, dropout, device)
    eval_model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
    eval_model.eval()
    gen = FlowGenAdapter(eval_model, mu_cond, sd_cond, mu_x, sd_x, ode_steps, 1).to(device)

    T = test.shape[0]
    cond_test = test[:, :l].unsqueeze(0).to(device)
    samples = torch.empty(T, 1000, device=device)
    with torch.no_grad():
        for s in range(0, 1000, 8):
            cur = min(8, 1000 - s)
            noise = torch.randn(1, T * cur, z_dim, device=device)
            cond_rep = cond_test.squeeze(0).repeat_interleave(cur, dim=0).unsqueeze(0)
            h0 = torch.zeros(1, T * cur, hid_g, device=device)
            c0 = torch.zeros(1, T * cur, hid_g, device=device)
            out = gen(noise, cond_rep, h0, c0).reshape(-1).view(T, cur)
            samples[:, s:s+cur] = out

    pu = (samples >= 0).float().mean(dim=1).cpu().numpy()
    pos = 2 * pu - 1
    print(f"{ticker} | TEST pu: mean={pu.mean():.4f} std={pu.std():.4f} min={pu.min():.3f} max={pu.max():.3f}", flush=True)
    print(f"{ticker} | pct 0.4<pu<0.6: {((pu>0.4)&(pu<0.6)).mean()*100:.1f}%", flush=True)
    print(f"{ticker} | mean|pos|: {np.abs(pos).mean():.4f}", flush=True)

print("\nSmoke test complete.", flush=True)
