"""
Trainer — train FlowFMPlus with a given loss function, return metrics.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim

# Add FlowFM_repo (GitHub clone) to path for FinGAN/flow_adapter code
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FlowFM_repo"))

import FinGAN
from flow_adapter_aux import CondFlowNet, FlowGenAdapter, EMA, fm_sample_from_x0
import config


def _load_data(ticker, tr=0.8, vl=0.1, h=1, l=10, pred=1):
    if ticker[0] == "X":
        train_np, val_np, test_np, _ = FinGAN.split_train_val_testraw(
            ticker, config.DATA_DIR, tr=tr, vl=vl, h=h, l=l, pred=pred, plotcheck=False
        )
    else:
        train_np, val_np, test_np, _ = FinGAN.split_train_val_test(
            ticker, config.DATA_DIR, config.ETF_LIST,
            tr=tr, vl=vl, h=h, l=l, pred=pred, plotcheck=False
        )
    return train_np, val_np, test_np


def _fast_val_sr(gen, val_data, l, z_dim=8, hid_g=8, nsamp=256, chunk=8, device=None):
    """Fast validation SR using Evaluation2 MC logic."""
    gen.eval()
    val_data = val_data.to(device).float()
    T = val_data.shape[0]
    cond = val_data[:, :l].contiguous()
    real = val_data[:, -1].contiguous()

    samples = torch.empty((T, nsamp), device=device)
    done = 0
    while done < nsamp:
        cur = min(chunk, nsamp - done)
        eff = T * cur
        noise = torch.randn((1, eff, z_dim), device=device)
        cond_rep = cond.repeat_interleave(cur, dim=0).unsqueeze(0).contiguous()
        h0 = torch.zeros((1, eff, hid_g), device=device)
        c0 = torch.zeros((1, eff, hid_g), device=device)
        out = gen(noise, cond_rep, h0, c0).reshape(T, cur)
        samples[:, done:done + cur] = out
        done += cur

    pu = (samples >= 0).float().mean(dim=1)
    pnl = 10000.0 * (2.0 * pu - 1.0) * real
    Td = int(0.5 * T)
    pnl_d = pnl[:2 * Td].reshape(Td, 2).sum(dim=1)
    mu = pnl_d.mean().item()
    sd = pnl_d.std(unbiased=False).item() + 1e-12
    return (mu / sd) * np.sqrt(252.0)


def _compute_custom_loss(model, x1_batch, cond_batch, loss_fn, epoch):
    """
    Compute custom loss with the standard interface.

    Flow matching setup:
        x_t = (1-t)*x0 + t*x1,  x0 ~ N(0,1)
        v_target = x1 - x0
        v_pred = model(x_t, cond, t)
        x_pred = x0 + v_pred  (one-step approximation of generated sample)
    """
    B = x1_batch.shape[0]
    t = torch.rand((B,), device=x1_batch.device).clamp(1e-4, 1.0 - 1e-4)
    x0 = torch.randn_like(x1_batch)
    x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1_batch
    v_target = x1_batch - x0
    v_pred = model(x_t, cond_batch, t)

    # One-step approximation: x_pred = x0 + v_pred
    x_pred = x0 + v_pred

    return loss_fn(
        v_pred=v_pred,
        v_target=v_target,
        x_pred=x_pred,
        x_real=x1_batch,
        condition=cond_batch,
        epoch=epoch,
    )


def train_single_ticker(
    ticker: str,
    loss_fn,
    max_epochs: int = None,
    eval_every: int = None,
    patience: int = None,
    device=None,
    verbose: bool = True,
) -> dict:
    """
    Train FlowFMPlus on one ticker with custom loss_fn.
    Returns dict with {val_sr, train_time_s, best_epoch, final_loss}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if max_epochs is None:
        max_epochs = config.STAGE1_MAX_EPOCHS
    if eval_every is None:
        eval_every = config.STAGE1_EVAL_EVERY
    if patience is None:
        patience = config.STAGE1_PATIENCE

    l = config.LOOKBACK
    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    train_np, val_np, _ = _load_data(ticker, l=l)
    train_t = torch.from_numpy(train_np).float().to(device)
    val_t = torch.from_numpy(val_np).float().to(device)

    cond_train = train_t[:, :l]
    x_train = train_t[:, l:l + 1]

    mu_cond = cond_train.mean(dim=0)
    sd_cond = cond_train.std(dim=0, unbiased=False) + 1e-8
    mu_x = float(x_train.mean().item())
    sd_x = float(x_train.std(unbiased=False).item() + 1e-8)

    cond_n = (cond_train - mu_cond) / sd_cond
    x1_n = (x_train - mu_x) / sd_x

    # Model
    model = CondFlowNet(
        cond_dim=l, hidden=config.HIDDEN, depth=config.DEPTH,
        t_dim=64, dropout=config.DROPOUT,
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=config.LR,
                      weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.95))
    ema = EMA(model, decay=config.EMA_DECAY)

    N = cond_n.shape[0]
    bs = config.BATCH_SIZE
    total_steps = int(np.ceil(N / bs)) * max_epochs
    warmup = max(100, total_steps // 50)

    def lr_mult(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    best_val_sr = -1e9
    best_epoch = 0
    evals_no_improve = 0
    global_step = 0
    t0 = time.time()
    use_amp = device.type == "cuda"

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        c = cond_n[perm]
        x = x1_n[perm]
        losses = []

        for i in range(0, N, bs):
            c_b = c[i:i + bs]
            x_b = x[i:i + bs]
            if config.COND_NOISE_STD > 0:
                c_b = c_b + config.COND_NOISE_STD * torch.randn_like(c_b)

            mult = lr_mult(global_step)
            for pg in opt.param_groups:
                pg["lr"] = config.LR * mult

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = _compute_custom_loss(model, x_b, c_b, loss_fn, ep)
            else:
                loss = _compute_custom_loss(model, x_b, c_b, loss_fn, ep)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            global_step += 1
            losses.append(float(loss.item()))

        if ep % eval_every == 0:
            eval_model = CondFlowNet(
                cond_dim=l, hidden=config.HIDDEN, depth=config.DEPTH,
                t_dim=64, dropout=config.DROPOUT,
            ).to(device)
            eval_model.load_state_dict(ema.shadow, strict=True)
            eval_model.eval()

            gen = FlowGenAdapter(
                flow_model=eval_model, mu_cond=mu_cond, sd_cond=sd_cond,
                mu_x=mu_x, sd_x=sd_x, ode_steps=config.ODE_STEPS, pred=1,
            ).to(device)

            val_sr = _fast_val_sr(
                gen, val_t, l, device=device,
                nsamp=config.VAL_MC_SAMPLES, chunk=config.VAL_MC_CHUNK,
            )

            if verbose:
                mean_loss = float(np.mean(losses))
                print(f"  {ticker} ep={ep:4d} loss={mean_loss:.5f} val_SR={val_sr:.4f}")

            if val_sr > best_val_sr:
                best_val_sr = val_sr
                best_epoch = ep
                evals_no_improve = 0
            else:
                evals_no_improve += 1
                if evals_no_improve >= patience:
                    if verbose:
                        print(f"  {ticker} early stopping at epoch {ep}")
                    break

    return {
        "ticker": ticker,
        "val_sr": best_val_sr,
        "best_epoch": best_epoch,
        "train_time_s": round(time.time() - t0, 1),
        "final_loss": float(np.mean(losses)) if losses else 0.0,
    }
