"""
Trainer — train FinGAN with a given financial loss, return metrics.
Uses original FinGAN Generator/Discriminator from FinGAN.py (zero rewrite).
Includes GradientCheck warmup for gradient norm balancing (from original paper).
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import FinGAN
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


def _param_grad_norm(model):
    """L2 norm of all parameter gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5


def _one_gen_step(gen, disc, gen_opt, criterion, financial_loss_fn,
                  cond_b, real_b, real_disc_in, tanh_temp,
                  h0g, c0g, h0d, c0d, z_dim, batch_size, device):
    """Single generator forward pass. Returns (bce_loss, fin_loss, fake_out)."""
    noise = torch.randn(1, batch_size, z_dim, device=device)
    fake_out = gen(noise, cond_b, h0g, c0g)
    fake_disc_in = torch.cat([cond_b, fake_out], dim=-1)
    d_fake = disc(fake_disc_in, h0d.detach().clone(), c0d.detach().clone())
    bce_loss = criterion(d_fake, torch.ones_like(d_fake))
    gen_returns = fake_out.squeeze()
    fin_loss = financial_loss_fn(gen_returns, real_b, tanh_temp)
    return bce_loss, fin_loss, fake_out


def _gradient_check_warmup(gen, disc, gen_opt, disc_opt, criterion,
                           financial_loss_fn, train_t, tanh_temp,
                           n_warmup, batch_size, l, z_dim, hid_g, hid_d, device):
    """
    GradientCheck warmup (from original FinGAN paper, FinGAN.py:793-931).
    Trains the model for n_warmup epochs while measuring gradient norms.
    Returns scale factor: median(||grad_BCE|| / ||grad_fin||).
    """
    N = train_t.shape[0]
    nbatches = N // batch_size
    bce_norms = []
    fin_norms = []

    gen.train()
    for ep in range(n_warmup):
        perm = torch.randperm(N, device=device)
        for b in range(nbatches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            cond_b = train_t[idx, :l].unsqueeze(0)
            real_b = train_t[idx, -1]
            real_disc_in = train_t[idx].unsqueeze(0)

            h0g = torch.zeros(1, batch_size, hid_g, device=device)
            c0g = torch.zeros(1, batch_size, hid_g, device=device)
            h0d = torch.zeros(1, batch_size, hid_d, device=device)
            c0d = torch.zeros(1, batch_size, hid_d, device=device)

            # Disc step (same as training)
            disc_opt.zero_grad()
            noise = torch.randn(1, batch_size, z_dim, device=device)
            with torch.no_grad():
                fake_out = gen(noise, cond_b, h0g, c0g)
            fake_disc_in = torch.cat([cond_b, fake_out], dim=-1)
            d_real = disc(real_disc_in, h0d.detach().clone(), c0d.detach().clone())
            d_fake = disc(fake_disc_in, h0d.detach().clone(), c0d.detach().clone())
            d_loss = (criterion(d_real, torch.ones_like(d_real)) +
                      criterion(d_fake, torch.zeros_like(d_fake))) / 2
            d_loss.backward()
            disc_opt.step()

            # Gen: measure BCE gradient norm
            bce_loss, fin_loss, _ = _one_gen_step(
                gen, disc, gen_opt, criterion, financial_loss_fn,
                cond_b, real_b, real_disc_in, tanh_temp,
                h0g, c0g, h0d, c0d, z_dim, batch_size, device)

            gen_opt.zero_grad()
            bce_loss.backward(retain_graph=True)
            bce_norm = _param_grad_norm(gen)

            # Gen: measure financial loss gradient norm
            gen_opt.zero_grad()
            fin_loss.backward()
            fin_norm = _param_grad_norm(gen)

            bce_norms.append(bce_norm)
            fin_norms.append(fin_norm + 1e-10)

            # Actual training step with equal weight (like original)
            gen_opt.zero_grad()
            bce2, fin2, _ = _one_gen_step(
                gen, disc, gen_opt, criterion, financial_loss_fn,
                cond_b, real_b, real_disc_in, tanh_temp,
                h0g, c0g, h0d, c0d, z_dim, batch_size, device)
            total = bce2 + fin2
            total.backward()
            gen_opt.step()

    # Compute scale factor: median is more robust than mean
    ratios = np.array(bce_norms) / np.array(fin_norms)
    scale = float(np.median(ratios))
    scale = max(0.01, min(100.0, scale))  # clip to sane range
    return scale


@torch.no_grad()
def _fast_val_sr_fingan(gen, data, l, z_dim, hid_g, device, nsamp=256):
    """Vectorized val SR for FinGAN generator."""
    gen.eval()
    data = data.to(device).float()
    T = data.shape[0]
    cond = data[:, :l].unsqueeze(0)
    real = data[:, -1]

    samples = torch.empty((T, nsamp), device=device)
    for i in range(nsamp):
        noise = torch.randn(1, T, z_dim, device=device)
        h0 = torch.zeros(1, T, hid_g, device=device)
        c0 = torch.zeros(1, T, hid_g, device=device)
        out = gen(noise, cond, h0, c0)
        samples[:, i] = out.squeeze()

    pu = (samples >= 0).float().mean(dim=1)
    pnl = 10000.0 * (2.0 * pu - 1.0) * real
    Td = T // 2
    pnl_d = pnl[:2 * Td].reshape(Td, 2).sum(dim=1)
    mu = pnl_d.mean().item()
    sd = pnl_d.std(unbiased=False).item() + 1e-12
    return (mu / sd) * np.sqrt(252.0)


def train_single_ticker_fingan(
    ticker: str,
    financial_loss_fn,
    max_epochs: int = 100,
    warmup_epochs: int = 15,
    eval_every: int = 20,
    patience: int = 6,
    max_time_s: float = 300.0,
    device=None,
    verbose: bool = True,
) -> dict:
    """
    Train FinGAN on one ticker with custom financial loss terms.
    Phase 1: GradientCheck warmup (warmup_epochs) -> compute scale factor
    Phase 2: Train with g_loss = bce + scale * fin_loss
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    l = config.LOOKBACK
    z_dim, hid_g, hid_d = 8, 8, 8
    batch_size = 100
    tanh_temp = 100.0

    torch.manual_seed(42)
    np.random.seed(42)

    train_np, val_np, _ = _load_data(ticker, l=l)
    train_t = torch.from_numpy(train_np).float().to(device)
    val_t = torch.from_numpy(val_np).float().to(device)

    N = train_t.shape[0]
    ref_mean = train_t[:batch_size].mean()
    ref_std = train_t[:batch_size].std()

    gen = FinGAN.Generator(noise_dim=z_dim, cond_dim=l, hidden_dim=hid_g,
                           output_dim=1, mean=ref_mean, std=ref_std).to(device)
    disc = FinGAN.Discriminator(in_dim=l + 1, hidden_dim=hid_d,
                                mean=ref_mean, std=ref_std).to(device)
    gen_opt = optim.RMSprop(gen.parameters(), lr=1e-4)
    disc_opt = optim.RMSprop(disc.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    t0 = time.time()

    # Check if financial_loss_fn is trivial (returns 0 for all inputs)
    test_out = torch.randn(10, device=device)
    test_real = torch.randn(10, device=device)
    try:
        test_val = financial_loss_fn(test_out, test_real, tanh_temp)
        is_trivial = (abs(test_val.item()) < 1e-8)
    except Exception:
        is_trivial = True

    if is_trivial:
        # velocity_only: skip GradientCheck, train with BCE only
        fin_scale = 0.0
        if verbose:
            print(f"  {ticker} fingan: trivial financial loss, scale=0")
    else:
        # Phase 1: GradientCheck warmup
        fin_scale = _gradient_check_warmup(
            gen, disc, gen_opt, disc_opt, criterion,
            financial_loss_fn, train_t, tanh_temp,
            warmup_epochs, batch_size, l, z_dim, hid_g, hid_d, device)
        if verbose:
            print(f"  {ticker} fingan: GradientCheck scale={fin_scale:.4f} (after {warmup_epochs} warmup epochs)")

    # Phase 2: Training with balanced loss
    best_val_sr = -1e9
    best_epoch = 0
    evals_no_improve = 0
    nbatches = N // batch_size

    for ep in range(warmup_epochs + 1, max_epochs + 1):
        if time.time() - t0 > max_time_s:
            if verbose:
                print(f"  {ticker} fingan timeout at epoch {ep}")
            break

        gen.train()
        disc.train()
        perm = torch.randperm(N, device=device)

        for b in range(nbatches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            cond_b = train_t[idx, :l].unsqueeze(0)
            real_b = train_t[idx, -1]
            real_disc_in = train_t[idx].unsqueeze(0)

            h0g = torch.zeros(1, batch_size, hid_g, device=device)
            c0g = torch.zeros(1, batch_size, hid_g, device=device)
            h0d = torch.zeros(1, batch_size, hid_d, device=device)
            c0d = torch.zeros(1, batch_size, hid_d, device=device)

            # Disc step
            disc_opt.zero_grad()
            noise = torch.randn(1, batch_size, z_dim, device=device)
            with torch.no_grad():
                fake_out = gen(noise, cond_b, h0g, c0g)
            fake_disc_in = torch.cat([cond_b, fake_out], dim=-1)
            d_real = disc(real_disc_in, h0d.detach().clone(), c0d.detach().clone())
            d_fake = disc(fake_disc_in, h0d.detach().clone(), c0d.detach().clone())
            d_loss = (criterion(d_real, torch.ones_like(d_real)) +
                      criterion(d_fake, torch.zeros_like(d_fake))) / 2
            d_loss.backward()
            disc_opt.step()

            # Gen step with balanced loss
            gen_opt.zero_grad()
            bce_loss, fin_loss, _ = _one_gen_step(
                gen, disc, gen_opt, criterion, financial_loss_fn,
                cond_b, real_b, real_disc_in, tanh_temp,
                h0g, c0g, h0d, c0d, z_dim, batch_size, device)
            g_loss = bce_loss + fin_scale * fin_loss
            g_loss.backward()
            gen_opt.step()

        # Eval
        if ep % eval_every == 0:
            val_sr = _fast_val_sr_fingan(gen, val_t, l, z_dim, hid_g, device)
            if verbose:
                print(f"  {ticker} fingan ep={ep:4d} val_SR={val_sr:.4f} (scale={fin_scale:.3f})")
            if val_sr > best_val_sr:
                best_val_sr = val_sr
                best_epoch = ep
                evals_no_improve = 0
            else:
                evals_no_improve += 1
                if evals_no_improve >= patience:
                    break

    return {
        "ticker": ticker,
        "model": "fingan",
        "val_sr": best_val_sr,
        "best_epoch": best_epoch,
        "train_time_s": round(time.time() - t0, 1),
        "fin_scale": fin_scale,
    }
