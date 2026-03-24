"""
Trainer — train FinGAN with a given financial loss, return metrics.
Uses original FinGAN Generator/Discriminator from FinGAN.py (zero rewrite).
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


@torch.no_grad()
def _fast_val_sr_fingan(gen, data, l, z_dim, hid_g, device, mean_val, std_val, nsamp=256):
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
    eval_every: int = 20,
    patience: int = 6,
    max_time_s: float = 300.0,
    device=None,
    verbose: bool = True,
) -> dict:
    """
    Train FinGAN on one ticker with custom financial loss terms.

    financial_loss_fn(gen_out, real, tanh_temp) -> scalar
        gen_out: (B,) generated returns (denormalized)
        real: (B,) actual returns
        tanh_temp: float
    Returns: dict with {ticker, model, val_sr, best_epoch, train_time_s}
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
    mean_val = float(train_t.mean().item())
    std_val = float(train_t.std().item()) + 1e-8

    # Use original FinGAN Generator and Discriminator classes
    gen = FinGAN.Generator(z_dim, l + 1, hid_g, mean_val, std_val).to(device)
    disc = FinGAN.Discriminator(l + 1, hid_d, mean_val, std_val).to(device)
    gen_opt = optim.RMSprop(gen.parameters(), lr=1e-4)
    disc_opt = optim.RMSprop(disc.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    best_val_sr = -1e9
    best_epoch = 0
    evals_no_improve = 0
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        if time.time() - t0 > max_time_s:
            if verbose:
                print(f"  {ticker} fingan timeout at epoch {ep}")
            break

        gen.train()
        disc.train()
        nbatches = N // batch_size
        perm = torch.randperm(N, device=device)

        for b in range(nbatches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            cond_b = train_t[idx, :l].unsqueeze(0)           # (1, B, l)
            real_b = train_t[idx, -1]                          # (B,)
            real_disc_in = train_t[idx].unsqueeze(0)           # (1, B, l+1) for disc

            # Disc step: disc takes concat(cond, real_or_fake) as input
            disc_opt.zero_grad()
            noise = torch.randn(1, batch_size, z_dim, device=device)
            h0g = torch.zeros(1, batch_size, hid_g, device=device)
            c0g = torch.zeros(1, batch_size, hid_g, device=device)
            h0d = torch.zeros(1, batch_size, hid_d, device=device)
            c0d = torch.zeros(1, batch_size, hid_d, device=device)

            with torch.no_grad():
                fake_out = gen(noise, cond_b, h0g, c0g)  # (1, B, 1)

            # Disc input = concat(cond, real) or concat(cond, fake)
            fake_disc_in = torch.cat([cond_b, fake_out], dim=-1)  # (1, B, l+1)

            d_real = disc(real_disc_in, h0d, c0d)
            d_fake = disc(fake_disc_in, h0d.clone(), c0d.clone())
            d_loss = (criterion(d_real, torch.ones_like(d_real)) +
                      criterion(d_fake, torch.zeros_like(d_fake))) / 2
            d_loss.backward()
            disc_opt.step()

            # Gen step
            gen_opt.zero_grad()
            noise = torch.randn(1, batch_size, z_dim, device=device)
            fake_out = gen(noise, cond_b, h0g, c0g)  # (1, B, 1)

            fake_disc_in = torch.cat([cond_b, fake_out], dim=-1)
            d_fake = disc(fake_disc_in, h0d.clone(), c0d.clone())
            bce_loss = criterion(d_fake, torch.ones_like(d_fake))

            # Financial loss on generated returns
            gen_returns = fake_out.squeeze()  # (B,)
            fin_loss = financial_loss_fn(gen_returns, real_b, tanh_temp)
            g_loss = bce_loss + fin_loss

            g_loss.backward()
            gen_opt.step()

        # Eval
        if ep % eval_every == 0:
            val_sr = _fast_val_sr_fingan(gen, val_t, l, z_dim, hid_g, device,
                                          mean_val, std_val)
            if verbose:
                print(f"  {ticker} fingan ep={ep:4d} val_SR={val_sr:.4f}")
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
    }
