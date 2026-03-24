"""
Trainer — train FinGAN with a given financial loss, return metrics.
Uses original FinGAN generator/discriminator architecture.
"""
import os
import sys
import time
import math
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


# ── FinGAN Generator (same as original FinGAN.py) ──────

def combine_vectors(x, y, dim=2):
    return torch.cat((x.float(), y.float()), dim=dim)

class Generator(nn.Module):
    def __init__(self, z_dim, cond_dim, hid_g, mean_val, std_val):
        super().__init__()
        self.hid_g = hid_g
        self.z_dim = z_dim
        self.mean = mean_val
        self.std = std_val
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=hid_g, num_layers=1)
        self.linear1 = nn.Linear(hid_g + z_dim, hid_g + z_dim)
        self.linear2 = nn.Linear(hid_g + z_dim, 1)
        self.relu = nn.ReLU()
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, noise, cond, h0, c0):
        cond_n = (cond - self.mean) / self.std
        _, (h_n, _) = self.lstm(cond_n, (h0, c0))
        out = combine_vectors(noise.float(), h_n.float(), dim=-1)
        out = self.relu(self.linear1(out))
        out = self.linear2(out)
        return out * self.std + self.mean

class Discriminator(nn.Module):
    def __init__(self, cond_dim, hid_d):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=hid_d, num_layers=1)
        self.linear1 = nn.Linear(hid_d + 1, hid_d)
        self.linear2 = nn.Linear(hid_d, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, cond, h0, c0):
        _, (h_n, _) = self.lstm(cond, (h0, c0))
        out = combine_vectors(noise.float(), h_n.float(), dim=-1)
        out = self.relu(self.linear1(out))
        out = self.sigmoid(self.linear2(out))
        return out


# ── Vectorized Eval ─────────────────────────────────────

@torch.no_grad()
def _fast_val_sr_fingan(gen, data, l, z_dim, hid_g, device, nsamp=256):
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


# ── Training ────────────────────────────────────────────

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
        gen_out: (N,) generated returns
        real: (N,) actual returns
        tanh_temp: float
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
    mean_val = float(train_t[:, :l].mean().item())
    std_val = float(train_t[:, :l].std().item()) + 1e-8

    gen = Generator(z_dim, l, hid_g, mean_val, std_val).to(device)
    disc = Discriminator(1, hid_d).to(device)
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
            cond_b = train_t[idx, :l].unsqueeze(0)
            real_b = train_t[idx, -1]

            # Disc step
            disc_opt.zero_grad()
            noise = torch.randn(1, batch_size, z_dim, device=device)
            h0 = torch.zeros(1, batch_size, hid_g, device=device)
            c0 = torch.zeros(1, batch_size, hid_g, device=device)
            h0d = torch.zeros(1, batch_size, hid_d, device=device)
            c0d = torch.zeros(1, batch_size, hid_d, device=device)

            with torch.no_grad():
                fake = gen(noise, cond_b, h0, c0).squeeze()

            d_real = disc(real_b.unsqueeze(0).unsqueeze(-1), cond_b, h0d, c0d)
            d_fake = disc(fake.unsqueeze(0).unsqueeze(-1), cond_b, h0d, c0d)
            d_loss = (criterion(d_real, torch.ones_like(d_real)) +
                      criterion(d_fake, torch.zeros_like(d_fake))) / 2
            d_loss.backward()
            disc_opt.step()

            # Gen step
            gen_opt.zero_grad()
            noise = torch.randn(1, batch_size, z_dim, device=device)
            fake = gen(noise, cond_b, h0, c0).squeeze()

            d_fake = disc(fake.unsqueeze(0).unsqueeze(-1), cond_b, h0d, c0d)
            bce_loss = criterion(d_fake, torch.ones_like(d_fake))

            fin_loss = financial_loss_fn(fake, real_b, tanh_temp)
            g_loss = bce_loss + fin_loss
            g_loss.backward()
            gen_opt.step()

        # Eval
        if ep % eval_every == 0:
            val_sr = _fast_val_sr_fingan(gen, val_t, l, z_dim, hid_g, device)
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
