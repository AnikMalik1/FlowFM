"""
Rewritten FinGAN: unified loss interface, vectorized MC, BSZ=N, 4-GPU parallel.
Original FinGAN.py has 3300+ lines with 10 copy-pasted TrainLoop functions.
This replaces them with ~200 lines.
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

dataloc = "/projects/s5e/quant/fingan/FlowFM/data/"
etflistloc = "/projects/s5e/quant/fingan/FlowFM_repo/stocks-etfs-list.csv"
loc = "/projects/s5e/quant/fingan/FlowFM/Fin-GAN-fast/"

TICKERS = [
    "AMZN","HD","NKE","CL","EL","KO","PEP","APA","OXY",
    "WFC","GS","BLK","PFE","HUM","FDX","GD",
    "IBM","TER","ECL","IP","DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]

# ── Model (same architecture as original) ───────────────────────────────
def combine_vectors(x, y, dim=2):
    return torch.cat((x.float(), y.float()), dim=dim)

class Generator(nn.Module):
    def __init__(self, z_dim, cond_dim, hid_g, mean, std):
        super().__init__()
        self.hid_g = hid_g
        self.z_dim = z_dim
        self.mean = mean
        self.std = std
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=hid_g, num_layers=1)
        self.linear1 = nn.Linear(hid_g + z_dim, hid_g + z_dim)
        self.linear2 = nn.Linear(hid_g + z_dim, 1)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, noise, cond, h0, c0):
        # cond: (1, B, l), noise: (1, B, z_dim)
        cond_n = (cond - self.mean) / self.std
        _, (h_n, _) = self.lstm(cond_n, (h0, c0))
        out = combine_vectors(noise.float(), h_n.float(), dim=-1)
        out = self.relu(self.linear1(out))
        out = self.linear2(out)
        return out * self.std + self.mean  # denormalize

class Discriminator(nn.Module):
    def __init__(self, cond_dim, hid_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 1, hid_d), nn.LeakyReLU(0.2),
            nn.Linear(hid_d, hid_d), nn.LeakyReLU(0.2),
            nn.Linear(hid_d, 1), nn.Sigmoid(),
        )
    def forward(self, cond, x):
        return self.net(torch.cat([cond, x], dim=-1))

# ── Unified Loss Functions ──────────────────────────────────────────────
def loss_forgan(gen_out, real, disc, cond, criterion, tanh_c):
    """Original ForGAN/BCE loss."""
    fake_pred = disc(cond, gen_out)
    return criterion(fake_pred, torch.ones_like(fake_pred))

def loss_mse(gen_out, real, disc, cond, criterion, tanh_c):
    return torch.mean((gen_out - real) ** 2)

def loss_pnl(gen_out, real, disc, cond, criterion, tanh_c):
    ft = torch.tanh(tanh_c * gen_out.squeeze(-1))
    return -torch.mean(ft * real)

def loss_sr(gen_out, real, disc, cond, criterion, tanh_c):
    ft = torch.tanh(tanh_c * gen_out.squeeze(-1))
    pnl = ft * real
    return -(torch.mean(pnl) / (torch.std(pnl) + 1e-8))

def loss_pnlmse(gen_out, real, disc, cond, criterion, tanh_c):
    return loss_pnl(gen_out, real, disc, cond, criterion, tanh_c) + loss_mse(gen_out, real, disc, cond, criterion, tanh_c)

def loss_pnlsr(gen_out, real, disc, cond, criterion, tanh_c):
    return loss_pnl(gen_out, real, disc, cond, criterion, tanh_c) + loss_sr(gen_out, real, disc, cond, criterion, tanh_c)

def loss_pnlstd(gen_out, real, disc, cond, criterion, tanh_c):
    ft = torch.tanh(tanh_c * gen_out.squeeze(-1))
    pnl = ft * real
    return -torch.mean(pnl) + torch.std(pnl)

def loss_pnlmsesr(gen_out, real, disc, cond, criterion, tanh_c):
    return loss_pnl(gen_out, real, disc, cond, criterion, tanh_c) + loss_mse(gen_out, real, disc, cond, criterion, tanh_c) + loss_sr(gen_out, real, disc, cond, criterion, tanh_c)

def loss_pnlmsestd(gen_out, real, disc, cond, criterion, tanh_c):
    return loss_pnlmse(gen_out, real, disc, cond, criterion, tanh_c) + loss_pnlstd(gen_out, real, disc, cond, criterion, tanh_c)

def loss_srmse(gen_out, real, disc, cond, criterion, tanh_c):
    return loss_sr(gen_out, real, disc, cond, criterion, tanh_c) + loss_mse(gen_out, real, disc, cond, criterion, tanh_c)

LOSS_FUNCTIONS = {
    "ForGAN": loss_forgan, "MSE": loss_mse, "PnL": loss_pnl,
    "PnLMSE": loss_pnlmse, "PnLMSESR": loss_pnlmsesr,
    "PnLMSESTD": loss_pnlmsestd, "PnLSR": loss_pnlsr,
    "PnLSTD": loss_pnlstd, "SR": loss_sr, "SRMSE": loss_srmse,
}

# ── Vectorized Evaluation ───────────────────────────────────────────────
@torch.no_grad()
def eval_sr(gen, data, l, z_dim, hid_g, dev, nsamp=1000):
    """Vectorized MC evaluation: generate all samples in one batch."""
    T = data.shape[0]
    cond = data[:, :l].unsqueeze(0).to(dev)  # (1, T, l)
    real = data[:, -1].to(dev)

    # Generate nsamp samples for all T windows at once
    noise = torch.randn(1, T * nsamp, z_dim, device=dev)
    cond_rep = cond.expand(1, T, -1).repeat(1, nsamp, 1).reshape(1, T * nsamp, l)
    h0 = torch.zeros(1, T * nsamp, hid_g, device=dev)
    c0 = torch.zeros(1, T * nsamp, hid_g, device=dev)

    out = gen(noise, cond_rep, h0, c0)  # (1, T*nsamp, 1)
    samples = out.squeeze().view(nsamp, T).T  # (T, nsamp)

    pu = (samples >= 0).float().mean(dim=1)
    pnl_ws = 10000.0 * (2.0 * pu - 1.0) * real
    Td = T // 2
    pnl_wd = pnl_ws[:2*Td].reshape(Td, 2).sum(dim=1)
    mu = pnl_wd.mean().item()
    sd = pnl_wd.std(unbiased=False).item() + 1e-12
    sr = (mu / sd) * np.sqrt(252.0)

    # Also return pu stats for analysis
    pu_np = pu.cpu().numpy()
    return sr, mu, float(pnl_wd.sum()), pu_np

# ── Training ────────────────────────────────────────────────────────────
def train_one_variant(gen, disc, train, val, l, z_dim, hid_g, dev,
                      loss_fn, loss_name, tanh_c=100, n_epochs=100,
                      lr_g=1e-4, lr_d=1e-4, diter=1):
    """Train one loss variant. BSZ=N, vectorized."""
    N = train.shape[0]
    cond_all = train[:, :l].unsqueeze(0).to(dev)  # (1, N, l)
    real_all = train[:, -1].to(dev)  # (N,)

    gen_opt = optim.Adam(gen.parameters(), lr=lr_g)
    disc_opt = optim.Adam(disc.parameters(), lr=lr_d)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    best_state = None

    for ep in range(n_epochs):
        # Train discriminator
        for _ in range(diter):
            disc.zero_grad()
            noise = torch.randn(1, N, z_dim, device=dev)
            h0 = torch.zeros(1, N, hid_g, device=dev)
            c0 = torch.zeros(1, N, hid_g, device=dev)
            with torch.no_grad():
                fake = gen(noise, cond_all, h0, c0)
            # Disc on real
            real_in = real_all.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
            cond_2d = cond_all.squeeze(0)  # (N, l) for disc
            d_real = disc(cond_2d, real_in.squeeze(0))
            d_fake = disc(cond_2d, fake.squeeze(0))
            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            disc_opt.step()

        # Train generator
        gen.zero_grad()
        noise = torch.randn(1, N, z_dim, device=dev)
        h0 = torch.zeros(1, N, hid_g, device=dev)
        c0 = torch.zeros(1, N, hid_g, device=dev)
        fake = gen(noise, cond_all, h0, c0)
        g_loss = loss_fn(fake.squeeze(0), real_all, disc, cond_2d, criterion, tanh_c)
        g_loss.backward()
        gen_opt.step()

        if ep >= 20 and g_loss.item() < best_loss:
            best_loss = g_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
    return best_state


def train_ticker(gpu_id, ticker, ti):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    dev = torch.device("cuda:0")
    torch.manual_seed(42 + ti)

    l, h, pred, tr, vl = 10, 1, 1, 0.8, 0.1
    z_dim, hid_g, hid_d = 8, 8, 8
    tanh_c = 100

    # Load data
    if ticker[0] == "X":
        train_np, val_np, test_np, _ = FinGAN.split_train_val_testraw(
            ticker, dataloc, tr=tr, vl=vl, h=h, l=l, pred=pred, plotcheck=False)
    else:
        train_np, val_np, test_np, _ = FinGAN.split_train_val_test(
            ticker, dataloc, etflistloc, tr=tr, vl=vl, h=h, l=l, pred=pred, plotcheck=False)

    train = torch.from_numpy(train_np).float()
    val = torch.from_numpy(val_np).float()
    test = torch.from_numpy(test_np).float()

    # Normalization stats
    mean_cond = float(train[:, :l].mean())
    std_cond = float(train[:, :l].std()) + 1e-8

    results = []
    t0 = time.time()

    for loss_name, loss_fn in LOSS_FUNCTIONS.items():
        # Fresh model for each variant
        gen = Generator(z_dim, l, hid_g, mean_cond, std_cond).to(dev)
        disc = Discriminator(l, hid_d).to(dev)

        best_state = train_one_variant(
            gen, disc, train, val, l, z_dim, hid_g, dev,
            loss_fn, loss_name, tanh_c=tanh_c, n_epochs=100,
        )

        # Save checkpoint
        ckpt_path = os.path.join(loc, "TrainedModels",
            f"{ticker}-FinGAN-100-epochs-{loss_name}_generator_checkpoint.pth")
        torch.save({"g_state_dict": best_state}, ckpt_path)

        # Eval on val and test
        gen.load_state_dict(best_state)
        gen.eval()

        val_sr, val_pnl, _, _ = eval_sr(gen, val, l, z_dim, hid_g, dev, nsamp=1000)
        test_sr, test_pnl, test_cum, pu_np = eval_sr(gen, test, l, z_dim, hid_g, dev, nsamp=1000)

        results.append({
            "ticker": ticker, "type": loss_name,
            "SR_w scaled val": val_sr, "PnL_w val": val_pnl,
            "SR_w scaled": test_sr, "PnL_w": test_pnl,
            "CumPnL": test_cum,
            "mean_pu": float(pu_np.mean()),
            "mean_abs_pos": float(np.abs(2 * pu_np - 1).mean()),
        })

    elapsed = time.time() - t0

    # Find best by val SR
    best = max(results, key=lambda r: r["SR_w scaled val"])
    print(f"[GPU{gpu_id}] {ticker} done {elapsed:.0f}s | "
          f"best={best['type']} val_SR={best['SR_w scaled val']:+.3f} "
          f"test_SR={best['SR_w scaled']:+.3f} "
          f"pu={best['mean_pu']:.3f} |pos|={best['mean_abs_pos']:.3f}", flush=True)

    return results


def worker(gpu_id, assignments, result_dict):
    rows = []
    for ti, ticker in assignments:
        rows.extend(train_ticker(gpu_id, ticker, ti))
    result_dict[gpu_id] = rows


if __name__ == "__main__":
    mp.set_start_method("spawn")

    for d in ["PnLs", "Results", "Plots", "TrainedModels"]:
        os.makedirs(os.path.join(loc, d), exist_ok=True)

    n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))
    print(f"GPUs: {n_gpus}, Tickers: {len(TICKERS)}, Models: {len(TICKERS)*10}", flush=True)

    assignments = [[] for _ in range(n_gpus)]
    for i, t in enumerate(TICKERS):
        assignments[i % n_gpus].append((i, t))

    for g in range(n_gpus):
        print(f"  GPU {g}: {[t for _,t in assignments[g]]}", flush=True)

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

    # Aggregate
    all_rows = []
    for g in range(n_gpus):
        if g in result_dict:
            all_rows.extend(result_dict[g])

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(loc, "Results", "fingan_full_results.csv"), index=False)

        best = df.sort_values(["ticker", "SR_w scaled val"], ascending=[True, False]).groupby("ticker").head(1)
        best.to_csv(os.path.join(loc, "Results", "fingan_best_by_valSR.csv"), index=False)

        total = time.time() - t0
        print(f"\nAll done in {total:.0f}s ({len(TICKERS)} tickers × 10 variants = {len(all_rows)} models)", flush=True)
        print(f"\nBest per ticker:", flush=True)
        for _, r in best.iterrows():
            print(f"  {r['ticker']:5s} | {r['type']:12s} | val={r['SR_w scaled val']:+.3f} "
                  f"test={r['SR_w scaled']:+.3f} | pu={r['mean_pu']:.3f} |pos|={r['mean_abs_pos']:.3f}", flush=True)
