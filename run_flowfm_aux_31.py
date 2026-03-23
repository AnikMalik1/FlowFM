import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import FinGAN
from flow_adapter_aux import CondFlowNet, fm_batch_loss, FlowGenAdapter, EMA
from eval2_fast_flow import Evaluation2


MODEL_TAG = "FlowFMPlus_Aux"

TICKERS_31 = [
    "AMZN","HD","NKE",
    "CL","EL","KO","PEP",
    "APA","OXY",
    "WFC","GS","BLK",
    "PFE","HUM",
    "FDX","GD",
    "IBM","TER",
    "ECL","IP",
    "DTE","WEC",
    "XLY","XLP","XLE","XLF","XLV","XLI","XLK","XLB","XLU",
]

# For a quick test first:
# TICKERS_31 = TICKERS_31[:2]


@torch.no_grad()
def fast_val_sr_eval2_logic(gen, val_data, l, z_dim, hid_g, nsamp, chunk, device):
    gen.eval()
    val_data = val_data.to(device).float()

    T = val_data.shape[0]
    cond_Tl = val_data[:, :l].contiguous()
    real = val_data[:, -1].contiguous()

    samples = torch.empty((T, nsamp), device=device, dtype=torch.float32)

    done = 0
    while done < nsamp:
        cur = min(chunk, nsamp - done)
        eff_batch = T * cur

        noise = torch.randn((1, eff_batch, z_dim), device=device, dtype=torch.float32)
        cond_rep = cond_Tl.repeat_interleave(cur, dim=0)
        cond_in = cond_rep.unsqueeze(0).contiguous()

        h0 = torch.zeros((1, eff_batch, hid_g), device=device, dtype=torch.float32)
        c0 = torch.zeros((1, eff_batch, hid_g), device=device, dtype=torch.float32)

        out = gen(noise, cond_in, h0, c0).reshape(-1)
        out = out.view(T, cur)
        samples[:, done:done + cur] = out
        done += cur

    pu = (samples >= 0).float().mean(dim=1)
    pnl_ws = 10000.0 * (2.0 * pu - 1.0) * real

    Td = int(0.5 * T)
    pnl_wd = pnl_ws[:2 * Td].reshape(Td, 2).sum(dim=1)

    mu = pnl_wd.mean().item()
    sd = pnl_wd.std(unbiased=False).item() + 1e-12
    sr_ann = (mu / sd) * np.sqrt(252.0)
    return sr_ann


def compute_portfolio_summary(pnl_dir, tickers, tag=MODEL_TAG):
    pnl_series = []
    used_tickers = []

    for ticker in tickers:
        f = os.path.join(pnl_dir, f"{ticker}-{tag}.csv")
        if not os.path.exists(f):
            continue
        s = pd.read_csv(f, header=None).iloc[:, 0].astype(float).values
        pnl_series.append(s)
        used_tickers.append(ticker)

    if len(pnl_series) == 0:
        return None

    min_len = min(len(x) for x in pnl_series)
    pnl_mat = np.vstack([x[:min_len] for x in pnl_series])
    port = pnl_mat.sum(axis=0)

    mean_daily = float(np.mean(port))
    std_daily = float(np.std(port))
    sharpe_ann = float((mean_daily / (std_daily + 1e-12)) * np.sqrt(252.0))
    cum = np.cumsum(port)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = float(np.min(drawdown))

    return {
        "n_tickers_used": len(used_tickers),
        "mean_daily_pnl_bp": mean_daily,
        "std_daily_pnl_bp": std_daily,
        "annualized_sharpe": sharpe_ann,
        "cumulative_pnl_end_bp": float(cum[-1]),
        "max_drawdown_bp": max_dd,
    }


def load_data_for_ticker(ticker, dataloc, etflistloc, tr, vl, h, l, pred):
    if ticker[0] == "X":
        train_np, val_np, test_np, _ = FinGAN.split_train_val_testraw(
            ticker, dataloc, tr=tr, vl=vl, h=h, l=l, pred=pred, plotcheck=False
        )
    else:
        train_np, val_np, test_np, _ = FinGAN.split_train_val_test(
            ticker, dataloc, etflistloc, tr=tr, vl=vl, h=h, l=l, pred=pred, plotcheck=False
        )
    return train_np, val_np, test_np


def make_model(l, hidden, depth, dropout, device):
    return CondFlowNet(
        cond_dim=l,
        hidden=hidden,
        depth=depth,
        t_dim=64,
        dropout=dropout,
    ).to(device)


def train_one_ticker(
    ticker,
    device,
    dataloc,
    etflistloc,
    loc,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------- config ----------
    h = 1
    l = 10
    pred = 1
    freq = 2
    tr = 0.8
    vl = 0.1

    max_epochs = 500
    batch_size = 4096
    lr = 3e-4
    weight_decay = 0.01
    ode_steps = 40

    hidden = 256
    depth = 4
    dropout = 0.05
    ema_decay = 0.999
    cond_noise_std = 0.02

    eval_every = 10
    val_nsamp_ckpt = 1024
    val_chunk_ckpt = 8
    patience_evals = 8

    final_nsamp = 1000
    final_chunk = 8

    # Evaluation2 compatibility
    hid_g = 8
    hid_d = 64
    z_dim = 8

    use_amp = (device.type == "cuda")

    # ---------- data ----------
    train_np, val_np, test_np = load_data_for_ticker(
        ticker=ticker,
        dataloc=dataloc,
        etflistloc=etflistloc,
        tr=tr,
        vl=vl,
        h=h,
        l=l,
        pred=pred,
    )

    train = torch.from_numpy(train_np).float().to(device)
    val   = torch.from_numpy(val_np).float().to(device)
    test  = torch.from_numpy(test_np).float().to(device)

    cond_train_real = train[:, :l]
    x_train_real    = train[:, l:l+1]

    mu_cond = cond_train_real.mean(dim=0)
    sd_cond = cond_train_real.std(dim=0, unbiased=False) + 1e-8
    mu_x = float(x_train_real.mean().item())
    sd_x = float(x_train_real.std(unbiased=False).item() + 1e-8)

    cond_train = (cond_train_real - mu_cond) / sd_cond
    x1_train   = (x_train_real - mu_x) / sd_x

    # ---------- model ----------
    model = make_model(l=l, hidden=hidden, depth=depth, dropout=dropout, device=device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    ema = EMA(model, decay=ema_decay)

    total_steps = int(np.ceil(cond_train.shape[0] / batch_size)) * max_epochs
    warmup = max(100, total_steps // 50)

    def lr_mult(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    global_step = 0
    best_val_sr = -1e9
    best_state = None
    evals_since_best = 0
    best_path = os.path.join(loc, "TrainedModels", f"{ticker}_{MODEL_TAG}_best.pt")

    N = cond_train.shape[0]
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()

        perm = torch.randperm(N, device=device)
        c = cond_train[perm]
        x = x1_train[perm]

        losses = []

        for i in range(0, N, batch_size):
            c_b = c[i:i + batch_size]
            x_b = x[i:i + batch_size]

            if cond_noise_std > 0:
                c_b = c_b + cond_noise_std * torch.randn_like(c_b)

            mult = lr_mult(global_step)
            for pg in opt.param_groups:
                pg["lr"] = lr * mult

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = fm_batch_loss(model, x_b, c_b)
            else:
                loss = fm_batch_loss(model, x_b, c_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ema.update(model)
            global_step += 1
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses))

        if ep % 10 == 0:
            print(f"{ticker} | epoch {ep:4d} | loss {mean_loss:.6f} | lr {opt.param_groups[0]['lr']:.2e}")

        if ep % eval_every == 0:
            eval_model = make_model(l=l, hidden=hidden, depth=depth, dropout=dropout, device=device)
            eval_model.load_state_dict(ema.shadow, strict=True)
            eval_model.eval()

            gen = FlowGenAdapter(
                flow_model=eval_model,
                mu_cond=mu_cond,
                sd_cond=sd_cond,
                mu_x=mu_x,
                sd_x=sd_x,
                ode_steps=ode_steps,
                pred=1,
            ).to(device)

            val_sr_ckpt = fast_val_sr_eval2_logic(
                gen=gen,
                val_data=val,
                l=l,
                z_dim=z_dim,
                hid_g=hid_g,
                nsamp=val_nsamp_ckpt,
                chunk=val_chunk_ckpt,
                device=device,
            )

            print(f"{ticker} | [val-checkpoint] epoch {ep:4d} | val SR = {val_sr_ckpt:.4f}")

            if val_sr_ckpt > best_val_sr:
                best_val_sr = val_sr_ckpt
                best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
                torch.save(best_state, best_path)
                evals_since_best = 0
                print(f"{ticker} | NEW BEST validation SR: {best_val_sr:.4f}")
            else:
                evals_since_best += 1
                print(f"{ticker} | no improvement count: {evals_since_best}/{patience_evals}")
                if evals_since_best >= patience_evals:
                    print(f"{ticker} | early stopping triggered")
                    break

    print(f"{ticker} | train seconds: {round(time.time() - t0, 2)}")
    print(f"{ticker} | best validation SR: {best_val_sr:.4f}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
        torch.save(best_state, best_path)

    # ---------- final eval ----------
    eval_model = make_model(l=l, hidden=hidden, depth=depth, dropout=dropout, device=device)
    eval_model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
    eval_model.eval()

    gen = FlowGenAdapter(
        flow_model=eval_model,
        mu_cond=mu_cond,
        sd_cond=sd_cond,
        mu_x=mu_x,
        sd_x=sd_x,
        ode_steps=ode_steps,
        pred=1,
    ).to(device)

    df_row, PnL_test, PnL_even, PnL_odd, *_ = Evaluation2(
        ticker=ticker,
        freq=freq,
        gen=gen,
        test_data=test,
        val_data=val,
        h=h,
        l=l,
        pred=pred,
        hid_d=hid_d,
        hid_g=hid_g,
        z_dim=z_dim,
        lrg=lr,
        lrd=lr,
        n_epochs=max_epochs,
        losstype=MODEL_TAG,
        sr_val=0,
        device=device,
        plotsloc=os.path.join(loc, "Plots") + "/",
        f_name=f"{ticker}-{MODEL_TAG}",
        plot=False,
        nsamp=final_nsamp,
        chunk=final_chunk,
        use_amp=False,
        verbose=False,
    )

    pnl_path = os.path.join(loc, "PnLs", f"{ticker}-{MODEL_TAG}.csv")
    pd.DataFrame(PnL_test).to_csv(pnl_path, index=False, header=False)

    return df_row.iloc[0].to_dict()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base = "."
    dataloc = base + "/data/"
    etflistloc = base + "/stocks-etfs-list.csv"
    loc = base + "/Fin-GAN/"

    os.makedirs(os.path.join(loc, "PnLs"), exist_ok=True)
    os.makedirs(os.path.join(loc, "Results"), exist_ok=True)
    os.makedirs(os.path.join(loc, "Plots"), exist_ok=True)
    os.makedirs(os.path.join(loc, "TrainedModels"), exist_ok=True)

    tickers = list(TICKERS_31)

    print("Number of tickers:", len(tickers))
    print("Tickers:", tickers)

    out_csv = os.path.join(loc, "Results", "flowfm_aux_31.csv")
    portfolio_csv = os.path.join(loc, "Results", "flowfm_aux_31_portfolio_summary.csv")

    existing = pd.DataFrame()
    done_tickers = set()
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        if "ticker" in existing.columns:
            done_tickers = set(existing["ticker"].astype(str).tolist())

    all_rows = []
    if len(existing) > 0:
        all_rows.extend(existing.to_dict(orient="records"))

    for i, ticker in enumerate(tickers):
        if ticker in done_tickers:
            print(f"Skipping {ticker} ({i+1}/{len(tickers)}) because it already exists in {out_csv}")
            continue

        print(f"\n===== {ticker} ({i+1}/{len(tickers)}) =====")

        row = train_one_ticker(
            ticker=ticker,
            device=device,
            dataloc=dataloc,
            etflistloc=etflistloc,
            loc=loc,
            seed=1000 + i,
        )

        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print("Updated results CSV:", out_csv)

        portfolio_summary = compute_portfolio_summary(
            pnl_dir=os.path.join(loc, "PnLs"),
            tickers=tickers,
            tag=MODEL_TAG,
        )
        if portfolio_summary is not None:
            pd.DataFrame([portfolio_summary]).to_csv(portfolio_csv, index=False)
            print("Updated portfolio summary.")

    print("\nDone.")
    print("Per-ticker summary:", out_csv)
    print("Portfolio summary:", portfolio_csv)


if __name__ == "__main__":
    main()