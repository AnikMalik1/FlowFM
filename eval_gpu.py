"""
All-GPU Evaluation2 replacement. No numpy, no for-loop MC sampling.
Drop-in replacement for FinGAN.Evaluation2.
"""
import torch
import numpy as np
import math


@torch.no_grad()
def Evaluation2_gpu(ticker, freq, gen, test_data, val_data, h, l, pred,
                    hid_d, hid_g, z_dim, lrg, lrd, n_epochs, losstype,
                    sr_val, device, plotsloc, f_name, plot=False,
                    nsamp=1000):
    """
    All-torch GPU evaluation. Replaces FinGAN.Evaluation2.
    MC sampling is vectorized: one forward pass for all nsamp samples.
    """
    gen.eval()
    dt = {'lrd': lrd, 'lrg': lrg, 'type': losstype, 'epochs': n_epochs,
          'ticker': ticker, 'hid_g': hid_g, 'hid_d': hid_d}

    def eval_split(data, split_name):
        T = data.shape[0]
        cond = data[:, :l].unsqueeze(0).to(device).float()  # (1, T, l)

        # Vectorized MC: generate all T*nsamp samples in one forward pass
        cond_rep = cond.expand(1, T, -1).repeat(1, nsamp, 1).reshape(1, T * nsamp, l)
        noise = torch.randn(1, T * nsamp, z_dim, device=device, dtype=torch.float32)
        h0 = torch.zeros(1, T * nsamp, hid_g, device=device, dtype=torch.float32)
        c0 = torch.zeros(1, T * nsamp, hid_g, device=device, dtype=torch.float32)

        fake = gen(noise, cond_rep, h0, c0)  # (1, T*nsamp, 1)
        samples = fake.squeeze().view(nsamp, T).T  # (T, nsamp)

        # Mean prediction
        mn = samples.mean(dim=1)  # (T,)
        real = data[:, -1].to(device).float()  # (T,)

        # RMSE, MAE
        rmse = torch.sqrt(torch.mean((mn - real) ** 2)).item()
        mae = torch.mean(torch.abs(mn - real)).item()

        # PnL (sign-based)
        sgn = torch.sign(mn)
        pnl_total = (10000 * torch.sum(sgn * real) / T).item()

        # pu-based Sharpe
        pu = (samples >= 0).float().mean(dim=1)  # (T,)
        pnl_ws = 10000.0 * (2.0 * pu - 1.0) * real  # (T,) half-step PnL

        Td = T // 2
        pnl_wd = pnl_ws[:2*Td].reshape(Td, 2).sum(dim=1)  # daily PnL
        pnl_even = pnl_ws[::2][:Td]
        pnl_odd = pnl_ws[1::2][:Td]

        pnl_mean = pnl_wd.mean().item()
        pnl_std = pnl_wd.std(unbiased=False).item() + 1e-12
        sr_ann = (pnl_mean / pnl_std) * math.sqrt(252)

        # Correlation
        mn_centered = mn - mn.mean()
        real_centered = real - real.mean()
        corr = (mn_centered * real_centered).sum() / (mn_centered.norm() * real_centered.norm() + 1e-12)

        # Pos mn / Neg mn
        pos_mn = (mn > 0).float().mean().item()
        neg_mn = (mn < 0).float().mean().item()

        # Narrow dist check
        narrow_dist = (samples[1].std().item() < 0.0002)
        narrow_means = (mn.std().item() < 0.0002)

        # pu stats
        mean_pu = pu.mean().item()
        mean_abs_pos = torch.abs(2.0 * pu - 1.0).mean().item()

        return {
            'RMSE': rmse, 'MAE': mae,
            'SR_w scaled': sr_ann, 'PnL_w': pnl_mean,
            'Corr': corr.item(),
            'Pos mn': pos_mn, 'Neg mn': neg_mn,
            'narrow dist': narrow_dist, 'narrow means dist': narrow_means,
            'mean_pu': mean_pu, 'mean_abs_pos': mean_abs_pos,
        }, pnl_wd.cpu().numpy(), pnl_even.cpu().numpy(), pnl_odd.cpu().numpy(), \
           mn.cpu().numpy(), real.cpu().numpy(), samples[1].cpu().numpy()

    # Test split
    test_res, PnL_test, PnL_even, PnL_odd, means_gen, reals_test, distcheck_test = eval_split(test_data, "test")
    dt.update({k: v for k, v in test_res.items()})

    # Eval on close-to-open / open-to-close
    T = test_data.shape[0]
    if T % 2 == 0:
        dt['Close-to-Open SR_w'] = math.sqrt(252) * float(np.mean(PnL_even)) / (float(np.std(PnL_even)) + 1e-12)
        dt['Open-to-Close SR_w'] = math.sqrt(252) * float(np.mean(PnL_odd)) / (float(np.std(PnL_odd)) + 1e-12)
    else:
        dt['Open-to-Close SR_w'] = math.sqrt(252) * float(np.mean(PnL_even)) / (float(np.std(PnL_even)) + 1e-12)
        dt['Close-to-Open SR_w'] = math.sqrt(252) * float(np.mean(PnL_odd)) / (float(np.std(PnL_odd)) + 1e-12)

    print(f"Annualised (test) SR_w: {test_res['SR_w scaled']:.4f}")
    print(f"Correlation {test_res['Corr']:.4f}")

    # Val split
    val_res, _, _, _, _, _, _ = eval_split(val_data, "val")
    dt.update({f'{k} val': v for k, v in val_res.items()
               if k in ['RMSE', 'MAE', 'SR_w scaled', 'PnL_w', 'Corr', 'Pos mn', 'Neg mn']})

    print(f"Annualised (val) SR_w: {val_res['SR_w scaled']:.4f}")

    import pandas as pd
    df_temp = pd.DataFrame(data=dt, index=[0])

    rl_test = reals_test[1] if len(reals_test) > 1 else 0.0

    return df_temp, PnL_test, PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test
