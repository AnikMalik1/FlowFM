"""
Dual-Model Evaluator — evaluates a financial loss on both FinGAN and FlowFMPlus.
Fitness = mean of both models' mean per-ticker Sharpe Ratio.
"""
import numpy as np
import torch
from typing import Callable

import config
from trainer import train_single_ticker  # FlowFMPlus
from trainer_fingan import train_single_ticker_fingan


def _adapt_ga_genome_to_financial_loss(genome_loss_fn):
    """
    Adapt GA genome loss_fn (FlowFMPlus interface) to FinGAN interface.

    FlowFMPlus: loss_fn(v_pred, v_target, x_pred, x_real, condition, epoch)
    FinGAN:     financial_loss_fn(gen_out, real, tanh_temp)

    The financial terms are the same, just different variable names.
    """
    def fingan_financial_loss(gen_out, real, tanh_temp):
        # Create dummy variables that FlowFMPlus loss expects
        # x_pred ≈ gen_out (both are model predictions)
        # x_real ≈ real (both are actual returns)
        # v_pred/v_target are not used by financial terms
        x_pred = gen_out.unsqueeze(-1) if gen_out.dim() == 1 else gen_out
        x_real = real.unsqueeze(-1) if real.dim() == 1 else real
        v_dummy = torch.zeros_like(x_pred)
        cond_dummy = torch.zeros(x_pred.shape[0], 10, device=x_pred.device)

        # Call the genome loss_fn, subtract velocity_loss (which is 0 with dummy)
        total = genome_loss_fn(v_dummy, v_dummy, x_pred, x_real, cond_dummy, epoch=100)

        # The genome loss includes velocity_MSE(0,0)=0, so total IS the financial part
        return total

    return fingan_financial_loss


def evaluate_dual(
    loss_fn: Callable,
    loss_name: str,
    tickers: list[str] = None,
    max_epochs_flow: int = None,
    max_epochs_fingan: int = 100,
    device=None,
    verbose: bool = True,
    weight_flow: float = 0.5,
    weight_fingan: float = 0.5,
) -> dict:
    """
    Evaluate a loss function on both FlowFMPlus and FinGAN.

    Returns {
        loss_name, fitness, sr_flow, sr_fingan,
        per_ticker_flow, per_ticker_fingan
    }
    """
    tickers = tickers or config.STAGE1_TICKERS
    max_epochs_flow = max_epochs_flow or config.STAGE1_MAX_EPOCHS

    fingan_loss = _adapt_ga_genome_to_financial_loss(loss_fn)

    flow_results = {}
    fingan_results = {}

    for ticker in tickers:
        # FlowFMPlus
        r = train_single_ticker(
            ticker=ticker, loss_fn=loss_fn,
            max_epochs=max_epochs_flow,
            eval_every=config.STAGE1_EVAL_EVERY,
            patience=config.STAGE1_PATIENCE,
            device=device, verbose=False,
        )
        flow_results[ticker] = r["val_sr"]

        # FinGAN
        r2 = train_single_ticker_fingan(
            ticker=ticker, financial_loss_fn=fingan_loss,
            max_epochs=max_epochs_fingan,
            eval_every=20, patience=6,
            device=device, verbose=False,
        )
        fingan_results[ticker] = r2["val_sr"]

    sr_flow = float(np.mean(list(flow_results.values())))
    sr_fingan = float(np.mean(list(fingan_results.values())))
    fitness = weight_flow * sr_flow + weight_fingan * sr_fingan

    if verbose:
        print(f"    {loss_name}:")
        print(f"      FlowFMPlus: {' | '.join(f'{t}={sr:.3f}' for t, sr in flow_results.items())} | mean={sr_flow:.4f}")
        print(f"      FinGAN:     {' | '.join(f'{t}={sr:.3f}' for t, sr in fingan_results.items())} | mean={sr_fingan:.4f}")
        print(f"      Fitness: {fitness:.4f} (flow={sr_flow:.4f} × {weight_flow} + fingan={sr_fingan:.4f} × {weight_fingan})")

    return {
        "loss_name": loss_name,
        "fitness": fitness,
        "sr_flow": sr_flow,
        "sr_fingan": sr_fingan,
        "per_ticker_flow": flow_results,
        "per_ticker_fingan": fingan_results,
    }
