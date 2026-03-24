"""
Evaluator — Stage 1 (fast screen) and Stage 2 (full validation).
"""
import numpy as np
from typing import Callable

import config
from trainer import train_single_ticker


def stage1_evaluate(loss_fn: Callable, loss_name: str,
                    device=None, verbose: bool = True) -> dict:
    """
    Stage 1: Fast screen on 3 tickers x 200 epochs.
    Returns {loss_name, mean_sr, per_ticker: {ticker: sr}, total_time_s}.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Stage 1: {loss_name} ({len(config.STAGE1_TICKERS)} tickers x {config.STAGE1_MAX_EPOCHS} epochs)")
        print(f"{'='*50}")

    results = {}
    total_time = 0.0

    for ticker in config.STAGE1_TICKERS:
        r = train_single_ticker(
            ticker=ticker,
            loss_fn=loss_fn,
            max_epochs=config.STAGE1_MAX_EPOCHS,
            eval_every=config.STAGE1_EVAL_EVERY,
            patience=config.STAGE1_PATIENCE,
            device=device,
            verbose=verbose,
        )
        results[ticker] = r["val_sr"]
        total_time += r["train_time_s"]

    mean_sr = float(np.mean(list(results.values())))

    if verbose:
        print(f"\nStage 1 Results for '{loss_name}':")
        for t, sr in results.items():
            print(f"  {t}: SR = {sr:.4f}")
        print(f"  Mean SR: {mean_sr:.4f}")
        print(f"  Total time: {total_time:.0f}s")

    return {
        "loss_name": loss_name,
        "stage": 1,
        "mean_sr": mean_sr,
        "per_ticker": results,
        "total_time_s": total_time,
    }


def stage2_evaluate(loss_fn: Callable, loss_name: str,
                    device=None, verbose: bool = True) -> dict:
    """
    Stage 2: Full validation on 31 tickers x 500 epochs.
    Returns {loss_name, mean_sr, per_ticker: {ticker: sr}, total_time_s}.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage 2: {loss_name} ({len(config.STAGE2_TICKERS)} tickers x {config.STAGE2_MAX_EPOCHS} epochs)")
        print(f"{'='*60}")

    results = {}
    total_time = 0.0

    for i, ticker in enumerate(config.STAGE2_TICKERS):
        if verbose:
            print(f"\n--- [{i+1}/{len(config.STAGE2_TICKERS)}] {ticker} ---")

        r = train_single_ticker(
            ticker=ticker,
            loss_fn=loss_fn,
            max_epochs=config.STAGE2_MAX_EPOCHS,
            eval_every=config.STAGE2_EVAL_EVERY,
            patience=config.STAGE2_PATIENCE,
            device=device,
            verbose=verbose,
        )
        results[ticker] = r["val_sr"]
        total_time += r["train_time_s"]

    mean_sr = float(np.mean(list(results.values())))

    if verbose:
        print(f"\nStage 2 Results for '{loss_name}':")
        for t, sr in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {t:6s}: SR = {sr:.4f}")
        print(f"  Mean SR: {mean_sr:.4f}")
        print(f"  Total time: {total_time:.0f}s")

    return {
        "loss_name": loss_name,
        "stage": 2,
        "mean_sr": mean_sr,
        "per_ticker": results,
        "total_time_s": total_time,
    }
