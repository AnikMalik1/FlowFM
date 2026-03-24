#!/usr/bin/env python3
"""
GA-based Meta-Loss Evolution — entry point.

Usage:
    python evolve_ga.py --generations 10 --pop-size 12
    python evolve_ga.py --generations 5 --pop-size 8  # quick test
"""
import argparse
import os
import sys
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ga_evolver import run_ga, genome_to_description
from trainer import train_single_ticker


def make_stage1_evaluator(device, tickers=None, max_epochs=None):
    """Create a Stage 1 evaluation function for the GA."""
    tickers = tickers or config.STAGE1_TICKERS
    max_epochs = max_epochs or config.STAGE1_MAX_EPOCHS

    def evaluate(loss_fn, name):
        results = {}
        for ticker in tickers:
            r = train_single_ticker(
                ticker=ticker,
                loss_fn=loss_fn,
                max_epochs=max_epochs,
                eval_every=config.STAGE1_EVAL_EVERY,
                patience=config.STAGE1_PATIENCE,
                device=device,
                verbose=False,
            )
            results[ticker] = r["val_sr"]
        mean_sr = float(np.mean(list(results.values())))
        print(f"    {name}: {' | '.join(f'{t}={sr:.3f}' for t, sr in results.items())} | mean={mean_sr:.4f}")
        return mean_sr

    return evaluate


def main():
    parser = argparse.ArgumentParser(description="GA Meta-Loss Evolution")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    results_dir = os.path.join(config.RESULTS_DIR, "ga")
    os.makedirs(results_dir, exist_ok=True)

    evaluator = make_stage1_evaluator(device)

    print(f"\nGA Config: {args.pop_size} pop x {args.generations} gens, "
          f"elite={args.elite}, mutation={args.mutation_rate}")
    print(f"Eval: {config.STAGE1_TICKERS} x {config.STAGE1_MAX_EPOCHS} epochs\n")

    all_results = run_ga(
        evaluate_fn=evaluator,
        pop_size=args.pop_size,
        n_generations=args.generations,
        elite_k=args.elite,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        results_dir=results_dir,
        verbose=True,
    )

    # Final leaderboard
    print(f"\n{'='*70}")
    print(f"{'GA FINAL LEADERBOARD':^70}")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Name':<35} {'SR':>8} {'Gen':>5}")
    print(f"{'-'*70}")

    seen = set()
    for r in all_results[:20]:
        key = json.dumps(r["genome"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        print(f"{len(seen):<5} {r['name'][:35]:<35} {r['fitness']:>8.4f} {r['generation']:>5}")
        if len(seen) >= 15:
            break

    print(f"{'='*70}")

    # Save final results
    final_path = os.path.join(results_dir, "ga_final_results.json")
    with open(final_path, "w") as f:
        json.dump(all_results[:50], f, indent=2, default=str)
    print(f"\nResults saved to {final_path}")


if __name__ == "__main__":
    main()
