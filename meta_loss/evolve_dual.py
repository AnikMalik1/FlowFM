#!/usr/bin/env python3
"""
Dual-Model GA Evolution — evolve loss functions evaluated on BOTH FinGAN and FlowFMPlus.

Usage:
    python evolve_dual.py --generations 10 --pop-size 12
"""
import argparse
import os
import sys
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ga_evolver import (
    run_ga, genome_to_loss_fn, genome_to_name, genome_to_description,
)
from evaluator_dual import evaluate_dual


def make_dual_evaluator(device):
    """Create a dual-model evaluation function for the GA."""
    def evaluate(loss_fn, name):
        result = evaluate_dual(
            loss_fn=loss_fn,
            loss_name=name,
            device=device,
            verbose=True,
        )
        return result["fitness"]
    return evaluate


def main():
    parser = argparse.ArgumentParser(description="Dual-Model GA Evolution")
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

    results_dir = os.path.join(config.RESULTS_DIR, "ga_dual")
    os.makedirs(results_dir, exist_ok=True)

    evaluator = make_dual_evaluator(device)

    print(f"\nDual-Model GA: {args.pop_size} pop x {args.generations} gens")
    print(f"Inner loop: FlowFMPlus ({config.STAGE1_MAX_EPOCHS}ep) + FinGAN (100ep)")
    print(f"Tickers: {config.STAGE1_TICKERS}")
    print(f"Fitness = 0.5 × SR_flow + 0.5 × SR_fingan\n")

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
    print(f"\n{'='*80}")
    print(f"{'DUAL-MODEL GA FINAL LEADERBOARD':^80}")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Name':<35} {'Fitness':>8} {'Gen':>5}")
    print(f"{'-'*80}")

    seen = set()
    for r in all_results[:20]:
        key = json.dumps(r["genome"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        print(f"{len(seen):<5} {r['name'][:35]:<35} {r['fitness']:>8.4f} {r['generation']:>5}")
        if len(seen) >= 15:
            break

    print(f"{'='*80}")

    final_path = os.path.join(results_dir, "ga_dual_final.json")
    with open(final_path, "w") as f:
        json.dump(all_results[:50], f, indent=2, default=str)
    print(f"\nSaved: {final_path}")


if __name__ == "__main__":
    main()
