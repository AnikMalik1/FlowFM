#!/usr/bin/env python3
"""
Meta-Loss Evolution Controller — main entry point.

Usage:
    # Evaluate FinGAN baselines only (no API needed)
    python evolve.py --baselines-only

    # Run full evolution loop
    ANTHROPIC_API_KEY=sk-... python evolve.py --rounds 20

    # Evaluate a single custom loss
    python evolve.py --eval-loss losses/my_custom_loss.py

    # Resume from previous results
    python evolve.py --rounds 20 --resume
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
import numpy as np

import config
from loss_registry import LossRegistry
from evaluator import stage1_evaluate, stage2_evaluate
from llm_proposer import propose_loss


def load_results(results_dir: str) -> list[dict]:
    """Load all round results from JSON files."""
    results = []
    if not os.path.exists(results_dir):
        return results
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".json") and f.startswith("round_"):
            with open(os.path.join(results_dir, f)) as fh:
                results.append(json.load(fh))
    return results


def save_result(result: dict, results_dir: str):
    """Save a single round result."""
    os.makedirs(results_dir, exist_ok=True)
    name = result.get("loss_name", "unknown")
    stage = result.get("stage", 1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"round_{ts}_{name}_s{stage}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def print_leaderboard(results: list[dict]):
    """Print current leaderboard sorted by mean SR."""
    if not results:
        print("No results yet.")
        return

    # Group by loss_name, keep best stage result
    best = {}
    for r in results:
        name = r["loss_name"]
        sr = r.get("mean_sr", -999)
        if name not in best or sr > best[name].get("mean_sr", -999):
            best[name] = r

    sorted_r = sorted(best.values(), key=lambda x: -x.get("mean_sr", -999))

    print(f"\n{'='*70}")
    print(f"{'LEADERBOARD':^70}")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Loss Name':<30} {'Mean SR':>8} {'Stage':>6} {'Time':>8}")
    print(f"{'-'*70}")

    for i, r in enumerate(sorted_r):
        name = r.get("loss_name", "?")[:30]
        sr = r.get("mean_sr", 0)
        stage = r.get("stage", "?")
        t = r.get("total_time_s", 0)
        marker = " *" if r.get("stage", 1) == 2 else ""
        print(f"{i+1:<5} {name:<30} {sr:>8.4f} {'S'+str(stage):>6} {t:>7.0f}s{marker}")

    print(f"{'='*70}")
    print("(* = Stage 2 validated)\n")


def get_stage2_threshold(results: list[dict], top_k: int = 3) -> float:
    """SR threshold for Stage 2 promotion: must beat top-K Stage 1 results."""
    s1_results = [r for r in results if r.get("stage", 1) == 1]
    if len(s1_results) < top_k:
        return -999.0
    srs = sorted([r.get("mean_sr", -999) for r in s1_results], reverse=True)
    return srs[min(top_k - 1, len(srs) - 1)]


def run_baselines(registry: LossRegistry, device, results: list[dict]) -> list[dict]:
    """Evaluate all FinGAN baselines (Stage 1)."""
    evaluated = {r["loss_name"] for r in results}

    for name in registry.list_names():
        if name in evaluated:
            print(f"Skipping '{name}' (already evaluated)")
            continue

        fn = registry.get(name)
        result = stage1_evaluate(fn, name, device=device)
        result["description"] = registry.get_meta(name).get("description", "")
        result["origin"] = "fingan_baseline"

        path = save_result(result, config.RESULTS_DIR)
        results.append(result)
        print(f"Saved: {path}")

    return results


def run_evolution(registry: LossRegistry, device, results: list[dict],
                  max_rounds: int = 30) -> list[dict]:
    """Run LLM-driven evolution loop."""
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'#'*70}")
        print(f"  EVOLUTION ROUND {round_num}/{max_rounds}")
        print(f"{'#'*70}")

        # 1. LLM proposes new loss
        print("\nAsking Claude for a new loss function...")
        try:
            proposal = propose_loss(results, round_num)
        except Exception as e:
            print(f"LLM proposal failed: {e}")
            continue

        name = proposal["name"]
        code = proposal["code"]
        reasoning = proposal["reasoning"]

        print(f"\nProposed: '{name}'")
        print(f"Reasoning: {reasoning[:200]}")
        print(f"Code preview:\n{code[:300]}...")

        # 2. Register and validate
        err = registry.register_from_code(name, code,
                                          description=proposal["description"],
                                          origin="llm")
        if err:
            print(f"Registration failed: {err}")
            result = {
                "loss_name": name, "stage": 0, "mean_sr": -999,
                "error": err, "reasoning": reasoning,
            }
            save_result(result, config.RESULTS_DIR)
            results.append(result)
            continue

        err = registry.validate_loss_fn(name)
        if err:
            print(f"Validation failed: {err}")
            result = {
                "loss_name": name, "stage": 0, "mean_sr": -999,
                "error": err, "reasoning": reasoning,
            }
            save_result(result, config.RESULTS_DIR)
            results.append(result)
            continue

        print("Validation passed.")

        # 3. Stage 1 evaluation
        fn = registry.get(name)
        try:
            result = stage1_evaluate(fn, name, device=device)
        except Exception as e:
            print(f"Stage 1 training failed: {e}")
            result = {
                "loss_name": name, "stage": 1, "mean_sr": -999,
                "error": str(e), "reasoning": reasoning,
            }
            save_result(result, config.RESULTS_DIR)
            results.append(result)
            continue

        result["reasoning"] = reasoning
        result["code"] = code
        result["origin"] = "llm"
        save_result(result, config.RESULTS_DIR)
        results.append(result)

        # 4. Stage 2 promotion check
        threshold = get_stage2_threshold(results, config.STAGE2_PROMOTION_TOP_K)
        if result["mean_sr"] > threshold:
            print(f"\nPromoted to Stage 2! (SR {result['mean_sr']:.4f} > threshold {threshold:.4f})")
            try:
                result2 = stage2_evaluate(fn, name, device=device)
                result2["reasoning"] = reasoning
                result2["code"] = code
                result2["origin"] = "llm"
                save_result(result2, config.RESULTS_DIR)
                results.append(result2)
            except Exception as e:
                print(f"Stage 2 failed: {e}")
        else:
            print(f"Not promoted (SR {result['mean_sr']:.4f} <= threshold {threshold:.4f})")

        print_leaderboard(results)

    return results


def eval_single_loss(path: str, registry: LossRegistry, device,
                     stage: int = 1):
    """Evaluate a single loss file."""
    name = os.path.splitext(os.path.basename(path))[0]

    with open(path) as f:
        code = f.read()

    err = registry.register_from_code(name, code, origin="manual")
    if err:
        print(f"Error: {err}")
        return

    err = registry.validate_loss_fn(name)
    if err:
        print(f"Validation error: {err}")
        return

    fn = registry.get(name)
    if stage == 1:
        result = stage1_evaluate(fn, name, device=device)
    else:
        result = stage2_evaluate(fn, name, device=device)

    path = save_result(result, config.RESULTS_DIR)
    print(f"\nSaved: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Meta-Loss Evolution Framework")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only evaluate FinGAN baselines (no API needed)")
    parser.add_argument("--rounds", type=int, default=config.MAX_ROUNDS,
                        help="Number of evolution rounds")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results")
    parser.add_argument("--eval-loss", type=str, default=None,
                        help="Evaluate a single loss .py file")
    parser.add_argument("--eval-stage", type=int, default=1,
                        help="Stage for --eval-loss (1 or 2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Registry
    registry = LossRegistry()
    print(f"Loaded {len(registry.list_names())} loss functions")

    # Results
    results = load_results(config.RESULTS_DIR) if args.resume else []
    if results:
        print(f"Resumed {len(results)} previous results")

    # Single loss evaluation
    if args.eval_loss:
        eval_single_loss(args.eval_loss, registry, device, args.eval_stage)
        return

    # Baselines
    results = run_baselines(registry, device, results)
    print_leaderboard(results)

    if args.baselines_only:
        print("Baselines-only mode. Done.")
        return

    # Evolution
    results = run_evolution(registry, device, results, args.rounds)
    print_leaderboard(results)
    print("Evolution complete.")


if __name__ == "__main__":
    main()
