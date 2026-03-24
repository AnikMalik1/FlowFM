#!/usr/bin/env python3
"""
LLM-Proposal Evolution — FinGAN model.
Same LLM approach as FlowFMPlus (evolve.py) but trains FinGAN.
"""
import argparse, os, sys, json, time
from datetime import datetime
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from loss_registry import LossRegistry
from llm_proposer import propose_loss
from evaluator_dual import _adapt_ga_genome_to_financial_loss
from trainer_fingan import train_single_ticker_fingan


def evaluate_fingan_stage1(loss_code_fn, name, device):
    """Evaluate a loss function on FinGAN, Stage 1 (3 tickers)."""
    fingan_loss = _adapt_ga_genome_to_financial_loss(loss_code_fn)
    results = {}
    for ticker in config.STAGE1_TICKERS:
        r = train_single_ticker_fingan(
            ticker=ticker, financial_loss_fn=fingan_loss,
            max_epochs=100, eval_every=20, patience=6,
            device=device, verbose=True,
        )
        results[ticker] = r["val_sr"]
    mean_sr = float(np.mean(list(results.values())))
    return {"loss_name": name, "mean_sr": mean_sr, "per_ticker": results, "stage": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    results_dir = os.path.join(config.RESULTS_DIR, "llm_fingan")
    os.makedirs(results_dir, exist_ok=True)

    registry = LossRegistry()
    all_results = []

    # Evaluate baselines first
    print("=== Evaluating baselines on FinGAN ===")
    for name in registry.list_names():
        fn = registry.get(name)
        print(f"\n--- {name} ---")
        r = evaluate_fingan_stage1(fn, name, device)
        r["origin"] = "baseline"
        all_results.append(r)
        print(f"  Mean SR: {r['mean_sr']:.4f}")

        path = os.path.join(results_dir, f"baseline_{name}.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2, default=str)

    # Print baseline leaderboard
    sorted_r = sorted(all_results, key=lambda x: -x["mean_sr"])
    print(f"\n{'='*60}")
    print(f"{'FinGAN BASELINE LEADERBOARD':^60}")
    print(f"{'='*60}")
    for i, r in enumerate(sorted_r):
        print(f"  {i+1}. {r['loss_name']:30s} SR={r['mean_sr']:.4f}")

    # LLM evolution rounds
    print(f"\n=== Starting LLM evolution ({args.rounds} rounds) ===")
    for rnd in range(1, args.rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  LLM ROUND {rnd}/{args.rounds} (FinGAN)")
        print(f"{'#'*60}")

        try:
            proposal = propose_loss(all_results, rnd)
        except Exception as e:
            print(f"LLM proposal failed: {e}")
            continue

        name = proposal["name"]
        code = proposal["code"]
        print(f"Proposed: '{name}'")
        print(f"Reasoning: {proposal['reasoning'][:200]}")

        err = registry.register_from_code(name, code, origin="llm")
        if err:
            print(f"Registration failed: {err}")
            all_results.append({"loss_name": name, "mean_sr": -999, "error": err, "origin": "llm"})
            continue

        err = registry.validate_loss_fn(name)
        if err:
            print(f"Validation failed: {err}")
            all_results.append({"loss_name": name, "mean_sr": -999, "error": err, "origin": "llm"})
            continue

        fn = registry.get(name)
        try:
            r = evaluate_fingan_stage1(fn, name, device)
        except Exception as e:
            print(f"Training failed: {e}")
            all_results.append({"loss_name": name, "mean_sr": -999, "error": str(e), "origin": "llm"})
            continue

        r["origin"] = "llm"
        r["reasoning"] = proposal["reasoning"]
        r["code"] = code
        all_results.append(r)
        print(f"  Mean SR: {r['mean_sr']:.4f}")

        path = os.path.join(results_dir, f"round_{rnd:03d}_{name}.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2, default=str)

    # Final leaderboard
    sorted_r = sorted(all_results, key=lambda x: -x.get("mean_sr", -999))
    print(f"\n{'='*60}")
    print(f"{'LLM+FinGAN FINAL LEADERBOARD':^60}")
    print(f"{'='*60}")
    for i, r in enumerate(sorted_r[:20]):
        origin = r.get("origin", "?")
        print(f"  {i+1}. {r['loss_name']:30s} SR={r.get('mean_sr',0):.4f} ({origin})")

    with open(os.path.join(results_dir, "final.json"), "w") as f:
        json.dump(sorted_r[:50], f, indent=2, default=str)
    print("Done.")

if __name__ == "__main__":
    main()
