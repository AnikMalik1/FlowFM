#!/usr/bin/env python3
"""GA Evolution — FinGAN only. Compare with FlowFMPlus-only GA (job 3306298)."""
import argparse, os, sys, json
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from ga_evolver import run_ga, genome_to_loss_fn, genome_to_name
from evaluator_dual import _adapt_ga_genome_to_financial_loss
from trainer_fingan import train_single_ticker_fingan

def make_fingan_evaluator(device):
    def evaluate(loss_fn, name):
        fingan_loss = _adapt_ga_genome_to_financial_loss(loss_fn)
        results = {}
        for ticker in config.STAGE1_TICKERS:
            r = train_single_ticker_fingan(
                ticker=ticker, financial_loss_fn=fingan_loss,
                max_epochs=100, eval_every=20, patience=6,
                device=device, verbose=False,
            )
            results[ticker] = r["val_sr"]
        mean_sr = float(np.mean(list(results.values())))
        print(f"    {name}: {' | '.join(f'{t}={sr:.3f}' for t, sr in results.items())} | mean={mean_sr:.4f}")
        return mean_sr
    return evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    results_dir = os.path.join(config.RESULTS_DIR, "ga_fingan")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nFinGAN-only GA: {args.pop_size} pop x {args.generations} gens")
    print(f"Tickers: {config.STAGE1_TICKERS}\n")

    all_results = run_ga(
        evaluate_fn=make_fingan_evaluator(device),
        pop_size=args.pop_size, n_generations=args.generations,
        elite_k=2, mutation_rate=0.3, seed=42,
        results_dir=results_dir, verbose=True,
    )

    print(f"\n{'='*70}")
    print(f"{'FINGAN-ONLY GA FINAL':^70}")
    print(f"{'='*70}")
    seen = set()
    for r in all_results[:15]:
        key = json.dumps(r["genome"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            print(f"  {len(seen)}. SR={r['fitness']:.4f} | {r['name'][:40]}")

    with open(os.path.join(results_dir, "final.json"), "w") as f:
        json.dump(all_results[:50], f, indent=2, default=str)
