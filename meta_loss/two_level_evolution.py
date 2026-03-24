#!/usr/bin/env python3
"""
Two-Level Evolution Strategy (v3)

Level 1 (GA): Optimize weights + combinations of known primitive terms
  - Fast, structured, no API needed
  - Runs every generation (~12 evals per generation)
  - Handles exploitation

Level 2 (LLM): Invent new primitive terms when GA plateaus
  - Slow, creative, needs API
  - Runs only when GA stagnates (no improvement for N generations)
  - Handles exploration
  - New terms get added to GA's term registry

Evolution loop:
  1. GA runs for K generations
  2. If best fitness hasn't improved for P generations:
     a. Send GA leaderboard + top genomes to LLM
     b. Ask LLM to invent 1-2 NEW primitive terms (not combos!)
     c. Add new terms to TERM_REGISTRY
     d. Inject random genomes using new terms into population
     e. Resume GA
  3. Repeat until budget exhausted

This combines GA's exploitation strength with LLM's creative exploration.
"""
import os
import sys
import json
import time
import copy
import random
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ga_evolver import (
    TERM_REGISTRY, MAXIMIZE_TERMS, MINIMIZE_TERMS,
    run_ga, random_genome, genome_to_loss_fn, genome_to_name,
    genome_to_description, fingan_seed_genomes, mutate,
    tournament_select, crossover,
)

import torch
import torch.nn.functional as F
import numpy as np


# ── LLM Term Invention ──────────────────────────────────

TERM_INVENTION_PROMPT = """You are a financial loss function researcher.

## Context
We are using a Genetic Algorithm to evolve loss functions for a flow matching model.
The GA optimizes weights and combinations of PRIMITIVE TERMS.

## Current Primitive Terms (already in the system):
{current_terms}

## GA Leaderboard (top 5 genomes):
{leaderboard}

## GA has STAGNATED for {stagnation_gens} generations.

## Your Task
Invent 1-2 NEW primitive loss terms that are FUNDAMENTALLY DIFFERENT from existing ones.

Each term must:
1. Take (x_pred, x_real, condition, v_pred, v_target, tanh_temp) as kwargs
2. Return a scalar Tensor
3. Be differentiable
4. Use only torch operations
5. Encode a novel financial insight not captured by existing terms

DO NOT:
- Combine existing terms (GA does that)
- Adjust weights (GA does that)
- Add curriculum scheduling (GA does that)

Output for EACH new term:
1. Name (snake_case)
2. Whether to maximize or minimize it
3. One-line financial intuition
4. Python function code

```python
def term_name(x_pred, x_real, condition, **kw):
    # ... novel computation ...
    return scalar_tensor
```"""


def ask_llm_for_new_terms(
    current_terms: list[str],
    leaderboard: list[dict],
    stagnation_gens: int,
) -> list[dict]:
    """
    Ask LLM to invent new primitive terms.
    Returns list of {"name": str, "code": str, "maximize": bool, "description": str}
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )

    terms_desc = "\n".join(f"- {name}" for name in current_terms)
    lb_desc = "\n".join(
        f"- SR={r['fitness']:.4f}: {r['description']}"
        for r in leaderboard[:5]
    )

    prompt = TERM_INVENTION_PROMPT.format(
        current_terms=terms_desc,
        leaderboard=lb_desc,
        stagnation_gens=stagnation_gens,
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": "You invent novel financial loss terms for genetic algorithm optimization."},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content
    return _parse_term_proposals(text)


def _parse_term_proposals(text: str) -> list[dict]:
    """Parse LLM response into term proposals."""
    import re

    proposals = []
    # Find all python code blocks
    blocks = re.findall(r'```python\s*(.*?)```', text, re.DOTALL)

    for block in blocks:
        # Extract function name
        m = re.search(r'def\s+(\w+)\s*\(', block)
        if not m:
            continue
        name = m.group(1)

        # Determine maximize/minimize from context before the block
        idx = text.find(block)
        context = text[max(0, idx - 200):idx].lower()
        maximize = "maximize" in context or "higher is better" in context

        proposals.append({
            "name": name,
            "code": block.strip(),
            "maximize": maximize,
            "description": name.replace("_", " "),
        })

    return proposals


def register_new_term(name: str, code: str, maximize: bool) -> Optional[str]:
    """
    Safely register a new term into GA's TERM_REGISTRY.
    Returns error string or None on success.
    """
    from loss_registry import _scan_code_safety

    # Safety scan
    err = _scan_code_safety(code)
    if err:
        return f"Safety scan failed: {err}"

    # Compile and test
    try:
        namespace = {"torch": torch, "F": F, "math": __import__("math")}
        exec(code, namespace)

        fn = namespace.get(name)
        if fn is None:
            return f"Function '{name}' not found in code"

        # Smoke test
        B = 16
        result = fn(
            x_pred=torch.randn(B, 1),
            x_real=torch.randn(B, 1),
            condition=torch.randn(B, 10),
            v_pred=torch.randn(B, 1),
            v_target=torch.randn(B, 1),
            tanh_temp=100.0,
        )
        if not isinstance(result, torch.Tensor) or result.dim() != 0:
            return f"Term must return scalar tensor, got {type(result)} shape={getattr(result, 'shape', '?')}"

        # Register
        TERM_REGISTRY[name] = fn
        if maximize:
            MAXIMIZE_TERMS.add(name)
        else:
            MINIMIZE_TERMS.add(name)

        return None

    except Exception as e:
        return f"Failed to register term '{name}': {e}"


# ── Two-Level Evolution Loop ────────────────────────────

def run_two_level(
    evaluate_fn,
    pop_size: int = 12,
    ga_gens_per_cycle: int = 5,
    max_cycles: int = 4,
    stagnation_threshold: int = 3,
    elite_k: int = 2,
    mutation_rate: float = 0.3,
    use_llm: bool = True,
    seed: int = 42,
    results_dir: str = "results/two_level",
    verbose: bool = True,
):
    """
    Two-level evolution:
    - Outer loop: LLM invents new terms when GA stagnates
    - Inner loop: GA optimizes combinations + weights
    """
    random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    global_best = -999.0
    global_best_genome = None
    stagnation_count = 0

    # Initialize population
    population = fingan_seed_genomes()
    while len(population) < pop_size:
        population.append(random_genome(n_terms=random.randint(2, 4)))

    for cycle in range(1, max_cycles + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"  CYCLE {cycle}/{max_cycles} | Terms: {len(TERM_REGISTRY)} | Best: {global_best:.4f}")
            print(f"{'='*70}")

        cycle_best = -999.0

        # ── Level 1: GA inner loop ──
        for gen in range(1, ga_gens_per_cycle + 1):
            global_gen = (cycle - 1) * ga_gens_per_cycle + gen

            if verbose:
                print(f"\n--- Cycle {cycle}, Gen {gen}/{ga_gens_per_cycle} (global gen {global_gen}) ---")

            fitnesses = []
            for i, genome in enumerate(population):
                name = f"c{cycle}g{gen}_{genome_to_name(genome)}_{i}"
                loss_fn = genome_to_loss_fn(genome)

                try:
                    sr = evaluate_fn(loss_fn, name)
                except Exception as e:
                    if verbose:
                        print(f"  [{i}] FAILED: {e}")
                    sr = -999.0

                fitnesses.append(sr)
                all_results.append({
                    "cycle": cycle, "generation": global_gen,
                    "name": name, "fitness": sr,
                    "genome": genome, "description": genome_to_description(genome),
                })

            gen_best = max(fitnesses)
            cycle_best = max(cycle_best, gen_best)

            if gen_best > global_best:
                global_best = gen_best
                best_idx = fitnesses.index(gen_best)
                global_best_genome = copy.deepcopy(population[best_idx])
                stagnation_count = 0
                if verbose:
                    print(f"  NEW GLOBAL BEST: {global_best:.4f} ({genome_to_name(global_best_genome)})")
            else:
                stagnation_count += 1

            # Selection + reproduction
            ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
            new_pop = [copy.deepcopy(ranked[i][1]) for i in range(min(elite_k, len(ranked)))]

            while len(new_pop) < pop_size:
                if random.random() < 0.7:
                    p1 = tournament_select(population, fitnesses)
                    p2 = tournament_select(population, fitnesses)
                    child = crossover(p1, p2)
                else:
                    child = tournament_select(population, fitnesses)
                child = mutate(child, mutation_rate)
                new_pop.append(child)

            population = new_pop

            # Save generation
            gen_path = os.path.join(results_dir, f"cycle{cycle}_gen{gen}.json")
            with open(gen_path, "w") as f:
                json.dump({
                    "cycle": cycle, "generation": global_gen,
                    "best_fitness": gen_best,
                    "global_best": global_best,
                    "stagnation": stagnation_count,
                    "n_terms": len(TERM_REGISTRY),
                }, f, indent=2)

        # ── Level 2: LLM term invention (if stagnated) ──
        if use_llm and stagnation_count >= stagnation_threshold:
            if verbose:
                print(f"\n{'*'*70}")
                print(f"  GA STAGNATED ({stagnation_count} gens). Asking LLM for new terms...")
                print(f"{'*'*70}")

            try:
                leaderboard = sorted(all_results, key=lambda x: -x["fitness"])[:5]
                proposals = ask_llm_for_new_terms(
                    current_terms=list(TERM_REGISTRY.keys()),
                    leaderboard=leaderboard,
                    stagnation_gens=stagnation_count,
                )

                for p in proposals:
                    err = register_new_term(p["name"], p["code"], p["maximize"])
                    if err:
                        if verbose:
                            print(f"  Term '{p['name']}' rejected: {err}")
                    else:
                        if verbose:
                            print(f"  NEW TERM registered: '{p['name']}' ({'max' if p['maximize'] else 'min'})")

                        # Inject genomes using new term
                        for _ in range(3):
                            g = random_genome(n_terms=random.randint(2, 3))
                            # Ensure new term is included
                            g["terms"].append({
                                "name": p["name"],
                                "weight": round(random.uniform(0.3, 1.5), 2),
                                "sign": -1 if p["maximize"] else 1,
                            })
                            population.append(g)

                        # Trim population back to size
                        if len(population) > pop_size:
                            # Keep elite + new injections
                            population = population[:pop_size]

                stagnation_count = 0

            except Exception as e:
                if verbose:
                    print(f"  LLM call failed: {e}")

    # Final report
    all_results.sort(key=lambda x: -x["fitness"])

    if verbose:
        print(f"\n{'='*70}")
        print(f"{'TWO-LEVEL EVOLUTION COMPLETE':^70}")
        print(f"{'='*70}")
        print(f"Total evaluations: {len(all_results)}")
        print(f"Final term count: {len(TERM_REGISTRY)}")
        print(f"Best SR: {global_best:.4f}")
        if global_best_genome:
            print(f"Best genome: {genome_to_description(global_best_genome)}")
        print(f"\nTop 10:")
        seen = set()
        for r in all_results:
            key = json.dumps(r["genome"], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            print(f"  SR={r['fitness']:.4f} | {r['description'][:60]}")
            if len(seen) >= 10:
                break

    # Save
    final_path = os.path.join(results_dir, "two_level_final.json")
    with open(final_path, "w") as f:
        json.dump({
            "global_best": global_best,
            "global_best_genome": global_best_genome,
            "total_evals": len(all_results),
            "final_terms": list(TERM_REGISTRY.keys()),
            "top_20": all_results[:20],
        }, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    import argparse
    from trainer import train_single_ticker

    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--ga-gens", type=int, default=5, help="GA generations per cycle")
    parser.add_argument("--cycles", type=int, default=4, help="Outer cycles (LLM invention)")
    parser.add_argument("--stagnation", type=int, default=3, help="Gens before LLM kicks in")
    parser.add_argument("--no-llm", action="store_true", help="GA only, no LLM")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def evaluate(loss_fn, name):
        results = {}
        for ticker in config.STAGE1_TICKERS:
            r = train_single_ticker(
                ticker=ticker, loss_fn=loss_fn,
                max_epochs=config.STAGE1_MAX_EPOCHS,
                eval_every=config.STAGE1_EVAL_EVERY,
                patience=config.STAGE1_PATIENCE,
                device=device, verbose=False,
            )
            results[ticker] = r["val_sr"]
        mean_sr = float(np.mean(list(results.values())))
        print(f"    {name}: mean={mean_sr:.4f}")
        return mean_sr

    run_two_level(
        evaluate_fn=evaluate,
        pop_size=args.pop_size,
        ga_gens_per_cycle=args.ga_gens,
        max_cycles=args.cycles,
        stagnation_threshold=args.stagnation,
        use_llm=not args.no_llm,
        results_dir=os.path.join(config.RESULTS_DIR, "two_level"),
    )
