#!/usr/bin/env python3
"""
SE-Agent Style Loss Evolution for FinGAN.

Based on: "SE-Agent: Self-Evolution Trajectory Optimization" (NeurIPS 2025)
Three operations: Revision, Recombination, Refinement.

Unlike GA (fixed primitives + weight optimization), this operates on
FULL LOSS FUNCTION CODE via LLM semantic operations.
"""
import argparse, os, sys, json, time, copy
from datetime import datetime
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from loss_registry import LossRegistry
from trainer_fingan import train_single_ticker_fingan
from evaluator_dual import _adapt_ga_genome_to_financial_loss


# ── LLM Client ──────────────────────────────────────────

def _llm_call(system: str, user: str, max_tokens: int = 3000) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    r = client.chat.completions.create(
        model=config.LLM_MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return r.choices[0].message.content


def _extract_code(text: str) -> str:
    if "```python" in text:
        s = text.index("```python") + len("```python")
        e = text.index("```", s)
        return text[s:e].strip()
    if "```" in text:
        s = text.index("```") + 3
        e = text.index("```", s)
        return text[s:e].strip()
    return text


LOSS_SYSTEM = """You design loss functions for FinGAN (GAN for probabilistic return forecasting).

Generator: LSTM + Linear, outputs predicted return (scalar).
Discriminator: LSTM + Linear + Sigmoid.

Financial loss interface:
```python
def loss_fn(gen_out, real, tanh_temp):
    # gen_out: (B,) generated returns
    # real: (B,) actual returns
    # tanh_temp: float (100.0)
    # Returns: scalar Tensor (will be ADDED to BCE loss with gradient-balanced weight)
    ...
```

Known effective terms:
- PnL: -mean(tanh(T*gen_out) * real)  — maximize trading profit
- MSE: mean((gen_out - real)^2)  — prediction accuracy
- Sharpe: -mean(pnl) / std(pnl)  — risk-adjusted return
- STD: std(tanh(T*gen_out) * real)  — PnL volatility
- Sortino: -mean(pnl) / sqrt(mean(relu(-pnl)^2))  — downside risk
- Winrate: mean(relu(margin - gen_out*real))  — directional accuracy hinge

Rules:
1. Return scalar Tensor, must be differentiable
2. Can use torch.*, math.*
3. Function name MUST be loss_fn
4. Can combine existing terms OR invent new ones
5. Can use curriculum scheduling via epoch (not available here, ignore)
"""


# ── Evaluation ───────────────────────────────────────────

def evaluate_loss(code: str, name: str, registry: LossRegistry, device) -> dict:
    """Register, validate, and evaluate a loss on FinGAN (3 tickers)."""
    err = registry.register_from_code(name, code, origin="se_agent")
    if err:
        return {"name": name, "sr": -999, "error": f"register: {err}", "code": code}

    err = registry.validate_loss_fn(name)
    if err:
        return {"name": name, "sr": -999, "error": f"validate: {err}", "code": code}

    fn = registry.get(name)
    fingan_fn = _adapt_ga_genome_to_financial_loss(fn)

    results = {}
    for ticker in config.STAGE1_TICKERS:
        try:
            r = train_single_ticker_fingan(
                ticker=ticker, financial_loss_fn=fingan_fn,
                max_epochs=100, warmup_epochs=15,
                eval_every=20, patience=6,
                device=device, verbose=False)
            results[ticker] = r["val_sr"]
        except Exception as e:
            results[ticker] = -999
            print(f"    {ticker} FAILED: {e}")

    mean_sr = float(np.mean([v for v in results.values() if v > -900]))
    if not any(v > -900 for v in results.values()):
        mean_sr = -999

    return {"name": name, "sr": mean_sr, "per_ticker": results, "code": code}


# ── Operation 1: Revision (Multi-Planning + Reflection) ─

PLANNING_STRATEGIES = [
    "Focus on maximizing Sharpe Ratio by penalizing large drawdowns.",
    "Focus on directional accuracy: reward correct sign prediction, ignore magnitude.",
    "Focus on risk parity: scale PnL by inverse volatility of recent returns.",
    "Focus on tail risk: use CVaR or expected shortfall instead of mean PnL.",
    "Focus on robust estimation: use Huber loss or trimmed mean for PnL.",
]

def generate_initial_population(n: int = 5) -> list[dict]:
    """Multi-Planning: generate diverse loss functions using different strategies."""
    population = []
    for i, strategy in enumerate(PLANNING_STRATEGIES[:n]):
        prompt = f"""Generate a novel loss function for FinGAN.

Strategy: {strategy}

Output ONLY the Python code block with loss_fn(gen_out, real, tanh_temp)."""

        text = _llm_call(LOSS_SYSTEM, prompt)
        code = _extract_code(text)
        population.append({
            "name": f"plan_{i}",
            "code": code,
            "strategy": strategy,
            "reasoning": text[:300],
        })
        print(f"  Generated plan_{i}: {strategy[:60]}")
    return population


def reflect_and_revise(candidate: dict, eval_result: dict) -> dict:
    """Reflection: analyze why a loss worked/failed, then revise."""
    prompt = f"""Analyze this loss function and its results, then create an improved version.

## Original Loss
```python
{candidate['code']}
```

## Results
Mean Sharpe Ratio: {eval_result['sr']:.4f}
Per-ticker: {json.dumps(eval_result.get('per_ticker', {}), indent=2)}

## Task
1. Identify what works and what doesn't in this loss
2. Propose specific improvements
3. Output a REVISED loss function that addresses the weaknesses

Output reasoning (2-3 sentences) then the Python code block."""

    text = _llm_call(LOSS_SYSTEM, prompt)
    code = _extract_code(text)
    return {
        "name": f"{candidate['name']}_rev",
        "code": code,
        "parent": candidate["name"],
        "reasoning": text[:300],
    }


# ── Operation 2: Recombination (Crossover + Transfer) ───

def crossover(cand_a: dict, cand_b: dict) -> dict:
    """Combine best elements from two loss functions."""
    prompt = f"""Combine the best elements from these two loss functions into a new, superior one.

## Loss A (SR={cand_a.get('sr', '?')})
```python
{cand_a['code']}
```

## Loss B (SR={cand_b.get('sr', '?')})
```python
{cand_b['code']}
```

## Task
1. Identify strengths of each loss
2. Create a NEW loss that combines the best parts
3. Don't just concatenate — create genuine synthesis

Output reasoning then Python code block."""

    text = _llm_call(LOSS_SYSTEM, prompt)
    code = _extract_code(text)
    return {
        "name": f"cross_{cand_a['name']}_{cand_b['name']}",
        "code": code,
        "parents": [cand_a["name"], cand_b["name"]],
        "reasoning": text[:300],
    }


def transfer(target: dict, references: list[dict]) -> dict:
    """Transfer winning patterns from reference losses to enhance target."""
    ref_texts = "\n\n".join(
        f"### {r['name']} (SR={r.get('sr', '?')})\n```python\n{r['code']}\n```"
        for r in references[:3]
    )

    prompt = f"""Enhance this target loss by transferring effective patterns from the reference losses.

## Target Loss (SR={target.get('sr', '?')})
```python
{target['code']}
```

## Reference Losses (successful)
{ref_texts}

## Task
1. Identify what makes the references successful
2. Transfer those patterns to improve the target
3. Keep what works in the target, add what's missing

Output reasoning then Python code block."""

    text = _llm_call(LOSS_SYSTEM, prompt)
    code = _extract_code(text)
    return {
        "name": f"transfer_{target['name']}",
        "code": code,
        "reasoning": text[:300],
    }


def restructure(pool: list[dict]) -> dict:
    """Global restructuring: synthesize from entire pool."""
    pool_text = "\n\n".join(
        f"### {c['name']} (SR={c.get('sr', '?')})\n```python\n{c['code']}\n```"
        for c in sorted(pool, key=lambda x: -x.get("sr", -999))[:5]
    )

    prompt = f"""Synthesize a new loss function from global analysis of this pool.

## Loss Function Pool (top 5)
{pool_text}

## Task
1. Find abstract patterns across ALL successful losses
2. Identify what the best losses share
3. Create a NEW loss that captures these patterns in a novel way
4. Don't copy any single loss — synthesize

Output reasoning then Python code block."""

    text = _llm_call(LOSS_SYSTEM, prompt)
    code = _extract_code(text)
    return {
        "name": f"restructure_{len(pool)}",
        "code": code,
        "reasoning": text[:300],
    }


# ── Operation 3: Refinement (Evaluate + Select) ─────────

def select_elite(pool: list[dict], k: int = 5) -> list[dict]:
    """Select top-K by SR, ensuring diversity."""
    sorted_pool = sorted(pool, key=lambda x: -x.get("sr", -999))
    elite = []
    seen_codes = set()
    for c in sorted_pool:
        # Simple diversity: skip near-duplicate code
        code_hash = hash(c["code"][:200])
        if code_hash in seen_codes:
            continue
        seen_codes.add(code_hash)
        elite.append(c)
        if len(elite) >= k:
            break
    return elite


# ── Main SE-Agent Loop ──────────────────────────────────

def run_se_agent(
    n_cycles: int = 3,
    n_initial: int = 5,
    device=None,
    results_dir: str = "results/se_agent",
    verbose: bool = True,
):
    os.makedirs(results_dir, exist_ok=True)
    registry = LossRegistry()
    all_results = []

    # ── Phase 1: Generate diverse initial population ─────
    print(f"\n{'='*60}")
    print(f"  SE-Agent: Generating {n_initial} diverse loss functions")
    print(f"{'='*60}")

    population = generate_initial_population(n_initial)

    # Evaluate initial population
    print(f"\n--- Evaluating initial population ---")
    for i, cand in enumerate(population):
        print(f"\n[{i+1}/{len(population)}] {cand['name']}")
        result = evaluate_loss(cand["code"], cand["name"], registry, device)
        cand["sr"] = result["sr"]
        cand["per_ticker"] = result.get("per_ticker", {})
        all_results.append(result)
        print(f"  SR = {result['sr']:.4f}")

    for cycle in range(1, n_cycles + 1):
        print(f"\n{'#'*60}")
        print(f"  SE-Agent CYCLE {cycle}/{n_cycles}")
        print(f"  Pool size: {len(population)}, Best SR: {max(c.get('sr',-999) for c in population):.4f}")
        print(f"{'#'*60}")

        new_candidates = []

        # ── Revision: reflect + revise each candidate ────
        print(f"\n--- Revision ---")
        for cand in select_elite(population, k=3):
            if cand.get("sr", -999) <= -900:
                continue
            result = next((r for r in all_results if r["name"] == cand["name"]), None)
            if result:
                revised = reflect_and_revise(cand, result)
                new_candidates.append(revised)
                print(f"  Revised {cand['name']} -> {revised['name']}")

        # ── Recombination: crossover top pairs ───────────
        print(f"\n--- Recombination ---")
        elite = select_elite(population, k=4)
        if len(elite) >= 2:
            # Crossover top-1 × top-2
            crossed = crossover(elite[0], elite[1])
            new_candidates.append(crossed)
            print(f"  Crossover: {elite[0]['name']} × {elite[1]['name']} -> {crossed['name']}")

            # Transfer: top references -> weakest elite
            if len(elite) >= 3:
                transferred = transfer(elite[-1], elite[:2])
                new_candidates.append(transferred)
                print(f"  Transfer: {elite[:2]} -> {transferred['name']}")

            # Restructure from entire pool
            restructured = restructure(population)
            new_candidates.append(restructured)
            print(f"  Restructure: pool -> {restructured['name']}")

        # ── Evaluate all new candidates ──────────────────
        print(f"\n--- Evaluate {len(new_candidates)} new candidates ---")
        for i, cand in enumerate(new_candidates):
            name = f"c{cycle}_{cand['name']}"
            cand["name"] = name
            print(f"\n[{i+1}/{len(new_candidates)}] {name}")
            result = evaluate_loss(cand["code"], name, registry, device)
            cand["sr"] = result["sr"]
            cand["per_ticker"] = result.get("per_ticker", {})
            all_results.append(result)
            print(f"  SR = {result['sr']:.4f}")

        # ── Refinement: select top-K for next cycle ──────
        population.extend(new_candidates)
        population = select_elite(population, k=8)

        # Leaderboard
        print(f"\n--- Cycle {cycle} Leaderboard ---")
        for i, c in enumerate(population[:10]):
            print(f"  {i+1}. SR={c.get('sr',-999):.4f} | {c['name']}")

        # Save cycle results
        cycle_path = os.path.join(results_dir, f"cycle_{cycle}.json")
        with open(cycle_path, "w") as f:
            json.dump({
                "cycle": cycle,
                "population": [{"name": c["name"], "sr": c.get("sr"), "code": c["code"]} for c in population],
                "new_candidates": [{"name": c["name"], "sr": c.get("sr")} for c in new_candidates],
            }, f, indent=2, default=str)

    # Final
    best = max(population, key=lambda x: x.get("sr", -999))
    print(f"\n{'='*60}")
    print(f"  SE-Agent COMPLETE")
    print(f"  Best: {best['name']} SR={best.get('sr',-999):.4f}")
    print(f"  Total evaluations: {len(all_results)}")
    print(f"{'='*60}")
    print(f"\nBest loss code:\n{best['code']}")

    final_path = os.path.join(results_dir, "se_agent_final.json")
    with open(final_path, "w") as f:
        json.dump({
            "best": {"name": best["name"], "sr": best.get("sr"), "code": best["code"]},
            "all_results": [{"name": r["name"], "sr": r["sr"]} for r in all_results],
            "leaderboard": [{"name": c["name"], "sr": c.get("sr")} for c in population],
        }, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--initial", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    run_se_agent(n_cycles=args.cycles, n_initial=args.initial, device=device)
