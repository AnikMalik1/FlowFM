#!/usr/bin/env python3
"""
SE-Agent Style Loss Evolution for FinGAN — v2.

Based on: "SE-Agent: Self-Evolution Trajectory Optimization" (NeurIPS 2025)
Three operations: Revision, Recombination, Refinement.

v2 improvements (inspired by Goldie et al. RLC 2025):
1. JSON structured output with "thought" field (forced reasoning)
2. Baseline-normalized SR reporting
3. Multi-seed evaluation for robustness
"""
import argparse, os, sys, json, time, copy, re
from datetime import datetime
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from loss_registry import LossRegistry
from trainer_fingan import train_single_ticker_fingan


# ── LLM Client ──────────────────────────────────────────

def _llm_call(system: str, user: str, max_tokens: int = 3000) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    kwargs = dict(
        model=config.LLM_MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    # Try JSON mode
    try:
        kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content
    except Exception:
        pass

    kwargs.pop("response_format", None)
    r = client.chat.completions.create(**kwargs)
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


def _parse_response(text: str) -> dict:
    """Parse LLM response: try JSON first, fallback to regex."""
    # 1. Direct JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "code" in data:
            data.setdefault("name", "unnamed")
            data.setdefault("thought", "")
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. JSON in markdown block
    for pattern in [r'```json\s*([\s\S]*?)```', r'```\s*(\{[\s\S]*?\})\s*```']:
        m = re.search(pattern, text)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, dict) and "code" in data:
                    data.setdefault("name", "unnamed")
                    data.setdefault("thought", "")
                    return data
            except (json.JSONDecodeError, TypeError):
                continue

    # 3. Brace-matching for embedded JSON
    brace_depth = 0
    json_start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start is not None:
                candidate = text[json_start:i+1]
                if '"code"' in candidate:
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict) and "code" in data:
                            data.setdefault("name", "unnamed")
                            data.setdefault("thought", "")
                            return data
                    except (json.JSONDecodeError, TypeError):
                        pass
                json_start = None

    # 4. Fallback: regex extraction
    code = _extract_code(text)
    m = re.search(r'[Nn]ame[:\s]+[`"]*(\w+)[`"]*', text)
    name = m.group(1) if m else "unnamed"
    return {"thought": text[:300], "name": name, "code": code}


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
- PnL: -mean(tanh(T*gen_out) * real)  -- maximize trading profit
- MSE: mean((gen_out - real)^2)  -- prediction accuracy
- Sharpe: -mean(pnl) / std(pnl)  -- risk-adjusted return
- STD: std(tanh(T*gen_out) * real)  -- PnL volatility
- Sortino: -mean(pnl) / sqrt(mean(relu(-pnl)^2))  -- downside risk
- Winrate: mean(relu(margin - gen_out*real))  -- directional accuracy hinge

Rules:
1. Return scalar Tensor, must be differentiable
2. Can use torch.*, math.*
3. Function name MUST be loss_fn
4. Can combine existing terms OR invent new ones

When you respond, output a JSON object with exactly three keys:
1. "thought": Your analysis and reasoning (2-4 sentences). Be specific about mechanisms.
2. "name": snake_case name for the loss function
3. "code": The exact Python code defining loss_fn (as a single string, use \\n for newlines)
"""


# ── Evaluation ───────────────────────────────────────────

def evaluate_loss(code: str, name: str, registry: LossRegistry, device,
                  n_seeds: int = 3, baseline_sr: float = 1.0,
                  tickers: list = None) -> dict:
    """Load loss_fn from code, validate, evaluate on tickers x n_seeds seeds."""
    from loss_registry import _scan_code_safety, _sanitize_name
    import importlib.util

    name = _sanitize_name(name)

    err = _scan_code_safety(code)
    if err:
        return {"name": name, "sr": -999, "norm_sr": -999, "error": f"safety: {err}", "code": code}

    path = os.path.join(config.LOSSES_DIR, f"{name}.py")
    with open(path, "w") as f:
        f.write(code)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.loss_fn
    except Exception as e:
        return {"name": name, "sr": -999, "norm_sr": -999, "error": f"load: {e}", "code": code}

    # Validate
    try:
        test_out = torch.randn(32, requires_grad=True)
        test_real = torch.randn(32)
        loss = fn(test_out, test_real, 100.0)
        if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
            return {"name": name, "sr": -999, "norm_sr": -999, "error": f"bad output: {type(loss)}", "code": code}
        loss.backward()
    except Exception as e:
        return {"name": name, "sr": -999, "norm_sr": -999, "error": f"validate: {e}", "code": code}

    # Evaluate on FinGAN
    if tickers is None:
        tickers = config.META_TRAIN_TICKERS
    seeds = list(range(42, 42 + n_seeds))
    results = {}
    for ticker in tickers:
        sr_per_seed = []
        for seed in seeds:
            try:
                r = train_single_ticker_fingan(
                    ticker=ticker, financial_loss_fn=fn,
                    max_epochs=100, warmup_epochs=15,
                    eval_every=20, patience=6,
                    device=device, verbose=False, seed=seed)
                sr_per_seed.append(r["val_sr"])
            except Exception as e:
                sr_per_seed.append(-999)
                print(f"    {ticker} seed={seed} FAILED: {e}")
        valid_sr = [s for s in sr_per_seed if s > -900]
        results[ticker] = float(np.mean(valid_sr)) if valid_sr else -999

    mean_sr = float(np.mean([v for v in results.values() if v > -900]))
    if not any(v > -900 for v in results.values()):
        mean_sr = -999

    norm_sr = mean_sr / baseline_sr if baseline_sr > 0 and mean_sr > -900 else -999

    return {"name": name, "sr": mean_sr, "norm_sr": norm_sr,
            "per_ticker": results, "code": code}


# ── Operation 1: Revision (Multi-Planning + Reflection) ─

PLANNING_STRATEGIES = [
    "Focus on maximizing Sharpe Ratio by penalizing large drawdowns.",
    "Focus on directional accuracy: reward correct sign prediction, ignore magnitude.",
    "Focus on risk parity: scale PnL by inverse volatility of recent returns.",
    "Focus on tail risk: use CVaR or expected shortfall instead of mean PnL.",
    "Focus on robust estimation: use Huber loss or trimmed mean for PnL.",
]

def generate_initial_population(n: int = 5, baseline_context: str = "") -> list[dict]:
    """Multi-Planning: generate improvements on baselines using different strategies."""
    population = []
    for i, strategy in enumerate(PLANNING_STRATEGIES[:n]):
        prompt = (
            f"Here are the current best loss functions for FinGAN:\n\n"
            f"{baseline_context}\n\n"
            f"Your task: IMPROVE on these baselines using this strategy:\n"
            f"  {strategy}\n\n"
            f"Start from the best baseline's structure and make targeted changes.\n"
            f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
        )

        text = _llm_call(LOSS_SYSTEM, prompt)
        parsed = _parse_response(text)
        population.append({
            "name": f"plan_{i}",
            "code": parsed["code"],
            "strategy": strategy,
            "thought": parsed.get("thought", ""),
            "reasoning": parsed.get("thought", text[:300]),
        })
        print(f"  Generated plan_{i}: {strategy[:60]}")
        if parsed.get("thought"):
            print(f"    Thought: {parsed['thought'][:150]}...")
    return population


def reflect_and_revise(candidate: dict, eval_result: dict, baseline_sr: float) -> dict:
    """Reflection: analyze why a loss worked/failed, then revise."""
    norm = eval_result['sr'] / baseline_sr if baseline_sr > 0 else 0
    prompt = (
        f"Analyze this loss function and its results, then create an improved version.\n\n"
        f"## Original Loss\n```python\n{candidate['code']}\n```\n\n"
        f"## Results\n"
        f"Mean Sharpe Ratio: {eval_result['sr']:.4f} (normalized: {norm:.3f})\n"
        f"Per-ticker: {json.dumps(eval_result.get('per_ticker', {}), indent=2)}\n\n"
        f"## Task\n"
        f"1. Identify what works and what doesn't in this loss\n"
        f"2. Propose specific improvements\n"
        f"3. Output a REVISED loss function that addresses the weaknesses\n\n"
        f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
    )

    text = _llm_call(LOSS_SYSTEM, prompt)
    parsed = _parse_response(text)
    return {
        "name": f"{candidate['name']}_rev",
        "code": parsed["code"],
        "parent": candidate["name"],
        "thought": parsed.get("thought", ""),
        "reasoning": parsed.get("thought", text[:300]),
    }


# ── Operation 2: Recombination (Crossover + Transfer) ───

def crossover(cand_a: dict, cand_b: dict) -> dict:
    """Combine best elements from two loss functions."""
    prompt = (
        f"Combine the best elements from these two loss functions into a new, superior one.\n\n"
        f"## Loss A (SR={cand_a.get('sr', '?')}, norm={cand_a.get('norm_sr', '?')})\n"
        f"```python\n{cand_a['code']}\n```\n\n"
        f"## Loss B (SR={cand_b.get('sr', '?')}, norm={cand_b.get('norm_sr', '?')})\n"
        f"```python\n{cand_b['code']}\n```\n\n"
        f"## Task\n"
        f"1. Identify strengths of each loss\n"
        f"2. Create a NEW loss that combines the best parts\n"
        f"3. Don't just concatenate -- create genuine synthesis\n\n"
        f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
    )

    text = _llm_call(LOSS_SYSTEM, prompt)
    parsed = _parse_response(text)
    return {
        "name": f"cross_{cand_a['name']}_{cand_b['name']}",
        "code": parsed["code"],
        "parents": [cand_a["name"], cand_b["name"]],
        "thought": parsed.get("thought", ""),
        "reasoning": parsed.get("thought", text[:300]),
    }


def transfer(target: dict, references: list[dict]) -> dict:
    """Transfer winning patterns from reference losses to enhance target."""
    ref_texts = "\n\n".join(
        f"### {r['name']} (SR={r.get('sr', '?')}, norm={r.get('norm_sr', '?')})\n```python\n{r['code']}\n```"
        for r in references[:3]
    )

    prompt = (
        f"Enhance this target loss by transferring effective patterns from the reference losses.\n\n"
        f"## Target Loss (SR={target.get('sr', '?')}, norm={target.get('norm_sr', '?')})\n"
        f"```python\n{target['code']}\n```\n\n"
        f"## Reference Losses (successful)\n{ref_texts}\n\n"
        f"## Task\n"
        f"1. Identify what makes the references successful\n"
        f"2. Transfer those patterns to improve the target\n"
        f"3. Keep what works in the target, add what's missing\n\n"
        f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
    )

    text = _llm_call(LOSS_SYSTEM, prompt)
    parsed = _parse_response(text)
    return {
        "name": f"transfer_{target['name']}",
        "code": parsed["code"],
        "thought": parsed.get("thought", ""),
        "reasoning": parsed.get("thought", text[:300]),
    }


def restructure(pool: list[dict], mode: str = "global") -> dict:
    """Restructure: synthesize new loss from pool. 3 modes for diversity."""
    sorted_pool = sorted(pool, key=lambda x: -x.get("sr", -999))
    pool_text = "\n\n".join(
        f"### {c['name']} (SR={c.get('sr', '?')}, norm={c.get('norm_sr', '?')})\n```python\n{c['code']}\n```"
        for c in sorted_pool[:5]
    )

    if mode == "contrast":
        best = sorted_pool[0]
        worst = sorted_pool[-1] if sorted_pool[-1].get("sr", -999) > -900 else sorted_pool[-2]
        prompt = (
            f"Compare the BEST and WORST loss functions. Extract what makes the difference.\n\n"
            f"## Best: {best['name']} (SR={best.get('sr', '?')})\n```python\n{best['code']}\n```\n\n"
            f"## Worst: {worst['name']} (SR={worst.get('sr', '?')})\n```python\n{worst['code']}\n```\n\n"
            f"## Task\n"
            f"1. WHY does the best work and worst fail? Be specific about mechanisms.\n"
            f"2. Create a NEW loss that maximizes the good and avoids the bad.\n\n"
            f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
        )
    elif mode == "novel":
        prompt = (
            f"Here are the current best losses:\n\n{pool_text}\n\n"
            f"## Task\n"
            f"All existing losses use variations of PnL, Sharpe, MSE, STD.\n"
            f"Create something GENUINELY NEW that no existing loss does:\n"
            f"- Consider information-theoretic measures (mutual information, entropy)\n"
            f"- Consider distributional matching (Wasserstein, MMD)\n"
            f"- Consider higher-order moments (skewness, kurtosis of PnL)\n"
            f"- Consider regime-adaptive weighting\n"
            f"Must still be differentiable and use the loss_fn(gen_out, real, tanh_temp) interface.\n\n"
            f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
        )
    else:  # global (default)
        prompt = (
            f"Synthesize a new loss function from global analysis of this pool.\n\n"
            f"## Loss Function Pool (top 5)\n{pool_text}\n\n"
            f"## Task\n"
            f"1. Find abstract patterns across ALL successful losses\n"
            f"2. Identify what the best losses share\n"
            f"3. Create a NEW loss that captures these patterns in a novel way\n"
            f"4. Don't copy any single loss -- synthesize\n\n"
            f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
        )

    text = _llm_call(LOSS_SYSTEM, prompt)
    parsed = _parse_response(text)
    return {
        "name": f"restructure_{mode}_{len(pool)}",
        "code": parsed["code"],
        "thought": parsed.get("thought", ""),
        "reasoning": parsed.get("thought", text[:300]),
    }


# ── Operation 3: Refinement (Evaluate + Select) ─────────

def select_elite(pool: list[dict], k: int = 5) -> list[dict]:
    """Select top-K by SR, ensuring diversity."""
    sorted_pool = sorted(pool, key=lambda x: -x.get("sr", -999))
    elite = []
    seen_codes = set()
    for c in sorted_pool:
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
    n_seeds: int = 3,
    device=None,
    results_dir: str = "results/se_agent_v3",
    verbose: bool = True,
    resume_path: str = None,
):
    os.makedirs(results_dir, exist_ok=True)
    registry = LossRegistry()
    all_results = []
    baseline_sr = 1.0
    start_cycle = 1

    # ── Resume from previous run ─────────────────────────
    if resume_path and os.path.exists(resume_path):
        print(f"\n{'='*60}")
        print(f"  Resuming from {resume_path}")
        print(f"{'='*60}")
        with open(resume_path) as f:
            data = json.load(f)
        population = data.get("population", [])
        baseline_sr = data.get("baseline_sr", 1.0)
        start_cycle = data.get("cycle", 0) + 1
        all_results = [{"name": r["name"], "sr": r["sr"],
                        "norm_sr": r.get("norm_sr")} for r in data.get("all_results", [])]
        print(f"  Loaded pool of {len(population)}, baseline_sr={baseline_sr:.4f}")
        print(f"  Starting from cycle {start_cycle}")

    else:
        # ── Phase 0: Evaluate ALL baselines with code ────
        print(f"\n{'='*60}")
        print(f"  SE-Agent v3: Evaluating 9 baselines as seed population")
        print(f"{'='*60}")
        from losses.fingan_baselines import FINGAN_BASELINE_CODES

        population = []
        for bname, bcode in FINGAN_BASELINE_CODES.items():
            print(f"\n--- {bname} ---")
            result = evaluate_loss(bcode, bname, registry, device,
                                   n_seeds=n_seeds, baseline_sr=1.0)
            population.append({
                "name": bname, "code": bcode,
                "sr": result["sr"], "norm_sr": 0,
                "per_ticker": result.get("per_ticker", {}),
                "origin": "baseline",
            })
            all_results.append(result)
            print(f"  SR = {result['sr']:.4f}")

        # Set baseline_sr from bce_only (no financial loss)
        bce_result = next((p for p in population if p["name"] == "baseline_bce_only"), None)
        if bce_result and bce_result["sr"] > 0:
            baseline_sr = bce_result["sr"]
        else:
            baseline_sr = max(p["sr"] for p in population if p["sr"] > -900) or 1.0

        # Update norm_sr for all baselines
        for p in population:
            p["norm_sr"] = p["sr"] / baseline_sr if p["sr"] > -900 else -999

        # Leaderboard
        sorted_bl = sorted(population, key=lambda x: -x.get("sr", -999))
        print(f"\n--- Baseline Leaderboard (norm by BCE-only SR={baseline_sr:.4f}) ---")
        for i, p in enumerate(sorted_bl):
            print(f"  {i+1}. SR={p['sr']:.4f} (norm={p.get('norm_sr',0):.3f}) | {p['name']}")

        # ── Phase 1: LLM improves on baselines ──────────
        print(f"\n{'='*60}")
        print(f"  SE-Agent v3: LLM generating {n_initial} improvements on baselines")
        print(f"{'='*60}")

        # Build context: top-3 baseline code + results
        top3 = sorted_bl[:3]
        baseline_context = "\n\n".join(
            f"### {p['name']} (SR={p['sr']:.4f}, norm={p.get('norm_sr',0):.3f})\n"
            f"```python\n{p['code']}\n```"
            for p in top3
        )

        llm_plans = generate_initial_population(n_initial, baseline_context=baseline_context)

        # Evaluate LLM plans
        print(f"\n--- Evaluating LLM improvements ---")
        for i, cand in enumerate(llm_plans):
            print(f"\n[{i+1}/{len(llm_plans)}] {cand['name']}")
            result = evaluate_loss(cand["code"], cand["name"], registry, device,
                                   n_seeds=n_seeds, baseline_sr=baseline_sr)
            cand["sr"] = result["sr"]
            cand["norm_sr"] = result.get("norm_sr", -999)
            cand["per_ticker"] = result.get("per_ticker", {})
            all_results.append(result)
            print(f"  SR = {result['sr']:.4f} (norm={result.get('norm_sr', -999):.3f})")

        population.extend(llm_plans)

        # Select top-8 for cycles
        population = select_elite(population, k=8)
        print(f"\n--- Init complete. Top-8 elite pool ---")
        for i, p in enumerate(population):
            print(f"  {i+1}. SR={p.get('sr',-999):.4f} (norm={p.get('norm_sr',0):.3f}) | {p['name']}")

    for cycle in range(start_cycle, n_cycles + 1):
        print(f"\n{'#'*60}")
        print(f"  SE-Agent CYCLE {cycle}/{n_cycles}")
        best_sr = max(c.get('sr', -999) for c in population)
        best_norm = best_sr / baseline_sr if baseline_sr > 0 else 0
        print(f"  Pool size: {len(population)}, Best SR: {best_sr:.4f} (norm={best_norm:.3f})")
        print(f"{'#'*60}")

        new_candidates = []

        # ── Revision: only best-1 (was 3, reduced to avoid crashes) ──
        print(f"\n--- Revision (best-1 only) ---")
        for cand in select_elite(population, k=1):
            if cand.get("sr", -999) <= -900:
                continue
            result = next((r for r in all_results if r["name"] == cand["name"]), None)
            if result:
                revised = reflect_and_revise(cand, result, baseline_sr)
                new_candidates.append(revised)
                print(f"  Revised {cand['name']} -> {revised['name']}")

        # ── Recombination: crossover + transfer ───────────
        print(f"\n--- Recombination ---")
        elite = select_elite(population, k=4)
        if len(elite) >= 2:
            crossed = crossover(elite[0], elite[1])
            new_candidates.append(crossed)
            print(f"  Crossover: {elite[0]['name']} x {elite[1]['name']} -> {crossed['name']}")

            if len(elite) >= 3:
                transferred = transfer(elite[-1], elite[:2])
                new_candidates.append(transferred)
                print(f"  Transfer: top-2 -> {transferred['name']}")

        # ── Restructure x3 (global, contrast, novel) ─────
        print(f"\n--- Restructure x3 ---")
        for mode in ["global", "contrast", "novel"]:
            restructured = restructure(population, mode=mode)
            new_candidates.append(restructured)
            print(f"  Restructure ({mode}): -> {restructured['name']}")

        # ── Evaluate all new candidates ──────────────────
        print(f"\n--- Evaluate {len(new_candidates)} new candidates ---")
        for i, cand in enumerate(new_candidates):
            name = f"c{cycle}_{cand['name']}"
            cand["name"] = name
            print(f"\n[{i+1}/{len(new_candidates)}] {name}")
            result = evaluate_loss(cand["code"], name, registry, device,
                                   n_seeds=n_seeds, baseline_sr=baseline_sr)
            cand["sr"] = result["sr"]
            cand["norm_sr"] = result.get("norm_sr", -999)
            cand["per_ticker"] = result.get("per_ticker", {})
            all_results.append(result)
            print(f"  SR = {result['sr']:.4f} (norm={result.get('norm_sr', -999):.3f})")

        # ── Refinement: select top-K for next cycle ──────
        population.extend(new_candidates)
        population = select_elite(population, k=8)

        # Leaderboard
        print(f"\n--- Cycle {cycle} Leaderboard ---")
        for i, c in enumerate(population[:10]):
            norm = c.get('sr', -999) / baseline_sr if baseline_sr > 0 else 0
            print(f"  {i+1}. SR={c.get('sr',-999):.4f} (norm={norm:.3f}) | {c['name']}")

        # Save cycle results (enables resume)
        cycle_path = os.path.join(results_dir, f"cycle_{cycle}.json")
        with open(cycle_path, "w") as f:
            json.dump({
                "cycle": cycle,
                "baseline_sr": baseline_sr,
                "population": [{"name": c["name"], "sr": c.get("sr"),
                                "norm_sr": c.get("norm_sr"), "code": c["code"]}
                               for c in population],
                "all_results": [{"name": r["name"], "sr": r.get("sr", r.get("mean_sr")),
                                 "norm_sr": r.get("norm_sr")} for r in all_results],
                "new_candidates": [{"name": c["name"], "sr": c.get("sr"),
                                    "norm_sr": c.get("norm_sr")} for c in new_candidates],
            }, f, indent=2, default=str)

    # ── Validation Phase: held-out tickers ─────────────────
    print(f"\n{'='*60}")
    print(f"  VALIDATION PHASE (held-out tickers: {config.META_VAL_TICKERS})")
    print(f"{'='*60}")
    top5 = select_elite(population, k=5)
    val_results = []
    for cand in top5:
        print(f"\n  Validating: {cand['name']} (train_SR={cand.get('sr', -999):.4f})")
        vr = evaluate_loss(cand["code"], f"val_{cand['name']}", registry, device,
                           n_seeds=config.META_VAL_SEEDS, baseline_sr=baseline_sr,
                           tickers=config.META_VAL_TICKERS)
        val_results.append({"name": cand["name"], "train_sr": cand.get("sr"),
                            "val_sr": vr["sr"], "val_per_ticker": vr.get("per_ticker", {}),
                            "code": cand["code"]})
        print(f"  val_SR = {vr['sr']:.4f} (train_SR was {cand.get('sr', -999):.4f})")

    val_results.sort(key=lambda x: -x.get("val_sr", -999))
    print(f"\n--- Validation Leaderboard ---")
    for i, vr in enumerate(val_results):
        val_norm = vr["val_sr"] / baseline_sr if baseline_sr > 0 else 0
        print(f"  {i+1}. val_SR={vr['val_sr']:.4f} (norm={val_norm:.3f}) "
              f"| train_SR={vr['train_sr']:.4f} | {vr['name']}")

    winner = val_results[0]
    winner_norm = winner["val_sr"] / baseline_sr if baseline_sr > 0 else 0

    # Final
    print(f"\n{'='*60}")
    print(f"  SE-Agent v4 COMPLETE")
    print(f"  Winner (by val_SR): {winner['name']}")
    print(f"    val_SR={winner['val_sr']:.4f} (norm={winner_norm:.3f})")
    print(f"    train_SR={winner['train_sr']:.4f}")
    print(f"  Total train evaluations: {len(all_results)}")
    print(f"{'='*60}")
    print(f"\nWinner loss code:\n{winner['code']}")

    final_path = os.path.join(results_dir, "se_agent_v4_final.json")
    with open(final_path, "w") as f:
        json.dump({
            "cycle": n_cycles,
            "baseline_sr": baseline_sr,
            "winner": winner,
            "validation_results": val_results,
            "all_train_results": [{"name": r["name"], "sr": r.get("sr", r.get("mean_sr")),
                                   "norm_sr": r.get("norm_sr")} for r in all_results],
            "final_population": [{"name": c["name"], "sr": c.get("sr"),
                                  "norm_sr": c.get("norm_sr"), "code": c["code"]}
                                 for c in population],
        }, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to cycle_N.json or se_agent_v3_final.json to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    run_se_agent(n_cycles=args.cycles, n_initial=args.initial,
                 n_seeds=args.n_seeds, device=device,
                 resume_path=args.resume)
