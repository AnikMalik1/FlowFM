#!/usr/bin/env python3
"""
LLM-Proposal Evolution with Conversation History v2 — FinGAN model.

Improvements over v1 (inspired by Goldie et al. RLC 2025):
1. Cumulative conversation history (LLM sees ALL prior rounds)
2. JSON structured output with "thought" field (forced reasoning)
3. Baseline-normalized SR reporting
4. Multi-seed evaluation for robustness
"""
import argparse, os, sys, json, time, re
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from loss_registry import _scan_code_safety, _sanitize_name
from trainer_fingan import train_single_ticker_fingan
import importlib.util


# ── System Prompt (v2: JSON output + thought field) ──────

LOSS_SYSTEM_V2 = """You design loss functions for FinGAN (GAN for probabilistic return forecasting).

Generator: LSTM + Linear, outputs predicted return (scalar).
Discriminator: LSTM + Linear + Sigmoid.
Training: Generator loss = BCE(discriminator) + scale * financial_loss
  where scale is auto-computed via GradientCheck (gradient norm balancing).

Financial loss interface:
```python
def loss_fn(gen_out, real, tanh_temp):
    # gen_out: (B,) generated returns
    # real: (B,) actual returns
    # tanh_temp: float (100.0)
    # Returns: scalar Tensor (added to BCE with gradient-balanced weight)
```

Known effective terms:
- PnL: -mean(tanh(T*gen_out) * real)  -- maximize trading profit
- MSE: mean((gen_out - real)^2)  -- prediction accuracy
- Sharpe: -mean(pnl) / std(pnl)  -- risk-adjusted return
- Sortino: -mean(pnl) / sqrt(mean(relu(-pnl)^2))  -- downside risk
- Winrate: mean(relu(margin - gen_out*real))  -- directional accuracy hinge

Rules:
1. Return scalar Tensor, differentiable
2. Can use torch.*, math.*
3. Function name MUST be loss_fn
4. Can combine existing terms OR invent new ones

When you respond, output a JSON object with exactly three keys:
1. "thought": Your analysis of previous results and reasoning for the new design (2-4 sentences). Be specific about which mechanisms worked/failed and why.
2. "name": snake_case name for the loss function
3. "code": The exact Python code defining loss_fn (as a single string, use \\n for newlines)

You are deeply familiar with financial loss design from the quantitative finance literature. Be creative and reference prior literature when possible. The user will return evaluation results after each proposal. Your goal is to maximize the Sharpe Ratio."""


# ── LLM Client (v2: conversation mode + JSON attempt) ────

def _llm_call_conversation(messages, max_tokens=2000):
    """LLM call with full conversation history. Tries JSON mode first."""
    from openai import OpenAI
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    kwargs = dict(model=config.LLM_MODEL, max_tokens=max_tokens, messages=messages)

    # Try with JSON response format (OpenAI-compatible providers may support it)
    try:
        kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content
    except Exception:
        pass

    # Fallback without JSON mode
    kwargs.pop("response_format", None)
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content


def _llm_call(system, user, max_tokens=2000):
    """Standalone LLM call (backward compat)."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return _llm_call_conversation(messages, max_tokens)


# ── Response Parsing ─────────────────────────────────────

def _extract_code(text):
    """Extract code from ```python blocks."""
    if "```python" in text:
        s = text.index("```python") + len("```python")
        e = text.index("```", s)
        return text[s:e].strip()
    if "```" in text:
        s = text.index("```") + 3
        e = text.index("```", s)
        return text[s:e].strip()
    return text


def _parse_response(text):
    """Parse LLM response: try JSON first, fallback to regex.

    Returns dict with keys: thought, name, code.
    """
    # 1. Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "code" in data:
            data.setdefault("name", "unnamed")
            data.setdefault("thought", "")
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Try to find JSON block embedded in markdown ```json ... ```
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

    # 3. Try to find a bare JSON object in text
    # Match outermost { ... } that contains "code"
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

    # 4. Fallback: extract code block + name via regex
    code = _extract_code(text)
    m = re.search(r'[Nn]ame[:\s]+[`"]*(\w+)[`"]*', text)
    name = m.group(1) if m else "unnamed"
    return {"thought": text[:300], "name": name, "code": code}


# ── Evaluation ───────────────────────────────────────────

def evaluate_fingan(code, name, device, n_seeds=3, tickers=None):
    """Load loss, validate, train on tickers x n_seeds seeds."""
    name = _sanitize_name(name)
    err = _scan_code_safety(code)
    if err:
        return {"loss_name": name, "mean_sr": -999, "error": f"safety: {err}", "code": code}

    path = os.path.join(config.LOSSES_DIR, f"{name}.py")
    with open(path, "w") as f:
        f.write(code)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.loss_fn
    except Exception as e:
        return {"loss_name": name, "mean_sr": -999, "error": f"load: {e}", "code": code}

    # Validate
    try:
        t_out = torch.randn(32, requires_grad=True)
        t_real = torch.randn(32)
        loss = fn(t_out, t_real, 100.0)
        loss.backward()
    except Exception as e:
        return {"loss_name": name, "mean_sr": -999, "error": f"validate: {e}", "code": code}

    # Train on tickers x n_seeds seeds
    if tickers is None:
        tickers = config.META_TRAIN_TICKERS
    seeds = list(range(42, 42 + n_seeds))
    per_ticker = {}
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
            except Exception:
                sr_per_seed.append(-999)
        valid_sr = [s for s in sr_per_seed if s > -900]
        per_ticker[ticker] = float(np.mean(valid_sr)) if valid_sr else -999

    valid = [v for v in per_ticker.values() if v > -900]
    mean_sr = float(np.mean(valid)) if valid else -999

    return {"loss_name": name, "mean_sr": mean_sr, "per_ticker": per_ticker, "code": code}


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to messages.json to resume from")
    args = parser.parse_args()

    n_seeds = args.n_seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    results_dir = os.path.join(config.RESULTS_DIR, "llm_fingan_v4")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    baseline_sr = None

    # ── Evaluate baselines using code strings ────────────
    print("=== Evaluating baselines (3-arg code strings) ===")
    from losses.fingan_baselines import FINGAN_BASELINE_CODES

    baseline_codes = {}  # name -> code string (for prompt)
    for bname, bcode in FINGAN_BASELINE_CODES.items():
        print(f"\n--- {bname} ---")
        result = evaluate_fingan(bcode, bname, device, n_seeds=n_seeds)
        result["origin"] = "baseline"
        all_results.append(result)
        baseline_codes[bname] = bcode
        print(f"  Mean SR: {result['mean_sr']:.4f}")

        if bname == "baseline_bce_only":
            baseline_sr = result["mean_sr"]

    if baseline_sr is None or baseline_sr <= 0:
        baseline_sr = max(r["mean_sr"] for r in all_results if r["mean_sr"] > -900) or 1.0

    # Leaderboard
    sorted_r = sorted(all_results, key=lambda x: -x["mean_sr"])
    print(f"\n{'='*50}")
    print(f"BASELINE LEADERBOARD (norm by BCE-only SR={baseline_sr:.4f})")
    for i, r in enumerate(sorted_r):
        norm = r["mean_sr"] / baseline_sr
        print(f"  {i+1}. {r['loss_name']:25s} SR={r['mean_sr']:.4f} (norm={norm:.3f})")

    # ── Build archive with CODE for initial prompt ───────
    archive_lines = []
    for r in sorted_r:
        norm = r["mean_sr"] / baseline_sr
        archive_lines.append(
            f"  {r['loss_name']:25s} SR={r['mean_sr']:.4f} (norm={norm:.3f}) "
            f"per_ticker={json.dumps(r.get('per_ticker', {}))}"
        )
    archive_str = "\n".join(archive_lines)

    # Top-3 baseline SOURCE CODE
    top3_code_str = "\n\n".join(
        f"### {r['loss_name']} (SR={r['mean_sr']:.4f}, norm={r['mean_sr']/baseline_sr:.3f})\n"
        f"```python\n{baseline_codes.get(r['loss_name'], 'N/A')}\n```"
        for r in sorted_r[:3]
        if r['loss_name'] in baseline_codes
    )

    # ── Initialize conversation ──────────────────────────
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            messages = json.load(f)
        start_round = len([m for m in messages if m["role"] == "assistant"]) + 1
        print(f"\nResumed from {args.resume}, starting at round {start_round}")
    else:
        messages = [{"role": "system", "content": LOSS_SYSTEM_V2}]
        first_prompt = (
            f"Here are the baseline results:\n{archive_str}\n\n"
            f"## Top-3 Baseline SOURCE CODE:\n{top3_code_str}\n\n"
            f"The BCE-only baseline (no financial loss) has SR={baseline_sr:.4f} (norm=1.000).\n"
            f"Your task: IMPROVE on the best baseline. Start from its structure and "
            f"make targeted improvements. Do not reinvent from scratch.\n\n"
            f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
        )
        messages.append({"role": "user", "content": first_prompt})
        start_round = 1

    # ── LLM Evolution with Conversation History ──────────
    print(f"\n=== LLM Evolution v2 ({args.rounds} rounds, {n_seeds} seeds) ===")

    for rnd in range(start_round, args.rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  ROUND {rnd}/{args.rounds}")
        print(f"{'#'*60}")

        # LLM proposes (sees full conversation history)
        t0 = time.time()
        response_text = _llm_call_conversation(messages, max_tokens=2000)
        llm_time = time.time() - t0
        messages.append({"role": "assistant", "content": response_text})

        # Parse response (JSON -> fallback regex)
        parsed = _parse_response(response_text)
        name = _sanitize_name(parsed.get("name", f"round_{rnd}"))
        code = parsed.get("code", "")
        thought = parsed.get("thought", "")

        print(f"  Thought: {thought[:200]}...")
        print(f"  Proposed: '{name}' (LLM {llm_time:.1f}s)")

        # Evaluate
        result = evaluate_fingan(code, name, device, n_seeds=n_seeds)
        result["origin"] = "llm"
        result["thought"] = thought
        result["code"] = code
        all_results.append(result)

        norm_sr = result["mean_sr"] / baseline_sr
        print(f"  Mean SR: {result['mean_sr']:.4f} (norm={norm_sr:.3f})")

        # Feed evaluation back into conversation
        if result["mean_sr"] > -900:
            above_below = "ABOVE" if norm_sr > 1.0 else "BELOW"
            feedback = (
                f"Evaluation result for '{name}':\n"
                f"  Mean SR: {result['mean_sr']:.4f} (normalized: {norm_sr:.3f}, {above_below} baseline)\n"
                f"  Per-ticker: {json.dumps(result.get('per_ticker', {}))}\n"
                f"\nCurrent leaderboard (top 5):\n"
            )
            top5 = sorted(all_results, key=lambda x: -x.get("mean_sr", -999))[:5]
            for i, r in enumerate(top5):
                n = r.get("mean_sr", 0) / baseline_sr
                key = r.get("loss_name", r.get("name", "?"))
                feedback += f"  {i+1}. {key:25s} SR={r.get('mean_sr',0):.4f} (norm={n:.3f})\n"
            feedback += (
                f"\nPlease analyze why this result occurred and propose a better loss function.\n"
                f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
            )
        else:
            error = result.get("error", "unknown")
            feedback = (
                f"The loss '{name}' FAILED: {error}\n"
                f"Please analyze the failure and propose a corrected loss function.\n"
                f'Respond with JSON: {{"thought": "...", "name": "...", "code": "..."}}'
            )

        messages.append({"role": "user", "content": feedback})

        # Save per-round result + conversation state
        with open(os.path.join(results_dir, f"round_{rnd:03d}_{name}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        with open(os.path.join(results_dir, "messages.json"), "w") as f:
            json.dump(messages, f, indent=2, default=str)

    # ── Train leaderboard ────────────────────────────────
    sorted_r = sorted(all_results, key=lambda x: -x.get("mean_sr", -999))
    print(f"\n{'='*60}")
    print(f"{'TRAIN LEADERBOARD':^60}")
    print(f"{'='*60}")
    for i, r in enumerate(sorted_r[:10]):
        norm = r.get("mean_sr", 0) / baseline_sr
        key = r.get("loss_name", r.get("name", "?"))
        print(f"  {i+1}. {key:25s} SR={r.get('mean_sr',0):.4f} (norm={norm:.3f}) ({r.get('origin','?')})")

    # ── Validation Phase: held-out tickers ────────────────
    print(f"\n{'='*60}")
    print(f"  VALIDATION PHASE (held-out tickers: {config.META_VAL_TICKERS})")
    print(f"{'='*60}")
    # Take top-5 unique losses (skip baselines, only LLM proposals)
    llm_only = [r for r in sorted_r if r.get("origin") == "llm" and r.get("mean_sr", -999) > -900]
    top5_llm = llm_only[:5]
    val_results = []
    for r in top5_llm:
        name = r.get("loss_name", r.get("name", "?"))
        code = r.get("code", "")
        print(f"\n  Validating: {name} (train_SR={r['mean_sr']:.4f})")
        vr = evaluate_fingan(code, f"val_{name}", device,
                             n_seeds=config.META_VAL_SEEDS,
                             tickers=config.META_VAL_TICKERS)
        val_results.append({"name": name, "train_sr": r["mean_sr"],
                            "val_sr": vr["mean_sr"], "code": code})
        print(f"  val_SR = {vr['mean_sr']:.4f} (train_SR was {r['mean_sr']:.4f})")

    val_results.sort(key=lambda x: -x.get("val_sr", -999))
    print(f"\n--- Validation Leaderboard ---")
    for i, vr in enumerate(val_results):
        val_norm = vr["val_sr"] / baseline_sr if baseline_sr > 0 else 0
        print(f"  {i+1}. val_SR={vr['val_sr']:.4f} (norm={val_norm:.3f}) "
              f"| train_SR={vr['train_sr']:.4f} | {vr['name']}")

    winner = val_results[0] if val_results else {"name": "none", "val_sr": -999, "train_sr": -999}
    print(f"\n  Winner (by val_SR): {winner['name']}")
    print(f"    val_SR={winner['val_sr']:.4f}, train_SR={winner['train_sr']:.4f}")

    with open(os.path.join(results_dir, "final_v4.json"), "w") as f:
        json.dump({
            "baseline_sr": baseline_sr,
            "n_seeds": n_seeds,
            "winner": winner,
            "validation_results": val_results,
            "train_results": [{k: v for k, v in r.items() if k != "code"}
                              for r in sorted_r[:20]],
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
