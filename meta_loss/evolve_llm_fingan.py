#!/usr/bin/env python3
"""
LLM-Proposal Evolution with Reflection — FinGAN model.

Each round:
1. Reflect on best + worst loss (why it worked/failed)
2. Propose new loss based on reflection
3. Evaluate on FinGAN
4. Update history

This is proper meta-learning: propose -> evaluate -> REFLECT -> propose better.
"""
import argparse, os, sys, json, time
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from loss_registry import LossRegistry, _scan_code_safety, _sanitize_name
from trainer_fingan import train_single_ticker_fingan
import importlib.util


LOSS_SYSTEM = """You design loss functions for FinGAN (GAN for probabilistic return forecasting).

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

Rules:
1. Return scalar Tensor, differentiable
2. Can use torch.*, math.*
3. Function name MUST be loss_fn
"""


def _llm_call(system, user, max_tokens=2000):
    from openai import OpenAI
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    r = client.chat.completions.create(
        model=config.LLM_MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content


def _extract_code(text):
    if "```python" in text:
        s = text.index("```python") + len("```python")
        e = text.index("```", s)
        return text[s:e].strip()
    if "```" in text:
        s = text.index("```") + 3
        e = text.index("```", s)
        return text[s:e].strip()
    return text


def evaluate_fingan(code, name, device):
    """Load loss, validate with FinGAN interface, train on 3 tickers."""
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

    # Train on 3 tickers
    per_ticker = {}
    for ticker in config.STAGE1_TICKERS:
        try:
            r = train_single_ticker_fingan(
                ticker=ticker, financial_loss_fn=fn,
                max_epochs=100, warmup_epochs=15,
                eval_every=20, patience=6,
                device=device, verbose=False)
            per_ticker[ticker] = r["val_sr"]
        except Exception as e:
            per_ticker[ticker] = -999

    valid = [v for v in per_ticker.values() if v > -900]
    mean_sr = float(np.mean(valid)) if valid else -999

    return {"loss_name": name, "mean_sr": mean_sr, "per_ticker": per_ticker, "code": code}


# ── Reflection ───────────────────────────────────────────

def reflect(best_result, worst_result):
    """LLM reflects on best vs worst loss: WHY did one work and other fail?"""
    prompt = f"""Analyze these two loss functions and their results.

## Best Loss: {best_result['loss_name']} (SR={best_result['mean_sr']:.4f})
Per-ticker: {json.dumps(best_result.get('per_ticker', {}))}
```python
{best_result.get('code', 'N/A')}
```

## Worst Loss: {worst_result['loss_name']} (SR={worst_result['mean_sr']:.4f})
Per-ticker: {json.dumps(worst_result.get('per_ticker', {}))}
```python
{worst_result.get('code', 'N/A')}
```

## Task
1. WHY does the best loss work better? Identify the specific mechanism.
2. WHY does the worst loss fail? What goes wrong?
3. What pattern should the next loss exploit?

Be specific about which loss TERMS and WEIGHTS matter."""

    return _llm_call(LOSS_SYSTEM, prompt, max_tokens=1000)


def propose_with_reflection(all_results, reflection, round_num):
    """Propose a new loss based on reflection insights."""
    sorted_r = sorted(all_results, key=lambda x: -x.get("mean_sr", -999))

    history = "\n".join(
        f"  {i+1}. {r['loss_name']:25s} SR={r.get('mean_sr',0):.4f}"
        for i, r in enumerate(sorted_r[:10])
    )

    # Include top-3 code
    top_code = "\n\n".join(
        f"### {r['loss_name']} (SR={r.get('mean_sr',0):.4f})\n```python\n{r.get('code','N/A')}\n```"
        for r in sorted_r[:3] if r.get("code")
    )

    prompt = f"""## Evolution Round {round_num}

## Leaderboard (top 10):
{history}

## Top-3 Source Code:
{top_code}

## Reflection from Previous Round:
{reflection}

## Task
Based on the reflection above, propose a NEW loss function that:
1. Exploits the patterns identified in the reflection
2. Addresses the weaknesses found
3. Is meaningfully different from existing losses

Output: name (snake_case), reasoning (2 sentences), then Python code block."""

    text = _llm_call(LOSS_SYSTEM, prompt)
    code = _extract_code(text)

    import re
    m = re.search(r'[Nn]ame[:\s]+[`"]*(\w+)[`"]*', text)
    name = m.group(1) if m else f"round_{round_num}"

    return {"name": name, "code": code, "reasoning": text[:300]}


# ── Main ─────────────────────────────────────────────────

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

    all_results = []

    # Evaluate baselines
    print("=== Evaluating baselines ===")
    registry = LossRegistry()
    from losses.fingan_baselines import FINGAN_BASELINES

    for bname, bfn in FINGAN_BASELINES.items():
        print(f"\n--- {bname} ---")
        # Wrap baseline to FinGAN interface
        def make_fingan_fn(base_fn):
            def fingan_fn(gen_out, real, tanh_temp):
                x_pred = gen_out.unsqueeze(-1) if gen_out.dim() == 1 else gen_out
                x_real = real.unsqueeze(-1) if real.dim() == 1 else real
                v_dummy = torch.zeros_like(x_pred)
                cond_dummy = torch.zeros(x_pred.shape[0], 10, device=x_pred.device)
                return base_fn(v_dummy, v_dummy, x_pred, x_real, cond_dummy, epoch=100)
            return fingan_fn

        fingan_fn = make_fingan_fn(bfn)
        per_ticker = {}
        for ticker in config.STAGE1_TICKERS:
            r = train_single_ticker_fingan(
                ticker=ticker, financial_loss_fn=fingan_fn,
                max_epochs=100, warmup_epochs=15,
                eval_every=20, patience=6,
                device=device, verbose=True)
            per_ticker[ticker] = r["val_sr"]

        mean_sr = float(np.mean(list(per_ticker.values())))
        result = {"loss_name": bname, "mean_sr": mean_sr, "per_ticker": per_ticker, "origin": "baseline"}
        all_results.append(result)
        print(f"  Mean SR: {mean_sr:.4f}")

    # Leaderboard
    sorted_r = sorted(all_results, key=lambda x: -x["mean_sr"])
    print(f"\n{'='*50}")
    print("BASELINE LEADERBOARD")
    for i, r in enumerate(sorted_r):
        print(f"  {i+1}. {r['loss_name']:25s} SR={r['mean_sr']:.4f}")

    # LLM evolution with reflection
    print(f"\n=== LLM Evolution with Reflection ({args.rounds} rounds) ===")
    reflection = "No reflection yet. This is the first round."

    for rnd in range(1, args.rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  ROUND {rnd}/{args.rounds}")
        print(f"{'#'*60}")

        # Step 1: Reflect on best vs worst
        valid = [r for r in all_results if r.get("mean_sr", -999) > -900]
        if len(valid) >= 2:
            best = max(valid, key=lambda x: x["mean_sr"])
            worst = min(valid, key=lambda x: x["mean_sr"])
            print(f"  Reflecting: best={best['loss_name']}({best['mean_sr']:.3f}) vs worst={worst['loss_name']}({worst['mean_sr']:.3f})")
            reflection = reflect(best, worst)
            print(f"  Reflection: {reflection[:200]}...")

        # Step 2: Propose based on reflection
        proposal = propose_with_reflection(all_results, reflection, rnd)
        name = proposal["name"]
        code = proposal["code"]
        print(f"  Proposed: '{name}'")

        # Step 3: Evaluate
        result = evaluate_fingan(code, name, device)
        result["origin"] = "llm"
        result["reasoning"] = proposal["reasoning"]
        result["code"] = code
        all_results.append(result)
        print(f"  Mean SR: {result['mean_sr']:.4f}")

        # Save
        with open(os.path.join(results_dir, f"round_{rnd:03d}_{name}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Final leaderboard
    sorted_r = sorted(all_results, key=lambda x: -x.get("mean_sr", -999))
    print(f"\n{'='*60}")
    print(f"{'LLM+REFLECTION FINAL':^60}")
    print(f"{'='*60}")
    for i, r in enumerate(sorted_r[:20]):
        print(f"  {i+1}. {r['loss_name']:25s} SR={r.get('mean_sr',0):.4f} ({r.get('origin','?')})")

    with open(os.path.join(results_dir, "final.json"), "w") as f:
        json.dump(sorted_r[:50], f, indent=2, default=str)


if __name__ == "__main__":
    main()
