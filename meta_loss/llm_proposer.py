"""
LLM Proposer — uses OpenAI-compatible API to propose new loss functions.

v2: Includes top-K source code in prompt + diversity pressure.
"""
import os
import json
from typing import Optional

import config


SYSTEM_PROMPT = """You are a loss function designer for financial time series probabilistic forecasting.

## Model
Conditional Flow Matching model that learns a velocity field v(x_t, condition, t).
- Base training loss: MSE(v_pred, v_target) where v_target = x1 - x0
- x0 ~ N(0,1) is noise, x1 is the real return (normalized)
- x_t = (1-t)*x0 + t*x1 is the noisy interpolation
- x_pred = x_t + (1-t)*v_pred is the first-order extrapolation to t=1
- At inference: ODE integration from t=0 to t=1 generates samples

## Loss Function Interface
```python
def loss_fn(v_pred, v_target, x_pred, x_real, condition, epoch):
    # v_pred:    (B,1) predicted velocity
    # v_target:  (B,1) true velocity = x1 - x0
    # x_pred:    (B,1) first-order extrapolation = x_t + (1-t)*v_pred
    # x_real:    (B,1) true normalized return
    # condition: (B,l) lookback window of past normalized returns
    # epoch:     int, current training epoch
    # Returns: scalar Tensor (differentiable)
```

## Known Loss Terms (from FinGAN paper)
- velocity_loss = MSE(v_pred, v_target)  — MUST be included as base term
- differentiable_pnl = mean(tanh(100*x_pred) * x_real)  — smooth sign * return
- mse_term = MSE(x_pred, x_real)  — prediction accuracy
- sharpe_term = mean(pnl) / std(pnl)  — mini-batch Sharpe
- std_term = std(tanh(100*x_pred) * x_real)  — PnL volatility
- sortino_term = mean(pnl) / sqrt(mean(relu(-pnl)^2))  — downside risk

## Rules
1. MUST include velocity_loss = F.mse_loss(v_pred, v_target) as base term
2. MUST return a scalar Tensor
3. MUST be differentiable (no .item(), no numpy, no detach on main path)
4. Can use: torch.*, torch.nn.functional.*, math.*
5. Can create new terms not in the list above
6. Can use epoch for curriculum/scheduling
7. Can use condition tensor for context-aware losses
8. Function name MUST be loss_fn

## Output Format
First explain your reasoning (2-3 sentences), then output ONLY the Python code block:
```python
import torch
import torch.nn.functional as F

def loss_fn(v_pred, v_target, x_pred, x_real, condition, epoch):
    ...
    return total_loss
```"""


def build_history_table(results: list[dict]) -> str:
    """Format evolution history as a table for the LLM prompt."""
    if not results:
        return "No previous results yet."

    sorted_r = sorted(results, key=lambda x: -x.get("mean_sr", -999))

    lines = ["| Rank | Loss Name | Mean SR | Stage | Description |",
             "|------|-----------|---------|-------|-------------|"]

    for i, r in enumerate(sorted_r[:15]):
        name = r.get("loss_name", "?")
        sr = r.get("mean_sr", 0)
        stage = r.get("stage", "?")
        desc = r.get("description", "")[:60]
        lines.append(f"| {i+1} | {name} | {sr:.4f} | S{stage} | {desc} |")

    return "\n".join(lines)


def build_top_k_code(results: list[dict], k: int = 3) -> str:
    """Include source code of top-K losses so LLM can learn from winners."""
    sorted_r = sorted(results, key=lambda x: -x.get("mean_sr", -999))
    blocks = []

    for r in sorted_r[:k]:
        name = r.get("loss_name", "?")
        sr = r.get("mean_sr", 0)
        code = r.get("code", "")

        # If no code in results, try reading from file
        if not code:
            path = os.path.join(config.LOSSES_DIR, f"{name}.py")
            if os.path.exists(path):
                with open(path) as f:
                    code = f.read()

        if code:
            blocks.append(f"### {name} (SR={sr:.4f})\n```python\n{code}\n```")

    return "\n\n".join(blocks) if blocks else "No source code available."


def build_failed_patterns(results: list[dict]) -> str:
    """Summarize patterns that didn't work, so LLM avoids them."""
    sorted_r = sorted(results, key=lambda x: x.get("mean_sr", -999))
    failures = [r for r in sorted_r if r.get("mean_sr", 0) < 0.9 and r.get("origin") == "llm"]

    if not failures:
        return ""

    lines = ["\n## Patterns That Did NOT Work (avoid these):"]
    for r in failures[:5]:
        name = r.get("loss_name", "?")
        sr = r.get("mean_sr", 0)
        reasoning = r.get("reasoning", "")[:100]
        lines.append(f"- **{name}** (SR={sr:.3f}): {reasoning}")

    return "\n".join(lines)


def propose_loss(history: list[dict], round_num: int) -> dict:
    """
    Call OpenAI-compatible API to propose a new loss function.

    Returns: {
        "name": str,           # suggested name
        "code": str,           # Python source code
        "reasoning": str,      # LLM's explanation
        "description": str,    # one-line description
    }
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
    )

    history_table = build_history_table(history)
    top_k_code = build_top_k_code(history, k=3)
    failed_patterns = build_failed_patterns(history)

    # Count how many evolved losses exist
    n_evolved = sum(1 for r in history if r.get("origin") == "llm")

    # Diversity pressure: after 3 failed rounds, explicitly demand novelty
    diversity_hint = ""
    if n_evolved >= 3:
        recent_names = [r.get("loss_name", "") for r in history if r.get("origin") == "llm"][-5:]
        diversity_hint = f"""
## IMPORTANT: Diversity Required
You have already proposed {n_evolved} losses. Recent attempts: {', '.join(recent_names)}.
These all follow similar patterns (curriculum + sortino variants).
You MUST try a fundamentally different approach. Consider:
- Using the `condition` tensor (lookback returns) for context-aware weighting
- Contrastive losses (positive vs negative return separation)
- Quantile-based losses instead of mean-based
- Volatility regime detection from condition
- Asymmetric loss weights based on predicted confidence
- Huber/quantile regression on returns instead of MSE
- Direct optimization of win rate or hit ratio
- Momentum-aware losses using condition autocorrelation
Do NOT propose another curriculum_sortino variant."""

    user_msg = f"""## Evolution Round {round_num}

## Previous Results (sorted by mean per-ticker Sharpe Ratio):
{history_table}

## Source Code of Top-3 Losses (learn from what works):
{top_k_code}
{failed_patterns}
{diversity_hint}

## Task
Propose a NEW loss function that improves mean per-ticker Sharpe Ratio.
- Analyze what worked and what didn't in previous rounds
- Try something meaningfully different from existing attempts
- Give the function a descriptive name

Respond with:
1. Name (snake_case, e.g. "pnl_with_momentum_penalty")
2. One-line description
3. Reasoning (2-3 sentences)
4. Python code block with the loss_fn function"""

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    text = response.choices[0].message.content

    # Parse response
    code = _extract_code_block(text)
    name = _extract_name(text)
    reasoning = _extract_reasoning(text, code)

    return {
        "name": name,
        "code": code,
        "reasoning": reasoning,
        "description": name.replace("_", " "),
        "raw_response": text,
    }


def _extract_code_block(text: str) -> str:
    """Extract Python code from markdown code block."""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    return text


def _extract_name(text: str) -> str:
    """Extract loss function name from response."""
    import re
    m = re.search(r'[Nn]ame[:\s]+[`"]*(\w+)[`"]*', text)
    if m:
        return m.group(1)
    return f"evolved_{hash(text) % 10000:04d}"


def _extract_reasoning(text: str, code: str) -> str:
    """Extract reasoning (everything before the code block)."""
    if "```" in text:
        return text[:text.index("```")].strip()
    return text[:200]
