"""
LLM Proposer — uses OpenAI-compatible API to propose new loss functions.
"""
import json
from typing import Optional

import config


SYSTEM_PROMPT = """You are a loss function designer for financial time series probabilistic forecasting.

## Model
Conditional Flow Matching model that learns a velocity field v(x_t, condition, t).
- Base training loss: MSE(v_pred, v_target) where v_target = x1 - x0
- x0 ~ N(0,1) is noise, x1 is the real return (normalized)
- x_t = (1-t)*x0 + t*x1 is the noisy interpolation
- At inference: ODE integration from t=0 to t=1 generates samples

## Loss Function Interface
```python
def loss_fn(v_pred, v_target, x_pred, x_real, condition, epoch):
    # v_pred:    (B,1) predicted velocity
    # v_target:  (B,1) true velocity = x1 - x0
    # x_pred:    (B,1) one-step approx of generated sample = x0 + v_pred
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

## Rules
1. MUST include velocity_loss = F.mse_loss(v_pred, v_target) as base term
2. MUST return a scalar Tensor
3. MUST be differentiable (no .item(), no numpy, no detach on main path)
4. Can use: torch.*, torch.nn.functional.*, math.*
5. Can create new terms not in the list above
6. Can use epoch for curriculum/scheduling
7. Function name MUST be loss_fn

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

    for i, r in enumerate(sorted_r[:15]):  # top 15
        name = r.get("loss_name", "?")
        sr = r.get("mean_sr", 0)
        stage = r.get("stage", "?")
        desc = r.get("description", "")[:60]
        lines.append(f"| {i+1} | {name} | {sr:.4f} | S{stage} | {desc} |")

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

    user_msg = f"""## Evolution Round {round_num}

## Previous Results (sorted by mean per-ticker Sharpe Ratio):
{history_table}

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
    # Look for "Name: xxx" or "name: xxx"
    m = re.search(r'[Nn]ame[:\s]+[`"]*(\w+)[`"]*', text)
    if m:
        return m.group(1)
    # Look for "def loss_fn" — use a generic name
    return f"evolved_{hash(text) % 10000:04d}"


def _extract_reasoning(text: str, code: str) -> str:
    """Extract reasoning (everything before the code block)."""
    if "```" in text:
        return text[:text.index("```")].strip()
    return text[:200]
