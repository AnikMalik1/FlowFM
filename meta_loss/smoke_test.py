#!/usr/bin/env python3
"""
Smoke test: train 1 ticker (AMZN) with velocity_only loss for 5 epochs.
Validates the entire pipeline end-to-end on GPU.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FlowFM_repo"))

import torch
import config

# Override for smoke test
config.VAL_MC_SAMPLES = 32
config.VAL_MC_CHUNK = 8

from losses.fingan_baselines import velocity_only, fingan_pnl_mse
from trainer import train_single_ticker
from llm_proposer import propose_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Test velocity_only
print("\n=== Test 1: velocity_only on AMZN (5 epochs) ===")
r1 = train_single_ticker("AMZN", velocity_only, max_epochs=5, eval_every=5,
                          patience=1, device=device, verbose=True)
print(f"Result: {r1}")

# 2. Test fingan_pnl_mse (trading-aware loss)
print("\n=== Test 2: fingan_pnl_mse on AMZN (5 epochs) ===")
r2 = train_single_ticker("AMZN", fingan_pnl_mse, max_epochs=5, eval_every=5,
                          patience=1, device=device, verbose=True)
print(f"Result: {r2}")

# 3. Test LLM proposal (if API available)
print("\n=== Test 3: LLM proposal ===")
try:
    proposal = propose_loss([], round_num=0)
    print(f"Name: {proposal['name']}")
    print(f"Code preview: {proposal['code'][:200]}...")
    print("LLM API: OK")
except Exception as e:
    print(f"LLM API: FAILED ({e})")

print("\n=== Smoke test complete ===")
