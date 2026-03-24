"""
Loss Registry — stores and manages loss functions (baselines + evolved).
"""
import os
import json
import importlib.util
from typing import Callable, Optional

from losses.fingan_baselines import FINGAN_BASELINES
import config


class LossRegistry:
    """In-memory registry of loss functions with persistence."""

    def __init__(self):
        self._fns: dict[str, Callable] = {}
        self._meta: dict[str, dict] = {}  # name -> {source, description, origin}
        self._load_baselines()

    def _load_baselines(self):
        for name, fn in FINGAN_BASELINES.items():
            self.register(
                name=name,
                fn=fn,
                source=fn.__doc__ or "",
                description=fn.__doc__ or name,
                origin="fingan_baseline",
            )

    def register(self, name: str, fn: Callable, source: str = "",
                 description: str = "", origin: str = "manual"):
        self._fns[name] = fn
        self._meta[name] = {
            "source": source,
            "description": description,
            "origin": origin,
        }

    def get(self, name: str) -> Callable:
        return self._fns[name]

    def list_names(self) -> list[str]:
        return list(self._fns.keys())

    def get_meta(self, name: str) -> dict:
        return self._meta.get(name, {})

    def register_from_code(self, name: str, code: str,
                           description: str = "", origin: str = "llm") -> Optional[str]:
        """
        Register a loss function from Python source code string.
        Returns None on success, error message on failure.
        """
        # Save to file
        path = os.path.join(config.LOSSES_DIR, f"{name}.py")
        with open(path, "w") as f:
            f.write(code)

        # Try to load it
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if not hasattr(mod, "loss_fn"):
                return f"Module {name} must define a 'loss_fn' function"

            fn = mod.loss_fn
            self.register(name=name, fn=fn, source=code,
                          description=description, origin=origin)
            return None

        except Exception as e:
            return f"Failed to load {name}: {e}"

    def validate_loss_fn(self, name: str) -> Optional[str]:
        """Smoke test: run loss_fn with dummy tensors, check output."""
        import torch
        fn = self._fns.get(name)
        if fn is None:
            return f"Loss '{name}' not found"

        try:
            B = 32
            v_pred = torch.randn(B, 1, requires_grad=True)
            v_target = torch.randn(B, 1)
            x_pred = torch.randn(B, 1, requires_grad=True)
            x_real = torch.randn(B, 1)
            condition = torch.randn(B, 10)

            loss = fn(v_pred, v_target, x_pred, x_real, condition, epoch=1)

            if not isinstance(loss, torch.Tensor):
                return f"loss_fn must return Tensor, got {type(loss)}"
            if loss.dim() != 0:
                return f"loss_fn must return scalar, got shape {loss.shape}"
            if not loss.requires_grad:
                return "loss_fn output is not differentiable"

            loss.backward()
            return None

        except Exception as e:
            return f"Validation failed: {e}"
