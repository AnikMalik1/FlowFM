"""
Loss Registry — stores and manages loss functions (baselines + evolved).
"""
import os
import re
import importlib.util
from typing import Callable, Optional

from losses.fingan_baselines import FINGAN_BASELINES
import config

# Imports that LLM-generated code is NOT allowed to use
BANNED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "http", "urllib", "requests", "ctypes", "signal",
    "multiprocessing", "threading", "pickle", "shelve",
    "importlib", "builtins", "code", "codeop", "compile",
}
BANNED_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\(",
    r"\b__import__\s*\(",
    r"\bgetattr\s*\(",
    r"\bglobals\s*\(",
    r"\bcompile\s*\(",
]


def _sanitize_name(name: str) -> str:
    """Allow only alphanumeric + underscore in loss names."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", name)
    if not cleaned:
        cleaned = "unnamed_loss"
    return cleaned


def _scan_code_safety(code: str) -> Optional[str]:
    """Static scan for dangerous patterns before execution. Returns error or None."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Check imports
        if stripped.startswith(("import ", "from ")):
            for banned in BANNED_IMPORTS:
                if re.search(rf"\b{banned}\b", stripped):
                    return f"Banned import detected: '{banned}' in '{stripped}'"
        # Check dangerous builtins
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, stripped):
                return f"Banned pattern detected: '{pattern}' in '{stripped}'"
    return None


class LossRegistry:
    """In-memory registry of loss functions with persistence."""

    def __init__(self):
        self._fns: dict[str, Callable] = {}
        self._meta: dict[str, dict] = {}
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
        # C1 fix: sanitize name (prevent path traversal)
        name = _sanitize_name(name)

        # C1 fix: static safety scan before any execution
        safety_err = _scan_code_safety(code)
        if safety_err:
            return f"Safety scan failed for '{name}': {safety_err}"

        # Save to file
        path = os.path.join(config.LOSSES_DIR, f"{name}.py")
        with open(path, "w") as f:
            f.write(code)

        # Load with importlib
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

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                return f"loss_fn returned {loss.item()}"

            loss.backward()
            return None

        except Exception as e:
            return f"Validation failed: {e}"
