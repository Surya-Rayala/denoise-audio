"""Backends for the denoise package.

A backend is a small wrapper around a denoising implementation (model).

Conventions:
- Each backend module exposes a Backend class (e.g., RNNoiseBackend) with:
  - name: str
  - description: str
  - add_cli_args(parser) -> None
  - config_from_cli(args) -> Config
  - denoise_file(input_wav, output_wav, config) -> Iterator[float] (optional yields)

This module provides a tiny registry so the CLI can:
- list available models
- select a model by name
"""

from __future__ import annotations

from typing import Dict, List, Type

# Import concrete backends here.
from .rnnoise import RNNoiseBackend
from .deepfilternet import DeepFilterNetBackend
from .fbdenoiser import FBDenoiserBackend


# Registry of model name -> backend class
_BACKENDS: Dict[str, Type] = {
    RNNoiseBackend.name: RNNoiseBackend,
    DeepFilterNetBackend.name: DeepFilterNetBackend,
    FBDenoiserBackend.name: FBDenoiserBackend,
}


def available_models() -> List[str]:
    """Return available model/backend names."""
    return sorted(_BACKENDS.keys())


def get_backend_class(name: str):
    """Get the backend class by model name."""
    try:
        return _BACKENDS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown model '{name}'. Available models: {', '.join(available_models())}"
        ) from e


def create_backend(name: str):
    """Instantiate the backend by model name."""
    cls = get_backend_class(name)
    return cls()


def backend_help() -> str:
    """Human-readable help for available backends."""
    lines: List[str] = []
    for model in available_models():
        cls = _BACKENDS[model]
        desc = getattr(cls, "description", "")
        if desc:
            lines.append(f"  - {model}: {desc}")
        else:
            lines.append(f"  - {model}")
    return "\n".join(lines)
