"""

Package layout:
- denoise/
  - __init__.py
      Exposes the public API (denoise_file, available_models).
  - denoise.py
      CLI entrypoint (``python -m denoise.denoise ...``).
  - backends/
      Model/backends live here.
      - __init__.py
          Backend registry + helpers (list models, create backend).
      - rnnoise.py
          RNNoise backend implementation (pyrnnoise).

"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Dict, Iterator

from .backends import available_models as available_models
from .backends import backend_help as backend_help
from .backends import create_backend
from .backends import get_backend_class


__all__ = [
    "__version__",
    "available_models",
    "backend_help",
    "model_description",
    "model_kwargs_help",
    "denoise_file",
]

# Keep this simple; you can replace with dynamic versioning later.
__version__ = "0.1.0"


def model_description(model: str) -> str:
    """Return the one-line description for a backend/model."""
    cls = get_backend_class(model)
    return str(getattr(cls, "description", ""))


def model_kwargs_help(model: str) -> Dict[str, Dict[str, Any]]:
    """Return allowed kwargs for `denoise_file(..., model=..., **kwargs)`.

    The kwargs correspond to fields on the backend's config dataclass.

    Returns:
        A mapping of field name -> {"type": <string>, "default": <value>}.
    """
    cls = get_backend_class(model)
    cfg = getattr(cls, "config_type", None)

    if cfg is None or not is_dataclass(cfg):
        return {}

    out: Dict[str, Dict[str, Any]] = {}

    for f in fields(cfg):
        # Determine a readable default.
        if f.default is not MISSING:
            default = f.default
        elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
            default = "<factory>"
        else:
            default = None

        # Best-effort readable type string.
        t = f.type
        type_str = getattr(t, "__name__", None) or str(t)

        out[f.name] = {"type": type_str, "default": default}

    return out


def denoise_file(
    input_wav: str,
    output_wav: str,
    model: str = "rnnoise",
    **model_kwargs: Any,
) -> Iterator[float]:
    """Denoise a WAV file.

    Args:
        input_wav: Path to input WAV.
        output_wav: Path to output WAV.
        model: Backend/model name (see available_models()).
        **model_kwargs: Model-specific kwargs. These map to fields on the backend's
            config dataclass.

    Returns:
        Iterator of optional per-frame metadata produced by the backend.
        For rnnoise, this yields per-frame speech probabilities.

    Example:
        >>> from denoise import denoise_file
        >>> list(denoise_file("in.wav", "out.wav", model="rnnoise"))
    """
    backend = create_backend(model)

    # Build a config object if the backend advertises a dataclass config type.
    config_type = getattr(backend, "config_type", None)
    config = None

    if config_type is not None and model_kwargs:
        # Fail fast on unknown keys.
        try:
            config = config_type(**model_kwargs)
        except TypeError as e:
            raise TypeError(
                f"Invalid arguments for model '{model}': {e}. "
                "Pass only model-specific fields (e.g., rnnoise_sample_rate=...)."
            ) from e

    # Backend returns an iterator; caller can iterate to completion.
    return backend.denoise_file(input_wav, output_wav, config=config)