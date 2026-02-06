
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

from typing import Any, Dict, Iterator, Optional

from .backends import available_models as available_models
from .backends import create_backend


__all__ = [
    "__version__",
    "available_models",
    "denoise_file",
]

# Keep this simple; you can replace with dynamic versioning later.
__version__ = "0.1.0"


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