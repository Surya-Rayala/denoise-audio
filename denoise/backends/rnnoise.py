"""\
RNNoise backend using pyrnnoise.

This backend is intentionally small and package-friendly:
- Provides a config dataclass with explicit, model-specific args.
- Provides CLI integration hooks (add_cli_args / config_from_cli).
- Provides a simple denoise_file() entry point.

Notes:
- pyrnnoise RNNoise runs internally at 48 kHz / 480-sample frames.
- We pass the *input* sample rate to pyrnnoise so its wrapper can resample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import os

import numpy as np

try:
    from audiolab import Reader
except Exception as e:  # pragma: no cover
    Reader = None  # type: ignore[assignment]
    _AUDIO_READER_IMPORT_ERROR = e
else:
    _AUDIO_READER_IMPORT_ERROR = None

try:
    from pyrnnoise import RNNoise
except Exception as e:  # pragma: no cover
    RNNoise = None  # type: ignore[assignment]
    _PYRNNOISE_IMPORT_ERROR = e
else:
    _PYRNNOISE_IMPORT_ERROR = None


BACKEND_NAME = "rnnoise"
BACKEND_DESCRIPTION = (
    "RNNoise. Fast lightweight speech denoising. "
    "CPU-only. Best for real-time and low-latency use."
)


@dataclass(frozen=True)
class RNNoiseConfig:
    """Model-specific configuration for RNNoise.

    Keep every model arg explicit and namespaced (rnnoise_*), so the top-level CLI
    can expose them clearly and avoid collisions with other backends.
    """

    rnnoise_sample_rate: Optional[int] = None
    """If set, forces the RNNoise wrapper sample_rate.

    If None, we infer it from the input WAV file (recommended).
    """


def _require_deps() -> None:
    if RNNoise is None:
        raise RuntimeError(f"pyrnnoise is not available: {_PYRNNOISE_IMPORT_ERROR}")
    if Reader is None:
        raise RuntimeError(f"audiolab is not available: {_AUDIO_READER_IMPORT_ERROR}")


def infer_wav_sample_rate(path: str) -> int:
    """Infer sample rate from a WAV file using audiolab.Reader."""
    _require_deps()

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    r = Reader(path, dtype=np.int16)
    try:
        rate = int(r.rate)
    finally:
        close_fn = getattr(r, "close", None)
        if callable(close_fn):
            close_fn()

    return rate


class RNNoiseBackend:
    """Backend wrapper for RNNoise."""

    name = BACKEND_NAME
    description = BACKEND_DESCRIPTION
    config_type = RNNoiseConfig

    @staticmethod
    def add_cli_args(parser) -> None:
        """Add RNNoise-specific CLI flags."""
        parser.add_argument(
            "--rnnoise-sample-rate",
            dest="rnnoise_sample_rate",
            type=int,
            default=None,
            help="Force RNNoise wrapper sample rate. Default: infer from input WAV.",
        )

    @staticmethod
    def config_from_cli(args) -> RNNoiseConfig:
        """Build RNNoiseConfig from parsed argparse args."""
        return RNNoiseConfig(
            rnnoise_sample_rate=getattr(args, "rnnoise_sample_rate", None),
        )

    def denoise_file(
        self,
        input_wav: str,
        output_wav: str,
        config: RNNoiseConfig | None = None,
    ) -> Iterator[float]:
        """Denoise a WAV file.

        Yields:
            Per-frame speech probability values produced by pyrnnoise.
        """
        _require_deps()
        cfg = config or RNNoiseConfig()

        sample_rate = (
            infer_wav_sample_rate(input_wav)
            if cfg.rnnoise_sample_rate is None
            else int(cfg.rnnoise_sample_rate)
        )

        denoiser = RNNoise(sample_rate=sample_rate)

        # denoise_wav yields per-frame speech probabilities; iteration drives processing.
        for speech_prob in denoiser.denoise_wav(input_wav, output_wav):
            # pyrnnoise may yield a python float OR an array-like (e.g., numpy array).
            # Convert robustly to a single float.
            arr = np.asarray(speech_prob)
            if arr.size == 1:
                yield float(arr.item())
            else:
                # If a vector is returned, keep the first element as the per-frame score.
                yield float(arr.reshape(-1)[0])