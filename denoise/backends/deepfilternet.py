"""\
DeepFilterNet backend (PyPI: deepfilternet).

This backend uses the Python API:

    from df.enhance import enhance, init_df, load_audio, save_audio

to load a pretrained model and enhance an input WAV.

Notes:
- DeepFilterNet is typically trained/optimized for 48kHz full-band audio.
- The helper `load_audio(..., sr=df_state.sr())` is used to resample on input.
- Output is written using `save_audio(..., sr=df_state.sr())`.

Install (CPU example):
    pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
    pip install deepfilternet
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import os

import numpy as np


# Optional dependency: deepfilternet
try:
    # deepfilternet exposes the `df` package.
    from df.enhance import enhance, init_df, load_audio, save_audio
except Exception as e:  # pragma: no cover
    enhance = None  # type: ignore[assignment]
    init_df = None  # type: ignore[assignment]
    load_audio = None  # type: ignore[assignment]
    save_audio = None  # type: ignore[assignment]
    _DF_IMPORT_ERROR = e
else:
    _DF_IMPORT_ERROR = None


BACKEND_NAME = "deepfilternet"
BACKEND_DESCRIPTION = (
    "DeepFilterNet. High-quality full-band audio denoising. "
    "Supports CPU/GPU. Best for offline batch processing."
)



@dataclass(frozen=True)
class DeepFilterNetConfig:
    """Model-specific configuration for DeepFilterNet.

    All args are explicit and namespaced (df_*), so they stay backend-specific.
    """

    df_model: Optional[str] = None
    """Model name or model base dir.

    If None, DeepFilterNet's default pretrained model is loaded.
    """

    df_pf: bool = False
    """Enable DeepFilterNet post-filter (slightly stronger attenuation in very noisy sections)."""

    df_compensate_delay: bool = False
    """Compensate delay introduced by the real-time STFT/ISTFT implementation (padding)."""


def _require_deps() -> None:
    if init_df is None or enhance is None or load_audio is None or save_audio is None:
        raise RuntimeError(
            "DeepFilterNet is not available. Install it with: pip install deepfilternet "
            f"(import error: {_DF_IMPORT_ERROR})"
        )


class DeepFilterNetBackend:
    """Backend wrapper for DeepFilterNet."""

    name = BACKEND_NAME
    description = BACKEND_DESCRIPTION
    config_type = DeepFilterNetConfig

    @staticmethod
    def add_cli_args(parser) -> None:
        # Keep only the commonly-used knobs from upstream CLI.
        parser.add_argument(
            "--df-model",
            dest="df_model",
            default=None,
            help=(
                "Select which DeepFilterNet pretrained model to load. "
                "You can pass a model name (e.g., DeepFilterNet, DeepFilterNet2, DeepFilterNet3) "
                "or a full path to a model base directory containing checkpoints/config. "
                "If omitted, DeepFilterNet's default pretrained model is used."
            ),
        )

        parser.add_argument(
            "--df-pf",
            dest="df_pf",
            action="store_true",
            help=(
                "Enable the DeepFilterNet post-filter. This can reduce residual noise, "
                "but may sound slightly more aggressive in very noisy sections."
            ),
        )

        parser.add_argument(
            "--df-compensate-delay",
            dest="df_compensate_delay",
            action="store_true",
            help=(
                "Add padding to compensate for the processing delay introduced by STFT/ISTFT and model lookahead. "
                "Useful if you need better alignment with the original audio."
            ),
        )

    @staticmethod
    def config_from_cli(args) -> DeepFilterNetConfig:
        return DeepFilterNetConfig(
            df_model=getattr(args, "df_model", None),
            df_pf=bool(getattr(args, "df_pf", False)),
            df_compensate_delay=bool(getattr(args, "df_compensate_delay", False)),
        )

    def denoise_file(
        self,
        input_wav: str,
        output_wav: str,
        config: DeepFilterNetConfig | None = None,
    ) -> Iterator[float]:
        """Enhance/denoise a WAV file with DeepFilterNet.

        Yields:
            A single float (1.0) upon successful completion.
            (DeepFilterNet's Python API does not expose per-frame probs like RNNoise.)
        """
        _require_deps()
        cfg = config or DeepFilterNetConfig()

        if not os.path.exists(input_wav):
            raise FileNotFoundError(input_wav)

        # Load model
        # Upstream: init_df() returns (model, df_state, suffix).
        # If a model name/base-dir is provided, pass it through.
        if cfg.df_model:
            model, df_state, _suffix = init_df(model_base_dir=cfg.df_model)
        else:
            model, df_state, _suffix = init_df()

        # df_state.sr() tells us which sampling rate the model expects.
        target_sr = int(df_state.sr())

        # Load audio and resample to target sr.
        audio, _sr = load_audio(input_wav, sr=target_sr)

        # Ensure we have a float32 numpy array (DeepFilterNet expects float audio).
        audio = np.asarray(audio, dtype=np.float32)

        # Some DeepFilterNet versions expect torch.Tensors (and will call torch.nn.functional
        # ops like pad()). Convert numpy -> torch here.
        try:
            import torch
        except Exception as e:
            raise RuntimeError(
                "DeepFilterNet requires torch to run. Ensure torch is installed in your environment. "
                f"(import error: {e})"
            ) from e

        audio_t = torch.as_tensor(audio, dtype=torch.float32)
        # Ensure shape is [T] or [C, T]. If [T, C] (rare), transpose.
        if audio_t.ndim == 2 and audio_t.shape[0] < audio_t.shape[1]:
            # Assume [C, T] already.
            pass
        elif audio_t.ndim == 2 and audio_t.shape[1] < audio_t.shape[0]:
            # Likely [T, C] -> [C, T]
            audio_t = audio_t.transpose(0, 1)

        # Enhance
        # DeepFilterNet's Python API has changed across versions. Some versions
        # accept flags like `pf` / `compensate_delay`, others do not. We adapt by
        # inspecting the enhance() signature and only passing supported kwargs.
        import inspect

        kwargs = {}
        try:
            params = set(inspect.signature(enhance).parameters.keys())
        except Exception:
            params = set()

        # Post-filter flag (varies by version)
        if cfg.df_pf:
            if "pf" in params:
                kwargs["pf"] = True
            elif "post_filter" in params:
                kwargs["post_filter"] = True
            elif "postfilter" in params:
                kwargs["postfilter"] = True
            elif "use_pf" in params:
                kwargs["use_pf"] = True

        # Delay compensation flag (varies by version)
        if cfg.df_compensate_delay:
            if "compensate_delay" in params:
                kwargs["compensate_delay"] = True
            elif "compensate" in params:
                kwargs["compensate"] = True
            elif "compensate_latency" in params:
                kwargs["compensate_latency"] = True
            elif "compensate_lookahead" in params:
                kwargs["compensate_lookahead"] = True

        try:
            enhanced = enhance(model, df_state, audio_t, **kwargs)
        except TypeError:
            # Last-resort fallback: call without extra flags.
            enhanced = enhance(model, df_state, audio_t)

        # Convert output to numpy float32 for saving.
        if hasattr(enhanced, "detach"):
            enhanced = enhanced.detach().cpu().numpy()
        enhanced = np.asarray(enhanced, dtype=np.float32)

        # Save at target sr
        save_audio(output_wav, enhanced, target_sr)

        yield 1.0