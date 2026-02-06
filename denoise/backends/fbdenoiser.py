"""\
FacebookResearch Denoiser backend (PyPI: denoiser).

This backend runs the upstream enhancement in-process (avoids CLI + torchaudio compatibility issues).

Why in-process?
- The denoiser project provides a documented enhance entrypoint and pretrained models.
- In-process lets us apply a small torchaudio compatibility shim for older/newer API differences.

Install:
    pip install denoiser

Examples:
    # Use pretrained model
    python -m denoise.denoise --model fbdenoiser --fb-model dns64 in.wav out.wav

    # Use a local trained model checkpoint (best.th)
    python -m denoise.denoise --model fbdenoiser --fb-model /path/to/best.th in.wav out.wav

Notes:
- Upstream expects WAV input. If you pass non-wav, convert first.
- Upstream enhance writes to an output directory; we place the result at output_wav.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import os
import shutil
import inspect
import torch
import torchaudio


# Optional dependency: denoiser
try:
    import denoiser  # noqa: F401
except Exception as e:  # pragma: no cover
    _DENOISER_IMPORT_ERROR = e
else:
    _DENOISER_IMPORT_ERROR = None


BACKEND_NAME = "fbdenoiser"
BACKEND_DESCRIPTION = (
    "Facebook Denoiser (Demucs-based). High-quality speech enhancement. "
    "GPU recommended. Best for noisy recordings."
)



@dataclass(frozen=True)
class FBDenoiserConfig:
    """Model-specific configuration for facebookresearch/denoiser.

    Args are explicit and namespaced (fb_*).
    """

    fb_model: str = "dns64"
    """Which model to use.

    Accepts:
    - "dns48" | "dns64" | "master64" for pretrained models.
    - A path to a local trained model checkpoint file (best.th).
    """

    fb_device: Optional[str] = None
    """Device for inference (e.g., "cpu", "cuda"). If None, let upstream decide."""

    fb_dry: Optional[float] = None
    """Dry/wet knob coefficient. 0 = only input, 1 = only denoised. If None, upstream default."""

    fb_streaming: bool = False
    """Enable streaming mode (true streaming evaluation for Demucs)."""

    fb_batch_size: Optional[int] = None
    """Batch size for processing (upstream flag: --batch_size). If None, upstream default."""

    fb_num_workers: Optional[int] = None
    """Number of workers (upstream flag: --num_workers). If None, upstream default."""

    fb_verbose: bool = False
    """Enable verbose logging (upstream flag: -v/--verbose)."""


def _require_deps() -> None:
    if _DENOISER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "facebookresearch/denoiser is not available. Install it with: pip install denoiser "
            f"(import error: {_DENOISER_IMPORT_ERROR})"
        )


def _is_pretrained_name(name: str) -> bool:
    return name in {"dns48", "dns64", "master64"}


def _ensure_torchaudio_offset_compat() -> None:
    """Ensure torchaudio.load accepts the `offset=` kwarg.

    facebookresearch/denoiser historically calls `torchaudio.load(..., offset=..., num_frames=...)`.
    Some torchaudio builds use `frame_offset` instead.

    We patch `torchaudio.load` in-process to accept `offset` and forward it to
    `frame_offset` when needed.
    """
    import torchaudio

    try:
        sig = inspect.signature(torchaudio.load)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    # Already compatible.
    if "offset" in params:
        return

    orig_load = torchaudio.load

    def load_compat(*args, **kwargs):
        # Map legacy `offset` -> `frame_offset`
        if "offset" in kwargs and "frame_offset" not in kwargs:
            kwargs["frame_offset"] = kwargs.pop("offset")

        # Coerce common scalar-like types (0-d torch tensors / numpy scalars) to int
        for k in ("frame_offset", "offset", "num_frames"):
            if k in kwargs and kwargs[k] is not None:
                v = kwargs[k]
                try:
                    # torch tensors
                    if hasattr(v, "item"):
                        v = v.item()
                    # numpy scalars
                    if hasattr(v, "dtype") and not isinstance(v, (int, float, bool, str, bytes)):
                        v = v.item()
                    # final int coercion for offsets/frames
                    if k in ("frame_offset", "offset", "num_frames"):
                        v = int(v)
                    kwargs[k] = v
                except Exception:
                    # If coercion fails, pass through unchanged.
                    pass

        return orig_load(*args, **kwargs)

    torchaudio.load = load_compat  # type: ignore[assignment]


class FBDenoiserBackend:
    """Backend wrapper for facebookresearch/denoiser."""

    name = BACKEND_NAME
    description = BACKEND_DESCRIPTION
    config_type = FBDenoiserConfig

    @staticmethod
    def add_cli_args(parser) -> None:
        parser.add_argument(
            "--fb-model",
            dest="fb_model",
            default="dns64",
            help=(
                "Model selection for facebookresearch/denoiser. "
                "Use one of: dns48, dns64, master64 (pretrained), or a path to a local best.th checkpoint."
            ),
        )
        parser.add_argument(
            "--fb-device",
            dest="fb_device",
            default=None,
            help="Device for inference (e.g., cpu or cuda). Default: upstream default.",
        )
        parser.add_argument(
            "--fb-dry",
            dest="fb_dry",
            type=float,
            default=None,
            help="Dry/wet mix: 0=input only, 1=denoised only. Default: upstream default.",
        )
        parser.add_argument(
            "--fb-streaming",
            dest="fb_streaming",
            action="store_true",
            help="Enable streaming mode (upstream: --streaming).",
        )
        parser.add_argument(
            "--fb-batch-size",
            dest="fb_batch_size",
            type=int,
            default=None,
            help="Batch size (upstream: --batch_size). Default: upstream default.",
        )
        parser.add_argument(
            "--fb-num-workers",
            dest="fb_num_workers",
            type=int,
            default=None,
            help="Num workers (upstream: --num_workers). Default: upstream default.",
        )
        parser.add_argument(
            "--fb-verbose",
            dest="fb_verbose",
            action="store_true",
            help="Verbose logging (upstream: -v/--verbose).",
        )

    @staticmethod
    def config_from_cli(args) -> FBDenoiserConfig:
        return FBDenoiserConfig(
            fb_model=str(getattr(args, "fb_model", "dns64")),
            fb_device=getattr(args, "fb_device", None),
            fb_dry=getattr(args, "fb_dry", None),
            fb_streaming=bool(getattr(args, "fb_streaming", False)),
            fb_batch_size=getattr(args, "fb_batch_size", None),
            fb_num_workers=getattr(args, "fb_num_workers", None),
            fb_verbose=bool(getattr(args, "fb_verbose", False)),
        )

    def denoise_file(
        self,
        input_wav: str,
        output_wav: str,
        config: FBDenoiserConfig | None = None,
    ) -> Iterator[float]:
        _require_deps()
        cfg = config or FBDenoiserConfig()

        in_path = Path(input_wav)
        out_path = Path(output_wav)

        if not in_path.exists():
            raise FileNotFoundError(str(in_path))

        # Load full file (no offset-based streaming reads).
        wav, sr = torchaudio.load(str(in_path))
        wav = wav.contiguous()
        orig_channels = wav.shape[0]

        # Pick device.
        device = None
        if cfg.fb_device:
            device = torch.device(cfg.fb_device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model.
        from denoiser import pretrained
        from denoiser.dsp import convert_audio

        model_sel = cfg.fb_model
        if _is_pretrained_name(model_sel):
            get_model = getattr(pretrained, model_sel, None)
            if not callable(get_model):
                raise RuntimeError(f"denoiser.pretrained has no callable '{model_sel}'")
            model = get_model()
        else:
            # Best-effort: support local checkpoint paths for advanced use.
            # The official CLI supports --model_path; however internal APIs vary.
            # We attempt to deserialize via torch.load and expect a dict with a 'model' entry.
            ckpt = torch.load(model_sel, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                model = ckpt["model"]
            else:
                raise RuntimeError(
                    "Unsupported --fb-model path format. "
                    "Use pretrained names (dns48/dns64/master64), or provide a checkpoint compatible with denoiser's internal format."
                )

        model = model.to(device)
        model.eval()

        # Convert audio to model expected rate/channels.
        wav_m = convert_audio(wav, sr, model.sample_rate, model.chin)
        wav_m = wav_m.to(device)

        with torch.no_grad():
            # Model expects batch dimension.
            den = model(wav_m[None])[0]

        # Optional dry/wet mix at model sample rate.
        dry = cfg.fb_dry
        if dry is not None:
            dry_f = float(dry)
            dry_f = 0.0 if dry_f < 0.0 else (1.0 if dry_f > 1.0 else dry_f)
            den = (1.0 - dry_f) * wav_m + dry_f * den

        # Move to CPU.
        den = den.detach().cpu()

        # Resample back to original sample rate if needed.
        if model.sample_rate != sr:
            den = torchaudio.functional.resample(den, model.sample_rate, sr)

        # Match original channel count.
        if orig_channels > 1 and den.shape[0] == 1:
            den = den.repeat(orig_channels, 1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), den, sr)

        yield 1.0
