

#!/usr/bin/env python3
"""CLI for the denoise package.

Examples:
  # Default model (rnnoise)
  python -m denoise.denoise input.wav output.wav

  # Pick a model explicitly
  python -m denoise.denoise --model rnnoise --input in.wav --output out.wav

  # List models
  python -m denoise.denoise --list-models

  # RNNoise-specific options (only valid when --model rnnoise)
  python -m denoise.denoise --model rnnoise --rnnoise-sample-rate 44100 --input in.wav --output out.wav

  # DeepFilterNet options (only valid when --model deepfilternet)
  python -m denoise.denoise --model deepfilternet --df-pf --input in.wav --output out.wav
"""

from __future__ import annotations

import argparse
import os
import sys

from . import denoise_file
from .backends import available_models, backend_help, create_backend, get_backend_class




def _build_base_parser() -> argparse.ArgumentParser:
    """Parser that only knows global args + positional IO."""
    p = argparse.ArgumentParser(
        prog="denoise",
        description="Denoise a WAV file using selectable backends/models.",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,  # <-- IMPORTANT: disable argparse auto-help
        epilog=(
            "Model-specific help:\n"
            "  python -m denoise --model <name> --help\n"
            "Examples:\n"
            "  python -m denoise --help\n"
            "  python -m denoise --model fbdenoiser --help\n"
        ),
    )

    # Manual help flag (so we can defer help printing until after model is known)
    p.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show this help message and exit. Use with --model for model-specific options.",
    )

    p.add_argument(
        "--model",
        default="rnnoise",
        help="Model/backend name. Use --list-models to see options.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List available models/backends and exit.",
    )

    # Input/output (preferred: flags, fallback: positional)
    p.add_argument(
        "--input",
        dest="input",
        help="Path to input WAV file",
    )

    p.add_argument(
        "--output",
        dest="output",
        help="Path to output WAV file",
    )
    return p


def _build_model_parser(model: str) -> argparse.ArgumentParser:
    """Full parser including only the selected backend's options."""

    p = _build_base_parser()

    # Attach options for the selected backend only.
    try:
        backend_cls = get_backend_class(model)
    except KeyError:
        # Let argparse display the unknown model error later in main.
        return p

    add_cli = getattr(backend_cls, "add_cli_args", None)
    if callable(add_cli):
        group = p.add_argument_group(f"{model} options")
        add_cli(group)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    # First pass: parse only global args (without exiting on --help)
    base_parser = _build_base_parser()
    base_args, _unknown = base_parser.parse_known_args(argv)

   # Was --model explicitly provided by user?
    model_was_set = "--model" in argv

    # Build appropriate parser
    if model_was_set:
        parser = _build_model_parser(str(base_args.model))
    else:
        # Only base parser (no backend args)
        parser = _build_base_parser()

    # If user asked for help, print correct help
    if getattr(base_args, "help", False):
        print(parser.format_help())
        return 0

    # Normal parse
    args = parser.parse_args(argv)

    if args.list_models:
        print("Available models:")
        print(backend_help() or "  (none)")
        return 0

    # Resolve input/output: flags take priority over positional
    in_path = args.input or args.input_pos
    out_path = args.output or args.output_pos

    if not in_path or not out_path:
        parser.error(
            "input and output are required. "
            "Use --input <file> --output <file>."
        )

    if not os.path.exists(in_path):
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 2

    model = str(args.model)

    try:
        backend = create_backend(model)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        print("\nUse --list-models to see available options.", file=sys.stderr)
        return 2

    cfg = None
    cfg_from_cli = getattr(backend, "config_from_cli", None)
    if callable(cfg_from_cli):
        cfg = cfg_from_cli(args)

    try:
        for _meta in backend.denoise_file(in_path, out_path, config=cfg):
            pass
    except Exception as e:
        print(f"Error: denoising failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())