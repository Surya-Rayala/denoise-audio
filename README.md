# denoise-audio

A small CLI tool for denoising **WAV** audio with selectable backends:

- **rnnoise** (fast, CPU-only, low-latency)
- **deepfilternet** (high-quality full-band denoising; offline/batch friendly)
- **fbdenoiser** (FacebookResearch Denoiser / causal Demucs; strong enhancement, GPU recommended)

> Inputs/outputs are WAV files. If your audio is not WAV, convert it first.

---

## Install `uv`

Install `uv` using Astral’s standalone installer:

https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

```bash
uv --version
```

---

## Install dependencies

Clone the repo and install dependencies from the lockfile:

git clone <YOUR_REPO_URL>
cd denoise-audio
uv sync

---

## CLI usage

Run everything with uv run to ensure you’re using the project environment:

Global help

uv run python -m denoise --help

List available backends

uv run python -m denoise --list-models

Model-specific help

Pass --model <name> and --help to see only that backend’s options:

uv run python -m denoise --model rnnoise --help
uv run python -m denoise --model deepfilternet --help
uv run python -m denoise --model fbdenoiser --help

---

## Basic command (all models)

uv run python -m denoise --model <model> --input <in.wav> --output <out.wav>

	•	<model> is one of: rnnoise, deepfilternet, fbdenoiser
	•	--input and --output must be WAV files

---

## Models

1) RNNoise (--model rnnoise)

Basic command

uv run python -m denoise --model rnnoise --input <in.wav> --output <out.wav>

Arguments
	•	--rnnoise-sample-rate <int>
Force the RNNoise wrapper sample rate.
If omitted, the tool infers the sample rate from the input WAV (recommended).

---

2) DeepFilterNet (--model deepfilternet)

Basic command

uv run python -m denoise --model deepfilternet --input <in.wav> --output <out.wav>

Arguments
	•	--df-model <name>
Select which pretrained DeepFilterNet model to load. Common options:
	•	DeepFilterNet
	•	DeepFilterNet2
	•	DeepFilterNet3
To see available models, visit:
https://github.com/Rikorose/DeepFilterNet/tree/main/models
	•	--df-pf
Enable the post-filter. This can reduce residual noise, but may sound slightly more aggressive in very noisy sections.
	•	--df-compensate-delay
Add padding to compensate processing delay introduced by STFT/ISTFT and model lookahead. Useful when you need better alignment with the original audio.

---

3) FacebookResearch Denoiser (--model fbdenoiser)

Basic command

uv run python -m denoise --model fbdenoiser --input <in.wav> --output <out.wav>

Arguments
	•	--fb-model <name>
Choose the pretrained model:
	•	dns48 — pre-trained real time H=48 model trained on DNS
	•	dns64 — pre-trained real time H=64 model trained on DNS
	•	master64 — pre-trained real time H=64 model trained on DNS and Valentini
	•	--fb-device <device>
Inference device, e.g. cpu or cuda.
If omitted, it automatically uses CUDA if available, otherwise CPU.
	•	--fb-dry <float>
Dry/wet mix:
	•	0.0 = original input only
	•	1.0 = denoised output only
Values outside [0.0, 1.0] are clamped.
	•	--fb-streaming
Enable streaming mode (flag is accepted by the CLI).
	•	--fb-batch-size <int>
Batch size (flag is accepted by the CLI).
	•	--fb-num-workers <int>
Number of workers (flag is accepted by the CLI).
	•	--fb-verbose
Enable verbose logging (flag is accepted by the CLI).

---

## Updating from Git

If you’re using the repo locally with `uv sync` + `uv run`, updating to new code from Git is:

```bash
cd denoise-audio
git pull
uv sync
```

### If you have local changes and `git pull` refuses

Use one of these (pick what you intend):

- Keep your local changes and rebase on top:

```bash
git pull --rebase
uv sync
```

- Discard local changes and reset to remote (⚠️ destructive):

```bash
git fetch origin
git reset --hard origin/main
uv sync
```
