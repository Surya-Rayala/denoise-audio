# denoise-audio

A CLI tool for denoising **WAV** audio with 3 leading backends:

- **rnnoise** (fast, CPU-only, low-latency)
- **deepfilternet** (high-quality full-band denoising)
- **fbdenoiser** (FacebookResearch Denoiser / causal Demucs; strong enhancement)

> Inputs/outputs are WAV files. If your audio is not WAV, convert it first.

---

## Install with `pip` (PyPI)

> Use this if you want to install and use the tool without cloning the repo.

### Install

```bash
pip install denoise-audio
```

### CLI usage (pip)

Global help:

```bash
python -m denoise --help
```

List available backends:

```bash
python -m denoise --list-models
```

Model-specific help:

```bash
python -m denoise --model rnnoise --help
python -m denoise --model deepfilternet --help
python -m denoise --model fbdenoiser --help
```

Basic command (all models):

```bash
python -m denoise --model <model> --input <in.wav> --output <out.wav>
```

- `<model>` is one of: `rnnoise`, `deepfilternet`, `fbdenoiser`
- `--input` and `--output` must be WAV files

---

## Python usage (import)

You can also use this package directly in your Python code after installing with `pip install denoise-audio`.

### Quick sanity check

```bash
python -c "import denoise; print(denoise.__version__)"
```

### List available models/backends

```python
from denoise import available_models, backend_help

print(available_models())
print(backend_help())
```

### Inspect supported keyword arguments for each model

```python
from denoise import model_kwargs_help

print(model_kwargs_help("rnnoise"))
print(model_kwargs_help("deepfilternet"))
print(model_kwargs_help("fbdenoiser"))
```

### Run denoising from a Python file

Create `run_denoise.py`:

```python
from denoise import denoise_file

IN_WAV = "input.wav"
OUT_WAV = "output.wav"

# Choose one: rnnoise | deepfilternet | fbdenoiser
MODEL = "rnnoise"

# Model-specific kwargs (examples below)
kwargs = {}

# Example: RNNoise
# kwargs = {"rnnoise_sample_rate": 48000}

# Example: DeepFilterNet
# kwargs = {"df_model": "DeepFilterNet3", "df_pf": True, "df_compensate_delay": True}

# Example: FacebookResearch Denoiser
# kwargs = {"fb_model": "dns64", "fb_device": "cpu", "fb_dry": 1.0}

for _ in denoise_file(IN_WAV, OUT_WAV, model=MODEL, **kwargs):
    pass

print(f"Wrote: {OUT_WAV}")
```

Run it:

```bash
python run_denoise.py
```

---

## Install from GitHub (uv)

Install `uv` using Astral’s standalone installer:

https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

```bash
uv --version
```

---

## Install dependencies

```bash
git clone https://github.com/Surya-Rayala/denoise-audio.git
cd denoise-audio
uv sync
```

---

## CLI usage

Run everything with uv run to ensure you’re using the project environment:

### Global help

```bash
uv run python -m denoise --help
```

### List available backends

```bash
uv run python -m denoise --list-models
```

### Model-specific help

Pass `--model <name>` and `--help` to see only that backend’s options:

```bash
uv run python -m denoise --model rnnoise --help
uv run python -m denoise --model deepfilternet --help
uv run python -m denoise --model fbdenoiser --help
```

---

## Basic command (all models)

```bash
uv run python -m denoise --model <model> --input <in.wav> --output <out.wav>
```

- `<model>` is one of: `rnnoise`, `deepfilternet`, `fbdenoiser`
- `--input` and `--output` must be WAV files

---

## Models

### RNNoise (`--model rnnoise`)

**Basic command**

```bash
uv run python -m denoise --model rnnoise --input <in.wav> --output <out.wav>
```

**Arguments**
- `--rnnoise-sample-rate <int>`: Force the RNNoise wrapper sample rate. If omitted, the tool infers the sample rate from the input WAV (recommended).

---

### DeepFilterNet (`--model deepfilternet`)

**Basic command**

```bash
uv run python -m denoise --model deepfilternet --input <in.wav> --output <out.wav>
```

**Arguments**
- `--df-model <name>`: Select which pretrained DeepFilterNet model to load. Common options: `DeepFilterNet`, `DeepFilterNet2`, `DeepFilterNet3`.
  - To see available models, visit: https://github.com/Rikorose/DeepFilterNet/tree/main/models
- `--df-pf`: Enable the post-filter (can reduce residual noise; may sound more aggressive in very noisy sections).
- `--df-compensate-delay`: Add padding to compensate processing delay (useful when you need better alignment with the original audio).

---

### FacebookResearch Denoiser (`--model fbdenoiser`)

**Basic command**

```bash
uv run python -m denoise --model fbdenoiser --input <in.wav> --output <out.wav>
```

**Arguments**
- `--fb-model <name>`: Choose the pretrained model:
  - `dns48`: pre-trained real time H=48 model trained on DNS
  - `dns64`: pre-trained real time H=64 model trained on DNS
  - `master64`: pre-trained real time H=64 model trained on DNS and Valentini
- `--fb-device <device>`: Inference device, e.g. `cpu` or `cuda`. If omitted, it automatically uses CUDA if available, otherwise CPU.
- `--fb-dry <float>`: Dry/wet mix (`0.0` = original input only, `1.0` = denoised output only). Values outside `[0.0, 1.0]` are clamped.
- `--fb-streaming`: Enable streaming mode (flag is accepted by the CLI).
- `--fb-batch-size <int>`: Batch size (flag is accepted by the CLI).
- `--fb-num-workers <int>`: Number of workers (flag is accepted by the CLI).
- `--fb-verbose`: Enable verbose logging (flag is accepted by the CLI).

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

## License
This project's source code is licensed under the MIT License.

**Note on Dependencies:** This tool relies on the `denoiser` library (Facebook Research), which is licensed under **CC-BY-NC 4.0** (Non-Commercial). Consequently, this tool as a whole is suitable for research and personal use only, unless you obtain a commercial license for the `denoiser` dependency.