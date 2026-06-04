# Reforge

A single-file LoRA fine-tuning pipeline for Hugging Face causal language models and vision-language models. Reforge handles dataset loading, automatic schema detection, text cleaning, tokenization, training with smart early stopping, LoRA merge, and final model export — all from one script, with a live-updating Tkinter dashboard for interactive control.

## Features

### Core
- **Automatic training mode detection** — inspects dataset columns and selects the right pipeline: SFT (instruction/response), causal (raw text), chat (messages), or multimodal (image+text).
- **Built-in data cleaning** — strips LaTeX, HTML, markdown artifacts, unicode garbage, emoji, junk tokens, and e-commerce noise. Configurable modes for math, LaTeX-preserving, and code-preserving workflows.
- **Model profiles** — ships with tuned configs for Phi-2, Phi-3, Phi-3-Vision, Llama 3, Mistral, Qwen, and Gemma. Falls back to sensible defaults for unlisted models.
- **Smart early stopping** — EMA-based loss plateau/worsening detection with exposure floors, LR gates, and dynamic hard caps. Prevents both underfitting and wasted compute.
- **Automatic sequence length** — measures token length distribution (P95) and sets `max_length` accordingly. No manual tuning needed.
- **Iterative training** — trains on a dataset, merges the LoRA adapter into the base, and uses the merged model as the starting point for the next dataset. Tracks training history in `training_chain.txt`.
- **Checkpoint resumption** — interrupted runs resume from the last checkpoint automatically.
- **Cross-platform file picker** — native file dialogs on Windows (PowerShell), macOS (osascript), and Linux (zenity/kdialog), with text input fallback.

### Interactive
- **Local model picker** — scans `D:\HF_Models\`, `~/HF_Models/`, and the Hugging Face hub cache, then presents a numbered list of every local model with its detected architecture and on-disk size. Type a number to pick, or paste a path. Filters to vision-capable models automatically when `--image` is set.
- **Post-training summary** — prints a formatted report of steps, throughput, loss statistics (first/final/best/Δ), trend direction, and a unicode sparkline of the loss curve. Writes a full `training_report.json` next to the merged model.
- **Snazzy CLI output** — ANSI color-coded banners, status bullets, and aligned key/value tables. Auto-detects terminal capability, enables Windows VT processing, falls back to plain text for redirected output. Honors `NO_COLOR` and `FORCE_COLOR` environment variables.
- **Tkinter dashboard** (`reforge_dashboard.py`) — single-window UI with config editor, start/stop, and a live loss chart updated in real-time from trainer stdout.

### Robustness
- **Local-files-only loading** — once a base model is downloaded, no further network calls are made. Train and resume fully offline.
- **CUDA kernel warmup** — runs a single dry forward pass before timed training to prime the allocator and kernel cache, eliminating the 70s→4s staircase slowdown.
- **Emergency save** — on training error, the current LoRA state is written to disk before merge so partial progress is never lost.
- **Resource cleanup** — `roc()` callback logs and reclaims CUDA memory between checkpoints.

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on 8–24 GB VRAM)
- Dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
# Basic SFT training on a local parquet file (opens file picker, then model picker)
python reforge.py

# Specify a Hugging Face Hub dataset directly
python reforge.py --hf_dataset username/dataset-name

# Override the base model path
python reforge.py --base_model /path/to/your/model

# Preserve code formatting during cleaning
python reforge.py --code

# Preserve LaTeX equations
python reforge.py --latex

# Multimodal image+text training (filters picker to vision models)
python reforge.py --image

# Train for exactly 500 steps
python reforge.py --steps 500

# Disable early stopping and run full epochs
python reforge.py --force --epoch 3

# Launch the GUI dashboard
python reforge_dashboard.py
```

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--base_model PATH` | Path to a local Hugging Face model directory (skips picker) |
| `--output_dir PATH` | Override the output/checkpoint directory |
| `--hf_dataset ID` | Load a dataset from the Hugging Face Hub |
| `--image` | Enable multimodal (vision+text) training |
| `--latex` | Preserve LaTeX/scientific notation during cleaning |
| `--code` | Preserve code formatting and indentation |
| `--epoch N` | Number of training epochs (default: 1) |
| `--steps N` | Train for exactly N steps (overrides epoch) |
| `--force` | Disable early stopping |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | Root for the Hugging Face model cache |
| `HF_HOME` | `~/.cache/huggingface` | Root for all HF cache types |
| `NO_COLOR` | unset | If set, disables all ANSI color output |
| `FORCE_COLOR` | unset | If set, forces color output even when stdout is not a TTY |
| `TERM=dumb` | unset | Disables color output (standard convention) |

## How It Works

1. **Load** — reads a `.parquet` file or HF Hub dataset, shuffles it, and auto-detects the training schema from column names.
2. **Clean** — applies mode-appropriate text normalization (math symbol conversion, markup stripping, junk token filtering).
3. **Tokenize** — measures P95 token length, sets `max_length`, and tokenizes with dynamic padding.
4. **Train** — configures LoRA adapters (rank, alpha, target modules per model profile), warms up CUDA kernels, and trains with the Hugging Face `Trainer`.
5. **Stop** — early stopping monitors EMA loss for plateaus and worsening trends. A dynamic hard cap prevents runaway training.
6. **Report** — prints a summary banner with loss curve, throughput, and trend; writes `training_report.json` alongside the merged model.
7. **Merge** — after training, the LoRA adapter is merged back into the base model and saved alongside the tokenizer/processor.
8. **Chain** — the merged model becomes the base for the next training run, enabling iterative refinement across datasets.

## Output

After a successful run you'll find:

```
OUTPUT_DIR/
├── merged/                        # the final, merged model
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer files
│   ├── training_report.json       # full training metrics
│   └── success.txt                # marker file
├── checkpoint-N/                  # intermediate checkpoints (kept: 2)
└── checkpoint-last/               # symlink (or copy) to the latest
```

The `training_report.json` includes per-step loss, throughput, eval entries, and trend analysis.

## Adding a New Model Profile

Edit the `MODEL_PROFILES` dictionary in `reforge.py`. The key is a lowercase substring matched against the model directory name (first match wins, so put more-specific keys before less-specific ones):

```python
"your-model": {
    "attn_implementation":   "sdpa",       # or "eager", "flash_attention_2"
    "trust_remote_code":     False,
    "use_fast_tokenizer":    True,
    "inject_special_tokens": False,
    "resize_embeddings":     False,
    "vision":                False,        # True if model accepts image inputs
    "lora_r":                16,
    "lora_alpha":            32,
    "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
    "modules_to_save":       None,
    "grad_accum":            8,
    "learning_rate":         2e-5,
    "safe_serialization":    True,
    "enable_input_grads":    False,
},
```

If your model isn't listed, the fallback `MODEL_PROFILE_DEFAULT` is used automatically.

## Troubleshooting

**`HFValidationError: Repo id must be in the form 'repo_name'...`**
The base model path you supplied doesn't exist on disk. Use the interactive picker (run with no args) to see what's actually available locally.

**First few training steps are much slower than the rest**
Should not happen with the built-in CUDA warmup. If you disabled it, expect a 70s→4s staircase on the first ~10 steps as kernels JIT.

**Out of memory during training**
Reduce `max_length` (Reforge auto-sizes it from P95, but you can lower it manually), or use a smaller LoRA rank in the model profile.

**Windows console shows `?` instead of `═` and `─`**
Reforge reconfigures stdout to UTF-8 at startup. If you still see this, your console is forcing a legacy code page; run `chcp 65001` before launching, or set `PYTHONIOENCODING=utf-8`.

## License

MIT
