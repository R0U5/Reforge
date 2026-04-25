# Reforge

A single-file LoRA fine-tuning pipeline for Hugging Face causal language models and vision-language models. Handles dataset loading, automatic schema detection, aggressive text cleaning, tokenization, training with smart early stopping, LoRA merge, and final model export — all from one script.

## Features

- **Automatic training mode detection** — inspects dataset columns and selects the right pipeline: SFT (instruction/response), causal (raw text), chat (messages), or multimodal (image+text).
- **Built-in data cleaning** — strips LaTeX, HTML, markdown artifacts, unicode garbage, emoji, junk tokens, and e-commerce noise. Configurable modes for math, LaTeX-preserving, and code-preserving workflows.
- **Model profiles** — ships with tuned configs for Phi-2, Phi-3, Phi-3-Vision, Llama 3, Mistral, Qwen, and Gemma. Falls back to sensible defaults for unlisted models.
- **Smart early stopping** — EMA-based loss plateau/worsening detection with exposure floors, LR gates, and dynamic hard caps. Prevents both underfitting and wasted compute.
- **Automatic sequence length** — measures token length distribution (P95) and sets `max_length` accordingly. No manual tuning needed.
- **Iterative training** — trains on a dataset, merges the LoRA adapter into the base, and uses the merged model as the starting point for the next dataset. Tracks training history in `training_chain.txt`.
- **Checkpoint resumption** — interrupted runs resume from the last checkpoint automatically.
- **Cross-platform file picker** — native file dialogs on Windows (PowerShell), macOS (osascript), and Linux (zenity/kdialog), with text input fallback.

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on 8–24 GB VRAM)
- Dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
# Basic SFT training on a local parquet file (opens file picker)
python reforge.py

# Specify a HuggingFace Hub dataset directly
python reforge.py --hf_dataset username/dataset-name

# Override the base model path
python reforge.py --base_model /path/to/your/model

# Preserve code formatting during cleaning
python reforge.py --code

# Preserve LaTeX equations
python reforge.py --latex

# Multimodal image+text training
python reforge.py --image --base_model /path/to/vision-model

# Train for exactly 500 steps
python reforge.py --steps 500

# Disable early stopping and run full epochs
python reforge.py --force --epoch 3
```

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--base_model PATH` | Path to a local Hugging Face model directory |
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
| `HF_MODEL_DIR` | `~/hf_models` | Root directory for downloaded HF models |
| `TRAINER_OUTPUT` | `~/trainer_output` | Root directory for checkpoints and merged output |
| `HF_ROOT` | `~/.cache/huggingface` | Hugging Face cache root |
| `HF_TMP` | System temp dir | Temporary file directory |

## How It Works

1. **Load** — reads a `.parquet` file or HF Hub dataset, shuffles it, and auto-detects the training schema from column names.
2. **Clean** — applies mode-appropriate text normalization (math symbol conversion, markup stripping, junk token filtering).
3. **Tokenize** — measures P95 token length, sets `max_length`, and tokenizes with dynamic padding.
4. **Train** — configures LoRA adapters (rank, alpha, target modules per model profile), warms up CUDA kernels, and trains with the Hugging Face `Trainer`.
5. **Stop** — early stopping monitors EMA loss for plateaus and worsening trends. A dynamic hard cap prevents runaway training.
6. **Merge** — after training, the LoRA adapter is merged back into the base model and saved alongside the tokenizer/processor.
7. **Chain** — the merged model becomes the base for the next training run, enabling iterative refinement across datasets.

## Adding a New Model Profile

Edit the `MODEL_PROFILES` dictionary in `reforge.py`. The key is a lowercase substring matched against the model directory name (first match wins):

```python
"your-model": {
    "attn_implementation":   "sdpa",       # or "eager", "flash_attention_2"
    "trust_remote_code":     False,
    "use_fast_tokenizer":    True,
    "inject_special_tokens": False,
    "resize_embeddings":     False,
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

## License

MIT
