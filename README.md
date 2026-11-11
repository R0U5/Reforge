# Reforge

A modular LoRA fine-tuning pipeline for Hugging Face causal language models and vision-language models. Reforge handles dataset loading, automatic schema detection, text cleaning, tokenization, training with smart early stopping, LoRA merge, and final model export. Includes a Tkinter dashboard for interactive control.

## Project Structure

```
src/reforge/
├── __init__.py          # Package marker
├── __main__.py          # CLI entry point
├── config.py            # Environment setup, paths, constants
├── profiles.py          # Model profiles (Phi-2, Phi-3, etc.)
├── display.py           # ANSI colors, banners, formatting
├── dataset.py           # Dataset loading, schema detection, cleaning
├── tokenization.py      # Tokenization, collation, length measurement
├── training.py          # LoRA setup, model loading, training loop
├── early_stopping.py    # EMA-based early stopping callback
├── reporting.py         # Training summary, loss curves, reports
├── agent.py             # Agent mode: NDJSON events + run_state.json
├── utils.py             # Model scanner, file picker, GC
└── dashboard.py           # Tkinter GUI — `reforge dashboard` (the only launch path)
```

`AGENTS.md` (repo root) is the orientation guide for AI agents working on or driving Reforge.

## Quick Start

```bash
# Run as a module
python -m reforge

# Or install and use the console script
pip install -e .
reforge

# Install with dashboard support (matplotlib)
pip install -e ".[dashboard]"

# Launch the GUI dashboard
reforge dashboard

# Specify a Hugging Face Hub dataset
python -m reforge --hf_dataset username/dataset-name

# Override the base model path
python -m reforge --base_model /path/to/your/model

# Preserve code formatting during cleaning
python -m reforge --code

# Preserve LaTeX equations
python -m reforge --latex

# Multimodal image+text training
python -m reforge --image

# Train for exactly 500 steps
python -m reforge --steps 500

# Disable early stopping and run full epochs
python -m reforge --force --epoch 3

# Agent mode: machine-readable NDJSON events on stdout + run_state.json
python -m reforge dataset.parquet --base_model /path/to/model --agent
```

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--base_model PATH` | Path to a local Hugging Face model directory (skips picker) |
| `--hf_dataset ID` | Load a dataset from the Hugging Face Hub |
| `--image` | Enable multimodal (vision+text) training |
| `--latex` | Preserve LaTeX/scientific notation during cleaning |
| `--code` | Preserve code formatting and indentation |
| `--epoch N` | Number of training epochs (default: 1) |
| `--steps N` | Train for exactly N steps (overrides epoch) |
| `--batch_size N` | Per-device batch size override (default: 1) |
| `--grad_accum N` | Gradient accumulation steps override (default: model profile) |
| `--force` | Disable early stopping |
| `--agent` | Agent mode: NDJSON events on stdout + pollable `run_state.json` |
| `--agent_schema` | Print the machine-readable agent contract as JSON and exit 0 (also: `reforge agent-schema`) |
| `--dry_run` | Preflight: validate args/dataset/model and print the resolved training plan as JSON (no weights loaded) |
| `--agent_log PATH` | Also tee the agent NDJSON stream to a file |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `~/.cache/huggingface` | Root for all HF cache types |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | Root for the Hugging Face model cache |
| `REFORGE_MODELS_DIR` | `~/HF_Models` | Directory to scan for local models |
| `REFORGE_OUTPUT_DIR` | `~/Reforge_Output` | Default output directory |
| `NO_COLOR` | unset | Disables all ANSI color output |
| `FORCE_COLOR` | unset | Forces color output even when stdout is not a TTY |
| `TERM=dumb` | unset | Disables color output (standard convention) |
| `REFORGE_AGENT=1` | unset | Enables agent mode (same as `--agent`) |

## Features

### Core
- **Automatic training mode detection** — inspects dataset columns and selects the right pipeline: SFT (instruction/response), causal (raw text), chat (messages), or multimodal (image+text).
- **Built-in data cleaning** — strips LaTeX, HTML, markdown artifacts, unicode garbage, emoji, junk tokens, and e-commerce noise. Configurable modes for math, LaTeX-preserving, and code-preserving workflows.
- **Model profiles** — ships with tuned configs for Phi-2, Phi-3, Phi-3-Vision, and more. Falls back to sensible defaults for unlisted models.
- **Smart early stopping** — EMA-based loss plateau/worsening detection with exposure floors, LR gates, and dynamic hard caps.
- **Automatic sequence length** — measures token length distribution (P95) and sets `max_length` accordingly.
- **Checkpoint resumption** — interrupted runs resume from the last checkpoint automatically.
- **Cross-platform file picker** — native dialogs on all platforms.

### Interactive
- **Local model picker** — scans `~/HF_Models/` and the Hugging Face hub cache, then presents an interactive numbered list.
- **Post-training summary** — formatted report with loss statistics, trend, and unicode sparkline. Writes a full `training_report.json`.
- **Tkinter dashboard** (`reforge dashboard`, `python -m reforge dashboard`, or `-m reforge.dashboard`) — config editor, start/stop, and a live loss chart.

### Agent mode
- **Self-describing contract** — `reforge --agent-schema` (or `reforge agent-schema`) prints the complete machine-readable contract as one JSON object: CLI arguments, event vocabulary and fields, `run_state.json` keys, exit codes, error codes, dataset schemas, model profiles, artifact map, and an example session. Generated from the live argparse parser and pipeline constants, so it cannot drift. **Agents should call this first.**
- **Preflight** — `reforge --dry_run --base_model PATH dataset.parquet` validates everything and prints the resolved training plan (steps, epochs, scheduler, warmup, early-stop caps, LoRA config) as JSON without loading weights. Exit 0 = safe to launch, 2 = errors listed with codes.
- **NDJSON event stream on stdout** — one JSON object per line: `run_started`, `dataset_loaded`, `training_started`, `step` (loss, percent, ETA), `checkpoint_saved`, `early_stop_check`, `early_stop_triggered`, `training_complete`, `merge_started`, `merge_complete`, `run_complete`, `run_failed`, `run_interrupted`. Each event carries a `stage` field (`loading`/`preparing`/`training`/`exporting`).
- **Pollable state file** — `<output_dir>/run_state.json` holds the latest snapshot (schema `version`, status, step, loss, ETA, stage, error, exit code), rewritten atomically after every event; safe to poll from a background process.
- **Structured failures** — `run_failed`/`training_failed` events carry a machine-branchable `error_code` and a remediation `hint` (`missing_base_model`, `undetectable_schema`, `cuda_oom`, `model_not_found`, `dataset_not_found`, `invalid_dataset`, `missing_dataset`, `dataset_too_small`, `checkpoint_base_mismatch`).
- **Rich `dataset_loaded`** — includes detected `training_mode`, raw `columns`, `mapped_roles` (question/chosen/rejected/image → column), `cleaning_mode`, and `shuffle_seed`.
- **Pure stdout in agent mode** — pretty/console output is diverted to stderr, so stdout parses line-by-line as JSON. Highly recommended to pass `--base_model` and a dataset argument, since interactive pickers are unavailable in agent mode (it fails fast with exit code 2 instead of blocking on stdin).
- **Meaningful exit codes** — `0` success, `1` runtime/training failure, `2` invalid arguments or dataset, `130` interrupted.

```jsonl
{"ts": "2026-11-10T19:05:14Z", "event": "step", "stage": "training", "step": 100, "total_steps": 1500, "loss": 2.4131, "pct": 6.7, "eta_sec": 612}
{"ts": "2026-11-10T19:16:12Z", "event": "run_complete", "stage": "complete", "merged_dir": "C:/Users/you/Reforge_Output/merged", "exit_code": 0}
```

See [`AGENTS.md`](AGENTS.md) for the full agent-facing guide (structure, monitoring, failure handling). A portable skill for skill-compatible agent harnesses is bundled at [`skills/reforge/SKILL.md`](skills/reforge/SKILL.md).

## How It Works

1. **Load** — reads a `.parquet` file or HF Hub dataset, auto-detects training schema.
2. **Clean** — mode-appropriate text normalization.
3. **Tokenize** — P95-based length detection, tokenizes with dynamic padding.
4. **Train** — LoRA adapters, CUDA warmup, Hugging Face `Trainer`.
5. **Stop** — early stopping via EMA plateau/worsening detection.
6. **Report** — summary banner, training_report.json.
7. **Merge** — LoRA adapter merged back into the base model.
8. **Chain** — merged model becomes the base for the next run.

## Adding a New Model Profile

Edit the `MODEL_PROFILES` dictionary in `src/reforge/profiles.py`.

## License

MIT
