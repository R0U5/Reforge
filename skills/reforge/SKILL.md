---
name: reforge
description: >
  Drive the Reforge LoRA fine-tuning pipeline non-interactively: launch
  training runs, monitor NDJSON progress or run_state.json, preflight
  configs with --dry_run, and diagnose run_failed error codes. Use when the
  user asks to fine-tune or train a model with reforge, or to inspect,
  monitor, or fix a reforge run.
---

# Driving Reforge

Reforge fine-tunes Hugging Face causal / vision-language models with LoRA:
**Load → Clean → Tokenize → Train → Early-stop → Report → Merge → Chain**.
One command goes from a `.parquet` dataset to a merged, ready-to-use model.

## First move: get the live contract

```bash
reforge --agent-schema
```

Prints the complete machine-readable contract as one JSON object (CLI args,
event fields, run_state keys, exit codes, error codes, dataset schemas,
model profiles, artifact map). It is generated from the real argparse parser
and pipeline constants — if this skill and the output disagree, trust the
command. Re-run it before relying on cached knowledge.

## Golden rules

1. **Non-interactive.** Always pass `--base_model /path/to/model` AND a
   dataset (positional `.parquet` path or `--hf_dataset ID`). Anything else
   fails fast with exit 2 instead of hanging on an interactive picker.
2. **stdout is pure NDJSON** in agent mode (activate with `--agent` or
   `REFORGE_AGENT=1`); human output goes to stderr. Parse stdout line by line.
3. **Errors are structured.** Branch on `error_code`, not prose (see below).
4. **Do not react to single `step` losses.** They are noisy; judge health
   from `early_stop_check` EMA trends. Reforge stops itself.
5. **Runs are offline by default** (`HF_HUB_OFFLINE=1`): base models must be
   local directories.

## Launch a run

```bash
reforge --agent \
  --base_model /path/to/model \
  data.parquet \
  --agent_log run.ndjson
```

Useful flags: `--steps N` (fixed step budget), `--epoch N`, `--force`
(disable early stopping), `--batch_size N`, `--grad_accum N`, `--image`
(multimodal), `--latex` / `--code` (preserve LaTeX / code during cleaning),
`--output_dir PATH`.

## Monitor

- **Stream stdout NDJSON** (preferred): events `run_started → dataset_loaded
  → training_started → step* → checkpoint_saved → merge_started →
  merge_complete → run_complete`, each with `ts`, `event`, `stage`.
- **Poll `<output_dir>/run_state.json`** (background-safe): it is rewritten
  atomically after every event; branch on `status`
  (`starting → preparing → training → exporting → complete | failed`) and
  `exit_code` (`0` ok, `1` runtime failure, `2` bad args/dataset, `130`
  interrupted). `pid` is there if you need to stop the run.
- Terminal events: `run_complete` (success) and `run_failed` (check
  `error_code` / `hint`).

## Preflight before a long GPU run

```bash
reforge --dry_run --base_model /path/to/model data.parquet
```

Validates args, dataset schema, and model path, then prints the resolved
training plan (steps, epochs, scheduler, warmup, early-stop caps, LoRA
config) as JSON without loading weights. Exit 0 = safe to launch; exit 2 =
fix the listed errors. Requires `--base_model` + dataset, same as agent mode.

## Diagnose failures (branch on `error_code`)

| error_code | Fix |
|---|---|
| `missing_base_model` | pass `--base_model /path/to/model` |
| `missing_dataset` | pass a `.parquet` path or `--hf_dataset ID` |
| `invalid_dataset` / `dataset_not_found` | fix the dataset path (`.parquet`) |
| `undetectable_schema` | rename columns to match a supported schema (see `--agent-schema` → `dataset_schemas`) |
| `model_not_found` | base model not on disk; pass a local model dir |
| `cuda_oom` | lower `--batch_size`, or use a smaller base model |
| `dataset_too_small` | more rows, or lower batch/accum |
| `checkpoint_base_mismatch` | checkpoint belongs to another base; run restarted fresh |
| `invalid_base_model` | `--base_model` points at a Trainer checkpoint, not a model dir |
| `tokenizer_load_failed` | install `sentencepiece`/`tiktoken`, or check tokenizer files |
| `unknown_args` | unrecognized CLI flag (typo); see `--agent-schema` |

## Dataset schemas

Detected from column names (alias matching): **sft** (`chosen` required,
`question` optional), **causal** (raw `text`/`prompt`), **chat**
(`messages`), **multimodal** (image + text, with `--image`). The
`dataset_loaded` event reports exactly how columns were mapped
(`mapped_roles`) and the `cleaning_mode` used — default cleaning strips
LaTeX/HTML/markdown/emoji, so pass `--latex` or `--code` when that content
is the point.

## Side effects to check before launching

Both silently change what the next run does (artifacts live under
`--output_dir`, default `~/Reforge_Output`):

- `<output_dir>/checkpoint-last` present → the run **resumes** from it.
- `<output_dir>/merged/success.txt` present → the run **chains**: the merged
  model becomes the base (chain training).

Final model always lands in `<output_dir>/merged/` next to
`training_report.json`.

## Going deeper

- `AGENTS.md` (repo root) — codebase structure, development conventions.
- `README.md` — human-oriented docs.
- `reforge --agent-schema` — the authoritative live contract.
