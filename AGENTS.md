# AGENTS.md — Guide for AI agents driving Reforge

This file is the fast-path orientation for coding agents (Codex, Cursor,
Claude Code, etc.) dropped into a shell with Reforge available. It explains
how Reforge is structured, how to drive it non-interactively, and how to
interpret its output.

**The authoritative, machine-readable contract is the CLI itself:**

```bash
reforge --agent-schema      # full agent contract as one JSON object (exit 0)
reforge --dry_run --base_model /path/to/model dataset.parquet   # preflight
reforge --help              # argparse help
```

A portable skill for skill-compatible agent harnesses is bundled at
`skills/reforge/SKILL.md`.

`--agent-schema` is generated from the live argparse parser and pipeline
constants, so it cannot drift from actual behavior. If this file and
`--agent-schema` ever disagree, trust `--agent-schema` and fix this file.

---

## 1. What Reforge is

A modular LoRA fine-tuning pipeline for Hugging Face causal language models
and vision-language models. One command goes from raw `.parquet` dataset to
a merged, ready-to-use model:

**Load → Clean → Tokenize → Train → Early-stop → Report → Merge → Chain**

"Chain" means the merged output (`<output_dir>/merged/`) becomes the base
model for the next run automatically (gated by `merged/success.txt`).

## 2. Codebase structure

```
src/reforge/
├── __main__.py          # CLI entry point, arg parsing, pipeline orchestration
├── config.py            # Env setup, paths, constants (SAVE_STEPS=25, LOG_STEPS=1, BATCH_SIZE=1)
├── profiles.py          # MODEL_PROFILES: per-model LoRA/LR/serialisation configs (phi-2, phi-3, phi-3-vision)
├── dataset.py           # Loading, TRAINING_SCHEMAS schema detection, cleaning
├── tokenization.py      # Tokenization, P95 length measurement, collator
├── training.py          # Model loading (4-bit), LoRA setup, Trainer loop
├── early_stopping.py    # EMA-based plateau/worsening detection, scheduler selection
├── reporting.py         # training_report.json + loss summaries
├── agent.py             # Agent mode: NDJSON events, run_state.json, --agent-schema contract
├── display.py           # Console output (diverted to stderr in agent mode)
├── utils.py             # Model scanner, file pickers, GC
└── dashboard.py         # Tkinter GUI (`reforge dashboard`)
```

Where to change things:

| Task | File |
|------|------|
| New/changed CLI flag | `__main__.py` parser + the flag table in `README.md` |
| New event or field | `agent.py` (`_EVENT_FIELDS` + the `emit(...)` call site) |
| New error code | `agent.py` `_ERROR_PATTERNS` (classified from message substrings) |
| New model defaults | `profiles.py` `MODEL_PROFILES` |
| New dataset schema/aliases | `dataset.py` `TRAINING_SCHEMAS` |
| Docs for agents | this file + `agent.py` `build_agent_schema()` |

## 3. The agent-mode contract (non-negotiable)

1. **Non-interactive.** Agent mode never opens a picker. `--base_model` AND
   a dataset (positional `.parquet` path or `--hf_dataset ID`) are required;
   otherwise the run exits 2 immediately without touching the GPU.
2. **stdout is pure NDJSON.** One JSON object per line, never colored, never
   banner-formatted. All human output goes to stderr. Parse stdout line by
   line; ignore stderr (or capture it for debugging).
3. **run_state.json is the pollable fallback.** If you cannot hold stdout
   open (background process), poll `<output_dir>/run_state.json` — it is
   rewritten atomically after every event and survives process death.
4. **Offline by default.** `HF_HUB_OFFLINE=1` and friends are set in
   `config.py`. Base models must exist on disk (local dir or HF cache);
   `--hf_dataset` downloads only because the loader temporarily lifts the
   offline env vars.
5. **Exit codes:** `0` success, `1` runtime/training failure, `2` invalid
   arguments or dataset, `130` interrupted.

Activation: `--agent` flag or `REFORGE_AGENT=1`.

```bash
reforge --agent --base_model ~/HF_Models/phi-2 data.parquet --agent_log run.ndjson
```

## 4. Monitoring a run

Preferred: stream stdout NDJSON. Each event has `ts`, `event`, `stage`, and
event-specific fields. Terminal events are `run_complete` (exit 0) and
`run_failed` (nonzero exit_code).

Fallback: poll `run_state.json` and branch on `status`
(`starting → preparing → training → exporting → complete | failed`) and
`exit_code`.

Judging training health — **do not** react to single `step` losses; they are
noisy. Use the `early_stop_check` event's `ema` / `best_ema` trend and its
`verdict` field. Only `early_stop_triggered` and terminal events require
action. Reforge stops itself; an external agent normally only needs to wait,
kill (see `pid` in run_state.json), or diagnose failures.

## 5. Failures: branch on `error_code`, not prose

`run_failed` / `training_failed` events (and `run_state.json`) carry
`error_code` + `hint` for recognized failures:

| error_code | Meaning / fix |
|------------|---------------|
| `missing_base_model` | pass `--base_model /path/to/model` |
| `missing_dataset` | pass a `.parquet` path or `--hf_dataset ID` |
| `invalid_dataset` | dataset argument must be a `.parquet` file |
| `dataset_not_found` | dataset path does not exist |
| `undetectable_schema` | columns match no supported schema — see §6 |
| `model_not_found` | base model not on disk (offline default) |
| `cuda_oom` | lower `--batch_size`, or use a smaller base model |
| `dataset_too_small` | too few rows for one optimizer step |
| `checkpoint_base_mismatch` | checkpoint-last belongs to a different base; run restarted fresh |
| `invalid_base_model` | `--base_model` points at a Trainer checkpoint, not a model dir — pass the original base |
| `tokenizer_load_failed` | install `sentencepiece`/`tiktoken`, or check the model's tokenizer files |
| `missing_dependency` | install the required extra (e.g. `pip install trl` for DPO) |
| `unknown_args` | unrecognized CLI flag (typo); see `--agent-schema` for valid flags |
| `batch_manifest_invalid` | `--batch` arg is not a valid manifest/parquet list; check the JSON or dataset paths |
| `batch_stage_failed` | a batch stage's child run failed; see `stage_failed` for the stage's own error_code |

Unknown failures emit the message only. `hint` contains the remediation.

## 6. Dataset schemas (how your data will be interpreted)

Modes are detected from column names via alias matching, checked in order:
`multimodal` (if an image column exists) → `chat` → `dpo` → `sft` →
`causal`. Full alias tables: `reforge --agent-schema` → `dataset_schemas`.

- **dpo** — needs BOTH `chosen` and `rejected` (preference pairs), optional
  `question` (prompt). Trains with TRL's `DPOTrainer` (LoRA on the 4-bit
  base; the adapter-disabled base weights act as the reference model).
  `--dpo_beta` sets beta (default 0.1); `--dpo` forces DPO interpretation.
  Workflow: SFT first, then DPO on a pair dataset — the training chain
  (merged/success.txt) makes the SFT-merged model the DPO base
  automatically.
- **sft** — needs `chosen` (response/answer/completion/…), optional
  `question` (instruction/prompt/…)
- **causal** — needs a raw-text column (`text`/`prompt`)
- **chat** — needs `messages` (OpenAI-style message lists)
- **multimodal** — image column + text; enable with `--image`
  (Phi-3-Vision class models)

If detection fails (`undetectable_schema`), rename columns to match an alias
list, or choose a different dataset. The `dataset_loaded` event shows exactly
how your columns were mapped (`mapped_roles`) and which cleaning mode ran
(`cleaning_mode`: `math` default, `latex`/`code` with the matching flags).

Note: cleaning strips LaTeX, HTML, markdown artifacts, emoji, and e-commerce
noise by default — pass `--latex` or `--code` when that content is the point.

## 7. Preflight before a real run

```bash
reforge --dry_run --base_model ~/HF_Models/phi-2 data.parquet
```

Prints one JSON object: validation errors (with codes), the resolved model
profile, detected schema + column mapping, and the full training plan
(steps, epochs, scheduler, warmup, early-stop caps, LoRA config) — computed
with the same math as the real run, without loading weights. Exit 0 = safe
to launch; exit 2 = fix the listed errors first.

Dry-run is non-interactive and requires `--base_model` + dataset, just like
agent mode. For batch runs, `--dry_run --batch manifest.json --base_model X`
validates every stage and prints a per-stage plan array.

## 8. Batch mode: multiple datasets, chained merges

```bash
reforge --agent --batch sft1.parquet sft2.parquet pairs.parquet --base_model ~/HF_Models/phi-2
reforge --agent --batch manifest.json --base_model ~/HF_Models/phi-2
```

Stages train **in order**; stage N trains on stage N-1's merged model. The
last stage's merged dir is the final output. A stage whose dataset has
`chosen`+`rejected` columns trains as DPO automatically.

- **Base model is required** (`--base_model` or manifest `base_model`):
  batch never chains from a stale `merged/` silently.
- **Manifest** (single `.json` arg): `{"base_model": ..., "stages":
  [{"dataset": ..., "epochs": 2}, {"dataset": ..., "code": true,
  "steps": 300}, {"dataset": ..., "dpo_beta": 0.1}]}`. Stage keys mirror CLI
  flags. Full spec: `reforge --agent-schema` -> `batch_mode`.
- **Events**: `batch_started` -> (`stage_started` -> `stage_completed`)* ->
  `batch_complete` (or `stage_failed`/`batch_interrupted`). In agent mode,
  `<batch_dir>/run_state.json` tracks the batch; each stage also writes its
  own `<batch_dir>/stage_0N/run_state.json` and `agent.ndjson` — poll those
  for per-step loss during a stage.
- **Resume**: Ctrl+C interrupts cleanly (the running stage saves its
  adapter). Rerun with `--batch_dir <batch_dir>` to skip completed stages
  and resume the interrupted one from its checkpoint. A stage whose merge
  was interrupted (no `success.txt`) is wiped and restarted automatically.
- **Disk**: stage N-1's merged is deleted once stage N completes (the
  running stage only ever needs its predecessor). `"keep_stage_merges":
  true` in the manifest keeps everything.

## 9. Artifacts (all under `--output_dir`, default `~/Reforge_Output`)

| Path | Meaning |
|------|---------|
| `run_state.json` | live run snapshot (pollable) |
| `console.log` | fallback console log for broken pipes |
| `checkpoint-<step>/` | Trainer checkpoints (every 25 steps, keep 2) |
| `checkpoint-last` | newest checkpoint; **its presence makes the next run resume** |
| `last_good_ckpt/` | backup of the last checkpoint |
| `training_chain.txt` | dataset usage log across chained runs |
| `merged/` | final merged model + tokenizer + `training_report.json` |
| `merged/success.txt` | merge marker; when present the NEXT run uses `merged/` as base |

Batch mode adds, under the batch dir:

| Path | Meaning |
|------|---------|
| `batch_state.json` | per-stage status; the resume anchor (rewrite-atomic) |
| `stage_01/` … `stage_0N/` | one complete reforge output dir per stage |
| `stage_0N/agent.ndjson` | the stage's full NDJSON event stream (per-step loss lives here) |
| `stage_0N/merged/` | the stage's merged model = next stage's base |

Agents should check for `merged/success.txt` / `checkpoint-last` before
launching — they silently change what the next run does (resume / chain).

## 10. Conventions for changing Reforge itself

- Keep agent output machine-parseable: never print pretty text to stdout in
  agent mode; use `_info/_ok/_warn/_err/_banner` (they divert to stderr).
- `BRIDGE.emit(...)` no-ops when agent mode is off — safe to call anywhere.
- Everything agent-output-related is wrapped defensively: a broken console
  or log file must never crash training.
- When you add an event or field, update `_EVENT_FIELDS` in `agent.py` and
  bump `SCHEMA_VERSION`; the contract is regenerated from real code, so
  nothing else can go stale.
- Emitters announce stage transitions via `emit(..., stage="training")`;
  later events inherit the current stage automatically.
- Keep `AGENTS.md`, the `agent.py` module docstring, and `README.md` in sync
  with any contract change.
