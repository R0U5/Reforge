"""Agent mode — machine-readable output for AI agents driving Reforge.

Activated with `--agent`, `--agent_log PATH`, or the REFORGE_AGENT=1
environment variable. Two delivery channels run together:

1. NDJSON event stream — one JSON object per line on stdout. Never ANSI
   colored, never banner-formatted, so an agent can parse stdout line by
   line (or tee it). Pretty/human console output is diverted to stderr in
   agent mode, keeping stdout pure NDJSON.
2. run_state.json — the latest run snapshot in the output directory,
   rewritten atomically after every event. A background agent can poll this
   file instead of holding a pipe open, and it survives process death.

Event vocabulary
    run_started           run configuration recorded (stage: loading)
    dataset_loaded        dataset resolved (rows, training mode, column
                          mapping, cleaning mode)
    training_started      total steps, LR, scheduler, early-stop plan
    step                  progress at every logged step (loss, pct, eta)
    checkpoint_saved      Trainer checkpoint written
    early_stop_check      periodic early-stopping health snapshot
    early_stop_triggered  early stopping decided to stop the run
    training_complete     final loss summary (report path included)
    training_failed       training raised (emergency save was attempted)
    run_interrupted       KeyboardInterrupt; state saved, export continues
    merge_started         LoRA merge/export begins (stage: exporting)
    merge_complete        merged model + tokenizer written
    run_complete          run finished successfully (exit_code 0)
    run_failed            run aborted with a nonzero exit_code

Structured failures
    Recognized failures carry a machine-branchable `error_code` and a
    remediation `hint` (see classify_error). Unknown failures emit the
    message only. Codes: missing_base_model, missing_dataset,
    invalid_dataset, dataset_not_found, undetectable_schema,
    model_not_found, cuda_oom, dataset_too_small, checkpoint_base_mismatch,
    invalid_base_model, tokenizer_load_failed, unknown_args.

Self-description
    `reforge --agent-schema` prints the complete agent contract as one
    JSON object (CLI args, events and their fields, run_state keys, exit
    codes, error codes, dataset schemas, model profiles, artifacts, an
    example session) and exits 0. Agents dropped into a shell should call
    it first: it is the single source of truth, generated from the same
    argparse parser and constants the pipeline runs on.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from transformers import TrainerCallback

from .display import _set_stream


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Bumped whenever the event vocabulary, run_state keys, or error codes
# change in a way an agent could observe. Agents should read this from
# run_state.json ("version") or `reforge --agent-schema`.
# v3: DPO training mode (trl.DPOTrainer), beta in training_started,
#     missing_dependency error code.
# v4: Batch mode (--batch): batch_started/stage_started/stage_completed/
#     stage_failed/batch_interrupted/batch_complete/batch_failed events,
#     batch_state.json, batch_manifest_invalid/batch_stage_failed error codes.
SCHEMA_VERSION = 4


# Status shown in run_state.json after each event type.
_STATUS_BY_EVENT = {
    "run_started":          "starting",
    "dataset_loaded":       "preparing",
    "training_started":     "training",
    "step":                 "training",
    "checkpoint_saved":     "training",
    "early_stop_check":     "training",
    "training_failed":      "failed",
    "training_complete":    "exporting",
    "run_interrupted":      "saving",
    "merge_started":        "exporting",
    "merge_complete":       "exporting",
    "run_complete":         "complete",
    "run_failed":           "failed",
    "batch_started":        "starting",
    "stage_started":        "training",
    "stage_completed":      "complete",
    "stage_failed":         "failed",
    "batch_interrupted":    "interrupted",
    "batch_complete":       "complete",
    "batch_failed":         "failed",
}

# Event fields that mirror into run_state.json.
_MIRRORED_FIELDS = {
    "dataset", "base_model", "training_mode", "rows", "eval_rows",
    "total_steps", "step", "pct", "eta_sec", "lr", "early_stop", "beta",
    "merged_dir", "report", "error", "exit_code", "interrupted",
    "error_code", "hint", "cleaning_mode", "mapped_roles",
    "columns", "shuffle_seed", "dpo_beta",
    "batch_dir", "total_stages", "index", "name", "final_merged",
    "resuming", "returncode",
}

# Substring match (case-insensitive) -> (error_code, hint). First match
# wins; order matters. Kept here so every emitters shares one classifier.
_ERROR_PATTERNS = [
    ("could not detect training mode from columns",
     "undetectable_schema",
     "Dataset columns match no supported schema. Supported modes and their "
     "column aliases are in `reforge --agent-schema` under dataset_schemas; "
     "rename columns or pick a mode-appropriate dataset."),
    ("out of memory",
     "cuda_oom",
     "GPU ran out of VRAM. Lower --batch_size (default 1), lower --grad_accum "
     "does not help VRAM but slows steps, or use a smaller base model."),
    ("not enough data to train",
     "dataset_too_small",
     "Dataset has too few rows for even one optimizer step. Provide more rows "
     "or lower --batch_size/--grad_accum."),
    ("checkpoint base mismatch",
     "checkpoint_base_mismatch",
     "checkpoint-last was trained on a different base model; the run restarted "
     "with a fresh LoRA adapter. Point --base_model at the original base, or "
     "remove the output_dir checkpoint to accept the restart."),
    ("points at a training checkpoint",
     "invalid_base_model",
     "--base_model was set to a Trainer checkpoint (contains adapter_config.json), "
     "not a base model. Pass the original base model directory instead; resuming "
     "from checkpoint-last happens automatically."),
    ("dataset file not found",
     "dataset_not_found",
     "Pass an existing .parquet path as the positional dataset argument, or "
     "use --hf_dataset ID for a Hub dataset."),
    ("no valid .parquet dataset provided",
     "invalid_dataset",
     "The positional dataset argument must be a .parquet file (or use "
     "--hf_dataset ID)."),
    ("--base_model is required in agent mode",
     "missing_base_model",
     "Agent mode is non-interactive: pass --base_model /path/to/model."),
    ("a dataset argument or --hf_dataset is required",
     "missing_dataset",
     "Pass a .parquet path as the positional dataset argument, or "
     "--hf_dataset ID."),
    ("does not appear to have a file named",
     "model_not_found",
     "Base model was not found on disk. Runs are offline by default "
     "(HF_HUB_OFFLINE=1): pass a local model directory (e.g. under "
     "REFORGE_MODELS_DIR or the HF hub cache), or pre-download the model."),
    ("couldn't instantiate the backend tokenizer",
     "tokenizer_load_failed",
     "Tokenizer could not be loaded. Install the converter backend "
     "(pip install sentencepiece, or tiktoken for some models), or verify "
     "the model directory contains tokenizer files."),
    ("dpo training requires the 'trl' package",
     "missing_dependency",
     "DPO training needs the TRL library. Install it with: pip install trl "
     "(or pip install -e . which now includes it)."),
    ("dpo mode requires 'chosen' and 'rejected'",
     "invalid_dataset",
     "DPO training needs a preference-pair dataset: both a chosen and a "
     "rejected column (aliases in `reforge --agent-schema` -> "
     "dataset_schemas)."),
    ("batch manifest",
     "batch_manifest_invalid",
     "The --batch argument must be either a single .json manifest file or a "
     "list of .parquet paths. Manifest format is in `reforge --agent-schema` "
     "-> batch_manifest."),
    ("batch requires an explicit base model",
     "missing_base_model",
     "Batch mode never chains from a stale merged/ silently: pass "
     "--base_model (or set 'base_model' in the manifest) for stage 1."),
]


def classify_error(message) -> tuple:
    """Map an error message to (error_code, hint). First match wins.

    Returns (None, None) when the message is unrecognized, so callers can
    emit prose-only failures without inventing a wrong code.
    """
    if not message:
        return None, None
    text = str(message).lower()
    for needle, code, hint in _ERROR_PATTERNS:
        if needle in text:
            return code, hint
    return None, None


class AgentBridge:
    """Emits NDJSON events and keeps run_state.json current.

    A module-level singleton (`BRIDGE`) is shared across the pipeline.
    Everything is wrapped defensively: a failure to write agent output can
    never crash the training run itself.
    """

    def __init__(self):
        self.enabled = False
        self.output_dir = None
        self.merge_dir = None
        self._state_path = None
        self._log_file = None
        self._state = {}
        self._stage = None

    # ---------------------------------------------------------------- setup

    def enable(self, output_dir, merge_dir=None, log_path=None) -> None:
        """Activate agent mode and divert pretty console output to stderr."""
        if self.enabled:
            return
        self.enabled = True
        self.output_dir = output_dir
        self.merge_dir = merge_dir
        if merge_dir is None and output_dir:
            self.merge_dir = os.path.join(output_dir, "merged")
        self._state_path = os.path.join(output_dir, "run_state.json")

        self._state = {
            "version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "status": "starting",
            "started_at": _now(),
            "updated_at": _now(),
            "step": 0,
            "total_steps": None,
            "pct": 0.0,
            "latest_loss": None,
            "best_loss": None,
            "eta_sec": None,
            "training_mode": None,
            "base_model": None,
            "dataset": None,
            "output_dir": output_dir,
            "merge_dir": self.merge_dir,
            "merged_dir": self.merge_dir,
            "report": None,
            "error": None,
            "error_code": None,
            "hint": None,
            "interrupted": False,
            "exit_code": None,
        }
        if log_path:
            self._state["agent_log"] = os.path.abspath(log_path)

        if log_path:
            try:
                directory = os.path.dirname(os.path.abspath(log_path))
                if directory:
                    os.makedirs(directory, exist_ok=True)
                self._log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            except Exception:
                self._log_file = None

        # Keep stdout machine-parseable: human output goes to stderr.
        try:
            _set_stream(sys.stderr)
        except Exception:
            pass

        self._write_state()

    # -------------------------------------------------------------- emit

    def emit(self, event: str, stage: str = None, **fields) -> None:
        """Send one JSON event line to stdout and refresh run_state.json.

        `stage` marks pipeline position (loading/preparing/training/
        exporting). It is remembered and stamped onto subsequent events
        automatically, so emitters only need to announce transitions.
        """
        if not self.enabled:
            return
        try:
            if stage:
                self._stage = stage
            record = {"ts": _now(), "event": event}
            if self._stage:
                record["stage"] = self._stage
            for key, value in fields.items():
                if value is not None:
                    record[key] = value

            line = json.dumps(record, default=str)
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except Exception:
                pass
            if self._log_file is not None:
                try:
                    self._log_file.write(line + "\n")
                except Exception:
                    pass

            self._update_state(event, fields)
            if stage:
                self._state["stage"] = stage
            self._write_state()
        except Exception:
            pass

    def _update_state(self, event: str, fields: dict) -> None:
        status = _STATUS_BY_EVENT.get(event)
        if status:
            self._state["status"] = status

        if "loss" in fields:
            try:
                loss = float(fields["loss"])
                self._state["latest_loss"] = loss
                best = self._state.get("best_loss")
                self._state["best_loss"] = loss if best is None else min(best, loss)
            except (TypeError, ValueError):
                pass

        for key in _MIRRORED_FIELDS:
            if key in fields and fields[key] is not None:
                self._state[key] = fields[key]

        if event == "run_interrupted":
            self._state["interrupted"] = True

        self._state["updated_at"] = _now()

    def _write_state(self) -> None:
        if not self._state_path:
            return
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
                f.write("\n")
            os.replace(tmp, self._state_path)
        except Exception:
            pass


BRIDGE = AgentBridge()


# ---------------------------------------------------------------- schema
# The machine-readable self-description (`reforge --agent-schema`).
# Generated from the live argparse parser and the real pipeline constants
# so it cannot drift from actual behavior.

_EVENT_FIELDS = {
    "run_started": {
        "description": "Run configuration recorded. First event of every run.",
        "stage": "loading",
        "fields": {
            "output_dir": "str — artifacts root (checkpoints, logs, run_state.json)",
            "dataset": "str|null — local dataset path, when used",
            "hf_dataset": "str|null — Hub dataset ID, when used",
            "base_model": "str|null — base model directory",
            "image_mode": "bool — multimodal (vision+text) training requested",
            "dpo": "bool — DPO training forced via --dpo",
            "dpo_beta": "float|null — DPO beta override (--dpo_beta)",
            "force": "bool — early stopping disabled",
            "epoch": "int — requested epochs",
            "steps": "int|null — fixed step budget (overrides epochs)",
        },
    },
    "dataset_loaded": {
        "description": "Dataset resolved: training mode, row counts, and how "
                       "columns were interpreted. Emitted after cleaning prep.",
        "stage": "preparing",
        "fields": {
            "dataset": "str — resolved dataset source",
            "training_mode": '"sft"|"causal"|"chat"|"multimodal"|"dpo" — detected schema',
            "rows": "int — training rows after cleaning/filtering",
            "eval_rows": "int|null — eval split size (usually null: no split)",
            "base_model": "str — resolved base model (may be a prior merged model)",
            "columns": "list[str] — raw dataset columns",
            "mapped_roles": "dict — {question, chosen, rejected, image} -> column",
            "cleaning_mode": 'str — "math", "latex", "code", or "latex+code"',
            "shuffle_seed": "int — shuffle seed used for the dataset",
        },
    },
    "training_started": {
        "description": "Training plan locked in; Trainer loop begins.",
        "stage": "training",
        "fields": {
            "total_steps": "int — optimizer steps planned",
            "epochs": "int|float — epochs this run",
            "training_mode": "str",
            "lr": "float — learning rate from the model profile",
            "batch_size": "int — per-device batch size",
            "grad_accum": "int — gradient accumulation steps",
            "early_stop": "bool — EMA early stopping enabled",
            "scheduler": "str — LR scheduler type",
            "warmup_steps": "int — LR warmup steps",
            "beta": "float — DPO beta (KL strength vs reference); present on DPO runs only",
        },
    },
    "step": {
        "description": "Progress at every logged step (LOG_STEPS=1). Highest-volume event.",
        "stage": "training",
        "fields": {
            "step": "int",
            "total_steps": "int",
            "pct": "float — percent complete",
            "loss": "float — current training loss",
            "eta_sec": "int|null — estimated seconds remaining",
        },
    },
    "checkpoint_saved": {
        "description": "Trainer checkpoint written (every SAVE_STEPS=25 steps).",
        "stage": "training",
        "fields": {"step": "int", "path": "str — checkpoint-<step> directory"},
    },
    "early_stop_check": {
        "description": "Early-stopping health snapshot, every 25 steps. Loss may "
                       "rise temporarily; only the `verdict` decides stopping.",
        "stage": "training",
        "fields": {
            "step": "int",
            "ema": "float|null — EMA-smoothed loss",
            "best_ema": "float|null — best EMA so far",
            "since_best": "int — steps since best EMA",
            "patience": "int — allowed steps past best",
            "verdict": '"continue"|"stop"',
        },
    },
    "early_stop_triggered": {
        "description": "Early stopping decided the run is done; export follows.",
        "stage": "training",
        "fields": {
            "step": "int", "ema": "float|null", "best_ema": "float|null",
            "steps_total": "int", "hard_cap_steps": "int",
        },
    },
    "training_complete": {
        "description": "Trainer finished normally; loss summary attached.",
        "stage": "exporting",
        "fields": {
            "steps": "int — final global step",
            "final_loss": "float|null", "best_loss": "float|null",
            "time_sec": "float|null — total training time",
            "report": "str — path to training_report.json",
        },
    },
    "training_failed": {
        "description": "Training raised. An emergency adapter save was attempted; "
                       "export is skipped unless the save succeeded.",
        "stage": "training",
        "fields": {
            "error": "str — exception message",
            "error_code": "str|null — machine-branchable code (classify_error)",
            "hint": "str|null — remediation suggestion",
            "emergency_save": "bool — always true on this event",
        },
    },
    "run_interrupted": {
        "description": "KeyboardInterrupt/SIGINT. State is saved and export "
                       "continues; run_state status becomes 'saving'.",
        "stage": "training",
        "fields": {},
    },
    "merge_started": {
        "description": "LoRA merge/export into the base model begins.",
        "stage": "exporting",
        "fields": {"merge_dir": "str — destination directory"},
    },
    "merge_complete": {
        "description": "Merged model + tokenizer (or processor) written.",
        "stage": "exporting",
        "fields": {
            "merged_dir": "str — final model directory",
            "report": "str — training_report.json path",
        },
    },
    "run_complete": {
        "description": "Run finished successfully. Terminal event.",
        "stage": "complete",
        "fields": {"merged_dir": "str", "exit_code": "int — always 0 here"},
    },
    "run_failed": {
        "description": "Run aborted. Terminal event. Always includes exit_code.",
        "stage": "failed",
        "fields": {
            "error": "str — human-readable failure message",
            "error_code": "str|null — machine-branchable code (see error_codes)",
            "hint": "str|null — remediation suggestion",
            "exit_code": "int — 1 runtime, 2 invalid arguments/dataset, 130 interrupted",
        },
    },
    "batch_started": {
        "description": "Batch mode beginning: the stage list and its resolved "
                       "status. First event of every batch run.",
        "stage": "loading",
        "fields": {
            "batch_dir": "str — batch_x/ directory holding batch_state.json + stage dirs",
            "base_model": "str — stage 1's base model (explicit; batch never chains silently)",
            "total_stages": "int",
            "datasets": "list[str] — dataset (or hf_dataset) per stage, in order",
            "resuming": "bool — true when --batch_dir resumed an existing batch",
        },
    },
    "stage_started": {
        "description": "A batch stage's child reforge process is launching.",
        "stage": "loading",
        "fields": {
            "index": "int — 0-based stage index",
            "name": "str — stage name (stage_01, ...)",
            "dataset": "str — this stage's dataset",
            "stage_dir": "str — the stage's own output dir (run_state.json, agent.ndjson)",
            "base_model": "str — previous stage's merged dir (or the explicit base for index 0)",
        },
    },
    "stage_completed": {
        "description": "A batch stage trained, merged, and wrote success.txt.",
        "stage": "complete",
        "fields": {
            "index": "int", "name": "str", "dataset": "str",
            "merged_dir": "str — this stage's merged model (the next stage's base)",
            "training_mode": "str|null — detected mode from the stage's run_state.json",
        },
    },
    "stage_failed": {
        "description": "A batch stage failed; the batch aborts (later stages stay pending). "
                       "error/error_code/hint come from the stage's own run_failed event.",
        "stage": "failed",
        "fields": {
            "index": "int", "name": "str", "dataset": "str",
            "error": "str", "error_code": "str|null", "hint": "str|null",
            "returncode": "int — the child process exit code",
        },
    },
    "batch_interrupted": {
        "description": "Ctrl+C during a batch: the interrupted child got "
                       "CTRL_BREAK, saved its state, and exited; remaining "
                       "stages stay pending. Resume with --batch_dir.",
        "stage": "interrupted",
        "fields": {
            "index": "int — the stage that was interrupted",
            "name": "str",
            "returncode": "int — usually 130",
            "hint": "str — the exact resume command",
        },
    },
    "batch_complete": {
        "description": "All batch stages completed. Terminal event.",
        "stage": "complete",
        "fields": {
            "final_merged": "str — the last stage's merged model directory",
            "completed_stages": "int",
        },
    },
    "batch_failed": {
        "description": "Batch aborted (manifest invalid or a stage failed). Terminal event.",
        "stage": "failed",
        "fields": {"error": "str", "error_code": "str|null", "hint": "str|null"},
    },
}

_ERROR_CODES_DOC = {code: hint for _, code, hint in _ERROR_PATTERNS}
# Raised directly in __main__ for unrecognized CLI flags (not message-derived).
_ERROR_CODES_DOC["unknown_args"] = (
    "Unrecognized CLI flags were passed. Check flag spelling; "
    "`reforge --agent-schema` lists every valid flag.")

_EXIT_CODES_DOC = {
    "0": "success",
    "1": "runtime/training failure (weights load, CUDA, Trainer error)",
    "2": "invalid arguments or dataset (missing --base_model/dataset, bad "
         "path, undetectable schema)",
    "130": "interrupted (Ctrl+C)",
}


def _cli_arguments_from_parser(parser):
    """Describe argparse options without a second hand-maintained table."""
    entries = []
    for action in parser._actions:
        if action.dest == "help":
            continue
        entry = {"dest": action.dest}
        if action.option_strings:
            entry["flag"] = action.option_strings[0]
            if isinstance(action, argparse._StoreTrueAction):
                entry["type"] = "bool"
                entry["default"] = False
            else:
                entry["type"] = getattr(action.type, "__name__", "string") if action.type else "string"
                entry["default"] = None if action.default is argparse.SUPPRESS else action.default
                if action.choices:
                    entry["choices"] = list(action.choices)
        else:
            entry["positional"] = True
            entry["required"] = bool(action.required)
        if action.help:
            entry["help"] = action.help
        entries.append(entry)
    return entries


def build_agent_schema(parser=None) -> dict:
    """Assemble the full agent contract as one JSON-serializable dict."""
    from .dataset import TRAINING_SCHEMAS
    from .profiles import MODEL_PROFILES, MODEL_PROFILE_DEFAULT

    contract = {
        "tool": "reforge",
        "description": "Modular LoRA fine-tuning pipeline for Hugging Face "
                       "causal and vision-language models.",
        "schema_version": SCHEMA_VERSION,
        "agent_contract": [
            "Agent mode is non-interactive: --base_model AND a dataset "
            "(positional .parquet path or --hf_dataset) are required, or the "
            "run exits 2 before touching the GPU.",
            "In agent mode stdout is pure NDJSON (one JSON object per line); "
            "all human output goes to stderr. Parse stdout line by line.",
            "If you cannot hold stdout open, poll "
            "<output_dir>/run_state.json — it is rewritten atomically after "
            "every event and survives process death.",
            "Treat `step` loss values as noisy: judge training health from "
            "early_stop_check EMA trends, not single steps.",
            "Re-run `reforge --agent-schema` before relying on any of this: "
            "this contract is versioned (schema_version) and may evolve.",
        ],
        "invocation": {
            "module": "python -m reforge [dataset.parquet] [flags]",
            "console_script": "reforge [dataset.parquet] [flags]",
            "agent_mode_activation": "--agent flag, or REFORGE_AGENT=1 env var",
            "self_description": "reforge --agent-schema (prints this JSON, exit 0)",
            "dry_run": "reforge --dry_run --base_model PATH dataset.parquet "
                       "(validates everything and prints the resolved training "
                       "plan as JSON without loading weights or training)",
        },
        "cli_arguments": _cli_arguments_from_parser(parser) if parser else [],
        "environment_variables": {
            "REFORGE_AGENT": '"1" enables agent mode (same as --agent)',
            "REFORGE_MODELS_DIR": "model scan dir (default ~/HF_Models)",
            "REFORGE_OUTPUT_DIR": "default output dir (default ~/Reforge_Output)",
            "HF_HOME": "HF cache root (default ~/.cache/huggingface)",
        },
        "exit_codes": _EXIT_CODES_DOC,
        "events": _EVENT_FIELDS,
        "error_codes": _ERROR_CODES_DOC,
        "run_state": {
            "path": "<output_dir>/run_state.json",
            "version": SCHEMA_VERSION,
            "write_policy": "atomically rewritten (tmp + os.replace) after every event",
            "keys": {
                "version": "schema version of this contract",
                "pid": "process id (kill this to stop a run)",
                "status": "starting|preparing|training|exporting|saving|complete|failed",
                "stage": "pipeline position (loading/preparing/training/exporting), "
                         "set by the latest stage-tagged event",
                "started_at": "run start (UTC)",
                "updated_at": "last event time (UTC)",
                "step": "current optimizer step",
                "total_steps": "planned steps (null until training_started)",
                "pct": "percent complete",
                "latest_loss": "most recent step loss",
                "best_loss": "lowest loss seen so far",
                "eta_sec": "estimated seconds remaining",
                "training_mode": "detected dataset schema mode",
                "base_model": "resolved base model path",
                "dataset": "resolved dataset source",
                "output_dir": "artifacts root",
                "merge_dir": "final merged-model directory (<output_dir>/merged)",
                "report": "training_report.json path",
                "error": "failure message (run_failed)",
                "error_code": "machine-branchable failure code (run_failed)",
                "hint": "remediation suggestion (run_failed)",
                "interrupted": "true after run_interrupted",
                "exit_code": "set when the run ends (0/1/2/130)",
                "agent_log": "absolute path of the teed NDJSON stream, when --agent_log is used",
            },
        },
        "dataset_schemas": {
            mode: {
                "required_columns": schema["required"],
                "optional_columns": schema.get("optional", []),
                "column_aliases": schema["aliases"],
            }
            for mode, schema in TRAINING_SCHEMAS.items()
        },
        "model_profiles": {
            "resolution": "substring match of the base-model basename "
                          "(e.g. 'phi-2'); otherwise MODEL_PROFILE_DEFAULT",
            "profiles": {**MODEL_PROFILES, "default": MODEL_PROFILE_DEFAULT},
            "notes": {
                "learning_rate": "seconds per step come from batch_size x grad_accum; "
                                 "override --grad_accum to change effective batch size",
                "modules_to_save": "lm_head is NOT trained by default on most models "
                                   "(~5 GB VRAM on large-vocab models); opt in with --train_lm_head",
                "safe_serialization": "some Phi-3 profiles must save with "
                                      "safe_serialization=False (shared tensors)",
            },
        },
        "artifacts": {
            "<output_dir>/run_state.json": "live run snapshot (pollable)",
            "<output_dir>/console.log": "fallback console log for broken pipes",
            "<output_dir>/checkpoint-<step>/": "Trainer checkpoints (save_steps=25, keep 2)",
            "<output_dir>/checkpoint-last": "symlink/copy of the newest checkpoint; "
                                            "present in output_dir => next run resumes from it",
            "<output_dir>/last_good_ckpt/": "backup copy of the last checkpoint",
            "<output_dir>/training_chain.txt": "dataset usage log for chained runs",
            "<output_dir>/merged/": "final merged model + tokenizer + training_report.json",
            "<output_dir>/merged/success.txt": "marker: merge finished. If present, the "
                                               "NEXT run uses merged/ as its base model (training chain)",
            "<output_dir>/merged/training_report.json": "loss statistics, trend, timing",
        },
        "batch_mode": {
            "invocation": "reforge --batch stage1.parquet stage2.parquet pairs.parquet "
                          "--base_model PATH  |  --batch manifest.json --base_model PATH",
            "manifest": {
                "path": "pass a single .json path to --batch for per-stage control",
                "keys": {
                    "base_model": "required (or pass --base_model): stage 1's base. "
                                  "Batch never chains from a stale merged/ silently",
                    "output_dir": "optional: batch root (batch_<timestamp> is created inside)",
                    "keep_stage_merges": "optional bool: keep every stage's merged model "
                                         "(disk-heavy); default false = only the newest "
                                         "completed stage's merged is kept",
                    "stages": "list of stage objects, trained in order; the LAST stage's "
                              "merged model is the final output",
                },
                "stage_keys": "dataset | hf_dataset (required), plus: name, epochs, steps, "
                              "force, dpo, dpo_beta, latex, code, batch_size, grad_accum",
                "note": "a stage whose dataset has chosen+rejected columns trains as DPO "
                        "automatically - no special batch keys needed",
            },
            "layout": {
                "batch_dir": "output_dir/batch_<timestamp>/ (or --batch_dir to resume)",
                "state": "<batch_dir>/batch_state.json - per-stage status; the resume anchor",
                "stages": "<batch_dir>/stage_01/..stage_0N/ - each a complete reforge "
                          "output dir; stage N trains on stage N-1's merged model",
            },
            "resume": "rerun the same --batch command with --batch_dir <batch_dir>; "
                      "completed stages are skipped, the interrupted stage resumes "
                      "from its own checkpoint-last",
            "run_state": "in batch mode, <batch_dir>/run_state.json tracks batch events; "
                         "each stage also writes its own <stage_dir>/run_state.json and "
                         "<stage_dir>/agent.ndjson (teeable via --agent_log)",
        },
        "example_session": [
            '$ reforge --agent --base_model ~/HF_Models/phi-2 data.parquet --agent_log run.ndjson',
            '{"ts":"2026-11-10T19:05:01Z","event":"run_started","output_dir":"...","base_model":"..."}',
            '{"ts":"2026-11-10T19:05:09Z","event":"dataset_loaded","dataset":"data.parquet",'
            '"training_mode":"sft","rows":4200,"base_model":"...","cleaning_mode":"math",'
            '"mapped_roles":{"question":"instruction","chosen":"response","rejected":null,"image":null}}',
            '{"ts":"2026-11-10T19:05:40Z","event":"training_started","total_steps":262,'
            '"lr":2e-05,"batch_size":1,"grad_accum":16,"early_stop":true,"scheduler":"linear"}',
            '{"ts":"2026-11-10T19:05:41Z","event":"step","step":1,"total_steps":262,'
            '"pct":0.38,"loss":2.4131,"eta_sec":600}',
            '{"ts":"2026-11-10T19:16:12Z","event":"run_complete",'
            '"merged_dir":"~/Reforge_Output/merged","exit_code":0}',
        ],
        "batch_example_session": [
            '$ reforge --agent --batch sft1.parquet sft2.parquet pairs.parquet --base_model ~/HF_Models/phi-2',
            '{"event":"batch_started","batch_dir":".../batch_20261113_143000","total_stages":3,"resuming":false}',
            '{"event":"stage_started","index":0,"dataset":"sft1.parquet","base_model":"~/HF_Models/phi-2"}',
            '{"event":"stage_completed","index":0,"merged_dir":".../stage_01/merged","training_mode":"sft"}',
            '{"event":"stage_started","index":1,"dataset":"sft2.parquet",'
            '"base_model":".../stage_01/merged"}',
            '{"event":"stage_completed","index":1,"merged_dir":".../stage_02/merged"}',
            '{"event":"stage_started","index":2,"dataset":"pairs.parquet","base_model":".../stage_02/merged"}',
            '{"event":"batch_complete","final_merged":".../stage_03/merged","exit_code":0}',
        ],
    }
    return contract


class AgentProgressCallback(TrainerCallback):
    """TrainerCallback emitting per-step progress and checkpoint events.

    Instantiate with an optional `EarlyStopByLoss` instance to also surface
    early-stopping health. `emit()` no-ops when agent mode is off, so the
    callback is harmless to attach unconditionally.
    """

    def __init__(self, bridge=BRIDGE, early_stop=None, check_every: int = 25):
        self.bridge = bridge
        self.early_stop = early_stop
        self.check_every = max(1, int(check_every))
        self._t0 = None
        self._step0 = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self.bridge.enabled:
            return
        loss = logs.get("loss")
        if loss is None:  # final train_runtime metrics land in training_complete
            return

        step = int(getattr(state, "global_step", 0) or 0)
        total = int(getattr(state, "max_steps", 0) or getattr(args, "max_steps", 0) or 0)

        eta_sec = None
        now = time.time()
        if total and step:
            if self._t0 is None:
                self._t0, self._step0 = now, step
            elapsed = now - self._t0
            done = step - self._step0
            if done > 0 and elapsed > 0:
                eta_sec = round(max(0.0, (total - step) / (done / elapsed)))

        self.bridge.emit(
            "step",
            step=step,
            total_steps=total,
            pct=round(100.0 * step / total, 2) if total else None,
            loss=round(float(loss), 6),
            eta_sec=eta_sec,
        )

        es = self.early_stop
        if es is not None and total and (step % self.check_every == 0 or step == total):
            self.bridge.emit(
                "early_stop_check",
                step=step,
                ema=round(es.ema, 6) if es.ema is not None else None,
                best_ema=round(es.best_ema, 6) if es.best_ema != float("inf") else None,
                since_best=es.since_best,
                patience=es.patience,
                verdict="stop" if es.triggered else "continue",
            )

    def on_save(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        out_dir = getattr(args, "output_dir", None) or self.bridge.output_dir or ""
        self.bridge.emit(
            "checkpoint_saved",
            step=step,
            path=os.path.join(out_dir, f"checkpoint-{step}"),
        )

    def on_train_end(self, args, state, control, **kwargs):
        es = self.early_stop
        if es is not None and getattr(es, "triggered", False):
            step = int(getattr(state, "global_step", 0) or 0)
            self.bridge.emit(
                "early_stop_triggered",
                step=step,
                ema=round(es.ema, 6) if es.ema is not None else None,
                best_ema=round(es.best_ema, 6) if es.best_ema != float("inf") else None,
                steps_total=es.steps_total,
                hard_cap_steps=es.hard_cap_steps,
            )
