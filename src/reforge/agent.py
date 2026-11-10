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
    run_started           run configuration recorded
    dataset_loaded        dataset resolved (rows, training mode)
    training_started      total steps, LR, scheduler, early-stop plan
    step                  progress at every logged step (loss, pct, eta)
    checkpoint_saved      Trainer checkpoint written
    early_stop_check      periodic early-stopping health snapshot
    early_stop_triggered  early stopping decided to stop the run
    training_complete     final loss summary (report path included)
    training_failed       training raised (emergency save was attempted)
    run_interrupted       KeyboardInterrupt; state saved, export continues
    merge_started         LoRA merge/export begins
    merge_complete        merged model + tokenizer written
    run_complete          run finished successfully (exit_code 0)
    run_failed            run aborted with a nonzero exit_code
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from transformers import TrainerCallback

from .display import _set_stream


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
}

# Event fields that mirror into run_state.json.
_MIRRORED_FIELDS = {
    "dataset", "base_model", "training_mode", "rows", "eval_rows",
    "total_steps", "step", "pct", "eta_sec", "lr", "early_stop",
    "merged_dir", "report", "error", "exit_code", "interrupted",
}


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
            "version": 1,
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
            "interrupted": False,
            "exit_code": None,
        }

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

    def emit(self, event: str, **fields) -> None:
        """Send one JSON event line to stdout and refresh run_state.json."""
        if not self.enabled:
            return
        try:
            record = {"ts": _now(), "event": event}
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
