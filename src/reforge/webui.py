"""Local web UI for Reforge — a modern browser dashboard.

Zero third-party dependencies: a stdlib ``http.server`` serves one
self-contained page (``webui_assets.py``) and a small JSON API. Launch with
``reforge webui`` (alias ``reforge ui``) and open http://127.0.0.1:7861.

Architecture (see docs/WEBUI_MIGRATION.md, decisions D1-D11):
    The server is a thin supervisor. ``POST /api/start`` validates the
    config and spawns ``python -m reforge --agent ...`` as a child process.
    Reader threads tail the child's stdout (NDJSON agent events) and stderr
    (human log + loss rows); events update a thread-safe ``Job`` snapshot
    that the page polls every second. ``POST /api/stop`` sends
    CTRL_BREAK/SIGINT for a graceful checkpoint save.

Endpoints:
    GET  /              the single-page app
    GET  /api/status    current job state (polled by the page)
    GET  /api/models    local model candidates for the base-model field
    GET  /api/places    filesystem shortcuts for the folder picker
    GET  /api/browse    directory listing for the folder picker
    GET  /api/analysis  loss-analysis math over the live loss ring
    GET  /api/report    training_report.json for the finished run
    POST /api/start     begin a training job   -> {ok} | {error}
    POST /api/stop      graceful stop (checkpoint save)
    POST /api/preview   validated command preview (no side effects)
    POST /api/mkdir     create a folder from the picker
    POST /api/reveal    open a file/folder in the OS file manager

Filesystem endpoints are localhost-only: they return 403 when the server
is bound to a non-loopback address and accessed remotely.
"""

from __future__ import annotations

import json
import math
import os
import platform
import signal
import string
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .config import DEFAULT_HF_MODELS_DIR, OUTPUT_DIR
from .display import _info, _title, _warn
from .webui_assets import INDEX_HTML
from .webui_config import build_argv, scan_models

# ── Tunables ─────────────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7861          # 7860 is Lethe's default; both may run side by side
MAX_LOG_LINES = 800          # ring buffer for the console pane
LOG_TAIL = 160               # lines sent per status poll
MAX_SIGNALS = 60             # recent activity timeline entries
MAX_ES_CHECKS = 240          # EMA points kept for the chart overlay
MAX_BROWSE_ENTRIES = 2000    # guard against enormous directories

# Exit codes that mean "the user interrupted the child". A stop requested
# through the UI is not a failure, but the child cannot always report 130:
#  - 130 is our own SystemExit after a handled KeyboardInterrupt;
#  - -2 / -15 are POSIX SIGINT/SIGTERM (Popen reports negative signals);
#  - 3221225786 (0xC000013A, or its signed form) is Windows' unhandled
#    CTRL_BREAK exit, e.g. a stop during the child's heavy imports before
#    its SIGBREAK handler is installed.
_INTERRUPT_EXIT_CODES = (130, -2, -15, 3221225786, -1073741510)
MAX_CHART_POINTS = 5000      # loss ring for the live chart

# Child stdout suppression (same list as the Tk dashboard keeps).
_SUPPRESS_PATTERNS = (
    "FutureWarning: `tokenizer` is deprecated",
    "UserWarning: The PEFT config's `base_model_name_or_path` was renamed",
    "Use `processing_class` instead.",
)


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(value):
    """Finite float or None — agent events must never poison the chart."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ── Job state ────────────────────────────────────────────────────────
class Job:
    """Thread-safe snapshot of the supervised run.

    Fields mirror the agent NDJSON contract (agent.py) plus the chart's
    loss ring. The page polls ``snapshot()`` once per second.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.state = "idle"          # idle|running|stopping|done|error
        self.phase = None            # active pipeline phase: dataset|model|train|merge
        self.phases = {p: "pending" for p in ("dataset", "model", "train", "merge")}
        self.step = 0
        self.total_steps = 0
        self.epochs = None             # requested epochs (training_started)
        self.pct = None
        self.eta_sec = None
        self.latest_loss = None
        self.latest_acc = None          # DPO runs: rewards/accuracies
        self.best_loss = None
        self.latest_grad_norm = None    # per-step telemetry (agent v6)
        self.learning_rate = None
        self.epoch = None
        self.rate_it_s = None
        self.training_mode = None
        self.base_model = None
        self.dataset = None
        self.stage = None            # batch mode: {"index": i, "total": n}
        self.log = deque(maxlen=MAX_LOG_LINES)
        # Each point: step, loss, grad_norm, lr, epoch (None when absent).
        self.losses = deque(maxlen=MAX_CHART_POINTS)
        # Early-stopping health (agent early_stop_check events).
        self.ema = None
        self.best_ema = None
        self.since_best = None
        self.patience = None
        self.verdict = None
        self.early_stop_enabled = None
        self.beta = None
        self.early_stop_triggered = None
        self.es_checks = deque(maxlen=MAX_ES_CHECKS)  # {step, ema, best_ema}
        # Checkpoints + a compact activity timeline for the UI.
        self.checkpoint = None          # {"step", "path", "ts"}
        self.checkpoint_count = 0
        self.signals = deque(maxlen=MAX_SIGNALS)
        # Training plan and dataset facts for the sidebar.
        self.plan = {}
        self.dataset_info = {}
        self.run_config = {}
        self.error = None
        self.error_code = None
        self.hint = None
        self.exit_code = None
        self.interrupted = False
        self.result = None           # {"merged_dir", "report"} on success
        self.proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []
        self._t_train = None         # wall clock of first step (it/s estimate)
        self._step0 = 0
        self.started_at = None       # wall clock of child launch
        self._reset_training_metrics()

    def _reset_training_metrics(self):
        """Clear per-stage training telemetry without touching run metadata."""
        self.step = 0
        self.pct = None
        self.eta_sec = None
        self.latest_loss = None
        self.latest_acc = None
        self.best_loss = None
        self.latest_grad_norm = None
        self.epoch = None
        self.rate_it_s = None
        self.losses.clear()
        self.ema = None
        self.best_ema = None
        self.since_best = None
        self.patience = None
        self.verdict = None
        self.early_stop_triggered = None
        self.es_checks.clear()
        self.checkpoint = None
        self.checkpoint_count = 0
        self._t_train = None
        self._step0 = 0

    def _signal(self, event, text, tone="info", **extra):
        """Append an activity-timeline entry (caller holds the lock)."""
        entry = {"ts": _now(), "event": event, "text": text, "tone": tone}
        entry.update(extra)
        self.signals.append(entry)

    def add_signal(self, event, text, tone="info", **extra):
        with self._lock:
            self._signal(event, text, tone, **extra)

    # -- mutation ------------------------------------------------------
    def begin(self, proc, base_model, dataset):
        with self._lock:
            self._reset()
            self._threads.clear()
            self.state = "running"
            self.phase = "dataset"
            self.phases["dataset"] = "active"
            self.base_model = base_model
            self.dataset = dataset
            self.proc = proc
            self.started_at = time.time()
            self.log.append(f"[{_now()}] launched: {os.path.basename(sys.executable)} -m reforge --agent …")
            self._signal("run_started", "Run started", "info")

    def add_log(self, line: str):
        with self._lock:
            self.log.append(line.rstrip("\r\n"))

    def request_stop(self):
        """Atomically move running -> stopping; return the child or None.

        The SPA renders a distinct "Stopping (saving)…" state and the
        supervisor refuses new starts while the child is winding down, so
        a second CTRL_BREAK can never interrupt the interrupted-adapter
        save path.
        """
        with self._lock:
            if self.state != "running" or self.proc is None:
                return None
            self.state = "stopping"
            return self.proc

    def cancel_stop(self):
        """Roll back request_stop() when the signal could not be sent."""
        with self._lock:
            if self.state == "stopping":
                self.state = "running"

    def _style_phase(self, event: str):
        """Phase machine — ported from the Tk dashboard's _apply_agent_event."""
        ph = self.phases
        order = ("dataset", "model", "train", "merge")
        if event == "run_started":
            self.phase = "dataset"
            ph["dataset"] = "active"
        elif event == "dataset_loaded":
            ph["dataset"] = "done"
            self.phase = "model"
            ph["model"] = "active"
        elif event == "training_started":
            ph["dataset"] = ph["model"] = "done"
            self.phase = "train"
            ph["train"] = "active"
            self._t_train = None
            self._step0 = 0
            self.rate_it_s = None
        elif event in ("training_complete", "merge_started"):
            ph["train"] = "done"
            self.phase = "merge"
            ph["merge"] = "active"
        elif event == "merge_complete":
            ph["merge"] = "done"
        elif event == "run_complete":
            for p in order:
                ph[p] = "done"
            self.phase = None
            self.pct = 100.0
        elif event in ("run_failed", "training_failed", "batch_failed",
                       "stage_failed"):
            if self.phase:
                ph[self.phase] = "failed"
        elif event == "batch_complete":
            for p in order:
                ph[p] = "done"
            self.phase = None
            self.pct = 100.0

    def apply_event(self, ev: dict):
        """Fold one agent NDJSON event into the snapshot."""
        event = ev.get("event")
        with self._lock:
            if event == "run_started":
                self.run_config = {k: ev.get(k) for k in (
                    "output_dir", "dataset", "hf_dataset", "base_model",
                    "image_mode", "dpo", "dpo_beta", "force", "epoch", "steps")}
            elif event == "dataset_loaded":
                self.training_mode = ev.get("training_mode")
                self.dataset = ev.get("dataset") or self.dataset
                self.base_model = ev.get("base_model") or self.base_model
                self.dataset_info = {k: ev.get(k) for k in (
                    "rows", "eval_rows", "columns", "mapped_roles",
                    "cleaning_mode", "shuffle_seed")}
                rows = ev.get("rows")
                rows_txt = f" · {rows:,} rows" if isinstance(rows, int) else ""
                mode = ev.get("training_mode") or "dataset"
                self._signal("dataset_loaded",
                             f"Dataset ready{rows_txt} · {mode}", "info")
            elif event == "training_started":
                self._reset_training_metrics()
                self.total_steps = int(ev.get("total_steps") or 0)
                self.epochs = ev.get("epochs")
                self.plan = {k: ev.get(k) for k in (
                    "lr", "batch_size", "grad_accum", "early_stop",
                    "scheduler", "warmup_steps", "beta")}
                self.learning_rate = _num(ev.get("lr"))
                self.early_stop_enabled = ev.get("early_stop")
                self.beta = ev.get("beta")
                steps_txt = (f" · {self.total_steps:,} steps"
                             if self.total_steps else "")
                self._signal("training_started",
                             f"Training started{steps_txt}", "info")
            elif event == "step":
                if self._t_train is None:
                    self._t_train = time.time()
                    self._step0 = int(ev.get("step") or 0)
                self.step = int(ev.get("step") or self.step)
                if ev.get("total_steps"):
                    self.total_steps = int(ev["total_steps"])
                self.pct = (float(ev["pct"]) if ev.get("pct") is not None
                            else (100.0 * self.step / self.total_steps
                                  if self.total_steps else None))
                self.eta_sec = ev.get("eta_sec")
                loss = _num(ev.get("loss"))
                if loss is not None:
                    # A diverging run can emit NaN/Inf. Storing it poisons
                    # best_loss and the chart (Math.min -> NaN); drop the
                    # point instead of only sanitizing the JSON response.
                    self.latest_loss = loss
                    self.best_loss = (loss if self.best_loss is None
                                      else min(self.best_loss, loss))
                    grad = _num(ev.get("grad_norm"))
                    if grad is not None:
                        self.latest_grad_norm = grad
                    lr = _num(ev.get("learning_rate"))
                    if lr is not None:
                        self.learning_rate = lr
                    epoch = _num(ev.get("epoch"))
                    if epoch is not None:
                        self.epoch = epoch
                    acc = _num(ev.get("acc"))
                    if acc is not None:
                        self.latest_acc = acc
                    self.losses.append({
                        "step": self.step, "loss": loss,
                        "grad_norm": grad,
                        "lr": self.learning_rate,
                        "epoch": self.epoch,
                    })
                    elapsed = (time.time() - self._t_train
                               if self._t_train else 0)
                    done = self.step - self._step0
                    if elapsed > 0 and done > 0:
                        self.rate_it_s = done / elapsed
            elif event == "early_stop_check":
                self.ema = _num(ev.get("ema"))
                self.best_ema = _num(ev.get("best_ema"))
                self.since_best = ev.get("since_best")
                self.patience = ev.get("patience")
                self.verdict = ev.get("verdict")
                self.es_checks.append({"step": int(ev.get("step") or 0),
                                       "ema": self.ema,
                                       "best_ema": self.best_ema})
                ema_txt = (f"{self.ema:.4f}" if self.ema is not None else "—")
                best_txt = (f"{self.best_ema:.4f}"
                            if self.best_ema is not None else "—")
                self._signal("early_stop_check",
                             f"EMA check · {ema_txt} (best {best_txt}) · "
                             f"{self.verdict or 'continue'}",
                             "warn" if self.verdict == "stop" else "info")
            elif event == "checkpoint_saved":
                step = int(ev.get("step") or 0)
                path = ev.get("path") or ""
                self.checkpoint = {"step": step, "path": path, "ts": ev.get("ts")}
                self.checkpoint_count += 1
                self._signal("checkpoint_saved",
                             f"Checkpoint saved · step {step:,}", "ok",
                             step=step, path=path)
            elif event == "early_stop_triggered":
                self.early_stop_triggered = {
                    k: ev.get(k) for k in (
                        "step", "ema", "best_ema", "steps_total",
                        "hard_cap_steps")}
                step = int(ev.get("step") or 0)
                self._signal("early_stop_triggered",
                             f"Early stop triggered · step {step:,}", "warn")
            elif event == "training_complete":
                self.step = int(ev.get("steps") or self.step)
                result = {"merged_dir": None, "report": ev.get("report"),
                          "final_loss": ev.get("final_loss"),
                          "best_loss": ev.get("best_loss"),
                          "time_sec": ev.get("time_sec")}
                self.result = {**(self.result or {}), **result}
                self._signal("training_complete",
                             "Training complete", "ok")
            elif event == "merge_started":
                self._signal("merge_started", "Merging adapter into base", "info")
            elif event == "merge_complete":
                self.result = {**(self.result or {}),
                               "merged_dir": ev.get("merged_dir"),
                               "report": ev.get("report") or (self.result or {}).get("report")}
                self._signal("merge_complete", "Merge complete", "ok")
            elif event == "run_complete":
                self.result = {**(self.result or {}),
                               "merged_dir": ev.get("merged_dir") or (self.result or {}).get("merged_dir")}
                self._signal("run_complete", "Run complete", "ok")
            elif event in ("run_failed", "training_failed"):
                self.error = ev.get("error") or self.error
                self.error_code = ev.get("error_code") or self.error_code
                self.hint = ev.get("hint") or self.hint
                self.exit_code = ev.get("exit_code") or self.exit_code
                self._signal("run_failed",
                             f"Run failed · {self.error_code or 'error'}", "bad")
            elif event in ("batch_failed", "stage_failed"):
                # Batch children report failures as batch_failed (stage_failed
                # is emitted just before it); without this the results view
                # degraded to a bare "child exited with code 1".
                self.error = ev.get("error") or self.error
                self.error_code = ev.get("error_code") or self.error_code
                self.hint = ev.get("hint") or self.hint
                self.exit_code = ev.get("exit_code") or self.exit_code
                self._signal(event, f"{event.replace('_', ' ').title()}", "bad")
            elif event == "batch_complete":
                self.result = {**(self.result or {}),
                               "merged_dir": (ev.get("final_merged")
                                              or (self.result or {}).get("merged_dir"))}
                self._signal("batch_complete", "Batch complete", "ok")
            elif event in ("run_interrupted", "batch_interrupted"):
                self.interrupted = True
                self._signal(event, "Run interrupted", "warn")
            elif event == "batch_started":
                total = ev.get("total_stages")
                self.stage = {"index": -1, "total": int(total)} if total else None
                if total:
                    self._signal("batch_started",
                                 f"Batch started · {int(total)} stages", "info")
            elif event == "stage_started":
                if self.stage:
                    self.stage = {"index": int(ev.get("index", 0)),
                                  "total": self.stage.get("total")}
                # A new batch stage re-enters the pipeline from the top.
                for p in ("dataset", "model", "train", "merge"):
                    self.phases[p] = "pending"
                self.phase = "dataset"
                self.phases["dataset"] = "active"
                self._reset_training_metrics()
                self.total_steps = 0
                self.error = None
                self.error_code = None
                self.hint = None
                index = int(ev.get("index", 0)) + 1
                total = (self.stage or {}).get("total")
                label = f"Stage {index}" + (f"/{total}" if total else "")
                self._signal("stage_started", f"{label} started", "info")
            elif event == "stage_completed":
                if self.stage:
                    self.stage["index"] = int(ev.get("index", 0))
                index = int(ev.get("index", 0)) + 1
                total = (self.stage or {}).get("total")
                label = f"Stage {index}" + (f"/{total}" if total else "")
                self._signal("stage_completed", f"{label} complete", "ok")
            self._style_phase(event)

    def finish(self, exit_code: int):
        """Child process exited: settle the terminal state."""
        with self._lock:
            if self.state == "stopping":
                if exit_code in _INTERRUPT_EXIT_CODES:
                    self.state = "done"
                    self.interrupted = True
                else:
                    self.state = "done" if exit_code == 0 else "error"
            elif self.state == "running":
                # No terminal event arrived (crash/old backend): derive state.
                self.state = "done" if exit_code == 0 else "error"
            self.exit_code = self.exit_code if self.exit_code is not None else exit_code
            if self.state == "error" and not self.error:
                self.error = f"child exited with code {exit_code}"
            if self.state == "error" and self.phase:
                # Crash with no terminal event: don't leave an active spinner.
                self.phases[self.phase] = "failed"
            self.phases["merge"] = "done" if self.state == "done" else self.phases["merge"]
            if self.state == "error" and not self.signals:
                self._signal("run_failed", f"Child exited with code {exit_code}",
                             "bad")
            self.proc = None

    # -- read ----------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "phase": self.phase,
                "phases": dict(self.phases),
                "step": self.step,
                "total_steps": self.total_steps,
                "epochs": self.epochs,
                "pct": None if self.pct is None else round(self.pct, 2),
                "eta_sec": self.eta_sec,
                "latest_loss": self.latest_loss,
                "latest_acc": self.latest_acc,
                "best_loss": self.best_loss,
                "latest_grad_norm": self.latest_grad_norm,
                "learning_rate": self.learning_rate,
                "epoch": self.epoch,
                "rate_it_s": (round(self.rate_it_s, 3)
                              if self.rate_it_s is not None else None),
                "train_elapsed": (int(time.time() - self._t_train)
                                  if self._t_train is not None else None),
                "run_elapsed": (int(time.time() - self.started_at)
                                if self.started_at is not None else None),
                "training_mode": self.training_mode,
                "base_model": self.base_model,
                "dataset": self.dataset,
                "dataset_info": dict(self.dataset_info),
                "plan": dict(self.plan),
                "run_config": dict(self.run_config),
                "ema": self.ema,
                "best_ema": self.best_ema,
                "since_best": self.since_best,
                "patience": self.patience,
                "verdict": self.verdict,
                "early_stop_enabled": self.early_stop_enabled,
                "early_stop_triggered": self.early_stop_triggered,
                "beta": self.beta,
                "checkpoint": dict(self.checkpoint) if self.checkpoint else None,
                "checkpoint_count": self.checkpoint_count,
                "es_checks": list(self.es_checks),
                "signals": list(self.signals),
                "stage": dict(self.stage) if self.stage else None,
                "log": list(self.log)[-LOG_TAIL:],
                "losses": list(self.losses),
                "error": self.error,
                "error_code": self.error_code,
                "hint": self.hint,
                "exit_code": self.exit_code,
                "interrupted": self.interrupted,
                "result": self.result,
            }


JOB = Job()


# ── Child supervision ────────────────────────────────────────────────
def _reader_stdout(pipe):
    """Tail child stdout: NDJSON agent events -> Job; echo the rest."""
    try:
        for raw in iter(pipe.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            if any(p in line for p in _SUPPRESS_PATTERNS):
                continue
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    obj = json.loads(stripped)
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and "event" in obj:
                    JOB.apply_event(obj)
                    # High-volume per-step events drive the metrics/chart;
                    # echoing them would flood the console pane.
                    if obj.get("event") not in ("step", "early_stop_check"):
                        JOB.add_log(f"» {obj.get('event')}")
                    continue
            if "'loss'" in line or '"loss"' in line:
                continue  # chart data arrives via step events
            JOB.add_log(line)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _reader_stderr(pipe):
    """Tail child stderr (human output in agent mode) into the log pane.

    Raw per-step loss dicts (the LossLineLogger rows) are dropped — they
    flood the pane and the same numbers arrive as `step` events for the
    chart; the stdout reader applies the same rule.
    """
    try:
        for raw in iter(pipe.readline, b""):
            line = raw.decode("utf-8", errors="replace").replace("\r", "")
            if not line.strip():
                continue
            if any(p in line for p in _SUPPRESS_PATTERNS):
                continue
            if "'loss'" in line or '"loss"' in line:
                continue
            JOB.add_log(line)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


_SPAWN_LOCK = threading.Lock()


def _spawn(params: dict) -> tuple[bool, str]:
    """Validate params, build argv, spawn the reforge child with readers."""
    # Serialize the check-then-spawn sequence: ThreadingHTTPServer handles
    # requests concurrently, so two rapid Start clicks could both pass the
    # state check and launch two children into the same output_dir.
    with _SPAWN_LOCK:
        if JOB.state in ("running", "stopping"):
            return False, "A training run is already active."

        if not isinstance(params, dict):
            params = {}
        try:
            cmd = build_argv(params)
        except ValueError as e:
            return False, str(e)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, **kwargs)
        except Exception as e:
            return False, f"Launch failed: {e}"

        dataset = params.get("dataset") or params.get("hf_dataset") or ""
        if isinstance(dataset, list):  # batch stages carry their own datasets
            dataset = ""
        JOB.begin(proc, base_model=params.get("base_model") or "",
                  dataset=str(dataset))

        for target, pipe in ((_reader_stdout, proc.stdout), (_reader_stderr, proc.stderr)):
            t = threading.Thread(target=target, args=(pipe,), daemon=True)
            t.start()
            JOB._threads.append(t)

        threading.Thread(target=_reaper, args=(proc,), daemon=True).start()
        return True, "started"


def _reaper(proc: subprocess.Popen):
    """Wait for the child, then settle the terminal state."""
    code = proc.wait()
    # Give the reader threads a beat to flush their last lines/events.
    for t in JOB._threads:
        t.join(timeout=2.0)
    JOB.finish(code)


def _stop() -> tuple[bool, str]:
    proc = JOB.request_stop()
    if proc is None:
        if JOB.state == "stopping":
            return True, "stop already requested"
        return False, "No active run."
    try:
        if platform.system() == "Windows":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception as e:
        JOB.cancel_stop()   # let the user try again
        return False, f"Could not send interrupt: {e}"
    JOB.add_log(f"[{_now()}] stop requested — child is saving state…")
    return True, "stop requested"


# ── Analysis & report (ports of the Tk dashboard's math, see D9) ─────
def _smooth_series(y, k):
    """Uniform moving average, edge-padded (same as dashboard.smooth_series)."""
    import numpy as np
    k = max(1, int(k))
    y = np.asarray(y, dtype=float)
    if k == 1 or len(y) < 2:
        return y
    kernel = np.ones(k) / k
    pad = k // 2
    ypad = np.pad(y, (pad, k - 1 - pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def _calculate_grade(slope_norm, snr, exposure_pct) -> str:
    """Grade A–D (ported verbatim from dashboard.calculate_grade)."""
    score = 0
    if slope_norm <= -3.0:   score += 4
    elif slope_norm <= -1.5: score += 3
    elif slope_norm <= -0.5: score += 2
    elif slope_norm <= -0.1: score += 1
    if snr >= 3.0:   score += 3
    elif snr >= 2.0: score += 2
    elif snr >= 1.2: score += 1
    if exposure_pct >= 36:   score += 2
    elif exposure_pct >= 18: score += 1
    return "A" if score >= 8 else "B" if score >= 6 else "C" if score >= 4 else "D"


def _trend_label(norm: float) -> str:
    if norm <= -3.0:     return "Strong Drop"
    elif norm <= -1.0:   return "Moderate Drop"
    elif norm < -0.1:    return "Weak Drop"
    elif abs(norm) <= 0.1: return "Flat"
    elif norm < 1.0:     return "Weak Rise"
    elif norm < 3.0:     return "Moderate Rise"
    return "Strong Rise"


def _analysis() -> dict:
    """Full loss analysis over the live loss ring (Tk _full_analysis port).

    Step events carry the real fractional `epoch` since agent schema v6.
    Older/partial streams fall back to a linear reconstruction from
    training_started's `epochs` and the step/total ratio.
    """
    try:
        import numpy as np
    except ImportError:
        return {"ok": False, "error": "numpy unavailable"}

    losses = list(JOB.snapshot()["losses"])
    if len(losses) < 3:
        return {"ok": False, "error": "Need at least 3 loss points."}

    steps = [float(e["step"]) for e in losses]
    y = np.array([float(e["loss"]) for e in losses])
    n = len(y)

    snap = JOB.snapshot()
    epochs_total = snap.get("epochs") or 0.0
    total = snap.get("total_steps") or 0
    # Prefer the real per-step epoch; reconstruct only when it is absent.
    epochs = []
    for entry, step in zip(losses, steps):
        epoch = _num(entry.get("epoch"))
        if epoch is None:
            epoch = (epochs_total * (step / total)
                     if epochs_total and total else 0.0)
        epochs.append(float(epoch))

    last_ep = epochs[-1]
    exposure = (last_ep - int(last_ep)) if last_ep >= 1.0 else last_ep
    exposure_pct = round(exposure * 100.0, 2)

    sk = 7 if n >= 7 else max(3, (n // 3) * 2 + 1)
    smoothed = _smooth_series(y, sk)

    y_mean, y_med = float(np.mean(y)), float(np.median(y))
    y_std = float(np.std(y))
    y_max = float(np.max(y))
    drop_pct = 100.0 * (y_max - float(np.min(y))) / max(y_max, 1e-8)

    x = np.array(steps)
    gslope_n = float(np.polyfit(x, smoothed, 1)[0]) * 100.0

    win = min(max(10, n // 10), 200)
    xw, yw = x[-win:], smoothed[-win:]
    w = np.linspace(0.5, 1.0, num=len(xw))
    slope = float(np.polyfit(xw, yw, 1, w=w)[0])
    rslope_n = slope * 100.0
    r_med = float(np.median(yw))
    r_mad = float(1.4826 * np.median(np.abs(yw - r_med)))
    r_noise = r_mad if r_mad > 1e-8 else float(np.std(yw))
    r_snr = float(abs(slope) / (r_noise + 1e-8))

    best_i = int(np.argmin(y))
    best_v = float(y[best_i])
    since_best = n - (best_i + 1)

    # Degenerate series (e.g. duplicate step values make polyfit singular)
    # can produce NaN/inf. Serializing that as JSON null would crash the
    # browser's .toFixed() calls; report "unavailable" instead.
    if not all(math.isfinite(float(v)) for v in
               (y_mean, y_med, y_std, drop_pct, gslope_n, rslope_n, r_snr,
                best_v)):
        return {"ok": False, "error": "Loss analysis unavailable (non-finite values)."}

    if slope < -1e-6:
        to_t = max(0, int((y[-1] - (best_v + 0.05)) / -slope))
        pred = f"step {int(steps[-1]) + to_t} (~{to_t} more)"
    else:
        pred = "—"

    grade = _calculate_grade(rslope_n, r_snr, exposure_pct)

    outliers = []
    if len(yw) > 1:
        sd = float(np.std(yw))
        if sd > 0:
            for j in range(win):
                idx = n - win + j
                if 0 < idx < n and abs(float(y[idx] - smoothed[idx])) > sd * 1.5:
                    outliers.append(int(steps[idx]))

    epoch_lines = []
    for j in range(1, n):
        if int(epochs[j - 1] * 100) != int(epochs[j] * 100) and j > 0:
            epoch_lines.append(int(steps[j]))

    return {
        "ok": True,
        "n": n, "smooth_k": sk,
        "avg": round(y_mean, 4), "med": round(y_med, 4), "std": round(y_std, 4),
        "best": round(best_v, 4), "best_step": int(steps[best_i]),
        "since_best": since_best,
        "grade": grade, "snr": round(r_snr, 2),
        "slope_global": round(gslope_n, 2), "slope_recent": round(rslope_n, 2),
        "trend_global": _trend_label(gslope_n), "trend_recent": _trend_label(rslope_n),
        "drop_pct": round(drop_pct, 1), "exposure_pct": exposure_pct,
        "latest_epoch": round(epochs[-1], 4),
        "ideal_zone": pred, "outliers": outliers, "epoch_lines": epoch_lines,
    }


_REPORT_CACHE = {"key": None, "data": None}


def _read_report() -> dict:
    """training_report.json from the finished run (cached per file version).

    The key includes mtime/size so a second run into the same output_dir
    cannot serve the previous run's cached report.
    """
    path = (JOB.snapshot().get("result") or {}).get("report")
    if not path:
        return {"ok": False, "error": "No report available."}
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        if _REPORT_CACHE["key"] == key and _REPORT_CACHE["data"] is not None:
            return _REPORT_CACHE["data"]
        with open(path, "r", encoding="utf-8") as f:
            import json as _json
            data = {"ok": True, "report": _json.load(f)}
    except Exception as e:
        return {"ok": False, "error": f"Could not read report: {e}"}
    _REPORT_CACHE["key"], _REPORT_CACHE["data"] = key, data
    return data


def _finite_json(obj):
    """Replace NaN/Infinity with None so the browser's JSON.parse succeeds.

    A diverging run can log a NaN loss; Python's json.dumps emits a bare
    ``NaN`` token which is invalid JSON, and one such status poll would
    break the whole SPA (it falls back to "server unreachable").
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_json(v) for v in obj]
    return obj


# ── Filesystem browsing (localhost-only) ─────────────────────────────
def _clean_path(path: str) -> str:
    raw = str(path or "").strip().strip('"')
    if not raw:
        return os.path.expanduser("~")
    return os.path.abspath(os.path.expandvars(os.path.expanduser(raw)))


def _places() -> dict:
    """Favorites + drive roots shown in the folder-picker sidebar."""
    home = os.path.expanduser("~")
    shortcuts = []
    for label, path, kind in (
        ("Home", home, "home"),
        ("Desktop", os.path.join(home, "Desktop"), "desktop"),
        ("Documents", os.path.join(home, "Documents"), "documents"),
        ("Downloads", os.path.join(home, "Downloads"), "downloads"),
        ("Models", DEFAULT_HF_MODELS_DIR, "models"),
        ("Reforge output", OUTPUT_DIR, "output"),
    ):
        try:
            if os.path.isdir(path):
                shortcuts.append({"label": label, "path": path, "kind": kind})
        except OSError:
            pass
    drives = []
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            try:
                if os.path.exists(root):
                    drives.append({"label": f"{letter}:", "path": root,
                                   "kind": "drive"})
            except OSError:
                pass
    return {
        "home": home,
        "shortcuts": shortcuts,
        "drives": drives,
        "default_output": OUTPUT_DIR,
        "models_dir": DEFAULT_HF_MODELS_DIR,
    }


def _crumbs(path: str) -> list:
    """Breadcrumb segments for a path, root first."""
    cur = os.path.abspath(path)
    parts = []
    seen = set()
    while cur not in seen:
        seen.add(cur)
        label = os.path.basename(cur.rstrip("\\/")) or cur
        parts.append({"label": label, "path": cur})
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        cur = parent
    return list(reversed(parts))


def _browse(path: str = "", mode: str = "dir", filter_ext: str = "",
            show_all: bool = False, show_hidden: bool = False) -> dict:
    """One directory listing for the SPA folder picker.

    ``mode="file"`` includes files (filtered by ``filter_ext`` unless
    ``show_all``); ``mode="dir"`` lists folders only. Errors are returned
    as ``{ok: false}`` so the picker can show them without a 500.
    """
    mode = "file" if mode == "file" else "dir"
    target = _clean_path(path)
    if os.path.isfile(target):
        target = os.path.dirname(target)
    base = {**_places(), "mode": mode, "filter": filter_ext,
            "show_all": show_all, "show_hidden": show_hidden}
    if not os.path.isdir(target):
        return {"ok": False, "error": f"Not a folder: {target}",
                "path": target, "parent": os.path.dirname(target),
                "crumbs": _crumbs(target) if target else [], **base}

    entries = []
    truncated = False
    try:
        with os.scandir(target) as it:
            for entry in it:
                if len(entries) >= MAX_BROWSE_ENTRIES:
                    truncated = True
                    break
                name = entry.name
                if not show_hidden and name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                if not is_dir:
                    if mode != "file":
                        continue
                    if (filter_ext and not show_all
                            and not name.lower().endswith(filter_ext.lower())):
                        continue
                is_model = False
                if is_dir:
                    try:
                        is_model = os.path.isfile(
                            os.path.join(entry.path, "config.json"))
                    except OSError:
                        pass
                size = mtime = None
                try:
                    st = entry.stat()
                    mtime = st.st_mtime
                    if not is_dir:
                        size = st.st_size
                except OSError:
                    pass
                entries.append({
                    "name": name,
                    "path": entry.path,
                    "is_dir": is_dir,
                    "is_model": is_model,
                    "size": size,
                    "mtime": mtime,
                })
    except OSError as e:
        return {"ok": False, "error": f"Could not read folder: {e}",
                "path": target, "parent": os.path.dirname(target),
                "crumbs": _crumbs(target), **base}

    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    parent = os.path.dirname(target)
    if parent == target:
        parent = None
    return {
        "ok": True,
        "path": target,
        "parent": parent,
        "crumbs": _crumbs(target),
        "entries": entries,
        "truncated": truncated,
        **base,
    }


def _mkdir(parent: str, name: str) -> dict:
    parent = _clean_path(parent)
    name = str(name or "").strip()
    if not name:
        return {"ok": False, "message": "Enter a folder name."}
    if name in (".", "..") or any(ch in name for ch in ("/", "\\")):
        return {"ok": False, "message": "Folder names cannot contain slashes."}
    if os.name == "nt" and (":" in name or name.endswith((" ", "."))):
        return {"ok": False,
                "message": "Windows folder names cannot contain ':' or end "
                           "with a space/dot."}
    if not os.path.isdir(parent):
        return {"ok": False, "message": f"Parent folder not found: {parent}"}
    target = os.path.join(parent, name)
    try:
        os.mkdir(target)
    except FileExistsError:
        return {"ok": False, "message": "A folder with that name already exists."}
    except OSError as e:
        return {"ok": False, "message": f"Could not create folder: {e}"}
    return {"ok": True, "path": target}


def _reveal(path: str) -> dict:
    target = _clean_path(path)
    if not os.path.exists(target):
        return {"ok": False, "message": f"Path not found: {target}"}
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(target)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception as e:
        return {"ok": False, "message": f"Could not open: {e}"}
    return {"ok": True, "message": "Opened."}


def _client_is_local(handler) -> bool:
    """Filesystem APIs are for the local operator, not network clients."""
    addr = (handler.client_address or ("",))[0]
    return (addr in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1")
            or addr.startswith("127.") or addr.startswith("::ffff:127."))


# ── HTTP handler ─────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console clean
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        payload = json.dumps(_finite_json(obj), allow_nan=False).encode("utf-8")
        self._send(code, payload, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            return {}
        if length <= 0 or length > 8 * 1024 * 1024:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        parsed = urlsplit(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"),
                       "text/html; charset=utf-8")
        elif route == "/api/status":
            self._send_json(JOB.snapshot())
        elif route == "/api/models":
            self._send_json({"models": scan_models()})
        elif route == "/api/places":
            if not _client_is_local(self):
                self._send_json({"ok": False, "error": "Local access only."}, 403)
                return
            self._send_json(_places())
        elif route == "/api/browse":
            if not _client_is_local(self):
                self._send_json({"ok": False, "error": "Local access only."}, 403)
                return
            q = parse_qs(parsed.query)
            self._send_json(_browse(
                path=(q.get("path") or [""])[0],
                mode=(q.get("mode") or ["dir"])[0],
                filter_ext=(q.get("filter") or [""])[0],
                show_all=(q.get("all") or ["0"])[0] in ("1", "true", "yes"),
                show_hidden=(q.get("hidden") or ["0"])[0] in ("1", "true", "yes"),
            ))
        elif route == "/api/analysis":
            self._send_json(_analysis())
        elif route == "/api/report":
            self._send_json(_read_report())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        route = urlsplit(self.path).path
        if route == "/api/start":
            ok, msg = _spawn(self._read_json())
            self._send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif route == "/api/stop":
            ok, msg = _stop()
            self._send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif route == "/api/preview":
            params = self._read_json()
            try:
                self._send_json({"ok": True, "argv": build_argv(params, strict=False)})
            except ValueError as e:
                self._send_json({"ok": False, "message": str(e)}, 200)
        elif route == "/api/mkdir":
            if not _client_is_local(self):
                self._send_json({"ok": False, "message": "Local access only."}, 403)
                return
            body = self._read_json()
            self._send_json(_mkdir(body.get("parent") or "",
                                   body.get("name") or ""))
        elif route == "/api/reveal":
            if not _client_is_local(self):
                self._send_json({"ok": False, "message": "Local access only."}, 403)
                return
            body = self._read_json()
            self._send_json(_reveal(body.get("path") or ""))
        else:
            self._send_json({"error": "not found"}, 404)


def _report_environment() -> None:
    print(f"  python        {sys.version.split()[0]}\n"
          f"  interpreter   {sys.executable}\n")


# ── Launch ───────────────────────────────────────────────────────────
def launch(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
           open_browser: bool = True) -> None:
    """Start the web UI server (blocks until Ctrl-C)."""
    _enable_utf8()
    _title("REFORGE", subtitle="Web dashboard")
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"\n  Reforge Web UI is live:  {url}\n")
    _report_environment()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _info("Shutting down.")
        if JOB.proc is not None:
            _stop()
            # Do not orphan the child: it lives in its own process group, so
            # it would keep training after this server exits. Give it time to
            # save and finish its export, then kill a stuck child (resume from
            # the checkpoint still works). Poll JOB.proc instead of calling
            # Popen.wait() here: the reaper thread owns that wait and settles
            # the job (and clears proc) when the child exits.
            deadline = time.time() + 120
            while JOB.proc is not None and time.time() < deadline:
                time.sleep(0.2)
            stuck = JOB.proc
            if stuck is not None:
                _warn("Child did not exit in time — terminating it.")
                try:
                    stuck.kill()
                    stuck.wait(timeout=10)
                except Exception:
                    pass
    finally:
        server.server_close()


def _enable_utf8():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
