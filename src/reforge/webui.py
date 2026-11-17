"""Local web UI for Reforge — a browser dashboard in the style of Lethe's UI.

Zero third-party dependencies: a stdlib ``http.server`` serves one
self-contained page and a small JSON API. Launch with ``reforge webui``
(alias ``reforge ui``) and open http://127.0.0.1:7861.

Architecture (see docs/WEBUI_MIGRATION.md, decisions D1-D11):
    The server is a thin supervisor. ``POST /api/start`` validates the
    config and spawns ``python -m reforge --agent ...`` as a child process
    (same isolation model as the Tk dashboard). Reader threads tail the
    child's stdout (NDJSON agent events) and stderr (human log + loss
    rows); events update a thread-safe ``Job`` snapshot that the page
    polls every second. ``POST /api/stop`` sends CTRL_BREAK/SIGINT for a
    graceful checkpoint save.

Endpoints:
    GET  /              the single-page app
    GET  /api/status    current job state (polled by the page)
    POST /api/start     begin a training job   -> {ok} | {error}
    POST /api/stop      graceful stop (checkpoint save)

Phases (WIP skeleton): config form and progress rendering land in Phases
3-4. /api/status already carries the full event-driven state machine.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .display import _info, _title
from .webui_config import build_argv, scan_models

# ── Tunables ─────────────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7861          # 7860 is Lethe's default; both may run side by side
MAX_LOG_LINES = 400          # ring buffer for the console pane
LOG_TAIL = 80                # lines sent per status poll
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
        self.best_loss = None
        self.rate_it_s = None
        self.training_mode = None
        self.base_model = None
        self.dataset = None
        self.stage = None            # batch mode: {"index": i, "total": n}
        self.log = deque(maxlen=MAX_LOG_LINES)
        self.losses = deque(maxlen=MAX_CHART_POINTS)   # {"step": n, "loss": f}
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

    # -- mutation ------------------------------------------------------
    def begin(self, proc, base_model, dataset):
        with self._lock:
            self._reset()
            self.state = "running"
            self.phase = "dataset"
            self.phases["dataset"] = "active"
            self.base_model = base_model
            self.dataset = dataset
            self.proc = proc
            self.log.append(f"[{_now()}] launched: {os.path.basename(sys.executable)} -m reforge --agent …")

    def add_log(self, line: str):
        with self._lock:
            self.log.append(line.rstrip("\n"))

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
        elif event == "run_failed":
            if self.phase:
                ph[self.phase] = "failed"

    def apply_event(self, ev: dict):
        """Fold one agent NDJSON event into the snapshot."""
        event = ev.get("event")
        with self._lock:
            if event == "run_started":
                pass  # base_model/dataset already known from argv
            elif event == "dataset_loaded":
                self.training_mode = ev.get("training_mode")
                self.dataset = ev.get("dataset") or self.dataset
                self.base_model = ev.get("base_model") or self.base_model
            elif event == "training_started":
                self.total_steps = int(ev.get("total_steps") or 0)
                self.epochs = ev.get("epochs")
                self.step = 0
                self.pct = None
                self.eta_sec = None
            elif event == "step":
                if self._t_train is None:
                    import time
                    self._t_train = time.time()
                    self._step0 = int(ev.get("step") or 0)
                self.step = int(ev.get("step") or self.step)
                if ev.get("total_steps"):
                    self.total_steps = int(ev["total_steps"])
                self.pct = (float(ev["pct"]) if ev.get("pct") is not None
                            else (100.0 * self.step / self.total_steps
                                  if self.total_steps else None))
                self.eta_sec = ev.get("eta_sec")
                loss = ev.get("loss")
                if loss is not None:
                    self.latest_loss = float(loss)
                    self.best_loss = (float(loss) if self.best_loss is None
                                      else min(self.best_loss, float(loss)))
                    self.losses.append({"step": self.step, "loss": float(loss)})
                    import time
                    elapsed = time.time() - self._t_train if self._t_train else 0
                    done = self.step - self._step0
                    if elapsed > 0 and done > 0:
                        self.rate_it_s = done / elapsed
            elif event == "training_complete":
                self.step = int(ev.get("steps") or self.step)
                result = {"merged_dir": None, "report": ev.get("report"),
                          "final_loss": ev.get("final_loss"),
                          "best_loss": ev.get("best_loss"),
                          "time_sec": ev.get("time_sec")}
                self.result = {**(self.result or {}), **result}
            elif event == "merge_complete":
                self.result = {**(self.result or {}),
                               "merged_dir": ev.get("merged_dir"),
                               "report": ev.get("report") or (self.result or {}).get("report")}
            elif event == "run_complete":
                self.result = {**(self.result or {}),
                               "merged_dir": ev.get("merged_dir") or (self.result or {}).get("merged_dir")}
            elif event == "run_failed":
                self.error = ev.get("error")
                self.error_code = ev.get("error_code")
                self.hint = ev.get("hint")
                self.exit_code = ev.get("exit_code")
            elif event == "run_interrupted":
                self.interrupted = True
            elif event == "batch_started":
                total = ev.get("total_stages")
                self.stage = {"index": -1, "total": int(total)} if total else None
            elif event == "stage_started":
                if self.stage:
                    self.stage = {"index": int(ev.get("index", 0)),
                                  "total": self.stage.get("total")}
                # A new batch stage re-enters the pipeline from the top.
                for p in ("dataset", "model", "train", "merge"):
                    self.phases[p] = "pending"
                self.phase = "dataset"
                self.phases["dataset"] = "active"
                self.step = 0
                self.total_steps = 0
                self.pct = None
                self.eta_sec = None
                self.losses.clear()
                self.error = None
                self.error_code = None
            # batch_completed stage tracking
            elif event == "stage_completed":
                if self.stage:
                    self.stage["index"] = int(ev.get("index", 0))
            self._style_phase(event)

    def finish(self, exit_code: int):
        """Child process exited: settle the terminal state."""
        with self._lock:
            if self.state == "stopping":
                self.state = "done" if exit_code in (0, 130) else "error"
            elif self.state == "running":
                # No terminal event arrived (crash/old backend): derive state.
                self.state = "done" if exit_code == 0 else "error"
            self.exit_code = self.exit_code if self.exit_code is not None else exit_code
            if self.state == "error" and not self.error:
                self.error = f"child exited with code {exit_code}"
            self.phases["merge"] = "done" if self.state == "done" else self.phases["merge"]
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
                "best_loss": self.best_loss,
                "rate_it_s": (round(self.rate_it_s, 3)
                              if self.rate_it_s is not None else None),
                "train_elapsed": (int(time.time() - self._t_train)
                                  if self._t_train is not None else None),
                "training_mode": self.training_mode,
                "base_model": self.base_model,
                "dataset": self.dataset,
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
    """Tail child stderr (human output in agent mode) into the log pane."""
    try:
        for raw in iter(pipe.readline, b""):
            line = raw.decode("utf-8", errors="replace").replace("\r", "")
            if not line.strip():
                continue
            if any(p in line for p in _SUPPRESS_PATTERNS):
                continue
            JOB.add_log(line)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _spawn(params: dict) -> tuple[bool, str]:
    """Validate params, build argv, spawn the reforge child with readers."""
    if JOB.state == "running":
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
    proc = JOB.proc
    if not proc:
        return False, "No active run."
    try:
        if platform.system() == "Windows":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception as e:
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

    The agent's `step` events carry no epoch, so the epoch schedule is
    reconstructed linearly from training_started's `epochs` — the same
    value the Tk analysis window read from the stderr loss rows.
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
    # Reconstructed per-point epoch: E * step/total (≈ what the stderr rows log)
    epochs = [epochs_total * (s / total) for s in steps] if epochs_total and total else [0.0] * n

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
        "ideal_zone": pred, "outliers": outliers, "epoch_lines": epoch_lines,
    }


_REPORT_CACHE = {"path": None, "data": None}


def _read_report() -> dict:
    """training_report.json from the finished run (cached per path)."""
    path = (JOB.snapshot().get("result") or {}).get("report")
    if not path:
        return {"ok": False, "error": "No report available."}
    if _REPORT_CACHE["path"] == path and _REPORT_CACHE["data"] is not None:
        return _REPORT_CACHE["data"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            import json as _json
            data = {"ok": True, "report": _json.load(f)}
    except Exception as e:
        return {"ok": False, "error": f"Could not read report: {e}"}
    _REPORT_CACHE["path"], _REPORT_CACHE["data"] = path, data
    return data


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
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send_json(JOB.snapshot())
        elif self.path == "/api/models":
            self._send_json({"models": scan_models()})
        elif self.path == "/api/analysis":
            self._send_json(_analysis())
        elif self.path == "/api/report":
            self._send_json(_read_report())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/start":
            ok, msg = _spawn(self._read_json())
            self._send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif self.path == "/api/stop":
            ok, msg = _stop()
            self._send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif self.path == "/api/preview":
            params = self._read_json()
            try:
                self._send_json({"ok": True, "argv": build_argv(params, strict=False)})
            except ValueError as e:
                self._send_json({"ok": False, "message": str(e)}, 200)
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
        if JOB.proc:
            _stop()
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


# ── Single-page app (Phases 3-5: config, progress, chart, results) ──
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reforge</title>
<style>
  :root{
    --bg:#101014; --card:#1a1a2e; --card2:#202032; --line:#33333f;
    --ink:#FFF8DC; --muted:#9096ad; --accent:#FFBF00; --accent2:#FFD700;
    --ok:#4caf50; --warn:#ffa726; --bad:#ef5350;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.5 "Segoe UI",system-ui,sans-serif}
  .wrap{max-width:1100px;margin:0 auto;padding:20px 20px 90px}
  header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:18px}
  h1{color:var(--accent);margin:0;font-size:26px;letter-spacing:1px}
  .chip{display:inline-flex;align-items:center;gap:7px;padding:3px 12px;
    border:1px solid var(--line);border-radius:20px;font-size:13px;color:var(--muted)}
  .chip .dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
  .chip.run .dot{background:var(--accent);animation:pulse 1.1s infinite}
  .chip.stop .dot{background:var(--warn)}
  .chip.done .dot{background:var(--ok)}
  .chip.err  .dot{background:var(--bad)}
  @keyframes pulse{50%{opacity:.35}}
  .spacer{flex:1}
  button{background:var(--card2);color:var(--ink);border:1px solid var(--line);
    border-radius:8px;padding:8px 18px;font-size:14px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.accent{background:var(--accent);border-color:var(--accent);
    color:#141414;font-weight:600}
  button.accent:hover{background:var(--accent2)}
  button.ghost{background:transparent}
  button:disabled{opacity:.45;cursor:not-allowed}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
    padding:16px 18px;margin-bottom:16px}
  .card h2{margin:0 0 12px;font-size:14px;color:var(--accent);
    text-transform:uppercase;letter-spacing:1.5px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}
  input[type=text],input[type=number]{width:100%;background:var(--card2);
    border:1px solid var(--line);border-radius:8px;color:var(--ink);
    padding:8px 10px;font-size:14px}
  input:focus{outline:none;border-color:var(--accent)}
  .checks{display:flex;gap:18px;flex-wrap:wrap;margin-top:6px}
  .checks label{display:flex;align-items:center;gap:7px;font-size:14px;
    color:var(--ink);margin:0}
  .checks input{width:auto}
  .hint{color:var(--muted);font-size:12.5px;margin-top:8px}
  pre{background:var(--card2);border:1px solid var(--line);border-radius:9px;
    padding:12px;font:12.5px/1.5 Consolas,monospace;color:var(--muted);
    white-space:pre-wrap;max-height:260px;overflow:auto;margin:0}
  .banner{border:1px solid var(--bad);background:rgba(239,83,80,.08);
    border-radius:10px;padding:12px 16px;margin-bottom:16px;white-space:pre-wrap}
  .banner.warn{border-color:var(--warn);background:rgba(255,167,38,.08)}
  /* phase tracker */
  .phases{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:14px}
  .ph{display:flex;align-items:center;gap:8px;padding:6px 14px;border:1px solid
    var(--line);border-radius:20px;font-size:12.5px;letter-spacing:1px;
    color:var(--muted)}
  .ph.active{border-color:var(--accent);color:var(--accent)}
  .ph.done{border-color:var(--ok);color:var(--ok)}
  .ph.failed{border-color:var(--bad);color:var(--bad)}
  .ph .spin{width:11px;height:11px;border:2px solid var(--accent);
    border-top-color:transparent;border-radius:50%;animation:rot .8s linear infinite}
  @keyframes rot{to{transform:rotate(360deg)}}
  .arrow{color:var(--line)}
  /* progress bar */
  .bar{position:relative;height:22px;background:var(--card2);border:1px solid
    var(--line);border-radius:11px;overflow:hidden;
    background-image:repeating-linear-gradient(90deg,transparent 0 10%,rgba(255,255,255,.05) 10% 10.2%)}
  .bar i{position:absolute;inset:0 auto 0 0;width:0;background:linear-gradient(90deg,#b58a00,var(--accent));
    transition:width .6s ease;border-radius:11px}
  .bar.indet i{width:18%;animation:sweep 1.15s ease-in-out infinite}
  @keyframes sweep{0%{left:-18%}100%{left:100%}}
  .bar b{position:absolute;right:10px;top:0;line-height:22px;font-size:12px;
    color:var(--ink);text-shadow:0 0 3px #000}
  .tqdm{font:13px Consolas,monospace;color:var(--muted);margin:8px 0 14px}
  .stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
  .stat{background:var(--card2);border:1px solid var(--line);border-radius:10px;
    padding:7px 14px;min-width:100px}
  .stat small{display:block;font-size:10.5px;color:var(--muted);letter-spacing:1px}
  .stat span{font:15px Consolas,monospace;color:var(--ink)}
  .stat.gold span{color:var(--accent)}
  .stagechip{background:rgba(255,191,0,.1);border-color:var(--accent);color:var(--accent)}
  canvas#chart{width:100%;height:230px;display:block;background:var(--card2);
    border:1px solid var(--line);border-radius:10px}
  .analysis{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
  .alog{font:12px/1.6 Consolas,monospace;color:var(--muted);margin-top:10px}
  .logpane{max-height:300px}
  .logpane .bar2{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
  .logpane label{display:flex;align-items:center;gap:6px;margin:0;color:var(--muted);font-size:12px}
  /* batch editor */
  .stage{display:grid;grid-template-columns:34px 1fr 1fr 90px 90px 30px 30px 30px;
    gap:8px;align-items:center;margin-bottom:8px}
  .stage .n{color:var(--muted);font:13px Consolas,monospace;text-align:center}
  .stage input{padding:6px 8px;font-size:13px}
  .stage button{padding:3px 8px;font-size:12px}
  .ibtn{background:transparent;border:1px solid var(--line);border-radius:6px;
    color:var(--muted);cursor:pointer}
  .ibtn:hover{color:var(--accent);border-color:var(--accent)}
  /* results */
  .res{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
  .res .stat{min-height:74px}
  .kv{font:13px/1.8 Consolas,monospace;color:var(--muted);white-space:pre-wrap}
  a{color:var(--accent)}
  [hidden]{display:none !important}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>REFORGE</h1>
    <span class="chip" id="chip"><span class="dot"></span><span id="chip-text">Idle</span></span>
    <span class="spacer"></span>
    <button id="btn-stop" hidden>Stop</button>
    <button id="btn-start" class="accent" hidden>Start training</button>
  </header>

  <div id="msg-banner" class="banner" hidden></div>

  <!-- ── Setup ─────────────────────────────────────────────────────── -->
  <section id="view-setup">
    <div class="card">
      <h2>Paths</h2>
      <div class="grid">
        <div><label>Base model (local path)</label>
          <input type="text" id="p-base" list="models" placeholder="C:\HF_Models\…">
          <datalist id="models"></datalist></div>
        <div><label>Output directory</label>
          <input type="text" id="p-out" placeholder="~/Reforge_Output"></div>
        <div><label>Dataset (.parquet)</label>
          <input type="text" id="p-ds" placeholder="C:\data\train.parquet"></div>
        <div><label>…or HuggingFace dataset ID</label>
          <input type="text" id="p-hf" placeholder="org/dataset"></div>
      </div>
    </div>
    <div class="card">
      <h2>Training &amp; performance</h2>
      <div class="grid">
        <div><label>Epochs</label><input type="text" id="p-epoch" placeholder="1"></div>
        <div><label>Fixed steps (overrides epochs)</label><input type="text" id="p-steps"></div>
        <div><label>Batch size</label><input type="text" id="p-batch"></div>
        <div><label>Grad. accumulation</label><input type="text" id="p-grad"></div>
      </div>
      <div class="checks">
        <label><input type="checkbox" id="p-force"> Force (no early stop)</label>
        <label><input type="checkbox" id="p-lmhead"> Train lm_head</label>
      </div>
    </div>
    <div class="card">
      <h2>Flags &amp; DPO</h2>
      <div class="checks">
        <label><input type="checkbox" id="p-latex"> Preserve LaTeX</label>
        <label><input type="checkbox" id="p-code"> Preserve code</label>
        <label><input type="checkbox" id="p-image"> Multimodal (image)</label>
        <label><input type="checkbox" id="p-dpo"> DPO</label>
      </div>
      <div class="grid" style="margin-top:10px">
        <div><label>DPO beta</label><input type="text" id="p-dpo-beta" placeholder="0.1"></div>
      </div>
      <div class="hint">DPO needs chosen/rejected columns. SFT→DPO chains: leave
        base model empty after an SFT merge and reforge chains from the merged dir.</div>
    </div>
    <div class="card">
      <h2>Batch mode</h2>
      <div class="checks">
        <label><input type="checkbox" id="p-batchmode"> Enable batch (stages below)</label>
      </div>
      <div id="batch-editor" hidden style="margin-top:12px">
        <div class="stage" style="font-size:11px;color:var(--muted)">
          <span></span><span>name (opt.)</span><span>dataset / hf id</span>
          <span>epochs</span><span>steps</span><span></span><span></span><span></span>
        </div>
        <div id="stage-list"></div>
        <button class="ghost" id="btn-addstage">+ Add stage</button>
        <div class="hint">Stage 1's base model is the Base model field above;
          later stages chain from the previous stage's merged output.</div>
      </div>
    </div>
    <div class="card">
      <h2>Command preview</h2>
      <pre id="cmd-preview">fill in the fields…</pre>
    </div>
  </section>

  <!-- ── Progress ──────────────────────────────────────────────────── -->
  <section id="view-progress" hidden>
    <div class="card">
      <div class="phases" id="phases"></div>
      <div class="bar indet" id="bar"><i></i><b id="bar-cap"></b></div>
      <div class="tqdm" id="tqdm">waiting for the child…</div>
      <div class="stats" id="stats"></div>
      <canvas id="chart"></canvas>
      <div class="analysis" id="analysis"></div>
      <div class="alog" id="alog"></div>
    </div>
    <div class="card logpane">
      <div class="bar2">
        <h2 style="margin:0">Console</h2>
        <label><input type="checkbox" id="log-follow" checked> follow</label>
      </div>
      <pre id="log"></pre>
    </div>
  </section>

  <!-- ── Results ───────────────────────────────────────────────────── -->
  <section id="view-results" hidden>
    <div class="card">
      <h2 id="res-title">Result</h2>
      <div id="res-body"></div>
      <div style="margin-top:14px;display:flex;gap:10px">
        <button class="accent" id="btn-again">Start another run</button>
      </div>
    </div>
    <div class="card logpane">
      <div class="bar2"><h2 style="margin:0">Console tail</h2></div>
      <pre id="res-log"></pre>
    </div>
  </section>
</div>

<script>
"use strict";
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const PHASES = ["dataset", "model", "train", "merge"];
const PHASE_LABEL = {dataset:"DATASET", model:"MODEL", train:"TRAIN", merge:"MERGE"};

let last = null;          // last /api/status snapshot
let analysis = null;      // last /api/analysis
let previewTimer = null;
let anaTimer = 0;

function fmtHMS(sec){
  if (sec === null || sec === undefined || isNaN(sec)) return "--:--";
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = sec%60;
  const p = n => String(n).padStart(2, "0");
  return h ? `${h}h${p(m)}m${p(s)}s` : `${p(m)}:${p(s)}`;
}

async function jget(url){ const r = await fetch(url); return r.json(); }
async function jpost(url, body){
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"},
                              body: JSON.stringify(body ?? {})});
  let data = {}; try{ data = await r.json(); }catch(e){}
  return {code: r.status, data};
}

/* ── params & preview ──────────────────────────────────────────────── */
function stages(){
  return window._stages || [];
}
function collectParams(){
  const v = id => $(id).value.trim();
  const c = id => $(id).checked;
  const p = {
    base_model: v("p-base"), output_dir: v("p-out"),
    dataset: v("p-ds"), hf_dataset: v("p-hf"),
    epoch: v("p-epoch"), steps: v("p-steps"),
    batch_size: v("p-batch"), grad_accum: v("p-grad"),
    force: c("p-force"), latex: c("p-latex"), code: c("p-code"),
    image: c("p-image"), dpo: c("p-dpo"), train_lm_head: c("p-lmhead"),
    dpo_beta: v("p-dpo-beta"),
  };
  if (c("p-batchmode")) p.stages = stages();
  for (const k of Object.keys(p)) if (p[k] === "" || p[k] === false) delete p[k];
  return p;
}
function schedulePreview(){
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try{
      const {data} = await jpost("/api/preview", collectParams());
      let text;
      if (data.ok){
        // group flag + value pairs, backslash-continued like the Tk preview
        const a = data.argv.slice(2);       // hide the interpreter prefix
        const parts = [];
        for (let i = 0; i < a.length; i++){
          if (a[i].startsWith("--") && i+1 < a.length && !a[i+1].startsWith("--"))
            { parts.push(a[i] + " " + a[i+1]); i++; }
          else parts.push(a[i]);
        }
        text = "python -m reforge \\\n    " + parts.join(" \\\n    ");
      } else {
        text = "⚠ " + data.message;
      }
      $("cmd-preview").textContent = text;
    }catch(e){ $("cmd-preview").textContent = "preview unavailable"; }
  }, 400);
}

/* ── batch stage editor ────────────────────────────────────────────── */
window._stages = [];
function renderStages(){
  const box = $("stage-list");
  box.innerHTML = "";
  window._stages.forEach((s, i) => {
    const row = document.createElement("div");
    row.className = "stage";
    row.innerHTML = `
      <span class="n">${i+1}</span>
      <input type="text" data-f="name" placeholder="stage name" value="${esc(s.name||"")}">
      <input type="text" data-f="dataset" placeholder="C:\\…\\data.parquet or org/hf-id" value="${esc(s.dataset||s.hf_dataset||"")}">
      <input type="text" data-f="epochs" placeholder="epochs" value="${esc(s.epochs??"")}">
      <input type="text" data-f="steps" placeholder="steps" value="${esc(s.steps??"")}">
      <button class="ibtn" data-act="up" title="move up">↑</button>
      <button class="ibtn" data-act="down" title="move down">↓</button>
      <button class="ibtn" data-act="del" title="remove">✕</button>`;
    row.addEventListener("change", () => {
      const ds = row.querySelector('[data-f="dataset"]').value.trim();
      s.name   = row.querySelector('[data-f="name"]').value.trim();
      s.epochs = row.querySelector('[data-f="epochs"]').value.trim();
      s.steps  = row.querySelector('[data-f="steps"]').value.trim();
      delete s.dataset; delete s.hf_dataset;
      if (/^[^\\\/]+\//.test(ds) && !/^[A-Za-z]:[\\\/]/.test(ds)) s.hf_dataset = ds;
      else s.dataset = ds;
      schedulePreview();
    });
    row.querySelector('[data-act="up"]').onclick = () => {
      if (i > 0){ [window._stages[i-1], window._stages[i]] = [window._stages[i], window._stages[i-1]];
        renderStages(); schedulePreview(); }
    };
    row.querySelector('[data-act="down"]').onclick = () => {
      if (i < window._stages.length - 1){
        [window._stages[i+1], window._stages[i]] = [window._stages[i], window._stages[i+1]];
        renderStages(); schedulePreview(); }
    };
    row.querySelector('[data-act="del"]').onclick = () => {
      window._stages.splice(i, 1); renderStages(); schedulePreview();
    };
    box.appendChild(row);
  });
}
$("btn-addstage").onclick = () => { window._stages.push({}); renderStages(); schedulePreview(); };
$("p-batchmode").onchange = e => {
  $("batch-editor").hidden = !e.target.checked;
  if (e.target.checked && !window._stages.length){ window._stages.push({}); renderStages(); }
  schedulePreview();
};

/* ── actions ───────────────────────────────────────────────────────── */
$("btn-start").onclick = async () => {
  const {code, data} = await jpost("/api/start", collectParams());
  if (!data.ok) showBanner(data.message || "Could not start.");
  else { hideBanner(); }
};
$("btn-stop").onclick = async () => {
  const {data} = await jpost("/api/stop");
  if (!data.ok) showBanner(data.message || "Could not stop.");
};
$("btn-again").onclick = () => { hideBanner(); route("idle"); };
function showBanner(msg){ const b = $("msg-banner"); b.textContent = msg; b.hidden = false; }
function hideBanner(){ $("msg-banner").hidden = true; }

/* ── views ─────────────────────────────────────────────────────────── */
function route(state){
  const setup = $("view-setup"), prog = $("view-progress"), res = $("view-results");
  setup.hidden = state !== "idle";
  prog.hidden  = !(state === "running" || state === "stopping");
  res.hidden   = !(state === "done" || state === "error");
  $("btn-start").hidden = !(state === "idle" || state === "done" || state === "error");
  $("btn-stop").hidden  = !(state === "running" || state === "stopping");
}

function chip(state, s){
  const el = $("chip");
  el.className = "chip";
  let text = "Idle", cls = "";
  if (state === "running"){ cls = "run"; text = (s.phase || "starting") + "…"; }
  else if (state === "stopping"){ cls = "stop"; text = "Stopping (saving)…"; }
  else if (state === "done"){ cls = "done"; text = "Complete"; }
  else if (state === "error"){ cls = "err"; text = "Failed"; }
  if (s.interrupted) text += " · interrupted";
  el.classList.add(cls);
  $("chip-text").textContent = text;
}

/* ── progress rendering ────────────────────────────────────────────── */
function renderPhases(phases){
  const box = $("phases");
  box.innerHTML = "";
  PHASES.forEach((p, i) => {
    if (i) { const a = document.createElement("span"); a.className = "arrow"; a.textContent = "→"; box.appendChild(a); }
    const el = document.createElement("span");
    const st = phases[p] || "pending";
    el.className = "ph" + (st === "active" ? " active" : st === "done" ? " done" : st === "failed" ? " failed" : "");
    el.innerHTML = (st === "active" ? '<span class="spin"></span>' : "") + esc(PHASE_LABEL[p]);
    box.appendChild(el);
  });
}

function renderStats(s){
  const stats = [];
  const add = (k, val, gold) => stats.push(
    `<div class="stat${gold ? " gold" : ""}${k==="STAGE" ? " stagechip" : ""}"><small>${k}</small><span>${val}</span></div>`);
  add("STEPS", `${s.step}/${s.total_steps || "—"}`);
  if (s.latest_loss !== null && s.latest_loss !== undefined)
    add("LOSS", Number(s.latest_loss).toFixed(4), true);
  if (s.best_loss !== null && s.best_loss !== undefined)
    add("BEST", Number(s.best_loss).toFixed(4));
  add("ETA", fmtHMS(s.eta_sec));
  if (s.rate_it_s) add("RATE", Number(s.rate_it_s).toFixed(2) + " it/s");
  if (s.stage) add("STAGE", `${(s.stage.index ?? -1) + 1}/${s.stage.total}`);
  if (s.training_mode) add("MODE", esc(s.training_mode));
  $("stats").innerHTML = stats.join("");
}

function renderTqdm(s){
  const el = $("tqdm");
  const bar = $("bar");
  if (s.pct !== null && s.pct !== undefined){
    bar.classList.remove("indet");
    bar.querySelector("i").style.width = Math.min(100, s.pct) + "%";
    $("bar-cap").textContent = (Number(s.pct).toFixed(1)) + "%";
  } else {
    bar.classList.add("indet");
    $("bar-cap").textContent = "";
  }
  const elapsed = s.train_elapsed !== null && s.train_elapsed !== undefined ? fmtHMS(s.train_elapsed) : "--:--";
  const eta = s.eta_sec !== null && s.eta_sec !== undefined ? fmtHMS(s.eta_sec) : "--:--";
  const rate = s.rate_it_s ? ` · ${Number(s.rate_it_s).toFixed(2)} it/s` : "";
  el.textContent = `step ${s.step}/${s.total_steps || "?"}`
    + (s.pct !== null && s.pct !== undefined ? ` · ${Number(s.pct).toFixed(1)}%` : "")
    + ` · [${elapsed}<${eta}${rate}]`;
}

/* ── loss chart ────────────────────────────────────────────────────── */
function smooth(arr, k){
  k = Math.max(1, Math.round(k));
  if (k === 1 || arr.length < 2) return arr.slice();
  const out = [];
  for (let i = 0; i < arr.length; i++){
    let a = Math.max(0, i - (k >> 1)), b = Math.min(arr.length, a + k);
    a = Math.max(0, b - k);
    let sum = 0; for (let j = a; j < b; j++) sum += arr[j];
    out.push(sum / (b - a));
  }
  return out;
}
function drawChart(s){
  const cv = $("chart");
  const raw = s.losses || [];
  if (!raw.length || cv.clientWidth === 0) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== cv.clientWidth * dpr){
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
  const W = cv.clientWidth, H = cv.clientHeight, padL = 46, padR = 12, padT = 12, padB = 20;

  // decimate for drawing
  let pts = raw;
  if (raw.length > 4000){
    const stride = Math.ceil(raw.length / 4000);
    pts = raw.filter((_, i) => i % stride === 0 || i === raw.length - 1);
  }
  const xs = pts.map(p => p.step), ys = pts.map(p => p.loss);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (ymax - ymin < 1e-6){ ymax += 0.5; ymin -= 0.5; }
  const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
  const xmin = xs[0], xmax = xs[xs.length - 1] || 1;
  const X = v => padL + (W - padL - padR) * (xmax === xmin ? 0.5 : (v - xmin) / (xmax - xmin));
  const Y = v => padT + (H - padT - padB) * (1 - (v - ymin) / (ymax - ymin));

  // grid + y labels
  ctx.strokeStyle = "rgba(255,255,255,.06)"; ctx.fillStyle = "#9096ad";
  ctx.font = "10px Consolas,monospace";
  for (let g = 0; g <= 4; g++){
    const yv = ymin + (ymax - ymin) * g / 4, py = Y(yv);
    ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(W - padR, py); ctx.stroke();
    ctx.fillText(yv.toFixed(2), 6, py + 3);
  }

  const k = Math.max(3, Math.round(raw.length / 50) | 1);
  const sm = smooth(raw.map(p => p.loss), k);
  const line = (arr, color, width, alpha) => {
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width;
    ctx.beginPath();
    arr.forEach((p, i) => { const x = X(p.step), y = Y(arr === sm ? sm[i] : p.loss);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  line(raw, "#FFBF00", 1, .22);
  ctx.strokeStyle = "#FFD700"; ctx.lineWidth = 1.8; ctx.beginPath();
  raw.forEach((p, i) => { const x = X(p.step), y = Y(sm[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();

  // best marker
  let bi = 0; raw.forEach((p, i) => { if (p.loss < raw[bi].loss) bi = i; });
  ctx.fillStyle = "#4caf50";
  ctx.beginPath(); ctx.arc(X(raw[bi].step), Y(raw[bi].loss), 3.5, 0, 7); ctx.fill();

  ctx.fillStyle = "#9096ad"; ctx.font = "11px Consolas,monospace";
  ctx.fillText(`step ${raw[raw.length-1].step} | now ${raw[raw.length-1].loss.toFixed(4)} | best ${raw[bi].loss.toFixed(4)} @ ${raw[bi].step}`, padL, H - 6);
}

/* ── analysis ──────────────────────────────────────────────────────── */
async function renderAnalysis(){
  try{ analysis = await jget("/api/analysis"); }catch(e){ return; }
  const box = $("analysis"), alog = $("alog");
  if (!analysis || !analysis.ok){
    box.innerHTML = ""; alog.textContent = analysis && analysis.error ? "analysis: " + esc(analysis.error) : "";
    return;
  }
  const card = (k, v, gold) => `<div class="stat${gold ? " gold" : ""}"><small>${k}</small><span>${esc(v)}</span></div>`;
  box.innerHTML =
    card("GRADE", analysis.grade, true) +
    card("TREND", `${analysis.trend_global} / ${analysis.trend_recent}`) +
    card("SNR", analysis.snr.toFixed(2)) +
    card("EXPOSURE", analysis.exposure_pct.toFixed(1) + "%") +
    card("BEST", `${analysis.best.toFixed(4)} @ ${analysis.best_step}`) +
    card("SINCE BEST", analysis.since_best);
  alog.innerHTML =
    `avg ${analysis.avg} | med ${analysis.med} | std ${analysis.std} | var ${analysis.drop_pct}%<br>` +
    `slope global ${analysis.slope_global >= 0 ? "+" : ""}${analysis.slope_global}/100 | recent ${analysis.slope_recent >= 0 ? "+" : ""}${analysis.slope_recent}/100<br>` +
    `est. ideal zone: ${esc(analysis.ideal_zone)}` +
    (analysis.outliers.length ? ` | outliers: ${analysis.outliers.length}` : "");
}

/* ── log pane ──────────────────────────────────────────────────────── */
function renderLog(s, elId){
  const el = $(elId);
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 4;
  el.textContent = (s.log || []).join("");
  if (elId === "log" && $("log-follow").checked && !atBottom) el.scrollTop = el.scrollHeight;
}

/* ── results view ──────────────────────────────────────────────────── */
async function renderResults(s){
  const body = $("res-body"), title = $("res-title");
  if (s.state === "error"){
    title.textContent = "Run failed";
    let html = `<div class="banner">${esc(s.error || "Unknown error")}`
      + (s.error_code ? ` <span class="chip"><span class="dot"></span>${esc(s.error_code)}</span>` : "")
      + `</div>`;
    if (s.hint) html += `<div class="hint">💡 ${esc(s.hint)}</div>`;
    if (s.interrupted) html += `<div class="banner warn">Run was interrupted — progress was saved.</div>`;
    body.innerHTML = html;
    renderLog(s, "res-log");
    return;
  }
  title.textContent = "Training complete";
  const r = s.result || {};
  const card = (k, v, gold) => `<div class="stat${gold ? " gold" : ""}"><small>${k}</small><span>${esc(v ?? "—")}</span></div>`;
  let html = `<div class="res">`
    + card("Merged to", r.merged_dir || "—", true)
    + card("Final loss", r.final_loss !== null && r.final_loss !== undefined ? Number(r.final_loss).toFixed(4) : null)
    + card("Best loss", r.best_loss !== null && r.best_loss !== undefined ? Number(r.best_loss).toFixed(4) : null)
    + card("Steps", s.total_steps)
    + card("Time", fmtHMS(r.time_sec))
    + `</div>`;
  body.innerHTML = html;
  try{
    const rep = await jget("/api/report");
    if (rep.ok && rep.report){
      const lo = rep.report.loss || rep.report;
      const kv = Object.entries(lo).slice(0, 12)
        .map(([k, v]) => `${k}: ${typeof v === "number" ? (Math.round(v * 10000) / 10000) : JSON.stringify(v)}`)
        .join("\n");
      body.innerHTML += `<pre class="kv" style="margin-top:12px">${esc(kv)}</pre>`;
    }
  }catch(e){}
  renderLog(s, "res-log");
}

/* ── main loop ─────────────────────────────────────────────────────── */
let viewState = null;
async function tick(){
  let s;
  try{ s = await jget("/api/status"); }
  catch(e){ chip("unreachable", {}); $("chip-text").textContent = "Server unreachable"; return; }
  last = s;
  chip(s.state, s);
  if (viewState !== s.state){
    viewState = s.state;
    hideBanner();
    route(s.state);
    if (s.state === "done" || s.state === "error") renderResults(s);
  }
  if (s.state === "running" || s.state === "stopping"){
    renderPhases(s.phases);
    renderTqdm(s);
    renderStats(s);
    drawChart(s);
    renderLog(s, "log");
    if (Date.now() - anaTimer > 5000){ anaTimer = Date.now(); renderAnalysis(); }
    // results render once at the state transition above
  }
}

/* ── boot ──────────────────────────────────────────────────────────── */
document.querySelectorAll("#view-setup input").forEach(el => {
  el.addEventListener("input", schedulePreview);
});
jget("/api/models").then(d => {
  $("models").innerHTML = (d.models || []).map(m => `<option value="${esc(m)}">`).join("");
}).catch(() => {});
route("idle");
setInterval(tick, 1000);
tick();
</script>
</body>
</html>
"""
