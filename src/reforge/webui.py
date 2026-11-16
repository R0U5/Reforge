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
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .display import _info, _title

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
                result = {"merged_dir": None, "report": ev.get("report")}
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
                "pct": None if self.pct is None else round(self.pct, 2),
                "eta_sec": self.eta_sec,
                "latest_loss": self.latest_loss,
                "best_loss": self.best_loss,
                "rate_it_s": (round(self.rate_it_s, 3)
                              if self.rate_it_s is not None else None),
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

    try:
        from .webui_config import build_argv
    except ImportError:
        return False, "argv builder lands in Phase 2 (docs/WEBUI_MIGRATION.md §6)"
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
    JOB.begin(proc, params.get("base_model") or "", dataset)

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
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/start":
            ok, msg = _spawn(self._read_json())
            self._send_json({"ok": ok, "message": msg}, 200 if ok else 409)
        elif self.path == "/api/stop":
            ok, msg = _stop()
            self._send_json({"ok": ok, "message": msg}, 200 if ok else 409)
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


# ── Placeholder page (Phase 3 replaces this with the full UI) ───────
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reforge</title>
<style>
  :root{ --bg:#101014; --card:#1a1a2e; --ink:#FFF8DC; --muted:#9096ad;
    --line:#33333f; --accent:#FFBF00; --ok:#4caf50; --bad:#ef5350; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 "Segoe UI",system-ui,sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:28px 20px 80px}
  h1{color:var(--accent);margin:0 0 4px}
  .hint{color:var(--muted);font-size:13px;margin-bottom:20px}
  pre{background:var(--card);border:1px solid var(--line);border-radius:9px;
    padding:12px;font:12px/1.5 Consolas,monospace;color:var(--muted);
    white-space:pre-wrap;max-height:220px;overflow:auto}
  .pill{display:inline-block;padding:2px 10px;border:1px solid var(--line);
    border-radius:20px;font-size:12px;color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <h1>REFORGE</h1>
  <div class="hint">Web dashboard — server skeleton. Full UI lands in Phase 3
    (docs/WEBUI_MIGRATION.md).</div>
  <div class="pill" id="state">idle</div>
  <pre id="status">polling /api/status …</pre>
</div>
<script>
async function tick(){
  try{
    const s = await (await fetch("/api/status")).json();
    document.getElementById("state").textContent = s.state;
    document.getElementById("status").textContent =
      JSON.stringify({phase:s.phase, step:s.step, total_steps:s.total_steps,
                      pct:s.pct, eta_sec:s.eta_sec, latest_loss:s.latest_loss,
                      stage:s.stage, exit_code:s.exit_code, error:s.error},
                     null, 1);
  }catch(e){ document.getElementById("status").textContent = "server unreachable"; }
}
setInterval(tick, 1000); tick();
</script>
</body>
</html>
"""
