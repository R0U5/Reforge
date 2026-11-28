#!/usr/bin/env python3
"""Reforge Training Watchdog v2 — supervisor for batch training.

Health model: child process alive AND run_state.json heartbeat fresh
(with startup grace for tokenization). See docs/watchdog-fmea.md for the
full failure-mode analysis this implements.

Usage:
  python -m reforge.watchdog --batch <manifest.json> --batch_dir <dir> \
      --base_model <path> [--agent_log <path>] [--stop_file <path>]

Exit codes: 0=batch complete, 2=stopped by request, 3=gave up (needs human).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOCK_NAME = "watchdog.lock"
STATE_NAME = "watchdog_state.json"
STOP_NAME = "stop_requested"
EVENTS_NAME = "watchdog_events.ndjson"
BRIEF_NAME = "situation_brief.md"
LEDGER_NAME = "watchdog_signatures.json"
ESCALATION_WINDOW_H = 6

MAX_CONSECUTIVE_FAILURES = 5
BACKOFF_SCHEDULE = [30, 120, 480, 1800, 1800]  # seconds
MIN_DISK_FREE_GB = 5.0
MIN_FREE_VRAM_MIB = 1500  # don't relaunch if a game holds the GPU
GPU_WAIT_POLL = 60
STARTUP_GRACE_S = 1200  # tokenization of 15k rows can take ~10-15 min
STALE_THRESHOLD_S = 180
KILL_ESCALATION_S = 90
LOG_ROTATE_BYTES = 50 * 1024 * 1024


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =====================================================================
# Situation brain — tier 1: deterministic diagnosis (zero tokens).
# Classifies deaths from evidence; writes a pre-digested brief so any
# AI consumer (agent monitor turn, human) reads 30 lines instead of
# mining logs. Tier 2 (LLM escalation) only fires on novel/repeat
# signatures — see should_escalate().
# =====================================================================

def collect_evidence(wd: "Watchdog") -> dict:
    ev = {
        "run_state": None,
        "last_events": [],
        "child_exit_code": wd.child.returncode if wd.child else None,
        "child_alive": bool(wd.child and wd.child.poll() is None),
        "free_vram_mib": wd.free_vram_mib(),
        "disk_free_gb": wd.disk_free_gb(),
        "checkpoints": [],
    }
    rsp = wd.run_state_path()
    if rsp:
        ev["run_state"] = wd.read_json_tolerant(rsp)
        ev["run_state_age_s"] = wd.state_age_seconds(rsp)
    # tail of the agent NDJSON stream (cheap bounded read)
    if wd.agent_log and wd.agent_log.exists():
        try:
            with open(wd.agent_log, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-30:]
            ev["last_events"] = []
            for line in tail:
                try:
                    j = json.loads(line)
                    ev["last_events"].append({k: j.get(k) for k in
                                              ("ts", "event", "step", "loss", "error", "error_code")
                                              if k in j})
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
    sd = wd.stage_dir()
    if sd:
        ckpts = [(p.name, p.stat().st_mtime) for p in sd.glob("checkpoint-*")
                 if p.name.split("-")[-1].isdigit()]
        ev["checkpoints"] = sorted(ckpts, key=lambda t: -t[1])[:3]
    return ev


def classify(ev: dict) -> list[dict]:
    """Rule engine: evidence -> findings with cause + confidence + action."""
    findings = []
    events = ev.get("last_events") or []
    rs = ev.get("run_state") or {}
    last = events[-1] if events else {}

    def add(cause, conf, action, note):
        findings.append({"cause": cause, "confidence": conf, "action": action, "note": note})

    ev_types = [e.get("event", "") for e in events]
    if any(t in ("run_complete", "training_complete") for t in ev_types):
        add("clean_completion", "high", "advance_batch",
            "completion events present — do not resume")
    if any(t == "early_stop_triggered" for t in ev_types):
        add("early_stop", "high", "advance_batch",
            "early-stop criterion fired — hand off to next stage")

    step = rs.get("step") or last.get("step")
    if not ev_types or last.get("event", "").startswith("step") or last.get("event") == "dataset_progress":
        if ev.get("child_exit_code") not in (None, 0):
            add("external_kill_during_work", "high", "resume_from_checkpoint",
                f"died mid-work at step {step}, exit={ev['child_exit_code']}, no terminal event — "
                "classic external kill (OOM/driver)")
        else:
            add("silent_death", "medium", "resume_from_checkpoint",
                f"no terminal event; last activity step {step} — treat as external kill")

    # OOM suspicion: GPU nearly full in the last observed state
    vram = ev.get("free_vram_mib")
    if vram is not None and vram < 500:
        add("cuda_oom_suspected", "medium", "resume_with_caution",
            "GPU memory was exhausted at death; if this repeats, lower --max_length")

    # Loss pathology in recent events
    losses = [e.get("loss") for e in events if isinstance(e.get("loss"), (int, float))]
    if losses and any(l != l or l in (float("inf"), float("-inf")) for l in losses[-5:]):
        add("loss_diverged", "high", "resume_and_flag",
            "NaN/inf loss in recent events — check LR / data before trusting resume")

    # Died during prep = zero training progress lost
    if any(t == "dataset_progress" for t in ev_types) and not any(
            t.startswith("step") for t in ev_types):
        add("died_during_prep", "high", "resume_from_checkpoint",
            "died in dataset prep/tokenization — no training steps lost")

    if not findings:
        add("unknown", "low", "resume_from_checkpoint",
            "no signature matched; default action is resume from checkpoint-last")
    return findings


def signature_of(findings: list[dict]) -> str:
    return "+".join(sorted(f["cause"] for f in findings)) or "unknown"


def should_escalate(sig: str, batch_dir: Path) -> tuple[bool, str]:
    """Tier 2 gate: burn tokens only for novel or recurring signatures."""
    ledger_path = batch_dir / LEDGER_NAME
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) \
            if ledger_path.exists() else {"seen": {}}
    except (OSError, json.JSONDecodeError):
        ledger = {"seen": {}}
    now = time.time()
    rec = ledger["seen"].get(sig, {"count": 0, "last_ts": 0, "escalated": False})
    rec["count"] += 1
    rec["last_ts"] = now
    within_window = (now - rec.get("first_ts", now)) < ESCALATION_WINDOW_H * 3600
    escalate = False
    reason = ""
    if rec["count"] == 1:
        escalate, reason = True, "novel signature — first occurrence warrants one AI review"
    elif within_window and rec["count"] >= 3:
        escalate, reason = True, f"recurring: {rec['count']}x in {ESCALATION_WINDOW_H}h — pattern needs reasoning"
    if escalate:
        rec["escalated"] = True
        rec["first_ts"] = rec.get("first_ts", now)
    ledger["seen"][sig] = rec
    try:
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    except OSError:
        pass
    return escalate, reason


def write_situation_brief(wd: "Watchdog", ev: dict, findings: list[dict],
                          verdict: str, detail: str) -> Path:
    """Human/LLM-readable digest: the whole situation in ~30 lines."""
    path = wd.batch_dir / BRIEF_NAME
    rs = ev.get("run_state") or {}
    step = rs.get("step", "?")
    best = rs.get("best_loss", "?")
    sig = signature_of(findings)
    primary = findings[0] if findings else {}
    disk = ev.get("disk_free_gb")
    disk_txt = f"{disk:.1f}" if isinstance(disk, (int, float)) else "?"
    lines = [
        f"# Training situation brief — {now_iso()}",
        f"", f"**Verdict:** {verdict} ({detail})  ",
        f"**Step:** {step} | best loss: {best}  ",
        f"**Child:** exit={ev.get('child_exit_code')} alive={ev.get('child_alive')}  ",
        f"**GPU free:** {ev.get('free_vram_mib')} MiB | **Disk free:** {disk_txt} GB",
        "",
        "## Diagnosis (deterministic, tier 1)",
    ]
    for f in findings:
        lines.append(f"- **{f['cause']}** ({f['confidence']} conf) → {f['action']} — {f['note']}")
    lines += ["", "## Last events"]
    for e in ev.get("last_events", [])[-8:]:
        lines.append(f"- `{e.get('ts','?')}` {e.get('event','?')}"
                     + (f" step={e['step']}" if e.get("step") is not None else "")
                     + (f" loss={e['loss']}" if e.get("loss") is not None else ""))
    lines += ["", "## Checkpoints",
              f"- " + ", ".join(n for n, _ in ev.get("checkpoints", [])) or "- none",
              "", f"**Signature:** `{sig}`", ""]
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass
    return path


class Watchdog:
    def __init__(self, args):
        self.manifest = Path(args.batch).resolve()
        self.base_model = args.base_model
        self.extra_child_args = list(getattr(args, "extra_child_args", []) or [])
        self.agent_log = Path(args.agent_log).resolve() if args.agent_log else None
        self._stop_file = Path(args.stop_file).resolve() if getattr(args, "stop_file", None) else None
        self.startup_grace = args.startup_grace
        self.stale_threshold = args.stale_threshold
        self.dry_run = args.dry_run

        # Fresh-batch discovery: when no --batch_dir is given, the batch runner
        # creates a timestamped batch_<ts> dir itself. Watchdog starts at the
        # discovery root and adopts the new batch_state.json once it appears.
        self.discover_root = Path(args.discover_root).resolve() if getattr(args, "discover_root", None) else None
        supplied = Path(args.batch_dir).resolve() if args.batch_dir else self.manifest.parent
        self._pending = (self.discover_root is not None
                         and not (supplied / "batch_state.json").exists())
        self.batch_dir = self.discover_root if self._pending else supplied
        self.discover_after: float | None = None

        self.child: subprocess.Popen | None = None
        self.consecutive_failures = 0
        self.child_started_at: float | None = None
        self.first_heartbeat_seen = False

        self.lock_path = self.batch_dir / LOCK_NAME  # fixed for the lifetime

    # --- paths that follow the (possibly adopted) batch dir ---

    @property
    def state_path(self) -> Path:
        return self.batch_dir / STATE_NAME

    @property
    def events_path(self) -> Path:
        return self.batch_dir / EVENTS_NAME

    @property
    def stop_file(self) -> Path:
        return self._stop_file or (self.batch_dir / STOP_NAME)

    @property
    def batch_state_path(self) -> Path:
        return self.batch_dir / "batch_state.json"

    # ---------- event/state logging ----------

    def emit(self, event: str, **kw) -> None:
        rec = {"ts": now_iso(), "event": f"watchdog_{event}", **kw}
        line = json.dumps(rec, default=str)
        try:
            self._rotate_if_big(self.events_path)
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        print(f"[WATCHDOG] {line}", flush=True)

    def write_state(self, status: str, **kw) -> None:
        data = {
            "watchdog_version": 2,
            "pid": os.getpid(),
            "status": status,
            "updated_at": now_iso(),
            "updated_epoch": time.time(),
            "batch_dir": str(self.batch_dir),
            "child_pid": self.child.pid if self.child else None,
            "consecutive_failures": self.consecutive_failures,
            **kw,
        }
        tmp = self.state_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_path)  # atomic on Windows
        except OSError as e:
            self.emit("state_write_failed", error=str(e))

    @staticmethod
    def _rotate_if_big(path: Path) -> None:
        if path.exists() and path.stat().st_size > LOG_ROTATE_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))

    # ---------- single-instance lock ----------

    def acquire_lock(self) -> bool:
        try:
            self.batch_dir.mkdir(parents=True, exist_ok=True)
            if self.lock_path.exists():
                old = self.lock_path.read_text(encoding="utf-8", errors="replace").strip()
                try:
                    old_pid = int(old)
                    # Windows: open process 0-access check via tasklist-free trick
                    probe = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {old_pid}"],
                        capture_output=True, text=True, timeout=15)
                    if old_pid != os.getpid() and old_pid in probe.stdout:
                        return False  # another live watchdog
                except (ValueError, subprocess.SubprocessError):
                    pass
                self.emit("stale_lock_broken", old=old)
            self.lock_path.write_text(str(os.getpid()), encoding="utf-8")
            return True
        except OSError as e:
            self.emit("lock_error", error=str(e))
            return False

    def release_lock(self) -> None:
        try:
            if self.lock_path.exists() and self.lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self.lock_path.unlink()
        except OSError:
            pass

    # ---------- health signals ----------

    def read_json_tolerant(self, path: Path, retries: int = 3) -> dict | None:
        """Scenario 10/11: partial writes, AV locks, unparseable content."""
        for attempt in range(retries):
            try:
                if not path.exists():
                    return None
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                time.sleep(1)
        return None

    def state_age_seconds(self, path: Path) -> float | None:
        """Age of state, mtime fallback when JSON is unreadable (10/11)."""
        data = self.read_json_tolerant(path)
        if data and data.get("updated_at"):
            try:
                ts = data["updated_at"].replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - dt).total_seconds()
                if -600 < age < 30 * 86400:  # reject absurd skew
                    return age
            except ValueError:
                pass
        try:
            return max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            return None

    def stage_dir(self) -> Path | None:
        bs = self.read_json_tolerant(self.batch_state_path)
        if bs:
            for st in bs.get("stages", []):
                if st.get("status") == "running":
                    return self.batch_dir / f"stage_{st.get('index', 0) + 1:02d}"
        # fallback: newest stage dir
        dirs = sorted(self.batch_dir.glob("stage_*"))
        return dirs[-1] if dirs else None

    def run_state_path(self) -> Path | None:
        sd = self.stage_dir()
        return sd / "run_state.json" if sd else None

    def batch_finished(self) -> bool:
        bs = self.read_json_tolerant(self.batch_state_path)
        if not bs:
            return False
        stages = bs.get("stages", [])
        return bool(stages) and all(s.get("status") in ("complete", "skipped") for s in stages)

    def free_vram_mib(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            return int(out.stdout.strip().splitlines()[0])
        except (subprocess.SubprocessError, ValueError, IndexError):
            return None

    def disk_free_gb(self) -> float | None:
        try:
            return shutil.disk_usage(str(self.batch_dir)).free / 2**30
        except OSError:
            return None

    # ---------- child lifecycle ----------

    def spawn_cmd(self) -> list[str]:
        cmd = [sys.executable, "-m", "reforge", "--agent",
               "--batch", str(self.manifest)]
        # Only pass --batch_dir when it is a real batch dir (resume case).
        # For a fresh batch the child creates its own timestamped dir and the
        # watchdog adopts it via discovery.
        if not self._pending and (self.batch_dir / "batch_state.json").exists():
            cmd += ["--batch_dir", str(self.batch_dir)]
        if self.base_model:
            cmd += ["--base_model", self.base_model]
        if self.agent_log:
            cmd += ["--agent_log", str(self.agent_log)]
        cmd += self.extra_child_args
        return cmd

    def spawn(self) -> None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.child = subprocess.Popen(self.spawn_cmd(), env=env, creationflags=flags,
                                      cwd=str(self.batch_dir))
        self.child_started_at = time.time()
        self.first_heartbeat_seen = False
        self.emit("child_spawned", pid=self.child.pid)

    def stop_child(self, reason: str) -> int | None:
        """Graceful CTRL_BREAK -> escalate kill (scenarios 4, 19)."""
        if not self.child or self.child.poll() is not None:
            return self.child.returncode if self.child else None
        pid = self.child.pid
        self.emit("stopping_child", pid=pid, reason=reason)
        try:
            self.child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        except OSError:
            pass
        try:
            self.child.wait(timeout=KILL_ESCALATION_S)
            self.emit("child_stopped_gracefully", pid=pid, returncode=self.child.returncode)
            return self.child.returncode
        except subprocess.TimeoutExpired:
            self.emit("child_kill_escalation", pid=pid)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=30)
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            return self.child.returncode

    def assess_health(self) -> tuple[str, str]:
        """Returns (verdict, detail): 'ok', 'dead', 'stale', 'warming_up'."""
        if self.child is None or self.child.poll() is not None:
            return "dead", f"exit={self.child.returncode if self.child else 'n/a'}"
        rsp = self.run_state_path()
        if rsp is None or not rsp.exists():
            # no run_state yet — inside startup grace?
            if time.time() - (self.child_started_at or 0) < self.startup_grace:
                return "warming_up", "no run_state yet (startup grace)"
            return "stale", "no run_state beyond startup grace"
        age = self.state_age_seconds(rsp)
        if age is None:
            return "warming_up", "run_state unreadable; treating as transient" \
                if time.time() - (self.child_started_at or 0) < self.startup_grace \
                else "stale"
        if not self.first_heartbeat_seen:
            # grace until first heartbeat lands (scenario 9)
            if time.time() - (self.child_started_at or 0) < self.startup_grace:
                return "warming_up", f"awaiting first heartbeat (age={age:.0f}s)"
            self.first_heartbeat_seen = True
        if age > self.stale_threshold:
            return "stale", f"run_state age={age:.0f}s"
        return "ok", f"age={age:.0f}s"

    # ---------- relaunch gating ----------

    def should_relaunch(self) -> tuple[bool, str]:
        if self.stop_file.exists():
            return False, "stop_requested file present"
        if self.batch_finished():
            return False, "batch complete"
        dfree = self.disk_free_gb()
        if dfree is not None and dfree < MIN_DISK_FREE_GB:
            return False, f"disk low: {dfree:.1f}GB free"
        vram = self.free_vram_mib()
        if vram is not None and vram < MIN_FREE_VRAM_MIB:
            return False, f"gpu_busy: only {vram}MiB free (game running?)"
        return True, "ok"

    def wait_for_gpu(self) -> bool:
        """Block until VRAM frees up or stop file appears (scenario 5)."""
        while not self.stop_file.exists():
            vram = self.free_vram_mib()
            if vram is None or vram >= MIN_FREE_VRAM_MIB:
                return True
            self.emit("waiting_gpu_free", free_mib=vram)
            self.write_state("waiting_gpu", free_vram_mib=vram)
            time.sleep(GPU_WAIT_POLL)
        return False

    # ---------- main loop ----------

    def _adopt_new_batch(self) -> None:
        """Fresh-batch mode: find the batch_state.json the child just created."""
        if not self._pending or self.discover_after is None:
            return
        try:
            candidates = []
            for p in self.discover_root.glob("**/batch_state.json"):
                try:
                    if p.stat().st_mtime >= self.discover_after - 5:
                        candidates.append(p)
                except OSError:
                    continue
            if not candidates:
                return
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            self.batch_dir = newest.parent
            self._pending = False
            self.emit("batch_adopted", batch_dir=str(self.batch_dir))
        except OSError as e:
            self.emit("adopt_scan_error", error=str(e))

    def run(self) -> int:
        if not self.acquire_lock():
            self.emit("already_running", detail="lock held; exiting")
            return 0
        self.emit("started", batch_dir=str(self.batch_dir), pending_discovery=self._pending)
        self.write_state("running")
        if self._pending:
            self.discover_after = time.time()

        stop_requested = {"flag": False}
        signal.signal(signal.SIGINT, lambda s, f: (stop_requested.__setitem__("flag", True),
                                                    self.emit("stop_signal", sig="SIGINT")))
        signal.signal(signal.SIGTERM, lambda s, f: (stop_requested.__setitem__("flag", True),
                                                    self.emit("stop_signal", sig="SIGTERM")))

        self.spawn()
        last_state_flush = 0.0

        try:
            while True:
                time.sleep(15)

                if self._pending:
                    self._adopt_new_batch()

                if stop_requested["flag"] or self.stop_file.exists():
                    rc = self.stop_child("manual_stop")
                    self.write_state("stopped_by_request", child_rc=rc)
                    self.emit("exit_stopped_by_request", child_rc=rc)
                    return 2

                verdict, detail = self.assess_health()
                if verdict == "ok":
                    self.consecutive_failures = 0  # healthy run resets the loop counter
                if verdict in ("dead", "stale"):
                    self.emit("unhealthy", verdict=verdict, detail=detail)
                    if verdict == "dead":
                        self.emit("child_died", returncode=self.child.returncode if self.child else None)
                    else:
                        # hung-but-alive: escalate to graceful stop then treat as death
                        self.stop_child("stale_state")

                    # Tier 1 diagnosis: deterministic, zero tokens.
                    ev = collect_evidence(self)
                    findings = classify(ev)
                    sig = signature_of(findings)
                    brief = write_situation_brief(self, ev, findings, verdict, detail)
                    escalate, esc_reason = should_escalate(sig, self.batch_dir)
                    self.emit("diagnosis", signature=sig,
                              findings=[f["cause"] for f in findings],
                              primary=findings[0] if findings else None,
                              brief=str(brief), escalate=escalate,
                              escalate_reason=esc_reason)
                    self.write_state("diagnosing", signature=sig,
                                     situation=findings[0] if findings else {},
                                     situation_brief=str(brief), escalate=escalate)

                    # Clean hand-off (early stop / completion): never relaunch.
                    if any(f["action"] == "advance_batch" for f in findings):
                        self.write_state("handed_off", signature=sig)
                        self.emit("handoff_no_relaunch", signature=sig)
                        return 0

                    relaunch, why = self.should_relaunch()
                    if not relaunch:
                        if "gpu_busy" in why:
                            if self.wait_for_gpu():
                                relaunch, why = True, "gpu freed"
                            else:
                                self.write_state("stopped_by_request", note="stop during gpu wait")
                                return 2
                    if not relaunch:
                        self.write_state("gave_up", reason=why)
                        self.emit("no_relaunch", reason=why)
                        return 3 if "disk" in why or "corrupt" in why else 0

                    self.consecutive_failures += 1
                    if self.consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                        self.write_state("gave_up", reason="restart_limit")
                        self.emit("restart_limit_reached", failures=self.consecutive_failures)
                        return 3

                    delay = BACKOFF_SCHEDULE[min(self.consecutive_failures - 1, len(BACKOFF_SCHEDULE) - 1)]
                    self.emit("backoff", seconds=delay, failure_n=self.consecutive_failures)
                    self.write_state("backoff", seconds=delay)
                    # interruptible backoff: check stop file during the wait
                    end = time.time() + delay
                    while time.time() < end:
                        if self.stop_file.exists():
                            self.write_state("stopped_by_request", note="stop during backoff")
                            return 2
                        time.sleep(5)

                    relaunch2, why2 = self.should_relaunch()
                    if relaunch2:
                        self.spawn()
                        self.write_state("running", relaunched=True)
                    else:
                        self.write_state("gave_up", reason=why2)
                        self.emit("no_relaunch", reason=why2)
                        return 3
                else:
                    # periodic state flush so the agent monitor sees freshness
                    if time.time() - last_state_flush > 60:
                        rsp = self.run_state_path()
                        step = None
                        rs = self.read_json_tolerant(rsp) if rsp else None
                        if rs:
                            step = rs.get("step")
                        self.write_state("running", step=step, health=detail)
                        last_state_flush = time.time()
        finally:
            self.release_lock()


def find_active_batch_dir(explicit: str | None) -> Path | None:
    """Locate the batch dir a watchdog is (or was) running in.

    Order: explicit arg > ACTIVE file > newest batch dir with watchdog_state.json.
    """
    if explicit:
        p = Path(explicit).resolve()
        return p if p.exists() else None
    try:
        from reforge.config import OUTPUT_DIR
        root = Path(OUTPUT_DIR).resolve() / "persistence_training"
    except Exception:
        return None
    if not root.exists():
        return None
    candidates = []
    # a pending-discovery watchdog writes its state at the root itself
    if (root / STATE_NAME).exists():
        try:
            candidates.append((time.time() - (root / STATE_NAME).stat().st_mtime, root))
        except OSError:
            pass
    for d in root.iterdir():
        if d.is_dir() and (d / STATE_NAME).exists():
            try:
                age = time.time() - (d / STATE_NAME).stat().st_mtime
            except OSError:
                continue
            candidates.append((age, d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def cmd_stop(batch_dir_arg: str | None) -> int:
    batch_dir = find_active_batch_dir(batch_dir_arg)
    if not batch_dir:
        print("No watchdog state found. Pass --batch_dir to target a specific batch.")
        return 1
    stop_path = batch_dir / STOP_NAME
    state = None
    try:
        state = json.loads((batch_dir / STATE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    stop_path.write_text(now_iso(), encoding="utf-8")
    status = (state or {}).get("status", "unknown")
    child = (state or {}).get("child_pid")
    print(f"Stop requested: {stop_path}")
    print(f"Watchdog status: {status} (child pid {child})")
    print("It will send a graceful stop, save the checkpoint, and exit without relaunching.")
    print(f"Watch progress:  python -m reforge.watchdog status")
    return 0


def cmd_status(batch_dir_arg: str | None) -> int:
    batch_dir = find_active_batch_dir(batch_dir_arg)
    if not batch_dir:
        print("No watchdog state found.")
        return 1
    try:
        state = json.loads((batch_dir / STATE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"State file unreadable at {batch_dir / STATE_NAME}")
        return 1
    age = time.time() - state.get("updated_epoch", 0)
    print(f"Batch:      {batch_dir}")
    print(f"Watchdog:   {state.get('status')} (pid {state.get('pid')}, state {age:.0f}s old)")
    print(f"Child:      pid {state.get('child_pid')} | step {state.get('step')} | {state.get('health', '')}")
    print(f"Failures:   {state.get('consecutive_failures', 0)} consecutive")
    if (batch_dir / STOP_NAME).exists():
        print("Stop:       REQUESTED (graceful shutdown in progress)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Reforge training watchdog v2",
                                epilog="Examples:\n"
                                       "  python -m reforge.watchdog start --batch manifest.json --batch_dir ... --base_model ...\n"
                                       "  python -m reforge.watchdog stop\n"
                                       "  python -m reforge.watchdog status",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("control", nargs="?", default="start", choices=["start", "stop", "status"],
                   help="start (default): supervise training | stop: graceful stop | status: show state")
    p.add_argument("--batch")
    p.add_argument("--batch_dir")
    p.add_argument("--base_model")
    p.add_argument("--agent_log")
    p.add_argument("--stop_file")
    p.add_argument("--startup_grace", type=int, default=STARTUP_GRACE_S)
    p.add_argument("--stale_threshold", type=int, default=STALE_THRESHOLD_S)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.control == "stop":
        return cmd_stop(args.batch_dir)
    if args.control == "status":
        return cmd_status(args.batch_dir)
    if not args.batch or not args.base_model:
        p.error("start requires --batch and --base_model")
    return Watchdog(args).run()


if __name__ == "__main__":
    sys.exit(main())
