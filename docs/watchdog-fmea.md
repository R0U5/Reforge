# Watchdog Failure-Mode Analysis (FMEA)
Project: Reforge training supervisor — 2026-11-27
Context: 12GB RTX 4070, Windows 11, batch LoRA training, history of silent deaths.

## Incident that triggered this
2026-11-28 ~03:46 UTC: training died at step 438/2500. No error event, no
early_stop event, batch runner died too. run_state.json updated_at froze.
Likely CUDA/driver OOM or external termination. ~3h of steps lost because
resume machinery was never trusted-tested before.

## Architecture
- **Watchdog** (supervisor, this file) owns the batch runner as a child process.
- Health = child alive AND run_state.json heartbeat fresh (with startup grace).
- Watchdog writes its own heartbeat (`watchdog_state.json`) so the agent's
  automation monitor can supervise the supervisor.
- The agent monitor (automation, every 10 min) reads watchdog_state.json:
  stale watchdog → agent relaunches watchdog itself (watchdog of watchdog).

## Scenario matrix

| # | Scenario | Detection | Mitigation | Relaunch? |
|---|----------|-----------|------------|-----------|
| 1 | CUDA OOM / driver kill (the 438 incident) | child exit + no terminal event | diagnose via last events + nvidia-smi; resume from checkpoint-last | YES, backoff |
| 2 | Windows silent kill (OOM, AV, job object) | run_state stale > threshold while child alive, or child gone | same as #1 | YES, backoff |
| 3 | Early stop / stage complete (clean) | `run_complete`/`training_complete` in events or batch_state stage status | do NOT restart; batch runner advances itself; watchdog follows batch_state | NO |
| 4 | **Manual stop (user wants GPU for games)** | `stop_requested` file in batch dir | watchdog sends CTRL_BREAK, waits ≤90s for checkpoint save, escalates taskkill, exits code 2, never relaunches | NO — never |
| 5 | User launches a game while training runs (GPU contention) | before each relaunch: free VRAM < 1.5GB via nvidia-smi | wait (poll every 60s) instead of fighting; log `waiting_gpu_free` | YES but deferred |
| 6 | Watchdog itself dies | watchdog_state.json stale (agent monitor) | agent monitor relaunches watchdog; child may be orphaned → monitor kills orphan first | YES |
| 7 | Relaunch storm (crash loop, e.g. dataset corrupt) | consecutive_failures counter | exponential backoff (30s→2m→8m→30m cap) + max 5 consecutive; after that exit code 3 and mark batch needs_attention | NO after limit |
| 8 | Child hangs (CUDA deadlock, dataloader stall) but process alive | run_state stale > threshold | grace, then CTRL_BREAK → escalate kill → resume | YES |
| 9 | False stale during tokenization/startup (no heartbeat yet) | startup grace window (default 20 min, covers 15k-row tokenize) | don't judge stale until first heartbeat OR grace expired | n/a (prevents false relaunch) |
| 10 | Partial/corrupt run_state.json (died mid-write) | JSON parse fails | tolerant read: retry 3×, then fall back to file mtime | YES |
| 11 | Clock skew / weird timestamps | updated_at in the future or unparseable | fall back to file mtime of run_state.json | YES |
| 12 | Disk full (checkpoint write fails → crash loop) | free space < 5GB on output drive before relaunch | refuse relaunch, exit 3, alert | NO |
| 13 | Power loss / reboot mid-run | everything gone after boot | resume from checkpoint; watchdog relaunched by agent monitor (session start check) | YES on boot |
| 14 | Orphaned child outlives watchdog | watchdog startup: scan for existing reforge python with batch_dir in cmdline | adopt (reattach monitoring) or kill if stale | adopt |
| 15 | Two watchdogs racing (double launch) | lock file with PID in batch dir (`watchdog.lock`) | second instance exits immediately | n/a |
| 16 | batch_state.json says batch complete but child exited 0 | read batch_state before relaunch | exit 0 clean | NO |
| 17 | Agent log grows unbounded | size check each heartbeat | rotate at 50MB (rename .1) | n/a |
| 18 | KeyboardInterrupt at console | signal handler | same as manual stop: graceful child shutdown, exit 2 | NO |
| 19 | CTRL_BREAK ignored by hung child | wait 90s after signal | taskkill /T /F escalation (last resort — checkpoint may be stale but checkpoint-last is still valid) | YES |
| 20 | Watchdog blocked on unreadable log/state (AV lock) | all file ops wrapped, timeouts | treat as transient, retry next cycle | n/a |

## Exit codes
- 0 = batch complete / nothing to do
- 2 = stopped by request (stop file, SIGINT, manual)
- 3 = gave up (restart limit, disk, unrecoverable) — needs human
- other = child's exit code propagated

## Rules
1. NEVER relaunch when stop_requested exists or a clean completion is recorded.
2. ALWAYS prefer waiting over killing when GPU is contended.
3. Every decision is written to watchdog_state.json + NDJSON events. No silent deaths, including the watchdog's own.
