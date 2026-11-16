# Reforge Web UI Migration — Master Guide

> **THIS IS THE SESSION ANCHOR DOCUMENT.**
> Start every session by reading this file top to bottom. Work only inside the
> current phase's tasks. Before ending a session: tick completed checkboxes,
> add a Session Log entry, and update "Next Immediate Steps". Never duplicate
> this doc — it is the single source of truth for the migration.

---

## 1. Goal

Replace the Tkinter dashboard (`src/reforge/dashboard.py`, ~80 KB) with a
**local web UI in the browser**, modeled on the UI of
[Lethe](https://github.com/R0U5/Lethe) (`C:\Users\willi\Desktop\Python Projects\lethe`).

Constraints / spirit:
- **Zero new heavy dependencies** — follow Lethe: Python stdlib
  `http.server` + one self-contained HTML page (inline CSS/JS, works offline,
  no npm/node build step). This also drops the matplotlib requirement.
- Everything stays **local-only** (bind 127.0.0.1).
- Feature parity with the Tk dashboard is the finish line;Tk dashboard stays
  working until parity (then deprecation is a separate decision).
- Reuse, don't reinvent: the **agent-mode NDJSON event contract** (agent.py)
  is the data source, exactly as the Tk dashboard already consumes it.

---

## 2. Session protocol

1. `cd C:\Users\willi\Desktop\Python Projects\reforge` — check `git status`,
   `git log --oneline -5`. There may be unrelated uncommitted WIP
   (dataset.py/profiles.py/tokenization.py/training.py + .bak files) — **never
   commit or revert those**.
2. Re-read this doc, especially §5 (decisions), §6 (roadmap), §8 (log).
3. Do work on a feature branch: `feature/web-ui` (create when starting Phase 1).
4. Test what you build (see §7 testing notes). `py_compile` everything touched.
5. Update §6 checkboxes + §8 session log + §9 next steps. Commit (code changes
   separately from doc updates is fine).

---

## 3. Reforge — state of the codebase (verified 2026-11-15)

### 3.1 Backend CLI (src/reforge/__main__.py, ~38 KB)
- Console script `reforge`, module `python -m reforge`. Subcommand dispatch on
  **first argv token**: `dashboard` (Tk UI), `agent-schema` / `agent_schema`
  (prints JSON contract, exit 0).
- Key flags: positional `.parquet`, `--base_model`, `--hf_dataset`, `--epoch`,
  `--steps`, `--batch_size`, `--grad_accum`, `--image`, `--latex`, `--code`,
  `--force`, `--train_lm_head`, `--dpo`, `--dpo_beta`, `--batch <paths|json>`,
  `--batch_dir`, `--output_dir`, `--log_file`, `--agent`, `--agent_log`,
  `--agent_schema`, `--dry_run`, `--no_tqdm`.
- `--log_file` is **incompatible with --agent** (exit 2) — don't combine in the
  web UI launcher.
- Agent mode is **non-interactive**: requires `--base_model` AND a dataset,
  else exit 2. Exit codes: 0 ok, 1 runtime/training failure, 2 invalid
  args/dataset, 130 interrupted.
- Training runs in this process (heavy imports at module load: torch etc.).

### 3.2 Agent event contract (src/reforge/agent.py, ~40 KB)
- `--agent` → **stdout is pure NDJSON** (one JSON obj/line, `{"ts", "event", ...}`);
  human output → stderr. `--agent_log PATH` tees the stream to a file (append
  mode). `<output_dir>/run_state.json` = atomically-rewritten latest snapshot
  (pollable). `reforge --agent-schema` = full self-description JSON.
- Events (fields in agent.py `_EVENT_FIELDS`):
  - run_started, dataset_loaded (rows, training_mode, cleaning_mode,
    mapped_roles), training_started (**total_steps**, epochs, lr, batch_size,
    grad_accum, early_stop, scheduler, warmup_steps, beta for DPO),
  - **step** (step, total_steps, pct, loss, eta_sec) — highest volume, LOG_STEPS=1,
  - checkpoint_saved, early_stop_check (ema/best_ema/since_best/verdict, every
    25 steps), early_stop_triggered, training_complete (steps, final_loss,
    best_loss, time_sec, report), training_failed (error, error_code, hint,
    emergency_save), run_interrupted, merge_started, merge_complete,
    run_complete (merged_dir, exit_code), run_failed (error, error_code, hint,
    exit_code),
  - batch: batch_started (batch_dir, total_stages, datasets, resuming),
    stage_started (index, name, dataset, stage_dir, base_model),
    stage_completed (index, merged_dir, training_mode), stage_failed
    (index, error, error_code, hint, returncode), batch_interrupted,
    batch_complete (final_merged), batch_failed.
- Batch parent (`--batch`) emits batch-level events; each stage child writes its
  own `<stage_dir>/run_state.json` + `<stage_dir>/agent.ndjson`.
- Error codes are machine-branchable (`_ERROR_PATTERNS` in agent.py).
- Artifacts in output_dir: `run_state.json`, `console.log`,
  `checkpoint-<step>/` (+`checkpoint-last` symlink), `last_good_ckpt/`,
  `training_chain.txt`, `merged/` (+`success.txt`, `training_report.json`).
- Batch layout: `output_dir/batch_<timestamp>/{batch_state.json, stage_01..N/}`;
  resume via `--batch_dir`.

### 3.3 Tk dashboard (src/reforge/dashboard.py, ~80 KB) — the feature checklist to port
Launch: `reforge dashboard`. Child process = `python -m reforge ... --no_tqdm
--agent --agent_log <out>/agent.ndjson`; reader threads parse stdout NDJSON
(events → progress panel) + stderr loss dicts (LossLineLogger rows: step, loss,
lr, grad_norm, epoch, acc — acc present on DPO runs only).

Feature inventory (each row = something the web UI must eventually do):
- [ ] Config editor cards: Paths (base model, output dir, dataset, HF id),
      Training (epochs / fixed steps), Performance (batch, grad accum),
      Flags (force/latex/code/image), DPO (force + beta + hint),
      Batch (stage list editor, ↑↓ reorder, manifest save/load JSON).
- [ ] Command preview (live-built argv, backslash-continued display).
- [ ] Start validations (dataset exists & .parquet, base model required,
      stage datasets exist, int fields ≥ 1, dpo_beta > 0).
- [ ] Start/Stop: Popen with `CREATE_NEW_PROCESS_GROUP` on Windows;
      graceful stop = `CTRL_BREAK_EVENT` (POSIX: SIGINT); status "Stopping (saving)…".
- [ ] Status chip: Idle → Starting… → Preparing… → Training… → Exporting… →
      Complete / Failed / Interrupted (saving)…
- [ ] Progress panel (tqdm/Docker-inspired): phase segments
      DATASET→MODEL→TRAIN→MERGE (pending dim / active gold + |/-\ spinner /
      done green / failed red), canvas bar with % caption + 10% ticks,
      indeterminate sweep animation for load/merge phases, tqdm info line
      `step 450/1000  ·  45.0% · [03:12<03:54 · 1.93 it/s]`, ETA chip,
      `STAGE k/N` chip for batch (per-stage reset).
- [ ] Live loss chart: raw (faint) + smoothed (gold, moving-average k adaptive)
      + best-point marker; pixel decimation at >4000 points; title
      `Step N | current | best`.
- [ ] Stat chips: STEPS / LOSS / BEST / ETA.
- [ ] Console echo: child stdout/stderr mirrored to the dashboard's console,
      suppression patterns (FutureWarning etc.), formatted table
      step/loss/lr/gnorm/epoch/acc.
- [ ] Full Analysis window: smoothed slope (global + recent weighted), SNR
      (1.4826·MAD), exposure %, grade A–D (`calculate_grade`), trend labels
      (`trend_label`), outlier flags (>1.5σ of window), epoch boundary lines,
      ideal-zone prediction. Pure numpy math — must be ported to JS or exposed
      as an API endpoint.
- [ ] Load JSONL / trainer_state.json import; Clear.
- [ ] Theme: Hermes dark — bg #101014, cards #1a1a2e, inputs #202032,
      borders #33333f, accent #FFBF00/#FFD700 gold, text #FFF8DC, muted
      #9096ad, ok #4caf50, warn #ffa726, err #ef5350; Segoe UI + Consolas.

### 3.4 Misc
- pyproject v1.2.0; extras: `dashboard = matplotlib` (Tk), `emoji`.
- py-min 3.10. Windows is the primary platform (user machine).
- config.py sets HF env vars, `REFORGE_OUTPUT_DIR` default `~/Reforge_Output`.

---

## 4. Lethe — reference architecture (verified 2026-11-15)

File: `lethe/src/abliterate/ui.py` (~780 lines, self-contained). Read it in
full before writing reforge's server. Key patterns to copy:

- **Server**: stdlib `http.server.ThreadingHTTPServer` + `BaseHTTPRequestHandler`;
  routes: `GET /` (INDEX_HTML), `GET /api/status` (job snapshot), `POST /api/start`
  (validated params → 409 if busy), `POST /api/test`. Silence
  `log_message`. Bind `127.0.0.1` only; `launch(host, port, open_browser)`,
  `--port/--no-browser` flags, `webbrowser.open`.
- **Job state**: dataclass with a lock; `set(**kw)`, `add_log` (ring buffer,
  cap 400), `snapshot()` returns trimmed view (log tail 60). Global `JOB` singleton;
  one active job; start rejected while running.
- **Worker**: daemon thread runs the pipeline; phase callbacks update
  `phase` text + `progress` (0..1) + `indeterminate` flag. Progress bands:
  per-stage fractions mapped onto global bands via `_lerp(band, frac)`
  (e.g. load 0.02–0.20, trials 0.42–0.90). Log lines come from a logging handler.
- **Frontend**: ONE inline HTML string `INDEX_HTML` — no external assets.
  Vanilla JS; `poll()` = `setInterval(tick, 1000)`. Views toggled by classes:
  setup → progress → results. Progress bar: `.bar > i` width transition;
  `.bar.indet` = CSS keyframe sliver animation (never frozen); spinner = CSS.
  Light/dark via `prefers-color-scheme` CSS vars. `esc()` for all interpolation.
  Results view: metric cards + before/after examples + "start another".
- **Error handling**: friendly `_friendly_error()` translation; job thread
  catches everything, sets state=error, keeps server alive.

Differences for reforge (deliberate divergences from Lethe):
1. **Lethe runs the pipeline in-process** (worker thread). Reforge keeps the
   **subprocess model** (spawn `python -m reforge --agent ...`), because
   reforge training is long/GPU-heavy and the codebase's crash-resilience
   (safe streams, CTRL_BREAK stop, checkpoint resume) is built around a
   child process. The server is a thin supervisor: it builds argv, spawns,
   tails stdout (NDJSON) + stderr (loss rows + human log), maps events →
   Job state. Stop = CTRL_BREAK/SIGINT to the child (same as Tk dashboard).
2. **Chart data** comes from `step` events (loss per step) — no matplotlib.
   Draw with `<canvas>`, adaptive moving average in JS.
3. reforge has richer progress (real step/total/pct/eta from events) — no
   fake global bands needed; use event fields directly, sweep only for
   load/merge phases (as the Tk panel already does).
4. Multiple **simultaneous views**: reforge has history (training_report.json,
   past runs in output dirs) — the Tk dashboard only had "Load JSONL"; the web
   UI can list output dirs later (post-parity idea, don't scope-creep MVP).

---

## 5. Design decisions (locked)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | New module `src/reforge/webui.py`, subcommand `reforge webui` (also accept `reforge ui`) | mirrors Lethe's `ui.py`; keeps dashboard.py untouched |
| D2 | Stdlib-only server + single inline INDEX_HTML | Lethe pattern; zero deps, offline |
| D3 | Child process runs training via `--agent --agent_log`; server spawns & tails | subprocess isolation = existing resilience; Tk parity |
| D4 | API: `GET /`, `GET /api/status`, `POST /api/start`, `POST /api/stop`, `GET /api/config/defaults` (later); polling 1 s (SSE only as a later enhancement) | Lethe parity; polling is proven at 250 ms–1 s |
| D5 | Job snapshot fields: state(idle/running/done/error/stopping), phase, phases{dataset,model,train,merge}, step, total_steps, pct, eta_sec, rate_it_s, stage {index,total} (batch), log tail (last ~80 lines), error, exit_code, losses (ring for chart, cap ~5000), result (merged_dir, report, metrics) | superset of Tk panel + Lethe snapshot |
| D6 | Default port **7861** (Lethe owns 7860; both may run side-by-side); `--port`, `--no-browser`, `--host` (default 127.0.0.1) | avoid collision |
| D7 | Dark-first Hermes palette as CSS vars (port from dashboard.py); light scheme optional later | brand continuity |
| D8 | Chart = `<canvas>` + JS moving average; Full Analysis math ported to JS (port `calculate_grade`, `trend_label`, MAD/SNR logic) | zero-dep requirement |
| D9 | argv construction moves to a shared place so web UI and Tk dashboard can't drift: extract `_build_cmd` logic or accept duplication short-term (v1: duplicate in webui.py, tagged TODO-unify) | speed over purity for MVP |
| D10 | `agent.ndjson` tee files are append-across-runs; server must consume the **child's stdout pipe** (not the file) for live state | correctness |
| D11 | pyproject: add no new deps; consider `webui` extra only if ever needed | zero-dep goal |

Open questions (resolve when reached, don't block):
- File pickers: browser can't browse local paths — v1 = text inputs (like
  Lethe's model path input). Optional later: `/api/browse` endpoint (stdlib
  `os.scandir` listing) for models/datasets dirs.
- Datasets on HF hub vs local: keep both fields like Tk dashboard.

---

## 6. Roadmap

Phase numbering = order of attack. Check off when done AND tested.

### Phase 0 — Research & guide ✅ (this session)
- [x] Understand reforge backend + agent contract (§3)
- [x] Understand Tk dashboard features (§3.3 inventory)
- [x] Understand Lethe UI architecture (§4)
- [x] Write this guide

### Phase 1 — Server skeleton ✅ (2026-11-15, done + tested)
- [x] Create branch `feature/web-ui` (branched from `feature/batch-training`
      tip f1c31e7, which includes batch.py — the web UI needs batch events;
      rebase onto main after the batch PR merges)
- [x] `webui.py`: `Job` dataclass (locked fields D5) + snapshot
- [x] `webui.py`: `_Handler` (routes D4), `launch()` (D6), `_report_environment`-style banner
- [x] `__main__.py`: dispatch `webui` / `ui` first-token → `webui.launch()`
      (before argparse; don't break `dashboard`/`agent-schema`)
- [x] Placeholder INDEX_HTML (header + live /api/status poll) — verified in browser-less HTTP test
- [x] Test: `python -m reforge webui --no-browser --port 7898` → GET /api/status
      returned `state: idle`; plus in-process thread test hitting all routes
- [x] py_compile + commit
- Bonus (pulled forward from Phase 2, it was cheap): `Job.apply_event` full
  event state machine (incl. batch `stage_started` per-stage reset),
  `_reader_stdout`/`_reader_stderr`, `_reaper` + `finish()` settlement,
  `POST /api/stop` handler. All tested. `POST /api/start` is a friendly stub
  until `webui_config.build_argv` exists (Phase 2).

### Phase 2 — Launch & supervise child (NEXT SESSION START HERE)
- [ ] Create `src/reforge/webui_config.py`: `build_argv(params) -> list[str]`
      (port `_build_cmd` + `_build_batch_cmd` from dashboard.py; append
      `--no_tqdm --agent --agent_log <out>/agent.ndjson`; create out dir;
      validate: base model + dataset required, .parquet exists, ints ≥ 1,
      dpo_beta > 0). Unify candidate for the Tk dashboard later (D9).
- [ ] Wire `_spawn` in webui.py to it (replace the ImportError stub already
      in place — readers, reaper and stop are DONE, see Phase 1 bonus)
- [ ] Optional: `POST /api/preview` returning the argv without launching
- [ ] Test: real short run (`--steps 5` tiny model) end-to-end via API only

### Phase 3 — Frontend: config + progress
- [ ] INDEX_HTML full layout: header, config cards (§3.3 inventory), command
      preview, Start/Stop buttons, status chip
- [ ] Progress panel: phase segments, bar (determinate + CSS indet sweep),
      tqdm info line, ETA + STAGE chips, stat chips
- [ ] Log pane (monospace tail, autoscroll)
- [ ] Results view: metrics from training_report.json (server reads file on
      run_complete), merged_dir, "Start another"
- [ ] Test: manual browser run; exercise both success and failure paths

### Phase 4 — Chart & analysis
- [ ] Loss chart `<canvas>`: raw + moving-average smoothed + best marker,
      decimation, dark grid styling (port style_analysis_axes aesthetics)
- [ ] Full Analysis panel: grade/trend/SNR/exposure/outliers/epoch lines
      (JS port of dashboard math)
- [ ] Test: replay a recorded agent.ndjson through a mock to render charts

### Phase 5 — Batch editor & parity polish
- [ ] Batch stage editor (add/edit/remove/reorder, manifest save/load via
      file dialogs replaced by textareas + /api endpoints for read/write JSON)
- [ ] Import JSONL / trainer_state.json (upload → /api/import)
- [ ] README section; pyproject entry point if desired (`reforge-webui` script?)
- [ ] Parity checklist walk (§3.3) — every box ticked or consciously deferred

### Phase 6 — Ship
- [ ] PR against main; keep Tk dashboard; note deprecation plan in PR body
- [ ] Update README dashboard section + structure tree

---

## 7. Testing notes

- No pytest infra in reforge yet — use ad-hoc scripts in
  `C:\Users\willi\AppData\Local\Temp\opencode\` (pattern used before:
  construct server/job objects directly, feed fake event dicts, assert state).
- Tk needs no mocking for web UI; instead instantiate the HTTP handler logic
  by calling module functions, or run the real server on a random port in a
  thread and hit it with `urllib.request`.
- Real-run smoke test command (cheap model): verify a local tiny model +
  small parquet exists before promising this; else test failure path with a
  missing dataset (exit 2 → run_failed event → UI error banner).
- Always `python -m py_compile` touched files. Web page: open it manually.

## 8. Session log

### Session 1 — 2026-11-15 (Phase 1 complete)
- Preserved last session's uncommitted Tk progress-panel work: committed
  README.md + dashboard.py on `feature/batch-training` (f1c31e7), then branched
  `feature/web-ui` from it.
- Built `src/reforge/webui.py` (~380 lines): `Job` (D5 field set, lock,
  snapshot), full event state machine via `apply_event` (incl. batch
  `stage_started` per-stage reset of phases/loss ring/error), `_reader_stdout`
  (NDJSON dispatch before the loss filter, same as Tk), `_reader_stderr`,
  `_reaper` + `finish()` (exit-0 → done, matching Tk semantics), `POST
  /api/stop` (CTRL_BREAK/SIGINT), `launch()` (D6: port 7861, --host/--port/
  --no-browser), placeholder INDEX_HTML with live status poll.
- Wired `reforge webui` / `reforge ui` first-token dispatch in `__main__.py`
  (`_launch_webui` parses --host/--port/--no-browser; warns if port == 7860).
- `/api/start` is a friendly stub (returns "Phase 2" message) until
  `webui_config.build_argv` exists.
- Tests (temp script, all passed): real HTTP server in thread — GET /, GET
  /api/status (all D5 keys), POST /api/start stub 409, POST /api/stop 409;
  Job state machine — happy path (phases → all done, pct 100, losses ring,
  result), failure path (train phase → failed red, error/error_code/exit_code),
  batch tracking (stage chip index/total, per-stage reset), finish()
  settlements (exit 0 no-event → done; exit 1 → error "child exited"; stopping
  + 130 → done). Plus real-subprocess check: `python -m reforge webui --port
  7898 --no-browser` → GET /api/status returned state:idle (torch import makes
  startup slow — allow ~25 s in tests; noted as gotcha).
- Test-script gotchas hit (worth remembering): urllib raises on 409 (read
  e.read()); event field `name` collides with test helper kwargs; eta 612 s =
  "10:12" not "03:54".
- Committed on `feature/web-ui`. NOTE: `python -m reforge webui` requires
  PYTHONPATH=src or `pip install -e .` in the target env.

### Session 0 — 2026-11-15
- Researched reforge (backend, agent contract, Tk dashboard inventory) and
  Lethe's `ui.py` (stdlib server + inline SPA + Job snapshot + polling).
- Wrote this guide; locked decisions D1–D11; planned Phases 0–6.
- No code written yet for the web UI.

## 9. Next immediate steps (start of next session)
1. `git -C ... reforge checkout feature/web-ui` (already created; based on
   `feature/batch-training` f1c31e7). Re-check `git status` for new user WIP.
2. Phase 2: create `src/reforge/webui_config.py` with `build_argv(params)`
   (port from dashboard.py `_build_cmd`/`_build_batch_cmd`, incl. validations;
   keep `--no_tqdm --agent --agent_log`), replace the `_spawn` ImportError
   stub in webui.py, then API-only end-to-end test with a real tiny run if a
   small model + parquet is available (otherwise test the failure path).
3. Update §6/§8/§9.
