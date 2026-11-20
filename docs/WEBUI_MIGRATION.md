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
> **Removed 2026-11-17** (user decision): dashboard.py deleted, `reforge
> dashboard` exits 2 with a pointer to `reforge webui`, the matplotlib
> `[dashboard]` extra dropped, README/AGENTS.md scrubbed. The web UI is the
> dashboard; the "ported from the Tk dashboard" comments in webui*.py are
> provenance notes for the math/logic origins. The checklist below is kept
> as the parity record.
Launch: `reforge dashboard`. Child process = `python -m reforge ... --no_tqdm
--agent --agent_log <out>/agent.ndjson`; reader threads parse stdout NDJSON
(events → progress panel) + stderr loss dicts (LossLineLogger rows: step, loss,
lr, grad_norm, epoch, acc — acc present on DPO runs only).

Feature inventory (each row = something the web UI must eventually do):
- [x] Config editor cards: Paths (base model, output dir, dataset, HF id),
      Training (epochs / fixed steps), Performance (batch, grad accum),
      Flags (force/latex/code/image), DPO (force + beta + hint),
      Batch (stage list editor, ↑↓ reorder, manifest save/load JSON).
      *Manifest save happens automatically (build_argv writes
      `<out>/batch_manifest.json`); loading an existing manifest file is
      consciously deferred — paste-stages-by-hand covers it for now.*
- [x] Command preview (live-built argv, backslash-continued display).
      *Debounced 400 ms → POST /api/preview (strict=False), shows the
      validation message inline when the form is incomplete/invalid.*
- [x] Start validations (dataset exists & .parquet, base model required,
      stage datasets exist, int fields ≥ 1, dpo_beta > 0).
      *Server-side in build_argv; failures return 409 + message shown in the
      page banner, job stays idle.*
- [x] Start/Stop: Popen with `CREATE_NEW_PROCESS_GROUP` on Windows;
      graceful stop = `CTRL_BREAK_EVENT` (POSIX: SIGINT); status "Stopping (saving)…".
- [x] Status chip: Idle → Starting… → Preparing… → Training… → Exporting… →
      Complete / Failed / Interrupted (saving)….
- [x] Progress panel (tqdm/Docker-inspired): phase segments
      DATASET→MODEL→TRAIN→MERGE (pending dim / active gold spinner /
      done green / failed red), bar with % caption + 10% ticks,
      indeterminate sweep animation, tqdm info line
      `step 450/1000 · 45.0% · [03:12<03:54, 1.93 it/s]`, ETA chip,
      `STAGE k/N` chip for batch (per-stage reset).
- [x] Live loss chart: raw (faint) + smoothed (gold, moving-average k adaptive)
      + best-point marker; decimation at >4000 points; footer
      `step N | now X | best Y @ step`.
- [x] Stat chips: STEPS / LOSS / BEST / ETA (plus RATE, STAGE, MODE extras).
- [x] Console echo: child stdout/stderr mirrored to the log pane,
      suppression patterns (FutureWarning etc.), follow toggle.
      *Tk's formatted step/loss/lr/gnorm table is not ported — agent-mode
      stdout already carries the data as events; consciously deferred.*
- [x] Full Analysis window: smoothed slope (global + recent weighted), SNR
      (1.4826·MAD), exposure %, grade A–D (`calculate_grade`), trend labels,
      outlier flags (>1.5σ of window), epoch boundaries (derived from
      training_started's `epochs`), ideal-zone prediction.
      *Ported as SERVER-side numpy (`_analysis()` in webui.py, verbatim math
      from dashboard.py) exposed at GET /api/analysis — simpler and safer
      than a JS reimplementation; UI renders cards + detail line.*
- [ ] Load JSONL / trainer_state.json import; Clear. → **Consciously
      deferred**: no /api/import endpoint; agent-mode runs always provide
      live data. Revisit if offline analysis is ever needed.
- [x] Theme: Hermes dark — bg #101014, cards #1a1a2e, inputs #202032,
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

### Phase 2 — Launch & supervise child ✅ (2026-11-16, done + tested)
- [x] Create `src/reforge/webui_config.py`: `build_argv(params) -> list[str]`
      (port `_build_cmd` + `_build_batch_cmd` from dashboard.py; append
      `--no_tqdm --agent --agent_log <out>/agent.ndjson`; create out dir;
      validate: base model + dataset required, .parquet exists, ints ≥ 1,
      dpo_beta > 0). ~~Unify candidate for the Tk dashboard later (D9).~~
      → **Unified 2026-11-17**: the Tk dashboard now delegates to
      `build_argv` via `ReforgeDashboard._collect_params`
      (strict=False for the live preview, strict=True + optional
      `manifest_path` for Start); `_build_cmd`/`_build_batch_cmd`/
      `_append_agent_args`/`_parse_int_field` and the duplicated
      `_start_training` validation block were removed. Smoke-tested the
      real dashboard class (window withdrawn): preview incl.
      incomplete-form message, all validation paths, fake-Popen happy
      path, batch manifest default + custom Save-Manifest path.
- [x] Wire `_spawn` in webui.py to it (real Popen with the readers/reaper)
- [x] `POST /api/preview` returning the argv without launching (strict=False)
- [x] Bonus: `GET /api/models` (scans `~/HF_Models` + HF hub cache for the
      datalist)
- [x] Test: real short run (`--steps 2` tiny model) end-to-end via API only —
      passed twice (fresh sandbox; graceful stop accepted). NOTE: always set
      `REFORGE_OUTPUT_DIR` to a temp sandbox or the child chains from the real
      `~/Reforge_Output/merged` (existing chain design, same as Tk).

### Phase 3 — Frontend: config + progress ✅ (2026-11-16, done + tested)
- [x] INDEX_HTML full layout: header, config cards (§3.3 inventory), command
      preview, Start/Stop buttons, status chip
- [x] Progress panel: phase segments, bar (determinate + CSS indet sweep),
      tqdm info line, ETA + STAGE chips, stat chips
- [x] Log pane (monospace tail, follow toggle instead of autoscroll-on)
- [x] Results view: done cards (merged_dir/final/best/steps/time) + report
      key-values via GET /api/report, error banner + error_code + hint +
      interrupted note, "Start another run"
- [x] Test: browser-less (Node 26): full script `node --check` + pure
      helpers (smooth/fmtHMS) + real `tick()` driven through
      idle→running→stopping→done→error with realistic /api/status payloads
      (DOM stubbed). Manual browser test still worth a pass at Phase 6.

### Phase 4 — Chart & analysis ✅ (2026-11-16, done + tested)
- [x] Loss chart `<canvas>`: raw (faint gold) + moving-average smoothed
      (gold, adaptive k = max(3, n/50|1)) + best marker, decimation at
      >4000 points, dark grid + y labels (D2: no matplotlib)
- [x] Full Analysis panel: grade/trend/SNR/exposure/outliers/ideal-zone —
      implemented as a SERVER-side numpy port (`_analysis()` in webui.py,
      math verbatim from dashboard.py `_full_analysis`) at
      `GET /api/analysis`; epoch exposure reconstructed from
      `training_started.epochs` × step/total (step events carry no epoch)
- [x] Bonus: `GET /api/report` (cached training_report.json reader) +
      `Job.epochs` / `train_elapsed` / final_loss+best_loss+time_sec in the
      training_complete result
- [x] Test: synthetic 60-step run with injected outlier folded through
      `Job.apply_event`, then asserted via HTTP (exposure 20% from
      epochs=2×60/100, outlier flagged, grade/trend labels sane;
      underflow error path; report reader happy/empty paths)

### Phase 5 — Batch editor & parity polish ✅ (2026-11-16, done; 2 deferrals)
- [x] Batch stage editor (add/edit/remove/↑↓ reorder; per-stage name/dataset/
      epochs/steps; dataset↔hf_dataset auto-detected from the path shape).
      *Manifest save is automatic (`build_argv` writes
      `<out>/batch_manifest.json`); loading an existing manifest file is
      deferred (was "file dialogs → textareas + /api endpoints").*
- [x] Import JSONL / trainer_state.json → **consciously deferred** (see
      §3.3): no /api/import; agent mode always yields live data.
- [x] Parity checklist walk (§3.3) — every box ticked or consciously
      deferred (console formatting table + manifest load are the deferrals).
- [ ] README section + pyproject entry point → moved into Phase 6 ship.

### Phase 6 — Ship ✅ (2026-11-17)
- [x] PR #24 against main; Tk dashboard stays; parity deferrals noted in PR body
- [x] Update README dashboard section + structure tree (web UI bullet,
      `reforge webui` quick-start line, `webui.py`/`webui_config.py` in tree)

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

### Session 2 — 2026-11-16 (template deep dive + fixes + PR #23)
- Chat-template deep dive (user request): found and FIXED three real issues
  (see commits d20f974, da9298e, d3fddf0):
  1. Mask boundary was length-only — now `_align_boundary` verifies the
     prompt render as a token-prefix of the full render and falls back to a
     content-token anchor (verified live on Qwen3.5-9B: seam divergence at
     `防`, token 198 vs 271). Empirical test script:
     `%TEMP%\opencode\template_boundary_test.py` / `template_fix_test.py`.
  2. Merged models had no `generation_config.eos_token_id` when the base
     ships no generation_config.json → HF generate() never stops; pinned at
     merge (`__main__.py`, only when None).
  3. Template-less tokenizers (Phi-2) crashed `apply_chat_template` inside
     `.map()`; SFT/chat now gate on `chat_template` → causal fallback + warn
     (`tokenization.py` + `dataset.py`).
- Committed the user's coherent WIP with permission: Qwen3.5 multimodal
  loading + chunked CE + REFORGE_LR/REFORGE_GRAD_ACCUM overrides
  (d3fddf0), overlong-row drop + template fixes (da9298e). `.bak` files
  remain untracked.
- Pushed `feature/web-ui`; opened **PR #23 stacked on #22**
  (feature/batch-training → main). After #22 merges, re-target/merge #23.
- Working tree now clean except untracked `.bak` files. Next: web UI
  Phase 2 per §9.

### Session 3 — 2026-11-16/17 (Phases 2–5 complete: full web UI)
- Synced main (PRs #22 + #23 merged, `33aacd4`), branched `feature/web-ui-2`.
- **Phase 2** (commit `2c801ff`): `webui_config.py` (`build_argv` port of the
  Tk `_build_cmd`/`_build_batch_cmd` with all validations; strict=False
  powers `/api/preview`; batch stages auto-write `<out>/batch_manifest.json`;
  `scan_models()` for the datalist), `_spawn` wired to a real Popen,
  `/api/preview` + `/api/models`. Tested: unit cases + a REAL end-to-end run
  with a purpose-built tiny model (`%TEMP%\opencode\tiny_model`, 84 MB,
  2-layer Qwen2, real Qwen3.5 tokenizer — vocab 248085) + 4-row parquet:
  agent events → phases → merge → run_complete, loss ring filled,
  success.txt + training_report.json written, graceful stop accepted. Log
  confirmed the eos-pin fix firing (`pinned to tokenizer eos (248046)`).
- **⚠ Side effect (disclosed to user):** the FIRST e2e run had no
  `REFORGE_OUTPUT_DIR` sandbox, so reforge's chain feature made the child
  train from the real `~/Reforge_Output/merged` and baked a negligible
  2-step tiny-LoRA delta into it. Existing chain design (Tk does the same);
  all later tests sandbox via `REFORGE_OUTPUT_DIR`.
- **Phases 3–5** (uncommitted at day's end → committed this session):
  full SPA in `INDEX_HTML` — setup view (Paths/Training/Performance/Flags+DPO/
  Batch cards, datalist from /api/models, debounced /api/preview live command
  preview), progress view (phase segments w/ spinner/done/failed, determinate
  + indeterminate bar, tqdm line, STEPS/LOSS/BEST/ETA/RATE/STAGE/MODE chips,
  canvas chart raw+smoothed+best+decimation, analysis cards + detail line,
  log pane with follow toggle), results view (done cards + report kv, error
  banner + error_code + hint + interrupted note, "Start another"). Server
  additions: `Job.epochs`, `train_elapsed`, final/best loss + time in the
  training_complete result, `_analysis()` (verbatim numpy port of the Tk
  `_full_analysis`: polyfit slopes, 1.4826·MAD SNR, exposure from
  reconstructed epochs, grade A–D, trend labels, outliers, ideal-zone) at
  `GET /api/analysis`, `_read_report()` at `GET /api/report`. Batch stage
  editor gained ↑↓ reorder (Tk parity).
- Testing without a browser: Node 26 — extracted the inline `<script>`,
  `node --check`, drove the real `tick()` through idle→running→stopping→done
  →error with realistic /api/status payloads (stubbed DOM + fetch) and
  asserted view routing, stats chips, indet bar, report kv, error banner.
  Two SPA bugs found & fixed this way: results view re-rendered every 1 s
  tick (flicker + /api/report refetch); "server unreachable" chip showed
  "Idle". Node functional test also validated `smooth()`/`fmtHMS()`.
- Final re-run of the full e2e on the finished file: PASS (after fixing a
  test-env drift: a leftover sandbox chain file made the child resume with
  global_step 4 ≥ max_steps 2 → zero training steps → empty loss ring;
  tests now rmtree the sandbox first).
- Visual pass 2026-11-17 (headless Edge screenshots against the live server,
  real tiny runs): setup / progress / results / interrupted views verified.
  Found & fixed:
  1. **Graceful stop was not graceful on Windows** — CTRL_BREAK_EVENT killed
     the child with 0xC000013A, no run_interrupted, no state save (the Tk
     dashboard's stop had the same flaw). Fix: `__main__.py` now installs
     SIGBREAK → default_int_handler on Windows so CTRL_BREAK raises
     KeyboardInterrupt → run_interrupted → adapter saved → clean exit.
     Verified live: stop mid-run → state done + interrupted note; also
     verified `run_interrupted`, "Interrupted state saved", merge/export in
     the console tail.
  2. Console pane flooded by raw stderr loss-dict rows → dropped in
     `_reader_stderr` (same rule as the stdout reader; data arrives as
     step events). Console now shows compact `» event` echoes.
  3. "Merged to" path clipped → `word-break:break-all` path style.
  4. Steps card showed planned steps on a stopped run → now `step/total`.
  5. Interrupted note added to the complete (not just error) results view;
     boot-time preview hint; gold checkbox accents; Tk reaper shows
     "Done" (green) for exit 130 stops.
- Tk dashboard removed (user decision, 2026-11-17): `git rm
  dashboard.py`; `reforge dashboard` now exits 2 with a pointer to
  `reforge webui`; `[dashboard]` (matplotlib) extra dropped from
  pyproject (`[emoji]` kept — dataset.py uses it optionally); README +
  AGENTS.md scrubbed. DPO accuracy visibility preserved through the
  removal: agent `step` events now carry `acc` (rewards/accuracies when
  present) and the web UI renders a gold ACC chip (agent.py schema docs
  updated too). Provenance comments in webui*.py kept.
- Post-removal sweep: requirements.txt no longer lists matplotlib/scipy
  as Tk-dashboard extras (scipy was unused by the web UI too); stale
  training.py comments updated (loss lines are for console workflows -
  the chart is fed by agent step events). Live "dashboard" references now
  only describe the web dashboard or the friendly removal pointer.
- Parity walk done: everything in §3.3 ticked except two conscious
  deferrals (JSONL/trainer_state import — no /api/import; Tk's formatted
  console table; batch manifest file-load). The Tk dashboard deprecation
  decision has since been made: it was removed (see the removal entry
  above).

### Session 4 — 2026-11-19 (full-repo review: confirmed bugs fixed)

Read-through of the whole package; every change below was re-verified with
the existing suites. No new runtime dependencies; the web UI stays
stdlib-only.

**Web UI supervisor / SPA**
- `chip()` called `classList.add("")` for idle/unreachable — a real-browser
  `SyntaxError` that aborted the first tick (so "Server unreachable" never
  rendered). Empty classes are now skipped.
- Stop semantics: `request_stop()` moves running→stopping atomically (a
  second CTRL_BREAK cannot hit the save path); Start and the state check are
  serialized by `_SPAWN_LOCK`; a stop during the child's imports (Windows
  exit `0xC000013A`) or a SIGINT exit now settles as `done` + `interrupted`
  instead of "Failed".
- Interrupts handled outside the training loop now emit `run_interrupted`
  and exit 130 (schema v5), not `run_failed` / `0xC000013A`. Ctrl+C on the
  server also no longer orphans a child in its own process group: shutdown
  waits (120 s) for the save/export, then terminates.
- NaN/Inf loss/acc values are dropped from the live state; `/api/status`,
  `/api/analysis` and `training_report.json` are strict JSON (a diverged run
  can no longer break `JSON.parse`, the chart, or the report min/max).
- Batch events (`batch_failed`, `stage_failed`, `batch_interrupted`,
  `batch_complete`) are folded into the Job snapshot; setup failures enable
  the bridge before emitting `batch_failed`; per-stage resets clear
  loss/acc/best/rate; the report cache invalidates on mtime+size; request
  bodies are validated and size-capped.

**Batch**
- Global CLI flags now actually apply to every stage (the README contract),
  inline lists and manifests alike, dry-run included; per-stage values win.
- `manifest_effective.json` records the resolved plan, so
  `--batch_dir <dir>` alone resumes with identical options (inline lists
  included). Web UI batch manifests bake global Performance/Flags form
  values into every stage.
- `image` and `train_lm_head` are valid stage keys.

**Training / data / reporting**
- `--latex --code` is parsed as a combined cleaning mode (`latex+code`);
  previously both formatters were stripped.
- Multimodal: two-turn Phi-3-Vision prompt with the answer supervised,
  `image_sizes` passed through the collator, overlong rows dropped instead
  of truncated mid-`<|end|>`, clearer drop warnings.
- `apply_chat_template` works on transformers versions without
  `return_dict`; every SFT/chat/measurement call site uses the wrapper.
- Checkpoint cleanup can never mask a training failure; emergency saves go
  to `<output_dir>/emergency_adapter`, never `merged/`.
- All rows dropped after tokenization now raises a clear error.
- Non-finite losses are filtered from report math and the eval display.

**Verification**: compileall + pyflakes clean; extracted SPA passes
`node --check` and the idle→running→stopping→done→error state test; phase
2/3-4 endpoint suites; tiny-model e2e (train→merge→done); early-stop e2e
(done + interrupted, exit `0xC000013A`); batch global-defaults /
effective-manifest / `--batch_dir` dry-run-resume tests; `latex+code` and
single-mode cleaning tests; multimodal mask/image_sizes test; interrupt
handlers exit 130 for single and batch runs. Pushed to PR #25.

### Session 0 — 2026-11-15
- Researched reforge (backend, agent contract, Tk dashboard inventory) and
  Lethe's `ui.py` (stdlib server + inline SPA + Job snapshot + polling).
- Wrote this guide; locked decisions D1–D11; planned Phases 0–6.
- No code written yet for the web UI.

## 9. Next immediate steps (start of next session)
1. PR #25 (`feature/web-ui-2`) is open with the web UI + this review sweep;
   the user merges it. No further code changes are queued here.
2. Known deliberate limitations (not bugs): no `/api/import` (JSONL /
   trainer_state analysis), no Tk-style formatted console table, no
   batch-manifest file-load in the SPA.
3. Watch item: a batch interrupt forwards CTRL_BREAK and waits only 60 s
   before killing the stage child. A large-model CPU merge can exceed that,
   leaving a partial `merged/`; batch resume wipes and retrains that stage
   (correct, but slow). Raise the timeout if this shows up in practice.
4. Optional polish: manual browser pass (success + failure paths).
   ~~unify webui_config.build_argv with dashboard.py's builders (D9)~~ →
   **done 2026-11-17**: dashboard.py delegates to build_argv (see Phase 2).
   (dashboard.py has since been removed.)
