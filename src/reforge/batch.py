"""Batch mode - sequential multi-stage training with chained merges.

Runs a list of datasets in order (SFT stage 1 -> SFT stage 2 -> DPO stage,
or any mix). Each stage is a complete, normal Reforge run in its own
subdirectory; the next stage trains on the previous stage's merged model.

    python -m reforge --batch sft1.parquet sft2.parquet dpo.parquet --base_model /path/to/base
    python -m reforge --batch manifest.json --base_model /path/to/base

On-disk layout (batch_dir):

    batch_<timestamp>/
    |-- batch_state.json      resume anchor: per-stage status
    |-- stage_01/             normal reforge output dir (checkpoints + merged/)
    |-- stage_02/
    +-- ...

Resume: rerun with --batch_dir pointing at the existing batch directory.
Completed stages are skipped (verified via merged/success.txt); the
interrupted stage resumes from its own checkpoint-last. A stage whose merge
was interrupted (merged/ without success.txt) is wiped and restarted - its
recorded base was overwritten, so its checkpoints are unusable.

Disk: stage N's merged is deleted once stage N+1 completes (only the
predecessor of the running stage is ever needed). "keep_stage_merges": true
in the manifest opts out.
"""

import json
import os
import signal
import subprocess
import sys
import time

from .config import OUTPUT_DIR
from .display import _banner, _info, _ok, _warn, _err
from .agent import BRIDGE, classify_error

# Per-stage keys that map 1:1 to child CLI flags.
_STAGE_FLAGS = {
    "force": "--force",
    "latex": "--latex",
    "code": "--code",
    "dpo": "--dpo",
    "no_tqdm": "--no_tqdm",
}
_STAGE_VALUES = {
    "epochs": "--epoch",
    "steps": "--steps",
    "batch_size": "--batch_size",
    "grad_accum": "--grad_accum",
    "dpo_beta": "--dpo_beta",
}
_STAGE_KEYS = set(_STAGE_FLAGS) | set(_STAGE_VALUES) | {"dataset", "hf_dataset", "name"}


# --------------------------------------------------------------------- manifest


def load_manifest(batch_args):
    """Turn --batch CLI values into a manifest dict.

    A single .json path is a manifest file; anything else is an inline list
    of datasets (per-stage overrides come from the global CLI flags).
    """
    if len(batch_args) == 1 and batch_args[0].lower().endswith(".json"):
        path = batch_args[0]
        if not os.path.isfile(path):
            raise _BatchError(f"Batch manifest not found: {path}")
        try:
            # utf-8-sig tolerates a BOM (Notepad / PowerShell edits on Windows).
            with open(path, "r", encoding="utf-8-sig") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise _BatchError(f"Batch manifest is not valid JSON: {path} ({e})")
        if not isinstance(manifest, dict):
            raise _BatchError("Batch manifest must be a JSON object with a 'stages' list")
        manifest.setdefault("_manifest_path", os.path.abspath(path))
        return manifest

    stages = []
    for i, arg in enumerate(batch_args):
        if arg.lower().endswith(".json"):
            raise _BatchError(
                "A .json manifest must be the only --batch argument; "
                "mixing manifests and inline datasets is ambiguous")
        stages.append({"dataset": arg, "name": f"stage_{i + 1:02d}"})
    return {"stages": stages}


def validate_manifest(manifest, base_model):
    """Validate + normalize. Returns (stages, warnings). Raises _BatchError."""
    warnings = []
    stages_raw = manifest.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise _BatchError("Batch manifest needs a non-empty 'stages' list")

    stages = []
    for i, raw in enumerate(stages_raw):
        if not isinstance(raw, dict):
            raise _BatchError(f"Stage {i + 1} must be an object")
        unknown = [k for k in raw if k not in _STAGE_KEYS]
        if unknown:
            warnings.append(f"Stage {i + 1}: unknown keys ignored: {', '.join(unknown)}")
        stage = {k: raw[k] for k in _STAGE_KEYS if k in raw}
        stage.setdefault("name", f"stage_{i + 1:02d}")

        if not stage.get("dataset") and not stage.get("hf_dataset"):
            raise _BatchError(f"Stage {i + 1}: needs 'dataset' (a .parquet path) or 'hf_dataset'")
        if stage.get("dataset") and not os.path.isfile(stage["dataset"]):
            raise _BatchError(f"Stage {i + 1}: dataset not found: {stage['dataset']}")
        for key, lo in (("epochs", 1), ("steps", 1)):
            if key in stage:
                try:
                    stage[key] = int(stage[key])
                except (TypeError, ValueError):
                    raise _BatchError(f"Stage {i + 1}: '{key}' must be an integer")
                if stage[key] < lo:
                    raise _BatchError(f"Stage {i + 1}: '{key}' must be >= {lo}")
        for key in ("batch_size", "grad_accum"):
            if key in stage:
                try:
                    stage[key] = int(stage[key])
                except (TypeError, ValueError):
                    raise _BatchError(f"Stage {i + 1}: '{key}' must be an integer")
                if stage[key] < 1:
                    raise _BatchError(f"Stage {i + 1}: '{key}' must be >= 1")
        if "dpo_beta" in stage:
            try:
                stage["dpo_beta"] = float(stage["dpo_beta"])
            except (TypeError, ValueError):
                raise _BatchError(f"Stage {i + 1}: 'dpo_beta' must be a number")
            if stage["dpo_beta"] <= 0:
                raise _BatchError(f"Stage {i + 1}: 'dpo_beta' must be > 0")
        stages.append(stage)

    if not base_model:
        raise _BatchError(
            "Batch mode requires an explicit base model for stage 1 "
            "(pass --base_model or set 'base_model' in the manifest); "
            "batch never chains from a previous merged/ silently")

    return stages, warnings


class _BatchError(Exception):
    pass


# ------------------------------------------------------------------- batch state


def _write_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


def _load_state(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ------------------------------------------------------------------- commands


def _stage_command(stage, base_model, stage_dir, extra_flags=()):
    cmd = [sys.executable, "-m", "reforge"]
    if stage.get("hf_dataset"):
        cmd += ["--hf_dataset", str(stage["hf_dataset"])]
    elif stage.get("dataset"):
        cmd.append(str(stage["dataset"]))
    cmd += ["--base_model", str(base_model), "--output_dir", stage_dir]
    for key, flag in _STAGE_FLAGS.items():
        if stage.get(key):
            cmd.append(flag)
    for key, flag in _STAGE_VALUES.items():
        if key in stage and stage[key] is not None:
            cmd += [flag, str(stage[key])]
    cmd += list(extra_flags)
    cmd += ["--agent", "--agent_log", os.path.join(stage_dir, "agent.ndjson")]
    return cmd


def _stage_env():
    env = dict(os.environ)
    # Make `python -m reforge` work from any cwd (source checkouts).
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing = env.get("PYTHONPATH", "")
    if pkg_parent not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (pkg_parent + os.pathsep + existing) if existing else pkg_parent
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _ndjson_tail(path, event_name, last=1):
    """Last event of a given name from a stage's agent.ndjson (None if absent)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        matches = [e for e in events if e.get("event") == event_name]
        return matches[-1] if matches else None
    except Exception:
        return None


# ---------------------------------------------------------------------- runner


class BatchRunner:
    def __init__(self, manifest, base_model, batch_dir=None, output_dir=None,
                 keep_stage_merges=None, extra_flags=()):
        self.manifest = manifest
        self.base_model = base_model
        self.keep_stage_merges = (manifest.get("keep_stage_merges")
                                  if keep_stage_merges is None else keep_stage_merges)
        self.extra_flags = list(extra_flags)
        self.warnings = []

        root = output_dir or manifest.get("output_dir") or OUTPUT_DIR
        if batch_dir:
            self.batch_dir = os.path.abspath(batch_dir)
            self.resuming = True
        else:
            self.batch_dir = os.path.join(root, time.strftime("batch_%Y%m%d_%H%M%S"))
            self.resuming = False
        self.state_path = os.path.join(self.batch_dir, "batch_state.json")
        self.state = _load_state(self.state_path) if self.resuming else {}
        self.stages, self.warnings = validate_manifest(manifest, base_model)

    # ------------------------------------------------------------------ run

    def run(self):
        os.makedirs(self.batch_dir, exist_ok=True)
        if not self.resuming or not self.state.get("stages"):
            self.state = {
                "version": 1,
                "batch_dir": self.batch_dir,
                "base_model": self.base_model,
                "manifest_path": self.manifest.get("_manifest_path"),
                "keep_stage_merges": self.keep_stage_merges,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stages": [{"index": i, "name": s.get("name"),
                            "dataset": s.get("dataset") or s.get("hf_dataset"),
                            "status": "pending"}
                           for i, s in enumerate(self.stages)],
            }
            _write_state(self.state_path, self.state)
        else:
            _info(f"Resuming batch: {self.batch_dir}")

        BRIDGE.emit(
            "batch_started",
            stage="loading",
            batch_dir=self.batch_dir,
            base_model=self.base_model,
            total_stages=len(self.stages),
            datasets=[s.get("dataset") or s.get("hf_dataset") for s in self.stages],
            resuming=self.resuming,
        )
        for w in self.warnings:
            _warn(w)
        if self.resuming:
            for w in self.state.get("warnings", []):
                if w not in self.warnings:
                    _warn(w)

        _banner("Batch")
        for i, s in enumerate(self.stages):
            ds = os.path.basename(s.get("dataset") or s.get("hf_dataset") or "?")
            _info(f"  [{i + 1:02d}] {s.get('name')}  {ds}  ({self.state['stages'][i]['status']})")

        final_merged = None
        try:
            for i, stage in enumerate(self.stages):
                final_merged = self._run_stage(i, stage)
        except _BatchError as e:
            code, hint = classify_error(str(e))
            BRIDGE.emit("batch_failed", error=str(e),
                        error_code=code or "batch_stage_failed", hint=hint,
                        batch_dir=self.batch_dir, exit_code=1)
            _err(f"Batch failed: {e}")
            return 1
        except KeyboardInterrupt:
            raise

        BRIDGE.emit(
            "batch_complete",
            stage="complete",
            batch_dir=self.batch_dir,
            final_merged=final_merged,
            completed_stages=len(self.stages),
            exit_code=0,
        )
        _banner("Batch complete")
        _ok(f"Final model: {final_merged}")
        _info(f"To resume or inspect this batch later: --batch_dir {self.batch_dir}")
        return 0

    # --------------------------------------------------------------- stages

    def _stage_dir(self, i):
        return os.path.join(self.batch_dir, f"stage_{i + 1:02d}")

    def _run_stage(self, i, stage):
        stage_dir = self._stage_dir(i)
        merged_dir = os.path.join(stage_dir, "merged")
        success_flag = os.path.join(merged_dir, "success.txt")
        state_entry = self.state["stages"][i]
        label = state_entry.get("name") or f"stage_{i + 1:02d}"

        # --- stage merge finished outside our bookkeeping (crash between
        # merge and state write): trust success.txt, mark complete.
        if os.path.isfile(success_flag) and state_entry["status"] != "complete":
            self._mark(i, "complete", finished=time.strftime("%Y-%m-%d %H:%M:%S"))

        if state_entry["status"] == "complete":
            _info(f"[{label}] already complete - skipping")
            return merged_dir

        # --- interrupted mid-merge: merged/ exists without success.txt. The
        # partial merge overwrote this stage's base? No - the base lives in
        # the PREVIOUS stage dir; a partial merged/ here only poisons this
        # stage's own merge output. Wipe the stage dir and start over.
        if os.path.isdir(stage_dir) and os.path.isdir(merged_dir) \
                and not os.path.isfile(success_flag):
            _warn(f"[{label}] partial merge found (no success.txt) - "
                  "wiping the stage dir and restarting this stage from scratch")
            _rmtree(stage_dir)

        base = self.base_model if i == 0 else os.path.join(self._stage_dir(i - 1), "merged")
        if i > 0 and not os.path.isfile(os.path.join(base, "success.txt")):
            raise _BatchError(
                f"[{label}] base for this stage is missing: {base} "
                f"(stage {i} did not complete; cannot proceed)")

        state_entry["status"] = "running"
        state_entry["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_state(self.state_path, self.state)

        ds_label = state_entry.get("dataset") or "?"
        BRIDGE.emit("stage_started", stage="loading", index=i, name=label,
                    dataset=ds_label, stage_dir=stage_dir,
                    base_model=base, batch_dir=self.batch_dir)
        _banner(f"Batch stage {i + 1}/{len(self.stages)}: {label}")
        _info(f"Dataset   : {ds_label}")
        _info(f"Base model: {base}")

        cmd = _stage_command(stage, base, stage_dir, extra_flags=self.extra_flags)
        env = _stage_env()
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(cmd, env=env, creationflags=creationflags)
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            # Forward a break to the child's own process group so its
            # KeyboardInterrupt handler can save state gracefully.
            if sys.platform == "win32":
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            try:
                returncode = proc.wait(timeout=60)
            except Exception:
                proc.kill()
                returncode = proc.wait()
            state_entry["status"] = "interrupted"
            state_entry["returncode"] = returncode
            _write_state(self.state_path, self.state)
            BRIDGE.emit("batch_interrupted", stage="training", index=i, name=label,
                        returncode=returncode, batch_dir=self.batch_dir,
                        hint=f"rerun with --batch_dir {self.batch_dir} to resume",
                        exit_code=130)
            _warn(f"[{label}] interrupted (child exit {returncode}). "
                  f"Resume with: --batch_dir {self.batch_dir}")
            raise

        failed = _ndjson_tail(os.path.join(stage_dir, "agent.ndjson"), "run_failed")
        if returncode != 0 or not os.path.isfile(success_flag):
            state_entry["status"] = "failed"
            state_entry["returncode"] = returncode
            _write_state(self.state_path, self.state)
            error = (failed or {}).get("error") or f"stage exited with code {returncode}"
            BRIDGE.emit("stage_failed", stage="training", index=i, name=label,
                        dataset=ds_label, error=error,
                        error_code=(failed or {}).get("error_code"),
                        hint=(failed or {}).get("hint"),
                        returncode=returncode, batch_dir=self.batch_dir, exit_code=1)
            raise _BatchError(f"{label} failed: {error}")

        self._mark(i, "complete", finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                   merged_dir=merged_dir)
        BRIDGE.emit(
            "stage_completed", stage="exporting", index=i, name=label,
            dataset=ds_label, merged_dir=merged_dir, batch_dir=self.batch_dir,
            training_mode=self._stage_mode(stage_dir),
        )

        # The just-completed stage's predecessor is no longer anyone's base.
        if i > 0 and not self.keep_stage_merges:
            prev_merged = os.path.join(self._stage_dir(i - 1), "merged")
            if os.path.isdir(prev_merged):
                _rmtree(prev_merged)
                _info(f"[{label}] reclaimed disk: removed {prev_merged}")
        return merged_dir

    def _mark(self, i, status, **extra):
        entry = self.state["stages"][i]
        entry["status"] = status
        entry.update(extra)
        _write_state(self.state_path, self.state)

    def _stage_mode(self, stage_dir):
        try:
            with open(os.path.join(stage_dir, "run_state.json"), "r", encoding="utf-8") as f:
                return (json.load(f) or {}).get("training_mode")
        except Exception:
            return None


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------------------------- dry run


def dry_run_batch(manifest, base_model):
    """Validate all batch stages without loading models. Returns (errors, plan)."""
    from .dataset import detect_training_mode
    import pyarrow.parquet as pq

    errors = []
    warnings = []
    stage_plans = []

    try:
        stages, warnings = validate_manifest(manifest, base_model)
    except _BatchError as e:
        return [{"error_code": "batch_manifest_invalid", "message": str(e)}], None

    for i, stage in enumerate(stages):
        entry = {"index": i, "name": stage.get("name")}
        problems = []
        if stage.get("hf_dataset"):
            entry["dataset"] = stage["hf_dataset"]
            entry["training_mode"] = "unknown (HF Hub - resolved at load time)"
        else:
            ds = stage["dataset"]
            entry["dataset"] = ds
            try:
                columns = pq.read_schema(ds).names
                mode = detect_training_mode(columns)
                entry["training_mode"] = mode
                entry["columns"] = columns
            except ValueError as e:
                code, hint = classify_error(str(e))
                problems.append({"error_code": code or "invalid_dataset",
                                 "message": f"Stage {i + 1}: {e}", "hint": hint})
            except Exception as e:
                warnings.append(f"Stage {i + 1}: could not peek columns: {e}")
        rows = None
        if not problems and stage.get("dataset"):
            try:
                rows = pq.read_metadata(stage["dataset"]).num_rows
                entry["rows"] = rows
            except Exception:
                pass
        if not problems:
            batch_size = stage.get("batch_size") or 1
            grad_accum = stage.get("grad_accum") or 8
            if stage.get("steps"):
                entry["steps"] = stage["steps"]
                entry["steps_fixed"] = True
            elif rows is not None:
                spe = max(1, -(-rows // (batch_size * grad_accum)))
                entry["estimated_steps_per_epoch"] = spe
                entry["epochs"] = stage.get("epochs", 1)
                entry["total_steps"] = spe * stage.get("epochs", 1)
            entry["early_stop"] = not stage.get("force")
            if stage.get("dpo_beta") is not None:
                entry["dpo_beta"] = stage["dpo_beta"]
        entry["ok"] = not problems
        stage_plans.append(entry)
        errors.extend(problems)

    if not errors and len(stages) > 1:
        warnings.append(
            "Each stage trains on the previous stage's merged model; "
            "early stopping resets per stage")

    plan = {
        "base_model": base_model,
        "total_stages": len(stages),
        "stages": stage_plans,
    }
    return errors, {"warnings": warnings, "batch": plan}
