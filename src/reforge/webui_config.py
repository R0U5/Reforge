"""Argv builder + validation for the Web UI's child training processes.

``build_argv(params)`` turns the JSON body of ``POST /api/start`` into the
``python -m reforge ...`` command line the web UI supervisor spawns.
Validation failures raise ``ValueError`` with a message that is shown to
the user verbatim.

This is the single source of truth for command construction and start
validation (D9; it previously also served the Tk dashboard, which was
removed when the web UI reached parity). ``strict=False`` (used by the
command preview) builds the same argv without filesystem existence
checks, so the preview works while typing. ``manifest_path`` optionally
overrides where a batch manifest is written; the default is
``<output_dir>/batch_manifest.json``.
"""

from __future__ import annotations

import json
import os
import sys

from .config import OUTPUT_DIR

# Numeric fields accepted from the form: name -> (coerce, minimum)
_INT_FIELDS = ("epoch", "steps", "batch_size", "grad_accum")
_FLAG_FIELDS = ("force", "latex", "code", "image", "dpo", "train_lm_head")


def _as_bool(v) -> bool:
    """Be liberal: the browser may send booleans or their string forms."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _parse_int(params: dict, name: str) -> "int | None":
    raw = params.get(name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer (got {raw!r}).") from None
    if val < 1:
        raise ValueError(f"{name} must be ≥ 1 (got {val}).")
    return val


def _parse_dpo_beta(params: dict) -> "float | None":
    raw = params.get("dpo_beta")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"dpo_beta must be a number (got {raw!r}).") from None
    if val <= 0:
        raise ValueError(f"dpo_beta must be > 0 (got {val}).")
    return val


def _resolve_output_dir(params: dict, strict: bool = True) -> str:
    out = str(params.get("output_dir") or "").strip() or OUTPUT_DIR
    if not strict:
        # The live command preview must never create directories as a side
        # effect of typing into the output-dir field.
        return out
    try:
        if os.path.exists(out) and not os.path.isdir(out):
            raise ValueError(f"Output path is not a directory:\n{out}")
        os.makedirs(out, exist_ok=True)
    except ValueError:
        raise
    except OSError as e:
        raise ValueError(
            f"Output directory could not be created:\n{out}\n({e})") from None
    return out


def _validate_stage_list(params: dict, strict: bool) -> "list | None":
    """Batch stages: each needs a dataset (local ones must exist)."""
    stages = params.get("stages")
    if stages is None:
        return None
    if not isinstance(stages, list):
        raise ValueError("stages must be a list of stage objects.")
    if any(not isinstance(s, dict) for s in stages):
        raise ValueError("Each batch stage must be an object.")
    if not stages:
        raise ValueError("Batch mode needs at least one stage.")
    cleaned = []
    for i, s in enumerate(stages):
        # The editor sends every cell on a row edit, including blank
        # epochs/steps/name. Drop blanks here: written verbatim into the
        # manifest they become int("") failures in the child's validation.
        s = {k: v for k, v in s.items() if v not in (None, "")}
        for key in ("dataset", "hf_dataset", "name"):
            if key in s:
                s[key] = str(s[key]).strip()
                if not s[key]:
                    del s[key]
        cleaned.append(s)
        ds = s.get("dataset") or ""
        hf = s.get("hf_dataset") or ""
        if not ds and not hf:
            raise ValueError(f"Stage {i + 1} has no dataset.")
        if strict and ds:
            if not os.path.isfile(ds):
                raise ValueError(f"Stage {i + 1} dataset not found:\n{ds}")
            if not ds.lower().endswith(".parquet"):
                raise ValueError(f"Stage {i + 1} dataset must be a .parquet file.")
    return cleaned


def _resume_batch_dir(params: dict) -> "str | None":
    """A --batch_dir resume target from the form, when it exists on disk."""
    raw = str(params.get("batch_dir") or "").strip()
    if not raw:
        return None
    if not os.path.isdir(raw):
        raise ValueError(f"Batch directory not found:\n{raw}")
    state = os.path.join(raw, "batch_state.json")
    if not os.path.isfile(state):
        raise ValueError(
            f"Not a batch directory (missing batch_state.json):\n{raw}")
    return raw


def build_argv(params: dict, strict: bool = True,
               manifest_path: "str | None" = None) -> "list[str]":
    """Validated child command line; raises ValueError on bad input."""
    if not isinstance(params, dict):
        raise ValueError("Request body must be a JSON object.")
    params = {k: v for k, v in params.items() if v not in (None, "")}

    base = str(params.get("base_model") or "").strip()
    ds = str(params.get("dataset") or "").strip()
    hf = str(params.get("hf_dataset") or "").strip()
    # A --batch_dir resume needs neither the form's stages nor its base
    # model: the child recovers both from the batch's recorded manifest.
    resume_dir = _resume_batch_dir(params)
    if resume_dir:
        stages = None
    else:
        stages = _validate_stage_list(params, strict)
    is_batch = resume_dir is not None or stages is not None

    if is_batch:
        # Batch mode: stage 1's base model comes from the manifest.
        if not resume_dir:
            # A fresh batch needs the form's base model; a resume recovers
            # it (and every stage override) from the recorded manifest.
            if not base:
                raise ValueError(
                    "Batch mode requires a Base Model (stage 1's base; batch "
                    "never chains from a previous merged/).")
            if strict and not os.path.exists(base):
                raise ValueError(f"Base model not found:\n{base}")
    else:
        if not ds and not hf:
            raise ValueError(
                "Select a .parquet file or enter a HuggingFace dataset ID.")
        if not base:
            raise ValueError(
                "A Base Model is required — the dashboard launches training "
                "in agent mode, which is non-interactive.")
        if ds:
            if strict:
                if not os.path.exists(ds):
                    raise ValueError(f"Dataset path does not exist:\n{ds}")
                if not ds.lower().endswith(".parquet"):
                    raise ValueError("Local dataset must be a .parquet file.")

    out_dir = _resolve_output_dir(params, strict=strict)

    # On resume the form's epoch/steps/beta are irrelevant (the recorded
    # manifest wins) — don't let typos in unused fields block a resume.
    if resume_dir:
        numeric = {name: None for name in _INT_FIELDS}
        dpo_beta = None
    else:
        numeric = {name: _parse_int(params, name) for name in _INT_FIELDS}
        dpo_beta = _parse_dpo_beta(params)

    cmd = [sys.executable, "-m", "reforge"]
    if is_batch:
        # Manifest path: the Tk Save-Manifest choice wins; otherwise the
        # default lives in the output dir. Written on strict builds (a real
        # start); a non-strict preview only shows where Start will save it.
        if resume_dir:
            # Resume: the CLI recovers the manifest recorded in the batch's
            # batch_state.json, so no fresh manifest is written here and no
            # --batch argument is needed.
            cmd += ["--batch_dir", resume_dir]
            if strict and str(params.get("output_dir") or "").strip():
                cmd += ["--output_dir", str(params["output_dir"]).strip()]
            cmd.append("--no_tqdm")
            cmd += ["--agent", "--agent_log", os.path.join(out_dir, "agent.ndjson")]
            return cmd
        path = manifest_path or os.path.join(out_dir, "batch_manifest.json")
        if strict:
            # Global Performance/Flags form values apply to every stage
            # (README batch contract); per-stage editor values win.
            stage_defaults = {}
            if numeric["epoch"] is not None:
                stage_defaults["epochs"] = numeric["epoch"]
            for name in ("steps", "batch_size", "grad_accum"):
                if numeric[name] is not None:
                    stage_defaults[name] = numeric[name]
            for name in _FLAG_FIELDS:
                if _as_bool(params.get(name)):
                    stage_defaults[name] = True
            if dpo_beta is not None:
                stage_defaults["dpo_beta"] = dpo_beta
            if stage_defaults:
                stages = [{**stage_defaults, **s} for s in stages]
            def _norm(p: str) -> str:
                # Normalize Windows paths to forward slashes for JSON safety
                return p.replace("\\", "/")

            manifest = {"base_model": _norm(base) if base else None, "stages": []}
            for s in stages:
                stage_copy = dict(s)
                if "dataset" in stage_copy:
                    stage_copy["dataset"] = _norm(stage_copy["dataset"])
                if "hf_dataset" in stage_copy:
                    stage_copy["hf_dataset"] = _norm(stage_copy["hf_dataset"])
                manifest["stages"].append(stage_copy)
            if not manifest["base_model"]:
                manifest.pop("base_model")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                raise ValueError(f"Manifest write failed: {e}") from None
        cmd += ["--batch", path]
        if base:
            cmd += ["--base_model", base]
        if str(params.get("output_dir") or "").strip():
            cmd += ["--output_dir", str(params["output_dir"]).strip()]
    else:
        if base:
            cmd += ["--base_model", base]
        if ds:
            cmd.append(ds)                      # positional dataset
        if hf:
            cmd += ["--hf_dataset", hf]         # hub ID stays on its own flag
        for name in _INT_FIELDS:
            if numeric[name] is not None:
                cmd += [f"--{name}", str(numeric[name])]
        for name in _FLAG_FIELDS:
            if _as_bool(params.get(name)):
                cmd.append(f"--{name}")
        if dpo_beta is not None:
            cmd += ["--dpo_beta", str(dpo_beta)]

    # Agent mode: NDJSON events on stdout drive the UI; the tee file is
    # the durable copy for post-run analysis (D10).
    cmd.append("--no_tqdm")
    cmd += ["--agent", "--agent_log", os.path.join(out_dir, "agent.ndjson")]
    return cmd


def scan_models() -> "list[str]":
    """Local model candidates for the base-model datalist (best effort).

    Scans ~/HF_Models/ subdirectories and the Hugging Face hub cache
    (models--* entries). Directories are returned newest-first; a missing
    or unreadable cache degrades to an empty list, never an error.
    """
    found: "list[str]" = []
    home = os.path.expanduser("~")

    hf_dir = os.path.join(home, "HF_Models")
    try:
        if os.path.isdir(hf_dir):
            for name in sorted(os.listdir(hf_dir), key=str.lower):
                if os.path.isfile(os.path.join(hf_dir, name, "config.json")):
                    found.append(os.path.join(hf_dir, name))
    except Exception:
        pass

    hub = os.path.join(home, ".cache", "huggingface", "hub")
    try:
        if os.path.isdir(hub):
            for name in sorted(os.listdir(hub), key=str.lower):
                if not name.startswith("models--"):
                    continue
                snap = os.path.join(hub, name, "snapshots")
                if not os.path.isdir(snap):
                    continue
                for entry in sorted(os.listdir(snap), key=str.lower):
                    if os.path.isfile(os.path.join(snap, entry, "config.json")):
                        found.append(os.path.join(snap, entry))
    except Exception:
        pass

    # De-duplicate (a model may exist in both roots) without reordering.
    seen, uniq = set(), []
    for p in found:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq
