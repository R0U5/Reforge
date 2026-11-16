"""Argv builder + validation for the Web UI's child training processes.

``build_argv(params)`` turns the JSON body of ``POST /api/start`` into the
``python -m reforge ...`` command line the supervisor (webui.py) spawns.
Validation failures raise ``ValueError`` with a message that is shown to
the user verbatim.

This is the web port of the Tk dashboard's ``_build_cmd`` /
``_build_batch_cmd`` / ``_start_training`` validation (D9: duplicate for
now; unify once the web UI is at parity). Differences from the Tk flow,
all deliberate:

* No file dialogs. Batch manifests are written from the JSON body's
  ``stages`` list into the output directory.
* ``strict=False`` (used by ``POST /api/preview``) builds the same argv
  without filesystem existence checks, so the preview works while typing.
* The child always runs in agent mode (``--no_tqdm --agent --agent_log``)
  — the supervisor consumes its stdout NDJSON stream (D10: the tee file
  is the durable copy for post-run analysis, not read by the UI).
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


def _resolve_output_dir(params: dict) -> str:
    out = str(params.get("output_dir") or "").strip() or OUTPUT_DIR
    try:
        os.makedirs(out, exist_ok=True)
    except Exception:
        pass
    return out


def _validate_stage_list(params: dict, strict: bool) -> "list | None":
    """Batch stages: each needs a dataset (local ones must exist)."""
    stages = params.get("stages")
    if stages is None:
        return None
    if not isinstance(stages, list):
        raise ValueError("stages must be a list of stage objects.")
    stages = [s for s in stages if isinstance(s, dict)]
    if not stages:
        raise ValueError("Batch mode needs at least one stage.")
    for i, s in enumerate(stages):
        ds = str(s.get("dataset") or "").strip()
        hf = str(s.get("hf_dataset") or "").strip()
        if not ds and not hf:
            raise ValueError(f"Stage {i + 1} has no dataset.")
        if strict and ds and not os.path.isfile(ds):
            raise ValueError(f"Stage {i + 1} dataset not found:\n{ds}")
    return stages


def _write_batch_manifest(out_dir: str, base_model: str, stages: list) -> str:
    path = os.path.join(out_dir, "batch_manifest.json")
    manifest = {"base_model": base_model or None, "stages": stages}
    if not manifest["base_model"]:
        manifest.pop("base_model")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def build_argv(params: dict, strict: bool = True) -> "list[str]":
    """Validated child command line; raises ValueError on bad input."""
    if not isinstance(params, dict):
        raise ValueError("Request body must be a JSON object.")
    params = {k: v for k, v in params.items() if v not in (None, "")}

    base = str(params.get("base_model") or "").strip()
    ds = str(params.get("dataset") or "").strip()
    hf = str(params.get("hf_dataset") or "").strip()
    stages = _validate_stage_list(params, strict)
    is_batch = stages is not None

    if is_batch:
        # Batch mode: stage 1's base model comes from the manifest.
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
                "A Base Model is required — the web UI launches training in "
                "agent mode, which is non-interactive.")
        if ds:
            if strict:
                if not os.path.exists(ds):
                    raise ValueError(f"Dataset path does not exist:\n{ds}")
                if not ds.lower().endswith(".parquet"):
                    raise ValueError("Local dataset must be a .parquet file.")

    out_dir = _resolve_output_dir(params)

    numeric = {name: _parse_int(params, name) for name in _INT_FIELDS}
    dpo_beta = _parse_dpo_beta(params)

    cmd = [sys.executable, "-m", "reforge"]
    if is_batch:
        manifest_path = _write_batch_manifest(out_dir, base, stages)
        cmd += ["--batch", manifest_path]
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
