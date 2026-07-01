import gc
import math
import os
import platform
import shutil
import subprocess

import torch

from .display import _info, _ok, _warn, _err
from .config import DEFAULT_HF_MODELS_DIR
from .profiles import MODEL_PROFILES, resolve_profile_key


def roc(label: str = "") -> None:
    MiB = 1024 * 1024
    if not torch.cuda.is_available():
        freed_objs = gc.collect()
        if freed_objs > 0:
            _info(f"GC: {freed_objs} objs collected (CPU only)")
        return

    torch.cuda.synchronize()
    before_alloc = torch.cuda.memory_allocated()
    before_res   = torch.cuda.memory_reserved()
    free_before, _ = torch.cuda.mem_get_info()
    freed_objs = gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_allocated()
    after_res   = torch.cuda.memory_reserved()
    free_after,  _ = torch.cuda.mem_get_info()
    freed_alloc    = (before_alloc - after_alloc) / MiB
    freed_reserved = (before_res   - after_res)   / MiB
    freed_driver   = (free_after   - free_before) / MiB

    def log_roc(label, freed_objs, freed_alloc, freed_reserved, freed_driver, eps=0.05):
        parts = []
        if freed_objs > 0:
            parts.append(f"{freed_objs} objs")
        def add(name, val):
            if not math.isclose(val, 0.0, abs_tol=eps):
                parts.append(f"{name} {val:+.1f} MiB")
        add("alloc", freed_alloc)
        add("reserved", freed_reserved)
        add("driver", freed_driver)
        if parts:
            tag = f" [{label}]" if label else ""
            _info(f"GC{tag}: " + "  |  ".join(parts))

    log_roc(label, freed_objs, freed_alloc, freed_reserved, freed_driver)


def log_dataset(dataset_path, log_file_path):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    try:
        with open(log_file_path, "a+", encoding="utf-8") as f:
            f.seek(0)
            content = f.read()
            logged_names = {line.strip() for line in content.splitlines()}
            if dataset_name not in logged_names:
                f.write(dataset_name + "\n")
                _ok(f"Logged '{dataset_name}' to training chain")
                return True
            else:
                _info(f"'{dataset_name}' already in training chain")
                return False
    except Exception as e:
        _warn(f"Could not log dataset: {e}")
        return False


def purge_checkpoints(output_dir):
    try:
        removed = 0
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                path = os.path.join(output_dir, name)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1

        last_link = os.path.join(output_dir, "checkpoint-last")
        if os.path.islink(last_link):
            os.unlink(last_link)
        elif os.path.isdir(last_link):
            shutil.rmtree(last_link, ignore_errors=True)

        last_good = os.path.join(output_dir, "last_good_ckpt")
        if os.path.isdir(last_good):
            shutil.rmtree(last_good, ignore_errors=True)

        _info(f"Purged {removed} old checkpoints")
    except Exception as e:
        _warn(f"Failed to purge checkpoints: {e}")


def select_file() -> str:
    try:
        if platform.system() == "Windows":
            ps_script = r"""
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$ofd = New-Object System.Windows.Forms.OpenFileDialog
$ofd.Filter = 'Parquet files (*.parquet)|*.parquet'
$ofd.Title = 'Select a Parquet file'
$ofd.Multiselect = $false
if ($ofd.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $ofd.FileName
}
"""
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
                capture_output=True, text=True, check=False
            )
            path = (proc.stdout or "").strip()
            if path:
                return path
    except Exception as e:
        _warn("PowerShell file picker failed \u2014 falling back to text input")
    return input("Dataset path (.parquet): ").strip()


_LOCAL_MODEL_MARKERS = (
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "tokenizer.model",
)


def _inspect_model_dir(path: str) -> dict | None:
    if not os.path.isdir(path):
        return None
    try:
        entries = set(os.listdir(path))
    except OSError:
        return None
    if not entries.intersection(_LOCAL_MODEL_MARKERS):
        return None

    size_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    size_bytes += os.path.getsize(fp)
            except OSError:
                pass

    return {
        "path":    path,
        "name":    os.path.basename(path),
        "arch":    resolve_profile_key(path),
        "size_gb": size_bytes / (1024 ** 3),
    }


def _scan_local_models() -> list:
    candidates = []
    seen = set()

    user_model_dirs = [DEFAULT_HF_MODELS_DIR]
    extra = os.environ.get("REFORGE_EXTRA_MODELS_DIR", "")
    if extra:
        user_model_dirs.append(extra)

    for user_dir in user_model_dirs:
        if not os.path.isdir(user_dir):
            continue
        for entry in os.listdir(user_dir):
            full = os.path.join(user_dir, entry)
            if not os.path.isdir(full) or full in seen:
                continue
            info = _inspect_model_dir(full)
            if info:
                candidates.append(info)
                seen.add(full)

    cache_roots = []
    env_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if env_cache:
        cache_roots.append(env_cache)
    cache_roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))

    for cache_root in cache_roots:
        if not os.path.isdir(cache_root):
            continue
        for entry in os.listdir(cache_root):
            if not entry.startswith("models--"):
                continue
            snapshots_dir = os.path.join(cache_root, entry, "snapshots")
            if not os.path.isdir(snapshots_dir):
                continue
            try:
                snaps = os.listdir(snapshots_dir)
            except OSError:
                continue
            if not snaps:
                continue
            snaps.sort(key=lambda s: os.path.getmtime(os.path.join(snapshots_dir, s)), reverse=True)
            full = os.path.join(snapshots_dir, snaps[0])
            if full in seen or not os.path.isdir(full):
                continue
            info = _inspect_model_dir(full)
            if info:
                info["name"] = entry.replace("models--", "").replace("--", "/", 1).replace("--", "/")
                candidates.append(info)
                seen.add(full)

    return candidates


def select_base_model(image_mode: bool = False) -> str:
    candidates = _scan_local_models()

    if image_mode:
        vision_archs = {k for k, v in MODEL_PROFILES.items() if v.get("vision")}
        before = len(candidates)
        candidates = [c for c in candidates if c["arch"] in vision_archs]
        if not candidates and before:
            _warn("--image was set but no vision-capable local models were found.")

    print()
    _info("Available local models:")
    if not candidates:
        _warn("No local models found in local directories or the HF hub cache.")
        print()
        typed = input("  Enter a model path, or Ctrl+C to abort: ").strip()
        if not typed:
            raise RuntimeError("[ERROR] No model selected.")
        if not os.path.isdir(typed):
            raise RuntimeError(f"[ERROR] Path does not exist or is not a directory: {typed!r}")
        return typed

    print()
    for i, c in enumerate(candidates, 1):
        size = c["size_gb"]
        size_str = f"{size:.2f} GB" if size < 100 else f"{size/1024:.2f} TB"
        print(f"  [{i}] {c['name']}   arch={c['arch']}, {size_str}")
        print(f"        {c['path']}")
    print()

    while True:
        choice = input(f"  Select base model [1-{len(candidates)}] or paste a path: ").strip()
        if not choice:
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]["path"]
            _warn(f"Enter a number between 1 and {len(candidates)}, or paste a path.")
            continue
        if os.path.isdir(choice):
            return choice
        raise RuntimeError(f"[ERROR] {choice!r} is not a valid number or existing directory.")
