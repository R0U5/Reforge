import argparse
import json
import os
import sys

from .config import OUTPUT_DIR, BATCH_SIZE
from .display import (_title, _info, _ok, _err, _warn, _banner, _safe_print,
                      _enable_utf8_stdout, _redirect_output, _install_safe_streams,
                      _set_stream)
from .agent import BRIDGE, build_agent_schema, classify_error
from .dataset import (load_and_prepare_dataset, detect_training_mode,
                      auto_map_roles, _resolve_model_name)
from .profiles import resolve_model_profile, resolve_profile_key
from .training import run_training
from .early_stopping import (select_scheduler, compute_min_steps,
                             dynamic_early_stop_cap)
from .utils import select_base_model, select_file


def _launch_dashboard() -> int:
    """Run the Tkinter dashboard in this process (`reforge dashboard`)."""
    _banner("Dashboard")
    _info("Starting Reforge Dashboard...")
    try:
        from .dashboard import main as dashboard_main
    except ImportError as e:
        _err(f"Dashboard needs matplotlib to run: {e}")
        _info("Install dashboard extras:  pip install -e .[dashboard]")
        return 1
    dashboard_main()
    return 0


def _resolve_cleaning_mode(parsed_args) -> str:
    """Shared by the real pipeline and --dry_run so they cannot disagree."""
    modes = []
    if parsed_args.latex:
        modes.append("latex")
    if parsed_args.code:
        modes.append("code")
    return "+".join(modes) if modes else "math"


def main():
    _enable_utf8_stdout()

    # Subcommand dispatch: `reforge dashboard` opens the Tk UI.
    # Intercepted before argparse so it isn't mistaken for the dataset
    # positional. (Dashboards are launched only from the first token;
    # a dataset file is expected to keep its .parquet extension.)
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        raise SystemExit(_launch_dashboard())

    # `reforge agent-schema` == `reforge --agent_schema`: print the agent
    # contract as JSON and exit 0.
    if len(sys.argv) > 1 and sys.argv[1] in ("agent-schema", "agent_schema"):
        sys.argv = [sys.argv[0], "--agent_schema"] + sys.argv[2:]

    parser = argparse.ArgumentParser(prog="reforge", description="LoRA fine-tuning pipeline")
    parser.add_argument('dataset', nargs='?', default=None, help="Path to .parquet dataset (or pass --hf_dataset)")
    parser.add_argument('--force',      action='store_true', help="Disable early stopping")
    parser.add_argument('--latex',      action='store_true', help="Preserve LaTeX and scientific equations")
    parser.add_argument('--code',       action='store_true', help="Preserve code formatting and indentation")
    parser.add_argument('--epoch',      type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument('--steps',      type=int, help="Terminate after a fixed number of steps")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Per-device batch size override (default: 1)")
    parser.add_argument('--grad_accum', type=int, default=None,
                        help="Gradient accumulation steps override (default: model profile)")
    parser.add_argument('--image',      action='store_true', help="Enable multimodal (image+text) training")
    parser.add_argument('--train_lm_head', action='store_true',
                        help="Opt-in: fully train the lm_head (~5 GB VRAM on large-vocab models like Qwen3). "
                             "Use for new-token vocab runs or head-capacity ablations. Ignored when resuming from checkpoint-last.")
    parser.add_argument('--hf_dataset', type=str, default=None, help="HuggingFace Hub dataset ID")
    parser.add_argument('--base_model', type=str, default=None, help="Override base model path")
    parser.add_argument('--output_dir', type=str, default=None, help="Override output directory")
    parser.add_argument('--log_file', type=str, default=None,
                        help="Redirect all console output to a text file (recommended for long runs; "
                             "immune to failing Windows console handles)")
    parser.add_argument('--agent', action='store_true',
                        help="Agent mode: emit NDJSON events on stdout (pretty output moves to stderr) "
                             "and keep run_state.json updated in the output directory")
    parser.add_argument('--agent_schema', '--agent-schema', dest='agent_schema',
                        action='store_true',
                        help="Print the machine-readable agent contract as one JSON object "
                             "(CLI args, events, run_state schema, error codes, dataset "
                             "schemas, model profiles) and exit 0. Agents should call this first.")
    parser.add_argument('--dry_run', '--dry-run', dest='dry_run',
                        action='store_true',
                        help="Preflight: validate arguments, dataset schema, and model path, then "
                             "print the resolved training plan as JSON (no weights loaded, no "
                             "training). Non-interactive: --base_model and a dataset are required.")
    parser.add_argument('--agent_log', type=str, default=None,
                        help="Also tee the agent-mode NDJSON event stream to this file")
    parser.add_argument('--no_tqdm', action='store_true',
                        help="Disable the tqdm progress bar and emit clean loss lines instead "
                             "(the dashboard sets this automatically)")
    parsed_args, _ = parser.parse_known_args()
    image_mode = parsed_args.image

    # `--agent_schema`: pure self-description. Handle before anything that
    # has side effects (no dirs created, no streams wrapped). Human output
    # is diverted to stderr so stdout carries exactly one JSON object.
    if parsed_args.agent_schema:
        _set_stream(sys.stderr)
        contract = build_agent_schema(parser)
        sys.stdout.write(json.dumps(contract, indent=2, default=str) + "\n")
        sys.stdout.flush()
        raise SystemExit(0)

    # `--dry_run`: non-interactive preflight, also keeps stdout pure JSON.
    if parsed_args.dry_run:
        _set_stream(sys.stderr)
        raise SystemExit(_dry_run(parsed_args))

    # Wrap stdout/stderr so a failing Windows console or pipe handle (OSError 22)
    # can never abort a run mid-training: writes silently fall back to a log file.
    if parsed_args.log_file:
        _redirect_output(parsed_args.log_file)
    _fallback_log = os.path.join(
        (parsed_args.output_dir or OUTPUT_DIR).rstrip("\\/"), "console.log")
    _install_safe_streams(_fallback_log)

    output_dir = parsed_args.output_dir or OUTPUT_DIR
    merge_dir = os.path.join(output_dir, "merged")
    merge_success_flag = os.path.join(merge_dir, "success.txt")
    os.makedirs(output_dir, exist_ok=True)

    # Agent mode: stdout becomes a pure NDJSON event stream and
    # <output_dir>/run_state.json tracks the latest snapshot.
    _agent_env = os.environ.get("REFORGE_AGENT", "").strip().lower()
    agent_mode = parsed_args.agent or _agent_env in ("1", "true", "yes")
    if agent_mode:
        BRIDGE.enable(output_dir, merge_dir, log_path=parsed_args.agent_log)

    _title("REFORGE", subtitle="LoRA fine-tuning pipeline")

    try:
        _run_pipeline(parsed_args, image_mode, output_dir, merge_dir, merge_success_flag)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        BRIDGE.emit("run_failed", error="Interrupted by user (KeyboardInterrupt)",
                    exit_code=130)
        raise
    except Exception as e:
        error = str(e)
        error_code, hint = classify_error(error)
        BRIDGE.emit("run_failed", error=error, error_code=error_code, hint=hint,
                    exit_code=1)
        raise


def _dry_run(parsed_args) -> int:
    """Preflight validation + resolved training plan, as one JSON object.

    Validates the same contract agent mode enforces (--base_model + a
    dataset), peeks the parquet schema, resolves the model profile, and
    mirrors the step/early-stop/scheduler math from run_training — without
    loading weights, tokenizers, or touching CUDA. Exit 0 when valid, 2
    otherwise; stdout always carries exactly one JSON object.
    """
    result = {"dry_run": True, "ok": True, "errors": [], "warnings": []}

    def _add_error(code, message, hint=None, **extra):
        entry = {"error_code": code, "message": message}
        if hint:
            entry["hint"] = hint
        entry.update(extra)
        result["errors"].append(entry)
        result["ok"] = False

    output_dir = parsed_args.output_dir or OUTPUT_DIR
    merge_dir = os.path.join(output_dir, "merged")
    merge_success_flag = os.path.join(merge_dir, "success.txt")
    result["output"] = {
        "output_dir": output_dir,
        "merged_dir": merge_dir,
        "run_state": os.path.join(output_dir, "run_state.json"),
    }

    # ---- base model (non-interactive contract, same as agent mode) ----
    resolved_base = parsed_args.base_model
    if not resolved_base:
        _add_error("missing_base_model",
                   "--base_model is required for --dry_run (non-interactive preflight)",
                   "pass --base_model /path/to/model")
    else:
        if not os.path.isdir(resolved_base):
            _add_error("model_not_found", f"Base model directory not found: {resolved_base}",
                       "pass an existing local model directory (runs are offline by "
                       "default: HF_HUB_OFFLINE=1)")
        elif not os.path.isfile(os.path.join(resolved_base, "config.json")):
            _add_error("model_not_found",
                       f"Base model directory has no config.json: {resolved_base}",
                       "point --base_model at a Hugging Face model directory")

    # ---- dataset ----
    dataset_path = parsed_args.dataset or parsed_args.hf_dataset
    rows = columns = training_mode = mapped_roles = None
    if not dataset_path:
        _add_error("missing_dataset",
                   "a dataset argument or --hf_dataset is required for --dry_run",
                   "pass a .parquet path as the positional dataset argument, or --hf_dataset ID")
    elif parsed_args.hf_dataset and not parsed_args.dataset:
        result["warnings"].append(
            "Hub dataset requested; skipping schema check (no network access in preflight)")
    else:
        if not dataset_path.lower().endswith(".parquet"):
            _add_error("invalid_dataset", "No valid .parquet dataset provided",
                       "the positional dataset argument must be a .parquet file")
        elif not os.path.isfile(dataset_path):
            _add_error("dataset_not_found", f"Dataset file not found: {dataset_path}",
                       "pass an existing .parquet path")
        else:
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(dataset_path)
                columns = parquet_file.schema_arrow.names
                rows = parquet_file.metadata.num_rows
            except Exception as e:
                _add_error("invalid_dataset", f"Could not read parquet metadata: {e}",
                           "verify the file is a valid .parquet file")
            if columns is not None and not parsed_args.image:
                try:
                    training_mode = detect_training_mode(columns)
                    mapped_roles = dict(zip(
                        ("question", "chosen", "rejected", "image"),
                        auto_map_roles(columns, training_mode)))
                except ValueError as e:
                    _add_error("undetectable_schema", str(e),
                               "rename columns to match a supported schema (see "
                               "`reforge --agent-schema` -> dataset_schemas)",
                               columns=columns)
            elif parsed_args.image:
                training_mode = "multimodal"

    # ---- resolved model (training chain may override --base_model) ----
    model_name = None
    chain_resume = False
    try:
        model_name = _resolve_model_name(merge_dir, merge_success_flag, resolved_base)
        chain_resume = os.path.exists(merge_success_flag)
    except RuntimeError as e:
        _add_error("merge_dir_conflict", str(e),
                   "verify or delete the leftover merged/ directory before rerunning")
    if chain_resume:
        result["warnings"].append(
            f"Training chain active: merged/success.txt found, base model will be {model_name}")

    # ---- resolved training plan (mirrors run_training math) ----
    plan = None
    if model_name and training_mode:
        profile = resolve_model_profile(model_name)
        image_mode = (training_mode == "multimodal")
        batch_size = getattr(parsed_args, "batch_size", None) or BATCH_SIZE
        grad_accum = getattr(parsed_args, "grad_accum", None) or profile["grad_accum"]
        if grad_accum < 1:
            grad_accum = profile["grad_accum"]
        if batch_size < 1:
            batch_size = BATCH_SIZE

        if rows is not None:
            estimated_steps = max(0, rows // (batch_size * grad_accum))
            if parsed_args.steps is not None:
                use_steps = True
                total_steps = parsed_args.steps
                epochs = max(1, total_steps // max(estimated_steps, 1))
            else:
                use_steps = False
                epochs = parsed_args.epoch
                total_steps = estimated_steps * epochs

            class _RowsView:
                def __init__(self, n): self._n = n
                def __len__(self): return self._n
            min_stop_steps = compute_min_steps(_RowsView(rows), batch_size, grad_accum)
            plan = {
                "rows": rows,
                "training_mode": training_mode,
                "cleaning_mode": _resolve_cleaning_mode(parsed_args),
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
                "steps_fixed": bool(use_steps),
                "epochs": epochs,
                "estimated_steps_per_epoch": estimated_steps,
                "total_steps": total_steps,
                "learning_rate": profile["learning_rate"],
                "warmup_steps": min(max(75, int(0.01 * total_steps)), max(1, total_steps // 10)),
                "early_stop": {
                    "enabled": not parsed_args.force and not use_steps,
                    "min_steps": min_stop_steps,
                    "hard_cap_steps": (
                        int(total_steps * dynamic_early_stop_cap(total_steps))
                        if not parsed_args.force and not use_steps and total_steps > 0 else None),
                },
                "scheduler": select_scheduler(rows, epochs, min_stop_steps),
                "lora": {
                    "profile_key": resolve_profile_key(model_name),
                    "r": 8 if image_mode else profile["lora_r"],
                    "alpha": 16 if image_mode else profile["lora_alpha"],
                    "targets": profile["lora_targets"],
                    "modules_to_save": None if image_mode else (
                        ["lm_head"] if parsed_args.train_lm_head else profile["modules_to_save"]),
                    "resize_embeddings": profile["resize_embeddings"],
                    "quantization": "none (vision loads bf16)" if image_mode
                                    else "4-bit NF4 (bitsandbytes)",
                },
            }
        else:
            plan = {"training_mode": "multimodal",
                    "warning": "row counts unavailable for Hub datasets; step plan skipped"}
        result["model"] = {
            "base_model": resolved_base,
            "resolved_model": model_name,
            "chain_resume": chain_resume,
            "profile": {k: profile[k] for k in (
                "attn_implementation", "trust_remote_code", "use_fast_tokenizer",
                "learning_rate", "grad_accum", "safe_serialization",
                "enable_input_grads")},
        }

    if plan:
        result["plan"] = plan
    if mapped_roles is not None:
        result.setdefault("dataset", {})["source"] = dataset_path
        result["dataset"]["rows"] = rows
        result["dataset"]["columns"] = columns
        result["dataset"]["training_mode"] = training_mode
        result["dataset"]["mapped_roles"] = mapped_roles
        result["dataset"]["cleaning_mode"] = _resolve_cleaning_mode(parsed_args)

    sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")
    sys.stdout.flush()
    return 0 if result["ok"] else 2


def _run_pipeline(parsed_args, image_mode, output_dir, merge_dir, merge_success_flag):
    merge_dir_actual = merge_dir

    # Agent mode is non-interactive by contract: interactive model/dataset
    # pickers would block forever waiting on stdin. Fail fast instead.
    if BRIDGE.enabled:
        _problems = []
        if not parsed_args.base_model:
            _problems.append("--base_model is required in agent mode (non-interactive run)")
        if not (parsed_args.dataset or parsed_args.hf_dataset):
            _problems.append("a dataset argument or --hf_dataset is required in agent mode")
        if _problems:
            _error_code, _hint = classify_error(_problems[0])
            BRIDGE.emit("run_failed", error="; ".join(_problems), problems=_problems,
                        error_code=_error_code, hint=_hint, exit_code=2)
            _err(_problems[0])
            raise SystemExit(2)

    BRIDGE.emit(
        "run_started",
        stage="loading",
        output_dir=output_dir,
        dataset=parsed_args.dataset,
        hf_dataset=parsed_args.hf_dataset,
        base_model=parsed_args.base_model,
        image_mode=image_mode,
        force=parsed_args.force,
        epoch=parsed_args.epoch,
        steps=parsed_args.steps,
    )

    if parsed_args.base_model:
        resolved_base = parsed_args.base_model
        _info(f"Base model override: {resolved_base}")
    else:
        resolved_base = select_base_model(image_mode=image_mode)

    modes = []
    if parsed_args.latex:
        modes.append("latex")
    if parsed_args.code:
        modes.append("code")
    cleaning_mode = "+".join(modes) if modes else "math"

    # dataset path: positional arg takes precedence, then --hf_dataset, then interactive
    if parsed_args.dataset:
        dataset_path = parsed_args.dataset
    elif parsed_args.hf_dataset:
        dataset_path = parsed_args.hf_dataset
        _info(f"HF Hub dataset: {dataset_path}")
    else:
        dataset_path = select_file()
    from_hf_hub = bool(parsed_args.hf_dataset) and not parsed_args.dataset
    if from_hf_hub:
        pass  # Hub IDs are not local .parquet files; validated at load time
    elif image_mode and dataset_path and not dataset_path.lower().endswith(".parquet"):
        if not ("/" in dataset_path and not os.path.exists(dataset_path)):
            _error_code, _hint = classify_error("No valid .parquet dataset provided")
            BRIDGE.emit("run_failed", error="No valid .parquet dataset provided (image mode)",
                        error_code=_error_code, hint=_hint, exit_code=2)
            _err("Please select a valid .parquet file or pass --hf_dataset")
            raise SystemExit(2)
    elif not image_mode:
        if (not dataset_path) or (not dataset_path.lower().endswith(".parquet")):
            _error_code, _hint = classify_error("No valid .parquet dataset provided")
            BRIDGE.emit("run_failed", error="No valid .parquet dataset provided",
                        error_code=_error_code, hint=_hint, exit_code=2)
            _err("Please select a valid .parquet file")
            raise SystemExit(2)
        if not os.path.isfile(dataset_path):
            _error_code, _hint = classify_error(f"Dataset file not found: {dataset_path}")
            BRIDGE.emit("run_failed", error=f"Dataset file not found: {dataset_path}",
                        dataset=dataset_path, error_code=_error_code, hint=_hint, exit_code=2)
            _err(f"Dataset file not found: {dataset_path}")
            raise SystemExit(2)

    # Pre-flight schema check: fail fast on undetectable schemas before any
    # model/tokenizer loading. Reads only parquet metadata, not the data.
    if not image_mode and os.path.isfile(dataset_path):
        try:
            import pyarrow.parquet as pq
            _peek_columns = pq.read_schema(dataset_path).names
            detect_training_mode(_peek_columns)
        except ValueError as e:
            _error_code, _hint = classify_error(str(e))
            BRIDGE.emit("run_failed", error=str(e), dataset=dataset_path,
                        columns=_peek_columns, error_code=_error_code, hint=_hint,
                        exit_code=2)
            _err(str(e))
            raise SystemExit(2)
        except Exception as e:
            _warn(f"Could not peek dataset columns: {e}")

    if image_mode:
        _info("Image mode enabled (Phi-3-Vision / multimodal)")

    train_dataset, eval_dataset, tokenizer, training_mode, model_name, processor, schema_info = load_and_prepare_dataset(
        dataset_path, cleaning_mode,
        image_mode=image_mode,
        base_model_override=resolved_base,
        output_dir=output_dir,
        merge_dir=merge_dir_actual,
        merge_success_flag=merge_success_flag,
    )
    BRIDGE.emit(
        "dataset_loaded",
        stage="preparing",
        dataset=dataset_path,
        training_mode=training_mode,
        rows=train_dataset.num_rows,
        eval_rows=(eval_dataset.num_rows if eval_dataset is not None else None),
        base_model=resolved_base,
        columns=schema_info["columns"],
        mapped_roles=schema_info["mapped_roles"],
        cleaning_mode=schema_info["cleaning_mode"],
        shuffle_seed=schema_info["shuffle_seed"],
    )

    model, trainer, training_error = run_training(
        train_dataset, tokenizer, training_mode, model_name, processor,
        parsed_args, resolved_base,
        output_dir=output_dir,
        merge_dir=merge_dir_actual,
    )

    if training_error is not None:
        raise training_error

    BRIDGE.emit("merge_started", stage="exporting", merge_dir=merge_dir_actual)

    from peft import PeftModel
    if isinstance(model, PeftModel):
        _banner("Saving")
        _info("Merging LoRA weights into base model...")
        try:
            merged_model = model.merge_and_unload()
            final_model_to_save = merged_model
            _ok("LoRA merge complete")
        except Exception as e:
            _warn(f"Merge failed: {e} \u2014 saving adapter as-is")
            final_model_to_save = model
    else:
        _banner("Saving")
        _info("Model already merged")
        final_model_to_save = model

    _info(f"Saving to: {merge_dir_actual}")
    os.makedirs(merge_dir_actual, exist_ok=True)
    save_profile = resolve_model_profile(resolved_base)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    try:
        final_model_to_save.save_pretrained(
            merge_dir_actual,
            safe_serialization=save_profile["safe_serialization"],
        )
    except Exception as e:
        _err(f"save_pretrained failed: {e}")
        _warn("Retrying with rebuilt base model...")
        import torch
        if image_mode:
            base = AutoModelForImageTextToText.from_pretrained(
                resolved_base,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                _attn_implementation="eager",
            ).to("cuda")
        else:
            base = AutoModelForCausalLM.from_pretrained(
                resolved_base,
                local_files_only=True,
                trust_remote_code=save_profile["trust_remote_code"],
                torch_dtype=torch.bfloat16,
                _attn_implementation="eager",
            ).to("cuda")
        if save_profile["resize_embeddings"]:
            try:
                base.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
        sd = {k: v.cpu() for k, v in final_model_to_save.state_dict().items()}
        missing_unexp = base.load_state_dict(sd, strict=False)
        _info(f"State-dict loaded \u2014 missing/unexpected: {missing_unexp}")
        base.save_pretrained(merge_dir_actual, safe_serialization=save_profile["safe_serialization"])

    if image_mode and processor is not None:
        processor.save_pretrained(merge_dir_actual)
        _ok("Processor saved")
    else:
        tokenizer.save_pretrained(merge_dir_actual)

    if training_error is None:
        with open(os.path.join(merge_dir_actual, "success.txt"), "w", encoding="utf-8") as f:
            f.write("Merge complete")
        BRIDGE.emit(
            "merge_complete",
            merged_dir=merge_dir_actual,
            report=os.path.join(merge_dir_actual, "training_report.json"),
        )
        _banner("Done")
        _ok(f"Model saved to {merge_dir_actual}")
        _safe_print()
        BRIDGE.emit("run_complete", merged_dir=merge_dir_actual, exit_code=0)
    else:
        raise training_error


if __name__ == '__main__':
    main()
