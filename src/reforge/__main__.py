import argparse
import os
import sys

from .config import OUTPUT_DIR
from .display import (_title, _info, _ok, _err, _warn, _banner, _safe_print,
                      _enable_utf8_stdout, _redirect_output, _install_safe_streams)
from .agent import BRIDGE
from .dataset import load_and_prepare_dataset, detect_training_mode
from .profiles import resolve_model_profile
from .training import run_training
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


def main():
    _enable_utf8_stdout()

    # Subcommand dispatch: `reforge dashboard` opens the Tk UI.
    # Intercepted before argparse so it isn't mistaken for the dataset
    # positional. (Dashboards are launched only from the first token;
    # a dataset file is expected to keep its .parquet extension.)
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        raise SystemExit(_launch_dashboard())

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
    parser.add_argument('--agent_log', type=str, default=None,
                        help="Also tee the agent-mode NDJSON event stream to this file")
    parser.add_argument('--no_tqdm', action='store_true',
                        help="Disable the tqdm progress bar and emit clean loss lines instead "
                             "(the dashboard sets this automatically)")
    parsed_args, _ = parser.parse_known_args()
    image_mode = parsed_args.image

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
        BRIDGE.emit("run_failed", error="Interrupted by user (KeyboardInterrupt)", exit_code=130)
        raise
    except Exception as e:
        BRIDGE.emit("run_failed", error=str(e), exit_code=1)
        raise


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
            BRIDGE.emit("run_failed", error="; ".join(_problems), exit_code=2)
            _err(_problems[0])
            raise SystemExit(2)

    BRIDGE.emit(
        "run_started",
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
            BRIDGE.emit("run_failed", error="No valid .parquet dataset provided (image mode)", exit_code=2)
            _err("Please select a valid .parquet file or pass --hf_dataset")
            raise SystemExit(2)
    elif not image_mode:
        if (not dataset_path) or (not dataset_path.lower().endswith(".parquet")):
            BRIDGE.emit("run_failed", error="No valid .parquet dataset provided", exit_code=2)
            _err("Please select a valid .parquet file")
            raise SystemExit(2)
        if not os.path.isfile(dataset_path):
            BRIDGE.emit("run_failed", error=f"Dataset file not found: {dataset_path}", exit_code=2)
            _err(f"Dataset file not found: {dataset_path}")
            raise SystemExit(2)

    # Pre-flight schema check: fail fast on undetectable schemas before any
    # model/tokenizer loading. Reads only parquet metadata, not the data.
    if not image_mode and os.path.isfile(dataset_path):
        try:
            import pyarrow.parquet as pq
            detect_training_mode(pq.read_schema(dataset_path).names)
        except ValueError:
            raise
        except Exception as e:
            _warn(f"Could not peek dataset columns: {e}")

    if image_mode:
        _info("Image mode enabled (Phi-3-Vision / multimodal)")

    train_dataset, eval_dataset, tokenizer, training_mode, model_name, processor = load_and_prepare_dataset(
        dataset_path, cleaning_mode,
        image_mode=image_mode,
        base_model_override=resolved_base,
        output_dir=output_dir,
        merge_dir=merge_dir_actual,
        merge_success_flag=merge_success_flag,
    )
    BRIDGE.emit(
        "dataset_loaded",
        dataset=dataset_path,
        training_mode=training_mode,
        rows=train_dataset.num_rows,
        eval_rows=(eval_dataset.num_rows if eval_dataset is not None else None),
        base_model=resolved_base,
    )

    model, trainer, training_error = run_training(
        train_dataset, tokenizer, training_mode, model_name, processor,
        parsed_args, resolved_base,
        output_dir=output_dir,
        merge_dir=merge_dir_actual,
    )

    if training_error is not None:
        raise training_error

    BRIDGE.emit("merge_started", merge_dir=merge_dir_actual)

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
