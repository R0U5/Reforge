import argparse
import os
import sys

import pandas as pd

from .config import OUTPUT_DIR, MERGED_DIR, MERGE_SUCCESS_FLAG, BATCH_SIZE, TRAINING_CHAIN_PATH
from .display import _title, _info, _ok, _err, _warn, _banner, _enable_utf8_stdout
from .dataset import load_and_prepare_dataset, detect_training_mode
from .profiles import resolve_model_profile
from .training import run_training, load_model
from .utils import select_base_model, select_file
from .reporting import summarize_training


def main():
    _enable_utf8_stdout()

    parser = argparse.ArgumentParser(prog="reforge", description="LoRA fine-tuning pipeline")
    parser.add_argument('--force',      action='store_true', help="Disable early stopping")
    parser.add_argument('--latex',      action='store_true', help="Preserve LaTeX and scientific equations")
    parser.add_argument('--code',       action='store_true', help="Preserve code formatting and indentation")
    parser.add_argument('--epoch',      type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument('--steps',      type=int, help="Terminate after a fixed number of steps")
    parser.add_argument('--image',      action='store_true', help="Enable multimodal (image+text) training")
    parser.add_argument('--hf_dataset', type=str, default=None, help="HuggingFace Hub dataset ID")
    parser.add_argument('--base_model', type=str, default=None, help="Override base model path")
    parser.add_argument('--output_dir', type=str, default=None, help="Override output directory")
    parsed_args, _ = parser.parse_known_args()
    force_epoch  = parsed_args.force
    image_mode   = parsed_args.image

    _title("REFORGE", subtitle="LoRA fine-tuning pipeline")

    output_dir = parsed_args.output_dir or OUTPUT_DIR
    merge_dir = os.path.join(output_dir, "merged")
    merge_success_flag = os.path.join(merge_dir, "success.txt")
    os.makedirs(output_dir, exist_ok=True)

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

    if parsed_args.hf_dataset:
        dataset_path = parsed_args.hf_dataset
        _info(f"HF Hub dataset: {dataset_path}")
    else:
        dataset_path = select_file()
        if image_mode and dataset_path and not dataset_path.lower().endswith(".parquet"):
            if not ("/" in dataset_path and not os.path.exists(dataset_path)):
                _err("Please select a valid .parquet file or pass --hf_dataset")
                return
        elif not image_mode:
            if (not dataset_path) or (not dataset_path.lower().endswith(".parquet")):
                _err("Please select a valid .parquet file")
                return

    if cleaning_mode in {"latex", "math"} and not image_mode:
        try:
            df_columns = pd.read_parquet(dataset_path, engine="pyarrow", columns=None).columns
            detect_training_mode(df_columns)
        except Exception as e:
            _warn(f"Could not peek dataset columns: {e}")

    if image_mode:
        _info("Image mode enabled (Phi-3-Vision / multimodal)")

    train_dataset, eval_dataset, tokenizer, training_mode, model_name, processor = load_and_prepare_dataset(
        dataset_path, cleaning_mode,
        image_mode=image_mode,
        base_model_override=resolved_base,
        output_dir=output_dir,
        merge_dir=merge_dir,
        merge_success_flag=merge_success_flag,
    )

    model, trainer, training_error = run_training(
        train_dataset, tokenizer, training_mode, model_name, processor,
        parsed_args, resolved_base,
        output_dir=output_dir,
        merge_dir=merge_dir,
    )

    if model is None:
        return

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

    _info(f"Saving to: {merge_dir}")
    os.makedirs(merge_dir, exist_ok=True)
    save_profile = resolve_model_profile(resolved_base)
    from transformers import AutoModelForCausalLM, AutoModelForVision2Seq
    try:
        final_model_to_save.save_pretrained(
            merge_dir,
            safe_serialization=save_profile["safe_serialization"],
        )
    except Exception as e:
        _err(f"save_pretrained failed: {e}")
        _warn("Retrying with rebuilt base model...")
        if image_mode:
            base = AutoModelForVision2Seq.from_pretrained(
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
        base.save_pretrained(merge_dir, safe_serialization=save_profile["safe_serialization"])

    if image_mode and processor is not None:
        processor.save_pretrained(merge_dir)
        _ok("Processor saved")
    else:
        tokenizer.save_pretrained(merge_dir)

    if training_error is None:
        with open(os.path.join(merge_dir, "success.txt"), "w", encoding="utf-8") as f:
            f.write("Merge complete")
        _banner("Done")
        _ok(f"Model saved to {merge_dir}")
        print()
    else:
        raise training_error


if __name__ == '__main__':
    main()
