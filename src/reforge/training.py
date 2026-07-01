import json
import os
import shutil

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModelForVision2Seq,
    TrainingArguments, Trainer
)

from .config import BATCH_SIZE, LOG_STEPS, SAVE_STEPS, SAVE_LIMITS, OUTPUT_DIR, MERGED_DIR
from .display import _info, _ok, _warn, _banner, _table
from .profiles import resolve_model_profile
from .early_stopping import select_scheduler, EarlyStopByLoss, compute_min_steps, dynamic_early_stop_cap
from .reporting import summarize_training
from .tokenization import DynamicCausalCollator
from .utils import roc


def load_model(tokenizer, model_name, image_mode=False, output_dir=None):
    from .profiles import resolve_model_profile
    profile = resolve_model_profile(model_name)
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if image_mode:
        _banner("Model")
        _info("Loading Phi-3-Vision (AutoModelForVision2Seq)...")
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            _attn_implementation="eager",
        ).to("cuda")
    else:
        _banner("Model")
        _info(f"Loading {os.path.basename(model_name)} (AutoModelForCausalLM)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=profile["trust_remote_code"],
            local_files_only=True,
            _attn_implementation=profile["attn_implementation"],
        ).to("cuda")

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    if not image_mode:
        _frozen_vision = 0
        for name, param in base_model.named_parameters():
            if any(tag in name.lower() for tag in ("vision", "img_projection", "image_newline")):
                param.requires_grad_(False)
                _frozen_vision += param.numel()
        if _frozen_vision > 0:
            _info(f"Froze vision encoder: {_frozen_vision / 1e6:.1f}M params (text-only SFT)")

    last_ckpt = os.path.join(output_dir, "checkpoint-last")

    use_last = os.path.isdir(last_ckpt)
    if use_last:
        recorded_base = None
        try:
            with open(os.path.join(last_ckpt, "adapter_config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            recorded_base = cfg.get("base_model_name_or_path")
        except Exception:
            pass

        base_has_config = os.path.isfile(os.path.join(model_name, "config.json"))
        if (not base_has_config) or (recorded_base and os.path.abspath(recorded_base) != os.path.abspath(model_name)):
            _warn("Checkpoint base mismatch \u2014 starting fresh LoRA")
            use_last = False

    if use_last:
        _ok("Resuming LoRA adapter from checkpoint-last")
        if profile["resize_embeddings"] and tokenizer.added_tokens_encoder:
            base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, last_ckpt, is_trainable=True)
    else:
        _info("Initialising fresh LoRA adapter")
        if image_mode:
            lora_r      = 8
            lora_alpha  = 16
            lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            lora_modules_to_save = None
        else:
            lora_r      = profile["lora_r"]
            lora_alpha  = profile["lora_alpha"]
            lora_targets = profile["lora_targets"]
            lora_modules_to_save = profile["modules_to_save"]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_targets,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            base_model_name_or_path=os.path.abspath(model_name),
            modules_to_save=lora_modules_to_save,
        )

        if profile["resize_embeddings"] and tokenizer.added_tokens_encoder:
            _info(f"Resizing embeddings: +{len(tokenizer.added_tokens_encoder)} new tokens")
            base_model.resize_token_embeddings(len(tokenizer))

        model = get_peft_model(base_model, peft_config)

        if lora_modules_to_save:
            for name, param in model.named_parameters():
                if any(
                    f"modules_to_save.default.{m}" in name
                    for m in lora_modules_to_save
                ):
                    param.requires_grad_(False)

    if profile["enable_input_grads"]:
        model.enable_input_require_grads()

    print_trainable_parameters(model)

    if image_mode:
        _info("Gradient checkpointing enabled (VRAM optimisation)")
        model.gradient_checkpointing_enable()

    return model


def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _ok(f"Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")


def run_training(train_dataset, tokenizer, training_mode, model_name, processor, parsed_args, resolved_base, output_dir=None, merge_dir=None):
    from .config import BATCH_SIZE
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if merge_dir is None:
        merge_dir = MERGED_DIR

    profile = resolve_model_profile(model_name)
    _info(f"Model profile: {os.path.basename(model_name)}  |  grad_accum: {profile['grad_accum']}  |  lr: {profile['learning_rate']}")
    estimated_steps = len(train_dataset) // (BATCH_SIZE * profile["grad_accum"])
    use_steps = parsed_args.steps is not None

    if use_steps:
        total_steps = parsed_args.steps
        num_train_epochs = max(1, total_steps // max(estimated_steps, 1))
        _info(f"Fixed steps: {total_steps}")
    else:
        num_train_epochs = parsed_args.epoch
        total_steps = estimated_steps * num_train_epochs
        _info(f"Estimated steps: {total_steps}")

    use_early_stop = not parsed_args.force and not use_steps
    min_stop_steps = compute_min_steps(train_dataset, BATCH_SIZE, profile["grad_accum"])
    early_stop = None

    if use_early_stop:
        cap = dynamic_early_stop_cap(total_steps)
        hard_cap_steps = int(total_steps * cap)
        early_stop = EarlyStopByLoss(
            steps_total=total_steps,
            mode=training_mode,
            active=True,
            hard_cap_steps=hard_cap_steps,
            exposure_floor=0.18,
            quality_lr_frac=0.60,
            window=128,
            ema_beta=0.90,
            std_floor=0.22,
            min_abs_improve=0.04,
            min_sigma_improve=0.50,
            slope_window=None,
            slope_thresh=0.010,
            patience=8,
            cooldown_after_best=3
        )
        _info(f"Early stop: floor \u2265 {int(100 * early_stop.exposure_floor)}%  |  hard cap {int(cap * 100)}% ({hard_cap_steps} steps)")
    else:
        _info("Early stop: disabled")

    selected_scheduler = select_scheduler(train_dataset.num_rows, num_train_epochs, min_stop_steps)
    dynamic_warmup = max(75, int(0.01 * total_steps))
    _table("Training Config", [
        ("Rows",          f"{len(train_dataset):,}"),
        ("Steps",         f"{total_steps:,}"),
        ("Epochs",        f"{num_train_epochs}"),
        ("Scheduler",     selected_scheduler.upper()),
        ("Learning rate", f"{profile['learning_rate']}"),
        ("Batch size",    f"{BATCH_SIZE}  (effective {BATCH_SIZE * profile['grad_accum']} with grad accum {profile['grad_accum']})"),
        ("Warmup",        f"{dynamic_warmup} steps"),
    ])

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=profile["grad_accum"],
        learning_rate=profile["learning_rate"],
        log_level="error",
        warmup_steps=dynamic_warmup,
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs if not use_steps else 1,
        max_steps=total_steps if use_steps else -1,
        eval_strategy="no",
        logging_steps=LOG_STEPS,
        logging_strategy="steps",
        logging_first_step=True,
        disable_tqdm=False,
        report_to=[],
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_LIMITS,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        lr_scheduler_type=selected_scheduler,
        bf16=True,
        group_by_length=True,
        length_column_name="length",
        gradient_checkpointing=False,
    )

    model = load_model(tokenizer, model_name, image_mode=(training_mode == "multimodal"), output_dir=output_dir)
    data_collator = DynamicCausalCollator(tokenizer, pad_to_multiple_of=8, image_mode=(training_mode == "multimodal"))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[] if early_stop is None else [early_stop],
    )

    _training_error = None
    try:
        if total_steps < 1:
            _err("Not enough data to train!")
            raise RuntimeError("Not enough data to train!")

        _banner("Training")
        _info(f"Train rows: {len(train_dataset):,}  |  Steps: {total_steps}  |  Warmup: {dynamic_warmup}")
        last_ckpt = os.path.join(output_dir, "checkpoint-last")

        if not image_mode and torch.cuda.is_available():
            try:
                _info("Warming up CUDA kernels (one dry-run forward pass)...")
                from torch.utils.data import DataLoader
                _warmup_loader = DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    collate_fn=data_collator,
                    shuffle=False,
                    num_workers=0,
                )
                _warmup_batch = next(iter(_warmup_loader))
                _warmup_batch = {k: v.to("cuda") for k, v in _warmup_batch.items() if isinstance(v, torch.Tensor)}
                model.eval()
                with torch.no_grad():
                    model(**_warmup_batch)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                model.train()
                del _warmup_loader, _warmup_batch
                _ok("CUDA warmup complete")
            except Exception as _wu_err:
                _warn(f"CUDA warmup skipped: {_wu_err}")
                try:
                    model.train()
                except Exception:
                    pass

        if os.path.isdir(last_ckpt):
            _ok(f"Resuming from checkpoint: {last_ckpt}")
            trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            trainer.train()
        summarize_training(
            trainer,
            dataset_size=len(train_dataset),
            num_train_epochs=num_train_epochs,
            batch_size=BATCH_SIZE,
            grad_accum=profile["grad_accum"],
            report_path=os.path.join(merge_dir, "training_report.json"),
        )
        roc()
    except KeyboardInterrupt:
        _warn("Keyboard interrupt \u2014 saving current state...")
        if isinstance(model, PeftModel):
            model.save_pretrained(merge_dir)
        else:
            trainer.save_model(merge_dir)
    except Exception as e:
        _training_error = e
        _err(f"Training error: {e}")
        _warn("Attempting emergency save before merge...")
        try:
            if isinstance(model, PeftModel):
                model.save_pretrained(merge_dir)
            else:
                trainer.save_model(merge_dir)
        except Exception as save_err:
            _err(f"Emergency save also failed: {save_err}")
    finally:
        _banner("Cleanup")
        _info("Cleaning checkpoints...")
        ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        ckpts = sorted(
            [d for d in ckpts if d.split("-")[-1].isdigit()],
            key=lambda d: int(d.split("-")[-1])
        )

        if ckpts:
            latest = ckpts[-1]
            last_path = os.path.join(output_dir, latest)
            symlink_path = os.path.join(output_dir, "checkpoint-last")

            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                try:
                    os.unlink(symlink_path)
                except OSError:
                    shutil.rmtree(symlink_path)

            try:
                os.symlink(last_path, symlink_path, target_is_directory=True)
            except (OSError, NotImplementedError):
                _warn("Could not create symlink \u2014 copying instead")
                shutil.copytree(last_path, symlink_path)
                _ok(f"checkpoint-last \u2192 {latest}")

            backup_path = os.path.join(output_dir, "last_good_ckpt")
            shutil.copytree(last_path, backup_path, dirs_exist_ok=True)
            _ok("Last checkpoint backed up")

    return model, trainer, _training_error
