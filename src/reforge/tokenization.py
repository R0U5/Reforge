import io
import math
import os

import numpy as np
import torch
from PIL import Image as PILImage

from .display import _info


class DynamicCausalCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8, image_mode=False):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.image_mode = image_mode

    def __call__(self, features):
        def _to_py(x):
            if isinstance(x, torch.Tensor):
                return x.tolist()
            return x
        base_feats = []
        for f in features:
            base_feats.append({
                "input_ids": _to_py(f["input_ids"]),
                "attention_mask": _to_py(f.get("attention_mask", []))
            })
        batch = self.tokenizer.pad(
            base_feats,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.image_mode:
            def _to_py(x):
                if isinstance(x, torch.Tensor):
                    return x.tolist()
                return list(x) if not isinstance(x, list) else x
            label_seqs = [_to_py(f["labels"]) for f in features]
            pad_len = batch["input_ids"].shape[1]
            padded_labels = []
            for seq in label_seqs:
                pad_needed = pad_len - len(seq)
                padded_labels.append(seq + [-100] * pad_needed)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        else:
            batch["labels"] = batch["input_ids"].masked_fill(batch["attention_mask"].eq(0), -100)

        if self.image_mode:
            pixel_list = [f["pixel_values"] for f in features if "pixel_values" in f]
            if pixel_list:
                try:
                    batch["pixel_values"] = torch.stack(pixel_list)
                except Exception:
                    batch["pixel_values"] = torch.cat(
                        [p.unsqueeze(0) if p.dim() == 3 else p for p in pixel_list], dim=0
                    )

        return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = torch.from_numpy(logits[..., :-1, :].copy())
    shift_labels = torch.from_numpy(labels[..., 1:].copy()).long()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    try:
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")
    return {"perplexity": perplexity, "eval_loss": loss.item()}


def _decode_image(raw):
    if isinstance(raw, PILImage.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict):
        raw = raw.get("bytes") or raw.get("path") or raw
    if isinstance(raw, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, str) and os.path.isfile(raw):
        return PILImage.open(raw).convert("RGB")
    raise ValueError(f"Cannot decode image from type: {type(raw)}")


def tokenize(example, tokenizer, training_mode="sft", max_length=750, processor=None):
    if processor is not None and training_mode == "multimodal" and "image" in example:
        try:
            image = _decode_image(example["image"])
        except Exception as e:
            print(f"[WARN] Skipping unreadable image: {e}")
            return {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": None}

        text = example.get("text", "")
        prompt = f"<|user|>\n<|image_1|>\n{text}<|end|>\n<|assistant|>"
        encoded = processor(
            text=prompt,
            images=image,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids      = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        pixel_values   = encoded["pixel_values"][0]

        assistant_token_ids = tokenizer.encode("<|assistant|>", add_special_tokens=False)
        labels = [-100] * len(input_ids)
        if assistant_token_ids:
            a_id = assistant_token_ids[0]
            boundary = -1
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == a_id:
                    boundary = i + 1
                    break
            if boundary > 0:
                labels[boundary:] = input_ids[boundary:]
            else:
                labels = list(input_ids)

        return {
            "input_ids":      [int(x) for x in input_ids],
            "attention_mask": [int(x) for x in attention_mask],
            "labels":         [int(x) for x in labels],
            "pixel_values":   pixel_values,
            **({"length": len(input_ids)} if "length" not in example else {"length": int(example["length"])}),
        }

    tokens = tokenizer(
        example["text"],
        padding=False,
        truncation=True,
        max_length=max_length
    )
    out = {
        "input_ids": [int(x) for x in tokens["input_ids"]],
        "attention_mask": [int(x) for x in tokens["attention_mask"]],
        "labels": [int(x) for x in tokens["input_ids"]],
    }
    if "length" in example:
        try:
            out["length"] = int(example["length"])
        except Exception:
            pass
    return out


def measure_lengths(ds, tokenizer):
    _num_proc = min(12, os.cpu_count() or 4)

    def _len_map(e):
        ids = tokenizer(
            e["text"],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False
        )["input_ids"]
        return {"length": len(ids)}

    ds = ds.map(_len_map, num_proc=_num_proc)
    lengths = ds["length"]
    if not lengths:
        raise ValueError("[ERROR] Dataset is empty after processing \u2014 no rows to measure lengths from.")
    p50 = int(np.percentile(lengths, 50))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    auto_len = max(128, min(1024, p95))
    _info(f"Lengths \u2014 p50: {p50}  p95: {p95}  p99: {p99}  \u2192  max_length set to {auto_len}")
    return ds, auto_len



