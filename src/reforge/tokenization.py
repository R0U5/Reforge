import io
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

        # Both SFT and multimodal: preserve pre-masked labels, pad with -100.
        # (The two branches used to be duplicated verbatim.)
        label_seqs = [_to_py(f["labels"]) for f in features]
        pad_len = batch["input_ids"].shape[1]
        padded_labels = []
        for seq in label_seqs:
            pad_needed = pad_len - len(seq)
            padded_labels.append(seq + [-100] * pad_needed)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

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


def split_sft_pair(text: str, question: str) -> tuple:
    """Split an SFT row into (question, answer).

    Preferred: the dataset carries the cleaned question separately, so the
    boundary between prompt and answer is unambiguous. Legacy fallback:
    merged text only, split on the first "\n\n". Unseparable rows return
    ("", text) and are trained causally.
    """
    question = str(question or "").strip()
    text = str(text or "")
    if question and text.startswith(question) and len(text) > len(question):
        return question, text[len(question):].lstrip("\r\n").strip()
    sep = "\n\n"
    sep_idx = text.find(sep)
    if sep_idx >= 0:
        return (question or text[:sep_idx].strip()), text[sep_idx + len(sep):].strip()
    return "", text


def build_sft_messages(text: str, question: str) -> "list | None":
    """Build the user/assistant message pair for an SFT row, or None."""
    q, a = split_sft_pair(text, question)
    if q and a:
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return None


def _causal_tokenize(text, tokenizer, max_length) -> tuple:
    """Plain causal tokenization shared by every fallback path."""
    tokens = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_length
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask", [1] * len(input_ids))
    return input_ids, attention_mask


def _join_chat_text(messages) -> str:
    """Flatten a messages list to plain text for causal fallbacks."""
    return "\n".join(
        str(m.get("content", "")) for m in (messages or [])
        if isinstance(m, dict)
    )


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
            "length":         len(input_ids),
        }

    # Chat mode: train on the last assistant response only.
    if training_mode == "chat":
        messages = example.get("messages")
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            messages = None
        last_assistant = None
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant":
                    last_assistant = i
                    break
        if messages and last_assistant is not None:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=max_length,
                return_dict=True,
            )
            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens.get("attention_mask", [1] * len(input_ids))

            # Everything before the final assistant message is the prompt.
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:last_assistant],
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=max_length,
                return_dict=True,
            )
            boundary = len(prompt_tokens["input_ids"])

            # Sanity: boundary must leave some response tokens to train on.
            if boundary <= 0 or boundary >= len(input_ids):
                input_ids, attention_mask = _causal_tokenize(
                    _join_chat_text(messages), tokenizer, max_length)
                boundary = 0
        else:
            # No assistant turn - train causally on the flattened conversation.
            input_ids, attention_mask = _causal_tokenize(
                _join_chat_text(messages), tokenizer, max_length)
            boundary = 0

        labels = [-100] * boundary + list(input_ids[boundary:])
        labels = labels[:len(input_ids)]

        return {
            "input_ids": [int(x) for x in input_ids],
            "attention_mask": [int(x) for x in attention_mask],
            "labels": [int(x) for x in labels],
        }

    # SFT mode: use chat template to format conversation, mask prompt tokens via exact token indexing
    if training_mode == "sft":
        text = example.get("text", "") or ""
        question, answer = split_sft_pair(text, str(example.get("question") or ""))

        if question and answer:
            # Tokenize the FULL conversation with chat template
            full_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer}],
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=max_length,
                return_dict=True,
            )
            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens.get("attention_mask", [1] * len(input_ids))

            # Tokenize ONLY the user message (question) with same chat template to find boundary
            # This gives us the EXACT token count of the prompt portion
            user_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,  # Include assistant prompt marker
                truncation=True,
                max_length=max_length,
                return_dict=True,
            )
            boundary = len(user_tokens["input_ids"])

            # Sanity: boundary must leave some response tokens to train on
            if boundary <= 0 or boundary >= len(input_ids):
                # Fallback: causal on full text
                input_ids, attention_mask = _causal_tokenize(text, tokenizer, max_length)
                boundary = 0
        else:
            # No separator or no question - treat as causal
            input_ids, attention_mask = _causal_tokenize(text, tokenizer, max_length)
            boundary = 0

        # Mask prompt tokens, train on assistant response only
        labels = [-100] * boundary + list(input_ids[boundary:])
        labels = labels[:len(input_ids)]

        return {
            "input_ids": [int(x) for x in input_ids],
            "attention_mask": [int(x) for x in attention_mask],
            "labels": [int(x) for x in labels],
        }

    # Causal / fallback
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


def measure_lengths(ds, tokenizer, training_mode=None):
    _num_proc = min(12, os.cpu_count() or 4)

    def _template_len(tokenizer, messages) -> int | None:
        """Token length of a conversation under its chat template, or None."""
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False)
            # Newer transformers wrap the ids in a BatchEncoding; plain lists
            # are returned by older versions.
            if hasattr(ids, "get"):
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return None

    def _len_map(e):
        # Chat rows: measure the actual templated conversation.
        if training_mode == "chat" and e.get("messages"):
            n = _template_len(tokenizer, e["messages"])
            if n is not None:
                return {"length": n}
            return {"length": len(_join_chat_text(e["messages"]))}
        # SFT rows: measure the templated user/assistant pair, not raw text,
        # so the P95 matches what training actually feeds the model.
        if training_mode == "sft" and e.get("question"):
            messages = build_sft_messages(e.get("text") or "", e["question"])
            if messages is not None:
                n = _template_len(tokenizer, messages)
                if n is not None:
                    return {"length": n}
        ids = tokenizer(
            e.get("text") or "",
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False
        )["input_ids"]
        return {"length": len(ids)}

    ds = ds.map(_len_map, num_proc=_num_proc)
    lengths = ds["length"]
    if not lengths:
        raise ValueError("[ERROR] Dataset is empty after processing — no rows to measure lengths from.")
    p50 = int(np.percentile(lengths, 50))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    # Explicit override wins (e.g. REFORGE_MAX_LENGTH=512) — needed on 12 GB
    # cards where the P95-derived length oversubscribes VRAM via WDDM paging.
    override = os.environ.get("REFORGE_MAX_LENGTH", "").strip()
    if override:
        auto_len = max(128, int(override))
        _info(f"Lengths — p50: {p50}  p95: {p95}  p99: {p99}  →  max_length OVERRIDDEN to {auto_len}")
    else:
        auto_len = max(128, min(1024, p95))
        _info(f"Lengths — p50: {p50}  p95: {p95}  p99: {p99}  →  max_length set to {auto_len}")
    return ds, auto_len
