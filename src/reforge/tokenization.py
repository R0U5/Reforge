import io
import os

import numpy as np
import torch
from PIL import Image as PILImage

from .display import _info


def _apply_chat_template_ids(tokenizer, messages, *, add_generation_prompt):
    """``apply_chat_template`` -> ``(input_ids, attention_mask)`` across versions.

    ``return_dict=True`` only exists in newer transformers; the declared
    minimum (>=4.41) returns a plain list. Try the structured call first,
    then fall back, so both behave identically here.
    """
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
        )
    except TypeError:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    if hasattr(out, "get"):
        ids = list(out["input_ids"])
        mask = out.get("attention_mask")
        return ids, list(mask) if mask is not None else [1] * len(ids)
    ids = list(out)
    return ids, [1] * len(ids)


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
            sizes = [f.get("image_sizes") for f in features]
            if sizes and all(s is not None for s in sizes):
                batch["image_sizes"] = torch.stack(
                    [torch.as_tensor(s, dtype=torch.long).reshape(-1)
                     for s in sizes])

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


def _completion_text_of(*completions) -> str:
    """Longest completion text among string or chat-message-list values."""
    best = ""
    for c in completions:
        if isinstance(c, list):
            text = "".join(str(m.get("content", "")) for m in c if isinstance(m, dict))
        else:
            text = str(c or "")
        if len(text) > len(best):
            best = text
    return best


def _find_content_start(full_ids, content, tokenizer) -> "int | None":
    """Index of the assistant content's first token inside full_ids, or None.

    Anchors on the encoded content text itself (full match first, then a
    short prefix), scanning from the end: templates may repeat prompt text
    inside the conversation, but the assistant content occurrence is the
    last one. Overlong rows are dropped upstream, so the content is always
    present in full (nothing truncated it away).
    """
    if not content:
        return None
    try:
        ids = tokenizer.encode(content, add_special_tokens=False)
    except Exception:
        return None
    if not ids:
        return None
    for anchor in (ids, ids[:8]):
        n = len(anchor)
        for i in range(len(full_ids) - n, -1, -1):
            if full_ids[i:i + n] == anchor:
                return i
    return None


def _align_boundary(full_ids, prompt_ids, content, tokenizer) -> int:
    """Token index in full_ids where assistant supervision should start.

    Primary technique: the prompt render (add_generation_prompt=True) is a
    token-prefix of the full render — true for most templates — so the mask
    boundary is simply its length. That is VERIFIED here, because a
    length-only boundary silently mis-masks when BPE merges across the seam:
    a SentencePiece '▁' can fuse the prompt's trailing space into the
    answer's first token (Llama-2 / Mistral-v1 family), and Qwen3-style
    think scaffolds tokenize differently solo ('\\n') vs in context
    ('\\n\\n'). The same prefix-length blind spot is why TRL anchors on
    response_template token ids instead. On mismatch, fall back to a
    token-level search for the assistant content; if even that fails, keep
    the historical length boundary (best effort).
    """
    boundary = len(prompt_ids)
    if prompt_ids and full_ids[:boundary] == list(prompt_ids):
        return boundary
    start = _find_content_start(full_ids, content, tokenizer)
    if start is not None and 0 < start < len(full_ids):
        return start
    return boundary


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
            return {"input_ids": [], "attention_mask": [], "labels": [],
                    "pixel_values": None, "image_sizes": None}

        text = example.get("text", "")
        q, a = split_sft_pair(text, example.get("question") or "")
        answer = a or text          # caption-only rows: the text is the answer
        # Two-turn Phi-3-Vision format. The answer must live in the assistant
        # turn: the old single-turn prompt put it in the user turn, leaving
        # every label at -100 (NaN loss).
        prompt = (f"<|user|>\n<|image_1|>\n{q}<|end|>\n"
                  f"<|assistant|>\n{answer}<|end|>")
        encoded = processor(
            text=prompt,
            images=image,
            truncation=False,       # drop overlong rows instead of cutting <|end|>
            return_tensors="pt"
        )
        input_ids      = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        pixel_values   = encoded["pixel_values"][0]
        # Phi3V's forward skips vision embeddings entirely unless image_sizes
        # is supplied; dropping it silently trains on negative placeholder
        # ids (i.e. no image information at all).
        raw_sizes = encoded.get("image_sizes")
        if raw_sizes is not None:
            raw_sizes = raw_sizes[0]
            image_sizes = (raw_sizes.tolist() if hasattr(raw_sizes, "tolist")
                           else list(raw_sizes))
        else:
            image_sizes = None
        if len(input_ids) > max_length:
            return {"input_ids": [], "attention_mask": [], "labels": [],
                    "length": len(input_ids), "pixel_values": None,
                    "image_sizes": None}

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
            "image_sizes":    image_sizes,
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
        if (messages and last_assistant is not None
                and getattr(tokenizer, "chat_template", None)):
            # Tokenize fully first: right-truncation would cut the trailing
            # <|im_end|> off long assistant turns, teaching the model to
            # never stop. Overlong rows are dropped instead (see below).
            input_ids, attention_mask = _apply_chat_template_ids(
                tokenizer, messages, add_generation_prompt=False)
            if len(input_ids) > max_length:
                return {"input_ids": [], "attention_mask": [], "labels": [], "length": len(input_ids)}

            # Everything before the final assistant message is the prompt.
            prompt_ids, _ = _apply_chat_template_ids(
                tokenizer, messages[:last_assistant], add_generation_prompt=True)
            boundary = _align_boundary(
                input_ids, prompt_ids,
                messages[last_assistant].get("content"), tokenizer)

            # Sanity: boundary must leave some response tokens to train on.
            if boundary <= 0 or boundary >= len(input_ids):
                input_ids, attention_mask = _causal_tokenize(
                    _join_chat_text(messages), tokenizer, max_length)
                boundary = 0
        else:
            # No assistant turn, or no chat template (template-less
            # tokenizers would raise inside apply_chat_template) — train
            # causally on the flattened conversation.
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

        # A chat template is required for the templated path: template-less
        # tokenizers (e.g. Phi-2) would raise inside apply_chat_template and
        # kill the whole .map(). They fall through to the causal branch.
        if question and answer and getattr(tokenizer, "chat_template", None):
            # Tokenize the FULL conversation with chat template first.
            # Right-truncating at max_length would sever the trailing
            # <|im_end|> on long assistant turns and teach the model to
            # never stop; overlong rows are dropped instead (labels stay
            # intact for every row that trains).
            input_ids, attention_mask = _apply_chat_template_ids(
                tokenizer,
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer}],
                add_generation_prompt=False)
            if len(input_ids) > max_length:
                return {"input_ids": [], "attention_mask": [], "labels": [], "length": len(input_ids)}

            # Tokenize ONLY the user message (question) with same chat template to find boundary
            # This gives us the EXACT token count of the prompt portion
            prompt_ids, _ = _apply_chat_template_ids(
                tokenizer,
                [{"role": "user", "content": question}],
                add_generation_prompt=True)  # include assistant prompt marker
            boundary = _align_boundary(
                input_ids, prompt_ids, answer, tokenizer)

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
            ids, _ = _apply_chat_template_ids(
                tokenizer, messages, add_generation_prompt=False)
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
        # DPO rows: measure prompt + preferred completion (the longer of the
        # pair drives sequence length in practice). Conversational rows go
        # through the chat template, matching what DPOTrainer will tokenize.
        if training_mode == "dpo":
            prompt = e.get("prompt")
            chosen = e.get("chosen")
            rejected = e.get("rejected")
            if isinstance(prompt, list) or isinstance(chosen, list):
                base_msgs = list(prompt) if isinstance(prompt, list) else []
                n_best = None
                for completion in (chosen, rejected):
                    comp_msgs = (list(completion) if isinstance(completion, list)
                                 else [{"role": "assistant", "content": str(completion or "")}])
                    n = _template_len(tokenizer, base_msgs + comp_msgs)
                    if n is not None:
                        n_best = n if n_best is None else max(n_best, n)
                if n_best is not None:
                    return {"length": n_best}
                text = "".join(str(m.get("content", "")) for m in base_msgs if isinstance(m, dict))
                text += _completion_text_of(chosen, rejected)
            else:
                text = (str(prompt or "") + _completion_text_of(chosen, rejected))
            ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False
            )["input_ids"]
            return {"length": len(ids)}
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
