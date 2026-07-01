import os
import random
import re

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, Sequence, Value

from .config import TRAINING_CHAIN_PATH, MERGED_DIR, MERGE_SUCCESS_FLAG
from .display import _info, _ok, _warn, _err, _banner
from .profiles import resolve_model_profile
from .tokenization import tokenize, measure_lengths, _decode_image
from .utils import roc, log_dataset, purge_checkpoints
from .config import OUTPUT_DIR

try:
    import emoji as _emoji_mod
    _EMOJI_AVAILABLE = True
except ImportError:
    _emoji_mod = None
    _EMOJI_AVAILABLE = False

TRAINING_SCHEMAS = {
    "sft": {
        "required": ["chosen"],
        "optional": ["question"],
        "aliases": {
            "chosen": ["response", "chosen", "answer", "answers", "completion",
                       "output", "solution", "expected_answer", "long_answer", "messages", "summary"],
            "question": ["instruction", "question", "prompt", "input",
                         "query", "context", "problem", "document"]
        }
    },
    "causal": {
        "required": ["chosen"],
        "optional": [],
        "aliases": {
            "chosen": ["prompt", "text"]
        }
    },
    "chat": {
        "required": ["question"],
        "optional": [],
        "aliases": {
            "question": ["messages"]
        }
    },
    "multimodal": {
        "required": ["image", "chosen"],
        "optional": ["question"],
        "aliases": {
            "image":    ["image", "img", "pixel", "photo", "image_path", "picture"],
            "chosen":   ["answer", "response", "chosen", "caption", "completion",
                         "output", "description", "label"],
            "question": ["question", "prompt", "instruction", "query", "text"]
        }
    }
}

RE_LATEX_INLINE = re.compile(r"(?<!\$)\$(?:\\.|[^$\\])+\$(?!\$)")                   # $...$
ANSWER_MARKER_RE = re.compile(r'^\s*#{3,6}\s*(?:final\s*answer\s*:)?\s*(?:answer\s*:)?\s*', re.I)
RE_TEX_INLINE = re.compile(r"\\\((?:\\.|[^\\])+\\\)|\\\[(?:\\.|[^\\])+\\\]", re.DOTALL)
RE_LATEX_ENV = re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.DOTALL)
RE_HTML = re.compile(r"<[^>]+>")
RE_MD_HEADER = re.compile(r"^\s*#{1,6}\s.*$", re.MULTILINE)
RE_JUNK_WORDS = re.compile(r"\b(?:Click here|Subscribe|Follow us|Advertisement)\b", re.IGNORECASE)
RE_ECOM_TAGS = re.compile(r"\b(?:SKU|ASIN|MSRP|Price:?\s*\$?\d[\d,\.]*)\b", re.IGNORECASE)
RE_UNICODE_GARBAGE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
RE_MEDIA_PLACEHOLDERS = re.compile(r"\[(?:image|video|audio|figure)[^\]]*\]", re.IGNORECASE)
RE_TRASH_PLACEHOLDER = re.compile(r"\[TRASH\]")
UNBOX_RE = re.compile(r'^[\(\[\{"]?\s*(?:\$+)?\\boxed\{([^}]*)\}(?:\$+)?\s*[\)\]\}"]?\.?\s*$')
BOX_INNER_WRAP_RE = re.compile(r'\\(?:text|mathrm)\{([^}]*)\}')
WHITESPACE_ESC_RE = re.compile(r'(?:[\n\r\t]+|\\\\[ntr])+')
FORMAT_STRIP_RE = re.compile(r"(?:[#*_`]+|[!?]{2,}|&[a-z]+;)")
PUNC_TRANS = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                             "\u2013": "-", "\u2014": "-", "\u2192": "->", "\u2190": "<-"})
MATH_SPACING_RE   = re.compile(r'\\(?:,|;|:|!|quad|qquad)\s*')
MATH_WRAPPERS_RE  = re.compile(r'\\(?:mathrm|text|operatorname)\{([^}]*)\}')
DOLLAR_INLINE_RE  = re.compile(r'(?<!\$)\$([^\$]+)\$(?!\$)')

_MATH_SIMPLE_MAP = {
    r'\cdot': '*',
    r'\times': '*',
    r'\le': '<=',
    r'\ge': '>=',
    r'\neq': '!=',
    r'\pm': '+/-',
    r'\div': '/',
    r'\triangle': 'triangle',
    r'\quad': ' ',
}
MATH_SIMPLE_MAP_RE = re.compile("|".join(map(re.escape, _MATH_SIMPLE_MAP.keys())))

BAD_TOKENS = [
    "embedreportprint", "cloneembedreport", "rawdownload", "printrawdownload",
    "reportprintclone", "embedreport", "printclone",
    "guiActive", "guiActiveUnfocused", "externalToEVA", "externalToEVAOnly",
    "PsyNetMessage", "vesselType", "activeRadarLock",
    "unfocusedRange", "targetType", "guiIcon", "stockLegacySensor", "KSPField",
    "persistent",
    "\u00c3\u0082", "\ufffd", "\u00c3\u00a2", "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u0153", "\u00e2\u20ac\u009d", "\u00e2\u20ac\u201c",
    "\u00e2\u20ac", "\u00c3\u00bc", "\u00c3\u00b6", "\u00c3\u0178",
    "[REJECTED ANSWER PLACEHOLDER]", "[PLACEHOLDER]", "[DUMMY]", "[INSERT]",
    "Traceback (most recent call last):", "NullPointerException",
    "undefined is not a function", "at com.",
    "You are a helpful assistant.", "LanguageModelOutput",
    "<|eot|>", "<unk>"
]
BAD_TOKENS_RE = re.compile("|".join(map(re.escape, BAD_TOKENS)))


def detect_training_mode(columns):
    lowered = {col.lower(): col for col in columns}
    image_aliases = set(TRAINING_SCHEMAS["multimodal"]["aliases"]["image"])
    has_image_col = any(col in image_aliases for col in lowered)
    if has_image_col:
        check_order = ["multimodal"] + [m for m in TRAINING_SCHEMAS if m != "multimodal"]
    else:
        check_order = list(TRAINING_SCHEMAS.keys())

    for mode in check_order:
        schema = TRAINING_SCHEMAS[mode]
        required = schema["required"]
        aliases = schema["aliases"]
        if all(
            any(alias in lowered for alias in aliases.get(role, []))
            for role in required
        ):
            return mode
    raise ValueError(f"[ERROR] Could not detect training mode from columns: {list(columns)}")


def auto_map_roles(columns, training_mode):
    lowered = {col.lower(): col for col in columns}
    aliases = TRAINING_SCHEMAS[training_mode]["aliases"]
    roles = TRAINING_SCHEMAS[training_mode]["required"] + TRAINING_SCHEMAS[training_mode].get("optional", [])
    mapped = {role: None for role in roles}
    for role in roles:
        for alias in aliases.get(role, []):
            if alias in lowered:
                mapped[role] = lowered[alias]
                break
    return mapped.get("question"), mapped.get("chosen"), mapped.get("rejected"), mapped.get("image")


def resolve_column(role, available_columns, training_mode):
    lowered = {col.lower(): col for col in available_columns}
    aliases = TRAINING_SCHEMAS[training_mode]["aliases"]
    for alias in aliases.get(role, []):
        if alias in lowered:
            return lowered[alias]
    return None


def mapped_summary(col_question, col_chosen, col_rejected, col_image=None):
    used = []
    if col_image:    used.append(f" image    \u2192 {col_image}")
    if col_question: used.append(f" question \u2192 {col_question}")
    if col_chosen:   used.append(f" chosen   \u2192 {col_chosen}")
    if col_rejected: used.append(f" rejected \u2192 {col_rejected}")
    if used:
        print("\nMapped Roles:")
        for line in used:
            print(" ", line)
    else:
        print("No usable column roles mapped!")


def unbox_field(text: str) -> str:
    t = text.strip()
    m = UNBOX_RE.match(t)
    if not m:
        return text
    inner = BOX_INNER_WRAP_RE.sub(r"\1", m.group(1))
    return inner.strip()


def strip_latex(match) -> str:
    val = match.group(0)
    return val if re.fullmatch(r"\$\d[\d,]*(\.\d{1,2})?\$", val) else ""


def clean_string(s, mode: str = "math"):
    s = "" if s is None else str(s)
    if not s:
        return ""

    s = unbox_field(s)
    if mode != "code":
        s = WHITESPACE_ESC_RE.sub(" ", s).strip()
    else:
        s = s.strip()
    if mode != "code":
        s = FORMAT_STRIP_RE.sub("", s)

    if mode == "math":
        s = MATH_WRAPPERS_RE.sub(r"\1", s)
        s = DOLLAR_INLINE_RE.sub(r"\1", s)
        s = MATH_SIMPLE_MAP_RE.sub(lambda m: _MATH_SIMPLE_MAP[m.group(0)], s)
        s = MATH_SPACING_RE.sub(" ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()

    if mode not in {"latex", "math"}:
        s = RE_LATEX_INLINE.sub(strip_latex, s)
        s = RE_TEX_INLINE.sub("", s)
        s = re.sub(r"\\mathrm\{.*?\}", "", s)
    s = RE_LATEX_ENV.sub("", s)
    s = RE_HTML.sub("", s)
    s = re.sub(r'(?:\$+)?\\boxed\{([^}]*)\}(?:\$+)?', r'\1', s)
    s = re.sub(r"\[/?(?:INST|SYS|USER|ASSISTANT)\]", "", s)
    s = ANSWER_MARKER_RE.sub("", s)
    s = re.sub(r"\{\{.*?\}\}", "", s)
    s = re.sub(r"<\|.*?\|>", "", s)
    s = RE_MD_HEADER.sub("", s)
    s = RE_JUNK_WORDS.sub("", s)
    s = RE_ECOM_TAGS.sub("", s)
    s = re.sub(r"\*\*.*?\*\*", "", s)
    s = RE_UNICODE_GARBAGE.sub("", s)
    s = RE_MEDIA_PLACEHOLDERS.sub("", s)
    if mode not in {"latex"}:
        s = RE_TRASH_PLACEHOLDER.sub("", s)
    s = s.translate(PUNC_TRANS)
    if _EMOJI_AVAILABLE:
        s = _emoji_mod.replace_emoji(s, "")
    if mode not in {"latex", "code"}:
        s = re.sub(r"[^\x00-\x7F]+", "", s)
    if BAD_TOKENS_RE.search(s):
        return ""
    return s.strip()


def synthesize_prompt_dataset(raw_dataset, col_question, col_chosen, col_rejected, training_mode, clean, col_image=None):
    original_count = len(raw_dataset)

    if training_mode == "multimodal":
        def build_multimodal_row(row):
            parts = []
            if col_question:
                q = row.get(col_question, "") or ""
                if str(q).strip():
                    parts.append(clean(str(q)))
            if col_chosen:
                a = row.get(col_chosen, "") or ""
                if str(a).strip():
                    parts.append(clean(str(a)))
            text = "\n\n".join(parts)
            result = {"text": text}
            if col_image and col_image != "image":
                result["image"] = row[col_image]
            return result

        processed = raw_dataset.map(
            build_multimodal_row,
            num_proc=1,
            writer_batch_size=50,
        )
        if col_image and col_image != "image" and col_image in processed.column_names:
            processed = processed.rename_column(col_image, "image")
        cols_to_drop = [c for c in processed.column_names if c not in ("text", "image")]
        if cols_to_drop:
            processed = processed.remove_columns(cols_to_drop)
        processed = processed.filter(lambda e: bool(str(e["text"]).strip()), num_proc=1)
        _info(f"Retained {len(processed):,} / {original_count:,} rows")
    else:
        df = raw_dataset.to_pandas()

        def build_prompt(row):
            parts = []
            if col_question and pd.notna(row.get(col_question, "")):
                parts.append(clean(str(row[col_question])))
            if training_mode == "sft":
                if col_chosen and pd.notna(row.get(col_chosen, "")):
                    parts.append(clean(str(row[col_chosen])))
            return "\n\n".join(parts)

        df["text"] = df.apply(build_prompt, axis=1)
        df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
        df = df[["text"]]
        _info(f"Retained {len(df):,} / {original_count:,} rows")
        processed = Dataset.from_pandas(df)

    if len(processed) > 0:
        first_raw = raw_dataset[0]
        q = clean(str(first_raw.get(col_question, ""))) if col_question else ""
        a = clean(str(first_raw.get(col_chosen, ""))) if col_chosen else ""
        _width = 58
        if q or a:
            print(f"\n  {'-' * _width}")
            print(f"  Sample row")
            print(f"  {'-' * _width}")
        if q:
            print(f"  Prompt   : {q[:120]}{chr(0x2026) if len(q) > 120 else ''}")
        if a:
            print(f"  Response : {a[:120]}{chr(0x2026) if len(a) > 120 else ''}")
        if q or a:
            print()
    else:
        _err("No usable prompt found!")

    return processed


def load_and_prepare_dataset(dataset_path, cleaning_mode, image_mode=False, base_model_override=None, output_dir=None, merge_dir=None, merge_success_flag=None):
    from transformers import AutoTokenizer, AutoProcessor

    _num_proc = min(12, os.cpu_count() or 4)

    is_hf_hub = not os.path.exists(dataset_path) and "/" in dataset_path
    if is_hf_hub:
        _info(f"Source: HF Hub \u2014 {dataset_path}")
        _prev_offline = os.environ.pop("HF_DATASETS_OFFLINE", None)
        _prev_tr_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
        _prev_hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            raw_dataset = load_dataset(dataset_path, split="train")
        finally:
            if _prev_offline:    os.environ["HF_DATASETS_OFFLINE"] = _prev_offline
            if _prev_tr_offline: os.environ["TRANSFORMERS_OFFLINE"] = _prev_tr_offline
            if _prev_hf_offline: os.environ["HF_HUB_OFFLINE"] = _prev_hf_offline
    else:
        raw_dataset = load_dataset("parquet", data_files=dataset_path, split="train")

    shuffle_seed = random.randint(1, 999999)
    raw_dataset = raw_dataset.shuffle(seed=shuffle_seed)
    columns = raw_dataset.column_names

    if image_mode:
        training_mode = "multimodal"
    else:
        training_mode = detect_training_mode(columns)

    col_question, col_chosen, col_rejected, col_image = auto_map_roles(columns, training_mode)
    _banner("Dataset")
    _info(f"Mode: {training_mode.upper()}  |  Cleaning: {cleaning_mode or 'math'}  |  Seed: {shuffle_seed}")
    img_label = "\u2014" if not col_image else col_image
    q_label = "\u2014" if not col_question else col_question
    a_label = "\u2014" if not col_chosen else col_chosen
    r_label = "\u2014" if not col_rejected else col_rejected
    _info(f"Image    \u2192 {img_label}")
    _info(f"Question \u2192 {q_label}")
    _info(f"Answer   \u2192 {a_label}")
    _info(f"Rejected \u2192 {r_label}")
    effective_mode = cleaning_mode or "math"
    clean = lambda s: clean_string(s, mode=effective_mode)

    if merge_dir is None:
        merge_dir = MERGED_DIR
    if merge_success_flag is None:
        merge_success_flag = MERGE_SUCCESS_FLAG

    model_name = _resolve_model_name(merge_dir, merge_success_flag, base_model_override)
    if os.path.exists(merge_success_flag):
        _ok(f"Resuming from previous merged model: {merge_dir}")
    else:
        if base_model_override:
            base_label = f"override: {base_model_override}"
        elif image_mode:
            base_label = "image base (Phi-3-Vision)"
        else:
            base_label = f"text base ({os.path.basename(base_model_override) if base_model_override else 'selected'})"
        _info(f"Base model: {model_name}")

    profile = resolve_model_profile(model_name)
    processor = None
    if image_mode:
        _info("Loading AutoProcessor (Phi-3-Vision)...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=profile["use_fast_tokenizer"],
            trust_remote_code=profile["trust_remote_code"],
            local_files_only=True
        )

    if not image_mode and profile["inject_special_tokens"]:
        special_tokens = {
            "pad_token": "<|pad|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>"
        }
        tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or tokenizer.unk_token

    if training_mode == "causal":
        _info("Causal mode \u2014 GPT-style synthesis will be applied")
        def synth_gpt_prompt(example):
            instruction = example.get("instruction", "").strip()
            input_ = example.get("input", "").strip()
            output = example.get("output", "").strip()
            if instruction and input_:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            return {"text": clean(text)}
        processed = raw_dataset.map(synth_gpt_prompt, num_proc=_num_proc)
    else:
        processed = synthesize_prompt_dataset(
            raw_dataset, col_question, col_chosen, col_rejected,
            training_mode, clean, col_image=col_image
        )

    _banner("Tokenization")
    _info("Measuring sequence lengths...")
    processed, auto_len = measure_lengths(processed, tokenizer)

    _info("Tokenizing dataset...")
    protected = {"length"}
    if image_mode:
        protected.add("image")
    cols_to_remove = [c for c in processed.column_names if c not in protected]

    tokenized = processed.map(
        lambda e: tokenize(
            example=e,
            tokenizer=tokenizer,
            training_mode=training_mode,
            max_length=auto_len,
            processor=processor
        ),
        remove_columns=cols_to_remove,
        batched=False,
        num_proc=1 if image_mode else _num_proc,
        writer_batch_size=50 if image_mode else 1000,
        keep_in_memory=False,
    )

    if image_mode:
        before = len(tokenized)
        tokenized = tokenized.filter(
            lambda e: len(e["input_ids"]) > 0 and e.get("pixel_values") is not None
        )
        dropped = before - len(tokenized)
        if dropped:
            _warn(f"Dropped {dropped} rows with unreadable images")

    _info("Casting columns to tensor format...")
    tokenized = tokenized.cast_column("input_ids", Sequence(Value("int64")))
    tokenized = tokenized.cast_column("attention_mask", Sequence(Value("int64")))
    tokenized = tokenized.cast_column("labels", Sequence(Value("int64")))
    if "length" in tokenized.column_names:
        tokenized = tokenized.cast_column("length", Value("int32"))

    fmt_cols = ["input_ids", "attention_mask", "labels"]
    if "length" in tokenized.column_names:
        fmt_cols.append("length")
    if image_mode and "pixel_values" in tokenized.column_names:
        fmt_cols.append("pixel_values")

    tokenized.set_format(type="torch", columns=fmt_cols)
    roc()

    train_dataset = tokenized
    eval_dataset = None
    if TRAINING_CHAIN_PATH:
        just_logged = log_dataset(dataset_path, TRAINING_CHAIN_PATH)
        if just_logged:
            purge_checkpoints(output_dir if output_dir else OUTPUT_DIR)
    return train_dataset, eval_dataset, tokenizer, training_mode, model_name, processor


def _resolve_model_name(merged_dir: str, merge_success_flag: str, base_model: str) -> str:
    if os.path.exists(merge_success_flag):
        return merged_dir
    if os.path.exists(merged_dir):
        raise RuntimeError(
            f"[ERROR] Refusing to overwrite {merged_dir} \u2014 no success.txt found, but directory exists. "
            "You may be about to overwrite a previous model. Please verify or delete the folder manually."
        )
    return base_model
