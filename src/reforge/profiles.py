import os

MODEL_PROFILES = {
    "phi-2": {
        "attn_implementation":   "sdpa",
        "trust_remote_code":     False,
        "use_fast_tokenizer":    False,
        "inject_special_tokens": True,
        "resize_embeddings":     True,
        "lora_r":                16,
        "lora_alpha":            32,
        "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save":       None,
        "grad_accum":            16,
        "learning_rate":         2e-5,
        "safe_serialization":    True,
        "enable_input_grads":    False,
    },
    "phi-3-vision": {
        "attn_implementation":   "eager",
        "trust_remote_code":     True,
        "use_fast_tokenizer":    True,
        "inject_special_tokens": False,
        "resize_embeddings":     False,
        "vision":                True,
        "lora_r":                16,
        "lora_alpha":            32,
        "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save":       ["lm_head"],
        "grad_accum":            8,
        "learning_rate":         1e-5,
        "safe_serialization":    False,
        "enable_input_grads":    True,
    },
    "phi-3": {
        "attn_implementation":   "sdpa",
        "trust_remote_code":     True,
        "use_fast_tokenizer":    True,
        "inject_special_tokens": False,
        "resize_embeddings":     False,
        "lora_r":                16,
        "lora_alpha":            32,
        "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save":       ["lm_head"],
        "grad_accum":            8,
        "learning_rate":         1e-5,
        "safe_serialization":    False,
        "enable_input_grads":    True,
    },
}

MODEL_PROFILE_DEFAULT = {
    "attn_implementation":   "sdpa",
    "trust_remote_code":     True,
    "use_fast_tokenizer":    True,
    "inject_special_tokens": False,
    "resize_embeddings":     False,
    "lora_r":                16,
    "lora_alpha":            32,
    "lora_targets":          ["q_proj", "k_proj", "v_proj", "o_proj"],
    "modules_to_save":       ["lm_head"],
    "grad_accum":            8,
    "learning_rate":         2e-5,
    "safe_serialization":    False,
    "enable_input_grads":    True,
}


def resolve_model_profile(model_path: str) -> dict:
    name = os.path.basename(model_path).lower()
    for key, profile in MODEL_PROFILES.items():
        if key in name:
            return profile
    return MODEL_PROFILE_DEFAULT


def resolve_profile_key(model_path: str) -> str:
    name = os.path.basename(model_path).lower()
    for key in MODEL_PROFILES:
        if key in name:
            return key
    return "default"
