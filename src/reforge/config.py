import os
import sys
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added", category=UserWarning)

import logging as _logging
_logging.getLogger("datasets").setLevel(_logging.ERROR)
_logging.getLogger("transformers.tokenization_utils_base").setLevel(_logging.ERROR)
_logging.getLogger("transformers").setLevel(_logging.ERROR)

import torch

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

HF_HOME = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("TORCH_HOME", os.path.join(HF_HOME, "torch"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

DEFAULT_HF_MODELS_DIR = os.environ.get(
    "REFORGE_MODELS_DIR",
    os.path.join(os.path.expanduser("~"), "HF_Models")
)

DEFAULT_OUTPUT_DIR = os.environ.get(
    "REFORGE_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), "Reforge_Output")
)

MAX_LENGTH = 750
SAVE_STEPS = 150
SAVE_LIMITS = 2
LOG_STEPS = 1
BATCH_SIZE = 1

OUTPUT_DIR = os.environ.get(
    "REFORGE_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), "Reforge_Output")
)
MERGED_DIR = os.path.join(OUTPUT_DIR, "merged")
MERGE_SUCCESS_FLAG = os.path.join(MERGED_DIR, "success.txt")

TRAINING_CHAIN_PATH = os.path.join(
    OUTPUT_DIR, "training_chain.txt"
)
