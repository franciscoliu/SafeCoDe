"""Loading the target MLLM that SafeCoDe decodes from.

Four architectures are supported, matching the paper: LLaVA-1.6, Qwen2.5-VL /
Qwen3-VL, InstructBLIP and IDEFICS.

Every ``transformers`` class is imported lazily, inside the branch that needs
it. That keeps ``import safecode.load_models`` working on any reasonably
recent transformers install: you only need the classes for the architecture
you actually run, and a version that lacks one of the others is not fatal.

Weights are cached wherever ``HF_HOME`` / ``SAFECODE_MODEL_CACHE_DIR`` points
(see safecode/config.py). Leave both unset to use the Hugging Face default of
``~/.cache/huggingface`` -- useful on clusters where $HOME has a small quota.
"""

import torch

from safecode.config import MODEL_CACHE_DIR

__all__ = ["load_model", "SUPPORTED_MODEL_TYPES"]

SUPPORTED_MODEL_TYPES = ("llava", "qwen", "instructionblip", "idefics")

_MIN_TRANSFORMERS = {
    "qwen": "4.49 (Qwen2.5-VL) / 4.57 (Qwen3-VL)",
    "idefics": "4.32",
    "llava": "4.39",
    "instructionblip": "4.31",
}


def _cache_kwargs():
    """Only pass cache_dir when the user actually configured one."""
    return {"cache_dir": MODEL_CACHE_DIR} if MODEL_CACHE_DIR else {}


def _missing(model_type, exc):
    import transformers

    return ImportError(
        f"Your transformers {transformers.__version__} does not provide the classes "
        f"needed for --model_type {model_type}. Requires roughly "
        f"{_MIN_TRANSFORMERS[model_type]} or newer.\n"
        f"  pip install -U 'transformers>=4.57.0'\n"
        f"Original error: {exc}"
    )


def load_model(args):
    """Return ``(model, processor, tokenizer)`` for ``args.model_type``."""
    model_type = args.model_type.lower()
    model_path = args.model_path
    dtype = torch.float16
    cache = _cache_kwargs()

    if model_type == "llava":
        try:
            from transformers import (
                LlavaNextForConditionalGeneration,
                LlavaNextProcessor,
            )
        except ImportError as exc:
            raise _missing(model_type, exc) from exc

        print(f"Loading LLaVA model from {model_path} ...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", torch_dtype=dtype, **cache
        )
        processor = LlavaNextProcessor.from_pretrained(model_path, **cache)
        return model, processor, processor.tokenizer

    if model_type == "qwen":
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise _missing(model_type, exc) from exc

        print(f"Loading Qwen-VL model from {model_path} ...")
        min_pixels = 256 * 28 * 28
        max_pixels = 256 * 28 * 28

        if "qwen3" in model_path.lower():
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError as exc:
                raise _missing(model_type, exc) from exc
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", **cache
            ).eval()
        else:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
            except ImportError as exc:
                raise _missing(model_type, exc) from exc
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.bfloat16, **cache
            ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, **cache)
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels, **cache
        )
        return model, processor, tokenizer

    if model_type == "instructionblip":
        try:
            from transformers import (
                InstructBlipForConditionalGeneration,
                InstructBlipProcessor,
            )
        except ImportError as exc:
            raise _missing(model_type, exc) from exc

        print(f"Loading InstructBLIP model from {model_path} ...")
        processor = InstructBlipProcessor.from_pretrained(model_path, **cache)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", torch_dtype=dtype, **cache
        ).eval()
        return model, processor, processor.tokenizer

    if model_type == "idefics":
        try:
            from transformers import (
                AutoProcessor,
                GenerationConfig,
                IdeficsForVisionText2Text,
            )
        except ImportError as exc:
            raise _missing(model_type, exc) from exc

        print(f"Loading IDEFICS model from {model_path} ...")
        processor = AutoProcessor.from_pretrained(model_path, **cache)
        model = IdeficsForVisionText2Text.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16, **cache
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_path, **cache)
        return model, processor, processor.tokenizer

    raise ValueError(
        f"Unsupported model type: {model_type!r}. "
        f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
    )
