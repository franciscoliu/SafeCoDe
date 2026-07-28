"""Stage 2 judge: caption the image, then emit a safe/unsafe verdict.

SafeCoDe needs a scene-level safety verdict to decide whether to boost or
suppress refusal tokens during decoding. Two interchangeable backends are
provided:

``gpt4o`` (default)
    Captions with GPT-4o and asks GPT-4o for the verdict. Reproduces the
    paper. Requires ``OPENAI_API_KEY`` and costs two API calls per sample.

``qwen``
    Uses a single local Qwen-VL model for both steps. No API key, no network
    after the weights are cached, no per-sample cost -- at some accuracy
    cost relative to GPT-4o.

Select a backend once at startup (``safecode.main`` does this from
``--judge``); every call site then goes through
:func:`prepare_caption_and_verdict`.
"""

from __future__ import annotations

from safecode import config

__all__ = ["set_judge", "get_judge", "prepare_caption_and_verdict", "AVAILABLE_JUDGES"]

AVAILABLE_JUDGES = ("gpt4o", "qwen")

_active_judge = "gpt4o"


def set_judge(name: str) -> None:
    """Choose the judge backend. Validates eagerly so a typo fails at startup."""
    global _active_judge
    name = (name or "gpt4o").lower()
    if name not in AVAILABLE_JUDGES:
        raise ValueError(
            f"Unknown judge {name!r}. Choose one of: {', '.join(AVAILABLE_JUDGES)}"
        )
    if name == "gpt4o":
        # Fail now, with a clear message, rather than thousands of samples in.
        config.require_openai_key("the gpt4o judge")
    _active_judge = name


def get_judge() -> str:
    return _active_judge


def prepare_caption_and_verdict(image_path: str, user_prompt: str):
    """Return ``(caption, verdict)`` where verdict is ``"safe"`` or ``"unsafe"``.

    Dispatches to whichever backend :func:`set_judge` selected.
    """
    # Imported lazily: safecode.utils pulls in torch and transformers, and we
    # do not want `import safecode.judge` to require them.
    if _active_judge == "qwen":
        from safecode.utils import prepare_caption_and_verdict_qwen

        return prepare_caption_and_verdict_qwen(
            image_path, user_prompt, model_repo=config.QWEN_JUDGE_MODEL
        )

    from safecode.utils import prepare_caption_and_verdict as _gpt_judge

    return _gpt_judge(image_path, user_prompt)
