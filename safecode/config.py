"""Central configuration for SafeCoDe.

Every tunable that used to be hardcoded (API keys, judge model names, model
cache directories) is read from the environment here, so nothing
machine-specific ever needs to live in the source tree.

Copy ``.env.example`` to ``.env`` and fill it in, or export the variables
directly. See the "Environment variables" section of the README.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "JUDGE_MODEL",
    "CAPTION_MODEL",
    "EVAL_MODEL",
    "MSS_JUDGE_MODEL",
    "MOSS_JUDGE_MODEL",
    "QWEN_JUDGE_MODEL",
    "HF_HOME",
    "MODEL_CACHE_DIR",
    "load_dotenv",
    "require_openai_key",
    "openai_headers",
]


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Load ``KEY=value`` pairs from a .env file into os.environ.

    Existing environment variables always win, so an explicit ``export`` or a
    CI secret overrides the file. Deliberately dependency-free -- we do not
    want to require python-dotenv just for this.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    path = Path(path)
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

# --- OpenAI ---------------------------------------------------------------
# Never hardcode a key. Leave unset to run with --judge qwen (fully local).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Point at a proxy or an Azure/vLLM-compatible gateway if you use one.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --- Judge / evaluator models --------------------------------------------
# Stage 2 of SafeCoDe: caption the image, then emit the safe/unsafe verdict.
CAPTION_MODEL = os.getenv("SAFECODE_CAPTION_MODEL", "gpt-4o")
JUDGE_MODEL = os.getenv("SAFECODE_JUDGE_MODEL", "gpt-4o")
# Local (no-API) judge used by `--judge qwen`.
QWEN_JUDGE_MODEL = os.getenv("SAFECODE_QWEN_JUDGE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# Benchmark-side scoring models (these grade outputs; they are not part of the
# method itself). Defaults reproduce the paper's setup.
EVAL_MODEL = os.getenv("SAFECODE_EVAL_MODEL", "gpt-4o")
MSS_JUDGE_MODEL = os.getenv("SAFECODE_MSS_JUDGE_MODEL", "gpt-4.1-nano")
MOSS_JUDGE_MODEL = os.getenv("SAFECODE_MOSS_JUDGE_MODEL", "gpt-4-turbo")

# --- Local model cache ----------------------------------------------------
# Replaces the previously hardcoded cluster scratch paths. If unset,
# transformers falls back to its own default (~/.cache/huggingface).
HF_HOME = os.getenv("HF_HOME")
MODEL_CACHE_DIR = os.getenv("SAFECODE_MODEL_CACHE_DIR") or HF_HOME or None


def require_openai_key(what: str = "this operation") -> str:
    """Return the API key, or fail with an actionable message."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set, but {what} needs it.\n"
            "  - export OPENAI_API_KEY=sk-...   (or copy .env.example to .env)\n"
            "  - or pass --judge qwen to run the local judge with no API access."
        )
    return OPENAI_API_KEY


def openai_headers() -> dict:
    """Standard auth headers for direct REST calls to the chat API."""
    return {
        "Authorization": f"Bearer {require_openai_key('an OpenAI request')}",
        "Content-Type": "application/json",
    }
