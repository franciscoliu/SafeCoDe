# Attribution and third-party licences

SafeCoDe evaluates on six external benchmarks. **None of their data is
redistributed in this repository** — `scripts/download_data.py` fetches
everything from the original distributors, so the licence you accept is
theirs, not ours.

Read this page before downloading. Two of these benchmarks carry restrictions
that are stricter than this repository's MIT licence.

## Benchmarks

| Benchmark | Licence | Upstream | Paper |
|---|---|---|---|
| MSSBench | MIT, © 2024 UCSC ERIC Lab | [eric-ai-lab/MSSBench](https://github.com/eric-ai-lab/MSSBench) · [`kzhou35/mssbench`](https://huggingface.co/datasets/kzhou35/mssbench) | [arXiv:2410.06172](https://arxiv.org/abs/2410.06172), ICLR 2025 |
| MOSSBench | **No LICENSE file.** README states CC BY-SA 4.0 | [xirui-li/MOSSBench](https://github.com/xirui-li/MOSSBench) · [`AIcell/MOSSBench`](https://huggingface.co/datasets/AIcell/MOSSBench) | [arXiv:2406.17806](https://arxiv.org/abs/2406.17806) |
| MM-SafetyBench | **No LICENSE file.** README states CC BY-NC 4.0 (data only) | [isXinLiu/MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench) | [arXiv:2311.17600](https://arxiv.org/abs/2311.17600), ECCV 2024 |
| HADES | MIT, © 2024 Yifan Li | [RUCAIBox/HADES](https://github.com/RUCAIBox/HADES) · [`Monosail/HADES`](https://huggingface.co/datasets/Monosail/HADES) | [arXiv:2403.09792](https://arxiv.org/abs/2403.09792), ECCV 2024 Oral |
| FigStep | MIT, © 2023 Contributors of FigStep | [ThuCCSLab/FigStep](https://github.com/ThuCCSLab/FigStep) | [arXiv:2311.05608](https://arxiv.org/abs/2311.05608), AAAI 2025 Oral |
| MSTS | see dataset card | [`felfri/MSTS`](https://huggingface.co/datasets/felfri/MSTS) | — |

## Restrictions worth reading twice

**MOSSBench** ships no licence file. Its README states the new contributions
are CC BY-SA 4.0, and that the dataset **may be used commercially as a test
set but must not be used as a training set**. Verify the current terms
upstream before relying on this summary.

**MM-SafetyBench** ships no licence file either. Its README declares the data
**CC BY-NC 4.0 — non-commercial, research use only**, and additionally
subject to the OpenAI and Stable Diffusion terms under which it was generated.
No code licence is stated at all.

**HADES** asks users to "not use the data for any illegal or harmful
activities" and to ensure responsible and ethical use.

**FigStep** warns prominently that the repository contains harmful model
responses. It is a jailbreak benchmark; treat its contents accordingly.

## Code adapted in this repository

Two files under `benchmarks/` are adapted from upstream rather than written
from scratch. Both carry a header naming their origin:

- `benchmarks/mssbench_judge.py` — from MSSBench's `utils/gpt4_eval.py`.
  Azure code path removed; model and credentials now read from
  `safecode/config.py`.
- `benchmarks/mossbench_evaluator.py` — from MOSSBench's `Evaluator.py`,
  `evaluation_prompts.py` and `utils/utils.py`. Rewritten against the OpenAI
  REST API instead of MOSSBench's multi-backend model wrapper, which removed
  five heavyweight dependencies (fschat, opencv-python, google-generativeai,
  anthropic, reka-api). Scoring prompts and parsing are unchanged.
- `benchmarks/prompts.py` — two task prompts copied verbatim from MSSBench.
