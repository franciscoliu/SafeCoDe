<div align="center">

# SafeCoDe

**Safety-aware Contrastive Decoding for Multimodal Large Language Models**

[![Paper](https://img.shields.io/badge/arXiv-2509.19212-b31b1b.svg)](https://arxiv.org/abs/2509.19212)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official implementation of
[*Steering Multimodal Large Language Models Decoding for Context-Aware Safety*](https://arxiv.org/abs/2509.19212)

Zheyuan Liu · Zhangchen Xu · Guangyao Dou · Xiangchi Yuan · Zhaoxuan Tan · Radha Poovendran · Meng Jiang

</div>

---

## What problem this solves

Multimodal LLMs fail at context-dependent safety in two opposite directions at
once. They are **undersensitive** — answering "any tips for running faster?"
without noticing the photo shows a cliff edge — and simultaneously
**oversensitive**, refusing benign requests because an image contains a knife
or a scalpel. Both failures come from the same place: refusal behaviour is
driven by textual priors, largely ignoring what the image actually shows.

SafeCoDe pushes the visual context back into the decoder. It is
**model-agnostic, training-free, and applied at inference time** — no
fine-tuning, no reward model, no changes to the underlying MLLM.

## How it works

```
                 ┌──────────────┐
   image  ───────┤ Gaussian     ├──── noised view ṽ
      │          │ noise (σ=0.1)│
      │          └──────────────┘
      │                                  Stage 1 — Contrastive decoding
      ├──────────────┬─────────────────────────────────────────────
      │              │      z_cd = z(v, x, y<t) − α · z(ṽ, x, y<t)
      │              │      surfaces the tokens that actually depend
      │              │      on what is in the image
      │              ▼
      │        ┌───────────┐
      └───────►│  judge    │─── "safe" / "unsafe"
   query  ────►│ (caption  │                       Stage 2 — Token modulation
                │ + verdict)│      unsafe → boost refusal tokens
                └───────────┘      safe   → suppress refusal tokens
                                   applied to the first few decoding steps,
                                   then normal generation completes the reply
```

**Stage 1** contrasts the logits of the real image against a Gaussian-noised
copy. Tokens whose probability barely moves were never grounded in the image;
tokens that shift sharply are the visually-sensitive ones.

**Stage 2** asks an auxiliary judge for a scene-level verdict, then nudges the
refusal token space accordingly — boosting it when the scene is genuinely
hazardous, suppressing it when the model is about to refuse something benign.
Only the first few decoding steps are steered; the rest of the response is
generated normally, which is what keeps helpfulness intact.

## Install

```bash
git clone https://github.com/franciscoliu/SafeCoDe
cd SafeCoDe
pip install -e ".[qwen]"          # drop [qwen] if you only run LLaVA/InstructBLIP/IDEFICS
```

To reproduce the paper's environment exactly, `pip install -r requirements-lock.txt`.

> **Match your torch build to your CUDA driver.** Plain `pip install torch`
> may fetch a wheel newer than your driver supports, in which case torch
> silently falls back to CPU and a 7B run crawls with no error message. Check
> `nvidia-smi` for your CUDA version and install the matching build, e.g. for
> CUDA 12.8:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```
> Confirm with `python demo.py --dry-run`, which prints `cuda available`.

## Environment variables

Copy the template and fill in what you need:

```bash
cp .env.example .env
```

`.env` is gitignored. Everything is read through `safecode/config.py`; no key
or path is ever hardcoded. Shell exports override the file.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | for `--judge gpt4o` | — | GPT-4o judge and benchmark scoring |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | Proxy or OpenAI-compatible gateway |
| `HF_HOME` | no | `~/.cache/huggingface` | Where model weights are cached |
| `SAFECODE_MODEL_CACHE_DIR` | no | falls back to `HF_HOME` | Overrides the cache for this project only |
| `HF_TOKEN` | no | — | Only for gated hub models/datasets |
| `SAFECODE_JUDGE_MODEL` | no | `gpt-4o` | Verdict model |
| `SAFECODE_CAPTION_MODEL` | no | `gpt-4o` | Caption model |
| `SAFECODE_QWEN_JUDGE_MODEL` | no | `Qwen/Qwen3-VL-8B-Instruct` | Local judge |
| `SAFECODE_EVAL_MODEL` | no | `gpt-4o` | Response scoring |
| `SAFECODE_MSS_JUDGE_MODEL` | no | `gpt-4.1-nano` | MSSBench scoring |
| `SAFECODE_MOSS_JUDGE_MODEL` | no | `gpt-4-turbo` | MOSSBench scoring |

**You can run the whole method without any API key** — pass `--judge qwen` and
the caption and verdict come from a local Qwen-VL model instead.

Be aware that judge quality bounds method quality, and a small local judge is
markedly weaker. Measured on all 300 MOSSBench samples — a set that is benign
by construction, so a perfect judge would answer `safe` every time — a
`Qwen2.5-VL-3B-Instruct` judge returned `unsafe` on 199 of 300. Use the 8B
default or GPU headroom for a larger judge if you can, and use `--judge gpt4o`
to match the paper. See [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Quickstart

Check the install — no weights, no GPU, no network:

```bash
python demo.py --dry-run
```

Then run it on an image:

```bash
python demo.py \
    --image your_photo.jpg \
    --prompt "Any tips on running faster here?" \
    --judge qwen \
    --compare_baseline
```

`--compare_baseline` prints the unmodified generation next to the SafeCoDe one,
which is the quickest way to see the method do something.

> No example image ships with this repo. The images in the paper belong to the
> benchmark authors and are not ours to redistribute — use your own photo, or
> fetch a benchmark with `scripts/download_data.py`.

## Benchmarks

```bash
python scripts/download_data.py --benchmark mssbench --out data/

python -m safecode.main \
    --model_type llava --model_path llava-hf/llava-v1.6-vicuna-7b-hf \
    --method safecode --judge gpt4o --output_name safecode_llava \
    --mss_data_root data/mssbench --mss_output_dir results/mssbench
```

| Benchmark | Measures | Data |
|---|---|---|
| MSSBench | undersensitivity | download |
| MOSSBench | oversensitivity | download |
| MM-SafetyBench | jailbreak robustness | download (images need a manual step) |
| FigStep | typographic jailbreaks | download |
| HADES | visual jailbreaks | streamed from the hub |
| MSTS | multilingual safety | streamed from the hub |
| MMMU / MathVista / MM-Vet / MIA-Bench | utility retention | via [patched VLMEvalKit](patches/README.md) |

Full commands in [docs/REPRODUCE.md](docs/REPRODUCE.md). Runs are resumable —
re-run with the same `--output_name` and finished samples are skipped.

## Method notes

Three things that materially affect results and are easy to miss:

- **Response length is verdict-dependent.** The decoder overrides
  `--total_max_new_tokens`: an `unsafe` verdict caps generation at 20 new
  tokens for LLaVA (30 Qwen, 100 InstructBLIP, 200 IDEFICS), while `safe` gets
  256–1024. This is part of the published configuration, not an accident, but
  it means the CLI flag is not in control.
- **The judge is part of the method.** Stage 2 needs a verdict; the quality of
  that verdict bounds the quality of the steering. `--judge gpt4o` reproduces
  the paper, `--judge qwen` trades accuracy for zero cost.
- **Generation is stochastic** (`do_sample=True`, temperature 0.7–0.8), so
  numbers move slightly between runs.

## What has been tested

For transparency about what is verified versus merely written:

| | Status |
|---|---|
| LLaVA-1.6-7B + local Qwen judge, single image | run on one A100 |
| MOSSBench, all 300 samples, `--method safecode` | run on one A100, completed |
| OpenAI judge/scoring layer | verified against a mocked transport |
| Import, lint, CLI, leak and secret scans | automated, all passing |
| Qwen / InstructBLIP / IDEFICS targets | **not yet run on hardware** |
| MSSBench, MM-SafetyBench, FigStep, HADES, MSTS drivers | **not yet run on hardware** |
| `--judge gpt4o` against the live API | **not yet run** |

Measured on one A100-80GB (LLaVA-1.6-7B target, Qwen2.5-VL-3B judge), per
MOSSBench sample: caption+verdict 0.87 s mean, decode 3.31 s mean (median
1.26 s, p95 8.18 s).

## Repository layout

```
safecode/       the method: decoding, judge, model loading, benchmark drivers
benchmarks/     minimal helpers adapted from MSSBench and MOSSBench
scripts/        data download
patches/        VLMEvalKit changes for utility evaluation
docs/           reproduction guide and third-party attribution
demo.py         single-image entry point
```

## Citation

```bibtex
@article{liu2025safecode,
  title   = {Steering Multimodal Large Language Models Decoding for Context-Aware Safety},
  author  = {Liu, Zheyuan and Xu, Zhangchen and Dou, Guangyao and Yuan, Xiangchi
             and Tan, Zhaoxuan and Poovendran, Radha and Jiang, Meng},
  journal = {arXiv preprint arXiv:2509.19212},
  year    = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE). This covers the code in this repository only.

The benchmarks are distributed by their own authors under their own terms and
are **not** redistributed here. Several carry restrictions stricter than MIT,
including non-commercial and no-training-set clauses. Read
[docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) before downloading them.

## Responsible use

This is safety research. Several of the benchmarks it evaluates on contain
jailbreak prompts and harmful model outputs by design. Use them for evaluating
and improving model safety, not for building attacks.
