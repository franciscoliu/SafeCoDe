# Reproducing the paper

Every command assumes you have installed the package (`pip install -e .`) and
set up `.env` (see the README).

## Models evaluated

| `--model_type` | `--model_path` |
|---|---|
| `llava` | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `instructionblip` | `Salesforce/instructblip-vicuna-7b` |
| `idefics` | `HuggingFaceM4/idefics-9b-instruct` |

Weights land in `$HF_HOME` if set, otherwise `~/.cache/huggingface`.

## Hyperparameters

The defaults in `safecode/main.py` are the paper's: `--alpha 0.3`,
`--max_steps 5`, `--top_k 20`, `--lambda_supp 1.0`, `--lambda_boost 1.0`,
Gaussian noise `stddev=0.1`.

## Contextual safety

MSSBench — undersensitivity, paired safe/unsafe images:

```bash
python scripts/download_data.py --benchmark mssbench --out data/

python -m safecode.main \
    --model_type llava --model_path llava-hf/llava-v1.6-vicuna-7b-hf \
    --method safecode --judge gpt4o --output_name safecode_llava \
    --mss_data_root data/mssbench --mss_output_dir results/mssbench
```

MOSSBench — oversensitivity, all-benign:

```bash
python scripts/download_data.py --benchmark mossbench --out data/

python -m safecode.main \
    --model_type llava --model_path llava-hf/llava-v1.6-vicuna-7b-hf \
    --method safecode --judge gpt4o --output_name safecode_llava \
    --moss_data_root data/mossbench --moss_output_dir results/mossbench
```

For the baseline row, swap `--method safecode` for `--method base`.

## General safety

MM-SafetyBench (needs the manual image download — see the script's output):

```bash
python -m safecode.main \
    --model_type llava --model_path llava-hf/llava-v1.6-vicuna-7b-hf \
    --method safecode --output_name safecode_llava \
    --mm_safebench_data_root data/mm-safetybench/data/processed_questions \
    --mm_safebench_img_root  data/mm-safetybench/data/imgs \
    --mm_safebench_output_dir results/mm-safetybench
```

FigStep:

```bash
python -m safecode.main \
    --model_type llava --model_path llava-hf/llava-v1.6-vicuna-7b-hf \
    --method safecode --output_name safecode_llava \
    --figstep_data_root data/figstep/data \
    --figstep_output_dir results/figstep
```

HADES and MSTS stream from the Hugging Face hub — no download step:

```bash
python -m safecode.main ... --hades_output_dir results/hades
python -m safecode.main ... --msts_output_dir results/msts --msts_language english
```

## Utility

See `patches/README.md`. Utility benchmarks run through a patched VLMEvalKit,
not through `safecode.main`.

## Cost and runtime

With `--judge gpt4o` every sample costs two API calls (caption + verdict).
MOSSBench is ~300 samples; MSSBench ~1820 pairs, i.e. two judged images each.
`--judge qwen` removes the API cost entirely at some accuracy cost, and adds an
8B model to GPU memory alongside the target model.

Each run writes `*_api_usage.json` (token counts and wall-clock per judge call)
and `*_efficiency.json` (caption and decode latency, mean/median/p95) next to
the results, which is where the efficiency table comes from.

Runs are resumable: re-running with the same `--output_name` skips samples
already present in the output JSON.

## Notes that affect the numbers

**Response length.** `contrastive_decode_multistep_with_modulation` overrides
`--total_max_new_tokens` based on the verdict and architecture: an `unsafe`
verdict caps generation at 20 new tokens for LLaVA, 30 for Qwen, 100 for
InstructBLIP, 200 for IDEFICS, while `safe` gets 256–1024. This is deliberate
and is part of the published configuration, but it means the CLI flag is not
in control — change the table at the top of that function if you want
uniform lengths.

**Judge choice changes results.** The paper's numbers use the GPT-4o judge.
`--judge qwen` is provided for cost-free replication and will not match the
published tables exactly.

**Sampling.** Generation uses `do_sample=True` (temperature 0.7–0.8), so runs
vary slightly. Set `torch.manual_seed` for a fixed contrastive perturbation;
the sampling itself is not seeded by default.
