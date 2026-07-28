# Utility evaluation via VLMEvalKit

Utility numbers in the paper (MMMU, MathVista, MM-Vet, MIA-Bench) come from
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) with SafeCoDe wired
into its inference loop. VLMEvalKit is Apache 2.0; rather than fork 500+ files
we describe the changes so you can apply them to a clean checkout.

> **Status:** this is a written patch guide, not yet a machine-applicable
> `.diff`. The original working tree had no git history to diff against, so
> generating a verified patch requires re-applying these edits to the pinned
> commit and running `git diff`. Doing that and committing the resulting
> `vlmevalkit.patch` here is a worthwhile follow-up.

## Pinned upstream

```bash
git clone https://github.com/open-compass/VLMEvalKit
cd VLMEvalKit
git checkout 989e33b28a2f14191ffea711936c56ccb96255ff
pip install -e .
```

## Changes

### 1. `run.py`

- Add a `--defender` argument accepting `none`, `paraphrase`, `retokenization`,
  `self-reminder`, `cot`, `ppl`, `self-exam`, `adashield`, `DPP`, `ours`.
  Only `none` and `ours` are needed to reproduce the SafeCoDe rows; the rest
  belong to the baseline study, which is not part of this release.
- As the first statement in `main()`, publish the choice to the environment so
  the inference layer can read it: `os.environ['MM_DEFENDER'] = (args.defender or 'none').lower()`.
- Prefix result filenames with the active defender so runs do not overwrite
  each other, and pass that filename through to `infer_data_job(...)` as
  `result_file_name=`.

### 2. `vlmeval/inference.py`

- Import SafeCoDe: `REFUSAL_PREFIXES` from `safecode.prompt`, and
  `apply_gaussian_noise` / `contrastive_decode_multistep_with_modulation` from
  `safecode.utils`, plus `prepare_caption_and_verdict` from `safecode.judge`.
  Install this repository (`pip install -e /path/to/SafeCoDe`) so these resolve
  as normal packages — **do not** replicate the original `sys.path.append('../')`
  hops.
- Add three keyword parameters to `infer_data(...)`: `defender="none"`,
  `defense=None`, `refuse_text="Sorry, I cannot answer your question."`. All
  have defaults, so `inference_mt.py` and `inference_video.py` keep working
  untouched.
- In the per-sample loop, replace the single `model.generate(...)` call with a
  branch: when the defender is `ours`, resolve the image to a path (writing
  PIL images to `{work_dir}/_ours_tmp_images/{hash}.png`), build the
  first-person system prompt, call `prepare_caption_and_verdict(...)` and
  `apply_gaussian_noise(image, stddev=0.1)`, then
  `contrastive_decode_multistep_with_modulation(...)` with
  `alpha=0.5, max_steps=3, top_k=5, lambda_supp=0.5, lambda_boost=0.5`.
  Otherwise fall through to the upstream call.
- Add `result_file_name=None` to `infer_data_job(...)` and key the `_PREV.pkl`
  cache and per-rank shards off it, so different defenders do not collide.

The SafeCoDe system prompt used here is the same first-person wording as
MSSBench's `PROMPT_CHAT_IF` (see `benchmarks/prompts.py`).

### 3. `vlmeval/config.py`

Register the InstructBLIP adapter below:

```python
from vlmeval.vlm.instructionblip import InstructionBLIP

supported_VLM.update({
    "InstructionBLIP_Vicuna_7B": partial(
        InstructionBLIP,
        model_path="Salesforce/instructblip-vicuna-7b",
        device_map="auto", max_new_tokens=512, temperature=0.0, top_p=1.0,
    ),
})
```

### 4. `vlmeval/vlm/instructionblip.py` (new)

A separate adapter from upstream's `instructblip.py` (note the spelling). It
exists because SafeCoDe needs **both** `self.processor` and `self.tokenizer` on
the model object, and upstream's adapter sets only the former. It also clamps
Q-Former input length to `min(qformer_max, 512) - 2`.

### 5. Optional environment fixes

- `vlmeval/vlm/qwen2_vl/model.py`: switch `attn_implementation` from
  `flash_attention_2` to `sdpa` if FlashAttention is not installed.
- `vlmeval/vlm/idefics.py`: in `IDEFICS.generate_inner`, convert image
  fragments to RGB explicitly and pass `prompts=[segments]` with
  `add_end_of_utterance_token=False`.

## Known issue

`Qwen2VLChat` exposes `.processor` but no `.tokenizer`. With `--defender ours`
the SafeCoDe branch therefore raises internally and **silently falls back to
plain generation** — a Qwen2-VL utility run will look like it applied SafeCoDe
when it did not. Either attach a tokenizer to that adapter or assert on the
attribute before trusting the numbers.
