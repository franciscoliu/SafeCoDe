"""Command-line entry point for SafeCoDe benchmark runs.

Each benchmark is opt-in: pass its ``--*_output_dir`` (and ``--*_data_root``
where the data is local) and that benchmark runs. Several can run in one
invocation.

Examples
--------
Run MSSBench with the local judge and no API key::

    python -m safecode.main \\
        --model_type llava --model_path llava-hf/llava-v1.6-vicuna-7b-hf \\
        --judge qwen --output_name safecode_llava \\
        --mss_data_root data/mssbench --mss_output_dir results/mssbench

Run MOSSBench with the GPT-4o judge from the paper::

    python -m safecode.main \\
        --model_type qwen --model_path Qwen/Qwen2.5-VL-7B-Instruct \\
        --judge gpt4o --output_name safecode_qwen \\
        --moss_data_root data/mossbench --moss_output_dir results/mossbench
"""

import argparse

from safecode.judge import AVAILABLE_JUDGES, set_judge
from safecode.load_models import SUPPORTED_MODEL_TYPES, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="SafeCoDe: safety-aware contrastive decoding for MLLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- model ---
    model = parser.add_argument_group("model")
    model.add_argument("--model_type", type=str, default="llava",
                       choices=SUPPORTED_MODEL_TYPES,
                       help="Target MLLM architecture")
    model.add_argument("--model_path", type=str, required=True,
                       help="Local path or Hugging Face hub ID")
    model.add_argument("--output_name", type=str, required=True,
                       help="Name used for this run's output files")

    # --- method ---
    method = parser.add_argument_group("method")
    method.add_argument("--method", type=str, default="safecode",
                        choices=["safecode", "base"],
                        help="'safecode' applies contrastive decoding + token "
                             "modulation; 'base' is vanilla generation")
    method.add_argument("--judge", type=str, default="gpt4o",
                        choices=list(AVAILABLE_JUDGES),
                        help="Stage-2 safety judge. 'gpt4o' needs OPENAI_API_KEY; "
                             "'qwen' runs locally with no API access")
    method.add_argument("--alpha", type=float, default=0.3,
                        help="Contrastive scaling factor on the noised-image logits")
    method.add_argument("--max_steps", type=int, default=5,
                        help="Number of guided decoding steps before normal generation")
    method.add_argument("--top_k", type=int, default=20,
                        help="Top-k tokens considered during guided steps")
    method.add_argument("--lambda_supp", type=float, default=1.0,
                        help="Refusal-token suppression strength on a 'safe' verdict")
    method.add_argument("--lambda_boost", type=float, default=1.0,
                        help="Refusal-token boost strength on an 'unsafe' verdict")
    method.add_argument("--total_max_new_tokens", type=int, default=256,
                        help="Generation budget. NOTE: the decoder overrides this "
                             "per verdict and per architecture -- see the "
                             "'Response length' note in the README")

    # --- benchmarks ---
    mss = parser.add_argument_group("MSSBench (contextual safety)")
    mss.add_argument("--mss_data_root", type=str, default=None,
                     help="Directory holding combined.json plus chat/ and embodied/")
    mss.add_argument("--mss_output_dir", type=str, default=None)

    moss = parser.add_argument_group("MOSSBench (oversensitivity)")
    moss.add_argument("--moss_data_root", type=str, default=None,
                      help="Directory holding images_information/information.json")
    moss.add_argument("--moss_output_dir", type=str, default=None)
    moss.add_argument("--moss_data_list", nargs="+", default=None,
                      help="Restrict to these sample ids")
    moss.add_argument("--moss_data_offset", type=int, default=0)
    moss.add_argument("--moss_inference", default=True,
                      action=argparse.BooleanOptionalAction)
    moss.add_argument("--moss_eval", default=True,
                      action=argparse.BooleanOptionalAction)

    mm = parser.add_argument_group("MM-SafetyBench (general safety)")
    mm.add_argument("--mm_safebench_data_root", type=str, default=None,
                    help="Directory of processed_questions/*.json")
    mm.add_argument("--mm_safebench_img_root", type=str, default=None,
                    help="Directory of <scenario>/<SD|SD_TYPO|TYPO>/<qid>.jpg")
    mm.add_argument("--mm_safebench_output_dir", type=str, default=None)

    fig = parser.add_argument_group("FigStep (typographic jailbreak)")
    fig.add_argument("--figstep_data_root", type=str, default=None,
                     help="Directory holding question/safebench.csv and images/")
    fig.add_argument("--figstep_output_dir", type=str, default=None)

    hades = parser.add_argument_group("HADES (downloads from Hugging Face)")
    hades.add_argument("--hades_output_dir", type=str, default=None)

    msts = parser.add_argument_group("MSTS (downloads from Hugging Face)")
    msts.add_argument("--msts_language", type=str, default="english",
                      help="Language split of felfri/MSTS")
    msts.add_argument("--msts_output_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate the judge before loading several GB of weights.
    set_judge(args.judge)

    # Imported here rather than at module scope so that --help stays instant
    # and does not require torch to be installed.
    from safecode.eval import (
        eval_figstep,
        eval_hades,
        eval_mm_safebench,
        eval_moss_bench,
        eval_mss_bench,
        eval_msts,
    )

    requested = [
        bool(args.mss_data_root and args.mss_output_dir),
        bool(args.moss_data_root and args.moss_output_dir),
        bool(args.mm_safebench_data_root and args.mm_safebench_output_dir),
        bool(args.figstep_data_root and args.figstep_output_dir),
        bool(args.hades_output_dir),
        bool(args.msts_output_dir),
    ]
    if not any(requested):
        raise SystemExit(
            "No benchmark selected. Pass at least one --*_output_dir "
            "(plus the matching --*_data_root for MSSBench, MOSSBench, "
            "MM-SafetyBench and FigStep). See --help."
        )

    model, processor, tokenizer = load_model(args)

    if args.mss_data_root and args.mss_output_dir:
        print("Running MSSBench evaluation ...")
        c_safe, c_unsafe, c_total, e_safe, e_unsafe, e_total = eval_mss_bench(
            args, model, processor, tokenizer
        )
        print(f"MSSBench chat:     safe={c_safe} unsafe={c_unsafe} total={c_total}")
        print(f"MSSBench embodied: safe={e_safe} unsafe={e_unsafe} total={e_total}")

    if args.moss_data_root and args.moss_output_dir:
        print("Running MOSSBench evaluation ...")
        eval_moss_bench(args, model, processor, tokenizer)

    if args.mm_safebench_data_root and args.mm_safebench_output_dir:
        print("Running MM-SafetyBench evaluation ...")
        eval_mm_safebench(args, model, processor, tokenizer)

    if args.figstep_data_root and args.figstep_output_dir:
        print("Running FigStep evaluation ...")
        eval_figstep(args, model, processor, tokenizer)

    if args.hades_output_dir:
        print("Running HADES evaluation ...")
        eval_hades(args, model, processor, tokenizer)

    if args.msts_output_dir:
        print(f"Running MSTS evaluation (language={args.msts_language}) ...")
        eval_msts(args, model, processor, tokenizer)


if __name__ == "__main__":
    main()
