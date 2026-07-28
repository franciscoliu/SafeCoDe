#!/usr/bin/env python3
"""Run SafeCoDe on a single image + query.

Two modes:

  # Full run -- needs a GPU, model weights, and (for --judge gpt4o) an API key
  python demo.py --image photo.jpg --prompt "Any tips for running faster here?"

  # Install check -- no weights, no GPU, no network
  python demo.py --dry-run

The full run prints the caption, the safety verdict, and the SafeCoDe output,
optionally next to an unmodified baseline generation so the difference is
visible.

No example image ships with this repository: the images used in the paper
belong to the benchmark authors and are not ours to redistribute. Use any
photo of your own, or fetch a benchmark with scripts/download_data.py.
"""

import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(
        description="Run SafeCoDe on one image + query",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", help="Path to the input image")
    p.add_argument("--prompt", help="User query about the image")
    p.add_argument("--model_type", default="llava",
                   choices=["llava", "qwen", "instructionblip", "idefics"])
    p.add_argument("--model_path", default="llava-hf/llava-v1.6-vicuna-7b-hf",
                   help="Local path or Hugging Face hub ID")
    p.add_argument("--judge", default="gpt4o", choices=["gpt4o", "qwen"],
                   help="'gpt4o' needs OPENAI_API_KEY; 'qwen' runs locally")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--lambda_supp", type=float, default=1.0)
    p.add_argument("--lambda_boost", type=float, default=1.0)
    p.add_argument("--noise_std", type=float, default=0.1,
                   help="Gaussian noise stddev for the contrastive view")
    p.add_argument("--compare_baseline", action="store_true",
                   help="Also generate without SafeCoDe, for side-by-side")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Check the install without loading any model")
    return p


def dry_run(args):
    """Verify everything that does not require model weights or network."""
    import numpy as np
    from PIL import Image

    print("SafeCoDe dry run -- no model weights, no GPU, no network.\n")
    ok = True

    from safecode import __version__, config
    print(f"[ok] safecode {__version__} imported")
    print(f"     OPENAI_API_KEY  : {'set' if config.OPENAI_API_KEY else 'not set'}")
    print(f"     OPENAI_BASE_URL : {config.OPENAI_BASE_URL}")
    print(f"     model cache dir : {config.MODEL_CACHE_DIR or 'default (~/.cache/huggingface)'}")
    print(f"     judge models    : caption={config.CAPTION_MODEL} verdict={config.JUDGE_MODEL}")
    print(f"     local judge     : {config.QWEN_JUDGE_MODEL}")

    from safecode.judge import AVAILABLE_JUDGES, get_judge, set_judge
    try:
        set_judge(args.judge)
        print(f"[ok] judge backend '{get_judge()}' selected")
    except RuntimeError as exc:
        ok = False
        print(f"[!!] judge backend '{args.judge}' unavailable: {str(exc).splitlines()[0]}")
        print(f"     available: {', '.join(AVAILABLE_JUDGES)}")

    from safecode.utils import apply_gaussian_noise
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    noisy = apply_gaussian_noise(img, stddev=args.noise_std)
    delta = float(np.abs(np.asarray(noisy, np.int16) - np.asarray(img, np.int16)).mean())
    print(f"[ok] stage 1 noise: {img.size} {img.mode} -> {noisy.size} {noisy.mode}, "
          f"mean |delta| = {delta:.1f}/255")

    from safecode.prompt import REFUSAL_PREFIXES
    print(f"[ok] stage 2 refusal token space: {len(REFUSAL_PREFIXES)} prefixes")

    import torch
    print(f"[ok] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("     note: a full run needs a CUDA GPU; CPU inference on a 7B "
              "MLLM is impractically slow.")

    print("\nDry run " + ("passed." if ok else "finished with warnings (see [!!] above)."))
    print("Next:  python demo.py --image YOUR_IMAGE.jpg --prompt \"your question\"")
    return 0 if ok else 1


def main():
    args = build_parser().parse_args()

    if args.dry_run:
        return dry_run(args)

    if not args.image or not args.prompt:
        build_parser().error(
            "--image and --prompt are required for a full run "
            "(or pass --dry-run to just check the install)."
        )

    from PIL import Image

    from safecode.judge import prepare_caption_and_verdict, set_judge
    from safecode.load_models import load_model
    from safecode.prompt import REFUSAL_PREFIXES
    from safecode.utils import (
        apply_gaussian_noise,
        contrastive_decode_multistep_with_modulation,
    )

    set_judge(args.judge)

    print(f"Loading {args.model_type} from {args.model_path} ...")
    model, processor, tokenizer = load_model(args)

    print("\n--- Stage 2: scene-level safety judgement ---")
    caption, verdict = prepare_caption_and_verdict(args.image, args.prompt)
    print(f"caption : {caption}")
    print(f"verdict : {verdict}")

    print("\n--- Stage 1: contrastive view ---")
    neutral = apply_gaussian_noise(Image.open(args.image), stddev=args.noise_std)
    print(f"gaussian noise applied, stddev={args.noise_std}")

    if args.compare_baseline:
        from safecode.eval import run_inference
        print("\n--- Baseline (no SafeCoDe) ---")
        print(run_inference(model, processor, tokenizer, args.prompt,
                            args.image, model_type=args.model_type))

    print("\n--- SafeCoDe ---")
    output, _ = contrastive_decode_multistep_with_modulation(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        real_img_path=args.image,
        neutral_img=neutral,
        conversation=args.prompt,
        verdict=verdict,
        refusal_prefixes=REFUSAL_PREFIXES,
        model_type=args.model_type,
        alpha=args.alpha,
        max_steps=args.max_steps,
        top_k=args.top_k,
        lambda_supp=args.lambda_supp,
        lambda_boost=args.lambda_boost,
        generate_rest=True,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
