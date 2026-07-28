#!/usr/bin/env python3
"""Fetch benchmark data from the original distributors.

No benchmark data is redistributed in this repository. Each dataset belongs to
its authors and carries its own licence -- read docs/ATTRIBUTION.md before
downloading, in particular the non-commercial and no-training-set clauses on
MM-SafetyBench and MOSSBench.

Usage:
    python scripts/download_data.py --benchmark mssbench --out data/
    python scripts/download_data.py --benchmark all --out data/
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BENCHMARKS = ["mssbench", "mossbench", "mm-safetybench", "figstep", "hades", "msts"]


def _hf_snapshot(repo_id: str, dest: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ! huggingface_hub is not installed:  pip install huggingface_hub")
        return False
    print(f"  downloading {repo_id} -> {dest}")
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=str(dest), local_dir_use_symlinks=False)
    return True


def get_mssbench(out: Path) -> bool:
    """MSSBench -- ICLR 2025, MIT licence, distributed on the HF hub."""
    print("MSSBench (contextual safety, 1820 pairs)")
    ok = _hf_snapshot("kzhou35/mssbench", out / "mssbench")
    if ok:
        print("  expected layout: combined.json, chat/, embodied/")
        print("  use with: --mss_data_root", out / "mssbench")
    return ok


def get_mossbench(out: Path) -> bool:
    """MOSSBench -- oversensitivity, 300 benign pairs.

    Fetched from GitHub rather than the hub: the HF dataset (AIcell/MOSSBench)
    ships the images but not images_information/information.json, which is the
    metadata SafeCoDe indexes on. The GitHub repo carries both under data/.
    """
    print("MOSSBench (oversensitivity, 300 pairs)")
    print("  NOTE: CC BY-SA 4.0; the authors prohibit use as a training set.")
    dest = out / "mossbench"
    if shutil.which("git") is None:
        print("  ! git not found; clone manually from "
              "https://github.com/xirui-li/MOSSBench")
        return False
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/xirui-li/MOSSBench", str(dest)],
            check=True,
        )
    info = dest / "data" / "images_information" / "information.json"
    if not info.is_file():
        print(f"  ! expected metadata missing at {info}")
        return False
    print(f"  metadata + images ready under {dest / 'data'}")
    print("  use with: --moss_data_root", dest / "data")
    return True


def get_mm_safetybench(out: Path) -> bool:
    """MM-SafetyBench -- questions from GitHub, images from the authors' Drive."""
    print("MM-SafetyBench (general safety, 13 scenarios)")
    print("  NOTE: data is CC BY-NC 4.0, research use only.")
    dest = out / "mm-safetybench"
    if shutil.which("git") is None:
        print("  ! git not found; clone manually from "
              "https://github.com/isXinLiu/MM-SafetyBench")
        return False
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/isXinLiu/MM-SafetyBench", str(dest)],
            check=True,
        )
    print(f"  questions ready at {dest / 'data' / 'processed_questions'}")
    print("  IMAGES ARE NOT ON THE HUB and must be fetched by hand:")
    print("    https://drive.google.com/file/d/1xjW9k-aGkmwycqGCXbru70FaSKhSDcR_/view")
    print(f"    unzip into {dest / 'data' / 'imgs'}")
    print("  then use: --mm_safebench_data_root", dest / "data" / "processed_questions",
          "--mm_safebench_img_root", dest / "data" / "imgs")
    return True


def get_figstep(out: Path) -> bool:
    """FigStep -- AAAI 2025, MIT licence, distributed only via GitHub."""
    print("FigStep (typographic jailbreak, SafeBench)")
    print("  WARNING: this benchmark contains harmful prompts by design.")
    dest = out / "figstep"
    if shutil.which("git") is None:
        print("  ! git not found; clone manually from "
              "https://github.com/ThuCCSLab/FigStep")
        return False
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ThuCCSLab/FigStep", str(dest)],
            check=True,
        )
    print("  use with: --figstep_data_root", dest / "data")
    return True


def get_hades(_out: Path) -> bool:
    print("HADES (ECCV 2024)")
    print("  No download needed -- streamed at run time from Monosail/HADES.")
    print("  The authors ask that it be used only for lawful, ethical research.")
    print("  use with: --hades_output_dir results/hades")
    return True


def get_msts(_out: Path) -> bool:
    print("MSTS (multilingual safety test suite)")
    print("  No download needed -- streamed at run time from felfri/MSTS.")
    print("  use with: --msts_output_dir results/msts --msts_language english")
    return True


GETTERS = {
    "mssbench": get_mssbench,
    "mossbench": get_mossbench,
    "mm-safetybench": get_mm_safetybench,
    "figstep": get_figstep,
    "hades": get_hades,
    "msts": get_msts,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS + ["all"])
    ap.add_argument("--out", default="data", type=Path,
                    help="Directory to download into")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    targets = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    failures = []
    for name in targets:
        print()
        try:
            if not GETTERS[name](args.out):
                failures.append(name)
        except Exception as exc:  # keep going; report at the end
            print(f"  ! {name} failed: {exc}")
            failures.append(name)

    print()
    if failures:
        print(f"Incomplete: {', '.join(failures)}. See the notes above.")
        return 1
    print("Done. See docs/REPRODUCE.md for the matching run commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
