#!/usr/bin/env python3
"""Run only the six metrics reported in the StyleForge paper.

The metric implementations are copied from the original open-source
evaluation scripts. This thin entry point keeps the public release focused on
Score, Pairwise Win Rate, CLIP-Text, CLIP-Style, VLM-F, and VLM-S.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = Path(__file__).resolve().parent


def _paths(values: Sequence[str]) -> List[str]:
    return [str(Path(value).expanduser()) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="metric", required=True)

    score = subparsers.add_parser("score", help="Qwen VLM Score metric.")
    score.add_argument("--summary", nargs="+", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--endpoint", default=None)
    score.add_argument("--model", default=None)
    score.add_argument("--prompt-mode", default=None)
    score.add_argument("--max-refs", type=int, default=None)
    score.add_argument("--limit", type=int, default=None)
    score.add_argument("--concurrency", type=int, default=None)
    score.add_argument("--max-new-tokens", type=int, default=None)
    score.add_argument("--timeout", type=float, default=None)
    score.add_argument("--retries", type=int, default=None)
    score.add_argument("--project-root", default=None)
    score.add_argument("--quiet", action="store_true", default=False)

    pairwise = subparsers.add_parser("pairwise", help="Qwen VLM Pairwise Win Rate.")
    pairwise.add_argument("--run-root", nargs="+", required=True)
    pairwise.add_argument("--methods", nargs="+", default=None)
    pairwise.add_argument("--dataset-root", default=None)
    pairwise.add_argument("--model-path", default=None)
    pairwise.add_argument("--judge-label", default=None)
    pairwise.add_argument("--output-name", default=None)
    pairwise.add_argument("--backend", choices=("api",), default="api")
    pairwise.add_argument("--endpoint", default=None)
    pairwise.add_argument("--model", default=None)
    pairwise.add_argument("--concurrency", type=int, default=None)
    pairwise.add_argument("--timeout", type=float, default=None)
    pairwise.add_argument("--retries", type=int, default=None)
    pairwise.add_argument("--image-max-side", type=int, default=None)
    pairwise.add_argument("--board-side", type=int, default=None)
    pairwise.add_argument("--num-shards", type=int, default=None)
    pairwise.add_argument("--shard-index", type=int, default=None)
    pairwise.add_argument("--limit", type=int, default=None)
    pairwise.add_argument("--prepare-only", action="store_true", default=False)
    pairwise.add_argument("--summarize-only", action="store_true", default=False)
    pairwise.add_argument("--quiet", action="store_true", default=False)

    freestyle = subparsers.add_parser("freestyle", help="FreeStyle VLM-F and VLM-S.")
    freestyle.add_argument("--summary", nargs="+", required=True)
    freestyle.add_argument("--output-dir", required=True)
    freestyle.add_argument("--endpoint", default=None)
    freestyle.add_argument("--model", default=None)
    freestyle.add_argument("--metrics", nargs="+", default=None)
    freestyle.add_argument("--concurrency", type=int, default=None)
    freestyle.add_argument("--timeout", type=float, default=None)
    freestyle.add_argument("--retries", type=int, default=None)
    freestyle.add_argument("--limit", type=int, default=None)
    freestyle.add_argument("--project-root", default=None)
    freestyle.add_argument("--retry-failures", action="store_true", default=False)
    freestyle.add_argument("--quiet", action="store_true", default=False)

    clip = subparsers.add_parser("clip", help="Paper CLIP-Text and CLIP-Style.")
    clip.add_argument("--summary", nargs="+", required=True)
    clip.add_argument("--output-dir", required=True)
    clip.add_argument("--project-root", default=None)
    clip.add_argument("--model-path", default=None)
    clip.add_argument("--canonical-style-metadata", default=None)
    clip.add_argument("--device", default=None)
    clip.add_argument("--limit", type=int, default=None)
    clip.add_argument("--quiet", action="store_true", default=False)
    args, rest = parser.parse_known_args()
    forwarded = []
    for name, value in vars(args).items():
        if name in {"metric", "summary", "output_dir", "run_root"} or value is None or value is False:
            continue
        option = "--" + name.replace("_", "-")
        if isinstance(value, list):
            forwarded.extend([option, *map(str, value)])
        elif value is True:
            forwarded.append(option)
        else:
            forwarded.extend([option, str(value)])
    if "--" in rest:
        rest = rest[rest.index("--") + 1 :]
    rest = [*forwarded, *rest]

    if args.metric == "score":
        command = [sys.executable, str(EVALUATION / "qwen_style_judge_api.py"), "--summary", *_paths(args.summary), "--output-dir", args.output_dir, *rest]
    elif args.metric == "pairwise":
        command = [sys.executable, str(EVALUATION / "qwen_pairwise_win_rate.py"), "--run-root", *_paths(args.run_root), *rest]
    elif args.metric == "freestyle":
        command = [sys.executable, str(EVALUATION / "freestyle_vlm_metrics_api.py"), "--summary", *_paths(args.summary), "--output-dir", args.output_dir, *rest]
    else:
        command = [sys.executable, str(EVALUATION / "paper_clip_metrics.py"), "--summary", *_paths(args.summary), "--output-dir", args.output_dir, *rest]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
