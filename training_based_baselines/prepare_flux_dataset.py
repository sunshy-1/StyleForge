#!/usr/bin/env python3
"""Convert the bundled style metadata into the JSONL format used by FLUX training.

The original FLUX implementations consume one ``data.jsonl`` per style class.
The StyleForge project stores the same references in ``metadata.json`` so that
the evaluation pipeline can use them directly.  This adapter only translates
that storage format; it does not alter any training objective or image.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/styles")
    parser.add_argument("--output-root", default="data/flux_training")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def style_tasks(dataset_root: Path, requested: list[str]) -> list[Path]:
    task_dirs = {path.name: path for path in dataset_root.iterdir() if path.is_dir()}
    names = sorted(requested) if requested else sorted(task_dirs)
    missing = sorted(set(names) - set(task_dirs))
    if missing:
        raise FileNotFoundError(f"Unknown style tasks: {', '.join(missing)}")
    return [task_dirs[name] for name in names]


def sample_records(task_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    metadata_path = task_dir / "metadata.json"
    metadata = read_json(metadata_path)
    style_prompt = str(metadata.get("style_prompt") or "").strip()
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{metadata_path} does not contain a non-empty samples list")

    records: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"Invalid sample in {metadata_path}")
        relative_image = str(sample.get("file") or "").strip()
        image_path = (task_dir / relative_image).resolve()
        if not relative_image or image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            raise FileNotFoundError(f"Missing sample image: {task_dir / relative_image}")
        content = str(sample.get("content") or "an image").strip()
        style_detail = style_prompt
        for prefix in ("In the style of ", "in the style of "):
            if style_detail.startswith(prefix):
                style_detail = style_detail[len(prefix) :]
                break
        short_prompt = f"{content}, rendered in [V] style."
        medium_prompt = f"The scene shows {content}, rendered in [V] style with {style_detail}."
        detailed_prompt = (
            f"The image shows {content}. The entire scene is rendered in [V] style, "
            f"featuring {style_detail}."
        )
        user_prompt = f"Generate a [V] style image showing {content}."
        record = {
            "local_path_list": [os.path.relpath(image_path, start=output_dir).replace(os.sep, "/")],
            "h*w": f"{int(sample.get('height', 0))}*{int(sample.get('width', 0))}",
            "short_en": short_prompt,
            "medium_en": medium_prompt,
            "detailed_en": detailed_prompt,
            "user_prompt_en": user_prompt,
            "short_zh": short_prompt,
            "medium_zh": medium_prompt,
            "detailed_zh": detailed_prompt,
            "user_prompt_zh": user_prompt,
        }
        if record["h*w"] == "0*0":
            from PIL import Image

            with Image.open(image_path) as image:
                record["h*w"] = f"{image.height}*{image.width}"
        records.append(record)
    return records


def write_task(task_dir: Path, output_root: Path, overwrite: bool) -> Path:
    output_dir = output_root / task_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.jsonl"
    if output_path.exists() and not overwrite:
        return output_path
    records = sample_records(task_dir, output_dir)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [write_task(task, output_root, args.overwrite) for task in style_tasks(dataset_root, args.tasks)]
    print(f"Prepared {len(paths)} FLUX training JSONL files under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
