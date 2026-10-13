#!/usr/bin/env python3
"""Generate StyleForge-compatible images from trained FLUX.2-Klein LoRAs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METHOD_CONFIGS = {
    "dopsd": {
        "experiment_prefix": "dopsd_editcontext_ema0.9999_onpolicy_4steptrain_{model_size}_",
        "adapter_subdir": "student",
    },
    "pso": {
        "experiment_prefix": "pso_db_lora_direct_distilled_4steptrain_{model_size}_",
        "adapter_subdir": "pso",
    },
    "vanilla_sft": {
        "experiment_prefix": "vanilla_sft_lora_direct_distilled_4steptrain_{model_size}_",
        "adapter_subdir": "sft",
    },
}
EXPERIMENT_SUFFIX = "_bsz1_lora_lr1e-5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_CONFIGS), required=True)
    parser.add_argument(
        "--checkpoint-root",
        default="training_based_baselines/checkpoints",
    )
    parser.add_argument("--checkpoint-step", type=int, default=1001)
    parser.add_argument("--model-size", choices=["4b", "9b"], default="4b")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--dataset-root",
        default="data/styles",
    )
    parser.add_argument(
        "--content-prompts-file",
        default="configs/content_prompts.json",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--prompt-indices", nargs="*", type=int, default=[])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=880000)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompt-template",
        default="{content_prompt}, rendered in [V] style.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def repository_path(path: Path) -> str:
    """Store runtime paths in summaries relative to the open-source repository."""
    return os.path.relpath(path, start=ROOT).replace(os.sep, "/")


def discover_tasks(dataset_root: Path, requested: list[str]) -> list[str]:
    available = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Unknown tasks: {', '.join(missing)}")
        return sorted(dict.fromkeys(requested))
    return available


def load_prompts(path: Path, prompt_indices: list[int]) -> list[tuple[int, str]]:
    data = read_json(path)
    prompts = data.get("content_prompts")
    if not isinstance(prompts, list) or not all(isinstance(item, str) and item.strip() for item in prompts):
        raise ValueError(f"Invalid content_prompts in {path}")
    normalized = [item.strip() for item in prompts]
    if not prompt_indices:
        return list(enumerate(normalized))
    invalid = sorted(set(prompt_indices) - set(range(len(normalized))))
    if invalid:
        raise ValueError(f"Invalid prompt indices: {invalid}")
    return [(index, normalized[index]) for index in sorted(dict.fromkeys(prompt_indices))]


def reference_images(dataset_root: Path, task_id: str) -> list[Path]:
    target_dir = dataset_root / task_id / "target_images"
    images = sorted(
        path.resolve()
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No reference images found: {target_dir}")
    return images


def checkpoint_for(checkpoint_root: Path, method: str, model_size: str, task_id: str, step: int) -> Path:
    config = METHOD_CONFIGS[method]
    experiment_prefix = config["experiment_prefix"].format(model_size=model_size)
    experiment_name = f"{experiment_prefix}{task_id}{EXPERIMENT_SUFFIX}"
    method_root = {"vanilla_sft": "vanilla_sft", "pso": "pso", "dopsd": "dopsd"}[method]
    candidate_roots = [
        checkpoint_root,
        checkpoint_root / f"{method_root}_{model_size}",
        checkpoint_root / f"{model_size}",
    ]
    attempted = []
    for root in candidate_roots:
        adapter = root / experiment_name / "checkpoints" / f"lora_gen_step_{step}" / config["adapter_subdir"]
        attempted.append(adapter)
        required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
        if all(path.is_file() for path in required):
            return adapter.resolve()
    raise FileNotFoundError(
        f"Missing checkpoint files for {method}/{task_id}; searched: {attempted}"
    )


def safe_adapter_name(method: str, task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_{task_id}")


def load_pipeline(args: argparse.Namespace):
    import torch
    from diffusers import Flux2KleinPipeline

    dtype = getattr(torch, args.dtype)
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, torch, dtype


def attach_adapter(pipe: Any, adapter_path: Path, adapter_name: str, previous_name: str | None) -> str:
    from peft import PeftModel

    if not isinstance(pipe.transformer, PeftModel):
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            str(adapter_path),
            adapter_name=adapter_name,
            is_trainable=False,
        )
    else:
        pipe.transformer.load_adapter(
            str(adapter_path),
            adapter_name=adapter_name,
            is_trainable=False,
        )
        pipe.transformer.set_adapter(adapter_name)
        if previous_name and previous_name != adapter_name:
            pipe.transformer.delete_adapter(previous_name)
    pipe.transformer.set_adapter(adapter_name)
    return adapter_name


def expected_record(
    args: argparse.Namespace,
    task_id: str,
    prompt_index: int,
    content_prompt: str,
    generated_image: Path,
    refs: list[Path],
    adapter_path: Path,
    style_prompt: str,
) -> dict[str, Any]:
    seed = args.seed + prompt_index
    generation_prompt = args.prompt_template.format(content_prompt=content_prompt)
    return {
        "task_id": task_id,
        "method": args.method,
        "content_prompt": content_prompt,
        "generation_prompt": generation_prompt,
        "prompt_index": prompt_index,
        "candidate_index": 0,
        "seed": seed,
        "generated_image": repository_path(generated_image.resolve()),
        "ref_images": [repository_path(path) for path in refs],
        "style_prompt": style_prompt,
        "checkpoint": repository_path(adapter_path),
        "checkpoint_step": args.checkpoint_step,
        "adapter_subdir": METHOD_CONFIGS[args.method]["adapter_subdir"],
        "base_model": repository_path(Path(args.model).expanduser().resolve()),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
        "max_sequence_length": args.max_sequence_length,
        "prompt_template": args.prompt_template,
    }


def reusable_record(image_path: Path, metadata_path: Path, expected: dict[str, Any]) -> bool:
    if not image_path.is_file() or not metadata_path.is_file():
        return False
    try:
        existing = read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return all(existing.get(key) == value for key, value in expected.items())


def main() -> int:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    prompts_path = Path(args.content_prompts_file).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not args.model:
        args.model = f"models/FLUX.2-klein-{args.model_size.upper()}"
    tasks = discover_tasks(dataset_root, args.tasks)
    prompts = load_prompts(prompts_path, args.prompt_indices)

    checkpoints = {}
    if not args.dry_run:
        checkpoints = {
            task_id: checkpoint_for(checkpoint_root, args.method, args.model_size, task_id, args.checkpoint_step)
            for task_id in tasks
        }
    refs_by_task = {task_id: reference_images(dataset_root, task_id) for task_id in tasks}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": args.method,
                    "tasks": len(tasks),
                    "prompts": len(prompts),
                    "expected_records": len(tasks) * len(prompts),
                    "checkpoint_step": args.checkpoint_step,
                    "adapter_subdir": METHOD_CONFIGS[args.method]["adapter_subdir"],
                    "output_root": str(output_root),
                },
                indent=2,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    pipe, torch, dtype = load_pipeline(args)
    current_adapter: str | None = None
    records: list[dict[str, Any]] = []

    for task_id in tasks:
        adapter_name = safe_adapter_name(args.method, task_id)
        current_adapter = attach_adapter(pipe, checkpoints[task_id], adapter_name, current_adapter)
        generated_dir = output_root / task_id / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        refs = refs_by_task[task_id]
        style_metadata = read_json(dataset_root / task_id / "metadata.json")
        style_prompt = str(style_metadata.get("style_prompt") or "").strip()
        if not style_prompt:
            raise ValueError(f"Canonical style prompt is empty for {task_id}")

        for prompt_index, content_prompt in prompts:
            seed = args.seed + prompt_index
            image_path = generated_dir / f"prompt_{prompt_index:02d}_seed_{seed}.png"
            metadata_path = image_path.with_suffix(".metadata.json")
            record = expected_record(
                args,
                task_id,
                prompt_index,
                content_prompt,
                image_path,
                refs,
                checkpoints[task_id],
                style_prompt,
            )
            if not args.overwrite and reusable_record(image_path, metadata_path, record):
                record["reused"] = True
                records.append(record)
                print(f"[{task_id}] reused prompt={prompt_index} -> {image_path}", flush=True)
                continue

            generator_device = "cuda" if str(args.device).startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            with torch.inference_mode():
                result = pipe(
                    prompt=record["generation_prompt"],
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                    max_sequence_length=args.max_sequence_length,
                )
            result.images[0].save(image_path)
            record["reused"] = False
            record["dtype"] = str(dtype)
            write_json(metadata_path, record)
            records.append(record)
            print(f"[{task_id}] generated prompt={prompt_index} -> {image_path}", flush=True)

        write_json(
            output_root / "summary.partial.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "method": args.method,
                "records": records,
            },
        )

    expected_count = len(tasks) * len(prompts)
    if len(records) != expected_count:
        raise RuntimeError(f"Expected {expected_count} records, found {len(records)}")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "method": args.method,
            "model_size": args.model_size,
            "checkpoint_root": repository_path(checkpoint_root),
            "checkpoint_step": args.checkpoint_step,
            "adapter_subdir": METHOD_CONFIGS[args.method]["adapter_subdir"],
            "model": repository_path(Path(args.model).expanduser().resolve()),
            "dataset_root": repository_path(dataset_root),
            "content_prompts_file": repository_path(prompts_path),
            "tasks": tasks,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "max_sequence_length": args.max_sequence_length,
            "dtype": args.dtype,
            "prompt_template": args.prompt_template,
        },
        "record_count": len(records),
        "records": records,
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.partial.json").unlink(missing_ok=True)
    print(f"Wrote {output_root / 'summary.json'} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
