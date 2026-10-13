#!/usr/bin/env python3
"""Side-effect-free checks and task discovery for the public workflow."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def tasks(dataset_root: Path) -> list[str]:
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    names = []
    for path in sorted(dataset_root.iterdir()):
        target = path / "target_images"
        if not path.is_dir() or not target.is_dir():
            continue
        if any(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in target.iterdir()):
            names.append(path.name)
    return names


def tasks_root_has_children(dataset_root: Path) -> bool:
    if not dataset_root.is_dir():
        return False
    return any(
        child.is_dir() and (child / "target_images").is_dir()
        for child in dataset_root.iterdir()
    )


def resolve_dataset_root(dataset_root: Path) -> Path:
    """Accept either a style directory or a dataset container with styles/."""
    if not tasks_root_has_children(dataset_root) and tasks_root_has_children(dataset_root / "styles"):
        return dataset_root / "styles"
    return dataset_root


def require_import(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"Python package is unavailable: {name}")


def checkpoint_exists(root: Path, method: str, model_size: str, task: str, step: str) -> bool:
    prefixes = {
        "vanilla_sft": "vanilla_sft_lora_direct_distilled_4steptrain",
        "pso": "pso_db_lora_direct_distilled_4steptrain",
        "dopsd": "dopsd_editcontext_ema0.9999_onpolicy_4steptrain",
    }
    adapters = {"vanilla_sft": "sft", "pso": "pso", "dopsd": "student"}
    experiment = f"{prefixes[method]}_{model_size}_{task}_bsz1_lora_lr1e-5"
    for candidate in (root / f"{method}_{model_size}", root, root / model_size):
        adapter = candidate / experiment / "checkpoints" / f"lora_gen_step_{step}" / adapters[method]
        if (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file():
            return True
    return False


def preflight(dataset_root: Path, stage: str) -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python >=3.10 is required; activate your prepared environment before running this script.")
    available = tasks(dataset_root)
    requested = [] if stage == "evaluation" else [
        item for item in os.environ.get("TASKS", "").replace(",", " ").split() if item
    ]
    missing_tasks = sorted(set(requested) - set(available))
    if missing_tasks:
        raise SystemExit("Unknown style tasks: " + ", ".join(missing_tasks))
    selected = [item for item in available if not requested or item in requested]
    if not selected:
        raise SystemExit(f"No style tasks found under {dataset_root}")
    for relative in ("configs/default.json", "configs/content_prompts.json"):
        if not (ROOT / relative).is_file():
            raise SystemExit(f"Required file is missing: {ROOT / relative}")
    packages = ("torch", "transformers", "PIL") if stage == "evaluation" else (
        "torch", "diffusers", "transformers", "peft", "safetensors", "accelerate", "PIL"
    )
    for package in packages:
        require_import(package)
    if stage in {"training_based", "training_free", "styleforge", "all"}:
        model_size = os.environ.get("MODEL_SIZE", "9b")
        model = Path(os.environ.get("MODEL", f"models/FLUX.2-klein-{model_size.upper()}"))
        if not model.is_absolute():
            model = ROOT / model
        if not model.is_dir():
            raise SystemExit(f"Base model is missing: {model}")
    if stage in {"training_based", "all"}:
        checkpoint_root = ROOT / os.environ.get("CHECKPOINT_ROOT", "training_based_baselines/checkpoints")
        step = os.environ.get("CHECKPOINT_STEP", "1001")
        missing = [
            f"{method}/{task}"
            for method in ("vanilla_sft", "pso", "dopsd")
            for task in selected
            if not checkpoint_exists(checkpoint_root, method, model_size, task, step)
        ]
        if missing:
            raise SystemExit("Missing checkpoints: " + ", ".join(missing))
    if stage in {"training_free", "styleforge", "all"}:
        entry_script = {
            "styleforge": "run_styleforge.sh",
            "training_free": "run_training_free_baselines.sh",
        }.get(stage, "run_styleforge.sh and run_training_free_baselines.sh")
        for name in ("VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY"):
            value = os.environ.get(name, "").strip()
            if not value or value in {"your-model", "your-key"} or "your-openai-compatible-endpoint" in value:
                raise SystemExit(f"Configure {name} at the top of {entry_script}")
    print(f"preflight ok: stage={stage}, tasks={len(selected)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("tasks", "preflight"))
    parser.add_argument("stage", nargs="?", default="all")
    parser.add_argument("--dataset-root", default=None)
    args = parser.parse_args()
    dataset = Path(args.dataset_root or os.environ.get("DATASET_ROOT", "data/styles"))
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    dataset = resolve_dataset_root(dataset)
    if args.command == "tasks":
        print("\n".join(tasks(dataset)))
    else:
        preflight(dataset, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
