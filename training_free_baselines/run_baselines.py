#!/usr/bin/env python3
"""Run the paper's training-free DSE and RCG baselines on FLUX.2-Klein.

The implementations below are copied from the original repository's
``scripts/run_baselines.py``:

1. ``dse`` (Direct Style Extraction) asks a vision-language model for a natural-
   language summary of the reference style, then combines that text with the
   target content prompt for text-to-image generation. This is the original
   ``vlm_naive`` path.
2. ``rcg`` (Reference-Guided Generation) passes the reference images through
   FLUX.2-Klein's native multi-reference ``image=`` input and asks the model to
   render the target content in their shared style. This is the original
   ``native_reference`` path.

Both methods use the same content prompts, seeds, resolution, step count, and
base checkpoint. Outputs are written in a common JSON schema for evaluation.

Example:

    python \
      training_free_baselines/run_baselines.py \
      --tasks megastyle_anime \
      --content-prompt "a red fox sitting in a forest clearing" \
      --methods dse rcg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styleforge.config import StyleForgeConfig  # noqa: E402
from styleforge.dataset import StyleTask, discover_tasks  # noqa: E402
from styleforge.paths import repository_path  # noqa: E402
from styleforge.vlm_client import (  # noqa: E402
    VisionLanguageModelClient,
    _content_for_prompt,
    _extract_answer,
)


METHOD_RCG = "rcg"
METHOD_DSE = "dse"
METHODS = (METHOD_DSE, METHOD_RCG)
DEFAULT_METHODS = (METHOD_DSE, METHOD_RCG)

DEFAULT_MODEL = "models/FLUX.2-klein-9B"
DEFAULT_CONFIG = ROOT / "configs" / "default.json"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "training_free_baselines"
DEFAULT_CONTENT_PROMPT = "a red fox sitting in a forest clearing"

DSE_VLM_INSTRUCTION = (
    "These images are all in the same art style. Write a prompt I can give to "
    "an AI image generator so it draws new pictures in this same style."
)
RCG_INSTRUCTION = (
    "Use the attached reference images as visual style references. Draw the requested "
    "new content in their shared art style while following the content request: {content}"
)

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(ROOT / "data" / "styles"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--max-ref-images", type=int, default=4)
    parser.add_argument("--content-prompt", action="append", default=[])
    parser.add_argument("--content-prompts-file", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--vlm-config", default=str(DEFAULT_CONFIG), help="JSON inference settings; API connection settings come from run_training_free_baselines.sh.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _strip_markdown_scaffolding(answer: str) -> str:
    fences = _FENCE_RE.findall(answer)
    if fences:
        return fences[0].strip()
    return answer.strip()


def _resolve_tasks(dataset_root: Path, requested: Sequence[str], max_refs: int) -> List[StyleTask]:
    if max_refs <= 0:
        raise ValueError("--max-ref-images must be positive")
    tasks = discover_tasks(dataset_root)
    if requested:
        by_id = {task.task_id: task for task in tasks}
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown tasks: {', '.join(missing)}")
        tasks = [by_id[task_id] for task_id in requested]
    return [
        StyleTask(task_id=task.task_id, root_dir=task.root_dir, ref_images=task.ref_images[:max_refs])
        for task in tasks
    ]


def _load_content_prompts(explicit: Sequence[str], path_value: str) -> List[str]:
    prompts = [str(prompt).strip() for prompt in explicit if str(prompt).strip()]
    if path_value:
        path = Path(path_value).expanduser()
        if path.suffix.lower() == ".jsonl":
            data = None
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    prompts.append(_prompt_from_item(json.loads(line)))
        else:
            data = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
        if isinstance(data, list):
            prompts.extend(_prompt_from_item(item) for item in data)
        elif isinstance(data, Mapping):
            value = data.get("prompts") or data.get("content_prompts") or data.get("scenes")
            if isinstance(value, list):
                prompts.extend(_prompt_from_item(item) for item in value)
            else:
                prompts.extend(_prompt_from_item(item) for item in data.values())
        elif data is not None:
            raise ValueError(f"Unsupported prompt JSON shape: {path}")
        elif path.suffix.lower() != ".jsonl":
            prompts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    prompts = [prompt for prompt in prompts if prompt]
    return prompts or [DEFAULT_CONTENT_PROMPT]


def _prompt_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        for key in ("prompt", "content_prompt", "text", "description"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
    return ""


def _torch_dtype(torch: Any, name: str, device: str) -> Any:
    if name == "auto":
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]
    return torch.float32 if not str(device).startswith("cuda") and dtype != torch.float32 else dtype


def _load_pipeline(args: argparse.Namespace) -> Tuple[Any, Any, Any, Any]:
    import torch
    from diffusers import Flux2KleinPipeline

    dtype = _torch_dtype(torch, args.dtype, args.device)
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)
    return pipe, torch, torch.device(args.device), dtype


def _load_reference_pil(paths: Sequence[Path]) -> List[Any]:
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _vlm_cache_key(task: StyleTask) -> str:
    digest = hashlib.sha256()
    digest.update(DSE_VLM_INSTRUCTION.encode("utf-8"))
    for path in task.ref_images:
        stat = path.stat()
        digest.update(f"{repository_path(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def _dse_style_summary(
    client: VisionLanguageModelClient,
    task: StyleTask,
    cache_dir: Path,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_vlm_cache_key(task)}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if str(cached.get("style_prompt") or "").strip():
            return cached

    content = _content_for_prompt(
        prompt_text=DSE_VLM_INSTRUCTION,
        images=task.ref_images,
        labels=[f"image {index + 1}" for index in range(len(task.ref_images))],
        image_max_side=client.image_max_side,
        image_quality=client.image_quality,
    )
    payload: Dict[str, Any] = {
        "model": client.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": client.temperature,
        "max_tokens": client.max_tokens,
    }
    response = client._post(payload)
    raw_answer = _extract_answer(response).strip()
    style_prompt = _strip_markdown_scaffolding(raw_answer)
    if not style_prompt:
        raise RuntimeError(f"vision-language model returned an empty style prompt for {task.task_id}")
    result = {
        "task_id": task.task_id,
        "instruction": DSE_VLM_INSTRUCTION,
        "style_prompt": style_prompt,
        "raw_answer": raw_answer,
        "model": client.model,
        "usage": response.get("usage") if isinstance(response, Mapping) else None,
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _generation_prompt(method: str, content_prompt: str, vlm_style_prompt: str = "") -> str:
    if method == METHOD_RCG:
        return RCG_INSTRUCTION.format(content=content_prompt)
    if method == METHOD_DSE:
        return f"{vlm_style_prompt.strip()}\n\nDepict the requested content: {content_prompt}".strip()
    raise ValueError(f"Unsupported method: {method}")


def _output_path(
    output_root: Path,
    task_id: str,
    method: str,
    prompt_index: int,
    candidate_index: int,
    seed: int,
) -> Path:
    directory = output_root / task_id / method / "generated"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"prompt_{prompt_index:04d}_candidate_{candidate_index:02d}_seed_{seed}.png"


def _build_vlm_client(args: argparse.Namespace, output_root: Path) -> VisionLanguageModelClient:
    config_path = Path(args.vlm_config).expanduser()
    config = StyleForgeConfig.from_file(config_path) if config_path.exists() else StyleForgeConfig()
    return VisionLanguageModelClient(config.backend, output_root / "_vlm_cache_raw")


def _record_base(
    args: argparse.Namespace,
    task: StyleTask,
    method: str,
    content_prompt: str,
    generation_prompt: str,
    prompt_index: int,
    candidate_index: int,
    seed: int,
    output_path: Path,
) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "method": method,
        "content_prompt": content_prompt,
        "generation_prompt": generation_prompt,
        "prompt_index": prompt_index,
        "candidate_index": candidate_index,
        "seed": seed,
        "generated_image": repository_path(output_path),
        "ref_images": [repository_path(path) for path in task.ref_images],
        "model": repository_path(Path(args.model)),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
    }


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _resolve_tasks(dataset_root, args.tasks, args.max_ref_images)
    prompts = _load_content_prompts(args.content_prompt, args.content_prompts_file)
    if not tasks:
        raise SystemExit("No style tasks found.")

    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": repository_path(dataset_root),
        "output_root": repository_path(output_root),
        "tasks": [task.task_id for task in tasks],
        "methods": list(args.methods),
        "content_prompts": prompts,
        "model": repository_path(Path(args.model)),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "max_ref_images": args.max_ref_images,
        "dse_instruction": DSE_VLM_INSTRUCTION,
        "rcg_instruction": RCG_INSTRUCTION,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return 0

    diffusers_bundle: Optional[Tuple[Any, Any, Any, Any]] = None
    def ensure_diffusers_pipeline() -> Tuple[Any, Any, Any, Any]:
        nonlocal diffusers_bundle
        if diffusers_bundle is None:
            diffusers_bundle = _load_pipeline(args)
        return diffusers_bundle

    vlm_client = _build_vlm_client(args, output_root) if METHOD_DSE in args.methods else None
    records: List[Dict[str, Any]] = []

    for task in tasks:
        reference_images = _load_reference_pil(task.ref_images)
        vlm_result: Dict[str, Any] = {}
        if vlm_client is not None:
            vlm_result = _dse_style_summary(vlm_client, task, output_root / "_dse_cache")

        for method in args.methods:
            for prompt_index, content_prompt in enumerate(prompts):
                generation_prompt = _generation_prompt(
                    method,
                    content_prompt,
                    str(vlm_result.get("style_prompt") or ""),
                )
                for candidate_index in range(args.candidate_count):
                    seed = args.seed + prompt_index * args.candidate_count + candidate_index
                    output_path = _output_path(
                        output_root,
                        task.task_id,
                        method,
                        prompt_index,
                        candidate_index,
                        seed,
                    )
                    record = _record_base(
                        args,
                        task,
                        method,
                        content_prompt,
                        generation_prompt,
                        prompt_index,
                        candidate_index,
                        seed,
                        output_path,
                    )
                    if method == METHOD_DSE:
                        record["dse"] = vlm_result
                        record["style_prompt"] = str(vlm_result.get("style_prompt") or "")
                    if output_path.exists() and not args.overwrite:
                        record["reused"] = True
                        records.append(record)
                        continue

                    pipe, torch, device, dtype = ensure_diffusers_pipeline()
                    generator_device = device.type if device.type == "cuda" else "cpu"
                    generator = torch.Generator(device=generator_device).manual_seed(seed)
                    kwargs: Dict[str, Any] = {
                        "prompt": generation_prompt,
                        "height": args.height,
                        "width": args.width,
                        "num_inference_steps": args.steps,
                        "guidance_scale": args.guidance,
                        "generator": generator,
                        "max_sequence_length": args.max_sequence_length,
                    }
                    if method == METHOD_RCG:
                        kwargs["image"] = reference_images

                    with torch.inference_mode():
                        result = pipe(**kwargs)
                    image = result.images[0]
                    image.save(output_path)
                    record["reused"] = False
                    metadata_path = output_path.with_suffix(".metadata.json")
                    metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    records.append(record)
                    print(f"[{task.task_id}] {method} prompt={prompt_index} candidate={candidate_index} -> {output_path}")

    summary = {
        "version": "training_free_style_baselines",
        "config": run_config,
        "record_count": len(records),
        "records": records,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
