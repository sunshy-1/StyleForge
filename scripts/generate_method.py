#!/usr/bin/env python3
"""Generate images from compiled style-schema prompts.

The script uses one generic construction path, runtime-selected task ids, and a
neutral content prompt or frozen prompt set shared by every selected task. It
does not call a VLM or read generated failure notes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styleforge.config import StyleForgeConfig
from styleforge.dataset import StyleTask, discover_tasks
from styleforge.style_schema import literal_reference_hits
from styleforge.flux2_generator import Flux2Generator
from styleforge.prompts import compose_generation_prompt
from styleforge.paths import repository_path, resolve_repository_path


CONFIG_PATH = ROOT / "configs" / "default.json"
STYLE_SCHEMA_RUN = ROOT / "outputs" / "style_schema"
OUTPUT_ROOT = ROOT / "outputs" / "styleforge"
DEFAULT_CONTENT_PROMPT = "a full-body person raising a glowing lantern in an open clearing at sunrise, wide view"
DEFAULT_SCENE_KEY = "distilled_reference_only_structure_open_clearing"
DEFAULT_SEED = 880000
DEFAULT_EVAL_PROMPTS = [
    "a full-body person holding a glowing lantern on a simple open walkway, wide view",
    "a quiet street corner with a bicycle leaning near a low wall, wide view",
    "a small cafe table with flowers and a teapot, medium-wide view",
    "a market stall with stacked fruit and hanging fabric, wide view",
]

_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_task_ids(value: str) -> List[str]:
    task_ids = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not task_ids:
        raise ValueError("--tasks must select at least one task id")
    return task_ids


def _load_content_prompts(path: Optional[Path], fallback: str) -> List[str]:
    if path is None:
        prompts = [fallback]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            value = data.get("content_prompts")
        else:
            value = data
        if not isinstance(value, list):
            raise ValueError("--content-prompts-file must be a JSON list or an object with content_prompts")
        prompts = [str(item).strip() for item in value if str(item).strip()]
    if not prompts:
        raise ValueError("content prompt set must contain at least one prompt")
    return prompts


def _selected_tasks(dataset_root: Path, task_ids: Sequence[str]) -> List[StyleTask]:
    tasks_by_id = {task.task_id: task for task in discover_tasks(dataset_root)}
    missing = [task_id for task_id in task_ids if task_id not in tasks_by_id]
    if missing:
        raise ValueError("Task ids not found in dataset: " + ", ".join(missing))
    return [tasks_by_id[task_id] for task_id in task_ids]


def _style_schema_path(task_id: str, style_schema_run: Path) -> Path:
    path = style_schema_run / task_id / "style_schema.json"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"missing compiled style schema record: {path}"
    )


def _style_schema_record(task_id: str, style_schema_run: Path) -> Dict[str, Any]:
    return _load_json(_style_schema_path(task_id, style_schema_run))


def _compiled_style_prompt(record: Mapping[str, Any]) -> str:
    compiled = record.get("compiled") or {}
    prompt = str(compiled.get("generation_style_prompt") or "").strip()
    if not prompt:
        raise ValueError("compiled style-schema generation_style_prompt is empty")
    return prompt


def _with_prompt_affixes(prompt: str, prefix: str = "", suffix: str = "") -> str:
    parts = [str(prefix or "").strip(), str(prompt or "").strip(), str(suffix or "").strip()]
    return " ".join(part for part in parts if part).strip()


def _literal_containers(record: Mapping[str, Any]) -> List[str]:
    schema = record.get("schema") or {}
    value = schema.get("literal_reference_containers") if isinstance(schema, Mapping) else []
    return [str(item) for item in value] if isinstance(value, list) else []


def _audit_inputs(
    records: Sequence[Mapping[str, Any]],
    content_prompts: Sequence[str],
    negative_prompt: str = "",
) -> Dict[str, Any]:
    errors: List[str] = []
    for prompt_index, content_prompt in enumerate(content_prompts):
        lowered_content = str(content_prompt).lower()
        for forbidden in ("window", "frame", "framing", "border", "portal", "stage", "room", "arched", "circular"):
            if forbidden in lowered_content:
                errors.append(f"content prompt {prompt_index} contains container literal: {forbidden}")
    per_task = {}
    for record in records:
        task_id = str(record["task_id"])
        literal_containers = record.get("literal_reference_containers") or []
        schema_hits = literal_reference_hits(str(record["generation_style_prompt"]), literal_containers)
        per_task[task_id] = {
            "schema_literal_hits": schema_hits,
        }
        if schema_hits:
            errors.append(f"{task_id}: reference schema prompt contains literal containers: {schema_hits}")
    candidate_groups: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        group_key = f"{record.get('task_id')}::prompt_{int(record.get('prompt_index') or 0)}"
        candidate_groups.setdefault(group_key, []).append(record)
    candidate_count_values = {
        int(record.get("candidate_count") or 1)
        for record in records
    }
    candidate_seed_stride_values = {
        int(record.get("candidate_seed_stride") or 0)
        for record in records
        if record.get("candidate_seed_stride") is not None
    }
    for group_key, group_records in candidate_groups.items():
        style_prompts = {str(record.get("generation_style_prompt") or "") for record in group_records}
        content_prompts_in_group = {str(record.get("content_prompt") or "") for record in group_records}
        scene_keys = {str(record.get("scene_key") or "") for record in group_records}
        candidate_indexes = [int(record.get("candidate_index") or 0) for record in group_records]
        group_candidate_counts = {int(record.get("candidate_count") or len(group_records)) for record in group_records}
        group_seed_strides = {
            int(record.get("candidate_seed_stride") or 0)
            for record in group_records
            if record.get("candidate_seed_stride") is not None
        }
        if len(style_prompts) > 1:
            errors.append(f"{group_key}: style prompt changed across seed candidates")
        if len(content_prompts_in_group) > 1:
            errors.append(f"{group_key}: content prompt changed across seed candidates")
        if len(scene_keys) > 1:
            errors.append(f"{group_key}: scene key changed across seed candidates")
        if len(candidate_indexes) != len(set(candidate_indexes)):
            errors.append(f"{group_key}: duplicate candidate_index values")
        if len(group_candidate_counts) > 1:
            errors.append(f"{group_key}: inconsistent candidate_count metadata")
        elif group_candidate_counts and next(iter(group_candidate_counts)) != len(group_records):
            errors.append(f"{group_key}: candidate_count metadata does not match group size")
        if len(group_seed_strides) > 1:
            errors.append(f"{group_key}: inconsistent candidate_seed_stride metadata")
        elif group_seed_strides:
            stride = next(iter(group_seed_strides))
            for record in group_records:
                prompt_index = int(record.get("prompt_index") or 0)
                candidate_index = int(record.get("candidate_index") or 0)
                base_seed = int(record.get("base_seed") or 0)
                expected_seed = base_seed + prompt_index + candidate_index * stride
                if stride > 0 and int(record.get("seed") or 0) != expected_seed:
                    errors.append(f"{group_key}: seed does not match generic candidate stride")
                    break
    negative_prompt = str(negative_prompt or "").strip()
    candidate_count = max(candidate_count_values) if candidate_count_values else 1
    candidate_seed_stride = max(candidate_seed_stride_values) if candidate_seed_stride_values else 0
    rules = [
        "runtime-selected task ids only",
        "same neutral content prompt set for every task",
        "same base seed for every task, with deterministic prompt-index offsets",
        "compiled prompt must not contain literal_reference_containers",
        "when a prompt set is provided, every task uses the same frozen content prompt set",
        "optional negative prompt is a CFG runtime control only and is not merged into positive prompt",
        "candidate variants keep style/content prompts fixed within each task x scene group; only seed/candidate metadata differs",
        "generation uses the compiled style schema only",
        "no VLM calls, no generated failures, no contact-sheet feedback used as prompt input",
    ]
    return {
        "ok": not errors,
        "errors": errors,
        "rules": rules,
        "content_prompts": list(content_prompts),
        "content_prompt_set": {
            "count": len(content_prompts),
            "sha256": _hash_text(json.dumps(list(content_prompts), ensure_ascii=False, sort_keys=True)),
        },
        "negative_prompt": {
            "enabled": bool(negative_prompt),
            "chars": len(negative_prompt),
            "sha256": _hash_text(negative_prompt) if negative_prompt else "",
        },
        "candidate_policy": {
            "candidate_count": candidate_count,
            "candidate_seed_stride": candidate_seed_stride,
            "candidate_group_count": len(candidate_groups),
            "prompt_invariance_ok": not any("prompt changed across seed candidates" in str(error) for error in errors),
        },
        "per_task": per_task,
    }


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(resolve_repository_path(path), base)


def _paste(sheet: Image.Image, image_path: Path, x: int, y: int, size: int) -> None:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((size, size), _RESAMPLE_LANCZOS)
    sheet.paste(image, (x + (size - image.width) // 2, y + (size - image.height) // 2))


def _write_contact_sheet(records: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    prompt_candidates = sorted({
        (int(record.get("prompt_index", 0)), int(record.get("candidate_index", 0)))
        for record in records
    })
    multi_candidate = any(candidate_index != 0 for _prompt_index, candidate_index in prompt_candidates)
    rows = ["reference"] + [
        f"prompt_{prompt_index}_candidate_{candidate_index}" if multi_candidate else f"prompt_{prompt_index}"
        for prompt_index, candidate_index in prompt_candidates
    ]
    task_ids = []
    for record in records:
        task_id = str(record["task_id"])
        if task_id not in task_ids:
            task_ids.append(task_id)
    first_by_task = {}
    generated_by_task_prompt = {}
    for record in records:
        task_id = str(record["task_id"])
        first_by_task.setdefault(task_id, record)
        generated_by_task_prompt[
            (
                task_id,
                int(record.get("prompt_index", 0)),
                int(record.get("candidate_index", 0)),
            )
        ] = record
    cell = 184
    left = 225
    title_h = 138
    row_h = cell + 44
    sheet = Image.new("RGB", (left + len(task_ids) * cell, title_h + len(rows) * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "Style Schema Generation", fill="black", font=_font(16))
    draw.text((10, 34), "compiled style schema prompt only", fill=(80, 80, 80), font=_font(10))
    content_prompt = str(records[0].get("content_prompt") or "") if records else DEFAULT_CONTENT_PROMPT
    draw.text((10, 50), content_prompt[:120], fill=(80, 80, 80), font=_font(10))
    for col, task_id in enumerate(task_ids):
        record = first_by_task[task_id]
        x = left + col * cell
        draw.text((x + 2, 78), str(record["task_id"])[:27], fill="black", font=_font(10))
        draw.text((x + 2, 94), f"seed={record['seed']}", fill=(80, 80, 80), font=_font(9))
        draw.text((x + 2, 110), str(record.get("scene_key") or "")[:27], fill=(80, 80, 80), font=_font(9))
    for row_idx, row_name in enumerate(rows):
        y = title_h + row_idx * row_h
        draw.text((10, y + 8), row_name, fill="black", font=_font(11))
        for col, task_id in enumerate(task_ids):
            x = left + col * cell
            if row_name == "reference":
                record = first_by_task[task_id]
                paths = record.get("ref_images") or []
                if paths:
                    _paste(sheet, resolve_repository_path(Path(paths[0])), x, y + 2, cell - 10)
            else:
                parts = row_name.split("_")
                prompt_index = int(parts[1])
                candidate_index = int(parts[3]) if len(parts) >= 4 else 0
                record = generated_by_task_prompt.get((task_id, prompt_index, candidate_index), {})
                if record.get("generated_image"):
                    label = str(record.get("content_prompt", ""))[:20]
                    if multi_candidate:
                        label = f"cand={candidate_index} " + label
                    draw.text((x + 2, y + 2), label, fill=(80, 80, 80), font=_font(8))
                    _paste(sheet, resolve_repository_path(Path(str(record["generated_image"]))), x, y + 12, cell - 20)
    sheet.save(output_root / "contact_sheet.jpg", quality=92)


def _write_html(records: Sequence[Mapping[str, Any]], audit: Mapping[str, Any], output_root: Path) -> None:
    sections = []
    for record in records:
        task_id = str(record["task_id"])
        refs = "".join(
            f'<img src="{html.escape(_rel(Path(path), output_root))}" loading="lazy" alt="ref">'
            for path in record.get("ref_images", [])[:4]
        )
        generated_path = Path(str(record["generated_image"])) if record.get("generated_image") else None
        candidate_index = int(record.get("candidate_index", 0))
        seed = int(record.get("seed", 0))
        generated_html = (
            f'<figure><img src="{html.escape(_rel(generated_path, output_root))}" loading="lazy" alt="generated"><figcaption>style schema · candidate {candidate_index} · seed {seed}</figcaption></figure>'
            if generated_path
            else ""
        )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(task_id)}</h2>
              <div class="refs">{refs}</div>
              <div class="grid">{generated_html}</div>
              <details><summary>Compiled style prompt</summary><pre>{html.escape(str(record["generation_style_prompt"]))}</pre></details>
              <details><summary>Full prompts</summary><pre>{html.escape(json.dumps(record.get("full_prompts", {}), ensure_ascii=False, indent=2))}</pre></details>
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Style Schema Generation</title>
<style>
body{{margin:0;background:#f6f7f9;color:#161b22;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:#111827;color:white;padding:20px 28px}} header a{{color:#93c5fd}}
main{{padding:18px 28px 48px}} section{{background:white;border:1px solid #d8dee4;border-radius:8px;padding:14px;margin:0 0 18px}}
.refs,.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0}}
img{{width:100%;aspect-ratio:1/1;object-fit:cover;border:1px solid #d8dee4;border-radius:6px;background:#e5e7eb}}
figure{{margin:0;background:#fbfcfe;border:1px solid #d8dee4;border-radius:8px;padding:9px}} figcaption{{font-size:12px;color:#57606a}}
details{{background:#fbfcfe;border:1px solid #d8dee4;border-radius:8px;padding:9px 11px;margin:10px 0}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
</style>
</head>
<body>
<header>
  <h1>Style Schema Generation</h1>
  <p>Protocol audit: {"PASS" if audit.get("ok") else "FAIL"} · <a href="contact_sheet.jpg">contact sheet</a></p>
  <p>{html.escape(str(audit.get("content_prompt", "")))}</p>
</header>
<main>
<details open><summary>Audit JSON</summary><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details>
{''.join(sections)}
</main>
</body>
</html>
"""
    (output_root / "index.html").write_text(html_text, encoding="utf-8")


def _prepare_records(args: argparse.Namespace, config: StyleForgeConfig, tasks: Sequence[StyleTask], content_prompts: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    candidate_count = max(1, int(getattr(args, "candidate_count", 1) or 1))
    candidate_seed_stride = max(1, int(getattr(args, "candidate_seed_stride", 1000) or 1000))
    for task in tasks:
        style_schema_path = _style_schema_path(task.task_id, args.style_schema_run)
        style_schema_record = _load_json(style_schema_path)
        generation_style_prompt = _compiled_style_prompt(style_schema_record)
        for prompt_index, content_prompt in enumerate(content_prompts):
            candidate_group_key = _hash_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "prompt_index": int(prompt_index),
                        "scene_key": args.scene_key,
                        "content_prompt": str(content_prompt),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )[:16]
            for candidate_index in range(candidate_count):
                records.append(
                    {
                        "task_id": task.task_id,
                        "ref_images": [repository_path(path) for path in task.ref_images[:4]],
                        "generation_style_prompt": generation_style_prompt,
                        "literal_reference_containers": _literal_containers(style_schema_record),
                        "style_schema_record": repository_path(style_schema_path),
                        "seed": int(args.seed) + int(prompt_index) + int(candidate_index) * candidate_seed_stride,
                        "base_seed": int(args.seed),
                        "prompt_index": int(prompt_index),
                        "candidate_index": int(candidate_index),
                        "candidate_count": int(candidate_count),
                        "candidate_seed_stride": int(candidate_seed_stride),
                        "candidate_group_key": candidate_group_key,
                        "scene_key": args.scene_key,
                        "content_prompt": str(content_prompt),
                    }
                )
    return records


def _generate_records(args: argparse.Namespace, config: StyleForgeConfig, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    generator = Flux2Generator(config.backend, args.output_root / "generated", config.prompt_template)
    out: List[Dict[str, Any]] = []
    for record in records:
        task_id = str(record["task_id"])
        new_record = dict(record)
        new_record["full_prompts"] = {}
        style_prompt = str(record["generation_style_prompt"])
        prompt_index = int(record.get("prompt_index", 0))
        params = dict(config.generation.to_dict())
        negative_prompt = str(args.negative_style_prompt or "").strip()
        params.update(
            {
                "n": 1,
                "pipeline_step": "style_schema_generation",
                "variant": "style_schema",
                "task_id": task_id,
                "prompt_index": prompt_index,
                "candidate_index": int(record.get("candidate_index", 0)),
                "candidate_count": int(record.get("candidate_count", 1)),
                "candidate_group_key": str(record.get("candidate_group_key") or ""),
                "scene_key": args.scene_key,
                "style_prompt_sha256": _hash_text(style_prompt),
                "style_schema_record": str(record["style_schema_record"]),
            }
        )
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
            params["negative_prompt_sha256"] = _hash_text(negative_prompt)
        seed = int(record["seed"])
        content_prompt = str(record["content_prompt"])
        candidate_index = int(record.get("candidate_index", 0))
        print(f"[generate] {task_id} prompt_index={prompt_index} candidate_index={candidate_index} style_schema seed={seed}", flush=True)
        started = time.time()
        image = generator.generate(style_prompt, content_prompt, seed, 1, params)[0]
        new_record["generated_image"] = repository_path(Path(image.local_path))
        new_record["generated_latency_s"] = round(time.time() - started, 6)
        composed_prompt = compose_generation_prompt(
            style_prompt,
            content_prompt,
            config.prompt_template,
        )
        new_record["full_prompts"]["style_schema"] = composed_prompt
        if negative_prompt:
            new_record["full_prompts"]["style_schema_negative"] = negative_prompt
        out.append(new_record)
        _write_json(args.output_root / "summary.partial.json", {"records": out})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "styles")
    parser.add_argument(
        "--style-schema-run",
        dest="style_schema_run",
        type=Path,
        default=STYLE_SCHEMA_RUN,
        help="Compiled style-schema directory.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tasks", required=True, help="Comma-separated task ids; runtime selection only.")
    parser.add_argument("--content-prompt", default=DEFAULT_CONTENT_PROMPT)
    parser.add_argument("--content-prompts-file", type=Path, default=None, help="Frozen JSON prompt set shared by every selected task.")
    parser.add_argument("--scene-key", default=DEFAULT_SCENE_KEY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-count", type=int, default=1, help="Number of seed candidates per task x scene.")
    parser.add_argument("--candidate-seed-stride", type=int, default=1000, help="Deterministic seed stride between candidates for the same task x scene.")
    parser.add_argument("--guidance", type=float, default=None, help="Override config.generation.guidance.")
    parser.add_argument("--steps", type=int, default=None, help="Override config.generation.steps.")
    parser.add_argument("--flux2-base-model", default="", help="Override backend.flux2_base_model.")
    parser.add_argument("--device", default="", help="Override backend.flux2_device.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="",
        help="Override backend.flux2_dtype.",
    )
    parser.add_argument("--negative-style-prompt", default="", help="Optional negative style prompt encoded as negative_prompt_embeds when CFG is active.")
    parser.add_argument("--allow-audit-fail", action="store_true")
    args = parser.parse_args(argv)

    config = StyleForgeConfig.from_file(args.config)
    config.dataset_root = str(args.dataset_root)
    if args.guidance is not None:
        config.generation.guidance = float(args.guidance)
    if args.steps is not None:
        config.generation.steps = int(args.steps)
    if args.flux2_base_model:
        config.backend.flux2_base_model = str(args.flux2_base_model)
    if args.device:
        config.backend.flux2_device = str(args.device)
    if args.dtype:
        config.backend.flux2_dtype = {
            "auto": "auto",
            "bfloat16": "bf16",
            "float16": "fp16",
            "float32": "fp32",
        }[args.dtype]
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = _selected_tasks(Path(config.dataset_root), _parse_task_ids(args.tasks))
    content_prompts = _load_content_prompts(args.content_prompts_file, args.content_prompt)
    records = _prepare_records(args, config, tasks, content_prompts)
    audit = _audit_inputs(records, content_prompts, args.negative_style_prompt)
    _write_json(args.output_root / "protocol_audit.json", audit)
    _write_json(args.output_root / "summary.contracts.json", {"audit": audit, "records": records})
    if not audit.get("ok") and not args.allow_audit_fail:
        raise ValueError("Protocol audit failed: " + "; ".join(str(error) for error in audit.get("errors") or []))

    generated = _generate_records(args, config, records)
    _write_contact_sheet(generated, args.output_root)
    summary = {"audit": audit, "records": generated}
    _write_json(args.output_root / "summary.json", summary)
    _write_html(generated, audit, args.output_root)
    print(f"ALL_DONE {args.output_root / 'summary.json'}")
    print(f"AUDIT {args.output_root / 'protocol_audit.json'} ok={audit['ok']}")
    print(f"HTML {args.output_root / 'index.html'}")
    print(f"CONTACT {args.output_root / 'contact_sheet.jpg'}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
