#!/usr/bin/env python3
"""Score StyleForge summaries through an OpenAI-compatible multimodal API."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qwen_style_judge import (
    CSV_FIELDS,
    JOINT_PROMPT_MODES,
    STYLE_LABEL_TO_SCORE,
    STYLE_VLM_SYSTEM_PROMPTS,
    STYLE_VLM_USER_PROMPTS,
    EvalItem,
    aggregate_scores,
    collect_from_summary,
    parse_style_label,
    resolve_cli_path,
    summarize_per_task,
    summarize_rows,
    unavailable_row,
)


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--endpoint", default="http://127.0.0.1:32133/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt-mode", choices=sorted(STYLE_VLM_SYSTEM_PROMPTS), default="joint_content_style")
    parser.add_argument("--max-refs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_output_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"response has no choices: {payload}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ).strip()
    return str(content or "").strip()


def post_json(endpoint: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("API returned non-object JSON")
    return result


def judge_pair(
    item: EvalItem,
    ref_index: int,
    ref_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    user_prompt = STYLE_VLM_USER_PROMPTS[args.prompt_mode]
    if args.prompt_mode in JOINT_PROMPT_MODES:
        user_prompt = user_prompt.format(content_prompt=str(item.metadata.get("content_prompt", "") or ""))
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": STYLE_VLM_SYSTEM_PROMPTS[args.prompt_mode]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(ref_path)}},
                    {"type": "image_url", "image_url": {"url": image_data_url(item.generated_image)}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": args.max_new_tokens,
        "regex": "(highly consistent|mostly consistent|slightly consistent|not consistent)",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error = ""
    for attempt in range(args.retries + 1):
        try:
            response = post_json(args.endpoint, payload, args.timeout)
            output_text = extract_output_text(response)
            label, score = parse_style_label(output_text)
            return {
                "key": pair_key(item, ref_index),
                "source_summary": item.source_summary,
                "summary_record_index": item.summary_record_index,
                "reference_index": ref_index,
                "reference_image": str(ref_path),
                "label": label,
                "score": score,
                "raw_output": output_text,
                "reason": "" if score is not None else "could not parse style label",
            }
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                try:
                    last_error = f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
                except Exception:
                    last_error = str(error)
            else:
                last_error = str(error)
            if attempt < args.retries:
                time.sleep(min(8.0, 2.0**attempt))
    return {
        "key": pair_key(item, ref_index),
        "source_summary": item.source_summary,
        "summary_record_index": item.summary_record_index,
        "reference_index": ref_index,
        "reference_image": str(ref_path),
        "label": None,
        "score": None,
        "raw_output": "",
        "reason": last_error or "API request failed",
    }


def pair_key(item: EvalItem, ref_index: int) -> str:
    return f"{item.source_summary}::{item.summary_record_index}::{ref_index}"


def judge_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": "sglang",
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "system_prompt": STYLE_VLM_SYSTEM_PROMPTS[args.prompt_mode],
        "user_prompt": STYLE_VLM_USER_PROMPTS[args.prompt_mode],
        "max_new_tokens": args.max_new_tokens,
    }


def load_pair_results(path: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("key"):
                results[str(row["key"])] = row
    return results


def append_pair_result(handle: Any, result: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()


def build_rows(items: Sequence[EvalItem], pair_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        generated_exists = bool(item.generated_image and item.generated_image.exists())
        existing_refs = [(index, path) for index, path in enumerate(item.ref_images) if path.exists()]
        if not item.generated_image:
            rows.append(unavailable_row(item, "missing generated_image field", False, [path for _, path in existing_refs]))
            continue
        if not generated_exists:
            rows.append(unavailable_row(item, f"generated image not found: {item.generated_image}", False, [path for _, path in existing_refs]))
            continue
        if not existing_refs:
            rows.append(unavailable_row(item, "no existing reference images", True, []))
            continue
        per_ref_results = [pair_results[pair_key(item, ref_index)] for ref_index, _ in existing_refs if pair_key(item, ref_index) in pair_results]
        aggregate = aggregate_scores([result.get("score") for result in per_ref_results])
        available = bool(aggregate["valid_ref_count"])
        reasons = sorted({str(result.get("reason") or "") for result in per_ref_results if result.get("reason")})
        if len(per_ref_results) != len(existing_refs):
            reasons.append("incomplete API results")
        row = {
            "source_summary": item.source_summary,
            "summary_record_index": item.summary_record_index,
            "task_id": item.task_id,
            "content_prompt": item.metadata.get("content_prompt", ""),
            "seed": item.metadata.get("seed", ""),
            "prompt_index": item.metadata.get("prompt_index", ""),
            "candidate_index": item.metadata.get("candidate_index", ""),
            "generated_image": str(item.generated_image),
            "generated_exists": True,
            "n_refs": len(item.ref_images),
            "available": available,
            "reason": "; ".join(reasons),
            "per_ref_results": per_ref_results,
            "qwen_vlm_j_labels": "|".join(str(result.get("label") or "") for result in per_ref_results),
            "qwen_vlm_j_raw_outputs": "|".join(str(result.get("raw_output") or "") for result in per_ref_results),
        }
        row.update(aggregate)
        rows.append(row)
    return rows


def stats(values: Iterable[float]) -> Dict[str, Any]:
    collected = list(values)
    if not collected:
        return {"count": 0}
    return {
        "count": len(collected),
        "mean": round(sum(collected) / len(collected), 4),
        "median": round(statistics.median(collected), 4),
        "min": round(min(collected), 4),
        "max": round(max(collected), 4),
    }


def summarize_per_source(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("source_summary", "")), []).append(row)
    return {source: summarize_rows(source_rows) for source, source_rows in sorted(grouped.items())}


def write_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    summary_paths: Sequence[Path],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "qwen_vlm_j_style_eval_api_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": {
            "name": "qwen_vlm_j_style_consistency",
            "score_scale": STYLE_LABEL_TO_SCORE,
            "aggregation": "Each generated image is judged against each reference separately; the image score is the mean of valid reference scores.",
            "style_only": args.prompt_mode not in JOINT_PROMPT_MODES,
            "content_prompt_used": args.prompt_mode in JOINT_PROMPT_MODES,
        },
        "config": {
            "inference_engine": "sglang",
            "endpoint": args.endpoint,
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "system_prompt": STYLE_VLM_SYSTEM_PROMPTS[args.prompt_mode],
            "user_prompt": STYLE_VLM_USER_PROMPTS[args.prompt_mode],
            "max_refs": args.max_refs,
            "max_new_tokens": args.max_new_tokens,
            "concurrency": args.concurrency,
            "enable_thinking": False,
        },
        "inputs": {
            "summaries": [str(path) for path in summary_paths],
            "summary_count": len(summary_paths),
            "record_count": len(rows),
        },
        "summary": summarize_rows(rows),
        "per_source": summarize_per_source(rows),
        "per_task": summarize_per_task(rows),
        "records": list(rows),
    }
    with (output_dir / "qwen_vlm_j_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "qwen_vlm_j_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in CSV_FIELDS})
    with (output_dir / "qwen_vlm_j_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    project_root = resolve_cli_path(args.project_root, DEFAULT_PROJECT_ROOT)
    summary_paths = [resolve_cli_path(value, project_root) for value in args.summary]
    items: List[EvalItem] = []
    for summary_path in summary_paths:
        items.extend(collect_from_summary(summary_path, project_root, max_refs=args.max_refs))
    if args.limit > 0:
        items = items[: args.limit]
    output_dir = resolve_cli_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "qwen_vlm_j_pair_results.jsonl"
    config = judge_config(args)
    # Legacy Transformers/API rows have no verified SGLang provenance.
    pair_results = {key: row for key, row in load_pair_results(pair_path).items()
                    if row.get("judge_config") == config}
    jobs: List[Tuple[EvalItem, int, Path]] = []
    for item in items:
        if not item.generated_image or not item.generated_image.exists():
            continue
        for ref_index, ref_path in enumerate(item.ref_images):
            key = pair_key(item, ref_index)
            if ref_path.exists() and (key not in pair_results or pair_results[key].get("score") is None):
                jobs.append((item, ref_index, ref_path))
    if not args.quiet:
        print(f"Collected {len(items)} records, {len(jobs)} pending image pairs, {len(pair_results)} cached pairs.", flush=True)
    lock = threading.Lock()
    completed = 0
    batch_size = max(1, args.concurrency * 4)
    with pair_path.open("a", encoding="utf-8") as pair_handle:
        for batch_start in range(0, len(jobs), batch_size):
            batch = jobs[batch_start : batch_start + batch_size]
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
                futures = {
                    executor.submit(judge_pair, item, ref_index, ref_path, args): (item, ref_index)
                    for item, ref_index, ref_path in batch
                }
                for future in as_completed(futures):
                    result = future.result()
                    result["judge_config"] = config
                    pair_results[str(result["key"])] = result
                    append_pair_result(pair_handle, result, lock)
                    completed += 1
                    if not args.quiet and (completed == 1 or completed == len(jobs) or completed % 25 == 0):
                        valid = sum(1 for value in pair_results.values() if value.get("score") is not None)
                        print(f"[{completed}/{len(jobs)}] latest={result.get('label')} cached_valid={valid}", flush=True)
    rows = build_rows(items, pair_results)
    write_outputs(output_dir, rows, summary_paths, args)
    summary = summarize_rows(rows)
    print(
        f"Wrote {output_dir}; available={summary['available']}/{summary['total']} "
        f"mean={summary['qwen_vlm_j_style_score'].get('mean', 'n/a')}",
        flush=True,
    )
    complete = all(
        pair_results.get(pair_key(item, ref_index), {}).get("score") is not None
        for item in items if item.generated_image and item.generated_image.exists()
        for ref_index, ref_path in enumerate(item.ref_images) if ref_path.exists()
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
