#!/usr/bin/env python3
"""Concurrent OpenAI-compatible transport for the official FreeStyle VLM-S/VLM-F.

Metric semantics are imported from ``freestyle_vlm_metrics.py``. This file only
adapts image transport to data URLs and submits the exact FreeStyle message
content to a local multimodal inference server.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

from freestyle_vlm_metrics import (
    DEFAULT_PROJECT_ROOT,
    INSTRUCTION_FOLLOW_PROMPT,
    METRICS,
    OFFICIAL_FOLLOW_SOURCE,
    OFFICIAL_FOLLOW_WRAPPER,
    OFFICIAL_STYLE_SOURCE,
    STYLE_5SCORE_PROMPT,
    load_existing,
    parse_overall_score_reason,
    parse_score_reason,
    read_summary,
    resolve_cli_path,
    row_complete,
    select_instruction_text,
    sha256_text,
    utc_now,
    write_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(METRICS))
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def style_messages(reference_url: str, generated_url: str) -> List[Dict[str, Any]]:
    # Exact order/text from FreeStyle style_similarity_dir.build_messages().
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image A:"},
                {"type": "image_url", "image_url": {"url": reference_url}},
                {"type": "text", "text": "Image B:"},
                {"type": "image_url", "image_url": {"url": generated_url}},
                {"type": "text", "text": STYLE_5SCORE_PROMPT},
            ],
        }
    ]


def follow_messages(generated_url: str, instruction: str) -> List[Dict[str, Any]]:
    # Exact order/text from FreeStyle edit_instruction_follow_dir.build_messages().
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Final image:"},
                {"type": "image_url", "image_url": {"url": generated_url}},
                {"type": "text", "text": f"Editing instruction:\n{instruction}"},
                {"type": "text", "text": INSTRUCTION_FOLLOW_PROMPT},
            ],
        }
    ]


def extract_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"response has no choices: {payload}")
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ).strip()
    return str(content or "").strip()


def post_chat(
    endpoint: str,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: int,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
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
    return extract_text(result)


def run_request(
    *,
    endpoint: str,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: int,
    parser: Any,
    prefix: str,
    timeout: float,
    retries: int,
) -> Dict[str, Any]:
    started = time.monotonic()
    raw = ""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            raw = post_chat(endpoint, model, messages, max_tokens, timeout)
            parsed = parser(raw)
            return {
                f"{prefix}_score": parsed["score"],
                f"{prefix}_reason": parsed["reason"],
                f"{prefix}_raw_output": raw,
                f"{prefix}_error": "",
                f"{prefix}_latency_s": round(time.monotonic() - started, 4),
            }
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                try:
                    last_error = f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
                except Exception:
                    last_error = str(error)
            else:
                last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(min(8.0, 2.0**attempt))
    return {
        f"{prefix}_score": None,
        f"{prefix}_reason": "",
        f"{prefix}_raw_output": raw,
        f"{prefix}_error": last_error or "request failed",
        f"{prefix}_latency_s": round(time.monotonic() - started, 4),
    }


def evaluate_record(base_row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    row = dict(base_row)
    generated_path = Path(row["generated_image"])
    reference_path = Path(row["style_reference"]) if row.get("style_reference") else None
    if not generated_path.is_file():
        error = f"FileNotFoundError: generated image not found: {generated_path}"
        for metric in args.metrics:
            row.update({f"{metric}_score": None, f"{metric}_error": error})
        return row

    generated_url = image_data_url(generated_path)
    if "vlm_s" in args.metrics:
        if reference_path is None or not reference_path.is_file():
            row.update(
                {
                    "vlm_s_score": None,
                    "vlm_s_error": f"FileNotFoundError: style reference not found: {reference_path}",
                }
            )
        else:
            row.update(
                run_request(
                    endpoint=args.endpoint,
                    model=args.model,
                    messages=style_messages(image_data_url(reference_path), generated_url),
                    max_tokens=512,
                    parser=parse_overall_score_reason,
                    prefix="vlm_s",
                    timeout=args.timeout,
                    retries=args.retries,
                )
            )
    if "vlm_f" in args.metrics:
        instruction = select_instruction_text(str(row.get("content_prompt") or "").strip(), "first_sentence")
        row.update(
            run_request(
                endpoint=args.endpoint,
                model=args.model,
                messages=follow_messages(generated_url, instruction),
                max_tokens=128,
                parser=parse_score_reason,
                prefix="vlm_f",
                timeout=args.timeout,
                retries=args.retries,
            )
        )
        row["vlm_f_instruction"] = instruction
    row["evaluated_at"] = utc_now()
    return row


def judge_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": "sglang",
        "model": args.model,
        "style_prompt_sha256": sha256_text(STYLE_5SCORE_PROMPT),
        "follow_prompt_sha256": sha256_text(INSTRUCTION_FOLLOW_PROMPT),
        "instruction_text_mode": "first_sentence",
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = [resolve_cli_path(value, project_root) for value in args.summary]
    records: List[Dict[str, Any]] = []
    for summary_path in summary_paths:
        records.extend(read_summary(summary_path, project_root))
    if args.limit > 0:
        records = records[: args.limit]

    records_path = output_dir / "records.jsonl"
    config = judge_config(args)
    active_keys = {row["record_key"] for row in records}
    existing = {key: row for key, row in load_existing(records_path).items()
                if key in active_keys and row.get("judge_config") == config}
    pending = [
        row
        for row in records
        if not row_complete(existing.get(row["record_key"], {}), args.metrics, args.retry_failures)
    ]
    metadata = {
        "inference_engine": "sglang",
        "evaluator": "FreeStyle VLM-S/VLM-F OpenAI-compatible API adapter",
        "official_style_source": OFFICIAL_STYLE_SOURCE,
        "official_follow_source": OFFICIAL_FOLLOW_SOURCE,
        "official_follow_wrapper": OFFICIAL_FOLLOW_WRAPPER,
        "style_prompt_sha256": sha256_text(STYLE_5SCORE_PROMPT),
        "follow_prompt_sha256": sha256_text(INSTRUCTION_FOLLOW_PROMPT),
        "endpoint": args.endpoint,
        "model": args.model,
        "metrics": list(args.metrics),
        "instruction_text_mode": "first_sentence",
        "style_reference_rule": "ref_images[0] (one reference per output; no aggregation)",
        "temperature": 0.0,
        "vlm_s_max_tokens": 512,
        "vlm_f_max_tokens": 128,
        "source_summaries": [str(path) for path in summary_paths],
        "started_at": utc_now(),
    }
    print(
        f"records={len(records)} existing={len(existing)} pending={len(pending)} "
        f"model={args.model} concurrency={args.concurrency}",
        flush=True,
    )
    lock = threading.Lock()
    completed = 0
    with records_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {executor.submit(evaluate_record, row, args): row["record_key"] for row in pending}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    row = dict(next(item for item in pending if item["record_key"] == key))
                    for metric in args.metrics:
                        row.update({f"{metric}_score": None, f"{metric}_error": f"{type(error).__name__}: {error}"})
                with lock:
                    row["judge_config"] = config
                    existing[key] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                completed += 1
                if not args.quiet and (completed == 1 or completed == len(pending) or completed % 25 == 0):
                    failures = {
                        metric: sum(value.get(f"{metric}_score") is None for value in existing.values())
                        for metric in args.metrics
                    }
                    print(f"[{completed}/{len(pending)}] failures={failures}", flush=True)

    write_snapshot(output_dir, list(existing.values()), metadata)
    failures = {
        metric: sum(row.get(f"{metric}_score") is None for row in existing.values())
        for metric in args.metrics
    }
    print(f"complete records={len(existing)} failures={failures} output={output_dir}", flush=True)
    return 0 if all(count == 0 for count in failures.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
