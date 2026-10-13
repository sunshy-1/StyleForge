#!/usr/bin/env python3
"""All-pairs AB/BA win-rate evaluation through a SGLang API Qwen judge.

For every matched (style, content prompt, seed) item, all unordered method
pairs are judged twice with candidate order reversed.  The raw A/B/tie
decisions are resumable and the final metric is Average Pairwise Win Rate:
win=1, tie=0.5, loss=0, first averaged for every method-vs-method matchup and
then equally over opponents.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import itertools
import io
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qwen_style_judge import resolve_cli_path, resolve_record_path as resolve_summary_record_path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_MODEL = "models/Qwen3-VL-8B-Instruct"
DEFAULT_METHODS = (
    "naive_gpt",
    "native_reference",
    "styleforge",
    "dopsd",
    "vanilla_sft",
    "pso",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
JUDGE_LABEL = "Qwen3-VL-8B-Instruct"
# v4 uses constrained A/B/C answers with both candidate orders.
VERSION = "average_pairwise_win_rate_abba_v4_letter_answers"


PAIRWISE_SYSTEM_PROMPT = """
You are a strict pairwise evaluator of text-guided visual style transfer.

You will receive:
- a target content prompt;
- a style-reference board containing examples of the target visual style;
- Candidate A and Candidate B, which were generated for the same target.

Choose which candidate is the better complete style-transfer result. Judge the
two candidates jointly on both criteria below.

Content correctness:
- The candidate should visibly depict the requested subject, scene, action,
  objects, scale, and viewpoint.
- Missing or materially changing an important requested element is a serious
  failure even when the candidate looks attractive.

Reference-style fidelity:
- The candidate should transfer the reference examples' visual style through
  medium and rendering method, palette behavior, lighting, texture, line and
  edge quality, abstraction level, geometry, graphic effects, and atmosphere.
- The reference-board content is not the requested content. Do not reward a
  candidate for copying the reference subjects, objects, scene, layout, or
  composition.
- Do not reward a generic match to a broad style name when the concrete visual
  mechanisms shown in the references are weak or absent.

Prefer the candidate in which correct target content and faithful reference
style are more successfully integrated. Do not assign independent numerical
scores. Use a tie only when neither candidate is clearly better overall, such
as an effectively indistinguishable result or a genuine balanced tradeoff.

The final answer must be exactly one uppercase letter and nothing else:
A = Candidate A is better
B = Candidate B is better
C = tie
""".strip()


@dataclass(frozen=True)
class Candidate:
    method: str
    task_id: str
    content_prompt: str
    seed: int
    prompt_index: int
    candidate_index: int
    generated_image: Path
    ref_images: Tuple[Path, ...]


@dataclass(frozen=True)
class PairJob:
    key: str
    run_name: str
    run_root: Path
    style: str
    content_prompt: str
    seed: int
    prompt_index: int
    candidate_index: int
    method_1: str
    method_2: str
    order: str
    candidate_a_method: str
    candidate_b_method: str
    candidate_a_image: Path
    candidate_b_image: Path
    ref_images: Tuple[Path, ...]
    ref_board: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        nargs="+",
        required=True,
        help="One or more run roots containing <method>/summary.json.",
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "data" / "styles"))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--judge-label", default=JUDGE_LABEL)
    parser.add_argument("--output-name", default="qwen3_vl_8b")
    parser.add_argument("--backend", choices=("api",), default="api",
                        help="All judges use the SGLang API.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:32133/v1/chat/completions")
    parser.add_argument("--model", default=None, help="API served model name; defaults to model-path.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--image-max-side", type=int, default=768)
    parser.add_argument("--board-side", type=int, default=1024)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Limit pending jobs per run after sharding; 0 means all.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.model = args.model or Path(args.model_path).name
    if args.concurrency < 1 or args.timeout <= 0 or args.retries < 0:
        parser.error("require concurrency >= 1, timeout > 0 and retries >= 0")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_record_path(value: Any, run_root: Path, summary_path: Path) -> Path:
    return resolve_summary_record_path(str(value or ""), (PROJECT_ROOT, summary_path.parent, run_root))


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_method_candidates(run_root: Path, method: str) -> Dict[Tuple[str, int, int], Candidate]:
    summary_path = run_root / method / "summary.json"
    payload = read_json(summary_path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"records is not a list: {summary_path}")
    candidates: Dict[Tuple[str, int, int], Candidate] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id") or "")
        prompt_index = integer(record.get("prompt_index"), -1)
        candidate_index = integer(record.get("candidate_index"), 0)
        key = (task_id, prompt_index, candidate_index)
        if key in candidates:
            raise ValueError(f"duplicate item key {key} in {summary_path}")
        generated = resolve_record_path(record.get("generated_image"), run_root, summary_path)
        refs = tuple(
            resolve_record_path(value, run_root, summary_path)
            for value in (record.get("ref_images") or [])
            if value
        )
        candidates[key] = Candidate(
            method=method,
            task_id=task_id,
            content_prompt=str(record.get("content_prompt") or ""),
            seed=integer(record.get("seed"), -1),
            prompt_index=prompt_index,
            candidate_index=candidate_index,
            generated_image=generated,
            ref_images=refs,
        )
    return candidates


def validate_panel(
    run_root: Path,
    methods: Sequence[str],
) -> Tuple[Dict[str, Dict[Tuple[str, int, int], Candidate]], List[Tuple[str, int, int]]]:
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("methods must contain at least two unique names")
    panel = {method: load_method_candidates(run_root, method) for method in methods}
    expected = set(panel[methods[0]])
    for method in methods[1:]:
        actual = set(panel[method])
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(f"unmatched panel for {run_root}/{method}; missing={missing}, extra={extra}")
    for key in expected:
        base = panel[methods[0]][key]
        if not base.content_prompt:
            raise ValueError(f"empty content prompt: {run_root}, {key}")
        for method in methods:
            candidate = panel[method][key]
            if candidate.content_prompt != base.content_prompt or candidate.seed != base.seed:
                raise ValueError(f"content/seed mismatch: {run_root}, {key}, {method}")
            if not candidate.generated_image.exists():
                raise FileNotFoundError(candidate.generated_image)
    return panel, sorted(expected)


def dataset_style_refs(
    task_id: str,
    max_refs: int = 4,
    dataset_root: Optional[Path] = None,
) -> Tuple[Path, ...]:
    target_dir = (dataset_root or (PROJECT_ROOT / "data" / "styles")) / task_id / "target_images"
    refs = tuple(
        path.resolve()
        for path in sorted(target_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[:max_refs]
    if not refs:
        raise ValueError(f"no dataset style references for {task_id}: {target_dir}")
    return refs


def stable_shard(key: str, num_shards: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def board_path_for(output_dir: Path, task_id: str, ref_images: Sequence[Path]) -> Path:
    signature = "\n".join(str(path) for path in ref_images)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return output_dir / "style_boards" / f"{task_id}_{digest}.jpg"


def create_style_board(ref_images: Sequence[Path], output_path: Path, side: int) -> None:
    from PIL import Image, ImageOps

    if output_path.exists():
        return
    if not ref_images:
        raise ValueError("cannot create a style board without reference images")
    selected = list(ref_images[:4])
    while len(selected) < 4:
        selected.append(selected[-1])
    tile_side = side // 2
    board = Image.new("RGB", (tile_side * 2, tile_side * 2), "white")
    opened = []
    try:
        for index, path in enumerate(selected):
            image = Image.open(path).convert("RGB")
            opened.append(image)
            tile = ImageOps.fit(image, (tile_side, tile_side), method=Image.Resampling.LANCZOS)
            board.paste(tile, ((index % 2) * tile_side, (index // 2) * tile_side))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        board.save(temporary, format="JPEG", quality=95, subsampling=0)
        os.replace(temporary, output_path)
    finally:
        board.close()
        for image in opened:
            image.close()


def output_dir_for(run_root: Path, output_name: str) -> Path:
    return run_root / "pairwise_win_rate" / output_name


def build_jobs_for_run(
    run_root: Path,
    methods: Sequence[str],
    output_name: str,
    board_side: int,
    dataset_root: Path,
) -> Tuple[List[PairJob], Dict[str, Any]]:
    panel, item_keys = validate_panel(run_root, methods)
    output_dir = output_dir_for(run_root, output_name)
    run_name = run_root.name
    jobs: List[PairJob] = []
    style_refs: Dict[str, Tuple[Path, ...]] = {}
    for item_key in item_keys:
        base = panel[methods[0]][item_key]
        canonical_refs = dataset_style_refs(base.task_id, max_refs=4, dataset_root=dataset_root)
        previous = style_refs.setdefault(base.task_id, canonical_refs)
        if previous != canonical_refs:
            raise ValueError(f"dataset references vary within style {base.task_id} in {run_root}")
        board_path = board_path_for(output_dir, base.task_id, canonical_refs)
        create_style_board(canonical_refs, board_path, board_side)
        for method_1, method_2 in itertools.combinations(methods, 2):
            for order in ("AB", "BA"):
                candidate_a_method, candidate_b_method = (
                    (method_1, method_2) if order == "AB" else (method_2, method_1)
                )
                key_parts = (
                    run_name,
                    base.task_id,
                    str(base.prompt_index),
                    str(base.candidate_index),
                    str(base.seed),
                    method_1,
                    method_2,
                    order,
                )
                job_key = "|".join(key_parts)
                jobs.append(
                    PairJob(
                        key=job_key,
                        run_name=run_name,
                        run_root=run_root,
                        style=base.task_id,
                        content_prompt=base.content_prompt,
                        seed=base.seed,
                        prompt_index=base.prompt_index,
                        candidate_index=base.candidate_index,
                        method_1=method_1,
                        method_2=method_2,
                        order=order,
                        candidate_a_method=candidate_a_method,
                        candidate_b_method=candidate_b_method,
                        candidate_a_image=panel[candidate_a_method][item_key].generated_image,
                        candidate_b_image=panel[candidate_b_method][item_key].generated_image,
                        ref_images=canonical_refs,
                        ref_board=board_path,
                    )
                )
    audit = {
        "run": run_name,
        "run_root": str(run_root),
        "methods": list(methods),
        "method_count": len(methods),
        "style_count": len(style_refs),
        "matched_item_count": len(item_keys),
        "unordered_pairs_per_item": math.comb(len(methods), 2),
        "orders_per_pair": 2,
        "expected_ordered_judgments": len(jobs),
        "style_reference_policy": "fixed 2x2 board of the first four reference images",
        "style_boards": {style: str(board_path_for(output_dir, style, refs)) for style, refs in sorted(style_refs.items())},
    }
    return jobs, audit


def load_latest_records(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("key"):
                    latest[str(row["key"])] = row
    return latest


def shard_record_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return output_dir / "shards" / f"records_shard_{shard_index:03d}_of_{num_shards:03d}.jsonl"


def judge_rgb(path: Path, max_side: int) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def make_messages(
    job: PairJob,
    reference_board: Any,
    candidate_a: Any,
    candidate_b: Any,
) -> List[Dict[str, Any]]:
    prompt = (
        f"Target content prompt:\n{job.content_prompt}\n\n"
        "The first image is the STYLE REFERENCE BOARD only; its depicted content is not the target. "
        "The second image is CANDIDATE A. The third image is CANDIDATE B. "
        "Select the better joint content-and-style result. Output exactly A, B, or C."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": PAIRWISE_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "STYLE REFERENCE BOARD:"},
                {"type": "image", "image": reference_board},
                {"type": "text", "text": "CANDIDATE A:"},
                {"type": "image", "image": candidate_a},
                {"type": "text", "text": "CANDIDATE B:"},
                {"type": "image", "image": candidate_b},
                {"type": "text", "text": "Final answer (A=Candidate A, B=Candidate B, C=tie):"},
            ],
        },
    ]


def parse_choice(raw_output: Any) -> Optional[str]:
    value = str(raw_output or "").strip()
    mapping = {"A": "A", "B": "B", "C": "tie", "0": "A", "1": "B", "2": "tie"}
    if value in mapping:
        return mapping[value]
    lowered = value.lower()
    aliases = {
        "a": "A",
        "candidate a": "A",
        "b": "B",
        "candidate b": "B",
        "tie": "tie",
    }
    return aliases.get(lowered)


def result_for(job: PairJob, raw_output: str, error: str = "", backend: str = "api") -> Dict[str, Any]:
    choice = None if error else parse_choice(raw_output)
    winner_method: Optional[str]
    if choice == "A":
        winner_method = job.candidate_a_method
    elif choice == "B":
        winner_method = job.candidate_b_method
    else:
        winner_method = None
    if choice == "tie":
        method_1_score: Optional[float] = 0.5
    elif winner_method == job.method_1:
        method_1_score = 1.0
    elif winner_method == job.method_2:
        method_1_score = 0.0
    else:
        method_1_score = None
    return {
        "version": VERSION,
        "backend": backend,
        "created_at": utc_now(),
        "key": job.key,
        "run": job.run_name,
        "style": job.style,
        "content_prompt": job.content_prompt,
        "seed": job.seed,
        "prompt_index": job.prompt_index,
        "candidate_index": job.candidate_index,
        "method_1": job.method_1,
        "method_2": job.method_2,
        "order": job.order,
        "candidate_a_method": job.candidate_a_method,
        "candidate_b_method": job.candidate_b_method,
        "candidate_a_image": str(job.candidate_a_image),
        "candidate_b_image": str(job.candidate_b_image),
        "reference_board": str(job.ref_board),
        "reference_images": [str(path) for path in job.ref_images],
        "raw_output": str(raw_output or ""),
        "choice": choice,
        "winner_method": winner_method,
        "method_1_score": method_1_score,
        "valid": method_1_score is not None,
        "error": error if error else ("" if choice is not None else "could not parse judge output"),
    }


def record_valid(row: Mapping[str, Any], backend: str) -> bool:
    # Older local records did not include a backend. Never reuse them for API.
    return (row.get("version") == VERSION and bool(row.get("valid"))
            and row.get("method_1_score") in (0.0, 0.5, 1.0)
            and row.get("backend", "local") == backend and not row.get("error"))


def make_api_messages(job: PairJob, image_max_side: int) -> List[Dict[str, Any]]:
    urls = []
    for path in (job.ref_board, job.candidate_a_image, job.candidate_b_image):
        image = judge_rgb(path, image_max_side)
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            urls.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
        finally:
            image.close()
    # Preserve the metric's prompt, image order, and resizing.
    messages = make_messages(job, *urls)
    for message in messages:
        for part in message["content"]:
            if part["type"] == "image":
                part["type"] = "image_url"
                part["image_url"] = {"url": part.pop("image")}
    return messages


def score_api_job(job: PairJob, args: argparse.Namespace) -> Dict[str, Any]:
    raw = ""
    last_error = ""
    started = time.monotonic()
    try:
        payload = {
            "model": args.model,
            "messages": make_api_messages(job, args.image_max_side),
            "temperature": 0.0,
            "max_tokens": 1,
            "regex": "[ABC]",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(args.retries + 1):
            try:
                request = urllib.request.Request(args.endpoint, data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    result = json.load(response)
                content = result["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(str(part.get("text", "")) for part in content
                                      if isinstance(part, dict))
                raw = str(content or "").strip()
                if raw not in {"A", "B", "C"}:
                    raise ValueError(f"Expected constrained A/B/C answer, got {raw!r}")
                last_error = ""
                break
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError):
                    last_error = f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
                else:
                    last_error = f"{type(error).__name__}: {error}"
                if attempt < args.retries:
                    time.sleep(min(8.0, 2.0**attempt))
    except Exception as error:
        last_error = f"{type(error).__name__}: {error}"
    row = result_for(job, raw, last_error, backend="api")
    row.update({"model": args.model, "endpoint": args.endpoint,
                "latency_s": round(time.monotonic() - started, 4)})
    return row


def run_api_jobs(pending_by_run: Sequence[Any], args: argparse.Namespace) -> int:
    failures = 0
    for run_root, record_path, pending, _ in pending_by_run:
        if not pending:
            continue
        record_path.parent.mkdir(parents=True, exist_ok=True)
        valid_completed = 0
        with record_path.open("a", encoding="utf-8") as handle, \
                ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(score_api_job, job, args) for job in pending]
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                valid_completed += int(row["valid"])
                failures += int(not row["valid"])
                if not args.quiet and (completed == 1 or completed % 32 == 0 or completed == len(pending)):
                    print(f"{run_root.name} SGLang shard {args.shard_index}: "
                          f"[{completed}/{len(pending)}] valid={valid_completed} "
                          f"failures={completed - valid_completed}", flush=True)
        if valid_completed != len(pending):
            print(f"WARNING {run_root.name}: rerun to retry {len(pending) - valid_completed} failed API judgments", flush=True)
    return int(failures > 0)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def rounded(value: float, digits: int = 6) -> Optional[float]:
    if not math.isfinite(value):
        return None
    return round(value, digits)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(
    output_dir: Path,
    jobs: Sequence[PairJob],
    audit: Dict[str, Any],
    methods: Sequence[str],
    judge_label: str,
    backend: str = "api",
) -> Dict[str, Any]:
    shard_paths = sorted((output_dir / "shards").glob("records_shard_*_of_*.jsonl"))
    records_by_key = load_latest_records(shard_paths)
    expected_keys = {job.key for job in jobs}
    unexpected_keys = sorted(set(records_by_key) - expected_keys)
    valid = {
        key: row
        for key, row in records_by_key.items()
        if key in expected_keys
        and record_valid(row, backend)
    }
    invalid = {key: row for key, row in records_by_key.items() if key in expected_keys and key not in valid}

    matchup_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    matchup_by_style: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    choices: Dict[str, int] = defaultdict(int)
    position_winners: Dict[str, int] = defaultdict(int)
    physical_pairs: Dict[Tuple[str, int, int, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in valid.values():
        method_1 = str(row["method_1"])
        method_2 = str(row["method_2"])
        score_1 = float(row["method_1_score"])
        style = str(row["style"])
        matchup_scores[(method_1, method_2)].append(score_1)
        matchup_scores[(method_2, method_1)].append(1.0 - score_1)
        matchup_by_style[(style, method_1, method_2)].append(score_1)
        matchup_by_style[(style, method_2, method_1)].append(1.0 - score_1)
        choices[str(row.get("choice"))] += 1
        position_winners[str(row.get("choice"))] += 1
        physical_key = (
            style,
            integer(row.get("prompt_index"), -1),
            integer(row.get("candidate_index"), 0),
            method_1,
            method_2,
        )
        physical_pairs[physical_key][str(row.get("order"))] = row

    pairwise_rows: List[Dict[str, Any]] = []
    matrix: Dict[str, Dict[str, Optional[float]]] = {method: {} for method in methods}
    expected_ordered_per_matchup = audit["matched_item_count"] * 2
    for method in methods:
        for opponent in methods:
            if method == opponent:
                matrix[method][opponent] = 0.5
                continue
            scores = matchup_scores.get((method, opponent), [])
            value = mean(scores)
            matrix[method][opponent] = rounded(value)
            pairwise_rows.append(
                {
                    "method": method,
                    "opponent": opponent,
                    "ordered_judgments": len(scores),
                    "expected_ordered_judgments": expected_ordered_per_matchup,
                    "win_rate": rounded(value),
                }
            )

    method_rows: List[Dict[str, Any]] = []
    for method in methods:
        opponent_rates = [matrix[method][opponent] for opponent in methods if opponent != method]
        numeric_rates = [float(value) for value in opponent_rates if value is not None]
        method_rows.append(
            {
                "method": method,
                "judge": judge_label,
                "opponent_count": len(numeric_rates),
                "expected_opponent_count": len(methods) - 1,
                "average_pairwise_win_rate": rounded(mean(numeric_rates)),
            }
        )

    styles = sorted({job.style for job in jobs})
    per_style_rows: List[Dict[str, Any]] = []
    for style in styles:
        for method in methods:
            opponent_rates = []
            ordered_count = 0
            for opponent in methods:
                if opponent == method:
                    continue
                scores = matchup_by_style.get((style, method, opponent), [])
                ordered_count += len(scores)
                if scores:
                    opponent_rates.append(mean(scores))
            per_style_rows.append(
                {
                    "style": style,
                    "method": method,
                    "ordered_judgments": ordered_count,
                    "average_pairwise_win_rate": rounded(mean(opponent_rates)),
                }
            )

    complete_physical = 0
    same_mapped_decision = 0
    swap_conflict = 0
    any_tie = 0
    for orders in physical_pairs.values():
        if set(orders) != {"AB", "BA"}:
            continue
        complete_physical += 1
        ab = orders["AB"]
        ba = orders["BA"]
        if ab.get("choice") == "tie" or ba.get("choice") == "tie":
            any_tie += 1
        if ab.get("winner_method") == ba.get("winner_method") and ab.get("choice") == ba.get("choice") == "tie":
            same_mapped_decision += 1
        elif ab.get("winner_method") is not None and ab.get("winner_method") == ba.get("winner_method"):
            same_mapped_decision += 1
        else:
            swap_conflict += 1

    matrix_rows = []
    for method in methods:
        row: Dict[str, Any] = {"method": method}
        row.update({opponent: matrix[method][opponent] for opponent in methods})
        matrix_rows.append(row)

    method_values = [row["average_pairwise_win_rate"] for row in method_rows]
    numeric_method_values = [float(value) for value in method_values if value is not None]
    conservation_mean = mean(numeric_method_values)
    max_matrix_error = 0.0
    for method_1, method_2 in itertools.combinations(methods, 2):
        left = matrix[method_1][method_2]
        right = matrix[method_2][method_1]
        if left is not None and right is not None:
            max_matrix_error = max(max_matrix_error, abs(float(left) + float(right) - 1.0))

    coverage = {
        **audit,
        "version": VERSION,
        "judge": judge_label,
        "backend": backend,
        "created_at": utc_now(),
        "record_shards": [str(path) for path in shard_paths],
        "record_keys": len(records_by_key),
        "valid_ordered_judgments": len(valid),
        "invalid_ordered_judgments": len(invalid),
        "missing_ordered_judgments": len(expected_keys - set(valid)),
        "unexpected_record_keys": len(unexpected_keys),
        "complete": len(valid) == len(expected_keys) and not unexpected_keys,
        "choice_counts": dict(sorted(choices.items())),
        "position_choice_counts": dict(sorted(position_winners.items())),
        "complete_abba_physical_pairs": complete_physical,
        "expected_physical_pairs": len(expected_keys) // 2,
        "same_mapped_decision_after_swap": same_mapped_decision,
        "swap_conflict_or_tie_disagreement": swap_conflict,
        "physical_pairs_with_any_tie": any_tie,
        "swap_agreement_rate": rounded(same_mapped_decision / complete_physical) if complete_physical else None,
        "mean_across_method_win_rates_should_be_0_5": rounded(conservation_mean),
        "max_pairwise_complement_error": rounded(max_matrix_error),
        "invalid_examples": list(invalid.values())[:10],
        "unexpected_key_examples": unexpected_keys[:10],
    }
    summary = {
        "version": VERSION,
        "metric": "Average Pairwise Win Rate",
        "backend": backend,
        "judge": judge_label,
        "run": audit["run"],
        "created_at": utc_now(),
        "scoring": {"win": 1.0, "tie": 0.5, "loss": 0.0},
        "aggregation": "mean over ordered AB/BA judgments per opponent, then equal mean over opponents",
        "style_aggregation": "31-style macro mean is identical to the overall mean because every style has 16 matched items",
        "methods": method_rows,
        "complete": coverage["complete"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(
        output_dir / "method_win_rates.csv",
        method_rows,
        ("method", "judge", "opponent_count", "expected_opponent_count", "average_pairwise_win_rate"),
    )
    write_csv(
        output_dir / "pairwise_win_rates.csv",
        pairwise_rows,
        ("method", "opponent", "ordered_judgments", "expected_ordered_judgments", "win_rate"),
    )
    write_csv(output_dir / "pairwise_matrix.csv", matrix_rows, ("method", *methods))
    write_csv(
        output_dir / "per_style_win_rates.csv",
        per_style_rows,
        ("style", "method", "ordered_judgments", "average_pairwise_win_rate"),
    )
    return {"summary": summary, "audit": coverage}


def resolve_run_roots(values: Sequence[str]) -> List[Path]:
    roots = []
    for value in values:
        path = resolve_cli_path(value, PROJECT_ROOT)
        if not path.is_dir():
            raise FileNotFoundError(path)
        roots.append(path.resolve())
    return roots


def main() -> int:
    args = parse_args()
    process_rank = int(os.environ.get("RANK", "0"))
    is_primary_process = process_rank == 0
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    run_roots = resolve_run_roots(args.run_root)
    dataset_root = resolve_cli_path(args.dataset_root, PROJECT_ROOT)
    run_payloads = []
    for run_root in run_roots:
        jobs, audit = build_jobs_for_run(
            run_root, args.methods, args.output_name, args.board_side, dataset_root
        )
        run_payloads.append((run_root, jobs, audit))
        if not args.quiet and is_primary_process:
            print(
                f"{run_root.name}: matched={audit['matched_item_count']} styles={audit['style_count']} "
                f"methods={audit['method_count']} ordered_jobs={len(jobs)}",
                flush=True,
            )
    if args.prepare_only:
        return 0
    if args.summarize_only:
        complete = True
        for run_root, jobs, audit in run_payloads:
            result = summarize_run(
                output_dir_for(run_root, args.output_name), jobs, audit, args.methods, args.judge_label, args.backend
            )
            complete = complete and result["summary"]["complete"]
            print(json.dumps(result["summary"], indent=2, ensure_ascii=False), flush=True)
        return 0 if complete else 1

    pending_by_run: List[Tuple[Path, Path, List[PairJob], Dict[str, Dict[str, Any]]]] = []
    total_pending = 0
    for run_root, jobs, _ in run_payloads:
        output_dir = output_dir_for(run_root, args.output_name)
        record_path = shard_record_path(output_dir, args.shard_index, args.num_shards)
        cached = load_latest_records([record_path])
        shard_jobs = [job for job in jobs if stable_shard(job.key, args.num_shards) == args.shard_index]
        pending = [
            job
            for job in shard_jobs
            if not record_valid(cached.get(job.key, {}), args.backend)
        ]
        if args.limit > 0:
            pending = pending[: args.limit]
        pending_by_run.append((run_root, record_path, pending, cached))
        total_pending += len(pending)
        if not args.quiet and is_primary_process:
            valid_cached = sum(record_valid(cached.get(job.key, {}), args.backend) for job in shard_jobs)
            print(
                f"{run_root.name}: shard={args.shard_index}/{args.num_shards} "
                f"assigned={len(shard_jobs)} cached_valid={valid_cached} pending={len(pending)}",
                flush=True,
            )
    if total_pending == 0:
        return 0

    return run_api_jobs(pending_by_run, args)


if __name__ == "__main__":
    raise SystemExit(main())
