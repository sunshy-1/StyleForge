#!/usr/bin/env python3
"""Evaluate StyleForge generations with FreeStyle's official VLM-S and VLM-F.

The prompts, output contracts, parsers, and instruction-selection rule below
are copied verbatim from the FreeStyle evaluation implementation:

* benchmark_infer/src/metrics/vlm/style_similarity_dir.py
* benchmark_infer/src/metrics/vlm/edit_instruction_follow_dir.py
* benchmark_infer/scripts/metrics/instruction_score.sh

Only the I/O and inference transport are adapted: StyleForge ``summary.json``
records replace FreeStyle's basename-matched directories, and the local
Transformers Qwen multimodal judge replaces its OpenAI-compatible HTTP endpoint.
FreeStyle defines VLM-S for one style-reference image and one output image, so
this adapter uses exactly ``ref_images[0]`` and performs no multi-reference
aggregation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qwen_style_judge import resolve_record_path


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
OFFICIAL_STYLE_SOURCE = "benchmark_infer/src/metrics/vlm/style_similarity_dir.py"
OFFICIAL_FOLLOW_SOURCE = "benchmark_infer/src/metrics/vlm/edit_instruction_follow_dir.py"
OFFICIAL_FOLLOW_WRAPPER = "benchmark_infer/scripts/metrics/instruction_score.sh"
METRICS = ("vlm_s", "vlm_f")


# English translation of FreeStyle's STYLE_5SCORE_PROMPT; scoring semantics are preserved.
STYLE_5SCORE_PROMPT = r"""
You will see two images: A and B. Evaluate their consistency only from the perspective of visual and artistic style.
Do not judge whether their subject matter is the same; the people, objects, and scenes may be completely different.

Give an overall similarity score from 0 to 10 by considering all dimensions together.

Dimension definitions (follow these meanings strictly):
1) BRUSHSTROKE (strokes, lines, and edge treatment)
   - Line width, line jitter, outlining method, and the visibility of brush marks
   - Hard or soft edges, outlines, and stroke characteristics such as smearing, splashing, pencil, or pen marks
2) TEXTURE (texture, material rendering, grain and noise, and canvas or paper feel)
   - Fine surface grain, such as sandpaper grain, film noise, watercolor-paper fibers, or canvas weave
   - How fine material structures are rendered, such as layered paint, spray dots, halftone dots, or embossed relief
   - TEXTURE is not color and is not the shape or contour
3) COLOR (palette and color distribution, color temperature, saturation, and contrast)
   - Dominant hues and temperature tendency (cool or warm), saturation, and light-dark contrast
   - Similarity of color distribution, such as whether the proportions and covered areas of the main colors are close (large background areas and main color blocks)
   - COLOR is not texture grain and is not brushwork or linework
4) SHAPE (shape language and form conventions)
   - Degree of geometric or realistic treatment, exaggerated proportions, and sharper or rounder contours
   - Ways of simplifying structure, such as flat cartoon treatment, chibi proportions, minimalist geometry, or realistic structure
5) PATTERN (patterns, motifs, and decorative organization)
   - Repeated motifs, decorative elements, and fixed symbolic themes, such as recurring ornament, repeated decorative lines, or patterned backgrounds
   - Similarity in pattern density, repetition, and the organization of decorative elements

Scoring standard (0-10; score strictly):
* 0: The styles are completely inconsistent; they cannot reasonably be considered the same in this dimension.
* 1-3: Mostly inconsistent, with only weak or incidental similarities.
* 4-6: Some similarities, but with clear omissions or key differences.
* 7-9: Broadly consistent, with only a few minor issues.
* 10: Completely consistent, with the key features closely matched.

Important instructions (must be followed):
* Score strictly. Do not give a high score unless the dimension is clearly and accurately matched.
* Evaluate only consistency in the style dimensions. Do not consider whether the person or object identity is the same, or whether the background or scene is the same.
* Do not evaluate aesthetic quality or how attractive the image is.

Output rules (very important):
* score must be an integer from 0 to 10.
* reason must be 1-2 sentences citing specific observable evidence.
* Use exactly the format score@reason (output only this line; do not output any other characters, punctuation, or newline)
""".strip()


# Copied verbatim from FreeStyle's INSTRUCTION_FOLLOW_PROMPT.
INSTRUCTION_FOLLOW_PROMPT = r"""
You will be given:
1) An image (the final edited result).
2) An editing instruction (prompt).

Task:
Score how well the final image follows the instruction, from 0 to 10.
Give PARTIAL CREDIT whenever the image implements some parts correctly.
Do NOT assume anything that is not clearly visible in the image.

Important evaluation rule:
- Ignore style/aesthetic differences (rendering style, brushwork, resolution, noise, overall beauty)
  ONLY IF the instruction does NOT explicitly ask for them.
- If the instruction explicitly mentions color, lighting, mood, style, lens, texture, etc.,
  then you MUST evaluate those aspects (do NOT ignore them).

How to score (internal steps; do not output these steps):
1) Break the instruction into atomic, checkable requirements:
   - main subject(s) / objects
   - key attributes (color, material, count, size, identity if specified)
   - actions / interactions
   - spatial/layout constraints (left/right, background, position)
   - any required text/logos (exact wording matters)
   Mark each requirement as MUST-HAVE or NICE-TO-HAVE.
2) For each requirement, judge: satisfied / partially satisfied / not satisfied / contradicted.
3) Combine into a final score using this guidance:
   - Start from 10 and subtract penalties:
     * Missing a MUST-HAVE requirement: -2 to -4 (bigger miss = bigger penalty)
     * Partially meeting a MUST-HAVE: -1 to -2
     * Minor detail wrong (NICE-TO-HAVE): -0.5 to -1
     * Direct contradiction (e.g., asked to add X but image removes X): -3 to -5
     * Wrong or missing required text/logo: -3 to -6
   Clamp the result to [0, 10].

Score anchors (use these to avoid extreme scores):
- 9–10: All MUST-HAVEs satisfied; at most tiny minor issues. Use 10 ONLY if nothing important is missing or wrong.
- 7–8.5: Most MUST-HAVEs satisfied; 1–2 notable misses or several small issues.
- 4–6.5: Some key parts are implemented, but major omissions/misapplied edits exist.
- 1–3.5: Slight relation, but most MUST-HAVEs are missing/wrong.
- 0–0.5: Completely unrelated or clearly opposite to the instruction. Use 0 ONLY when it is truly a total failure.

Output rules (very important):
* Output ONLY one line in the format: score@reason
* score must be a number from 0 to 10, allowing .5 steps (e.g., 7.5). Do NOT always use 0 or 10.
* reason must be 1-2 short sentences, specific and directly observable (mention 1-2 key satisfied/missed requirements).
""".strip()


# The following parser helpers are copied verbatim from the corresponding
# FreeStyle files. Keep them separate because VLM-S accepts integer scores and
# has a legacy JSON fallback, while VLM-F accepts floating-point scores.
def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    m = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return s


def _clamp_score_0_10(score: int) -> int:
    if score < 0:
        return 0
    if score > 10:
        return 10
    return score


def parse_overall_score_reason(raw_text: str) -> Dict[str, Any]:
    clean = strip_code_fences(raw_text).strip()
    if not clean:
        raise ValueError("Model output is empty")
    first_line = clean.splitlines()[0].strip()
    m = re.match(r"^\s*(\d{1,2})\s*@\s*(.+?)\s*$", first_line)
    if m:
        score = _clamp_score_0_10(int(m.group(1)))
        reason = m.group(2).strip()
        if not reason:
            raise ValueError("reason is empty")
        return {"score": score, "reason": reason}
    try:
        obj = json.loads(clean)
    except Exception:
        raise ValueError(f"Could not parse score@reason, and the output is not JSON: {first_line!r}")
    if not isinstance(obj, dict):
        raise ValueError("Model JSON output is not an object")
    overall = obj.get("OVERALL", None)
    if not isinstance(overall, dict):
        raise ValueError("Model JSON is missing the OVERALL object")
    score_raw = overall.get("score", None)
    reason_raw = overall.get("reason", "")
    if isinstance(score_raw, bool) or score_raw is None:
        raise ValueError("OVERALL.score is invalid")
    score = int(float(score_raw)) if isinstance(score_raw, str) else int(score_raw)
    score = _clamp_score_0_10(score)
    reason = reason_raw if isinstance(reason_raw, str) else str(reason_raw)
    reason = reason.strip()
    if not reason:
        raise ValueError("OVERALL.reason is empty")
    return {"score": score, "reason": reason}


def parse_score_reason(raw_text: str) -> Dict[str, Any]:
    clean = strip_code_fences(raw_text).strip()
    if not clean:
        raise ValueError("Model output is empty")
    first_line = clean.splitlines()[0].strip()
    if "@" not in first_line:
        raise ValueError(f"Output does not match score@reason: {first_line!r}")
    score_text, reason = first_line.split("@", 1)
    score_text = score_text.strip()
    reason = reason.strip()
    try:
        score = float(score_text)
    except Exception as e:
        raise ValueError(f"score is not numeric: {score_text!r}") from e
    if score < 0.0:
        score = 0.0
    if score > 10.0:
        score = 10.0
    if not reason:
        raise ValueError("reason is empty")
    return {"score": score, "reason": reason}


def select_instruction_text(text: str, mode: str) -> str:
    mode = (mode or "full").lower()
    s = text or ""
    if mode == "first_sentence":
        parts = s.split(".", 1)
        return parts[0].strip()
    if mode == "full":
        return s
    raise ValueError(f"Unsupported instruction_text_mode: {mode}")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FreeStyle VLM-S/VLM-F evaluation using a local Qwen multimodal judge."
    )
    parser.add_argument("--summary", nargs="+", required=True, help="StyleForge summary.json file(s).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default="",
        help="Optional Transformers device map, for example 'auto', for sharding larger local VLMs.",
    )
    parser.add_argument(
        "--tp-plan",
        default="",
        help="Optional Transformers tensor-parallel plan, for example 'auto'.",
    )
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--image-max-side", type=int, default=0, help="0 preserves original resolution.")
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(METRICS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_cli_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), project_root, project_root.parent):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (project_root / path).resolve()


def read_summary(path: Path, project_root: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Expected an object with a records list: {path}")
    records: List[Dict[str, Any]] = []
    bases = (project_root, path.parent, path.parent.parent)
    backbone_dir = path.parent.parent.name
    backbone = backbone_dir[len("runs_"):] if backbone_dir.startswith("runs_") else backbone_dir
    method = path.parent.name
    for index, raw in enumerate(payload["records"]):
        if not isinstance(raw, dict):
            continue
        refs = raw.get("ref_images") or []
        refs = refs if isinstance(refs, list) else []
        generated = raw.get("generated_image") or ""
        record = {
            "record_key": f"{path.resolve()}::{index}",
            "source_summary": str(path.resolve()),
            "summary_record_index": index,
            "backbone": backbone,
            "method": method,
            "task_id": str(raw.get("task_id") or ""),
            "prompt_index": raw.get("prompt_index"),
            "candidate_index": raw.get("candidate_index"),
            "seed": raw.get("seed"),
            "content_prompt": str(raw.get("content_prompt") or ""),
            "generated_image": str(resolve_record_path(str(generated), bases)) if generated else "",
            "style_reference": str(resolve_record_path(str(refs[0]), bases)) if refs else "",
            "source_ref_count": len(refs),
        }
        records.append(record)
    return records


def load_rgb_image(path: Path, max_side: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side))
    return image


def build_style_messages(reference_image: Any, generated_image: Any) -> List[Dict[str, Any]]:
    # Exact content order and text from FreeStyle build_messages(). The PIL
    # image objects are the local-Transformers equivalent of image_url payloads.
    content = []
    content.append({"type": "text", "text": "Image A:"})
    content.append({"type": "image", "image": reference_image})
    content.append({"type": "text", "text": "Image B:"})
    content.append({"type": "image", "image": generated_image})
    content.append({"type": "text", "text": STYLE_5SCORE_PROMPT})
    return [{"role": "user", "content": content}]


def build_follow_messages(generated_image: Any, instruction: str) -> List[Dict[str, Any]]:
    # Exact content order and text from FreeStyle build_messages().
    content = []
    content.append({"type": "text", "text": "Final image:"})
    content.append({"type": "image", "image": generated_image})
    content.append({"type": "text", "text": f"Editing instruction:\n{instruction}"})
    content.append({"type": "text", "text": INSTRUCTION_FOLLOW_PROMPT})
    return [{"role": "user", "content": content}]


class LocalQwen3VL:
    def __init__(
        self,
        model_path: str,
        device: str,
        torch_dtype: str,
        attn_implementation: str,
        allow_download: bool,
        device_map: str = "",
        tp_plan: str = "",
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[torch_dtype]
        kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": dtype,
            "local_files_only": not allow_download,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.torch = torch
        self.device = device
        self.device_map = device_map.strip()
        self.tp_plan = tp_plan.strip()
        self.model_type = self._detect_model_type(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=not allow_download,
        )
        loaders = []
        if self.model_type.startswith("qwen3_vl"):
            try:
                from transformers import Qwen3VLForConditionalGeneration

                loaders.append(Qwen3VLForConditionalGeneration)
            except ImportError:
                pass
        if self.model_type.startswith("qwen3_5"):
            try:
                from transformers import Qwen3_5ForConditionalGeneration

                loaders.append(Qwen3_5ForConditionalGeneration)
            except ImportError:
                pass
        loaders.append(AutoModelForImageTextToText)
        if self.device_map:
            kwargs["device_map"] = self.device_map
        if self.tp_plan:
            kwargs["tp_plan"] = self.tp_plan
        last_error = None
        for loader in loaders:
            try:
                self.model = loader.from_pretrained(**kwargs)
                if not self.device_map and not self.tp_plan:
                    self.model = self.model.to(device)
                break
            except Exception as error:
                last_error = error
        else:
            raise RuntimeError(f"failed to load local multimodal judge: {last_error}")
        self.model.eval()

    @staticmethod
    def _detect_model_type(model_path: str) -> str:
        config_path = Path(model_path).expanduser() / "config.json"
        if not config_path.is_file():
            return ""
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return ""
        return str(config.get("model_type") or config.get("text_config", {}).get("model_type") or "").lower()

    @property
    def input_device(self):
        if self.device_map or self.tp_plan:
            return next(self.model.parameters()).device
        return self.device

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        image_inputs = [
            part["image"]
            for message in messages
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is not None
        ]
        disable_thinking = self.model_type.startswith(("qwen3_vl", "qwen3_5"))
        if disable_thinking:
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.processor(
                text=[rendered],
                images=[image_inputs],
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.input_device)
        else:
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            inputs = self.processor.apply_chat_template(messages, **template_kwargs).to(self.input_device)
        inputs.pop("token_type_ids", None)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def run_vlm_s(
    judge: LocalQwen3VL,
    reference_path: Path,
    generated_path: Path,
    image_max_side: int,
) -> Dict[str, Any]:
    reference_image = generated_image = None
    started = time.monotonic()
    try:
        reference_image = load_rgb_image(reference_path, image_max_side)
        generated_image = load_rgb_image(generated_path, image_max_side)
        raw = judge.generate(build_style_messages(reference_image, generated_image), max_new_tokens=512)
        parsed = parse_overall_score_reason(raw)
        return {
            "vlm_s_score": parsed["score"],
            "vlm_s_reason": parsed["reason"],
            "vlm_s_raw_output": raw,
            "vlm_s_error": "",
            "vlm_s_latency_s": round(time.monotonic() - started, 4),
        }
    except Exception as exc:
        return {
            "vlm_s_score": None,
            "vlm_s_reason": "",
            "vlm_s_raw_output": locals().get("raw", ""),
            "vlm_s_error": f"{type(exc).__name__}: {exc}",
            "vlm_s_latency_s": round(time.monotonic() - started, 4),
        }
    finally:
        if reference_image is not None:
            reference_image.close()
        if generated_image is not None:
            generated_image.close()


def run_vlm_f(
    judge: LocalQwen3VL,
    generated_path: Path,
    content_prompt: str,
    image_max_side: int,
) -> Dict[str, Any]:
    generated_image = None
    started = time.monotonic()
    try:
        generated_image = load_rgb_image(generated_path, image_max_side)
        instruction = select_instruction_text(content_prompt.strip(), "first_sentence")
        raw = judge.generate(build_follow_messages(generated_image, instruction), max_new_tokens=128)
        parsed = parse_score_reason(raw)
        return {
            "vlm_f_score": parsed["score"],
            "vlm_f_reason": parsed["reason"],
            "vlm_f_raw_output": raw,
            "vlm_f_error": "",
            "vlm_f_instruction": instruction,
            "vlm_f_latency_s": round(time.monotonic() - started, 4),
        }
    except Exception as exc:
        return {
            "vlm_f_score": None,
            "vlm_f_reason": "",
            "vlm_f_raw_output": locals().get("raw", ""),
            "vlm_f_error": f"{type(exc).__name__}: {exc}",
            "vlm_f_instruction": select_instruction_text(content_prompt.strip(), "first_sentence"),
            "vlm_f_latency_s": round(time.monotonic() - started, 4),
        }
    finally:
        if generated_image is not None:
            generated_image.close()


def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            key = row.get("record_key")
            if key:
                rows[str(key)] = row
    return rows


def row_complete(row: Dict[str, Any], metrics: Sequence[str], retry_failures: bool) -> bool:
    for metric in metrics:
        score_key = f"{metric}_score"
        error_key = f"{metric}_error"
        if score_key not in row and error_key not in row:
            return False
        if retry_failures and row.get(score_key) is None:
            return False
    return True


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def stats(values: Iterable[Any]) -> Dict[str, Any]:
    numeric = [float(value) for value in values if valid_number(value)]
    if not numeric:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(numeric),
        "mean": round(statistics.fmean(numeric), 6),
        "median": round(statistics.median(numeric), 6),
        "min": round(min(numeric), 6),
        "max": round(max(numeric), 6),
    }


def write_snapshot(output_dir: Path, rows: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    ordered = sorted(rows, key=lambda row: (row["source_summary"], row["summary_record_index"]))
    jsonl_path = output_dir / "records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "backbone", "method", "task_id", "prompt_index", "candidate_index", "seed",
        "content_prompt", "generated_image", "style_reference", "source_ref_count",
        "vlm_s_score", "vlm_s_reason", "vlm_s_error", "vlm_s_latency_s",
        "vlm_f_score", "vlm_f_reason", "vlm_f_error", "vlm_f_latency_s",
        "source_summary", "summary_record_index", "record_key",
    ]
    with (output_dir / "records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in ordered:
        groups.setdefault((row["backbone"], row["method"]), []).append(row)
    aggregate_rows = []
    for (backbone, method), group in sorted(groups.items()):
        aggregate_rows.append(
            {
                "backbone": backbone,
                "method": method,
                "record_count": len(group),
                "vlm_s": stats(row.get("vlm_s_score") for row in group),
                "vlm_f": stats(row.get("vlm_f_score") for row in group),
            }
        )
    summary = {
        "metadata": {**metadata, "updated_at": utc_now()},
        "record_count": len(ordered),
        "groups": aggregate_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(args.model_path).expanduser().resolve())
    summary_paths = [resolve_cli_path(value, project_root) for value in args.summary]
    records: List[Dict[str, Any]] = []
    for summary_path in summary_paths:
        records.extend(read_summary(summary_path, project_root))
    if args.limit > 0:
        records = records[: args.limit]

    records_path = output_dir / "records.jsonl"
    existing = load_existing(records_path)
    pending = [
        row for row in records
        if not row_complete(existing.get(row["record_key"], {}), args.metrics, args.retry_failures)
    ]
    metadata = {
        "evaluator": "FreeStyle VLM-S/VLM-F local Qwen multimodal adapter",
        "official_style_source": OFFICIAL_STYLE_SOURCE,
        "official_follow_source": OFFICIAL_FOLLOW_SOURCE,
        "official_follow_wrapper": OFFICIAL_FOLLOW_WRAPPER,
        "style_prompt_sha256": sha256_text(STYLE_5SCORE_PROMPT),
        "follow_prompt_sha256": sha256_text(INSTRUCTION_FOLLOW_PROMPT),
        "model_path": model_path,
        "metrics": list(args.metrics),
        "instruction_text_mode": "first_sentence",
        "style_reference_rule": "ref_images[0] (one reference per output; no aggregation)",
        "temperature_equivalent": "do_sample=False (greedy)",
        "vlm_s_max_new_tokens": 512,
        "vlm_f_max_new_tokens": 128,
        "image_max_side": args.image_max_side,
        "source_summaries": [str(path) for path in summary_paths],
        "started_at": utc_now(),
    }
    if not args.quiet:
        print(
            f"records={len(records)} existing={len(existing)} pending={len(pending)} "
            f"metrics={','.join(args.metrics)} model={model_path}",
            flush=True,
        )
    if pending:
        judge = LocalQwen3VL(
            model_path=model_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            allow_download=args.allow_download,
            device_map=args.device_map,
            tp_plan=args.tp_plan,
        )
    else:
        judge = None

    for position, base_row in enumerate(pending, start=1):
        key = base_row["record_key"]
        row = {**base_row, **existing.get(key, {})}
        generated_path = Path(row["generated_image"])
        reference_path = Path(row["style_reference"]) if row["style_reference"] else None
        if not generated_path.is_file():
            missing = f"FileNotFoundError: generated image not found: {generated_path}"
            for metric in args.metrics:
                row.update({f"{metric}_score": None, f"{metric}_error": missing})
        else:
            if "vlm_s" in args.metrics and (
                "vlm_s_score" not in row or (args.retry_failures and row.get("vlm_s_score") is None)
            ):
                if reference_path is None or not reference_path.is_file():
                    row.update(
                        {
                            "vlm_s_score": None,
                            "vlm_s_error": f"FileNotFoundError: style reference not found: {reference_path}",
                        }
                    )
                else:
                    row.update(run_vlm_s(judge, reference_path, generated_path, args.image_max_side))
            if "vlm_f" in args.metrics and (
                "vlm_f_score" not in row or (args.retry_failures and row.get("vlm_f_score") is None)
            ):
                row.update(run_vlm_f(judge, generated_path, row["content_prompt"], args.image_max_side))
        row["evaluated_at"] = utc_now()
        existing[key] = row
        # Append immediately for crash-safe progress. The final snapshot removes
        # duplicate retry rows and writes deterministic ordering.
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not args.quiet:
            elapsed = sum(
                float(row.get(f"{metric}_latency_s") or 0.0) for metric in args.metrics
            )
            scores = " ".join(f"{metric}={row.get(f'{metric}_score')}" for metric in args.metrics)
            print(
                f"[{position}/{len(pending)}] {row['backbone']}/{row['method']} "
                f"{row['task_id']} prompt={row['prompt_index']} {scores} time={elapsed:.2f}s",
                flush=True,
            )
        if position % 25 == 0:
            write_snapshot(output_dir, list(existing.values()), metadata)

    write_snapshot(output_dir, list(existing.values()), metadata)
    failures = {
        metric: sum(1 for row in existing.values() if row.get(f"{metric}_score") is None)
        for metric in args.metrics
    }
    print(f"complete records={len(existing)} failures={failures} output={output_dir}", flush=True)
    return 0 if all(value == 0 for value in failures.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
