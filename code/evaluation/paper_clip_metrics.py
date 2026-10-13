#!/usr/bin/env python3
"""Paper-aligned CLIP-Text and CLIP-Style evaluation.

The image-text scorer and score scale are copied from the original
``evaluation/evaluate.py`` implementation. This release exposes only the two
CLIP metrics reported by the paper; no reference-image diagnostic or optional
image-quality metric is computed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qwen_style_judge import resolve_record_path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_MODEL_PATH = "models/CLIP-ViT-H-14-laion2B-s32B-b79K"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class EvalItem:
    source: str
    task_id: str
    generated_image: Optional[Path]
    ref_images: List[Path]
    metadata: Dict[str, Any]
    source_summary: str = ""
    summary_record_index: Optional[int] = None


def resolve_cli_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), project_root, project_root.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return project_root / path


def read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def first_nonempty_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_style_text(record: Dict[str, Any]) -> str:
    direct = first_nonempty_string(
        record.get("generation_style_prompt"),
        record.get("style_prompt"),
        record.get("compiled_style_prompt"),
    )
    if direct:
        return direct

    full_prompts = record.get("full_prompts")
    if isinstance(full_prompts, dict):
        # Fallback only: these prompts can include content text, but are still
        # useful for legacy summaries that did not store a pure style prompt.
        return first_nonempty_string(
            full_prompts.get("style"),
            full_prompts.get("style_schema"),
            full_prompts.get("prompt"),
        )
    return ""


def canonical_style_text(task_id: str, metadata_root: Optional[Path]) -> str:
    if metadata_root is None:
        return ""
    metadata_path = metadata_root / task_id / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"canonical style metadata not found: {metadata_path}")
    metadata = read_json(metadata_path)
    style_text = first_nonempty_string(metadata.get("style_prompt"))
    if not style_text:
        raise ValueError(f"canonical style metadata has no style_prompt: {metadata_path}")
    return style_text


def collect_from_summary(
    summary_path: Path,
    project_root: Path,
    canonical_style_metadata_root: Optional[Path] = None,
) -> List[EvalItem]:
    data = read_json(summary_path)
    raw_records = data.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError(f"summary records must be a list: {summary_path}")

    items: List[EvalItem] = []
    base_dirs = [project_root, summary_path.parent, summary_path.parent.parent]
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id") or summary_path.parent.name)
        gen_value = record.get("generated_image")
        generated_image = None
        if gen_value:
            generated_image = resolve_record_path(str(gen_value), base_dirs)
        ref_values = record.get("ref_images") or []
        if not isinstance(ref_values, list):
            ref_values = []
        ref_images = [resolve_record_path(str(value), base_dirs) for value in ref_values if value]
        style_text = extract_style_text(record)
        if canonical_style_metadata_root is not None:
            style_text = canonical_style_text(task_id, canonical_style_metadata_root)
        metadata = {
            "content_prompt": record.get("content_prompt", ""),
            "style_prompt": style_text,
            "seed": record.get("seed", ""),
            "prompt_index": record.get("prompt_index", ""),
            "candidate_index": record.get("candidate_index", ""),
        }
        items.append(
            EvalItem(
                source="summary",
                task_id=task_id,
                generated_image=generated_image,
                ref_images=ref_images,
                metadata=metadata,
                source_summary=str(summary_path),
                summary_record_index=index,
            )
        )
    return items


def collect_items(
    summary_paths: Sequence[Path],
    project_root: Path,
    canonical_style_metadata_root: Optional[Path] = None,
    limit: int = 0,
) -> List[EvalItem]:
    items: List[EvalItem] = []
    for summary_path in summary_paths:
        items.extend(
            collect_from_summary(
                summary_path,
                project_root,
                canonical_style_metadata_root=canonical_style_metadata_root,
            )
        )
        if limit and len(items) >= limit:
            return items[:limit]
    return items


def feature_tensor(output: Any, feature_name: str) -> Any:
    torch = sys.modules.get("torch")
    if torch is not None and torch.is_tensor(output):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if torch is None:
            if value is not None:
                return value
        elif torch.is_tensor(value):
            return value
    if isinstance(output, (list, tuple)):
        for value in output:
            if torch is None:
                if value is not None:
                    return value
            elif torch.is_tensor(value):
                return value
    raise TypeError(f"{feature_name} must be a tensor or a model output with projected features.")


class PaperCLIPScorer:
    """Paper-aligned CLIP image-text scorer.

    arXiv:2606.13809 reports CLIP-Text/CLIP-Style on the usual CLIP score
    scale. DiffSynth's CLIPMetric computes normalized image-text cosine and
    multiplies it by ``model.logit_scale.exp()``. For the local ViT-H/14
    checkpoint this scale is 100, so a raw cosine of 0.233 is reported as 23.3.
    """

    def __init__(self, model_path: str, device: str):
        self.model_path = Path(model_path).expanduser()
        self.device_arg = device
        self.backend_name = "clip_hf_vit_h_14_laion2b_s32b_b79k_logit_scale"
        self._torch = None
        self._model = None
        self._processor = None
        self._device = None
        self._load_error = ""
        self._text_cache: Dict[str, Any] = {}
        self._image_cache: Dict[str, Any] = {}
        self._load()

    @property
    def available(self) -> bool:
        return self._model is not None and self._processor is not None and not self._load_error

    def _load(self) -> None:
        if not self.model_path.exists():
            self._load_error = f"paper CLIP model path not found: {self.model_path}"
            return
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel

            device = self.device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
            self._torch = torch
            self._device = torch.device(device)
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model = CLIPModel.from_pretrained(
                self.model_path,
                local_files_only=True,
            ).to(self._device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on local runtime
            self._model = None
            self._processor = None
            self._load_error = f"{type(exc).__name__}: {exc}"

    def score(self, image_path: Path, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"available": False, "reason": "empty text"}
        if not self.available:
            return {"available": False, "reason": self._load_error or "paper CLIP scorer unavailable"}
        try:
            torch = self._torch
            image_feature = self._image_feature(image_path)
            text_feature = self._text_feature(text)
            raw_cosine = float(torch.dot(image_feature, text_feature))
            scale = self._score_scale()
            return {
                "available": True,
                "raw_cosine": round(raw_cosine, 6),
                "score": round(raw_cosine * scale, 6),
                "scale": round(scale, 6),
                "backend": self.backend_name,
            }
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _tensor_inputs(self, inputs: Any) -> Dict[str, Any]:
        torch = self._torch
        return {
            name: value.to(self._device)
            for name, value in dict(inputs).items()
            if torch.is_tensor(value)
        }

    def _image_feature(self, path: Path) -> Any:
        key = str(path.resolve())
        if key in self._image_cache:
            return self._image_cache[key]
        torch = self._torch
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            inputs = self._tensor_inputs(self._processor(images=image, return_tensors="pt"))
        with torch.no_grad():
            features = feature_tensor(self._model.get_image_features(**inputs), "image_features")
            features = torch.nn.functional.normalize(features, dim=-1)
        feature = features.squeeze(0)
        self._image_cache[key] = feature
        return feature

    def _text_feature(self, text: str) -> Any:
        if text in self._text_cache:
            return self._text_cache[text]
        torch = self._torch
        inputs = self._tensor_inputs(
            self._processor(
                text=[text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        )
        with torch.no_grad():
            features = feature_tensor(self._model.get_text_features(**inputs), "text_features")
            features = torch.nn.functional.normalize(features, dim=-1)
        feature = features.squeeze(0)
        self._text_cache[text] = feature
        return feature

    def _score_scale(self) -> float:
        torch = self._torch
        logit_scale = getattr(self._model, "logit_scale", None)
        if logit_scale is None:
            return 1.0
        with torch.no_grad():
            return float(logit_scale.detach().float().exp().cpu().item())


def item_identity(item: EvalItem) -> Dict[str, Any]:
    content_prompt = str(item.metadata.get("content_prompt", "") or "")
    style_prompt = str(item.metadata.get("style_prompt", "") or "")
    return {
        "task_id": item.task_id,
        "source_summary": item.source_summary,
        "summary_record_index": item.summary_record_index,
        "generated_image": str(item.generated_image) if item.generated_image else "",
        "content_prompt": content_prompt,
        "style_prompt": style_prompt,
        "seed": item.metadata.get("seed", ""),
        "prompt_index": item.metadata.get("prompt_index", ""),
        "candidate_index": item.metadata.get("candidate_index", ""),
    }


def score_item(item: EvalItem, scorer: PaperCLIPScorer) -> Dict[str, Any]:
    row = {**item_identity(item), "available": False, "reason": ""}
    content_prompt = row["content_prompt"]
    style_prompt = row["style_prompt"]
    for prefix, text in (("clip_text", content_prompt), ("clip_style", style_prompt)):
        row[f"{prefix}_raw_cosine"] = None
        row[f"{prefix}_score"] = None
        row[f"{prefix}_scale"] = None
        row[f"{prefix}_backend"] = ""
        row[f"{prefix}_reason"] = ""
        if item.generated_image is None or not item.generated_image.is_file():
            row[f"{prefix}_reason"] = "generated image not found"
            continue
        result = scorer.score(item.generated_image, text)
        if result.get("available"):
            row[f"{prefix}_raw_cosine"] = result.get("raw_cosine")
            row[f"{prefix}_score"] = result.get("score")
            row[f"{prefix}_scale"] = result.get("scale")
            row[f"{prefix}_backend"] = result.get("backend", "")
            row["available"] = True
        else:
            row[f"{prefix}_reason"] = result.get("reason", "unavailable")
    if not row["available"]:
        row["reason"] = "both CLIP-Text and CLIP-Style unavailable"
    return row


def stats(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--canonical-style-metadata",
        default="",
        help="Optional root containing <task_id>/metadata.json with the paper canonical style_prompt.",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    project_root = resolve_cli_path(args.project_root, PROJECT_ROOT)
    summary_paths = [resolve_cli_path(value, project_root) for value in args.summary]
    canonical_style_metadata_root = (
        resolve_cli_path(args.canonical_style_metadata, project_root)
        if args.canonical_style_metadata
        else None
    )
    items = collect_items(
        summary_paths,
        project_root,
        canonical_style_metadata_root=canonical_style_metadata_root,
        limit=args.limit,
    )
    if not items:
        raise SystemExit("No records found.")
    output_dir = resolve_cli_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = {}
    identity_fields = tuple(item_identity(items[0]))
    try:
        previous = read_json(output_dir / "clip_metrics.json")
        same_model = (resolve_cli_path(previous.get("model_path", ""), project_root).resolve()
                      == resolve_cli_path(args.model_path, project_root).resolve())
        if previous.get("version") == "style_transfer_clip_metrics" and same_model:
            for row in previous.get("records", []):
                if all(isinstance(row.get(field), (int, float)) and math.isfinite(row[field])
                       for field in ("clip_text_score", "clip_style_score")):
                    cached[tuple(row.get(field) for field in identity_fields)] = row
    except (OSError, ValueError, TypeError):
        pass
    identities = [tuple(item_identity(item).get(field) for field in identity_fields) for item in items]
    pending = sum(key not in cached for key in identities)
    if not args.quiet:
        print(f"CLIP: records={len(items)} cached_valid={len(items) - pending} pending={pending}", flush=True)
    scorer = PaperCLIPScorer(args.model_path, args.device) if pending else None
    rows = [cached[key] if key in cached else score_item(item, scorer)
            for item, key in zip(items, identities)]
    available = [row for row in rows if row["available"]]
    result = {
        "version": "style_transfer_clip_metrics",
        "config": {
            "project_root": str(project_root),
            "model_path": str(Path(args.model_path).expanduser()),
            "device": args.device or "auto",
            "canonical_style_metadata_root": (
                str(canonical_style_metadata_root) if canonical_style_metadata_root else ""
            ),
        },
        "inputs": {
            "mode": "summary",
            "summary_count": len(summary_paths),
            "summaries": [str(path) for path in summary_paths],
        },
        "metric": {
            "names": ["CLIP-Text", "CLIP-Style"],
            "definition": "normalized image-text cosine multiplied by CLIP logit_scale.exp()",
            "backend": (scorer.backend_name if scorer is not None else rows[0]["clip_text_backend"]),
        },
        "model_path": str(Path(args.model_path).expanduser()),
        # Keep the short alias for consumers of the initial release format.
        "canonical_style_metadata": (
            str(canonical_style_metadata_root) if canonical_style_metadata_root else ""
        ),
        "summary": {
            "total": len(rows),
            "available": len(available),
            "unavailable": len(rows) - len(available),
            "clip_text_score": stats(available, "clip_text_score"),
            "clip_style_score": stats(available, "clip_style_score"),
        },
        "records": rows,
    }
    json_path = output_dir / "clip_metrics.json"
    csv_path = output_dir / "clip_metrics.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if not args.quiet:
        print(f"Wrote {json_path}")
        print(f"Wrote {csv_path}")
    return 0 if all(row.get("clip_text_score") is not None and row.get("clip_style_score") is not None
                    for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
