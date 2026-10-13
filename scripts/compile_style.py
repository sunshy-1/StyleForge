#!/usr/bin/env python3
"""Compile strict reference-image style contracts into a portable style schema.

This script does not generate images. By default it also does not call a VLM: it reads
only detail_mechanism_contract.json files produced by the vision-language model
reference-image extraction step, drafts the style schema with one generic compiler
path, and audits that literal_reference_containers do not enter positive prompt
wording.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styleforge.style_schema import (
    DEFAULT_ACTUATOR_MODE,
    POSITIVE_MECHANISM_KEYS,
    REQUIRED_SCHEMA_KEYS,
    audit_style_schema,
    build_style_schema_from_contract,
    compile_style_schema,
    literal_reference_hits,
)
from styleforge.paths import repository_path


DEFAULT_DETAIL_CONTRACT_ROOT = ROOT / "outputs" / "extraction" / "contracts"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "style_schema"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.json"
SCRIPT_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "styleforge" / "style_schema.py"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _discover_tasks(contract_root: Path) -> List[str]:
    if not contract_root.exists():
        raise FileNotFoundError(contract_root)
    tasks = []
    for child in sorted(contract_root.iterdir()):
        if child.is_dir() and (child / "detail_mechanism_contract.json").exists():
            tasks.append(child.name)
    return tasks


def _contract_path(contract_root: Path, task_id: str) -> Path:
    path = contract_root / task_id / "detail_mechanism_contract.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_region_guided_color_light_allocation(
    palette_root: Path,
    task_id: str,
    palette_mode: str = "sampled",
) -> Tuple[List[Mapping[str, Any]], Path]:
    task_dir = palette_root / task_id
    mode = str(palette_mode or "sampled").strip().lower()
    if mode == "qa-gated":
        candidates = [
            task_dir / "qa_gated_color_light_allocation.json",
            task_dir / "palette_merge_decisions.json",
        ]
    elif mode == "sampled":
        candidates = [
            task_dir / "region_guided_color_light_allocation.json",
            task_dir / "region_guided_palette.json",
        ]
    else:
        raise ValueError(f"unsupported region-guided palette mode: {palette_mode}")
    for path in candidates:
        if not path.exists():
            continue
        data = _load_json(path)
        allocation = data.get("color_light_allocation")
        if not isinstance(allocation, list) or not allocation:
            raise ValueError(f"{task_id}: {mode} region-guided palette has no color_light_allocation: {path}")
        return [item for item in allocation if isinstance(item, Mapping)], path
    raise FileNotFoundError(f"{task_id}: no {mode} region-guided palette JSON under {palette_root}")


def _load_reference_color_distribution(palette_root: Path, task_id: str) -> Tuple[Dict[str, Any], Optional[Path]]:
    path = palette_root / task_id / "reference_color_distribution.json"
    if not path.exists():
        return {}, None
    data = _load_json(path)
    return data if isinstance(data, dict) else {}, path


def _reference_images_for_task(dataset_root: Path, task_id: str) -> List[Path]:
    target_dir = dataset_root / task_id / "target_images"
    if not target_dir.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _merge_missing_reference_distribution_fields(
    base: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (fallback or {}).items():
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def _reference_color_distribution_from_reference_images(paths: Sequence[Path], *, max_images: int = 8) -> Dict[str, Any]:
    pixels = []
    for path in list(paths)[:max_images]:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((256, 256))
        arr = np.asarray(image).astype("float32") / 255.0
        if arr.size:
            pixels.append(arr.reshape(-1, 3))
    if not pixels:
        return {}
    arr = np.concatenate(pixels, axis=0)
    red = arr[:, 0]
    green = arr[:, 1]
    blue = arr[:, 2]
    maxc = arr.max(axis=1)
    minc = arr.min(axis=1)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    sat = np.zeros_like(maxc)
    nonzero = maxc > 1e-6
    sat[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
    hue = np.zeros_like(maxc)
    delta = maxc - minc
    chroma = delta > 1e-6
    red_max = chroma & (maxc == red)
    green_max = chroma & (maxc == green)
    blue_max = chroma & (maxc == blue)
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0
    hue = (hue * 60.0) % 360.0
    chromatic = sat > 0.08
    chromatic_light = (luma > 0.62) & (sat > 0.12)
    luma_p05 = float(np.quantile(luma, 0.05))
    luma_p95 = float(np.quantile(luma, 0.95))

    quantized = np.clip((arr * 15.0).astype("int32"), 0, 15)
    codes = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    unique, counts = np.unique(codes, return_counts=True)
    order = np.argsort(counts)[::-1][:10]
    clusters = []
    total = float(arr.shape[0])
    for index in order:
        mask = codes == unique[index]
        rgb = arr[mask].mean(axis=0)
        r, g, b = [float(item) for item in rgb]
        mx = max(r, g, b)
        mn = min(r, g, b)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        saturation = (mx - mn) / mx if mx > 1e-6 else 0.0
        if mx == mn:
            hue_value = 0.0
        elif mx == r:
            hue_value = (((g - b) / (mx - mn)) % 6.0) * 60.0
        elif mx == g:
            hue_value = (((b - r) / (mx - mn)) + 2.0) * 60.0
        else:
            hue_value = (((r - g) / (mx - mn)) + 4.0) * 60.0
        clusters.append(
            {
                "hex": "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255)),
                "fraction": round(float(counts[index] / total), 6),
                "luma": round(float(lum), 6),
                "saturation": round(float(saturation), 6),
                "hue": round(float(hue_value % 360.0), 6),
            }
        )
    return {
        "source": "reference_image_color_distribution",
        "image_count": min(len(paths), max_images),
        "sampled_pixels": int(arr.shape[0]),
        "luma_mean": round(float(luma.mean()), 6),
        "luma_p05": round(luma_p05, 6),
        "luma_p95": round(luma_p95, 6),
        "luma_range_p95_p05": round(float(luma_p95 - luma_p05), 6),
        "saturation_mean": round(float(sat.mean()), 6),
        "high_saturation_fraction": round(float((sat > 0.45).mean()), 6),
        "strong_saturation_fraction": round(float((sat > 0.60).mean()), 6),
        "dark_fraction": round(float((luma < 0.18).mean()), 6),
        "light_fraction": round(float((luma > 0.82).mean()), 6),
        "chromatic_fraction": round(float(chromatic.mean()), 6),
        "chromatic_light_fraction": round(float(chromatic_light.mean()), 6),
        "clusters": clusters,
        "notes": [
            "reference-image pixel distribution, audit metadata only",
            "numeric values must not enter prompts",
        ],
    }


def _color_value_distribution_from_reference_distribution(distribution: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(distribution, Mapping) or not distribution:
        return {}
    dark = float(distribution.get("dark_fraction") or 0.0)
    light = float(distribution.get("light_fraction") or 0.0)
    sat_mean = float(distribution.get("saturation_mean") or 0.0)
    chromatic_fraction = float(distribution.get("chromatic_fraction") or 0.0)
    chromatic_light_fraction = float(distribution.get("chromatic_light_fraction") or 0.0)
    high_saturation_fraction = float(distribution.get("high_saturation_fraction") or 0.0)
    strong_saturation_fraction = float(distribution.get("strong_saturation_fraction") or 0.0)
    luma_range = float(distribution.get("luma_range_p95_p05") or 0.0)
    clusters = [item for item in distribution.get("clusters") or [] if isinstance(item, Mapping)]
    dark_cluster = sum(float(item.get("fraction") or 0.0) for item in clusters if float(item.get("luma") or 0.0) < 0.12)
    white_cluster = sum(
        float(item.get("fraction") or 0.0)
        for item in clusters
        if float(item.get("luma") or 0.0) > 0.78 and float(item.get("saturation") or 0.0) < 0.16
    )
    hue_family_weights: Dict[str, float] = {}
    light_hue_family_weights: Dict[str, float] = {}
    chromatic_cluster = 0.0
    chromatic_light_cluster = 0.0
    for item in clusters:
        fraction = float(item.get("fraction") or 0.0)
        luma = float(item.get("luma") or 0.0)
        saturation = float(item.get("saturation") or 0.0)
        if saturation >= 0.12 and luma >= 0.62 and fraction >= 0.015:
            family = _hue_family(float(item.get("hue") or 0.0))
            light_hue_family_weights[family] = light_hue_family_weights.get(family, 0.0) + fraction
            chromatic_light_cluster += fraction
        if saturation < 0.25 or luma < 0.10 or fraction < 0.015:
            continue
        family = _hue_family(float(item.get("hue") or 0.0))
        hue_family_weights[family] = hue_family_weights.get(family, 0.0) + fraction
        chromatic_cluster += fraction
    if dark >= 0.55 or dark_cluster >= 0.40:
        dark_field_role = "dominant dark/black field"
    elif dark >= 0.30 or dark_cluster >= 0.18:
        dark_field_role = "major dark/black field"
    elif dark > 0.12:
        dark_field_role = "local dark structure"
    else:
        dark_field_role = "limited dark structure"
    if light >= 0.18 or white_cluster >= 0.14:
        light_field_role = "broad light/white base field"
    elif light >= 0.04 or white_cluster >= 0.04:
        light_field_role = "selective light islands"
    else:
        light_field_role = "limited light islands"
    if sat_mean < 0.12 and chromatic_fraction < 0.25:
        chroma_key = "near-monochrome"
    elif sat_mean >= 0.30 or chromatic_cluster >= 0.14:
        chroma_key = "high-chroma"
    else:
        chroma_key = "moderate-chroma"
    if chroma_key == "near-monochrome":
        chromatic_role = "neutral-led"
    elif chromatic_cluster >= 0.10:
        chromatic_role = "large chromatic material fields"
    elif chromatic_cluster >= 0.035:
        chromatic_role = "small chromatic accents"
    else:
        chromatic_role = "neutral-led"
    if chroma_key == "near-monochrome":
        high_chroma_role = "restrained saturation"
    elif high_saturation_fraction >= 0.34 or strong_saturation_fraction >= 0.18 or chromatic_cluster >= 0.45:
        high_chroma_role = "dominant saturated carrier fields"
    elif high_saturation_fraction >= 0.16 or chromatic_cluster >= 0.14 or sat_mean >= 0.30:
        high_chroma_role = "visible saturated carrier fields"
    elif high_saturation_fraction >= 0.05:
        high_chroma_role = "local saturated accents"
    else:
        high_chroma_role = "restrained saturation"
    if chroma_key == "near-monochrome":
        chromatic_light_role = "limited colored light fields"
    elif chromatic_light_fraction >= 0.16 or chromatic_light_cluster >= 0.10:
        chromatic_light_role = "broad colored light fields"
    elif chromatic_light_fraction >= 0.06 or chromatic_light_cluster >= 0.04:
        chromatic_light_role = "selective colored light fields"
    else:
        chromatic_light_role = "limited colored light fields"
    ranked_families = [
        family
        for family, _weight in sorted(hue_family_weights.items(), key=lambda item: (-item[1], item[0]))
    ][:4]
    ranked_light_families = [
        family
        for family, _weight in sorted(light_hue_family_weights.items(), key=lambda item: (-item[1], item[0]))
    ][:4]
    contrast_key = ""
    if (dark >= 0.30 or dark_cluster >= 0.18) and (light >= 0.08 or white_cluster >= 0.08):
        contrast_key = "dark-light split"
    elif chroma_key == "near-monochrome" and (dark >= 0.45 or dark_cluster >= 0.25):
        contrast_key = "hard dark-light separation"
    if luma_range >= 0.70 or "dark-light" in contrast_key:
        value_contrast_role = "wide dark-light value range"
    elif luma_range and luma_range <= 0.42:
        value_contrast_role = "compressed soft value range"
    else:
        value_contrast_role = "moderate value range"
    return {
        "source": "reference_color_distribution",
        "value_key": "low-key" if dark >= 0.55 else ("high-key" if light >= 0.32 and dark < 0.12 else "mid-key"),
        "dark_field_role": dark_field_role,
        "light_field_role": light_field_role,
        "chroma_key": chroma_key,
        "chromatic_role": chromatic_role,
        "high_chroma_role": high_chroma_role,
        "chromatic_light_role": chromatic_light_role,
        "dominant_hue_families": ranked_families,
        "dominant_light_hue_families": ranked_light_families,
        "contrast_key": contrast_key,
        "value_contrast_role": value_contrast_role,
        "luma_mean": float(distribution.get("luma_mean") or 0.0),
        "luma_p05": float(distribution.get("luma_p05") or 0.0),
        "luma_p95": float(distribution.get("luma_p95") or 0.0),
        "luma_range_p95_p05": luma_range,
        "dark_fraction": dark,
        "light_fraction": light,
        "saturation_mean": sat_mean,
        "high_saturation_fraction": high_saturation_fraction,
        "strong_saturation_fraction": strong_saturation_fraction,
        "chromatic_fraction": chromatic_fraction,
        "chromatic_light_fraction": chromatic_light_fraction,
        "dark_cluster_fraction": round(float(dark_cluster), 6),
        "white_cluster_fraction": round(float(white_cluster), 6),
        "chromatic_cluster_fraction": round(float(chromatic_cluster), 6),
        "chromatic_light_cluster_fraction": round(float(chromatic_light_cluster), 6),
        "notes": [
            "derived from reference-image color distribution",
            "numeric values are audit metadata only and must not enter prompts",
        ],
    }


def _hue_family(hue: float) -> str:
    hue = hue % 360.0
    if hue < 15.0 or hue >= 345.0:
        return "red"
    if hue < 30.0:
        return "orange-red"
    if hue < 65.0:
        return "orange-gold"
    if hue < 85.0:
        return "yellow"
    if hue < 165.0:
        return "green"
    if hue < 205.0:
        return "cyan"
    if hue < 255.0:
        return "blue"
    if hue < 305.0:
        return "violet"
    return "magenta"


def _tonal_balance_from_reference_images(paths: Sequence[Path], *, max_images: int = 8) -> Dict[str, Any]:
    stats = []
    for path in list(paths)[:max_images]:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((256, 256))
        arr = np.asarray(image).astype("float32") / 255.0
        if arr.size == 0:
            continue
        red = arr[..., 0]
        green = arr[..., 1]
        blue = arr[..., 2]
        maxc = arr.max(axis=-1)
        minc = arr.min(axis=-1)
        luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        sat = np.zeros_like(maxc)
        nonzero = maxc > 1e-6
        sat[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
        stats.append(
            {
                "luma_mean": float(luma.mean()),
                "dark_fraction": float((luma < 0.18).mean()),
                "light_fraction": float((luma > 0.82).mean()),
                "midtone_fraction": float(((luma >= 0.18) & (luma <= 0.82)).mean()),
                "saturation_mean": float(sat.mean()),
            }
        )
    if not stats:
        return {}
    mean = {
        key: float(sum(item[key] for item in stats) / len(stats))
        for key in ("luma_mean", "dark_fraction", "light_fraction", "midtone_fraction", "saturation_mean")
    }
    dark = mean["dark_fraction"]
    light = mean["light_fraction"]
    mid = mean["midtone_fraction"]
    sat = mean["saturation_mean"]
    if dark >= 0.55:
        value_key = "low-key"
    elif light >= 0.38 and dark < 0.12:
        value_key = "high-key"
    else:
        value_key = "mid-key"
    return {
        **mean,
        "value_key": value_key,
        "shadow_extent": "large dark fields" if dark >= 0.45 else "limited dark structure",
        "highlight_extent": "broad light fields" if light >= 0.25 else "selective highlights",
        "midtone_extent": "broad midtone carriers" if mid >= 0.45 else "compressed midtones",
        "chroma_key": "low-chroma" if sat < 0.14 else ("high-chroma" if sat >= 0.35 else "moderate-chroma"),
        "notes": [
            "reference-image tonal statistics, converted to qualitative prompt wording",
            "numeric values are audit metadata only and must not enter prompts",
        ],
    }


def _build_record(
    *,
    task_id: str,
    detail_contract_path: Path,
    dataset_root: Path,
    enable_tonal_balance: bool,
    region_guided_palette_root: Optional[Path],
    region_guided_palette_mode: str,
    max_chars: int,
    actuator_mode: str = DEFAULT_ACTUATOR_MODE,
) -> Dict[str, Any]:
    detail_contract = _load_json(detail_contract_path)
    ref_images = _reference_images_for_task(dataset_root, task_id) if enable_tonal_balance else []
    tonal_balance = _tonal_balance_from_reference_images(ref_images) if enable_tonal_balance else {}
    source_files: Dict[str, str] = {"detail_contract": repository_path(detail_contract_path)}
    color_light_allocation_override = None
    palette_source_path = None
    reference_distribution: Dict[str, Any] = {}
    reference_distribution_path: Optional[Path] = None
    if region_guided_palette_root is not None:
        color_light_allocation_override, palette_source_path = _load_region_guided_color_light_allocation(
            region_guided_palette_root,
            task_id,
            palette_mode=region_guided_palette_mode,
        )
        source_files["region_guided_palette"] = repository_path(palette_source_path)
        reference_distribution, reference_distribution_path = _load_reference_color_distribution(
            region_guided_palette_root,
            task_id,
        )
        if reference_distribution_path is not None:
            source_files["reference_color_distribution"] = repository_path(reference_distribution_path)
    dataset_reference_distribution = _reference_color_distribution_from_reference_images(
        _reference_images_for_task(dataset_root, task_id)
    )
    if reference_distribution:
        reference_distribution = _merge_missing_reference_distribution_fields(
            reference_distribution,
            dataset_reference_distribution,
        )
    else:
        reference_distribution = dataset_reference_distribution
    color_value_distribution = _color_value_distribution_from_reference_distribution(reference_distribution)

    schema = build_style_schema_from_contract(
        task_id=task_id,
        detail_contract=detail_contract,
        color_light_allocation_override=color_light_allocation_override,
        color_value_distribution=color_value_distribution,
        tonal_balance=tonal_balance,
        source_files=source_files,
    )
    compiled = compile_style_schema(schema, max_chars=max_chars, actuator_mode=actuator_mode)
    audit = audit_style_schema(schema, compiled, max_chars=max_chars)
    compiled_dict = compiled.to_dict()

    return {
        "task_id": task_id,
        "actuator_mode": compiled.actuator_mode,
        "schema": schema.to_dict(),
        "compiled": compiled_dict,
        "audit": audit,
        "source_files": schema.source_files,
        "reference_image_count_for_tonal_balance": len(ref_images),
        "reference_color_distribution_source": repository_path(reference_distribution_path) if reference_distribution_path else "dataset_reference_images",
        "color_value_distribution": color_value_distribution,
        "region_guided_palette_source": repository_path(palette_source_path) if palette_source_path else "",
        "region_guided_palette_mode": str(region_guided_palette_mode or ""),
    }


def _write_task_record(record: Mapping[str, Any], output_root: Path) -> None:
    task_dir = output_root / str(record["task_id"])
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_json(task_dir / "style_schema.json", record)
    compiled = record.get("compiled") or {}
    (task_dir / "compiled_positive_prompt.txt").write_text(
        str(compiled.get("positive_prompt") or "").rstrip() + "\n",
        encoding="utf-8",
    )
    (task_dir / "compiled_generation_style_prompt.txt").write_text(
        str(compiled.get("generation_style_prompt") or "").rstrip() + "\n",
        encoding="utf-8",
    )


def _source_genericity_errors(task_ids: Sequence[str]) -> List[str]:
    errors: List[str] = []
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (SCRIPT_PATH, MODULE_PATH))
    for task_id in task_ids:
        if task_id and task_id in source_text:
            errors.append(f"task id literal appears in compiler/script source: {task_id}")
    return errors


def _source_file_errors(records: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    forbidden_path_bits = ("contact_sheet", "/generated/", "\\generated\\")
    for record in records:
        task_id = str(record.get("task_id") or "")
        source_files = record.get("source_files") or {}
        if not isinstance(source_files, Mapping):
            errors.append(f"{task_id}: source_files is not a mapping")
            continue
        for label, raw in source_files.items():
            path = Path(str(raw))
            lowered = str(path).lower()
            if path.suffix.lower() in image_suffixes:
                errors.append(f"{task_id}: source file {label} points to image data: {path}")
            if any(bit in lowered for bit in forbidden_path_bits):
                errors.append(f"{task_id}: source file {label} points to generated/review output: {path}")
    return errors


def _audit_batch(records: Sequence[Mapping[str, Any]], task_ids: Sequence[str]) -> Dict[str, Any]:
    errors: List[str] = []
    per_task_ok = {}
    for record in records:
        task_id = str(record.get("task_id") or "")
        audit = record.get("audit") or {}
        per_task_ok[task_id] = bool(audit.get("ok"))
        for err in audit.get("errors") or []:
            errors.append(f"{task_id}: {err}")

    schema_keys = None
    positive_keys = None
    for record in records:
        schema = record.get("schema") or {}
        keys = tuple(sorted(schema.keys())) if isinstance(schema, Mapping) else ()
        positive = schema.get("positive_style_mechanisms") if isinstance(schema, Mapping) else {}
        pkeys = tuple(sorted(positive.keys())) if isinstance(positive, Mapping) else ()
        if schema_keys is None:
            schema_keys = keys
            positive_keys = pkeys
        elif keys != schema_keys or pkeys != positive_keys:
            errors.append(f"{record.get('task_id')}: schema key mismatch")

    errors.extend(_source_genericity_errors(task_ids))
    errors.extend(_source_file_errors(records))

    literal_hit_counts = {
        str(record.get("task_id") or ""): sum(
            len((record.get("audit") or {}).get("prompt_literal_hits", {}).get(label, []) or [])
            for label in ("positive_prompt", "generation_style_prompt")
        )
        for record in records
    }

    return {
        "ok": not errors,
        "errors": errors,
        "rules": [
            "same style-schema keys for every audited style",
            "same positive_style_mechanisms keys for every audited style",
            "literal_reference_containers do not enter compiled positive prompt",
            "soft_composition_tendencies enter only through optional weak wording",
            "compile-only audit: no image generation, no VLM call",
            "source files are detail_mechanism_contract.json and optional region-guided palette JSON from reference-only pixel sampling",
            "compiler and script contain no task-id-specific prompt branches",
        ],
        "task_ids": list(task_ids),
        "per_task_ok": per_task_ok,
        "schema_keys": list(schema_keys or REQUIRED_SCHEMA_KEYS),
        "positive_mechanism_keys": list(positive_keys or POSITIVE_MECHANISM_KEYS),
        "literal_hit_counts": literal_hit_counts,
    }


def _write_html(records: Sequence[Mapping[str, Any]], audit: Mapping[str, Any], output_root: Path) -> None:
    sections = []
    for record in records:
        schema = record.get("schema") or {}
        compiled = record.get("compiled") or {}
        sections.append(
            f"""
            <section>
              <h2>{html.escape(str(record.get("task_id", "")))}</h2>
              <h3>Compiled Positive Prompt</h3>
              <pre>{html.escape(str(compiled.get("positive_prompt", "")))}</pre>
              <h3>Compiled Generation Style Prompt</h3>
              <pre>{html.escape(str(compiled.get("generation_style_prompt", "")))}</pre>
              <h3>literal_reference_containers</h3>
              <pre>{html.escape(json.dumps(schema.get("literal_reference_containers", []), ensure_ascii=False, indent=2))}</pre>
              <h3>Per Task Audit</h3>
              <pre>{html.escape(json.dumps(record.get("audit", {}), ensure_ascii=False, indent=2))}</pre>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Style Schema Prompt Audit</title>
<style>
body{{margin:0;background:#f6f7f9;color:#161b22;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:#111827;color:white;padding:20px 28px}}
main{{padding:18px 28px 48px}}
section{{background:white;border:1px solid #d8dee4;border-radius:8px;padding:14px;margin:0 0 18px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f3f4f6;padding:12px;border-radius:6px}}
h2{{margin-top:0}}
</style>
</head>
<body>
    <header><h1>Style Schema Prompt Audit</h1><p>Compile-only audit. No image generation. Literal reference containers are audit-only.</p></header>
<main>
<section><h2>Batch Audit</h2><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></section>
{''.join(sections)}
</main>
</body>
</html>
"""
    (output_root / "index.html").write_text(html_text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--detail-contract-root", type=Path, default=DEFAULT_DETAIL_CONTRACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "styles")
    parser.add_argument(
        "--region-guided-palette-root",
        type=Path,
        default=None,
        help="Optional root containing per-task region_guided_color_light_allocation.json files.",
    )
    parser.add_argument(
        "--region-guided-palette-mode",
        choices=("sampled", "qa-gated"),
        default="sampled",
        help="Palette file to read when --region-guided-palette-root is set. Default keeps existing sampled behavior.",
    )
    parser.add_argument(
        "--enable-tonal-balance",
        action="store_true",
        help="Optionally derive qualitative tonal-balance wording from reference images. Default off.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Task ids to audit. Defaults to all contracts in the root.")
    parser.add_argument("--max-chars", type=int, default=1300)
    parser.add_argument(
        "--actuator-mode",
        default=DEFAULT_ACTUATOR_MODE,
        help=(
            "Generic compiler actuator mode. Supported current smoke modes include "
            "t4_0_compact_carrier_area, t4_1_perceptual_hue_carrier, "
            "t4_2_scene_slot_carrier_binding, t4_3_perceptual_carrier_anchor, "
            "t4_4_bright_scene_recast, t4_7_bright_chroma_carrier, "
            "t4_8_role_metadata, and t4_9_field_precedence."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    task_ids = list(args.tasks) if args.tasks else _discover_tasks(args.detail_contract_root)
    if not task_ids:
        raise ValueError("no tasks selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for task_id in task_ids:
        record = _build_record(
            task_id=task_id,
            detail_contract_path=_contract_path(args.detail_contract_root, task_id),
            dataset_root=args.dataset_root,
            enable_tonal_balance=bool(args.enable_tonal_balance),
            region_guided_palette_root=args.region_guided_palette_root,
            region_guided_palette_mode=str(args.region_guided_palette_mode),
            max_chars=int(args.max_chars),
            actuator_mode=str(args.actuator_mode),
        )
        records.append(record)
        _write_task_record(record, args.output_root)

    audit = _audit_batch(records, task_ids)
    summary = {
        "mode": "style_schema_compile",
        "config": repository_path(args.config),
        "detail_contract_root": repository_path(args.detail_contract_root),
        "region_guided_palette_root": repository_path(args.region_guided_palette_root) if args.region_guided_palette_root else "",
        "region_guided_palette_mode": str(args.region_guided_palette_mode),
        "actuator_mode": str(args.actuator_mode),
        "output_root": repository_path(args.output_root),
        "records": records,
        "audit": audit,
    }
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "audit.json", audit)
    _write_html(records, audit, args.output_root)

    print(f"ALL_DONE {args.output_root / 'summary.json'}")
    print(f"AUDIT {args.output_root / 'audit.json'} ok={audit['ok']}")
    print(f"HTML {args.output_root / 'index.html'}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
