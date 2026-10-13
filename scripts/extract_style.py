#!/usr/bin/env python3
"""Extract reference-only style mechanism contracts.

Anti-overfit policy:

- only attached reference images and universal schema/instructions are used;
- no generated images, contact sheets, legacy prompts, color contracts, or
  human failure notes are used while extracting the contract;
- no task-id-specific branches, thresholds, or rescue wording;
- selected task ids are runtime arguments only;
- prompt-ready contract text is audited before downstream compilation.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from styleforge.config import StyleForgeConfig
from styleforge.dataset import StyleTask, discover_tasks
from styleforge.paths import repository_path, resolve_repository_path
from styleforge.vlm_client import VisionLanguageModelClient, _content_for_prompt


CONFIG_PATH = ROOT / "configs" / "default.json"
OUTPUT_ROOT = ROOT / "outputs" / "extraction"
DEFAULT_SCENE_KEY = "strict_reference_only_open_clearing"
DEFAULT_CONTENT_PROMPT = "a full-body person raising a glowing lantern in an open clearing at sunrise, wide view"
DEFAULT_SEED = 1234
MAX_DETAIL_CHARS = 900

COLOR_LIGHT_ALLOCATION_VALUES = {
    "dominant field",
    "large material color",
    "large material-light",
    "medium light",
    "structural dark",
    "tiny accent",
    "local material color",
}

COLOR_LIGHT_ALLOCATION_REQUIRED_KEYS = {
    "allocation",
    "label",
    "hexes",
    "carrier",
    "carrier_scale",
    "carrier_geometry",
    "light_role",
    "material_vs_glow_boundary",
    "evidence_across_references",
    "generator_safe_wording",
}

REGION_SAMPLING_PLAN_REQUIRED_KEYS = {
    "allocation",
    "label",
    "reference_index",
    "bbox_norm",
    "sample_policy",
    "transferable_carrier",
    "avoid_literal_content",
}

MECHANISM_TRANSFER_REQUIRED_KEYS = {
    "construction_rule",
    "carrier_binding_rule",
    "content_recasting_rule",
    "application_scope",
}

HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")

CONTRACT_KEYS = [
    "medium_mark_system",
    "line_edge_language",
    "shape_proportion_language",
    "surface_artifact",
    "detail_density_policy",
    "composition_depth_prior",
    "color_carrier_binding",
    "color_light_allocation",
    "portable_vs_literal_carrier_policy",
    "literalization_risks",
    "generator_safe_wording",
]

PROMPT_READY_KEYS = {
    "generator_safe_wording",
    "compiled_prompt_note",
}

NEGATIVE_OR_AUDIT_KEYS = {
    "color_light_allocation",
    "region_sampling_plan",
    "literalization_risks",
    "reference_content_not_to_copy",
    "validation_questions",
    "source_policy",
}

STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "along",
    "also",
    "around",
    "away",
    "before",
    "behind",
    "below",
    "beside",
    "between",
    "content",
    "every",
    "from",
    "front",
    "image",
    "into",
    "near",
    "object",
    "objects",
    "open",
    "person",
    "scene",
    "small",
    "style",
    "subject",
    "their",
    "there",
    "these",
    "those",
    "through",
    "under",
    "using",
    "with",
    "without",
}

BASE_STYLE_CONTROL_WORDS = {
    "art",
    "gold",
    "lighting",
    "neon",
    "palette",
    "paper",
    "photographic",
    "style",
    "texture",
}

LITERAL_CONTAINER_TERMS = {
    "border",
    "borders",
    "container",
    "containers",
    "frame",
    "framed",
    "frames",
    "framing",
    "portal",
    "portals",
    "room",
    "rooms",
    "stage",
    "staged",
    "window",
    "windows",
}

CONTAINER_GUARD_REPLACEMENTS = [
    (
        re.compile(r"\barched\s+or\s+circular\s+framing\b", re.IGNORECASE),
        "curvilinear visual rhythm",
    ),
    (
        re.compile(r"\barched\s+or\s+circular\s+rhythm\b", re.IGNORECASE),
        "curvilinear ornamental rhythm",
    ),
    (
        re.compile(r"\barched\s+and\s+circular\s+geometries\b", re.IGNORECASE),
        "curvilinear ornamental geometries",
    ),
    (
        re.compile(r"\bcircular\s+framing\b", re.IGNORECASE),
        "curved visual organization",
    ),
    (
        re.compile(r"\bpatterned\s+borders\s+that\s+lead\s+the\s+eye\s+inward\b", re.IGNORECASE),
        "delicate ornamental linework that guides the eye",
    ),
    (
        re.compile(r"\bpatterned\s+borders\b", re.IGNORECASE),
        "delicate ornamental linework",
    ),
    (
        re.compile(r"\bborders\b", re.IGNORECASE),
        "edge accents",
    ),
    (
        re.compile(r"\bborder\b", re.IGNORECASE),
        "edge accent",
    ),
    (
        re.compile(r"\bframes\b", re.IGNORECASE),
        "edge accents",
    ),
    (
        re.compile(r"\bframe\b", re.IGNORECASE),
        "edge accent",
    ),
    (
        re.compile(r"\bframing\b", re.IGNORECASE),
        "visual organization",
    ),
    (
        re.compile(r"\bcontainer(s)?\b", re.IGNORECASE),
        "literal scene enclosure",
    ),
    (
        re.compile(r"\bportal(s)?\b", re.IGNORECASE),
        "curved visual opening",
    ),
    (
        re.compile(r"\bstage-like\b", re.IGNORECASE),
        "shallow poster-like",
    ),
]

def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(resolve_repository_path(path), base)
    except ValueError:
        return str(path)


def _parse_task_ids(value: str) -> List[str]:
    task_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not task_ids:
        raise ValueError("--tasks must list at least one task id")
    return task_ids


def _tokenize(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{3,}", text.lower())
        if token not in STOPWORDS
    }


def _instruction(
    *,
    task_id: str,
    max_detail_chars: int,
) -> str:
    source_policy = "reference images only; universal schema/instructions only; no legacy prompt, color contract, generated images, or failure notes"
    input_context_section = (
        "Input package:\n"
        "- Style evidence: attached reference images only.\n"
        "- Estimate palette anchors, visual carriers, material/light roles, and all mechanism wording directly from those images.\n"
        "- No legacy prompt, precomputed color contract, generated-image feedback, failure note, or human review note is provided.\n\n"
    )
    color_allocation_rule = (
        "- The agent, not the downstream compiler, must extract color/light allocation from the attached reference images. "
        "Estimate qualitative hex anchors from visible reference colors; do not use numeric percentages.\n"
    )
    hex_source = "exact #RRGGBB anchors estimated from the attached reference images only"
    return (
        "You are designing a minimal detail-mechanism contract for a style-transfer pipeline.\n\n"
        "Goal: from the allowed inputs only, extract portable visual mechanisms for a later schema compiler. "
        "The output must improve transfer to arbitrary future content without copying reference content.\n\n"
        "Strict source policy:\n"
        f"- Use only this source policy: {source_policy}.\n"
        "- Do not use generated images, previous failures, contact sheets, or human review notes.\n"
        "- Do not infer rules from the task id; it is only an identifier.\n"
        f"- Task identifier for JSON bookkeeping only: {task_id}.\n"
        "- Use the same schema for every style.\n"
        "- Do not write task-specific rescue text.\n\n"
        "Important portability rule:\n"
        "- Separate portable style mechanisms from literal reference containers or scene content.\n"
        "- If the references repeat a compositional container, border, stage, window, panel, screen, room setup, prop, or subject family, "
        "decide whether it is a reusable visual mechanism, a soft composition tendency, or literal content that must not be forced into new scenes.\n"
        "- Prompt-ready wording should prefer transferable behavior: mark-making, line language, shape/proportion language, surface artifacts, "
        "detail density, depth treatment, and color-carrier binding.\n"
        + color_allocation_rule
        + "- For each color/light allocation, separate tangible material carriers from light-only glow, shadow structure, and tiny glints.\n"
        "- For important color/light allocations, also propose reference-image sampling regions for downstream deterministic pixel sampling. "
        "Use normalized bbox coordinates [x0,y0,x1,y1] from 0 to 1. "
        "The region plan chooses where a tool should sample real pixels; it is audit metadata, not prompt-ready wording.\n"
        "- When the references show broad, thick, continuous, layered, repeated, or radiating material carriers, capture that as carrier scale "
        "and carrier geometry instead of leaving it as generic glow or atmosphere.\n"
        "- Identify whether large material colors can plausibly attach to subject-worn surfaces, object surfaces, broad scene fields, or only local materials.\n"
        "- Extract how arbitrary requested content should be visually rebuilt through the reference material/process system. Preserve future content identity, "
        "but make content material words subordinate when they conflict with the extracted style mechanism.\n"
        "- Avoid forcing every future content prompt into the same literal container, object family, pose, location, or scene layout.\n\n"
        + input_context_section
        + "Required JSON schema:\n"
        "{\n"
        '  "task_id": "exact task identifier above",\n'
        f'  "source_policy": "{source_policy}",\n'
        '  "detail_mechanism_summary": "one concise reference-derived summary",\n'
        '  "contract": {\n'
        '    "medium_mark_system": "portable medium, mark, rendering, or material behavior",\n'
        '    "line_edge_language": "portable line, contour, boundary, silhouette, and edge behavior",\n'
        '    "shape_proportion_language": "portable shape simplification, anatomy/object proportion, abstraction, and form language",\n'
        '    "surface_artifact": "portable grain, stroke, artifact, shimmer, paper, scan, cut, brush, or texture behavior",\n'
        '    "detail_density_policy": "where detail is dense, sparse, repeated, or deliberately simplified",\n'
        '    "composition_depth_prior": "portable depth, camera, staging, spacing, and layout tendency without fixed content",\n'
        '    "color_carrier_binding": "how color should attach to visual carriers, using ordinal role language only",\n'
        '    "mechanism_transfer": {\n'
        '      "construction_rule": "how the reference style visibly constructs subjects, objects, and background carriers",\n'
        '      "carrier_binding_rule": "how extracted material/color/light roles stay bound to carrier scale and geometry instead of becoming generic object color",\n'
        '      "content_recasting_rule": "how future requested content keeps identity/action while surfaces, edges, depth, and light are rebuilt through extracted carriers",\n'
        '      "application_scope": "where the mechanism must apply: subjects, objects, background planes, shadows, highlights, negative spaces, or local details; no percentages"\n'
        '    },\n'
        '    "color_light_allocation": [\n'
        '      {\n'
        '        "allocation": "one of: dominant field, large material color, large material-light, medium light, structural dark, tiny accent, local material color",\n'
        '        "label": "reference-derived color/light role label, no subject nouns",\n'
        f'        "hexes": ["{hex_source}"],\n'
        '        "carrier": "portable visual carrier: subject-worn material surfaces, object material surfaces, broad scene fields, radiating material bands, backglow, reflected highlights, linework, paper base, silhouette clarity, contact shadows, glints, etc.",\n'
        '        "carrier_scale": "qualitative scale/thickness/area such as broad thick continuous, medium, fine thin, tiny, or local mixed; no numbers",\n'
        '        "carrier_geometry": "portable distribution geometry such as continuous bands, layered bands, repeated stripes, radial flow, flat fields, linework, glow field, contact shadow, or glints; no content nouns",\n'
        '        "light_role": "material color, material light, secondary light, shadow structure, accent sparkle, or local material color",\n'
        '        "material_vs_glow_boundary": "how to keep tangible material color/material-light distinct from light-only glow, shadow, or glint roles",\n'
        '        "evidence_across_references": "brief reference-only evidence for this allocation",\n'
        '        "generator_safe_wording": "short content-agnostic wording for this allocation, no numeric percentages"\n'
        '      }\n'
        '    ],\n'
        '    "region_sampling_plan": [\n'
        '      {\n'
        '        "allocation": "one of the same allocation values used in color_light_allocation",\n'
        '        "label": "matches or refines a transferable color/light role label, no subject nouns",\n'
        '        "reference_index": "integer 0-3 selecting one attached reference image",\n'
        '        "bbox_norm": [0.0, 0.0, 1.0, 1.0],\n'
        '        "sample_policy": "what pixels to sample or exclude inside the box, expressed as portable carrier criteria, no percentages",\n'
        '        "transferable_carrier": "portable carrier represented by this region, not the literal object name",\n'
        '        "avoid_literal_content": "literal subject/container in the box that must not be transferred"\n'
        '      }\n'
        '    ],\n'
        '    "portable_vs_literal_carrier_policy": "which repeated visual containers or scene structures transfer as mechanisms, and which must not be forced literally",\n'
        '    "literalization_risks": ["generic risks where prompt wording could copy content instead of style"],\n'
        '    "generator_safe_wording": "compact prompt-ready instruction, no bullets, no reference subjects, no eval content nouns"\n'
        "  },\n"
        '  "reference_content_not_to_copy": ["generic names of content families or containers that are visible but should not be required"],\n'
        '  "validation_questions": ["short visual review questions"],\n'
        f'  "compiled_prompt_note": "one compact prompt-ready paragraph <= {int(max_detail_chars)} chars"\n'
        "}\n"
    )


def _validate_contract(data: Mapping[str, Any], *, task_id: str, max_detail_chars: int) -> None:
    if str(data.get("task_id") or "") != task_id:
        raise ValueError(f"task_id mismatch: {data.get('task_id')} != {task_id}")
    source_policy = str(data.get("source_policy") or "").strip()
    if not source_policy:
        raise ValueError("source_policy is required")
    contract = data.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("missing contract object")
    for key in CONTRACT_KEYS:
        value = contract.get(key)
        if key == "literalization_risks":
            if not isinstance(value, list) or not value:
                raise ValueError(f"contract.{key} must be a non-empty list")
        elif key == "color_light_allocation":
            _validate_color_light_allocation(value)
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"contract.{key} must be a non-empty string")
    if "region_sampling_plan" in contract:
        _validate_region_sampling_plan(contract.get("region_sampling_plan"))
    if "mechanism_transfer" in contract:
        _validate_mechanism_transfer(contract.get("mechanism_transfer"))
    note = data.get("compiled_prompt_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("missing compiled_prompt_note")
    if len(note) > max_detail_chars:
        raise ValueError(f"compiled_prompt_note too long: {len(note)} > {max_detail_chars}")


def _validate_color_light_allocation(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("contract.color_light_allocation must be a non-empty list")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"contract.color_light_allocation[{index}] must be an object")
        missing = sorted(key for key in COLOR_LIGHT_ALLOCATION_REQUIRED_KEYS if key not in item)
        if missing:
            raise ValueError(f"contract.color_light_allocation[{index}] missing keys: {', '.join(missing)}")
        allocation = str(item.get("allocation") or "").strip()
        if allocation not in COLOR_LIGHT_ALLOCATION_VALUES:
            raise ValueError(f"contract.color_light_allocation[{index}].allocation is invalid: {allocation}")
        hexes = item.get("hexes")
        if not isinstance(hexes, list) or not hexes:
            raise ValueError(f"contract.color_light_allocation[{index}].hexes must be a non-empty list")
        for hex_value in hexes:
            if not isinstance(hex_value, str) or not HEX_RE.fullmatch(hex_value.strip()):
                raise ValueError(f"contract.color_light_allocation[{index}].hexes contains invalid hex: {hex_value}")
        for key in (
            "label",
            "carrier",
            "carrier_scale",
            "carrier_geometry",
            "light_role",
            "material_vs_glow_boundary",
            "evidence_across_references",
            "generator_safe_wording",
        ):
            text = str(item.get(key) or "").strip()
            if not text:
                raise ValueError(f"contract.color_light_allocation[{index}].{key} must be non-empty")
            if "0." in text or re.search(r"\b\d+\s*%", text):
                raise ValueError(f"contract.color_light_allocation[{index}].{key} contains numeric ratio/percentage")


def _validate_region_sampling_plan(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("contract.region_sampling_plan must be a non-empty list")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"contract.region_sampling_plan[{index}] must be an object")
        missing = sorted(key for key in REGION_SAMPLING_PLAN_REQUIRED_KEYS if key not in item)
        if missing:
            raise ValueError(f"contract.region_sampling_plan[{index}] missing keys: {', '.join(missing)}")
        allocation = str(item.get("allocation") or "").strip()
        if allocation not in COLOR_LIGHT_ALLOCATION_VALUES:
            raise ValueError(f"contract.region_sampling_plan[{index}].allocation is invalid: {allocation}")
        ref_index = item.get("reference_index")
        if not isinstance(ref_index, int) or not 0 <= ref_index <= 3:
            raise ValueError(f"contract.region_sampling_plan[{index}].reference_index must be an integer 0-3")
        bbox = item.get("bbox_norm")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"contract.region_sampling_plan[{index}].bbox_norm must be [x0,y0,x1,y1]")
        try:
            x0, y0, x1, y1 = [float(coord) for coord in bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"contract.region_sampling_plan[{index}].bbox_norm must contain numbers") from exc
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"contract.region_sampling_plan[{index}].bbox_norm must be normalized and ordered")
        for key in ("label", "sample_policy", "transferable_carrier", "avoid_literal_content"):
            text = str(item.get(key) or "").strip()
            if not text:
                raise ValueError(f"contract.region_sampling_plan[{index}].{key} must be non-empty")
            if "0." in text or re.search(r"\b\d+\s*%", text):
                raise ValueError(f"contract.region_sampling_plan[{index}].{key} contains numeric ratio/percentage")


def _validate_mechanism_transfer(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("contract.mechanism_transfer must be an object")
    missing = sorted(key for key in MECHANISM_TRANSFER_REQUIRED_KEYS if key not in value)
    if missing:
        raise ValueError(f"contract.mechanism_transfer missing keys: {', '.join(missing)}")
    for key in sorted(MECHANISM_TRANSFER_REQUIRED_KEYS):
        text = str(value.get(key) or "").strip()
        if not text:
            raise ValueError(f"contract.mechanism_transfer.{key} must be non-empty")
        if "0." in text or re.search(r"\b\d+\s*%", text):
            raise ValueError(f"contract.mechanism_transfer.{key} contains numeric ratio/percentage")


def _call_vlm_contract(
    *,
    client: VisionLanguageModelClient,
    config: StyleForgeConfig,
    task: StyleTask,
    max_detail_chars: int,
) -> Dict[str, Any]:
    prompt = _instruction(
        task_id=task.task_id,
        max_detail_chars=max_detail_chars,
    )
    content = _content_for_prompt(
        prompt_text=prompt,
        images=list(task.ref_images[:4]),
        labels=[f"reference image {idx + 1}" for idx in range(min(4, len(task.ref_images)))],
        image_max_side=config.backend.vlm_image_max_side,
        image_quality=config.backend.vlm_image_quality,
    )
    last_error: Optional[Exception] = None
    for attempt in range(max(1, config.max_json_retries)):
        try:
            data, meta = client.chat_json(content, cache_tag=f"detail_mechanism_min_{task.task_id}_{attempt}")
            _validate_contract(data, task_id=task.task_id, max_detail_chars=max_detail_chars)
            data["_vlm_meta"] = meta
            return data
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            print(
                f"[contract] validation failed for {task.task_id} "
                f"(attempt {attempt + 1}/{max(1, config.max_json_retries)}): {exc}",
                flush=True,
            )
            content = list(content) + [
                {
                    "type": "text",
                    "text": (
                        "The previous response was invalid. Return one JSON object only, exactly following the schema. "
                        f"Validation error: {exc}"
                    ),
                }
            ]
    raise RuntimeError(f"detail-mechanism contract failed for {task.task_id}: {last_error}") from last_error


def _prompt_ready_strings(value: Any, *, key: str = "") -> Iterable[str]:
    if key in NEGATIVE_OR_AUDIT_KEYS or key.startswith("_"):
        return
    if isinstance(value, str) and key in PROMPT_READY_KEYS:
        yield value
    elif isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _prompt_ready_strings(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _prompt_ready_strings(child)


def _audit_contract(
    *,
    contract: Mapping[str, Any],
    task_id: str,
    content_prompt: str,
) -> List[str]:
    errors: List[str] = []
    note = str(contract.get("compiled_prompt_note") or "")
    text = "\n".join(_prompt_ready_strings(contract))
    eval_tokens = _tokenize(content_prompt)
    text_tokens = _tokenize(text)
    leaked = sorted(token for token in eval_tokens & text_tokens if token not in {"window"})
    for token in leaked:
        errors.append(f"prompt-ready text contains eval content token: {token}")
    if task_id.lower() in text.lower():
        errors.append("prompt-ready text contains task id")
    if len(note) > MAX_DETAIL_CHARS:
        errors.append(f"compiled_prompt_note too long: {len(note)}")
    if "0." in text or re.search(r"\b\d+\s*%", text):
        errors.append("prompt-ready text contains numeric ratio/percentage")
    if len(note.split()) < 24:
        errors.append("compiled_prompt_note is too short to carry detail mechanisms")
    return errors


def _style_control_words_from_task_ids(task_ids: Sequence[str]) -> Set[str]:
    words = set(BASE_STYLE_CONTROL_WORDS)
    for task_id in task_ids:
        for token in re.split(r"[^A-Za-z0-9]+", str(task_id).lower()):
            if not token or token in {"style", "megastyle", "style30k"}:
                continue
            if len(token) >= 4 or token in {"2d", "3d"}:
                words.add(token)
    return words


def _audit_content_prompt(content_prompt: str, *, task_ids: Sequence[str] = ()) -> List[str]:
    errors = []
    lowered = content_prompt.lower()
    for term in sorted(_style_control_words_from_task_ids(task_ids)):
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered):
            errors.append(f"content prompt contains style/control word: {term}")
    if re.search(r"#[0-9a-fA-F]{6}", content_prompt):
        errors.append("content prompt contains hex color")
    return errors


def _literal_container_risk(contract: Mapping[str, Any]) -> bool:
    values: List[str] = []
    inner = contract.get("contract")
    if isinstance(inner, Mapping):
        risks = inner.get("literalization_risks")
        if isinstance(risks, list):
            values.extend(str(item) for item in risks)
        policy = inner.get("portable_vs_literal_carrier_policy")
        if isinstance(policy, str):
            values.append(policy)
    ref_content = contract.get("reference_content_not_to_copy")
    if isinstance(ref_content, list):
        values.extend(str(item) for item in ref_content)
    tokens = _tokenize(" ".join(values))
    return bool(tokens & LITERAL_CONTAINER_TERMS)


def _apply_container_guard(text: str) -> str:
    guarded = text
    for pattern, replacement in CONTAINER_GUARD_REPLACEMENTS:
        guarded = pattern.sub(replacement, guarded)
    guarded = re.sub(r"\s+", " ", guarded).strip()
    guarded = re.sub(r"\s+([,.;:])", r"\1", guarded)
    return guarded


def _compiled_detail_note(contract: Mapping[str, Any], *, container_guard: bool) -> str:
    if container_guard and _literal_container_risk(contract):
        inner = contract.get("contract")
        if isinstance(inner, Mapping):
            safe = str(inner.get("generator_safe_wording") or "").strip()
            if safe:
                return _apply_container_guard(safe)
    return str(contract.get("compiled_prompt_note") or "").strip()


def _write_html(records: Sequence[Mapping[str, Any]], output_root: Path, *, audit: Mapping[str, Any]) -> None:
    sections = []
    for record in records:
        task_id = str(record["task_id"])
        refs = "".join(
            f'<img src="{html.escape(_rel(Path(path), output_root))}" loading="lazy" alt="ref">'
            for path in record.get("ref_images", [])[:4]
        )
        sections.append(
            f"""
            <section id="{html.escape(task_id)}">
              <h2>{html.escape(task_id)}</h2>
              <p class="meta">contract chars={len(str(record.get("detail_note") or ""))}</p>
              <div class="refs">{refs}</div>
              <details open><summary>Detail-mechanism note</summary><pre>{html.escape(str(record.get("detail_note") or ""))}</pre></details>
              <details><summary>Contract JSON</summary><pre>{html.escape(json.dumps(record.get("contract") or {}, ensure_ascii=False, indent=2))}</pre></details>
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reference Style Contract Extraction</title>
<style>
body{{margin:0;background:#f6f7f9;color:#161b22;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:#111827;color:white;padding:20px 28px}} header a{{color:#93c5fd}}
main{{padding:18px 28px 48px}} section{{background:white;border:1px solid #d8dee4;border-radius:8px;padding:14px;margin:0 0 18px}}
.refs,.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0}}
img,.placeholder{{width:100%;aspect-ratio:1/1;object-fit:cover;border:1px solid #d8dee4;border-radius:6px;background:#eef1f4;display:flex;align-items:center;justify-content:center}}
figure{{margin:0;background:#fbfcfe;border:1px solid #d8dee4;border-radius:8px;padding:9px}} figcaption{{font-size:12px;color:#57606a}}
details{{background:#fbfcfe;border:1px solid #d8dee4;border-radius:8px;padding:9px 11px;margin:10px 0}} summary{{font-weight:700;cursor:pointer}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.meta{{color:#57606a}}
</style>
</head>
<body>
<header>
  <h1>Reference Style Contract Extraction</h1>
  <p>Scene: {html.escape(DEFAULT_CONTENT_PROMPT)}</p>
  <p>Protocol audit: {"PASS" if audit.get("ok") else "FAIL"}</p>
</header>
<main>
<details open><summary>Protocol audit JSON</summary><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details>
{''.join(sections)}
</main>
</body>
</html>
"""
    (output_root / "index.html").write_text(html_text, encoding="utf-8")


def _selected_tasks(dataset_root: Path, task_ids: Sequence[str]) -> List[StyleTask]:
    tasks_by_id = {task.task_id: task for task in discover_tasks(dataset_root)}
    missing = [task_id for task_id in task_ids if task_id not in tasks_by_id]
    if missing:
        raise ValueError("Task ids not found in dataset: " + ", ".join(missing))
    return [tasks_by_id[task_id] for task_id in task_ids]


def _build_contract_records(args: argparse.Namespace, config: StyleForgeConfig, tasks: Sequence[StyleTask]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_root = args.output_root / "contracts"
    client = VisionLanguageModelClient(config.backend, args.output_root / "_vlm_cache")
    records: List[Dict[str, Any]] = []
    all_task_ids = [task.task_id for task in discover_tasks(Path(config.dataset_root))]
    audit_errors = _audit_content_prompt(args.content_prompt, task_ids=all_task_ids)
    for idx, task in enumerate(tasks, start=1):
        contract_path = contract_root / task.task_id / "detail_mechanism_contract.json"
        source_contract_path = None
        if args.contract_source_root:
            source_candidate = args.contract_source_root / "contracts" / task.task_id / "detail_mechanism_contract.json"
            if source_candidate.exists():
                source_contract_path = source_candidate
        if source_contract_path is not None and not args.force_contracts:
            contract = _load_json(source_contract_path)
            _validate_contract(contract, task_id=task.task_id, max_detail_chars=args.max_detail_chars)
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(contract_path, contract)
            print(f"[contract] copy {idx}/{len(tasks)} {task.task_id} from {source_contract_path}", flush=True)
        elif contract_path.exists() and not args.force_contracts:
            contract = _load_json(contract_path)
            _validate_contract(contract, task_id=task.task_id, max_detail_chars=args.max_detail_chars)
            print(f"[contract] reuse {idx}/{len(tasks)} {task.task_id}", flush=True)
        else:
            print(f"[contract] VLM {idx}/{len(tasks)} {task.task_id}", flush=True)
            started = time.time()
            contract = _call_vlm_contract(
                client=client,
                config=config,
                task=task,
                max_detail_chars=args.max_detail_chars,
            )
            contract["_latency_s"] = round(time.time() - started, 6)
            contract["_style_prompt_source"] = ""
            contract["_b5_contract_source"] = ""
            contract["_reference_only_inputs"] = True
            _write_json(contract_path, contract)
            print(f"[contract] completed {task.task_id} in {contract['_latency_s']:.1f}s", flush=True)
        detail_note = _compiled_detail_note(contract, container_guard=bool(args.container_guard))
        guard_applied = bool(args.container_guard and _literal_container_risk(contract))
        audit_contract = dict(contract)
        audit_contract["compiled_prompt_note"] = detail_note
        if isinstance(audit_contract.get("contract"), Mapping):
            inner = dict(audit_contract["contract"])
            inner["generator_safe_wording"] = detail_note
            audit_contract["contract"] = inner
        task_errors = _audit_contract(
            contract=audit_contract,
            task_id=task.task_id,
            content_prompt=args.content_prompt,
        )
        audit_errors.extend(f"{task.task_id}: {error}" for error in task_errors)
        records.append(
            {
                "task_id": task.task_id,
                "ref_images": [repository_path(path) for path in task.ref_images[:4]],
                "style_prompt_source": "",
                "b5_contract_source": "",
                "contract_source": repository_path(contract_path),
                "contract": contract,
                "detail_note": detail_note,
                "container_guard_applied": guard_applied,
                "seed": int(args.seed),
                "scene_key": args.scene_key,
                "content_prompt": args.content_prompt,
            }
        )
        _write_json(args.output_root / "summary.partial.json", {"records": records})
    audit = {
        "ok": not audit_errors,
        "errors": audit_errors,
        "rules": [
            "same contract schema for every task",
            "selected task ids are runtime arguments",
            "reference images only during extraction; no legacy prompt or color contract",
            "no eval content tokens in prompt-ready contract text",
            "no numeric color ratios or percentages in prompt-ready text",
            "one neutral content prompt and one seed for every task",
            "container guard keeps literal container risks out of compiled prompt-ready wording when enabled",
        ],
        "scene_key": args.scene_key,
        "content_prompt": args.content_prompt,
        "seed": int(args.seed),
        "max_detail_chars": int(args.max_detail_chars),
        "container_guard": bool(args.container_guard),
        "contract_source_root": repository_path(args.contract_source_root) if args.contract_source_root else "",
        "reference_only_inputs": True,
    }
    _write_json(args.output_root / "protocol_audit.json", audit)
    _write_json(args.output_root / "summary.contracts.json", {"audit": audit, "records": records})
    _write_html(records, args.output_root, audit=audit)
    return records, audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "styles")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tasks", required=True, help="Comma-separated task ids; runtime selection only.")
    parser.add_argument("--scene-key", default=DEFAULT_SCENE_KEY)
    parser.add_argument("--content-prompt", default=DEFAULT_CONTENT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-detail-chars", type=int, default=MAX_DETAIL_CHARS)
    parser.add_argument("--contract-source-root", type=Path, default=None)
    parser.add_argument("--container-guard", action="store_true")
    parser.add_argument(
        "--reference-only-inputs",
        action="store_true",
        help="Compatibility flag; extraction is always reference-only.",
    )
    parser.add_argument("--contracts-only", action="store_true")
    parser.add_argument("--force-contracts", action="store_true")
    parser.add_argument("--allow-audit-fail", action="store_true")
    args = parser.parse_args(argv)

    config = StyleForgeConfig.from_file(args.config)
    config.dataset_root = str(args.dataset_root)
    if int(args.max_detail_chars) > 900:
        raise ValueError("--max-detail-chars must stay <= 900 for this extraction contract")
    config.backend.vlm_max_tokens = max(int(config.backend.vlm_max_tokens), 5200)

    task_ids = _parse_task_ids(args.tasks)
    tasks = _selected_tasks(Path(config.dataset_root), task_ids)
    records, audit = _build_contract_records(args, config, tasks)

    if not audit.get("ok") and not args.allow_audit_fail:
        raise ValueError("Protocol audit failed: " + "; ".join(str(error) for error in audit.get("errors") or []))
    print(f"CONTRACTS_DONE {args.output_root / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
