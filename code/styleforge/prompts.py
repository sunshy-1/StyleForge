"""Prompt builders for extractor, refiner, and critic roles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import CriticFeedback, IterationState, StyleAxis, StyleDescriptor


def axis_schema_text(axes: List[StyleAxis]) -> str:
    lines = []
    for axis in axes:
        lines.append(f"- {axis.name}: {axis.description}")
    return "\n".join(lines)


def build_extract_prompt(
    ref_image_paths: List[str],
    axes: List[StyleAxis],
    reference_color_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "role": "StyleExtractor",
        "system": (
            "You extract only shared style traits across all reference images. "
            "Ignore subject identity and one-off content. Return strict JSON."
        ),
        "images": ref_image_paths,
        "axes": [axis.to_dict() for axis in axes],
        "reference_color_contract": reference_color_contract,
        "instruction": (
            "Summarize the intersection of the reference image style across these axes:\n"
            f"{axis_schema_text(axes)}\n"
            "Return JSON with global_summary, per_axis, and rendered_prompt_fragment."
        ),
    }


def build_refine_prompt(
    prev_descriptor: StyleDescriptor,
    critic_feedback: CriticFeedback,
    history: List[IterationState],
    ref_image_paths: List[str],
    axes: List[StyleAxis],
    history_limit: int = 4,
    reference_color_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compact_history = [
        {
            "round_idx": state.round_idx,
            "overall_score": state.feedback.overall_score,
            "axis_scores": state.feedback.axis_scores(),
            "objective_color": _objective_color_summary(state.objective_metrics),
        }
        for state in history[-history_limit:]
    ]
    return {
        "role": "StyleRefiner",
        "system": (
            "You refine an existing style descriptor. Preserve stable axes and edit only "
            "dimensions with concrete critic gaps. Return strict JSON."
        ),
        "images": ref_image_paths,
        "axes": [axis.to_dict() for axis in axes],
        "prev_descriptor": prev_descriptor.to_dict(),
        "critic_feedback": critic_feedback.to_dict(),
        "history": compact_history,
        "reference_color_contract": reference_color_contract,
        "instruction": (
            "Use suggested_edit values as strong signals. Do not rewrite converged axes. "
            "Return JSON with global_summary, per_axis, and rendered_prompt_fragment."
        ),
    }


def build_critic_prompt(
    ref_image_paths: List[str],
    generated_image_path: str,
    axes: List[StyleAxis],
    prev_feedback: Optional[CriticFeedback],
) -> Dict[str, Any]:
    return {
        "role": "StyleCritic",
        "system": (
            "Compare reference style against generated image style. Ignore subject matter "
            "differences. Score each axis independently in [0, 1]. Return strict JSON."
        ),
        "reference_images": ref_image_paths,
        "generated_image": generated_image_path,
        "axes": [axis.to_dict() for axis in axes],
        "prev_feedback": prev_feedback.to_dict() if prev_feedback else None,
        "calibration": {
            "0.9": "near style match; only minor prompt refinements remain",
            "0.7": "recognizable style overlap but important axis gaps remain",
            "0.5": "some generic similarity but weak axis fidelity",
            "0.2": "mostly unrelated style",
        },
        "instruction": (
            "For each axis, describe the gap as reference style versus generated style. "
            "Make suggested_edit prompt-actionable."
        ),
    }


def compose_generation_prompt(style_fragment: str, content_prompt: str, template: str) -> str:
    style_fragment = _clean_prompt_part(style_fragment)
    content_prompt = _clean_prompt_part(content_prompt)
    if template == "instructional":
        return (
            f"Use this visual style first: {style_fragment}. "
            f"Depict the requested content: {content_prompt}."
        )
    if template == "concat":
        return f"{content_prompt}. Visual style: {style_fragment}."
    if template == "in_style_of":
        return f"Depict {content_prompt}. Render it in this visual style: {style_fragment}."
    if template == "structured":
        return f"Content: {content_prompt}\nVisual style: {style_fragment}"
    raise ValueError(f"Unknown prompt template: {template}")


def _clean_prompt_part(text: str) -> str:
    return " ".join(str(text or "").strip(" ,.;").split())


def _objective_color_summary(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not metrics.get("available") or not isinstance(metrics.get("color"), dict):
        return None
    color = metrics["color"]
    return {
        "similarity": color.get("similarity"),
        "band": color.get("band"),
        "palette_delta_e76": color.get("palette_delta_e76"),
        "hue_histogram_l1": color.get("hue_histogram_l1"),
        "saturation_diff": color.get("saturation_diff"),
        "value_diff": color.get("value_diff"),
        "summary": color.get("summary"),
    }
