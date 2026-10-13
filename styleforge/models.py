"""Typed state objects and validation helpers for StyleForge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _require_keys(data: Mapping[str, Any], keys: List[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{label} missing required keys: {', '.join(missing)}")


def _check_score(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {score}")
    return score


def _check_grade(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer grade in {{1, 2, 3, 4}}")
    if value not in {1, 2, 3, 4}:
        raise ValueError(f"{label} must be in {{1, 2, 3, 4}}, got {value}")
    return value


def grade_from_score(score: float) -> int:
    return max(1, min(4, int(score * 4 + 0.5)))


@dataclass(frozen=True)
class StyleAxis:
    name: str
    description: str
    weight: float = 1.0
    examples_pos: List[str] = field(default_factory=list)
    examples_neg: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StyleDescriptor:
    global_summary: str
    per_axis: Dict[str, str]
    rendered_prompt_fragment: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], axes: List[StyleAxis]) -> "StyleDescriptor":
        _require_keys(
            data,
            ["global_summary", "per_axis", "rendered_prompt_fragment"],
            "StyleDescriptor",
        )
        if not isinstance(data["global_summary"], str):
            raise ValueError("StyleDescriptor.global_summary must be a string")
        if not isinstance(data["rendered_prompt_fragment"], str):
            raise ValueError("StyleDescriptor.rendered_prompt_fragment must be a string")
        if not isinstance(data["per_axis"], Mapping):
            raise ValueError("StyleDescriptor.per_axis must be an object")

        expected = [axis.name for axis in axes]
        missing = [name for name in expected if name not in data["per_axis"]]
        if missing:
            raise ValueError("StyleDescriptor.per_axis missing axes: " + ", ".join(missing))

        per_axis: Dict[str, str] = {}
        for name in expected:
            value = data["per_axis"][name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"StyleDescriptor.per_axis.{name} must be a non-empty string")
            per_axis[name] = value

        return cls(
            global_summary=data["global_summary"].strip(),
            per_axis=per_axis,
            rendered_prompt_fragment=data["rendered_prompt_fragment"].strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AxisFeedback:
    axis_name: str
    score: float
    grade: int
    matched: bool
    gap_description: str
    suggested_edit: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], axis_names: List[str]) -> "AxisFeedback":
        _require_keys(
            data,
            ["axis_name", "matched", "gap_description", "suggested_edit"],
            "AxisFeedback",
        )
        axis_name = data["axis_name"]
        if axis_name not in axis_names:
            raise ValueError(f"AxisFeedback.axis_name is not enabled: {axis_name}")
        if "grade" in data:
            grade = _check_grade(data["grade"], f"AxisFeedback[{axis_name}].grade")
            score = grade / 4.0
        else:
            _require_keys(data, ["score"], "AxisFeedback")
            score = _check_score(data["score"], f"AxisFeedback[{axis_name}].score")
            grade = grade_from_score(score)
        if not isinstance(data["matched"], bool):
            raise ValueError("AxisFeedback.matched must be a boolean")
        if not isinstance(data["gap_description"], str):
            raise ValueError("AxisFeedback.gap_description must be a string")
        if not isinstance(data["suggested_edit"], str):
            raise ValueError("AxisFeedback.suggested_edit must be a string")
        return cls(
            axis_name=axis_name,
            score=score,
            grade=grade,
            matched=data["matched"],
            gap_description=data["gap_description"].strip(),
            suggested_edit=data["suggested_edit"].strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CriticFeedback:
    per_axis: List[AxisFeedback]
    overall_score: float
    overall_comment: str
    converged: bool
    style_match: Optional[int] = None
    improvement_plan: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], axes: List[StyleAxis]) -> "CriticFeedback":
        axis_names = [axis.name for axis in axes]
        if "style_match" in data:
            style_match = cls._check_style_match(data["style_match"])
            if not isinstance(data.get("improvement_plan", ""), str):
                raise ValueError("CriticFeedback.improvement_plan must be a string")
            overall_comment = str(data.get("overall_comment") or data.get("improvement_plan") or "").strip()
            improvement_plan = str(data.get("improvement_plan") or "").strip()
            feedback = cls._feedback_from_binary(style_match, improvement_plan, axis_names)
            return cls(
                per_axis=feedback,
                overall_score=float(style_match),
                overall_comment=overall_comment,
                converged=bool(style_match),
                style_match=style_match,
                improvement_plan=improvement_plan,
            )

        _require_keys(data, ["per_axis", "overall_score", "overall_comment", "converged"], "CriticFeedback")
        if not isinstance(data["per_axis"], list):
            raise ValueError("CriticFeedback.per_axis must be a list")
        feedback = [AxisFeedback.from_dict(item, axis_names) for item in data["per_axis"]]
        cls._validate_axis_coverage(feedback, axis_names)
        if not isinstance(data["overall_comment"], str):
            raise ValueError("CriticFeedback.overall_comment must be a string")
        if not isinstance(data["converged"], bool):
            raise ValueError("CriticFeedback.converged must be a boolean")
        return cls(
            per_axis=feedback,
            overall_score=_check_score(data["overall_score"], "CriticFeedback.overall_score"),
            overall_comment=data["overall_comment"].strip(),
            converged=data["converged"],
            style_match=data.get("style_match") if data.get("style_match") in (0, 1) else None,
            improvement_plan=str(data.get("improvement_plan") or "").strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_axis": [item.to_dict() for item in self.per_axis],
            "overall_score": self.overall_score,
            "overall_comment": self.overall_comment,
            "converged": self.converged,
            "style_match": self.style_match,
            "improvement_plan": self.improvement_plan,
        }

    def axis_scores(self) -> Dict[str, float]:
        return {item.axis_name: item.score for item in self.per_axis}

    @staticmethod
    def _check_style_match(value: Any) -> int:
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError("CriticFeedback.style_match must be integer 0 or 1")
        return int(value)

    @staticmethod
    def _validate_axis_coverage(feedback: List[AxisFeedback], axis_names: List[str]) -> None:
        seen = [item.axis_name for item in feedback]
        missing = [name for name in axis_names if name not in seen]
        if missing:
            raise ValueError("CriticFeedback.per_axis missing axes: " + ", ".join(missing))
        if len(seen) != len(set(seen)):
            raise ValueError("CriticFeedback.per_axis contains duplicate axes")

    @staticmethod
    def _feedback_from_binary(
        style_match: int,
        improvement_plan: str,
        axis_names: List[str],
    ) -> List[AxisFeedback]:
        matched = bool(style_match)
        grade = 4 if matched else 1
        score = 1.0 if matched else 0.0
        gap = "" if matched else "VLM judged the generated image style as not yet consistent with the references."
        edit = "No further style edit needed." if matched else improvement_plan
        return [
            AxisFeedback(
                axis_name=name,
                score=score,
                grade=grade,
                matched=matched,
                gap_description=gap,
                suggested_edit=edit,
            )
            for name in axis_names
        ]


@dataclass
class GeneratedImage:
    local_path: str
    prompt: str
    seed: int
    params: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.local_path:
            raise ValueError("GeneratedImage.local_path is required")
        if not isinstance(self.seed, int):
            raise ValueError("GeneratedImage.seed must be an integer")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IterationState:
    round_idx: int
    descriptor: StyleDescriptor
    content_prompt: str
    gen_image: GeneratedImage
    feedback: CriticFeedback
    cost: Dict[str, float]
    latency_s: Dict[str, float]
    objective_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "descriptor": self.descriptor.to_dict(),
            "content_prompt": self.content_prompt,
            "gen_image": self.gen_image.to_dict(),
            "feedback": self.feedback.to_dict(),
            "objective_metrics": self.objective_metrics,
            "cost": self.cost,
            "latency_s": self.latency_s,
        }


@dataclass
class StabilityReport:
    n_samples: int
    mean: float
    variance: float
    per_axis_variance: Dict[str, float]
    samples: List[Dict[str, Any]]
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StyleForgeResult:
    task_id: str
    config: Dict[str, Any]
    states: List[IterationState]
    final_descriptor: StyleDescriptor
    stability: StabilityReport
    total_cost: Dict[str, float]
    stop_reason: str
    output_dir: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config": self.config,
            "states": [state.to_dict() for state in self.states],
            "final_descriptor": self.final_descriptor.to_dict(),
            "stability": self.stability.to_dict(),
            "total_cost": self.total_cost,
            "stop_reason": self.stop_reason,
            "output_dir": self.output_dir,
            "metadata": self.metadata,
        }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
