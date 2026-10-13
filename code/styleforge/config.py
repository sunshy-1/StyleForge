"""Configuration for StyleForge runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_CONTENT_PROMPTS = [
    "a person sitting by a window",
    "a tree on a hillside at sunset",
    "a quiet street corner",
    "a small cafe table with flowers",
    "a bicycle leaning near a garden wall",
    "a mountain cabin beside a lake",
    "a child reading under a lamp",
    "a simple market stall",
]


DEFAULT_ENABLED_AXES = [
    "color_palette",
    "composition",
    "brushwork",
    "lighting",
    "texture",
    "subject_treatment",
    "mood",
]


@dataclass
class BackendConfig:
    style_agent: str = "mock"
    critic: str = "mock"
    generator: str = "mock"
    vlm_max_tokens: int = 1800
    vlm_timeout_s: float = 300.0
    vlm_request_retries: int = 2
    vlm_stream: bool = True
    vlm_protocol: str = "openai"
    vlm_temperature: float = 0.0
    vlm_response_format: bool = False
    vlm_image_max_side: int = 768
    vlm_image_quality: int = 85
    flux2_base_model: str = "models/FLUX.2-klein-9B"
    flux2_device: str = ""
    flux2_dtype: str = "auto"
    flux2_lora_path: str = ""
    flux2_adapter_name: str = "student"
    flux2_output_type: str = ""
    flux2_max_sequence_length: int = 1024

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationConfig:
    width: int = 512
    height: int = 512
    steps: int = 28
    guidance: float = 3.5
    n: int = 1
    base_seed: int = 1234
    negative_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StyleForgeConfig:
    dataset_root: str = "data/styles"
    output_root: str = "outputs"
    max_ref_images: int = 0
    max_iter: int = 8
    tau_overall: float = 0.80
    tau_axis: float = 0.75
    early_stop_patience: int = 2
    improvement_epsilon: float = 0.03
    stability_n_samples: int = 4
    stability_failure_policy: str = "raise"
    stability_critic_timeout_s: float = 0.0
    stability_critic_request_retries: int = -1
    score_quantization_step: float = 0.25
    recompute_overall_from_axes: bool = True
    objective_metrics_enabled: bool = True
    objective_metrics_max_side: int = 256
    objective_palette_size: int = 24
    # Literal reference colors are diagnostic; object colors can contaminate style.
    objective_color_score_enabled: bool = False
    objective_color_prompt_guidance_enabled: bool = False
    objective_color_prompt_guidance_threshold: float = 0.85
    reference_color_contract_enabled: bool = False
    reference_color_contract_max_swatches: int = 10
    reference_color_contract_role_swatches: int = 2
    # Zero selects the number of salient colors from reference complexity.
    salient_color_contract_enabled: bool = False
    salient_color_contract_max_colors: int = 0
    salient_color_contract_min_area: float = 0.002
    salient_color_nonregression_guard_enabled: bool = True
    salient_color_guard_tolerance: float = 0.04
    salient_color_signature_min_score: float = 0.65
    salient_color_signature_max_hue_l1: float = 0.35
    salient_color_accent_min_score: float = 0.25
    # Reference color/lighting relationships guide refinement and selection.
    color_signature_enabled: bool = True
    color_signature_feedback_enabled: bool = True
    color_signature_max_side: int = 256
    color_signature_sat_threshold: float = 0.18
    # Weight of color-grading fidelity vs the SigLIP style score in the
    # objective composite (0 = ignore color, 1 = color only).
    color_signature_select_weight: float = 0.5
    # v13 optional offline final-stage repair. This runs after final descriptor
    # selection and before stability, using a small set of generated samples to
    # rewrite only reusable color/lighting wording in the final prompt.
    final_color_contract_repair_enabled: bool = False
    final_color_contract_sample_indices: List[int] = field(default_factory=lambda: [0, 5])
    final_color_contract_max_chars: int = 600
    final_color_contract_keep_sample_images: bool = True
    # v13.2 non-regression gate: after inserting the final lighting/color
    # recipe, generate one or more offline validation samples and require the
    # same binary critic to still accept the result as the reference style.
    final_color_contract_validation_enabled: bool = True
    final_color_contract_validation_sample_indices: List[int] = field(default_factory=lambda: [0])
    # Best-so-far hill climbing: always refine from the highest-composite
    # descriptor and gate convergence on the objective composite instead of the
    # subjective GPT critic (the critic stays as qualitative refiner input).
    objective_hillclimb_enabled: bool = True
    # Which signal drives convergence and final selection:
    # "critic_binary" = VLM decides 0/1 style_match and provides the next edit;
    # "critic" = legacy VLM scalar score; "composite" = deterministic objective
    # blend. For real style transfer, prefer critic_binary because the objective
    # numbers cannot reliably see shape language, linework, material finish, or
    # whether the style is visibly the same.
    hillclimb_metric: str = "composite"
    tau_composite: float = 0.80
    # Explicit generation style compiler switch. The default keeps the legacy
    # StyleDescriptor compiler. ``style_schema`` loads reference-only schema
    # records by task id and compiles them with the public schema compiler.
    generation_style_prompt_compiler: str = "legacy"
    generation_style_prompt_max_chars: int = 1300
    generation_style_prompt_include_axes: bool = False
    style_schema_run_root: str = "outputs/style_schema"
    style_schema_max_chars: int = 2200
    enabled_axes: List[str] = field(default_factory=lambda: list(DEFAULT_ENABLED_AXES))
    content_prompts: List[str] = field(default_factory=lambda: list(DEFAULT_CONTENT_PROMPTS))
    content_prompt_strategy: str = "fixed"
    fixed_content_prompt_index: int = -1
    prompt_template: str = "instructional"
    critic_calibration: str = "anchored"
    max_json_retries: int = 3
    backend: BackendConfig = field(default_factory=BackendConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_file(cls, path: Path) -> "StyleForgeConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StyleForgeConfig":
        base = cls()
        merged = _deep_update(base.to_dict(), dict(data))
        backend_data = merged.pop("backend", {})
        generation_data = merged.pop("generation", {})
        return cls(
            **merged,
            backend=BackendConfig(**backend_data),
            generation=GenerationConfig(**generation_data),
        )

    def with_overrides(
        self,
        *,
        dataset_root: Optional[str] = None,
        output_root: Optional[str] = None,
        max_iter: Optional[int] = None,
        backend: Optional[str] = None,
        style_agent_backend: Optional[str] = None,
        critic_backend: Optional[str] = None,
        generator_backend: Optional[str] = None,
    ) -> "StyleForgeConfig":
        data = self.to_dict()
        if dataset_root is not None:
            data["dataset_root"] = dataset_root
        if output_root is not None:
            data["output_root"] = output_root
        if max_iter is not None:
            data["max_iter"] = max_iter
        if backend is not None:
            data["backend"]["style_agent"] = backend
            data["backend"]["critic"] = backend
            data["backend"]["generator"] = backend
        if style_agent_backend is not None:
            data["backend"]["style_agent"] = style_agent_backend
        if critic_backend is not None:
            data["backend"]["critic"] = critic_backend
        if generator_backend is not None:
            data["backend"]["generator"] = generator_backend
        return self.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["backend"] = self.backend.to_dict()
        data["generation"] = self.generation.to_dict()
        return data

    def validate(self) -> None:
        if self.max_ref_images < 0:
            raise ValueError("max_ref_images must be >= 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.stability_n_samples < 1:
            raise ValueError("stability_n_samples must be >= 1")
        if self.stability_failure_policy not in {"raise", "record_unavailable"}:
            raise ValueError("stability_failure_policy must be raise or record_unavailable")
        if self.stability_critic_timeout_s < 0.0:
            raise ValueError("stability_critic_timeout_s must be >= 0")
        if self.stability_critic_request_retries < -1:
            raise ValueError("stability_critic_request_retries must be >= -1")
        if self.score_quantization_step < 0.0 or self.score_quantization_step > 1.0:
            raise ValueError("score_quantization_step must be in [0, 1]")
        if self.objective_metrics_max_side < 32:
            raise ValueError("objective_metrics_max_side must be >= 32")
        if self.objective_palette_size < 2:
            raise ValueError("objective_palette_size must be >= 2")
        if (
            self.objective_color_prompt_guidance_threshold < 0.0
            or self.objective_color_prompt_guidance_threshold > 1.0
        ):
            raise ValueError("objective_color_prompt_guidance_threshold must be in [0, 1]")
        if self.reference_color_contract_max_swatches < 3:
            raise ValueError("reference_color_contract_max_swatches must be >= 3")
        if self.reference_color_contract_role_swatches < 1:
            raise ValueError("reference_color_contract_role_swatches must be >= 1")
        if self.salient_color_contract_max_colors < 0 or self.salient_color_contract_max_colors > 12:
            raise ValueError("salient_color_contract_max_colors must be in [0, 12]")
        if self.salient_color_contract_min_area < 0.0 or self.salient_color_contract_min_area > 0.05:
            raise ValueError("salient_color_contract_min_area must be in [0, 0.05]")
        if self.salient_color_guard_tolerance < 0.0 or self.salient_color_guard_tolerance > 1.0:
            raise ValueError("salient_color_guard_tolerance must be in [0, 1]")
        if self.salient_color_signature_min_score < 0.0 or self.salient_color_signature_min_score > 1.0:
            raise ValueError("salient_color_signature_min_score must be in [0, 1]")
        if self.salient_color_signature_max_hue_l1 < 0.0 or self.salient_color_signature_max_hue_l1 > 2.0:
            raise ValueError("salient_color_signature_max_hue_l1 must be in [0, 2]")
        if self.salient_color_accent_min_score < 0.0 or self.salient_color_accent_min_score > 1.0:
            raise ValueError("salient_color_accent_min_score must be in [0, 1]")
        if self.generation_style_prompt_max_chars < 120:
            raise ValueError("generation_style_prompt_max_chars must be >= 120")
        if self.generation_style_prompt_compiler not in {"legacy", "style_schema"}:
            raise ValueError("generation_style_prompt_compiler must be legacy or style_schema")
        if self.style_schema_max_chars < 180:
            raise ValueError("style_schema_max_chars must be >= 180")
        if self.generation_style_prompt_compiler == "style_schema" and not self.style_schema_run_root:
            raise ValueError("style_schema_run_root must be set when style_schema compiler is enabled")
        if self.generation_style_prompt_compiler == "style_schema" and self.final_color_contract_repair_enabled:
            raise ValueError("style_schema compiler cannot be combined with final_color_contract_repair_enabled")
        if self.color_signature_select_weight < 0.0 or self.color_signature_select_weight > 1.0:
            raise ValueError("color_signature_select_weight must be in [0, 1]")
        if self.final_color_contract_max_chars < 120:
            raise ValueError("final_color_contract_max_chars must be >= 120")
        if self.final_color_contract_repair_enabled:
            if not self.final_color_contract_sample_indices:
                raise ValueError("final_color_contract_sample_indices must not be empty when color repair is enabled")
            bad_sample_indices = [
                idx
                for idx in self.final_color_contract_sample_indices
                if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= len(self.content_prompts)
            ]
            if bad_sample_indices:
                raise ValueError(
                    "final_color_contract_sample_indices out of range for content_prompts: "
                    + ", ".join(str(idx) for idx in bad_sample_indices)
                )
            if self.final_color_contract_validation_enabled:
                if not self.final_color_contract_validation_sample_indices:
                    raise ValueError(
                        "final_color_contract_validation_sample_indices must not be empty when validation is enabled"
                    )
                bad_validation_indices = [
                    idx
                    for idx in self.final_color_contract_validation_sample_indices
                    if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= len(self.content_prompts)
                ]
                if bad_validation_indices:
                    raise ValueError(
                        "final_color_contract_validation_sample_indices out of range for content_prompts: "
                        + ", ".join(str(idx) for idx in bad_validation_indices)
                    )
        if self.tau_composite < 0.0 or self.tau_composite > 1.0:
            raise ValueError("tau_composite must be in [0, 1]")
        if self.color_signature_max_side < 32:
            raise ValueError("color_signature_max_side must be >= 32")
        if self.hillclimb_metric not in {"composite", "critic", "critic_binary"}:
            raise ValueError("hillclimb_metric must be 'composite', 'critic', or 'critic_binary'")
        if not self.content_prompts:
            raise ValueError("content_prompts must not be empty")
        if self.content_prompt_strategy not in {"fixed", "rotating", "multi"}:
            raise ValueError("content_prompt_strategy must be fixed, rotating, or multi")
        if self.fixed_content_prompt_index >= len(self.content_prompts):
            raise ValueError("fixed_content_prompt_index is out of range for content_prompts")
        if self.prompt_template not in {"instructional", "concat", "in_style_of", "structured"}:
            raise ValueError("prompt_template must be instructional, concat, in_style_of, or structured")


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_update(dict(result[key]), dict(value))
        else:
            result[key] = value
    return result
