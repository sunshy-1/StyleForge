#!/usr/bin/env python3
"""Local Qwen VLM-J style-transfer judge for StyleForge summaries.

By default this script evaluates style only. The ``joint_content_style`` prompt
mode instead judges whether the candidate satisfies the target content prompt
and transfers the reference visual style as one integrated style-transfer
result. The ``joint_content_style_strict`` mode uses the same joint objective but
calibrates the style criterion more conservatively.
For each generated image, it scores the image against each reference image
separately and aggregates the per-reference scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_QWEN_VL_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DOPSD_STYLE_VLM_SYSTEM_PROMPT = """
You are a strict visual style consistency evaluator.

You will receive two images:
- The first image is the reference style image.
- The second image is the candidate image.

The subject and composition are allowed to change. Ignore whether the objects,
people, layout, and scene content are the same. Judge only whether the candidate
preserves the reference image's visual style, including color palette, rendering
method, lighting treatment, texture, edge quality, graphic effects, and overall
visual atmosphere.

Scoring labels:
- highly consistent: the candidate strongly preserves the reference style.
- mostly consistent: the candidate preserves most style cues with minor drift.
- slightly consistent: only a few style cues are preserved.
- not consistent: the candidate does not preserve the reference style.

Output exactly one label from this list:
highly consistent
mostly consistent
slightly consistent
not consistent
""".strip()

STYLE_CATEGORY_VLM_SYSTEM_PROMPT = """
You are a visual style category evaluator.

You will receive two images:
- The first image is a reference image showing a target visual style.
- The second image is a generated candidate image.

The subject, objects, scene, layout, and composition may be completely different.
Do not judge content similarity or subject identity.

Judge whether the candidate belongs to the same visual style category as the
reference. Focus on broad and recognizable style cues such as medium, rendering
approach, color treatment, lighting mood, texture, line/edge quality, graphic
design language, and overall visual atmosphere.

Do not require the candidate to copy the exact colors, composition, objects, or
local details of the reference image. A candidate can be highly consistent if it
clearly expresses the same style category with different content.

Scoring labels:
- highly consistent: clearly the same visual style category with strong style cues.
- mostly consistent: mostly the same style category, with some drift or weaker cues.
- slightly consistent: only a few style cues match, or the style is ambiguous.
- not consistent: the candidate belongs to a different style or looks generic.

Output exactly one label from this list:
highly consistent
mostly consistent
slightly consistent
not consistent
""".strip()

STYLE_TRANSFER_VLM_SYSTEM_PROMPT = """
You are a visual style transfer evaluator.

You will receive two images:
- The first image provides the reference visual style.
- The second image is a generated candidate image.

The two images may depict different subjects, scenes, layouts, and compositions.
Base the judgment on visual style rather than content correspondence.

Evaluate whether the candidate would be recognized as an example of the
reference's visual style category. Use style cues such as medium, rendering
approach, color treatment, lighting mood, texture, line/edge quality, graphic
design language, and overall atmosphere as supporting evidence.

Consider the style at the category level, using the reference image as an
example of that category. A style judgment should be based on the combined
impression of the candidate image rather than a checklist of reference-specific
details.

Scoring labels:
- highly consistent: the candidate is a strong example of the target style category.
- mostly consistent: the candidate is a plausible example of the target style category.
- slightly consistent: the candidate shows some target-style cues, but the category is weak or ambiguous.
- not consistent: the candidate is better described as a different or generic visual category.

Output exactly one label from this list:
highly consistent
mostly consistent
slightly consistent
not consistent
""".strip()

JOINT_CONTENT_STYLE_VLM_SYSTEM_PROMPT = """
You are a strict visual style-transfer evaluator for text-guided image generation.

You will receive:
- A target content prompt describing what the candidate image should depict.
- A first image that is a reference for visual style only.
- A second image that is the generated candidate image.

Judge the candidate as a complete style-transfer result. The candidate must
simultaneously satisfy the target content prompt and express the reference
image's visual style. The reference image content is not the target content:
do not require the candidate to copy the reference subject, objects, scene,
layout, or composition. Use the first image only to judge style.

Content criterion:
- Check whether the candidate visibly depicts the requested subject, scene,
  action, objects, scale, and viewpoint from the target content prompt.
- A generic, missing, wrong, or materially changed main subject should lower the
  judgment even if the style looks plausible.

Style criterion:
- Check whether the candidate transfers the reference's visual style through
  medium, rendering method, palette behavior, lighting treatment, texture,
  line and edge quality, abstraction level, graphic effects, and atmosphere.
- Do not give high credit for generic style-category words alone when the
  concrete visual style cues are weak or absent.

Joint criterion:
- Give a high label only when the requested content and the reference style are
  both present and naturally integrated in the same image.
- Judge the overall semantic success directly. Do not compute a mechanical
  minimum, average, or checklist score between content and style.

Scoring labels:
- highly consistent: the candidate clearly depicts the target content and strongly transfers the reference visual style as an integrated result.
- mostly consistent: the candidate depicts the main target content and transfers most important style cues, with only minor omissions or drift.
- slightly consistent: the candidate has partial target content or partial style transfer, but one side is weak, incomplete, generic, or poorly integrated.
- not consistent: the candidate misses the target content, misses the reference style, or looks like a different/generic result.

Output exactly one label from this list:
highly consistent
mostly consistent
slightly consistent
not consistent
""".strip()

JOINT_CONTENT_STYLE_STRICT_VLM_SYSTEM_PROMPT = """
You are a strict visual style-transfer evaluator for text-guided image generation.

You will receive:
- A target content prompt describing what the candidate image should depict.
- A first image that is a reference for visual style only.
- A second image that is the generated candidate image.

Judge the candidate as a complete style-transfer result. The candidate must
simultaneously satisfy the target content prompt and reproduce the reference
image's visual style. The reference image content is not the target content:
do not require the candidate to copy the reference subject, objects, scene,
layout, or composition. Use the first image only to judge style.

Be conservative. Start from "not consistent" and increase the label only when
there is clear visual evidence in the candidate image. Do not reward a candidate
for merely naming or loosely resembling a broad style category such as anime,
watercolor, oil painting, noir, art deco, sketch, or cyberpunk.

Content criterion:
- The candidate must visibly depict the requested subject, scene, action,
  objects, scale, and viewpoint from the target content prompt.
- Missing the main subject, changing it into a different subject, dropping
  important requested objects, or changing the requested scale/viewpoint should
  cap the final label at "slightly consistent" even if the style looks plausible.
- If the content is generic or ambiguous but not fully wrong, prefer "slightly
  consistent" or "mostly consistent" depending on the style evidence.

Strict style criterion:
- Compare the candidate to the reference image's concrete visual mechanisms,
  not just the style name. Look for evidence across medium/rendering method,
  palette structure, value contrast, lighting treatment, surface texture, brush
  or stroke behavior, line and edge quality, abstraction level, geometry or
  ornament language, shading/depth treatment, and graphic effects.
- A high style judgment requires multiple distinctive mechanisms from the
  reference to appear in the candidate. Generic smooth digital art, generic
  photorealism, generic illustration, or a single superficial cue is not enough.
- If the candidate only matches the broad category but misses the reference's
  distinctive palette, texture, edge language, lighting, or rendering behavior,
  cap the final label at "slightly consistent".
- If style evidence is present but weak, partial, or mixed with an unrelated
  default model style, prefer "slightly consistent". Use "mostly consistent"
  only when most important reference-specific style mechanisms are visible.

Joint criterion:
- Give a high label only when the requested content and the reference-specific
  style mechanisms are both present and naturally integrated in the same image.
- Judge the overall semantic success directly. Do not compute a mechanical
  minimum, average, or checklist score between content and style.
- The highest label should be rare. Use it only when a human evaluator would
  clearly recognize the candidate as the requested content rendered in the
  reference's specific visual style, not merely in the same broad category.

Scoring labels:
- highly consistent: the target content is clearly correct, and the candidate strongly reproduces several distinctive reference-specific style mechanisms as an integrated result.
- mostly consistent: the main target content is correct, and most important reference-specific style mechanisms are visible, with only moderate omissions or drift.
- slightly consistent: the candidate has partial target content or partial style transfer, or it only matches a broad/generic style category while missing distinctive reference cues.
- not consistent: the candidate misses the target content, misses the reference-specific style, or looks like a different/generic result.

Output exactly one label from this list, and do not output any explanation:
highly consistent
mostly consistent
slightly consistent
not consistent
""".strip()

STYLE_VLM_SYSTEM_PROMPTS = {
    "style_transfer": STYLE_TRANSFER_VLM_SYSTEM_PROMPT,
    "style_category": STYLE_CATEGORY_VLM_SYSTEM_PROMPT,
    "dopsd": DOPSD_STYLE_VLM_SYSTEM_PROMPT,
    "joint_content_style": JOINT_CONTENT_STYLE_VLM_SYSTEM_PROMPT,
    "joint_content_style_strict": JOINT_CONTENT_STYLE_STRICT_VLM_SYSTEM_PROMPT,
}
STYLE_VLM_USER_PROMPTS = {
    "style_transfer": "Evaluate whether the second image expresses the same visual style category as the first image. Output one label only.",
    "style_category": "Judge whether the second image belongs to the same visual style category as the first image. Output one label only.",
    "dopsd": "Judge whether the second image preserves the visual style of the first image. Output one label only.",
    "joint_content_style": (
        "Target content prompt:\n{content_prompt}\n\n"
        "The first image is a style reference only. The second image is the generated candidate. "
        "Judge whether the second image both depicts the target content prompt and transfers "
        "the first image's visual style as one integrated result. Output one label only."
    ),
    "joint_content_style_strict": (
        "Target content prompt:\n{content_prompt}\n\n"
        "The first image is a style reference only. The second image is the generated candidate. "
        "Judge strictly whether the second image both depicts the target content prompt and "
        "reproduces the first image's reference-specific visual style mechanisms as one "
        "integrated result. Output one label only."
    ),
}
DEFAULT_PROMPT_MODE = "style_transfer"
STYLE_VLM_SYSTEM_PROMPT = STYLE_TRANSFER_VLM_SYSTEM_PROMPT
JOINT_PROMPT_MODES = {"joint_content_style", "joint_content_style_strict"}

STYLE_LABEL_TO_SCORE = {
    "highly consistent": 4,
    "mostly consistent": 3,
    "slightly consistent": 2,
    "not consistent": 1,
}
NUMERIC_SCORE_TO_LABEL = {score: label for label, score in STYLE_LABEL_TO_SCORE.items()}

CSV_FIELDS = [
    "source_summary",
    "summary_record_index",
    "task_id",
    "content_prompt",
    "seed",
    "prompt_index",
    "candidate_index",
    "generated_image",
    "generated_exists",
    "n_refs",
    "valid_ref_count",
    "available",
    "reason",
    "qwen_vlm_j_style_score",
    "qwen_vlm_j_style_score_median",
    "qwen_vlm_j_style_score_min",
    "qwen_vlm_j_style_score_max",
    "qwen_vlm_j_style_match_rate",
    "qwen_vlm_j_labels",
    "qwen_vlm_j_raw_outputs",
]


@dataclass
class EvalItem:
    source_summary: str
    summary_record_index: int
    task_id: str
    generated_image: Optional[Path]
    ref_images: List[Path]
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score StyleForge generations with local Qwen VLM-J style consistency."
    )
    parser.add_argument("--summary", nargs="+", required=True, help="One or more StyleForge summary.json files.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <summary parent>/qwen_vlm_j for one summary, or <project root>/qwen_vlm_j_eval.",
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--qwen-vl-model-path", default=DEFAULT_QWEN_VL_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default="",
        help="Optional Transformers device map, e.g. auto, for sharding larger local VLMs.",
    )
    parser.add_argument(
        "--tp-plan",
        default="",
        help="Optional Transformers tensor-parallel plan, e.g. auto; launch with torchrun.",
    )
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument(
        "--prompt-mode",
        choices=sorted(STYLE_VLM_SYSTEM_PROMPTS),
        default=DEFAULT_PROMPT_MODE,
        help=(
            "Judge prompt variant. style_transfer is the default style-only category judge; "
            "style_category is a stricter category-level style judge; dopsd reproduces the "
            "original D-OPSD prompt; joint_content_style judges target content and reference "
            "style as one semantic style-transfer result."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download model files. Default is local-files-only.",
    )
    parser.add_argument("--image-max-side", type=int, default=0, help="Resize image long side before judging. 0 disables.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-refs", type=int, default=0, help="Use at most this many references per generated image. 0 means all.")
    parser.add_argument("--limit", type=int, default=0, help="Only score first N records.")
    parser.add_argument("--dry-run", action="store_true", help="Collect records and write empty scores without loading Qwen.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_cli_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), project_root, project_root.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return project_root / path


def resolve_record_path(value: str, base_dirs: Sequence[Path]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        # Rebase stale repository paths without modifying the source summary.
        for anchor in ("outputs", "data", "data-all"):
            if anchor in path.parts:
                candidate = base_dirs[0] / Path(*path.parts[path.parts.index(anchor):])
                if candidate.exists():
                    return candidate
        return path
    candidates = [base / path for base in base_dirs]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def collect_from_summary(summary_path: Path, project_root: Path, max_refs: int = 0) -> List[EvalItem]:
    data = read_json(summary_path)
    raw_records = data.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError(f"summary records must be a list: {summary_path}")

    items: List[EvalItem] = []
    base_dirs = [project_root, summary_path.parent, summary_path.parent.parent]
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            continue
        generated_image = None
        if record.get("generated_image"):
            generated_image = resolve_record_path(str(record["generated_image"]), base_dirs)
        ref_values = record.get("ref_images") or []
        if not isinstance(ref_values, list):
            ref_values = []
        ref_images = [resolve_record_path(str(value), base_dirs) for value in ref_values if value]
        if max_refs > 0:
            ref_images = ref_images[:max_refs]
        items.append(
            EvalItem(
                source_summary=str(summary_path),
                summary_record_index=index,
                task_id=str(record.get("task_id") or summary_path.parent.name),
                generated_image=generated_image,
                ref_images=ref_images,
                metadata={
                    "content_prompt": record.get("content_prompt", ""),
                    "seed": record.get("seed", ""),
                    "prompt_index": record.get("prompt_index", ""),
                    "candidate_index": record.get("candidate_index", ""),
                },
            )
        )
    return items


def collect_items(args: argparse.Namespace, project_root: Path) -> Tuple[List[EvalItem], List[Path]]:
    summary_paths = [resolve_cli_path(value, project_root) for value in args.summary]
    items: List[EvalItem] = []
    for summary_path in summary_paths:
        items.extend(collect_from_summary(summary_path, project_root, max_refs=args.max_refs))
    if args.limit:
        items = items[: args.limit]
    return items, summary_paths


def default_output_dir(args: argparse.Namespace, project_root: Path, summary_paths: Sequence[Path]) -> Path:
    if args.output_dir:
        return resolve_cli_path(args.output_dir, project_root)
    if len(summary_paths) == 1:
        return summary_paths[0].parent / "qwen_vlm_j"
    return project_root / "qwen_vlm_j_eval"


def dtype_from_name(name: str, torch: Any, device: str) -> Any:
    if not str(device).startswith("cuda"):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def normalize_model_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path)
    # If the user passes a repository model identifier and a local copy exists
    # under the configurable relative models directory, prefer that.
    if "/" in value:
        model_root = os.environ.get("MODEL_ROOT", "models")
        candidate = Path(model_root) / value.rstrip("/").split("/")[-1]
        if candidate.exists():
            return str(candidate)
    return value


def detect_local_model_type(model_path: str) -> str:
    path = Path(model_path).expanduser()
    config_path = path / "config.json"
    if not config_path.exists():
        return ""
    try:
        data = read_json(config_path)
    except Exception:
        return ""
    model_type = data.get("model_type") or data.get("text_config", {}).get("model_type") or ""
    return str(model_type).lower()


def ensure_internvl_optional_deps() -> None:
    try:
        import timm  # noqa: F401

        return
    except Exception:
        pass
    # ``timm`` is a normal project dependency. No machine-specific site-
    # packages fallback is used in the open-source repository.


def load_rgb_image(path: Path, max_side: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side))
    return image


def build_internvl_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def internvl_find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def internvl_dynamic_preprocess(image: Any, max_num: int = 12, image_size: int = 448) -> List[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 1 <= i * j <= max_num
    }
    sorted_ratios = sorted(target_ratios, key=lambda value: value[0] * value[1])
    target_aspect_ratio = internvl_find_closest_aspect_ratio(
        aspect_ratio,
        sorted_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_image = image.resize((target_width, target_height))
    processed_images = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_image.crop(box))
    if len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_pixel_values(path: Path, torch: Any, dtype: Any, device: str, max_num: int = 12):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    try:
        transform = build_internvl_transform(input_size=448)
        images = internvl_dynamic_preprocess(image, max_num=max_num, image_size=448)
        pixel_values = torch.stack([transform(tile) for tile in images])
        return pixel_values.to(dtype).to(device)
    finally:
        image.close()


def parse_style_label(text: str) -> Tuple[Optional[str], Optional[int]]:
    lowered = str(text or "").strip().lower()
    for label, score in STYLE_LABEL_TO_SCORE.items():
        if label in lowered:
            return label, score
    for token in ("4", "3", "2", "1"):
        if token in lowered.split():
            score = int(token)
            return NUMERIC_SCORE_TO_LABEL[score], score
    return None, None


def aggregate_scores(scores: Sequence[Optional[int]]) -> Dict[str, Any]:
    valid = [float(score) for score in scores if isinstance(score, (int, float))]
    if not valid:
        return {
            "qwen_vlm_j_style_score": None,
            "qwen_vlm_j_style_score_median": None,
            "qwen_vlm_j_style_score_min": None,
            "qwen_vlm_j_style_score_max": None,
            "qwen_vlm_j_style_match_rate": None,
            "valid_ref_count": 0,
        }
    return {
        "qwen_vlm_j_style_score": round(sum(valid) / len(valid), 4),
        "qwen_vlm_j_style_score_median": round(statistics.median(valid), 4),
        "qwen_vlm_j_style_score_min": round(min(valid), 4),
        "qwen_vlm_j_style_score_max": round(max(valid), 4),
        "qwen_vlm_j_style_match_rate": round(sum(1 for value in valid if value >= 3.0) / len(valid), 4),
        "valid_ref_count": len(valid),
    }


class QwenStyleJudge:
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        torch_dtype: str,
        attn_implementation: str,
        allow_download: bool,
        image_max_side: int,
        max_new_tokens: int,
        prompt_mode: str,
        device_map: str = "",
        tp_plan: str = "",
    ) -> None:
        self.model_path = normalize_model_path(model_path)
        self.device = device
        self.device_map = device_map.strip()
        self.tp_plan = tp_plan.strip()
        self.torch_dtype_name = torch_dtype
        self.attn_implementation = attn_implementation
        self.allow_download = bool(allow_download)
        self.image_max_side = image_max_side
        self.max_new_tokens = max_new_tokens
        self.prompt_mode = prompt_mode
        self.system_prompt = STYLE_VLM_SYSTEM_PROMPTS[prompt_mode]
        self.user_prompt = STYLE_VLM_USER_PROMPTS[prompt_mode]
        self.model_type = detect_local_model_type(self.model_path)
        # Qwen3-VL and Qwen3.5-VL chat templates default to reasoning mode on
        # recent Transformers.  Pairwise judging constrains the first generated
        # token to A/B/C, so leaving the hidden ``<think>`` prefix enabled would
        # force an arbitrary label prior (especially A).  The pairwise scorer
        # explicitly renders ``enable_thinking=False`` whenever this flag is set.
        self.disable_thinking = self.model_type.startswith(("glm", "qwen3_vl", "qwen3_5", "gemma4"))
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.load_error = ""
        self._load()

    @property
    def available(self) -> bool:
        return self.processor is not None and self.model is not None and not self.load_error

    @property
    def input_device(self) -> Any:
        if self.tp_plan:
            return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        if self.device_map and self.model is not None:
            return next(self.model.parameters()).device
        return self.device

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor

            model_loaders = []
            if self.model_type.startswith("glm"):
                try:
                    from transformers import Glm4vForConditionalGeneration

                    model_loaders.append(Glm4vForConditionalGeneration)
                except Exception:
                    pass
            elif self.model_type.startswith("internvl"):
                ensure_internvl_optional_deps()
                from transformers import AutoModel, AutoTokenizer

                self.torch = torch
                dtype = dtype_from_name(self.torch_dtype_name, torch, self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=not self.allow_download,
                    trust_remote_code=True,
                    use_fast=False,
                )
                self.processor = self.tokenizer
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    local_files_only=not self.allow_download,
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,
                    trust_remote_code=True,
                ).to(self.device)
                self.model.eval()
                self.model.system_message = self.system_prompt
                return
            elif self.model_type.startswith("qwen3_vl"):
                try:
                    from transformers import Qwen3VLForConditionalGeneration

                    model_loaders.append(Qwen3VLForConditionalGeneration)
                except Exception:
                    pass
            try:
                from transformers import AutoModelForImageTextToText

                model_loaders.append(AutoModelForImageTextToText)
            except Exception:
                pass
            if not model_loaders:
                raise ImportError("transformers does not expose a compatible image-text model loader")

            self.torch = torch
            dtype = dtype_from_name(self.torch_dtype_name, torch, self.device)
            kwargs: Dict[str, Any] = {
                "pretrained_model_name_or_path": self.model_path,
                "torch_dtype": dtype,
                "local_files_only": not self.allow_download,
            }
            if self.attn_implementation:
                kwargs["attn_implementation"] = self.attn_implementation
            if self.device_map:
                kwargs["device_map"] = self.device_map
            if self.tp_plan:
                kwargs["tp_plan"] = self.tp_plan

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=not self.allow_download,
            )
            last_error = None
            for loader in model_loaders:
                try:
                    self.model = loader.from_pretrained(**kwargs)
                    if not self.device_map and not self.tp_plan:
                        self.model = self.model.to(self.device)
                    self.model.eval()
                    return
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(f"failed to load Qwen VLM: {last_error}")
        except Exception as exc:
            self.processor = None
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def user_prompt_for(self, content_prompt: str) -> str:
        template = self.user_prompt
        if "{content_prompt}" in template:
            content = str(content_prompt or "").strip() or "(empty content prompt)"
            return template.format(content_prompt=content)
        return template

    def judge_pair_internvl(self, reference_path: Path, candidate_path: Path, content_prompt: str = "") -> Dict[str, Any]:
        if not self.available or self.tokenizer is None:
            return {
                "reference_image": str(reference_path),
                "label": None,
                "score": None,
                "raw_output": None,
                "reason": self.load_error or "InternVL unavailable",
            }
        try:
            dtype = dtype_from_name(self.torch_dtype_name, self.torch, self.device)
            reference_pixels = load_internvl_pixel_values(reference_path, self.torch, dtype, self.device)
            candidate_pixels = load_internvl_pixel_values(candidate_path, self.torch, dtype, self.device)
            pixel_values = self.torch.cat((reference_pixels, candidate_pixels), dim=0)
            num_patches_list = [reference_pixels.size(0), candidate_pixels.size(0)]
            question = (
                "Image-1: <image>\n"
                "Image-2: <image>\n"
                f"{self.user_prompt_for(content_prompt)}"
            )
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            }
            with self.torch.no_grad():
                output_text = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                )
            output_text = str(output_text or "").strip()
            label, score = parse_style_label(output_text)
            return {
                "reference_image": str(reference_path),
                "label": label,
                "score": score,
                "raw_output": output_text,
                "reason": "" if score is not None else "could not parse style label",
            }
        except Exception as exc:
            return {
                "reference_image": str(reference_path),
                "label": None,
                "score": None,
                "raw_output": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def judge_pair(self, reference_path: Path, candidate_path: Path, content_prompt: str = "") -> Dict[str, Any]:
        if self.model_type.startswith("internvl"):
            return self.judge_pair_internvl(reference_path, candidate_path, content_prompt)
        if not self.available:
            return {
                "reference_image": str(reference_path),
                "label": None,
                "score": None,
                "raw_output": None,
                "reason": self.load_error or "Qwen VLM unavailable",
            }
        reference_image = None
        candidate_image = None
        try:
            reference_image = load_rgb_image(reference_path, self.image_max_side)
            candidate_image = load_rgb_image(candidate_path, self.image_max_side)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.user_prompt_for(content_prompt),
                        },
                        {"type": "image", "image": reference_image},
                        {"type": "image", "image": candidate_image},
                    ],
                },
            ]
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if self.disable_thinking:
                template_kwargs["enable_thinking"] = False
            try:
                inputs = self.processor.apply_chat_template(messages, **template_kwargs).to(self.input_device)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                inputs = self.processor.apply_chat_template(messages, **template_kwargs).to(self.input_device)
            inputs.pop("token_type_ids", None)
            with self.torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            label, score = parse_style_label(output_text)
            return {
                "reference_image": str(reference_path),
                "label": label,
                "score": score,
                "raw_output": output_text,
                "reason": "" if score is not None else "could not parse style label",
            }
        except Exception as exc:
            return {
                "reference_image": str(reference_path),
                "label": None,
                "score": None,
                "raw_output": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if reference_image is not None:
                reference_image.close()
            if candidate_image is not None:
                candidate_image.close()


def unavailable_row(item: EvalItem, reason: str, generated_exists: bool, existing_refs: Sequence[Path]) -> Dict[str, Any]:
    row = {
        "source_summary": item.source_summary,
        "summary_record_index": item.summary_record_index,
        "task_id": item.task_id,
        "content_prompt": item.metadata.get("content_prompt", ""),
        "seed": item.metadata.get("seed", ""),
        "prompt_index": item.metadata.get("prompt_index", ""),
        "candidate_index": item.metadata.get("candidate_index", ""),
        "generated_image": str(item.generated_image) if item.generated_image else "",
        "generated_exists": generated_exists,
        "n_refs": len(item.ref_images),
        "available": False,
        "reason": reason,
        "per_ref_results": [],
        "qwen_vlm_j_labels": "",
        "qwen_vlm_j_raw_outputs": "",
    }
    row.update(aggregate_scores([]))
    row["valid_ref_count"] = len(existing_refs)
    return row


def score_items(
    items: Sequence[EvalItem],
    judge: Optional[QwenStyleJudge],
    dry_run: bool,
    quiet: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        generated_exists = bool(item.generated_image and item.generated_image.exists())
        existing_refs = [path for path in item.ref_images if path.exists()]
        if not item.generated_image:
            rows.append(unavailable_row(item, "missing generated_image field", False, existing_refs))
            continue
        if not generated_exists:
            rows.append(unavailable_row(item, f"generated image not found: {item.generated_image}", False, existing_refs))
            continue
        if not existing_refs:
            rows.append(unavailable_row(item, "no existing reference images", True, existing_refs))
            continue

        per_ref_results = []
        if dry_run:
            per_ref_results = [
                {
                    "reference_image": str(path),
                    "label": None,
                    "score": None,
                    "raw_output": None,
                    "reason": "dry_run",
                }
                for path in existing_refs
            ]
        else:
            if judge is None:
                raise ValueError("judge must be provided unless dry_run is enabled")
            for ref_path in existing_refs:
                per_ref_results.append(
                    judge.judge_pair(
                        ref_path,
                        item.generated_image,
                        str(item.metadata.get("content_prompt", "") or ""),
                    )
                )

        aggregate = aggregate_scores([result.get("score") for result in per_ref_results])
        available = bool(aggregate["valid_ref_count"])
        reason = "" if available else "; ".join(
            sorted({str(result.get("reason") or "unavailable") for result in per_ref_results})
        )
        labels = [str(result.get("label") or "") for result in per_ref_results]
        raw_outputs = [str(result.get("raw_output") or "") for result in per_ref_results]
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
            "reason": reason,
            "per_ref_results": per_ref_results,
            "qwen_vlm_j_labels": "|".join(labels),
            "qwen_vlm_j_raw_outputs": "|".join(raw_outputs),
        }
        row.update(aggregate)
        rows.append(row)

        if not quiet and (index == 1 or index == total or index % 10 == 0):
            print(
                f"[{index}/{total}] {item.task_id} "
                f"qwen_style={row.get('qwen_vlm_j_style_score')} "
                f"valid_refs={row.get('valid_ref_count')}/{len(existing_refs)}",
                flush=True,
            )
    return rows


def numeric_values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def stats_for(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = numeric_values(rows, key)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    available_rows = [row for row in rows if row.get("available")]
    return {
        "total": len(rows),
        "available": len(available_rows),
        "unavailable": len(rows) - len(available_rows),
        "qwen_vlm_j_style_score": stats_for(available_rows, "qwen_vlm_j_style_score"),
        "qwen_vlm_j_style_match_rate": stats_for(available_rows, "qwen_vlm_j_style_match_rate"),
        "valid_ref_count": stats_for(available_rows, "valid_ref_count"),
    }


def summarize_per_task(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("task_id", "")), []).append(row)
    return {task_id: summarize_rows(task_rows) for task_id, task_rows in sorted(grouped.items())}


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in CSV_FIELDS})


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    summary_paths: Sequence[Path],
    args: argparse.Namespace,
    project_root: Path,
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    payload = {
        "version": "qwen_vlm_j_style_eval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": {
            "name": "qwen_vlm_j_style_consistency",
            "score_scale": {
                "highly consistent": 4,
                "mostly consistent": 3,
                "slightly consistent": 2,
                "not consistent": 1,
            },
            "aggregation": "Each generated image is judged against each reference image separately; per-generated score is the mean of valid per-reference scores.",
            "style_only": args.prompt_mode not in JOINT_PROMPT_MODES,
            "content_prompt_used": args.prompt_mode in JOINT_PROMPT_MODES,
            "dreambooth_subject_identity": False,
        },
        "config": {
            "project_root": str(project_root),
            "qwen_vl_model_path": normalize_model_path(args.qwen_vl_model_path),
            "device": args.device,
            "device_map": args.device_map,
            "tp_plan": args.tp_plan,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "prompt_mode": args.prompt_mode,
            "system_prompt": STYLE_VLM_SYSTEM_PROMPTS[args.prompt_mode],
            "user_prompt": STYLE_VLM_USER_PROMPTS[args.prompt_mode],
            "allow_download": bool(args.allow_download),
            "image_max_side": args.image_max_side,
            "max_new_tokens": args.max_new_tokens,
            "max_refs": args.max_refs,
            "dry_run": bool(args.dry_run),
        },
        "inputs": {
            "summaries": [str(path) for path in summary_paths],
            "summary_count": len(summary_paths),
            "record_count": len(rows),
        },
        "summary": summary,
        "per_task": summarize_per_task(rows),
        "records": list(rows),
    }
    json_path = output_dir / "qwen_vlm_j_metrics.json"
    csv_path = output_dir / "qwen_vlm_j_metrics.csv"
    jsonl_path = output_dir / "qwen_vlm_j_records.jsonl"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    return json_path, csv_path, jsonl_path, summary


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        args.quiet = True
    project_root = resolve_cli_path(args.project_root, DEFAULT_PROJECT_ROOT)
    items, summary_paths = collect_items(args, project_root)
    if not items:
        raise SystemExit("No records found.")
    output_dir = default_output_dir(args, project_root, summary_paths)

    if not args.quiet:
        print(f"Collected {len(items)} generated images for Qwen VLM-J style judging.", flush=True)
        print(f"Model path: {normalize_model_path(args.qwen_vl_model_path)}", flush=True)
        print(f"Prompt mode: {args.prompt_mode}", flush=True)

    judge = None if args.dry_run else QwenStyleJudge(
        model_path=args.qwen_vl_model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        allow_download=args.allow_download,
        image_max_side=args.image_max_side,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=args.prompt_mode,
        device_map=args.device_map,
        tp_plan=args.tp_plan,
    )
    if judge is not None and not judge.available and not args.quiet:
        print(f"Qwen VLM unavailable: {judge.load_error}", flush=True)

    rows = score_items(items, judge=judge, dry_run=args.dry_run, quiet=args.quiet)
    if rank != 0:
        return 0
    json_path, csv_path, jsonl_path, summary = write_outputs(
        output_dir=output_dir,
        rows=rows,
        summary_paths=summary_paths,
        args=args,
        project_root=project_root,
    )
    if not args.quiet:
        mean_summary = summary.get("qwen_vlm_j_style_score", {})
        print(f"Wrote JSON:  {json_path}", flush=True)
        print(f"Wrote CSV:   {csv_path}", flush=True)
        print(f"Wrote JSONL: {jsonl_path}", flush=True)
        print(
            f"Available: {summary['available']}/{summary['total']} "
            f"mean_qwen_style={mean_summary.get('mean', 'n/a')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
