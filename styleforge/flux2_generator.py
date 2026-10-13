"""Local FLUX.2-Klein generator backend."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from .config import BackendConfig
from .models import GeneratedImage
from .prompts import compose_generation_prompt
from .paths import repository_path


class Flux2Generator:
    def __init__(self, backend_config: BackendConfig, output_dir: Path, prompt_template: str) -> None:
        self.backend_config = backend_config
        self.output_dir = output_dir
        self.prompt_template = prompt_template
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipe = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._lora_path: Optional[Path] = None
        self._model_traits: Optional[Dict[str, Any]] = None
        self.call_idx = 0

    def generate(
        self,
        style_fragment: str,
        content_prompt: str,
        seed: int,
        n: int,
        params: Dict[str, object],
    ) -> List[GeneratedImage]:
        final_prompt = compose_generation_prompt(style_fragment, content_prompt, self.prompt_template)
        effective_params = dict(params)
        max_sequence_length = int(
            effective_params.get("max_sequence_length")
            or getattr(self.backend_config, "flux2_max_sequence_length", 1024)
            or 1024
        )
        effective_params["max_sequence_length"] = max_sequence_length
        model_traits = self._get_model_traits()
        model_is_distilled = bool(model_traits.get("is_distilled"))
        negative_prompt = str(effective_params.get("negative_prompt") or "").strip()
        cfg_controls = {
            "model_is_distilled": model_is_distilled,
            "guidance_requested": float(effective_params.get("guidance", 1.0)),
            "guidance_effective": not model_is_distilled,
            "negative_prompt_requested": bool(negative_prompt),
            "negative_prompt_effective": bool(negative_prompt) and not model_is_distilled,
            "negative_prompt_skipped_reason": (
                "step-wise distilled FLUX.2-Klein ignores CFG controls"
                if negative_prompt and model_is_distilled
                else ""
            ),
        }
        generated: List[GeneratedImage] = []
        for sample_idx in range(n):
            next_call_idx = self.call_idx + 1
            sample_seed = seed + sample_idx
            path = self.output_dir / f"flux2_gen_{next_call_idx:04d}_seed_{sample_seed}.png"
            metadata_path = path.with_suffix(".metadata.json")
            reusable = self._reuse_existing_image(
                path=path,
                metadata_path=metadata_path,
                final_prompt=final_prompt,
                style_fragment=style_fragment,
                content_prompt=content_prompt,
                sample_seed=sample_seed,
                params=effective_params,
            )
            self.call_idx = next_call_idx
            if reusable is not None:
                generated.append(reusable)
                continue

            pipe, torch, device = self._load_pipeline()
            generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(sample_seed)
            kwargs = {
                "prompt": final_prompt,
                "height": int(params.get("height", 1024)),
                "width": int(params.get("width", 1024)),
                "num_inference_steps": int(params.get("steps", 4)),
                "guidance_scale": float(params.get("guidance", 1.0)),
                "generator": generator,
                "max_sequence_length": max_sequence_length,
            }
            if negative_prompt and not model_is_distilled:
                with torch.no_grad():
                    negative_prompt_embeds, _ = pipe.encode_prompt(
                        prompt=negative_prompt,
                        device=device,
                        num_images_per_prompt=1,
                        max_sequence_length=max_sequence_length,
                    )
                negative_prompt_embeds = negative_prompt_embeds.detach()
                kwargs["negative_prompt_embeds"] = negative_prompt_embeds
            elif negative_prompt and model_is_distilled:
                _log("skip negative_prompt for step-wise distilled FLUX.2-Klein; CFG controls are ignored")
            if self.backend_config.flux2_output_type:
                kwargs["output_type"] = self.backend_config.flux2_output_type
            result = pipe(**kwargs)
            image = result.images[0] if hasattr(result, "images") else result[0]
            image.save(path)
            metadata = {
                "prompt": final_prompt,
                "style_fragment": style_fragment,
                "content_prompt": content_prompt,
                "prompt_template": self.prompt_template,
                "seed": sample_seed,
                "params": dict(effective_params),
                "base_model": repository_path(Path(self.backend_config.flux2_base_model)),
                "model_traits": model_traits,
                "cfg_controls": cfg_controls,
                "lora_path": repository_path(self._lora_path) if self._lora_path else None,
                "device": str(device),
                "dtype": str(self._dtype),
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            generated.append(
                GeneratedImage(
                    local_path=str(path),
                    prompt=final_prompt,
                    seed=sample_seed,
                    params=dict(
                        effective_params,
                        metadata_path=str(metadata_path),
                        style_fragment=style_fragment,
                        content_prompt=content_prompt,
                        prompt_template=self.prompt_template,
                        model_traits=model_traits,
                        cfg_controls=cfg_controls,
                    ),
                )
            )
        return generated

    def _reuse_existing_image(
        self,
        *,
        path: Path,
        metadata_path: Path,
        final_prompt: str,
        style_fragment: str,
        content_prompt: str,
        sample_seed: int,
        params: Dict[str, object],
    ) -> Optional[GeneratedImage]:
        if not path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = {
            "prompt": final_prompt,
            "style_fragment": style_fragment,
            "content_prompt": content_prompt,
            "prompt_template": self.prompt_template,
            "seed": sample_seed,
            "params": dict(params),
            "base_model": repository_path(Path(self.backend_config.flux2_base_model)),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                return None
        return GeneratedImage(
            local_path=str(path),
            prompt=final_prompt,
            seed=sample_seed,
            params=dict(
                params,
                metadata_path=str(metadata_path),
                style_fragment=style_fragment,
                content_prompt=content_prompt,
                prompt_template=self.prompt_template,
            ),
        )

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe, self._torch, self._device

        load_start = time.time()
        try:
            _log("import torch")
            import torch
            _log("import diffusers.Flux2KleinPipeline")
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError(
                "flux2_generator requires torch and diffusers.Flux2KleinPipeline. "
                "Run this backend in the same environment used by inference_test.py."
            ) from exc

        device = self.backend_config.flux2_device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _torch_dtype_from_name(torch, self.backend_config.flux2_dtype, device)
        base_model = self.backend_config.flux2_base_model
        if not base_model:
            raise ValueError("backend.flux2_base_model is empty")

        _log(f"load pipeline from {base_model} dtype={dtype}")
        pipe = Flux2KleinPipeline.from_pretrained(base_model, torch_dtype=dtype)
        _log(f"move pipeline to device={device}")
        pipe.to(device)

        lora_path = self._resolve_lora_path()
        if lora_path is not None:
            try:
                _log(f"load LoRA adapter from {lora_path}")
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("backend.flux2_lora_path was set, but peft is not installed") from exc
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer,
                str(lora_path),
                torch_dtype=dtype,
            ).to(device)

        self._pipe = pipe
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self._lora_path = lora_path
        _log(f"pipeline ready in {time.time() - load_start:.3f}s")
        return self._pipe, self._torch, self._device

    def _resolve_lora_path(self) -> Optional[Path]:
        if not self.backend_config.flux2_lora_path:
            return None
        path = Path(self.backend_config.flux2_lora_path).expanduser()
        if path.name.startswith("lora_gen_step_") and (path / self.backend_config.flux2_adapter_name).exists():
            return path / self.backend_config.flux2_adapter_name
        return path

    def _get_model_traits(self) -> Dict[str, Any]:
        if self._model_traits is None:
            self._model_traits = _flux2_model_traits(self.backend_config.flux2_base_model)
            self._model_traits["model_index"] = repository_path(Path(self._model_traits["model_index"]))
        return dict(self._model_traits)


def _torch_dtype_from_name(torch, name: str, device: str):
    if name == "auto":
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported backend.flux2_dtype: {name}")


def _flux2_model_traits(base_model: str) -> Dict[str, Any]:
    model_dir = Path(str(base_model or "")).expanduser()
    index_path = model_dir / "model_index.json"
    traits: Dict[str, Any] = {
        "model_index": str(index_path),
        "is_distilled": False,
    }
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        traits["model_index_available"] = False
        return traits
    traits["model_index_available"] = True
    traits["is_distilled"] = bool(data.get("is_distilled"))
    class_name = data.get("_class_name")
    if isinstance(class_name, str):
        traits["class_name"] = class_name
    return traits


def _log(message: str) -> None:
    print(f"[styleforge:flux2] {message}", file=sys.stderr, flush=True)
