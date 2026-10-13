"""Vision-language model client with configurable API protocol."""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import mimetypes
import os
from contextlib import contextmanager
from pathlib import Path
import socket
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple
import urllib.error
import urllib.request

from .config import BackendConfig
from .models import CriticFeedback, IterationState, StyleAxis, StyleDescriptor
from .prompts import build_critic_prompt, build_extract_prompt, build_refine_prompt


class VisionLanguageModelClient:
    def __init__(self, backend_config: BackendConfig, cache_dir: Path) -> None:
        # The calling Bash entry point exports its API settings to this process.
        self.protocol = str(backend_config.vlm_protocol or "openai").strip().lower()
        if self.protocol not in {"anthropic", "openai"}:
            raise ValueError(f"Unsupported VLM protocol: {self.protocol}")
        # Connection settings use the same names for every API protocol.
        self.base_url = os.environ.get("VLM_BASE_URL", "").strip().rstrip("/")
        self.model = os.environ.get("VLM_MODEL", "").strip()
        self.api_key = os.environ.get("VLM_API_KEY", "").strip()
        self.reasoning_effort = os.environ.get("VLM_REASONING_EFFORT", "").strip()
        missing = [
            name for name, value in (
                ("VLM_BASE_URL", self.base_url),
                ("VLM_MODEL", self.model),
                ("VLM_API_KEY", self.api_key),
            ) if not value
        ]
        if missing:
            raise ValueError(
                "Missing " + ", ".join(missing)
                + "; configure the API settings in run_styleforge.sh or "
                "run_training_free_baselines.sh and run that script."
            )
        self.max_tokens = int(backend_config.vlm_max_tokens or 1800)
        self.timeout_s = float(backend_config.vlm_timeout_s or 120)
        self.request_retries = int(backend_config.vlm_request_retries)
        self.stream = bool(backend_config.vlm_stream)
        self.temperature = float(backend_config.vlm_temperature)
        self.use_response_format = bool(backend_config.vlm_response_format)
        self.image_max_side = int(backend_config.vlm_image_max_side)
        self.image_quality = int(backend_config.vlm_image_quality)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_usage: Optional[Dict[str, Any]] = None

    def chat_json(self, content: List[Dict[str, Any]], *, cache_tag: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}

        cache_key = _cache_key(cache_tag, payload)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            response = cached["response"]
        else:
            compatible_cached = self._find_compatible_cached_response(cache_tag, payload)
            if compatible_cached is not None:
                response = compatible_cached["response"]
            else:
                response = self._post(payload)
            cache_path.write_text(
                json.dumps(
                    {"cache_tag": cache_tag, "payload": _payload_without_image_bytes(payload), "response": response},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.last_usage = response.get("usage") if isinstance(response, dict) else None
        answer = _extract_answer(response)
        return parse_json_object(answer), {"raw_answer": answer, "usage": self.last_usage}

    def _find_compatible_cached_response(
        self,
        cache_tag: str,
        payload: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        current_payload = _payload_for_cache_match(_payload_without_image_bytes(payload))
        for path in sorted(self.cache_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if cached.get("cache_tag") != cache_tag:
                continue
            cached_payload = _payload_for_cache_match(cached.get("payload", {}))
            if cached_payload == current_payload and isinstance(cached.get("response"), dict):
                return cached
        return None

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.protocol == "anthropic":
            return self._post_anthropic(payload)
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        request_payload = dict(payload, stream=self.stream)
        payload_bytes = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        attempts = max(1, self.request_retries + 1)
        last_error: Optional[BaseException] = None
        for attempt in range(attempts):
            request = urllib.request.Request(
                endpoint,
                data=payload_bytes,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "text/event-stream" if self.stream else "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "StyleForge/1.0",
                },
                method="POST",
            )
            started = time.time()
            print(
                f"[vlm] request {attempt + 1}/{attempts}: model={self.model}, "
                f"stream={self.stream}, timeout={self.timeout_s:g}s",
                flush=True,
            )
            try:
                with _request_progress():
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                        print(f"[vlm] response headers received after {time.time() - started:.1f}s", flush=True)
                        if "text/event-stream" in response.headers.get("Content-Type", "").lower():
                            result = _read_chat_stream(response)
                        else:
                            result = json.loads(response.read().decode("utf-8"))
                print(
                    f"[vlm] response complete after {time.time() - started:.1f}s "
                    f"({len(_extract_answer(result))} characters)",
                    flush=True,
                )
                return result
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                elapsed = time.time() - started
                if _is_retriable_http_status(error.code) and attempt + 1 < attempts:
                    last_error = error
                    print(f"[vlm] HTTP {error.code} after {elapsed:.1f}s; retrying", flush=True)
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(
                    f"VLM request failed: HTTP {error.code} {error.reason} after {elapsed:.3f}s "
                    f"(payload={len(payload_bytes)} bytes): {body[:1000]}"
                ) from error
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                http.client.HTTPException,
            ) as error:
                # Includes http.client.RemoteDisconnected (a ConnectionResetError
                # / BadStatusLine), which the server raises intermittently on the
                # larger multi-image critic payloads. These are transient, so we
                # back off and retry instead of crashing the whole run.
                last_error = error
                if attempt + 1 < attempts:
                    print(
                        f"[vlm] {type(error).__name__} after {time.time() - started:.1f}s: "
                        f"{getattr(error, 'reason', error)}; retrying",
                        flush=True,
                    )
                    time.sleep(min(2 ** attempt, 8))
                    continue
                elapsed = time.time() - started
                reason = getattr(error, "reason", error)
                raise RuntimeError(
                    f"VLM request failed after {attempts} attempt(s), last wait {elapsed:.3f}s "
                    f"(timeout={self.timeout_s}s, payload={len(payload_bytes)} bytes): {reason}"
                ) from error
        raise RuntimeError(f"VLM request failed: {last_error}")

    def _post_anthropic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = self.base_url.rstrip("/") + "/v1/messages"
        request_payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": payload["messages"],
        }
        if self.stream:
            request_payload["stream"] = True
        payload_bytes = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        attempts = max(1, self.request_retries + 1)
        last_error: Optional[BaseException] = None
        for attempt in range(attempts):
            request = urllib.request.Request(
                endpoint,
                data=payload_bytes,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Accept": "text/event-stream" if self.stream else "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "StyleForge/1.0",
                },
                method="POST",
            )
            started = time.time()
            print(f"[vlm] Anthropic request {attempt + 1}/{attempts}: model={self.model}, stream={self.stream}", flush=True)
            try:
                with _request_progress():
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                        print(f"[vlm] response headers received after {time.time() - started:.1f}s", flush=True)
                        if "text/event-stream" in response.headers.get("Content-Type", "").lower():
                            result = _read_anthropic_stream(response)
                        else:
                            result = _normalize_anthropic_response(json.loads(response.read().decode("utf-8")))
                print(f"[vlm] response complete after {time.time() - started:.1f}s ({len(_extract_answer(result))} characters)", flush=True)
                return result
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                if error.code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    last_error = error
                    print(f"[vlm] HTTP {error.code}; retrying", flush=True)
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"Anthropic request failed: HTTP {error.code}: {body[:1000]}") from error
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException) as error:
                last_error = error
                if attempt + 1 < attempts:
                    print(f"[vlm] {type(error).__name__}; retrying", flush=True)
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"Anthropic request failed after {attempts} attempts: {error}") from error
        raise RuntimeError(f"Anthropic request failed: {last_error}")


def _is_retriable_http_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


@contextmanager
def _request_progress():
    """Report long requests, including waits before headers or between chunks."""
    stopped = threading.Event()
    started = time.monotonic()

    def report() -> None:
        while not stopped.wait(30):
            print(f"[vlm] waiting for complete response: {time.monotonic() - started:.0f}s elapsed", flush=True)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        yield
    finally:
        stopped.set()
        reporter.join()


def _stream_events(response):
    """Read SSE data fields, including comments, CRLF, and multiline events."""
    data_lines = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
        elif line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield "\n".join(data_lines)


def _read_chat_stream(response) -> Dict[str, Any]:
    """Assemble one streamed completion into the usual cached response format.

    A connection ending without a completion marker is retryable; partial
    content must never be cached or used as a DSE style description.
    """
    parts: List[str] = []
    result: Dict[str, Any] = {}
    finish_reason = None
    done = False
    for raw_event in _stream_events(response):
        if raw_event.strip() == "[DONE]":
            done = True
            break
        try:
            event = json.loads(raw_event)
        except json.JSONDecodeError as error:
            raise ConnectionError("Incomplete or invalid VLM stream event") from error
        if event.get("error"):
            raise RuntimeError(f"VLM stream error: {event['error']}")
        for key in ("id", "model", "created", "usage"):
            if event.get(key) is not None:
                result[key] = event[key]
        for choice in event.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    part["text"] for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if not done and finish_reason is None:
        raise ConnectionError(
            f"VLM stream disconnected before completion ({sum(map(len, parts))} characters received)"
        )
    result["choices"] = [{
        "index": 0,
        "message": {"role": "assistant", "content": "".join(parts)},
        "finish_reason": finish_reason,
    }]
    return result


def _normalize_anthropic_response(data: Dict[str, Any]) -> Dict[str, Any]:
    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "usage": data.get("usage"),
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": data.get("stop_reason")}],
    }


def _read_anthropic_stream(response) -> Dict[str, Any]:
    parts: List[str] = []
    result: Dict[str, Any] = {}
    done = False
    stop_reason = None
    for raw_event in _stream_events(response):
        if raw_event.strip() == "[DONE]":
            done = True
            break
        try:
            event = json.loads(raw_event)
        except json.JSONDecodeError as error:
            raise ConnectionError("Incomplete or invalid Anthropic stream event") from error
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message") or {}
            result.update({key: message.get(key) for key in ("id", "model", "usage") if message.get(key) is not None})
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                parts.append(str(delta.get("text") or ""))
        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason") or stop_reason
            if event.get("usage"):
                result["usage"] = event["usage"]
        elif event_type == "message_stop":
            done = True
    if not done:
        raise ConnectionError(f"Anthropic stream disconnected before completion ({sum(map(len, parts))} characters received)")
    result["choices"] = [{"message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": stop_reason}]
    return result


class VisionStyleAgent:
    def __init__(
        self,
        client: VisionLanguageModelClient,
        *,
        max_json_retries: int,
    ) -> None:
        self.client = client
        self.max_json_retries = max_json_retries

    def extract(
        self,
        ref_images: List[Path],
        axes: List[StyleAxis],
        reference_color_contract: Optional[Dict[str, Any]] = None,
    ) -> StyleDescriptor:
        prompt = build_extract_prompt([str(path) for path in ref_images], axes, reference_color_contract)
        content = _content_for_prompt(
            prompt_text=_extractor_instruction(prompt),
            images=ref_images,
            image_max_side=self.client.image_max_side,
            image_quality=self.client.image_quality,
        )
        data = self._call_json_with_retries(content, cache_tag="extract")
        return StyleDescriptor.from_dict(data, axes)

    def refine(
        self,
        prev_descriptor: StyleDescriptor,
        critic_feedback: CriticFeedback,
        history: List[IterationState],
        ref_images: List[Path],
        axes: List[StyleAxis],
        reference_color_contract: Optional[Dict[str, Any]] = None,
    ) -> StyleDescriptor:
        prompt = build_refine_prompt(
            prev_descriptor,
            critic_feedback,
            history,
            [str(path) for path in ref_images],
            axes,
            reference_color_contract=reference_color_contract,
        )
        content = _content_for_prompt(
            prompt_text=_refiner_instruction(prompt),
            images=ref_images,
            image_max_side=self.client.image_max_side,
            image_quality=self.client.image_quality,
        )
        data = self._call_json_with_retries(content, cache_tag=f"refine_{len(history)}")
        return StyleDescriptor.from_dict(data, axes)

    def _call_json_with_retries(self, content: List[Dict[str, Any]], *, cache_tag: str) -> Dict[str, Any]:
        retry_content = list(content)
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.max_json_retries)):
            try:
                data, _meta = self.client.chat_json(retry_content, cache_tag=f"{cache_tag}_{attempt}")
                return data
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                retry_content = list(content) + [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response was invalid JSON for the required schema. "
                            f"Return only one JSON object. Error: {exc}"
                        ),
                    }
                ]
        raise RuntimeError(f"VLM style agent failed to return valid JSON: {last_error}") from last_error


class VisionStyleCritic:
    def __init__(
        self,
        client: VisionLanguageModelClient,
        *,
        max_json_retries: int,
    ) -> None:
        self.client = client
        self.max_json_retries = max_json_retries

    def evaluate(
        self,
        ref_images: List[Path],
        gen_image,
        axes: List[StyleAxis],
        prev_feedback: Optional[CriticFeedback],
    ) -> CriticFeedback:
        prompt = build_critic_prompt(
            [str(path) for path in ref_images],
            gen_image.local_path,
            axes,
            prev_feedback,
        )
        content = _content_for_prompt(
            prompt_text=_critic_instruction(
                prompt,
                generated_prompt=gen_image.prompt,
                content_prompt=str(gen_image.params.get("content_prompt") or ""),
            ),
            images=list(ref_images) + [Path(gen_image.local_path)],
            labels=[f"reference image {idx + 1}" for idx in range(len(ref_images))] + ["generated image"],
            image_max_side=self.client.image_max_side,
            image_quality=self.client.image_quality,
        )
        retry_content = list(content)
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.max_json_retries)):
            try:
                data, _meta = self.client.chat_json(retry_content, cache_tag=f"critic_{Path(gen_image.local_path).name}_{attempt}")
                return CriticFeedback.from_dict(data, axes)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                retry_content = list(content) + [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response was invalid JSON for the required CriticFeedback schema. "
                            f"Return only one JSON object. Error: {exc}"
                        ),
                    }
                ]
        raise RuntimeError(f"VLM critic failed to return valid JSON: {last_error}") from last_error


class VisionColorContractAgent:
    """Final-stage v13.1 lighting/color-grading repair agent.

    The runner treats this as an offline prompt-engineering step: it can inspect
    reference images and a few generated samples, but the output must be a single
    reusable, content-agnostic lighting/color-grading recipe for future
    inference.
    """

    def __init__(
        self,
        client: VisionLanguageModelClient,
        *,
        max_json_retries: int,
    ) -> None:
        self.client = client
        self.max_json_retries = max_json_retries

    def repair(
        self,
        ref_images: List[Path],
        descriptor: StyleDescriptor,
        generated_images: List[Any],
        axes: List[StyleAxis],
        sample_metrics: List[Dict[str, Any]],
        *,
        max_chars: int,
    ) -> Tuple[StyleDescriptor, Dict[str, Any]]:
        prompt_text = _color_contract_instruction(
            descriptor=descriptor,
            generated_images=generated_images,
            axes=axes,
            sample_metrics=sample_metrics,
            max_chars=max_chars,
        )
        images = list(ref_images) + [Path(item.local_path) for item in generated_images]
        labels = [f"reference image {idx + 1}" for idx in range(len(ref_images))]
        for idx, image in enumerate(generated_images):
            content_prompt = str(getattr(image, "params", {}).get("content_prompt") or "")
            labels.append(f"generated sample {idx + 1}; content prompt: {content_prompt}")
        content = _content_for_prompt(
            prompt_text=prompt_text,
            images=images,
            labels=labels,
            image_max_side=self.client.image_max_side,
            image_quality=self.client.image_quality,
        )
        retry_content = list(content)
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.max_json_retries)):
            try:
                data, _meta = self.client.chat_json(
                    retry_content,
                    cache_tag=f"final_color_contract_repair_{attempt}",
                )
                repaired_payload = data.get("repaired_descriptor")
                if not isinstance(repaired_payload, Mapping):
                    raise ValueError("missing repaired_descriptor object")
                repaired = StyleDescriptor.from_dict(repaired_payload, axes)
                lighting_recipe_prompt = data.get("lighting_recipe_prompt") or data.get("color_contract_prompt")
                if not isinstance(lighting_recipe_prompt, str) or not lighting_recipe_prompt.strip():
                    raise ValueError("missing non-empty lighting_recipe_prompt")
                if "lighting_recipe_prompt" not in data:
                    data["lighting_recipe_prompt"] = lighting_recipe_prompt
                return repaired, data
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                retry_content = list(content) + [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response was invalid JSON for the required final lighting/color-grading recipe schema. "
                            f"Return only one JSON object. Error: {exc}"
                        ),
                    }
                ]
        raise RuntimeError(f"VLM color contract agent failed to return valid JSON: {last_error}") from last_error


def _extractor_instruction(prompt: Mapping[str, Any]) -> str:
    color_contract = _reference_color_contract_instruction(prompt)
    return (
        f"{prompt['system']}\n\n"
        f"{prompt['instruction']}\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "global_summary": "style-only summary shared by all references",\n'
        '  "per_axis": {"axis_name": "axis-specific style description for every enabled axis"},\n'
        '  "rendered_prompt_fragment": "the final, vivid, generator-facing style description (see structure below)"\n'
        "}\n\n"
        "Enabled axes:\n"
        + json.dumps(prompt["axes"], ensure_ascii=False, indent=2)
        + color_contract
        + "\n\n"
        + _style_fragment_spec(has_color_contract=bool(color_contract))
    )


def _style_fragment_spec(*, has_color_contract: bool = False) -> str:
    """Shared spec for the generator-facing rendered_prompt_fragment.

    The descriptor must be reference-derived rather than category-derived. The
    prompt should describe the visual relations that repeat across the reference
    set, not force the style into a preset family with canned technique words.
    """

    color_rule = (
        "Use the provided reference color contract to disambiguate repeated style colors. When a palette or lighting "
        "role uses one of those anchors, include its exact #RRGGBB code after the color name. Do not invent hex codes; "
        "do not force one-off object colors into future content.\n"
        if has_color_contract
        else ""
    )
    hex_rule = (
        "Allowed raw hex colors are only the exact codes from the reference color contract; no numeric scores, and no "
        "reference-image subject nouns. "
        if has_color_contract
        else "No raw hex colors, no numeric scores, and no reference-image subject nouns. "
    )
    return (
        "Write rendered_prompt_fragment as one vivid, concrete English style paragraph (about 110 to 180 words) "
        "that a text-to-image model can follow with no reference image. Derive every claim from repeated visual "
        "relations across the reference set. Do NOT name or describe any concrete subject, object, character, "
        "scene, logo, or identity from the reference images; describe only content-adaptive style rules that "
        "generalize to ANY future content. Do not force the style into a canned family; only name a medium or "
        "genre when it is visually necessary and supported by the references. Cover these in order:\n"
        "1) STYLE IDENTITY: give a compact style identity based on the shared rendering behavior, not on the "
        "reference subject matter or a memorized preset label.\n"
        "2) PALETTE RELATIONS: describe the repeated color system with real, specific color names, saturation, "
        "value range, dominant/accent/neutral roles, and how color roles behave across highlights, shadows, "
        "edges, backgrounds, and focal forms. If the palette is opponent, analogous, monochrome, pastel, "
        "low-key, high-key, multi-hue, or sparse-accent, say so because the references show it. Do not reduce "
        "the style to a flat list of colors, and do not impose a warm/cool frame unless the references actually "
        "use that relation.\n"
        + color_rule
        + "3) SIGNATURE STRUCTURE: identify the repeated structural rules that make the style recognizable: edge "
        "hardness, line presence or absence, shape simplification, region size, pattern repetition, texture "
        "frequency, surface finish, artifacts, grain, blockiness, contour behavior, spatial layering, and any "
        "other visible marks. Name concrete observed mechanisms, but do not import technique names that are not "
        "visible in the references.\n"
        "4) LIGHTING AND VALUE ROLES: describe how light, shadow, highlight, rim/edge light, ambient fill, "
        "contrast, bloom/haze, and black/white levels are organized. Tie color to lighting roles when the "
        "references do so; avoid confusing local object color with style lighting.\n"
        "5) RENDERING AND FORM: describe dimensionality, shading model, material abstraction, texture/grain, "
        "edge treatment, detail density, proportions, and how forms remain readable across new content.\n"
        "6) MOOD AND DESIGN DISCIPLINE: describe the shared emotional tone and restraint/energy level as visual "
        "rules, not story content.\n"
        "7) AVOID CLAUSE: end with a short 'Avoid: ...' listing only visual looks that contradict the observed "
        "reference relations or common base-model drifts for this descriptor. Avoid terms must remain style-only "
        "and content-agnostic.\n"
        "Be concrete and evocative; avoid vague analytical jargon. No bullet lists, no JSON words. "
        + hex_rule
        + "In per_axis, give color_palette, "
        "lighting, texture and brushwork the same reference-derived relation treatment."
    )


def _refiner_instruction(prompt: Mapping[str, Any]) -> str:
    color_contract = _reference_color_contract_instruction(prompt)
    return (
        f"{prompt['system']}\n\n"
        f"{prompt['instruction']}\n\n"
        "Previous descriptor:\n"
        + json.dumps(prompt["prev_descriptor"], ensure_ascii=False, indent=2)
        + "\n\nCritic feedback:\n"
        + json.dumps(prompt["critic_feedback"], ensure_ascii=False, indent=2)
        + "\n\nCompact history:\n"
        + json.dumps(prompt["history"], ensure_ascii=False, indent=2)
        + color_contract
        + "\n\nRequired JSON schema is exactly the same as extraction: global_summary, per_axis, rendered_prompt_fragment.\n"
        "Refine by REWRITING one clean generator-facing style paragraph, not by appending every previous edit. "
        "Use the critic's binary improvement_plan as the primary instruction. "
        "If style_match is 0, rewrite the descriptor to fix the named visual gaps in style identity, line/edge treatment, "
        "shape/proportion language, material or brush finish, lighting/value behavior, palette roles, texture/detail density, "
        "and composition rhythm. "
        "Preserve traits that the critic did not question and change only style wording, not sample content. "
        "Do not overfit to the current sample object or scene; keep everything content-agnostic so it generalizes. "
        "Any objective_color metrics in the history are only a soft calibration hint, not semantic proof; "
        "use them to adjust reference-derived palette roles, hue coverage, saturation, value, and light/shadow "
        "role consistency; never hardcode a reference image's literal object colors or override plausible local "
        "colors of new subjects.\n\n"
        + _style_fragment_spec(has_color_contract=bool(color_contract))
    )


def _reference_color_contract_instruction(prompt: Mapping[str, Any]) -> str:
    contract = prompt.get("reference_color_contract")
    if not isinstance(contract, Mapping) or not contract.get("available"):
        return ""
    text = str(contract.get("prompt_text") or "").strip()
    if not text:
        return ""
    version = str(contract.get("version") or "")
    if version.startswith("salient_color_contract"):
        return (
            "\n\nSalient color contract from deterministic pixel tool:\n"
            + json.dumps(contract, ensure_ascii=False, indent=2)
            + "\n\nThis is an adaptive set of high-perceptual-weight style colors selected from reference complexity. "
            "Use only these exact hex codes when the descriptor names one of the listed palette or lighting roles. "
            "Do not invent new hex codes, do not enumerate nearby variants, and do not expand this into colors outside "
            "the contract. Describe the relation among all listed anchors: dominant fields, visible accents, support "
            "colors, bright planes, and shadow structure as applicable. If the contract contains multiple support colors, "
            "keep them visibly recurring in the style prompt instead of compressing the style to one dominant/accent pair. "
            "Preserve plausible local object colors for future content; these anchors describe style color roles, not fixed "
            "object identity. During refinement, keep the same color contract and only adjust increase/decrease wording for "
            "these existing anchors."
        )
    return (
        "\n\nReference color contract from deterministic pixel tool:\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
        + "\n\nUse this contract to reduce vague color-word ambiguity. "
        "When naming a repeated palette or lighting color, include the nearest provided hex code exactly, "
        "for example vivid orange #F07825 for rim light. Do not invent new hex codes. "
        "Assign anchors to style roles such as dominant fields, accents, highlights, midtones, shadows, rim light, "
        "glow, reflections, linework, or background planes. Do not force every future object to these colors."
    )


def _critic_instruction(
    prompt: Mapping[str, Any],
    *,
    generated_prompt: str,
    content_prompt: str,
) -> str:
    return (
        f"{prompt['system']}\n\n"
        f"{prompt['instruction']}\n\n"
        "Generated image prompt:\n"
        f"{generated_prompt}\n\n"
        "Generated content prompt only:\n"
        f"{content_prompt or '<unknown>'}\n\n"
        "Calibration:\n"
        + json.dumps(prompt["calibration"], ensure_ascii=False, indent=2)
        + "\n\nPrevious feedback:\n"
        + json.dumps(prompt["prev_feedback"], ensure_ascii=False, indent=2)
        + "\n\nEnabled axes:\n"
        + json.dumps(prompt["axes"], ensure_ascii=False, indent=2)
        + "\n\nRequired JSON schema:\n"
        "{\n"
        '  "style_match": 0,\n'
        '  "improvement_plan": "if style_match is 0, concrete style-only edits for the next prompt; if 1, say no further edit needed",\n'
        '  "overall_comment": "brief explanation of the binary decision"\n'
        "}\n\n"
        "Binary decision policy:\n"
        "- Be a strict same-style judge. Output style_match = 1 only when the generated image would plausibly belong in the same style dataset as the references. "
        "It does not need the same subject, but it must share the same style identity, rendering language, line/edge treatment, shape/proportion language, material or brush finish, lighting/value behavior, palette roles, texture/detail density, and composition rhythm. "
        "If a person could immediately tell that the generated image is a different subtype, different production style, different drawing/rendering language, or only a loose color imitation, output 0.\n"
        "- Output style_match = 0 when the image merely shares colors, broad genre words, or a few superficial traits but is visibly a different style. "
        "Specific automatic 0 cases: toy-like/chibi proportions when the references use cinematic character proportions; flat vector or poster language when the references use painterly/anime/3D rendering; black-silhouette or low-key poster lighting when the references use readable illustrated forms; missing linework, brush/texture artifacts, surface finish, detail density, or camera/framing language that defines the references. "
        "When uncertain, choose 0 and describe what must change.\n"
        "- Do not output decimal scores, grades, or per-axis numeric ratings. The only convergence signal is style_match 0 or 1.\n\n"
        "Ignore content differences. Judge only the visual style. "
        "Do not penalize the generated image for not containing the same subjects as the references. "
        "For composition, judge framing language, focal hierarchy, spatial depth, rhythm, and negative-space behavior appropriate to the generated content. "
        "For subject_treatment, judge realism level, abstraction, proportions, simplification, and shape language without requiring the generated subject to become a reference subject. "
        "For color_palette, judge the reference-derived palette relations while allowing plausible local colors for new content. "
        "If the palette is off, say which observed color roles are missing or reassigned: dominant fields, accents, neutrals, "
        "highlight colors, shadow colors, edge/rim colors, or focal-form colors. Distinguish a real style color role from a "
        "new object's local color. "
        "For lighting, judge light direction, key level, value range, contrast, highlight behavior, edge/rim light, ambient fill, "
        "shadow softness/color, and whether colors occupy the same light/shadow roles seen in the references. "
        "Color and lighting are the traits humans notice most, so be especially precise and demanding on those two axes. "
        "For texture and brushwork, infer the reference style's own repeated signature structure - edge hardness, linework, "
        "shape simplification, texture frequency, artifacts, material finish, grain, blockiness, contour rules, and region size. "
        "If the generated image lacks those observed structures, set style_match to 0 and name the missing observed mechanisms. "
        "Do not judge by a fixed checklist of known styles; derive the checklist from the references in front of you. "
        "improvement_plan must be concise but complete enough for the refiner to rewrite the generator prompt. "
        "It must name the most visible reference-vs-generated gaps and concrete prompt-actionable fixes. "
        "Avoid sample-specific objects, raw hex colors, numeric scores, metric names, and style-specific hardcoded rules."
    )


def _color_contract_instruction(
    *,
    descriptor: StyleDescriptor,
    generated_images: List[Any],
    axes: List[StyleAxis],
    sample_metrics: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    samples = []
    for idx, image in enumerate(generated_images):
        params = getattr(image, "params", {}) or {}
        samples.append(
            {
                "sample_index": idx,
                "content_prompt": params.get("content_prompt"),
                "seed": getattr(image, "seed", None),
                "compiled_prompt": getattr(image, "prompt", ""),
                "diagnostics": sample_metrics[idx] if idx < len(sample_metrics) else {},
            }
        )
    return (
        "You are the v13.1 final lighting/color-grading recipe repair agent for a style prompt pipeline.\n\n"
        "Goal: improve color and lighting style fidelity using natural-language prompt engineering only. "
        "You will see reference images and generated sample images made from the current final descriptor. "
        "Extract the reusable lighting/color-grading recipe that the references share, identify the generated samples' "
        "lighting and color drift, and rewrite only the descriptor's color and lighting wording. This is an offline one-time "
        "repair step: the final prompt must later work for arbitrary new content without seeing references or "
        "performing any extra color tuning.\n\n"
        "Hard constraints:\n"
        "- Derive every color rule from repeated reference evidence, not from style names, dataset ids, or presets.\n"
        "- Stay content-agnostic. Do not name or require reference subjects, sample subjects, objects, scenes, identities, or logos.\n"
        "- Do not use per-style if/else logic, canned style patches, raw hex colors, palette lists copied from objects, metric names, or numeric scores.\n"
        "- Preserve all non-color traits unless they are inseparable from color/lighting roles: composition, camera, shape language, linework, material finish, texture, proportions, and mood discipline.\n"
        "- Repair only palette roles, key/fill/rim/bounce/reflection/shadow/haze/bloom/grade roles, saturation/value policy, local-color preservation policy, and common lighting/color drift to avoid.\n"
        "- Distinguish style lighting/color-grading roles from local object color. New future content must remain semantically plausible; the recipe should not force semantic objects into reference-object colors.\n"
        "- Be explicit about WHERE the reference-derived hues live: key light, ambient fill, shadow masses, rim/edge light, bounce light, reflection/specular response, haze/bloom, background planes, broad value regions, and overall color grade.\n"
        "- Do not solve palette fidelity by making clothing, skin, hair, faces, products, props, text, or object identity carry the palette. That overfits to local object color and fails on future content.\n"
        "- Do allow the reference-derived colored lights and grade to visibly tint highlights, shadows, edges, surface turns, glossy reflections, and environment planes across both foreground subjects and background. The model should read colored light, not recolored objects.\n"
        f"- lighting_recipe_prompt must be <= {int(max_chars)} characters and should be 2 compact English sentences: one sentence for lighting/color-grading roles, one sentence for local-color preservation plus drift to avoid.\n\n"
        "Current descriptor:\n"
        + json.dumps(descriptor.to_dict(), ensure_ascii=False, indent=2)
        + "\n\nGenerated samples and diagnostics:\n"
        + json.dumps(samples, ensure_ascii=False, indent=2)
        + "\n\nEnabled axes:\n"
        + json.dumps([axis.to_dict() for axis in axes], ensure_ascii=False, indent=2)
        + "\n\nRequired JSON schema:\n"
        "{\n"
        '  "lighting_recipe_match": 0,\n'
        '  "reference_lighting_color_grading_recipe": {\n'
        '    "dominant_roles": "dominant color roles and value/saturation behavior",\n'
        '    "accent_roles": "accent and focal color roles",\n'
        '    "key_light_roles": "key light color, strength, direction, and value behavior",\n'
        '    "ambient_fill_roles": "ambient fill or environmental cast behavior",\n'
        '    "highlight_roles": "highlight, glow, specular, bloom, or lit-plane color roles",\n'
        '    "shadow_roles": "shadow hue, saturation, softness, and low-value behavior",\n'
        '    "edge_or_rim_roles": "rim, contour, edge, or outline color roles",\n'
        '    "bounce_reflection_roles": "bounce light, reflection, glossy surface, and color-bleed behavior",\n'
        '    "background_environment_roles": "background planes, atmosphere, haze, or broad field color roles",\n'
        '    "neutral_roles": "neutral, white, black, gray, paper, or background roles",\n'
        '    "saturation_value_policy": "how saturation and value are distributed",\n'
        '    "local_color_policy": "how future content local colors should be preserved, shifted, or stylized without turning semantic objects into palette swatches"\n'
        "  },\n"
        '  "generated_drift_pattern": "what the samples get wrong in reusable lighting/color-grading terms",\n'
        '  "prompt_missing_rules": "which reusable rules are missing or underweighted in the current descriptor",\n'
        '  "sample_specific_details_to_ignore": "content details from samples/references that must not enter the final prompt",\n'
        '  "generalizable_repair": "the reusable repair logic, style-only and content-agnostic",\n'
        '  "lighting_recipe_prompt": "2 compact sentences to insert at the front of the final prompt; assign hues to lighting/color-grading roles and surfaces; no bullets, no hex, no metric names",\n'
        '  "repaired_descriptor": {\n'
        '    "global_summary": "copy the original unless a color/lighting phrase truly must be clarified",\n'
        '    "per_axis": {"axis_name": "complete axis text for every enabled axis; only color_palette and lighting should materially change"},\n'
        '    "rendered_prompt_fragment": "full final style paragraph with the color contract integrated and all non-color traits preserved"\n'
        "  }\n"
        "}\n\n"
        "Decision policy: set lighting_recipe_match=1 only if generated samples already follow the reference lighting/color-grading recipe. "
        "Even then, return a concise lighting_recipe_prompt that preserves the current correct recipe. "
        "If samples drift, make the repair precise but not brittle: name color roles and relationships, not image-specific colors. "
        "The repaired_descriptor must satisfy the same generator-facing fragment spec below:\n\n"
        + _style_fragment_spec()
    )


def _content_for_prompt(
    *,
    prompt_text: str,
    images: List[Path],
    labels: Optional[List[str]] = None,
    image_max_side: int = 768,
    image_quality: int = 85,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for index, image in enumerate(images):
        label = labels[index] if labels and index < len(labels) else f"image {index + 1}"
        content.extend(
            [
                {"type": "text", "text": label},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(
                            image,
                            max_side=image_max_side,
                            quality=image_quality,
                        )
                    },
                },
            ]
        )
    return content


def image_data_url(path: Path, *, max_side: int = 768, quality: int = 85) -> str:
    compressed = _compressed_jpeg_bytes(path, max_side=max_side, quality=quality)
    if compressed is not None:
        encoded = base64.b64encode(compressed).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _compressed_jpeg_bytes(path: Path, *, max_side: int, quality: int) -> Optional[bytes]:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if max_side > 0:
                image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=max(1, min(int(quality), 95)),
                optimize=True,
            )
            return buffer.getvalue()
    except Exception:
        return None


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        depth = 0
        end = -1
        in_string = False
        escape = False
        for idx, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end < 0:
            raise
        data = json.loads(text[start:end])
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _extract_answer(response: Mapping[str, Any]) -> str:
    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part.strip() for part in parts if part.strip())
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()
    return ""


def _cache_key(cache_tag: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(cache_tag.encode("utf-8"))
    digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _payload_without_image_bytes(payload: Mapping[str, Any]) -> Dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key == "url" and isinstance(item, str) and item.startswith("data:"):
                    out[key] = item[:64] + f"...<data-url-bytes:{len(item)}>"
                else:
                    out[key] = clean(item)
            return out
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(dict(payload))


def _payload_for_cache_match(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _payload_for_cache_match(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_payload_for_cache_match(item) for item in payload]
    if isinstance(payload, str):
        if payload.startswith("data:") or "...<data-url-bytes:" in payload:
            return "<data-url>"
    return payload
