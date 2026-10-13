from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_IMAGE_KEYS = (
    "local_path_list",
    "local_paths",
    "image_path_list",
    "image_paths",
    "image_path",
    "path",
)


def _first_path(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("image path list is empty")
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid image path value: {value!r}")
    return value


def resolve_image_path(image_path: str, jsonl_path: str | Path | None = None, data_root: str | Path | None = None) -> Path:
    path = Path(image_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    bases: list[Path] = []
    if data_root is not None:
        bases.append(Path(data_root).expanduser())
    if jsonl_path is not None:
        bases.append(Path(jsonl_path).expanduser().resolve().parent)
    bases.extend([PROJECT_ROOT, Path.cwd()])

    seen: set[Path] = set()
    candidates: list[Path] = []
    for base in bases:
        candidate = (base / path).resolve()
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
            if candidate.exists():
                return candidate

    return candidates[0] if candidates else path.resolve()


def get_local_image_path(
    item: Mapping[str, Any],
    jsonl_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> str:
    for key in LOCAL_IMAGE_KEYS:
        if key in item:
            return str(resolve_image_path(_first_path(item[key]), jsonl_path=jsonl_path, data_root=data_root))

    if "oss_path_list" in item:
        raw_path = _first_path(item["oss_path_list"])
        if raw_path.startswith("oss://"):
            raise ValueError("expected local image path, got OSS path")
        return str(resolve_image_path(raw_path, jsonl_path=jsonl_path, data_root=data_root))

    raise KeyError(f"item must contain one of {LOCAL_IMAGE_KEYS}")


def resolve_existing_path(path: str | Path, base_dir: str | Path = PROJECT_ROOT) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute() or resolved_path.exists():
        return resolved_path.resolve()

    base_candidate = (Path(base_dir).expanduser() / resolved_path).resolve()
    if base_candidate.exists():
        return base_candidate

    return resolved_path.resolve()
