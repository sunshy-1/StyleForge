"""Paths persisted in run artifacts are relative to the repository root."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def repository_path(path: Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate)


def resolve_repository_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate
