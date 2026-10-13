"""Dataset discovery for style reference tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class StyleTask:
    task_id: str
    root_dir: Path
    ref_images: List[Path]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "root_dir": str(self.root_dir),
            "ref_images": [str(path) for path in self.ref_images],
        }


def discover_tasks(dataset_root: Path) -> List[StyleTask]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    tasks: List[StyleTask] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        target_dir = child / "target_images"
        if not target_dir.is_dir():
            continue
        images = _find_images(target_dir)
        if images:
            tasks.append(StyleTask(task_id=child.name, root_dir=child, ref_images=images))
    return tasks


def load_task(dataset_root: Path, task_id: str) -> StyleTask:
    tasks = {task.task_id: task for task in discover_tasks(dataset_root)}
    if task_id not in tasks:
        available = ", ".join(sorted(tasks)) or "<none>"
        raise ValueError(f"Task not found: {task_id}. Available tasks: {available}")
    return tasks[task_id]


def _find_images(directory: Path) -> List[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
