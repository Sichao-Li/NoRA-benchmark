from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from nora._scoring.data_types import ReasoningInstance, ValidationError


def load_instance(source: str | Path | Mapping[str, Any]) -> ReasoningInstance:
    """Load a single action-rooted support graph instance from a path or mapping."""

    if isinstance(source, Mapping):
        return ReasoningInstance.from_dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raise ValidationError(f"{path} contains multiple instances; use load_instances instead.")
    return ReasoningInstance.from_dict(payload)


def load_instances(source: str | Path | Mapping[str, Any] | list[Mapping[str, Any]]) -> list[ReasoningInstance]:
    """Load one or more instances from a file, directory, or in-memory payload."""

    if isinstance(source, Mapping):
        return [ReasoningInstance.from_dict(source)]
    if isinstance(source, list):
        return [ReasoningInstance.from_dict(item) for item in source]
    path = Path(source)
    if path.is_dir():
        return load_instances_from_dir(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [ReasoningInstance.from_dict(item) for item in payload]
    return [ReasoningInstance.from_dict(payload)]


def load_instances_from_dir(directory: str | Path, pattern: str = "*.json") -> list[ReasoningInstance]:
    """Load all JSON instance files from a directory."""

    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")
    instances: list[ReasoningInstance] = []
    for file_path in sorted(path.glob(pattern)):
        if not file_path.is_file():
            continue
        instances.extend(load_instances(file_path))
    return instances


def dump_instance(instance: ReasoningInstance, destination: str | Path) -> None:
    """Write a single instance to JSON."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance.to_dict(), indent=2), encoding="utf-8")


def dump_json(payload: Any, destination: str | Path) -> None:
    """Serialize arbitrary JSON payloads with stable formatting."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dump_rows_to_csv(rows: Iterable[Mapping[str, Any]], destination: str | Path) -> None:
    """Write rows to CSV, using the union of observed keys as headers."""

    rows_list = list(rows)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows_list:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def index_instances_by_id(instances: Iterable[ReasoningInstance]) -> dict[str, ReasoningInstance]:
    """Index instances by instance_id and reject duplicates."""

    indexed: dict[str, ReasoningInstance] = {}
    for instance in instances:
        if instance.instance_id in indexed:
            raise ValidationError(f"Duplicate instance_id {instance.instance_id!r} in batch input.")
        indexed[instance.instance_id] = instance
    return indexed
