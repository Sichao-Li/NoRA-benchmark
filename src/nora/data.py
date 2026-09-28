"""Bundled test references, pinned training data, and strict JSONL boundaries."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path

DATASET_ID = "MINTLABJHUANU/NoRA"
DATASET_REVISION = "463cb2eb2cf1d7d6e81b38c44ca8b9f72416bd61"
DATA_HASHES = {
    "train": "5c0d6988b88907b2f66d54f9b23788adec3566c7cc0f0cf1a1cc2a3e017ecb58",
    "test": "68f36c07a73fb173b73f6b941aad38ad614f4ebc62c13153c75915fb4d035d9d",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    """Malformed JSONL is a file error, never silently dropped."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else [payload]
    else:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"invalid_jsonl: line {line_number}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("row_not_object")
    return rows


def row_id(row):
    value = row.get("clip_id", row.get("instance_id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_clip_id")
    if value != value.strip():
        raise ValueError("clip_id_has_whitespace")
    if "instance_id" in row and row["instance_id"] != value:
        raise ValueError("conflicting_instance_id")
    return value


def index_rows(rows):
    result = {}
    for row in rows:
        key = row_id(row)
        if key in result:
            raise ValueError(f"duplicate_clip_id: {key}")
        result[key] = row
    return result


def reference_path(split="test", *, offline=False):
    if split not in DATA_HASHES:
        raise ValueError("split must be train or test")
    if split == "test":
        path = Path(str(files("nora") / "assets/nora_test.jsonl"))
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(
            DATASET_ID, "data/nora_train.jsonl", repo_type="dataset",
            revision=DATASET_REVISION, local_files_only=offline,
        ))
    if digest(path) != DATA_HASHES[split]:
        raise ValueError("dataset_checksum_mismatch")
    return path


def load_references(split="test", *, offline=False):
    """Return reference annotation dictionaries for the requested split.

    Test uses bundled annotations; train uses the pinned HF release (downloaded if needed).
    These are reference answers, not model inputs. Pass rows to predict, which
    exposes only clip IDs, pre-action media, and prompts to the model callback.
    """
    rows = read_rows(reference_path(split, offline=offline))
    index_rows(rows)
    return rows


def load_prompts():
    """Return bundled prompt dictionaries for direct, deliberate, and structured modes."""
    return json.loads((files("nora") / "assets/prediction_prompts.json").read_text())["prompts"]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
