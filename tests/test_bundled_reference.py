import json

import pytest

from nora import evaluate, load_prompts, load_references, reconstruct
from nora.adapters import to_instance
from nora.cli import build_parser
from nora.data import digest, read_rows, reference_path


def test_bundled_test_annotations_are_exact_complete_and_offline(monkeypatch):
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda *a, **kw: pytest.fail("test references must be bundled"))
    assert digest(reference_path(offline=True)) == (
        "68f36c07a73fb173b73f6b941aad38ad614f4ebc62c13153c75915fb4d035d9d")
    rows = load_references(offline=True)
    assert len(rows) == len({row["clip_id"] for row in rows}) == 190
    assert [sum(len(row[field]) for row in rows) for field in
            ("facts", "reasons", "actions")] == [1324, 1007, 635]
    for row in rows:
        assert set(row) == {"clip_id", "split", "prompt_variant", "facts", "reasons",
                            "actions", "video_source"}
        assert len(to_instance(row, "annotation").action_graphs) == len(row["actions"])
        assert "actions_observed" not in row
        assert all(set(action) == {"action_id", "description", "reasons_to_do"}
                   for action in row["actions"])


def test_annotation_format_and_bundled_test_are_defaults(tmp_path):
    args = build_parser().parse_args(["evaluate", "--predictions", "input", "--output", "output"])
    assert args.prediction_format == "annotation" and args.references is None
    row = load_references()[0]
    path = tmp_path / "pred.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = evaluate(path, output=tmp_path / "run", _demo=True)
    assert result["coverage"]["expected"] == 190
    assert result["coverage"]["valid"] == 1
    assert result["scores"]["soft"]["reasonableness_score"] == pytest.approx(1)


def test_prompts_are_bundled_and_versioned():
    prompts = load_prompts()
    assert {row["mode"] for row in prompts} == {"direct", "deliberate", "structured"}
    assert all(row["prompt_id"].startswith("nora_v2_") for row in prompts)


def test_reconstruction_uses_same_annotation_path(tmp_path):
    prediction = {"clip_id": "x", "facts": [], "reasons": [], "actions": [
        {"action_id": "A1", "description": "I wait.", "reasons_to_do": []},
        {"action_id": "A2", "description": "I step aside.", "reasons_to_do": []}]}
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"clip_id": "x", "response": "I wait or step aside.",
                                "secret_gold": "must not enter extraction"}) + "\n")
    def client(instructions, content):
        assert set(json.loads(content)) == {"clip_id", "response"}
        assert "secret_gold" not in instructions + content
        return json.dumps(prediction)
    output = tmp_path / "annotations.jsonl"
    assert reconstruct(path, output=output, model="fake", client=client)["invalid"] == 0
    saved = read_rows(output)[0]
    assert saved["prediction"] == prediction
    assert to_instance(saved, "annotation").to_dict() == to_instance(prediction, "annotation").to_dict()
    assert to_instance(saved, "annotation").chosen_action_id is None
    with pytest.raises(FileExistsError):
        reconstruct(path, output=output, model="fake", client=client)


@pytest.mark.parametrize("prediction", [
    {"clip_id": "wrong", "facts": [], "reasons": [], "actions": []},
    {"clip_id": "x", "facts": [], "reasons": [], "actions": [], "chosen_action_id": "A1"},
    {"clip_id": "x", "facts": [], "reasons": [], "actions": [{
        "action_id": "A1", "description": "I wait.", "reasons_to_do": [{"reason_id": "missing"}]}]},
    {"clip_id": "x", "facts": [], "reasons": [], "actions": [{
        "action_id": "A1", "description": "I wait.", "reasons_to_do": [], "reasons_not_to_do": []}]},
])
def test_reconstruction_does_not_repair_invalid_annotations(tmp_path, prediction):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"clip_id":"x","response":"I wait."}\n')
    output = tmp_path / "pred.jsonl"
    result = reconstruct(path, output=output, model="fake", client=lambda *a: json.dumps(prediction))
    assert result["invalid"] == 1
    assert read_rows(output)[0]["status"] == "failed"
    assert "prediction" not in read_rows(output)[0]
