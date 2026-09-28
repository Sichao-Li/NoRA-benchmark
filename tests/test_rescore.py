import numpy as np
import pytest

from nora.adapters import to_instance
from nora.cli import main
from nora.scoring import CachedCrossEncoder
from nora._scoring import HybridSimilarityScorer, evaluate_instance
from nora._scoring.embedding_backend import SentenceTransformerBackend


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return np.array([0.7 if a == b else 2.5 for a, b in pairs])


def test_batch_cache_preserves_singleton_normalization_and_metrics():
    graph = {"instance_id": "x", "action_graphs": [{
        "action_id": "a", "action_text": "Wait.",
        "facts": [{"id": "f", "text": "I see a door."}],
        "reasons": [{"id": "r", "text": "I avoid a collision."}],
        "edges": [{"src": "f", "dst": "r", "type": "supports"},
                  {"src": "r", "dst": "a", "type": "motivates"}]}]}
    pred = to_instance(graph, "graph")
    graph["action_graphs"][0]["facts"][0]["text"] = "I see another door."
    gold = to_instance(graph, "graph")
    batch_model = FakeModel()
    batch = CachedCrossEncoder(batch_model, "cross-encoder/test")
    serial = SentenceTransformerBackend(model_name="cross-encoder/test", allow_fallback=False)
    serial._model = FakeModel()
    batch.prepare(pred, gold)
    for mode in ("soft", "hungarian"):
        actual = evaluate_instance(pred, gold, HybridSimilarityScorer(backend=batch), mode).to_dict()
        expected = evaluate_instance(pred, gold, HybridSimilarityScorer(backend=serial), mode).to_dict()
        assert actual == expected
    assert len(batch_model.calls) == 1


def test_validate_honors_upstream_failed_status(tmp_path, capsys):
    path = tmp_path / "pred.jsonl"
    path.write_text('{"instance_id":"x","status":"failed","action_graphs":[]}\n')
    assert main(["validate", str(path), "--prediction-format", "graph"]) == 1
    assert "generation_failed" in capsys.readouterr().out


def test_public_scorer_is_independent_of_legacy_packages(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    (tmp_path / "reason_eval.py").write_text('raise RuntimeError("legacy import")')
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run([sys.executable, "-m", "nora.cli", "demo"], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(source)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def supported_graph(prefix):
    return to_instance({"instance_id": "x", "action_graphs": [{
        "action_id": prefix + "a", "action_text": "Wait.",
        "facts": [{"id": prefix + "f", "text": "I see a door."}],
        "reasons": [{"id": prefix + "r", "text": "I avoid a collision.",
                     "normative_label": "safety"}],
        "edges": [{"src": prefix + "f", "dst": prefix + "r", "type": "supports"},
                  {"src": prefix + "r", "dst": prefix + "a", "type": "motivates"}]}]}, "graph")


@pytest.mark.parametrize("mode", ["soft", "hungarian"])
def test_renumbering_preserves_scores_and_empty_prediction_is_poor(mode):
    from nora._scoring.embedding_backend import _HashingFallbackBackend
    scorer = HybridSimilarityScorer(backend=_HashingFallbackBackend())
    gold, pred = supported_graph("old"), supported_graph("new")
    result = evaluate_instance(pred, gold, scorer, mode)
    assert result.reasonableness_score == pytest.approx(1)
    empty = to_instance({"instance_id": "x", "action_graphs": []}, "graph")
    result = evaluate_instance(empty, gold, scorer, mode)
    assert result.action_f1 == result.fact_f1 == result.reasoning_f1 == 0
    assert result.reasonableness_score == pytest.approx(0, abs=1e-10)
    assert result.fact_available and result.graph_available


@pytest.mark.parametrize("edge", [
    {"src": "absent", "dst": "r", "type": "supports"},
    {"src": "f", "dst": "r", "type": "motivates"},
])
def test_invalid_support_edges_rejected(edge):
    payload = supported_graph("").to_dict()
    payload["action_graphs"][0]["edges"] = [edge]
    with pytest.raises(ValueError):
        to_instance(payload, "graph")
