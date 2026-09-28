from __future__ import annotations

from pathlib import Path
from importlib.resources import files

from nora._scoring import (
    HybridSimilarityScorer,
    build_sentence_transformer_backend,
    evaluate_instance,
    load_instance,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = files("nora") / "assets"
FIXTURES = Path(__file__).parent / "fixtures"


def build_fallback_scorer() -> HybridSimilarityScorer:
    backend = build_sentence_transformer_backend(
        model_name="missing-local-model",
        cache_root=ROOT / ".reason_eval_embedding_cache",
        local_files_only=True,
        allow_fallback=True,
    )
    return HybridSimilarityScorer(backend=backend)


def test_grounded_demo_scores_above_generic_demo() -> None:
    scorer = build_fallback_scorer()
    gold = load_instance(ASSETS / "demo_gold.json")
    grounded = load_instance(ASSETS / "demo_pred.json")
    generic = load_instance(FIXTURES / "demo_pred_generic.json")

    grounded_metrics = evaluate_instance(grounded, gold, scorer=scorer, mode="soft")
    generic_metrics = evaluate_instance(generic, gold, scorer=scorer, mode="soft")

    assert grounded_metrics.reasonableness_score > generic_metrics.reasonableness_score
    assert grounded_metrics.reasoning_f1 > generic_metrics.reasoning_f1
    assert grounded_metrics.chosen_action_reasonableness is not None


def test_annotation_fixtures_convert_without_losing_actions() -> None:
    from nora.adapters import to_instance
    from nora.data import read_rows
    paths = list((FIXTURES / "annotations").glob("*.json"))
    assert len(paths) == 2
    for path in paths:
        row = read_rows(path)[0]
        assert len(to_instance(row, "annotation").action_graphs) == len(row["actions"]) >= 3
