from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from nora._scoring.data_types import ActionGraph, FactNode, ReasoningInstance
from nora._scoring.graph_metrics import LocalGraphScore, LocalGraphWeights, score_local_graph
from nora._scoring.io import index_instances_by_id
from nora._scoring.matching import PoolingMetrics, harmonic_mean, hungarian_match, max_pooling_metrics
from nora._scoring.similarity import HybridSimilarityScorer

AlignmentMode = Literal["soft", "hungarian"]


@dataclass(frozen=True, slots=True)
class PairDiagnostic:
    pred_action_id: str
    gold_action_id: str
    action_similarity: float
    fact_score: float
    reasoning_score: float
    combined_score: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "pred_action_id": self.pred_action_id,
            "gold_action_id": self.gold_action_id,
            "action_similarity": self.action_similarity,
            "fact_score": self.fact_score,
            "reasoning_score": self.reasoning_score,
            "combined_score": self.combined_score,
        }


@dataclass(slots=True)
class InstanceMetrics:
    instance_id: str
    mode: AlignmentMode
    action_precision: float
    action_recall: float
    action_f1: float
    fact_precision: float
    fact_recall: float
    fact_f1: float
    reasoning_precision: float
    reasoning_recall: float
    reasoning_f1: float
    summary_score: float
    reasonableness_score: float
    chosen_action_alignment: float | None = None
    chosen_local_fact_f1: float | None = None
    chosen_support_binding: float | None = None
    chosen_action_summary: float | None = None
    chosen_action_reasonableness: float | None = None
    chosen_action_best_gold_action_id: str | None = None
    chosen_action_best_gold_action_text: str | None = None
    chosen_pred_action_id: str | None = None
    chosen_pred_action_text: str | None = None
    fact_available: bool = False
    graph_available: bool = False
    action_matrix: list[list[float]] = field(default_factory=list)
    fact_matrix: list[list[float]] = field(default_factory=list)
    reasoning_matrix: list[list[float]] = field(default_factory=list)
    pair_matrix: list[list[float]] = field(default_factory=list)
    pair_diagnostics: list[PairDiagnostic] = field(default_factory=list)
    local_scores: dict[str, dict[str, LocalGraphScore]] = field(default_factory=dict)

    @property
    def action_text_precision(self) -> float:
        return self.action_precision

    @property
    def action_text_recall(self) -> float:
        return self.action_recall

    @property
    def action_text_f1(self) -> float:
        return self.action_f1

    @property
    def shared_fact_precision(self) -> float:
        return self.fact_precision

    @property
    def shared_fact_recall(self) -> float:
        return self.fact_recall

    @property
    def shared_fact_f1(self) -> float:
        return self.fact_f1

    @property
    def action_graph_precision(self) -> float:
        return self.reasoning_precision

    @property
    def justification_binding_precision(self) -> float:
        return self.reasoning_precision

    @property
    def action_graph_recall(self) -> float:
        return self.reasoning_recall

    @property
    def justification_binding_recall(self) -> float:
        return self.reasoning_recall

    @property
    def action_graph_f1(self) -> float:
        return self.reasoning_f1

    @property
    def justification_binding_f1(self) -> float:
        return self.reasoning_f1

    @property
    def final_score(self) -> float:
        """Backward-compatible alias for the availability-aware summary score."""

        return self.summary_score

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "mode": self.mode,
            "action_precision": self.action_precision,
            "action_recall": self.action_recall,
            "action_f1": self.action_f1,
            "fact_precision": self.fact_precision,
            "fact_recall": self.fact_recall,
            "fact_f1": self.fact_f1,
            "reasoning_precision": self.reasoning_precision,
            "reasoning_recall": self.reasoning_recall,
            "reasoning_f1": self.reasoning_f1,
            "action_text_precision": self.action_text_precision,
            "action_text_recall": self.action_text_recall,
            "action_text_f1": self.action_text_f1,
            "shared_fact_precision": self.shared_fact_precision,
            "shared_fact_recall": self.shared_fact_recall,
            "shared_fact_f1": self.shared_fact_f1,
            "justification_binding_precision": self.justification_binding_precision,
            "justification_binding_recall": self.justification_binding_recall,
            "justification_binding_f1": self.justification_binding_f1,
            "action_graph_precision": self.action_graph_precision,
            "action_graph_recall": self.action_graph_recall,
            "action_graph_f1": self.action_graph_f1,
            "fact_available": self.fact_available,
            "graph_available": self.graph_available,
            "summary_score": self.summary_score,
            "reasonableness_score": self.reasonableness_score,
            "final_score": self.final_score,
            "chosen_action_alignment": self.chosen_action_alignment,
            "chosen_local_fact_f1": self.chosen_local_fact_f1,
            "chosen_support_binding": self.chosen_support_binding,
            "chosen_action_summary": self.chosen_action_summary,
            "chosen_action_reasonableness": self.chosen_action_reasonableness,
            "chosen_action_best_gold_action_id": self.chosen_action_best_gold_action_id,
            "chosen_action_best_gold_action_text": self.chosen_action_best_gold_action_text,
            "chosen_pred_action_id": self.chosen_pred_action_id,
            "chosen_pred_action_text": self.chosen_pred_action_text,
            "action_matrix": self.action_matrix,
            "fact_matrix": self.fact_matrix,
            "reasoning_matrix": self.reasoning_matrix,
            "pair_matrix": self.pair_matrix,
            "pair_diagnostics": [item.to_dict() for item in self.pair_diagnostics],
            "local_scores": {
                pred_id: {gold_id: score.to_dict() for gold_id, score in row.items()}
                for pred_id, row in self.local_scores.items()
            },
        }


@dataclass(frozen=True, slots=True)
class BatchMetrics:
    instances: tuple[InstanceMetrics, ...]
    aggregate: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "instances": [instance.to_dict() for instance in self.instances],
            "aggregate": self.aggregate,
        }


def _pool_metrics(
    matrix: np.ndarray,
    pred_items: Sequence[object],
    gold_items: Sequence[object],
    mode: AlignmentMode,
    threshold: float = 0.0,
) -> tuple[PoolingMetrics, list[tuple[int, int, float]]]:
    if mode == "soft":
        metrics = max_pooling_metrics(matrix)
        diagnostics: list[tuple[int, int, float]] = []
        if matrix.size:
            for pred_index in range(matrix.shape[0]):
                gold_index = int(np.argmax(matrix[pred_index]))
                diagnostics.append((pred_index, gold_index, float(matrix[pred_index, gold_index])))
        return metrics, diagnostics
    match = hungarian_match(pred_items, gold_items, matrix=matrix, threshold=threshold)
    if not pred_items and not gold_items:
        return PoolingMetrics(precision=1.0, recall=1.0, f1=1.0, row_maxima=(), column_maxima=()), []
    if not pred_items or not gold_items:
        return PoolingMetrics(precision=0.0, recall=0.0, f1=0.0, row_maxima=(), column_maxima=()), []
    score_sum = float(np.sum([pair.score for pair in match.pairs]))
    precision = score_sum / len(pred_items) if pred_items else 1.0
    recall = score_sum / len(gold_items) if gold_items else 1.0
    metrics = PoolingMetrics(
        precision=precision,
        recall=recall,
        f1=harmonic_mean(precision, recall),
        row_maxima=tuple(pair.score for pair in match.pairs),
        column_maxima=tuple(pair.score for pair in match.pairs),
    )
    diagnostics = [(pair.pred_index, pair.gold_index, pair.score) for pair in match.pairs]
    return metrics, diagnostics


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def _normalize_fact_text(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_action_text(text: str) -> str:
    return " ".join(text.lower().split())


def collect_fact_pool(instance: ReasoningInstance) -> list[FactNode]:
    """Collect a deduplicated clip-level fact pool, independent of action grouping."""

    fact_pool: list[FactNode] = []
    seen_keys: set[str] = set()
    for graph in instance.action_graphs:
        for fact in graph.facts:
            key = _normalize_fact_text(fact.text)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fact_pool.append(fact)
    return fact_pool


def _has_graph_support(graph: ActionGraph) -> bool:
    return bool(graph.facts or graph.reasons or graph.edges)


def _zero_metrics() -> PoolingMetrics:
    return PoolingMetrics(precision=0.0, recall=0.0, f1=0.0, row_maxima=(), column_maxima=())


def _resolve_chosen_pred_index(instance: ReasoningInstance, pred_graphs: Sequence[ActionGraph]) -> int | None:
    if not pred_graphs:
        return None

    if instance.chosen_action_id:
        for index, graph in enumerate(pred_graphs):
            if graph.action_id == instance.chosen_action_id:
                return index

    if instance.chosen_action_text:
        normalized = _normalize_action_text(instance.chosen_action_text)
        for index, graph in enumerate(pred_graphs):
            if _normalize_action_text(graph.action_text) == normalized:
                return index

    return 0


def _chosen_support_binding(local_score: LocalGraphScore, weights: LocalGraphWeights | None) -> float:
    active_weights = (weights or LocalGraphWeights()).normalized()
    support_weight = active_weights.reason + active_weights.edge + active_weights.normative
    if support_weight <= 0.0:
        return 0.0
    raw_score = (
        active_weights.reason * local_score.reason_f1
        + active_weights.edge * local_score.edge_f1
        + active_weights.normative * local_score.normative_agreement
    ) / support_weight
    return float(np.clip(raw_score, 0.0, 1.0))


def _justification_metrics(
    action_matrix: np.ndarray,
    justification_matrix: np.ndarray,
    pred_graphs: Sequence[ActionGraph],
    gold_graphs: Sequence[ActionGraph],
    mode: AlignmentMode,
) -> tuple[PoolingMetrics, list[tuple[int, int, float]]]:
    """Aggregate local justification quality over action alignments.

    Action similarity determines alignment; local graph quality determines the
    score of each aligned pair. Action similarity is intentionally not
    multiplied into the justification score again.
    """

    pred_count, gold_count = action_matrix.shape
    if pred_count == 0 and gold_count == 0:
        return PoolingMetrics(precision=1.0, recall=1.0, f1=1.0, row_maxima=(), column_maxima=()), []
    if pred_count == 0 or gold_count == 0:
        return _zero_metrics(), []

    if mode == "soft":
        row_scores: list[float] = []
        diagnostics: list[tuple[int, int, float]] = []
        for pred_index in range(pred_count):
            gold_index = int(np.argmax(action_matrix[pred_index]))
            score = float(justification_matrix[pred_index, gold_index])
            row_scores.append(score)
            diagnostics.append((pred_index, gold_index, score))

        column_scores: list[float] = []
        for gold_index in range(gold_count):
            pred_index = int(np.argmax(action_matrix[:, gold_index]))
            column_scores.append(float(justification_matrix[pred_index, gold_index]))

        precision = float(np.mean(row_scores))
        recall = float(np.mean(column_scores))
        return (
            PoolingMetrics(
                precision=precision,
                recall=recall,
                f1=harmonic_mean(precision, recall),
                row_maxima=tuple(row_scores),
                column_maxima=tuple(column_scores),
            ),
            diagnostics,
        )

    action_alignment = hungarian_match(pred_graphs, gold_graphs, matrix=action_matrix, threshold=0.0)
    matched_scores = [float(justification_matrix[pair.pred_index, pair.gold_index]) for pair in action_alignment.pairs]
    score_sum = float(np.sum(matched_scores))
    precision = score_sum / pred_count if pred_count else 1.0
    recall = score_sum / gold_count if gold_count else 1.0
    diagnostics = [
        (pair.pred_index, pair.gold_index, float(justification_matrix[pair.pred_index, pair.gold_index]))
        for pair in action_alignment.pairs
    ]
    return (
        PoolingMetrics(
            precision=precision,
            recall=recall,
            f1=harmonic_mean(precision, recall),
            row_maxima=tuple(matched_scores),
            column_maxima=tuple(matched_scores),
        ),
        diagnostics,
    )


def _headline_components(
    action_metrics: PoolingMetrics,
    fact_metrics: PoolingMetrics,
    reasoning_metrics: PoolingMetrics,
    *,
    fact_available: bool,
    graph_available: bool,
) -> list[float]:
    components = [action_metrics.f1]
    if fact_available:
        components.append(fact_metrics.f1)
    if graph_available:
        components.append(reasoning_metrics.f1)
    return components


def _summary_score(components: Sequence[float]) -> float:
    if not components:
        return 0.0
    return float(np.mean(components))


def _reasonableness_score(components: Sequence[float]) -> float:
    if not components:
        return 0.0
    clipped = np.clip(np.asarray(components, dtype=float), 0.0, 1.0)
    return float(np.exp(np.mean(np.log(np.clip(clipped, 1e-12, 1.0)))))


def evaluate_instance(
    pred_instance: ReasoningInstance,
    gold_instance: ReasoningInstance,
    scorer: HybridSimilarityScorer | None = None,
    mode: AlignmentMode = "soft",
    lambda_red: float = 0.10,
    lambda_con: float = 0.20,
    local_graph_weights: LocalGraphWeights | None = None,
) -> InstanceMetrics:
    """Evaluate one predicted instance against one gold instance.

    In this benchmark variant:
    - `action_f1` is action-text pool quality.
    - `fact_f1` is shared-fact-set quality when facts are available.
    - `reasoning_f1` / `justification_binding_f1` is local justification
      quality aggregated over action alignments.
    - auxiliary chosen-action metrics isolate the model's selected action
      against its best-matching gold action, for case studies and fine-grained
      analysis.
    - `summary_score` is the mean of the available headline metrics.
    - `reasonableness_score` is the geometric mean of the available headline
      metrics, so weak components reduce the score more sharply.

    Precision and recall remain the primary directional signals:
    - precision: prediction-to-gold quality
    - recall: gold coverage
    - F1: harmonic summary of the two

    `lambda_red` and `lambda_con` are compatibility parameters and do not affect
    the scores.
    """

    if pred_instance.instance_id != gold_instance.instance_id:
        raise ValueError(
            f"instance_id mismatch: pred={pred_instance.instance_id!r}, gold={gold_instance.instance_id!r}"
        )
    _ = (lambda_red, lambda_con)
    active = scorer or HybridSimilarityScorer()
    pred_graphs = list(pred_instance.action_graphs)
    gold_graphs = list(gold_instance.action_graphs)
    pred_facts = collect_fact_pool(pred_instance)
    gold_facts = collect_fact_pool(gold_instance)

    fact_available = bool(pred_facts or gold_facts)
    graph_available = any(_has_graph_support(graph) for graph in pred_graphs) or any(
        _has_graph_support(graph) for graph in gold_graphs
    )

    action_matrix = np.zeros((len(pred_graphs), len(gold_graphs)), dtype=float)
    fact_matrix = np.zeros((len(pred_facts), len(gold_facts)), dtype=float)
    reasoning_matrix = np.zeros_like(action_matrix)
    pair_matrix = np.zeros_like(action_matrix)
    local_scores: dict[str, dict[str, LocalGraphScore]] = {}

    for pred_index, pred_graph in enumerate(pred_graphs):
        local_scores[pred_graph.action_id] = {}
        for gold_index, gold_graph in enumerate(gold_graphs):
            action_similarity = active.action_similarity(pred_graph.action_text, gold_graph.action_text)
            action_matrix[pred_index, gold_index] = action_similarity
            local_score = score_local_graph(
                pred_graph,
                gold_graph,
                scorer=active,
                weights=local_graph_weights,
            )
            local_scores[pred_graph.action_id][gold_graph.action_id] = local_score
            if graph_available:
                raw_graph_pair_score = local_score.total
                calibrated_graph_pair_score = active.calibrate_action_graph_score(raw_graph_pair_score)
                reasoning_matrix[pred_index, gold_index] = calibrated_graph_pair_score
                pair_matrix[pred_index, gold_index] = calibrated_graph_pair_score

    if fact_available:
        for pred_index, pred_fact in enumerate(pred_facts):
            for gold_index, gold_fact in enumerate(gold_facts):
                fact_matrix[pred_index, gold_index] = active.fact_similarity(pred_fact.text, gold_fact.text)

    action_metrics, action_pairs = _pool_metrics(action_matrix, pred_graphs, gold_graphs, mode=mode)
    if fact_available:
        fact_metrics, _ = _pool_metrics(fact_matrix, pred_facts, gold_facts, mode=mode)
    else:
        fact_metrics = _zero_metrics()
    if graph_available:
        reasoning_metrics, reasoning_pairs = _justification_metrics(
            action_matrix,
            pair_matrix,
            pred_graphs,
            gold_graphs,
            mode=mode,
        )
    else:
        reasoning_metrics, reasoning_pairs = _zero_metrics(), []

    diagnostics = action_pairs if (mode == "hungarian" and not graph_available) else reasoning_pairs
    pair_diagnostics = [
        PairDiagnostic(
            pred_action_id=pred_graphs[pred_index].action_id,
            gold_action_id=gold_graphs[gold_index].action_id,
            action_similarity=float(action_matrix[pred_index, gold_index]),
            fact_score=float(local_scores[pred_graphs[pred_index].action_id][gold_graphs[gold_index].action_id].fact_f1),
            reasoning_score=float(local_scores[pred_graphs[pred_index].action_id][gold_graphs[gold_index].action_id].total),
            combined_score=float(score),
        )
        for pred_index, gold_index, score in diagnostics
    ]

    headline_components = _headline_components(
        action_metrics,
        fact_metrics,
        reasoning_metrics,
        fact_available=fact_available,
        graph_available=graph_available,
    )
    summary_score = _summary_score(headline_components)
    reasonableness_score = _reasonableness_score(headline_components)

    chosen_pred_index = _resolve_chosen_pred_index(pred_instance, pred_graphs)
    chosen_action_alignment: float | None = None
    chosen_local_fact_f1: float | None = None
    chosen_support_binding: float | None = None
    chosen_action_summary: float | None = None
    chosen_action_reasonableness: float | None = None
    chosen_action_best_gold_action_id: str | None = None
    chosen_action_best_gold_action_text: str | None = None
    chosen_pred_action_id: str | None = None
    chosen_pred_action_text: str | None = None
    if chosen_pred_index is not None:
        chosen_pred_graph = pred_graphs[chosen_pred_index]
        chosen_pred_action_id = chosen_pred_graph.action_id
        chosen_pred_action_text = chosen_pred_graph.action_text
        if gold_graphs:
            chosen_gold_index = int(np.argmax(action_matrix[chosen_pred_index]))
            chosen_gold_graph = gold_graphs[chosen_gold_index]
            chosen_action_best_gold_action_id = chosen_gold_graph.action_id
            chosen_action_best_gold_action_text = chosen_gold_graph.action_text
            chosen_local_score = local_scores[chosen_pred_graph.action_id][chosen_gold_graph.action_id]
            chosen_action_alignment = float(np.clip(action_matrix[chosen_pred_index, chosen_gold_index], 0.0, 1.0))
            chosen_local_fact_f1 = float(np.clip(chosen_local_score.fact_f1, 0.0, 1.0))
            chosen_support_binding = _chosen_support_binding(chosen_local_score, local_graph_weights)
        else:
            chosen_action_alignment = 0.0
            chosen_local_fact_f1 = 0.0
            chosen_support_binding = 0.0

        chosen_components = [
            chosen_action_alignment,
            chosen_local_fact_f1,
            chosen_support_binding,
        ]
        chosen_action_summary = _summary_score(chosen_components)
        chosen_action_reasonableness = _reasonableness_score(chosen_components)

    return InstanceMetrics(
        instance_id=pred_instance.instance_id,
        mode=mode,
        action_precision=action_metrics.precision,
        action_recall=action_metrics.recall,
        action_f1=action_metrics.f1,
        fact_precision=fact_metrics.precision,
        fact_recall=fact_metrics.recall,
        fact_f1=fact_metrics.f1,
        reasoning_precision=reasoning_metrics.precision,
        reasoning_recall=reasoning_metrics.recall,
        reasoning_f1=reasoning_metrics.f1,
        summary_score=float(np.clip(summary_score, 0.0, 1.0)),
        reasonableness_score=float(np.clip(reasonableness_score, 0.0, 1.0)),
        chosen_action_alignment=chosen_action_alignment,
        chosen_local_fact_f1=chosen_local_fact_f1,
        chosen_support_binding=chosen_support_binding,
        chosen_action_summary=(
            None if chosen_action_summary is None else float(np.clip(chosen_action_summary, 0.0, 1.0))
        ),
        chosen_action_reasonableness=(
            None
            if chosen_action_reasonableness is None
            else float(np.clip(chosen_action_reasonableness, 0.0, 1.0))
        ),
        chosen_action_best_gold_action_id=chosen_action_best_gold_action_id,
        chosen_action_best_gold_action_text=chosen_action_best_gold_action_text,
        chosen_pred_action_id=chosen_pred_action_id,
        chosen_pred_action_text=chosen_pred_action_text,
        fact_available=fact_available,
        graph_available=graph_available,
        action_matrix=_matrix_to_list(action_matrix),
        fact_matrix=_matrix_to_list(fact_matrix),
        reasoning_matrix=_matrix_to_list(reasoning_matrix),
        pair_matrix=_matrix_to_list(pair_matrix),
        pair_diagnostics=pair_diagnostics,
        local_scores=local_scores,
    )


def aggregate_metrics(instances: Sequence[InstanceMetrics]) -> dict[str, float]:
    """Average per-instance metrics across a batch.

    For partial-overlap prediction sets, read precision and recall directly:
    precision measures pred-to-gold quality, and recall measures gold coverage.
    """

    if not instances:
        return {
            "count": 0.0,
            "action_precision": 0.0,
            "action_recall": 0.0,
            "action_f1": 0.0,
            "fact_precision": 0.0,
            "fact_recall": 0.0,
            "fact_f1": 0.0,
            "reasoning_precision": 0.0,
            "reasoning_recall": 0.0,
            "reasoning_f1": 0.0,
            "action_text_precision": 0.0,
            "action_text_recall": 0.0,
            "action_text_f1": 0.0,
            "shared_fact_precision": 0.0,
            "shared_fact_recall": 0.0,
            "shared_fact_f1": 0.0,
            "justification_binding_precision": 0.0,
            "justification_binding_recall": 0.0,
            "justification_binding_f1": 0.0,
            "action_graph_precision": 0.0,
            "action_graph_recall": 0.0,
            "action_graph_f1": 0.0,
            "fact_available_rate": 0.0,
            "graph_available_rate": 0.0,
            "summary_score": 0.0,
            "reasonableness_score": 0.0,
            "final_score": 0.0,
            "chosen_action_available_rate": 0.0,
            "chosen_action_alignment": 0.0,
            "chosen_local_fact_f1": 0.0,
            "chosen_support_binding": 0.0,
            "chosen_action_summary": 0.0,
            "chosen_action_reasonableness": 0.0,
        }
    action_precision = float(np.mean([item.action_precision for item in instances]))
    action_recall = float(np.mean([item.action_recall for item in instances]))
    action_f1 = float(np.mean([item.action_f1 for item in instances]))
    fact_precision = float(np.mean([item.fact_precision for item in instances]))
    fact_recall = float(np.mean([item.fact_recall for item in instances]))
    fact_f1 = float(np.mean([item.fact_f1 for item in instances]))
    reasoning_precision = float(np.mean([item.reasoning_precision for item in instances]))
    reasoning_recall = float(np.mean([item.reasoning_recall for item in instances]))
    reasoning_f1 = float(np.mean([item.reasoning_f1 for item in instances]))
    chosen_instances = [item for item in instances if item.chosen_action_alignment is not None]

    def _mean_optional(field_name: str) -> float:
        if not chosen_instances:
            return 0.0
        return float(np.mean([getattr(item, field_name) for item in chosen_instances]))

    return {
        "count": float(len(instances)),
        "action_precision": action_precision,
        "action_recall": action_recall,
        "action_f1": action_f1,
        "fact_precision": fact_precision,
        "fact_recall": fact_recall,
        "fact_f1": fact_f1,
        "reasoning_precision": reasoning_precision,
        "reasoning_recall": reasoning_recall,
        "reasoning_f1": reasoning_f1,
        "action_text_precision": action_precision,
        "action_text_recall": action_recall,
        "action_text_f1": action_f1,
        "shared_fact_precision": fact_precision,
        "shared_fact_recall": fact_recall,
        "shared_fact_f1": fact_f1,
        "justification_binding_precision": reasoning_precision,
        "justification_binding_recall": reasoning_recall,
        "justification_binding_f1": reasoning_f1,
        "action_graph_precision": reasoning_precision,
        "action_graph_recall": reasoning_recall,
        "action_graph_f1": reasoning_f1,
        "fact_available_rate": float(np.mean([1.0 if item.fact_available else 0.0 for item in instances])),
        "graph_available_rate": float(np.mean([1.0 if item.graph_available else 0.0 for item in instances])),
        "summary_score": float(np.mean([item.summary_score for item in instances])),
        "reasonableness_score": float(np.mean([item.reasonableness_score for item in instances])),
        "final_score": float(np.mean([item.final_score for item in instances])),
        "chosen_action_available_rate": float(len(chosen_instances) / len(instances)),
        "chosen_action_alignment": _mean_optional("chosen_action_alignment"),
        "chosen_local_fact_f1": _mean_optional("chosen_local_fact_f1"),
        "chosen_support_binding": _mean_optional("chosen_support_binding"),
        "chosen_action_summary": _mean_optional("chosen_action_summary"),
        "chosen_action_reasonableness": _mean_optional("chosen_action_reasonableness"),
    }


def evaluate_batch(
    pred_instances: Sequence[ReasoningInstance],
    gold_instances: Sequence[ReasoningInstance],
    scorer: HybridSimilarityScorer | None = None,
    mode: AlignmentMode = "soft",
    lambda_red: float = 0.10,
    lambda_con: float = 0.20,
    local_graph_weights: LocalGraphWeights | None = None,
) -> BatchMetrics:
    """Evaluate a batch of prediction instances against a batch of gold instances.

    `lambda_red` and `lambda_con` are compatibility parameters and do not affect
    the scores.
    """

    gold_by_id = index_instances_by_id(gold_instances)
    pred_by_id = index_instances_by_id(pred_instances)
    metrics: list[InstanceMetrics] = []
    for instance_id, pred_instance in pred_by_id.items():
        if instance_id not in gold_by_id:
            raise KeyError(f"Missing gold instance for {instance_id!r}.")
        metrics.append(
            evaluate_instance(
                pred_instance=pred_instance,
                gold_instance=gold_by_id[instance_id],
                scorer=scorer,
                mode=mode,
                lambda_red=lambda_red,
                lambda_con=lambda_con,
                local_graph_weights=local_graph_weights,
            )
        )
    return BatchMetrics(instances=tuple(metrics), aggregate=aggregate_metrics(metrics))
