from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nora._scoring.data_types import ActionGraph, FactNode, ReasonNode
from nora._scoring.graph_utils import ancestor_node_ids, edge_signature, translate_edge
from nora._scoring.matching import MatchPair, harmonic_mean, hungarian_match
from nora._scoring.similarity import HybridSimilarityScorer


def _soft_scores(pairs: list[MatchPair[object, object]], pred_count: int, gold_count: int) -> tuple[float, float, float]:
    if pred_count == 0 and gold_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or gold_count == 0:
        return 0.0, 0.0, 0.0
    matched_sum = float(np.sum([pair.score for pair in pairs]))
    precision = matched_sum / pred_count
    recall = matched_sum / gold_count
    return precision, recall, harmonic_mean(precision, recall)


def _exact_prf(match_count: int, pred_count: int, gold_count: int) -> tuple[float, float, float]:
    if pred_count == 0 and gold_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or gold_count == 0:
        return 0.0, 0.0, 0.0
    precision = match_count / pred_count
    recall = match_count / gold_count
    return precision, recall, harmonic_mean(precision, recall)


@dataclass(frozen=True, slots=True)
class LocalGraphWeights:
    """Weights for the headline local justification score.

    The benchmark evaluator keeps `ancestor` for backward compatibility, but the
    headline score no longer uses it. Ancestor overlap remains diagnostic only.
    """

    fact: float = 0.20
    reason: float = 0.30
    normative: float = 0.15
    edge: float = 0.35
    ancestor: float = 0.0

    def __post_init__(self) -> None:
        values = self.to_dict()
        if any(value < 0.0 for value in values.values()):
            raise ValueError("Local graph weights must be non-negative.")
        headline_total = self.fact + self.reason + self.normative + self.edge
        if headline_total <= 0.0:
            raise ValueError("At least one headline local graph weight must be positive.")

    def normalized(self) -> LocalGraphWeights:
        """Return headline weights normalized to sum to 1.

        `ancestor` is preserved for diagnostics and does not participate in the
        headline total.
        """

        headline_total = self.fact + self.reason + self.normative + self.edge
        if np.isclose(headline_total, 1.0):
            return self
        return LocalGraphWeights(
            fact=self.fact / headline_total,
            reason=self.reason / headline_total,
            normative=self.normative / headline_total,
            edge=self.edge / headline_total,
            ancestor=self.ancestor,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "fact": self.fact,
            "reason": self.reason,
            "normative": self.normative,
            "edge": self.edge,
            "ancestor": self.ancestor,
        }


@dataclass(slots=True)
class LocalGraphScore:
    fact_precision: float
    fact_recall: float
    fact_f1: float
    reason_precision: float
    reason_recall: float
    reason_f1: float
    normative_agreement: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    ancestor_overlap: float
    total: float
    fact_alignment: dict[str, str] = field(default_factory=dict)
    reason_alignment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | dict[str, str]]:
        return {
            "fact_precision": self.fact_precision,
            "fact_recall": self.fact_recall,
            "fact_f1": self.fact_f1,
            "reason_precision": self.reason_precision,
            "reason_recall": self.reason_recall,
            "reason_f1": self.reason_f1,
            "normative_agreement": self.normative_agreement,
            "edge_precision": self.edge_precision,
            "edge_recall": self.edge_recall,
            "edge_f1": self.edge_f1,
            "ancestor_overlap": self.ancestor_overlap,
            "total": self.total,
            "fact_alignment": self.fact_alignment,
            "reason_alignment": self.reason_alignment,
        }


def _alignment_map(pairs: list[MatchPair[FactNode | ReasonNode, FactNode | ReasonNode]]) -> dict[str, str]:
    return {pair.pred_item.id: pair.gold_item.id for pair in pairs}


def score_local_graph(
    pred_graph: ActionGraph,
    gold_graph: ActionGraph,
    scorer: HybridSimilarityScorer,
    fact_threshold: float = 0.0,
    reason_threshold: float = 0.0,
    weights: LocalGraphWeights | None = None,
) -> LocalGraphScore:
    """Score one aligned action pair's local justification binding.

    The headline score emphasizes the binding structure:
    fact -> reason -> action. Ancestor overlap is still returned, but it no
    longer contributes to the headline total.
    """

    fact_result = hungarian_match(
        pred_graph.facts,
        gold_graph.facts,
        similarity_fn=lambda pred, gold: scorer.fact_similarity(pred.text, gold.text),
        threshold=fact_threshold,
    )
    reason_result = hungarian_match(
        pred_graph.reasons,
        gold_graph.reasons,
        similarity_fn=lambda pred, gold: scorer.reason_similarity(
            pred.text,
            gold.text,
            pred_label=pred.normative_label,
            gold_label=gold.normative_label,
        ),
        threshold=reason_threshold,
    )

    fact_precision, fact_recall, fact_f1 = _soft_scores(
        fact_result.pairs,
        pred_count=len(pred_graph.facts),
        gold_count=len(gold_graph.facts),
    )
    reason_precision, reason_recall, reason_f1 = _soft_scores(
        reason_result.pairs,
        pred_count=len(pred_graph.reasons),
        gold_count=len(gold_graph.reasons),
    )

    if not pred_graph.reasons and not gold_graph.reasons:
        normative_agreement = 1.0
    elif not reason_result.pairs:
        normative_agreement = 0.0
    else:
        numerator = 0.0
        denominator = 0.0
        for pair in reason_result.pairs:
            weight = pair.score
            numerator += weight * scorer.normative_agreement(
                pair.pred_item.normative_label,
                pair.gold_item.normative_label,
            )
            denominator += weight
        normative_agreement = 0.0 if denominator == 0.0 else numerator / denominator

    fact_alignment = _alignment_map(fact_result.pairs)
    reason_alignment = _alignment_map(reason_result.pairs)
    node_alignment = {
        **fact_alignment,
        **reason_alignment,
        pred_graph.action_id: gold_graph.action_id,
    }

    gold_edges = {edge_signature(edge) for edge in gold_graph.edges}
    matched_edges = 0
    for edge in pred_graph.edges:
        translated = translate_edge(edge, node_alignment)
        if translated is not None and translated in gold_edges:
            matched_edges += 1
    edge_precision, edge_recall, edge_f1 = _exact_prf(
        match_count=matched_edges,
        pred_count=len(pred_graph.edges),
        gold_count=len(gold_graph.edges),
    )

    pred_ancestors = ancestor_node_ids(pred_graph)
    mapped_pred_ancestors = {node_alignment[node_id] for node_id in pred_ancestors if node_id in node_alignment}
    gold_ancestors = ancestor_node_ids(gold_graph)
    if not mapped_pred_ancestors and not gold_ancestors:
        ancestor_overlap = 1.0
    elif not mapped_pred_ancestors or not gold_ancestors:
        ancestor_overlap = 0.0
    else:
        ancestor_overlap = len(mapped_pred_ancestors & gold_ancestors) / len(mapped_pred_ancestors | gold_ancestors)

    active_weights = (weights or LocalGraphWeights()).normalized()
    total = float(
        np.clip(
            active_weights.fact * fact_f1
            + active_weights.reason * reason_f1
            + active_weights.edge * edge_f1
            + active_weights.normative * normative_agreement,
            0.0,
            1.0,
        )
    )
    return LocalGraphScore(
        fact_precision=fact_precision,
        fact_recall=fact_recall,
        fact_f1=fact_f1,
        reason_precision=reason_precision,
        reason_recall=reason_recall,
        reason_f1=reason_f1,
        normative_agreement=float(np.clip(normative_agreement, 0.0, 1.0)),
        edge_precision=edge_precision,
        edge_recall=edge_recall,
        edge_f1=edge_f1,
        ancestor_overlap=float(np.clip(ancestor_overlap, 0.0, 1.0)),
        total=total,
        fact_alignment=fact_alignment,
        reason_alignment=reason_alignment,
    )
