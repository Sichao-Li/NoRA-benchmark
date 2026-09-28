from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Sequence, TypeVar

import numpy as np
from scipy.optimize import linear_sum_assignment


T = TypeVar("T")
U = TypeVar("U")


def harmonic_mean(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class MatchPair(Generic[T, U]):
    pred_index: int
    gold_index: int
    score: float
    pred_item: T
    gold_item: U


@dataclass(slots=True)
class MatchingResult(Generic[T, U]):
    pairs: list[MatchPair[T, U]] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_gold: list[int] = field(default_factory=list)
    matrix: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class PoolingMetrics:
    precision: float
    recall: float
    f1: float
    row_maxima: tuple[float, ...]
    column_maxima: tuple[float, ...]


def build_similarity_matrix(
    pred_items: Sequence[T],
    gold_items: Sequence[U],
    similarity_fn: Callable[[T, U], float],
) -> np.ndarray:
    """Build a dense similarity matrix in [0, 1]."""

    matrix = np.zeros((len(pred_items), len(gold_items)), dtype=float)
    for i, pred_item in enumerate(pred_items):
        for j, gold_item in enumerate(gold_items):
            matrix[i, j] = float(np.clip(similarity_fn(pred_item, gold_item), 0.0, 1.0))
    return matrix


def hungarian_match(
    pred_items: Sequence[T],
    gold_items: Sequence[U],
    similarity_fn: Callable[[T, U], float] | None = None,
    matrix: np.ndarray | None = None,
    threshold: float = 0.0,
) -> MatchingResult[T, U]:
    """Find a one-to-one alignment that maximizes total similarity."""

    if matrix is None:
        if similarity_fn is None:
            raise ValueError("Either similarity_fn or matrix must be provided.")
        matrix = build_similarity_matrix(pred_items, gold_items, similarity_fn)
    else:
        matrix = np.asarray(matrix, dtype=float)
    pred_count, gold_count = matrix.shape
    if pred_count == 0 or gold_count == 0:
        return MatchingResult(
            pairs=[],
            unmatched_pred=list(range(pred_count)),
            unmatched_gold=list(range(gold_count)),
            matrix=matrix,
        )
    row_ind, col_ind = linear_sum_assignment(1.0 - matrix)
    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    pairs: list[MatchPair[T, U]] = []
    for pred_index, gold_index in zip(row_ind, col_ind):
        score = float(matrix[pred_index, gold_index])
        if score < threshold:
            continue
        matched_pred.add(int(pred_index))
        matched_gold.add(int(gold_index))
        pairs.append(
            MatchPair(
                pred_index=int(pred_index),
                gold_index=int(gold_index),
                score=score,
                pred_item=pred_items[pred_index],
                gold_item=gold_items[gold_index],
            )
        )
    return MatchingResult(
        pairs=pairs,
        unmatched_pred=[index for index in range(pred_count) if index not in matched_pred],
        unmatched_gold=[index for index in range(gold_count) if index not in matched_gold],
        matrix=matrix,
    )


def max_pooling_metrics(matrix: np.ndarray) -> PoolingMetrics:
    """Compute set-level soft precision/recall/F1 with max-pooling."""

    matrix = np.asarray(matrix, dtype=float)
    pred_count, gold_count = matrix.shape
    if pred_count == 0 and gold_count == 0:
        return PoolingMetrics(precision=1.0, recall=1.0, f1=1.0, row_maxima=(), column_maxima=())
    if pred_count == 0:
        return PoolingMetrics(precision=0.0, recall=0.0, f1=0.0, row_maxima=(), column_maxima=tuple([0.0] * gold_count))
    if gold_count == 0:
        return PoolingMetrics(precision=0.0, recall=0.0, f1=0.0, row_maxima=tuple([0.0] * pred_count), column_maxima=())
    row_maxima = tuple(float(value) for value in matrix.max(axis=1))
    column_maxima = tuple(float(value) for value in matrix.max(axis=0))
    precision = float(np.mean(row_maxima))
    recall = float(np.mean(column_maxima))
    return PoolingMetrics(
        precision=precision,
        recall=recall,
        f1=harmonic_mean(precision, recall),
        row_maxima=row_maxima,
        column_maxima=column_maxima,
    )


def aligned_scores_from_matching(
    matrix: np.ndarray,
    threshold: float = 0.0,
) -> PoolingMetrics:
    """Compute precision/recall/F1 from a Hungarian alignment over a score matrix."""

    matrix = np.asarray(matrix, dtype=float)
    pred_count, gold_count = matrix.shape
    if pred_count == 0 and gold_count == 0:
        return PoolingMetrics(precision=1.0, recall=1.0, f1=1.0, row_maxima=(), column_maxima=())
    if pred_count == 0 or gold_count == 0:
        return PoolingMetrics(precision=0.0, recall=0.0, f1=0.0, row_maxima=(), column_maxima=())
    row_ind, col_ind = linear_sum_assignment(1.0 - matrix)
    matched_scores = [float(matrix[i, j]) for i, j in zip(row_ind, col_ind) if matrix[i, j] >= threshold]
    score_sum = float(np.sum(matched_scores))
    precision = score_sum / pred_count if pred_count else 1.0
    recall = score_sum / gold_count if gold_count else 1.0
    return PoolingMetrics(
        precision=precision,
        recall=recall,
        f1=harmonic_mean(precision, recall),
        row_maxima=tuple(matched_scores),
        column_maxima=tuple(matched_scores),
    )
