from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np

from nora._scoring.calibrate import ScoreCalibrator, load_saved_calibrator
from nora._scoring.embedding_backend import SentenceTransformerBackend, SimilarityBackend
from nora._scoring.lexical import lexical_similarity
from nora._scoring.llm_judge import JudgeResult, LLMJudge, NullLLMJudge


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


CalibratorInput: TypeAlias = ScoreCalibrator | str | Path | None


@dataclass(slots=True)
class BaseSimilarityWeights:
    """Weights for the primary semantic signal and light lexical backing.

    `embedding` weights the backend semantic score, which can come from a
    cross-encoder or a sentence-transformer backend.
    """

    embedding: float = 0.90
    lexical: float = 0.10
    entailment: float = 0.0
    contradiction_penalty: float = 0.0


@dataclass(slots=True)
class SimilarityBreakdown:
    embedding: float
    lexical: float
    pred_entails_gold: float
    gold_entails_pred: float
    contradiction: float
    entailment_score: float
    normative_agreement: float
    operational_compatibility: float
    base_similarity: float
    final_similarity: float
    used_llm: bool
    judge_result: JudgeResult | None = None


@dataclass(slots=True)
class HybridSimilarityScorer:
    """Typed text similarity with separately reported auxiliary signals.

    Offline backend scores are not reused as entailment or operational-compatibility
    evidence.

    Reason-node matching is semantic-only. Norm labels are reported separately
    through `normative_agreement` and do not change `reason_similarity` directly.
    """

    backend: SimilarityBackend = field(default_factory=SentenceTransformerBackend)
    judge: LLMJudge = field(default_factory=NullLLMJudge)
    use_wordnet: bool = False
    base_weights: BaseSimilarityWeights = field(default_factory=BaseSimilarityWeights)
    reason_alpha: float = 0.85
    reason_beta: float = 0.15
    action_alpha: float = 0.8
    action_beta: float = 0.2
    action_calibrator: ScoreCalibrator | None = None
    fact_calibrator: ScoreCalibrator | None = None
    reason_calibrator: ScoreCalibrator | None = None
    action_graph_calibrator: ScoreCalibrator | None = None

    def normative_agreement(self, pred_label: str | None, gold_label: str | None) -> float:
        if not pred_label or not gold_label:
            return 0.5
        return 1.0 if pred_label.strip().lower() == gold_label.strip().lower() else 0.0

    @staticmethod
    def _apply_calibrator(raw_score: float, calibrator: ScoreCalibrator | None) -> float:
        if calibrator is None:
            return clip01(raw_score)
        calibrated = np.asarray(calibrator.transform([clip01(raw_score)]), dtype=float)
        if calibrated.size == 0:
            return clip01(raw_score)
        return clip01(float(calibrated[0]))

    def calibrate_action_score(self, raw_score: float) -> float:
        return self._apply_calibrator(raw_score, self.action_calibrator)

    def calibrate_fact_score(self, raw_score: float) -> float:
        return self._apply_calibrator(raw_score, self.fact_calibrator)

    def calibrate_reason_score(self, raw_score: float) -> float:
        return self._apply_calibrator(raw_score, self.reason_calibrator)

    def calibrate_action_graph_score(self, raw_score: float) -> float:
        return self._apply_calibrator(raw_score, self.action_graph_calibrator)

    def _base_breakdown(self, node_type: str, pred_text: str, gold_text: str) -> SimilarityBreakdown:
        semantic = clip01(self.backend.similarity(pred_text, gold_text))
        lexical = clip01(lexical_similarity(pred_text, gold_text, use_wordnet=self.use_wordnet))
        used_llm = self.judge.available
        judge_result = self.judge.judge_pair(node_type, pred_text, gold_text) if used_llm else None
        pred_entails_gold = 0.0 if judge_result is None else judge_result.pred_entails_gold
        gold_entails_pred = 0.0 if judge_result is None else judge_result.gold_entails_pred
        contradiction = 0.0 if judge_result is None else judge_result.contradiction
        entailment_score = 0.5 * (pred_entails_gold + gold_entails_pred)
        operational_compatibility = semantic if judge_result is None else judge_result.operational_compatibility
        base_similarity = (
            self.base_weights.embedding * semantic
            + self.base_weights.lexical * lexical
            + self.base_weights.entailment * entailment_score
            - self.base_weights.contradiction_penalty * contradiction
        )
        base_similarity = clip01(base_similarity)
        return SimilarityBreakdown(
            embedding=semantic,
            lexical=lexical,
            pred_entails_gold=pred_entails_gold,
            gold_entails_pred=gold_entails_pred,
            contradiction=contradiction,
            entailment_score=entailment_score,
            normative_agreement=0.5,
            operational_compatibility=operational_compatibility,
            base_similarity=base_similarity,
            final_similarity=base_similarity,
            used_llm=used_llm,
            judge_result=judge_result,
        )

    def fact_breakdown(self, pred_text: str, gold_text: str) -> SimilarityBreakdown:
        breakdown = self._base_breakdown("fact", pred_text, gold_text)
        breakdown.final_similarity = self.calibrate_fact_score(breakdown.base_similarity)
        return breakdown

    def reason_breakdown(
        self,
        pred_text: str,
        gold_text: str,
        pred_label: str | None = None,
        gold_label: str | None = None,
    ) -> SimilarityBreakdown:
        breakdown = self._base_breakdown("reason", pred_text, gold_text)
        normative = self.normative_agreement(pred_label, gold_label)
        breakdown.normative_agreement = normative
        # Refined metric design keeps norm-category agreement as an explicit
        # graph-level term, rather than blending it into reason similarity.
        raw_reason_score = clip01(breakdown.base_similarity)
        breakdown.final_similarity = self.calibrate_reason_score(raw_reason_score)
        return breakdown

    def action_breakdown(self, pred_text: str, gold_text: str) -> SimilarityBreakdown:
        breakdown = self._base_breakdown("action", pred_text, gold_text)
        if breakdown.used_llm:
            raw_action_score = clip01(
                self.action_alpha * breakdown.base_similarity
                + self.action_beta * breakdown.operational_compatibility
            )
        else:
            raw_action_score = clip01(breakdown.base_similarity)
        breakdown.final_similarity = self.calibrate_action_score(raw_action_score)
        return breakdown

    def fact_similarity(self, pred_text: str, gold_text: str) -> float:
        return self.fact_breakdown(pred_text, gold_text).final_similarity

    def reason_similarity(
        self,
        pred_text: str,
        gold_text: str,
        pred_label: str | None = None,
        gold_label: str | None = None,
    ) -> float:
        return self.reason_breakdown(pred_text, gold_text, pred_label=pred_label, gold_label=gold_label).final_similarity

    def action_similarity(self, pred_text: str, gold_text: str) -> float:
        return self.action_breakdown(pred_text, gold_text).final_similarity


def s_fact(pred_text: str, gold_text: str, scorer: HybridSimilarityScorer | None = None) -> float:
    active = scorer or HybridSimilarityScorer()
    return active.fact_similarity(pred_text, gold_text)


def s_reason(
    pred_text: str,
    gold_text: str,
    pred_label: str | None = None,
    gold_label: str | None = None,
    scorer: HybridSimilarityScorer | None = None,
) -> float:
    active = scorer or HybridSimilarityScorer()
    return active.reason_similarity(pred_text, gold_text, pred_label=pred_label, gold_label=gold_label)


def s_action(pred_text: str, gold_text: str, scorer: HybridSimilarityScorer | None = None) -> float:
    active = scorer or HybridSimilarityScorer()
    return active.action_similarity(pred_text, gold_text)


def resolve_calibrator(spec: CalibratorInput) -> ScoreCalibrator | None:
    """Resolve an in-memory calibrator or load one from disk."""

    if spec is None:
        return None
    if isinstance(spec, ScoreCalibrator):
        return spec
    return load_saved_calibrator(spec)


def build_calibrated_scorer(
    *,
    backend: SimilarityBackend | None = None,
    judge: LLMJudge | None = None,
    use_wordnet: bool = False,
    base_weights: BaseSimilarityWeights | None = None,
    reason_alpha: float = 0.85,
    reason_beta: float = 0.15,
    action_alpha: float = 0.8,
    action_beta: float = 0.2,
    action_calibrator: CalibratorInput = None,
    fact_calibrator: CalibratorInput = None,
    reason_calibrator: CalibratorInput = None,
    action_graph_calibrator: CalibratorInput = None,
) -> HybridSimilarityScorer:
    """Build a scorer with optional per-type calibrators.

    Calibrators can be passed either as fitted `ScoreCalibrator` instances or as
    saved paths produced by `TemperatureScaler.save(...)` /
    `IsotonicCalibrator.save(...)`.
    """

    return HybridSimilarityScorer(
        backend=backend or SentenceTransformerBackend(),
        judge=judge or NullLLMJudge(),
        use_wordnet=use_wordnet,
        base_weights=base_weights or BaseSimilarityWeights(),
        reason_alpha=reason_alpha,
        reason_beta=reason_beta,
        action_alpha=action_alpha,
        action_beta=action_beta,
        action_calibrator=resolve_calibrator(action_calibrator),
        fact_calibrator=resolve_calibrator(fact_calibrator),
        reason_calibrator=resolve_calibrator(reason_calibrator),
        action_graph_calibrator=resolve_calibrator(action_graph_calibrator),
    )
