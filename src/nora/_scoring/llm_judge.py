from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class JudgeResult:
    equivalent: bool
    pred_entails_gold: float
    gold_entails_pred: float
    contradiction: float
    normative_compatibility: float
    operational_compatibility: float
    confidence: float
    brief_rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pred_entails_gold", _clip01(self.pred_entails_gold))
        object.__setattr__(self, "gold_entails_pred", _clip01(self.gold_entails_pred))
        object.__setattr__(self, "contradiction", _clip01(self.contradiction))
        object.__setattr__(self, "normative_compatibility", _clip01(self.normative_compatibility))
        object.__setattr__(self, "operational_compatibility", _clip01(self.operational_compatibility))
        object.__setattr__(self, "confidence", _clip01(self.confidence))

    @classmethod
    def neutral(cls, rationale: str = "LLM judge unavailable.") -> "JudgeResult":
        return cls(
            equivalent=False,
            pred_entails_gold=0.0,
            gold_entails_pred=0.0,
            contradiction=0.0,
            normative_compatibility=0.5,
            operational_compatibility=0.5,
            confidence=0.0,
            brief_rationale=rationale,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JudgeResult":
        return cls(
            equivalent=bool(payload.get("equivalent", False)),
            pred_entails_gold=float(payload.get("pred_entails_gold", 0.0)),
            gold_entails_pred=float(payload.get("gold_entails_pred", 0.0)),
            contradiction=float(payload.get("contradiction", 0.0)),
            normative_compatibility=float(payload.get("normative_compatibility", 0.5)),
            operational_compatibility=float(payload.get("operational_compatibility", 0.5)),
            confidence=float(payload.get("confidence", 0.0)),
            brief_rationale=str(payload.get("brief_rationale", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMJudge(ABC):
    """Abstract interface for semantic pairwise verification."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the judge is configured and callable."""

    @abstractmethod
    def judge_pair(self, node_type: str, pred_text: str, gold_text: str) -> JudgeResult:
        """Judge a pair of node texts."""


class NullLLMJudge(LLMJudge):
    """A no-op judge used in deterministic offline mode."""

    @property
    def available(self) -> bool:
        return False

    def judge_pair(self, node_type: str, pred_text: str, gold_text: str) -> JudgeResult:
        return JudgeResult.neutral("Null judge in use.")


class OpenAICompatibleJudge(LLMJudge):
    """Optional OpenAI-compatible judge with JSONL caching."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        cache_path: str | Path = ".reason_eval_llm_cache.jsonl",
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_API_BASE",
    ) -> None:
        self.model = model
        self.cache_path = Path(cache_path)
        self.api_key = os.getenv(api_key_env)
        self.base_url = os.getenv(base_url_env)
        # self._cache: dict[str, JudgeResult] = {}
        self._client = None
        # self._load_cache()
        self._init_client()

    @property
    def available(self) -> bool:
        return self._client is not None

    def _init_client(self) -> None:
        if not self.api_key:
            return
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        except Exception:
            self._client = None

    # def _load_cache(self) -> None:
    #     if not self.cache_path.exists():
    #         return
    #     for line in self.cache_path.read_text(encoding="utf-8").splitlines():
    #         if not line.strip():
    #             continue
    #         record = json.loads(line)
    #         self._cache[str(record["key"])] = JudgeResult.from_dict(record["result"])

    # def _cache_key(self, node_type: str, pred_text: str, gold_text: str) -> str:
    #     payload = json.dumps(
    #         {
    #             "model": self.model,
    #             "node_type": node_type,
    #             "pred_text": pred_text,
    #             "gold_text": gold_text,
    #         },
    #         sort_keys=True,
    #     )
    #     return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # def _append_cache(self, key: str, result: JudgeResult) -> None:
    #     self.cache_path.parent.mkdir(parents=True, exist_ok=True)
    #     with self.cache_path.open("a", encoding="utf-8") as handle:
    #         handle.write(json.dumps({"key": key, "result": result.to_dict()}) + "\n")

    def judge_pair(self, node_type: str, pred_text: str, gold_text: str) -> JudgeResult:
        # key = self._cache_key(node_type, pred_text, gold_text)
        # if key in self._cache:
        #     return self._cache[key]
        if not self.available:
            return JudgeResult.neutral("OPENAI_API_KEY missing or OpenAI client unavailable.")
        result = self._request_judgment(node_type=node_type, pred_text=pred_text, gold_text=gold_text)
        # self._cache[key] = result
        # self._append_cache(key, result)
        return result

    def _request_judgment(self, node_type: str, pred_text: str, gold_text: str) -> JudgeResult:
        system_prompt = (
            "You compare two moral reasoning graph nodes and return only JSON. "
            "All numeric fields must be probabilities in [0,1]."
        )
        user_prompt = (
            f"node_type: {node_type}\n"
            f"pred_text: {pred_text}\n"
            f"gold_text: {gold_text}\n\n"
            "Return a JSON object with keys: "
            "equivalent, pred_entails_gold, gold_entails_pred, contradiction, "
            "normative_compatibility, operational_compatibility, confidence, brief_rationale."
        )
        try:
            print(f"LLM Judge Request:\nSystem: {system_prompt}\nUser: {user_prompt}\n")
            response = self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return JudgeResult.from_dict(payload)
        except Exception as exc:
            return JudgeResult.neutral(f"LLM judge request failed: {exc.__class__.__name__}.")
