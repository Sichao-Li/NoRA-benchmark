from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


REQUIRED_COLUMNS = {"pred_text", "gold_text", "node_type", "human_score", "raw_score"}


def load_calibration_data(path: str | Path) -> pd.DataFrame:
    """Load and validate a calibration CSV."""

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Calibration CSV is missing required columns: {missing_list}")
    return frame


class ScoreCalibrator(ABC):
    """Abstract score calibrator."""

    @abstractmethod
    def fit(self, raw_scores: np.ndarray, human_scores: np.ndarray) -> "ScoreCalibrator":
        """Fit calibrator parameters."""

    @abstractmethod
    def transform(self, raw_scores: np.ndarray | list[float]) -> np.ndarray:
        """Transform raw scores into calibrated scores."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted calibrator."""


@dataclass(slots=True)
class TemperatureScaler(ScoreCalibrator):
    temperature: float = 1.0

    @staticmethod
    def _logit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))

    def fit(self, raw_scores: np.ndarray, human_scores: np.ndarray) -> "TemperatureScaler":
        raw = np.asarray(raw_scores, dtype=float)
        human = np.asarray(human_scores, dtype=float)

        def objective(temp: float) -> float:
            scaled = self._sigmoid(self._logit(raw) / max(temp, 1e-3))
            return float(np.mean((scaled - human) ** 2))

        result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    def transform(self, raw_scores: np.ndarray | list[float]) -> np.ndarray:
        raw = np.asarray(raw_scores, dtype=float)
        logits = self._logit(raw)
        return self._sigmoid(logits / max(self.temperature, 1e-3))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"method": "temperature", "temperature": self.temperature}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(temperature=float(payload["temperature"]))


@dataclass(slots=True)
class IsotonicCalibrator(ScoreCalibrator):
    model: IsotonicRegression = field(default_factory=lambda: IsotonicRegression(out_of_bounds="clip"))

    def fit(self, raw_scores: np.ndarray, human_scores: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(np.asarray(raw_scores, dtype=float), np.asarray(human_scores, dtype=float))
        return self

    def transform(self, raw_scores: np.ndarray | list[float]) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(raw_scores, dtype=float)), dtype=float)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump({"method": "isotonic", "model": self.model}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(model=payload["model"])


def fit_calibrator_from_csv(
    path: str | Path,
    method: str = "temperature",
) -> ScoreCalibrator:
    """Fit a calibrator directly from a labeled calibration CSV."""

    frame = load_calibration_data(path)
    raw = frame["raw_score"].to_numpy(dtype=float)
    human = frame["human_score"].to_numpy(dtype=float)
    if method == "temperature":
        return TemperatureScaler().fit(raw, human)
    if method == "isotonic":
        return IsotonicCalibrator().fit(raw, human)
    raise ValueError(f"Unsupported calibration method: {method!r}")


def load_saved_calibrator(path: str | Path) -> ScoreCalibrator:
    location = Path(path)
    if location.suffix.lower() == ".json":
        return TemperatureScaler.load(location)
    return IsotonicCalibrator.load(location)
