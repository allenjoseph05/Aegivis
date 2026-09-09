"""
Isolation Forest Anomaly Detector — Phase 3.3

Unsupervised ML on per-session feature vectors.
Catches runaway agents and abnormal session behaviour patterns.

Design
------
- Singleton backed by sklearn.ensemble.IsolationForest
- Refits every REFIT_INTERVAL new samples (not on every call)
- Normalizes sklearn decision_function to [0, 1] anomaly score
- Returns None until MIN_SAMPLES reached (insufficient training data)
- Graceful degradation if scikit-learn is not installed
- Thread-safe via module-level lock
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_REFIT_INTERVAL = 10   # refit model every N new samples
_MIN_SAMPLES = 10      # don't score until this many sessions seen

# Feature order must stay stable across calls
_FEATURE_KEYS = [
    "llm_call_count",
    "tool_call_rate",
    "error_rate",
    "session_duration_min",
    "max_injection_score",
]


@dataclass
class IsolationForestResult:
    """Result of Isolation Forest scoring for a single session."""
    anomaly_score: float         # 0.0 = normal, 1.0 = maximally anomalous
    is_anomaly: bool
    features: dict[str, float]  # input feature vector
    samples_seen: int

    def to_dict(self) -> dict:
        return {
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomaly":    self.is_anomaly,
            "features":      {k: round(v, 4) for k, v in self.features.items()},
            "samples_seen":  self.samples_seen,
        }


class _IsolationForestModel:
    """Thread-safe singleton wrapper around sklearn.ensemble.IsolationForest."""

    def __init__(self, contamination: float = 0.05) -> None:
        self._contamination = contamination
        self._lock = threading.Lock()
        self._model = None       # sklearn model; None until first fit
        self._data: list[list[float]] = []
        self._samples_seen = 0
        self._sklearn_available: bool | None = None  # None = not yet checked

    def _is_sklearn_available(self) -> bool:
        if self._sklearn_available is None:
            try:
                import sklearn  # noqa: F401
                self._sklearn_available = True
            except ImportError:
                self._sklearn_available = False
                logger.info(
                    "scikit-learn not installed -- Isolation Forest disabled. "
                    "Install with: pip install 'aegivis-proxy[ml]'"
                )
        return self._sklearn_available

    def _feature_vector(self, features: dict[str, float]) -> list[float]:
        return [features.get(k, 0.0) for k in _FEATURE_KEYS]

    def _refit(self) -> None:
        """Refit the model on all accumulated data. Must be called under lock."""
        if not self._is_sklearn_available():
            return
        if len(self._data) < _MIN_SAMPLES:
            return
        try:
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                contamination=self._contamination,
                random_state=42,
                n_estimators=100,
            )
            model.fit(self._data)
            self._model = model
            logger.debug(
                "Isolation Forest refitted on %d samples", len(self._data)
            )
        except Exception as exc:
            logger.warning("IsolationForest refit error: %s", exc)

    def fit_and_score(self, features: dict[str, float]) -> IsolationForestResult | None:
        """
        Add a new session sample, refit if needed, and score it.

        Returns None if:
        - scikit-learn is not installed
        - fewer than MIN_SAMPLES seen so far
        """
        if not self._is_sklearn_available():
            return None

        vec = self._feature_vector(features)

        with self._lock:
            self._data.append(vec)
            self._samples_seen += 1
            n = self._samples_seen

            # Refit every REFIT_INTERVAL samples
            if n % _REFIT_INTERVAL == 0:
                self._refit()

            if self._model is None or n < _MIN_SAMPLES:
                return None

            try:
                import numpy as np
                X = np.array([vec])
                # decision_function: positive = normal, negative = anomaly
                raw_score = float(self._model.decision_function(X)[0])

                # Normalize to [0, 1]:
                # sklearn decision_function typically ranges ~[-0.5, 0.5]
                # We map: score <= -0.5 -> 1.0 (max anomaly), score >= 0.5 -> 0.0 (normal)
                normalized = max(0.0, min(1.0, (0.5 - raw_score) / 1.0))

                # Default anomaly threshold: normalized score > 0.6
                is_anomaly = raw_score < 0.0

                return IsolationForestResult(
                    anomaly_score=normalized,
                    is_anomaly=is_anomaly,
                    features=dict(features),
                    samples_seen=n,
                )
            except Exception as exc:
                logger.warning("IsolationForest scoring error: %s", exc)
                return None


# Module-level singleton
_model: _IsolationForestModel | None = None
_model_lock = threading.Lock()


def _get_model(contamination: float = 0.05) -> _IsolationForestModel:
    global _model
    with _model_lock:
        if _model is None:
            _model = _IsolationForestModel(contamination=contamination)
        return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_and_score(
    features: dict[str, float],
    contamination: float = 0.05,
) -> IsolationForestResult | None:
    """
    Add a session feature vector to the training set and score it.

    Feature keys (all floats):
        llm_call_count        -- total LLM calls in the session
        tool_call_rate        -- tool_calls / max(llm_calls, 1)
        error_rate            -- SYSTEM_ERROR events / max(llm_calls, 1)
        session_duration_min  -- session duration in minutes
        max_injection_score   -- maximum injection score seen in the session

    Returns:
        IsolationForestResult if a model is fitted and MIN_SAMPLES reached.
        None otherwise (not enough data or sklearn not installed).

    Never raises.
    """
    try:
        model = _get_model(contamination=contamination)
        return model.fit_and_score(features)
    except Exception as exc:
        logger.warning("fit_and_score error (skipped): %s", exc)
        return None


def samples_seen() -> int:
    """Return number of sessions observed so far."""
    global _model
    with _model_lock:
        if _model is None:
            return 0
        return _model._samples_seen
