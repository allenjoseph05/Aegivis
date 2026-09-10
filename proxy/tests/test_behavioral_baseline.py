"""
Tests for proxy.app.security.behavioral_baseline — Phase 20.

Test philosophy:
  - BaselineRecord: Welford incremental updates, variance, std_dev, z_score.
  - BaselineRecord: z_score when std_dev=0 (zero-variance baseline).
  - BaselineRecord.to_dict: correct keys, JSON-serialisable.
  - InMemoryBackend: get_record creates on miss, save/get round-trip, clear.
  - BehavioralBaselineStore.observe: updates record, respects max_tool_count cap.
  - BehavioralBaselineStore.score_anomaly: has_baseline=False below min_observations.
  - BehavioralBaselineStore.score_anomaly: z-score correctly classifies safe/alert/block.
  - BehavioralBaselineStore.score_anomaly: zero-variance baseline handles spike.
  - BehavioralBaselineStore.score_anomaly: AnomalyResult.to_dict JSON-serialisable.
  - BehavioralBaselineStore: record_count tracks observation count.
  - BehavioralBaselineStore: clear removes all records.
  - AnomalyResult.to_dict: inf z-score converted to finite value.
"""
from __future__ import annotations

import json
import math
import pytest

from app.security.behavioral_baseline import (
    AnomalyResult,
    BaselineRecord,
    BehavioralBaselineStore,
    InMemoryBackend,
    baseline_store,
)


# ---------------------------------------------------------------------------
# BaselineRecord — Welford stats
# ---------------------------------------------------------------------------

class TestBaselineRecord:
    def test_initial_state(self):
        r = BaselineRecord()
        assert r.count == 0
        assert r.mean == 0.0
        assert r.variance == 0.0
        assert r.std_dev == 0.0

    def test_single_update(self):
        r = BaselineRecord()
        r.update(5.0)
        assert r.count == 1
        assert r.mean == pytest.approx(5.0)
        assert r.variance == 0.0  # Bessel correction: n-1=0

    def test_two_updates_mean(self):
        r = BaselineRecord()
        r.update(2.0)
        r.update(4.0)
        assert r.mean == pytest.approx(3.0)

    def test_two_updates_variance(self):
        r = BaselineRecord()
        r.update(2.0)
        r.update(4.0)
        # Sample variance of [2, 4] = ((2-3)^2 + (4-3)^2) / (2-1) = 2.0
        assert r.variance == pytest.approx(2.0)
        assert r.std_dev == pytest.approx(math.sqrt(2.0))

    def test_five_updates(self):
        r = BaselineRecord()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            r.update(v)
        assert r.count == 5
        assert r.mean == pytest.approx(3.0)
        # sample variance = 2.5
        assert r.variance == pytest.approx(2.5)

    def test_z_score_within_baseline(self):
        r = BaselineRecord()
        for v in [10.0, 10.0, 10.0, 11.0, 9.0]:
            r.update(v)
        # Mean ≈ 10, std ≈ small → z_score(10) ≈ 0
        z = r.z_score(10.0)
        assert abs(z) < 1.0

    def test_z_score_extreme_outlier(self):
        r = BaselineRecord()
        for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
            r.update(v)
        z = r.z_score(100.0)
        # Mean=1, std=0 → infinite z-score for any different value
        assert z == float("inf")

    def test_z_score_zero_variance_same_value(self):
        r = BaselineRecord()
        for v in [5.0, 5.0, 5.0]:
            r.update(v)
        # Same value as mean → z=0
        assert r.z_score(5.0) == 0.0

    def test_negative_z_score(self):
        r = BaselineRecord()
        for v in [10.0, 10.0, 10.0, 10.0, 10.0, 12.0]:
            r.update(v)
        z = r.z_score(1.0)  # below mean
        assert z < 0

    def test_to_dict_keys(self):
        r = BaselineRecord()
        r.update(3.0)
        r.update(5.0)
        d = r.to_dict()
        assert "count" in d
        assert "mean" in d
        assert "std_dev" in d
        assert "variance" in d

    def test_to_dict_json_serialisable(self):
        r = BaselineRecord()
        r.update(2.0)
        r.update(4.0)
        json.dumps(r.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------

class TestInMemoryBackend:
    def test_get_record_creates_on_miss(self):
        b = InMemoryBackend()
        r = b.get_record("org|agent|tool")
        assert isinstance(r, BaselineRecord)
        assert r.count == 0

    def test_get_record_same_object_on_second_call(self):
        b = InMemoryBackend()
        r1 = b.get_record("k")
        r2 = b.get_record("k")
        assert r1 is r2

    def test_save_and_retrieve(self):
        b = InMemoryBackend()
        r = BaselineRecord()
        r.update(7.0)
        b.save_record("k", r)
        retrieved = b.get_record("k")
        assert retrieved.count == 1
        assert retrieved.mean == pytest.approx(7.0)

    def test_clear_removes_all(self):
        b = InMemoryBackend()
        b.get_record("k1")
        b.get_record("k2")
        assert len(b) == 2
        b.clear()
        assert len(b) == 0

    def test_len(self):
        b = InMemoryBackend()
        assert len(b) == 0
        b.get_record("a")
        b.get_record("b")
        assert len(b) == 2


# ---------------------------------------------------------------------------
# BehavioralBaselineStore.observe
# ---------------------------------------------------------------------------

class TestBaselineStoreObserve:
    def _store(self, **kw) -> BehavioralBaselineStore:
        return BehavioralBaselineStore(InMemoryBackend(), **kw)

    def test_observe_increments_count(self):
        s = self._store()
        s.observe("org", "agent", "tool", 3)
        assert s.record_count("org", "agent", "tool") == 1

    def test_observe_multiple_sessions(self):
        s = self._store()
        for _ in range(5):
            s.observe("org", "agent", "tool", 2)
        assert s.record_count("org", "agent", "tool") == 5

    def test_max_count_cap(self):
        s = self._store(max_tool_count=10)
        # Observe a huge count — should be capped at 10
        s.observe("org", "agent", "tool", 9_999)
        b = InMemoryBackend()
        s2 = BehavioralBaselineStore(b, max_tool_count=10)
        s2.observe("org", "agent", "tool", 9_999)
        record = b.get_record("org|agent|tool")
        assert record.mean == pytest.approx(10.0)  # capped

    def test_observe_zero_count_allowed(self):
        s = self._store()
        s.observe("org", "agent", "tool", 0)
        assert s.record_count("org", "agent", "tool") == 1

    def test_observe_isolated_by_tool(self):
        s = self._store()
        s.observe("org", "agent", "tool_a", 5)
        s.observe("org", "agent", "tool_b", 10)
        assert s.record_count("org", "agent", "tool_a") == 1
        assert s.record_count("org", "agent", "tool_b") == 1

    def test_observe_isolated_by_agent(self):
        s = self._store()
        s.observe("org", "agent1", "tool", 5)
        s.observe("org", "agent2", "tool", 5)
        assert s.record_count("org", "agent1", "tool") == 1
        assert s.record_count("org", "agent2", "tool") == 1


# ---------------------------------------------------------------------------
# BehavioralBaselineStore.score_anomaly — no baseline yet
# ---------------------------------------------------------------------------

class TestScoreAnomalyNoBaseline:
    def _store(self, min_obs=5) -> BehavioralBaselineStore:
        return BehavioralBaselineStore(InMemoryBackend(), min_observations=min_obs)

    def test_has_baseline_false_on_no_data(self):
        s = self._store()
        result = s.score_anomaly("org", "agent", "tool", 100)
        assert result.has_baseline is False
        assert result.is_anomalous is False

    def test_has_baseline_false_below_min_obs(self):
        s = self._store(min_obs=5)
        for _ in range(4):  # 4 < 5 min observations
            s.observe("org", "agent", "tool", 2)
        result = s.score_anomaly("org", "agent", "tool", 100)
        assert result.has_baseline is False

    def test_has_baseline_true_at_min_obs(self):
        s = self._store(min_obs=5)
        for i in range(5):  # exactly 5 sessions observed
            s.observe("org", "agent", "tool", 2)
        result = s.score_anomaly("org", "agent", "tool", 2)
        assert result.has_baseline is True

    def test_no_baseline_result_not_anomalous(self):
        s = self._store()
        result = s.score_anomaly("org", "agent", "tool", 999)
        assert result.is_anomalous is False
        assert result.should_block is False


# ---------------------------------------------------------------------------
# BehavioralBaselineStore.score_anomaly — with baseline
# ---------------------------------------------------------------------------

class TestScoreAnomalyWithBaseline:
    def _trained_store(
        self,
        normal_count: int = 2,
        n: int = 10,
        alert_z: float = 3.0,
        block_z: float = 5.0,
    ) -> BehavioralBaselineStore:
        """Return a store trained on n sessions each with normal_count calls."""
        s = BehavioralBaselineStore(
            InMemoryBackend(),
            min_observations=n,
            alert_z_score=alert_z,
            block_z_score=block_z,
        )
        for _ in range(n):
            s.observe("org", "agent", "tool", normal_count)
        return s

    def test_normal_count_not_anomalous(self):
        s = self._trained_store(normal_count=2, n=10)
        result = s.score_anomaly("org", "agent", "tool", 2)
        assert result.has_baseline is True
        assert result.is_anomalous is False
        assert result.z_score == pytest.approx(0.0, abs=1e-6)

    def test_mild_spike_not_anomalous(self):
        # Train with some variance: [1, 2, 3, 2, 2, 1, 3, 2, 2, 2]
        s = BehavioralBaselineStore(
            InMemoryBackend(), min_observations=10, alert_z_score=3.0, block_z_score=5.0
        )
        for v in [1, 2, 3, 2, 2, 1, 3, 2, 2, 2]:
            s.observe("org", "agent", "tool", v)
        result = s.score_anomaly("org", "agent", "tool", 3)
        assert result.is_anomalous is False

    def test_extreme_spike_triggers_alert(self):
        # All sessions had 0–1 calls; now suddenly 50 calls
        s = BehavioralBaselineStore(
            InMemoryBackend(), min_observations=10, alert_z_score=3.0, block_z_score=5.0
        )
        for v in [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]:
            s.observe("org", "agent", "tool", v)
        result = s.score_anomaly("org", "agent", "tool", 50)
        assert result.has_baseline is True
        assert result.is_anomalous is True

    def test_extreme_spike_triggers_block(self):
        s = BehavioralBaselineStore(
            InMemoryBackend(), min_observations=10, alert_z_score=3.0, block_z_score=5.0
        )
        for v in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
            s.observe("org", "agent", "tool", v)
        # z_score of 1000 on a zero-variance baseline → infinite → block
        result = s.score_anomaly("org", "agent", "tool", 1000)
        assert result.should_block is True

    def test_zero_variance_spike_is_anomalous(self):
        """Baseline is always 0; any non-zero call count is anomalous."""
        s = BehavioralBaselineStore(
            InMemoryBackend(), min_observations=5, alert_z_score=3.0, block_z_score=5.0
        )
        for _ in range(5):
            s.observe("org", "agent", "tool", 0)
        result = s.score_anomaly("org", "agent", "tool", 5)
        assert result.is_anomalous is True

    def test_z_score_is_float(self):
        s = self._trained_store(n=10, normal_count=5)
        result = s.score_anomaly("org", "agent", "tool", 5)
        assert isinstance(result.z_score, float)

    def test_result_fields_populated(self):
        s = self._trained_store(n=10)
        result = s.score_anomaly("org", "agent", "tool", 2)
        assert result.tool_name == "tool"
        assert result.observed_count == 2
        assert isinstance(result.baseline, dict)
        assert isinstance(result.reason, str)

    def test_anomaly_result_to_dict_json_serialisable(self):
        s = self._trained_store(n=10, normal_count=1)
        result = s.score_anomaly("org", "agent", "tool", 100)
        json.dumps(result.to_dict())  # must not raise

    def test_to_dict_inf_z_score_capped(self):
        """AnomalyResult.to_dict must not emit inf (not JSON-serialisable)."""
        s = BehavioralBaselineStore(
            InMemoryBackend(), min_observations=5
        )
        for _ in range(5):
            s.observe("org", "agent", "tool", 0)
        result = s.score_anomaly("org", "agent", "tool", 100)
        d = result.to_dict()
        assert math.isfinite(d["z_score"])

    def test_isolated_by_org_id(self):
        """Different orgs have independent baselines."""
        s = BehavioralBaselineStore(InMemoryBackend(), min_observations=5)
        for _ in range(5):
            s.observe("org_a", "agent", "tool", 1)
        # org_b has no data
        result = s.score_anomaly("org_b", "agent", "tool", 1)
        assert result.has_baseline is False


# ---------------------------------------------------------------------------
# BehavioralBaselineStore — management
# ---------------------------------------------------------------------------

class TestBaselineStoreManagement:
    def test_record_count_zero_before_observe(self):
        s = BehavioralBaselineStore(InMemoryBackend())
        assert s.record_count("org", "agent", "tool") == 0

    def test_clear_resets_all_records(self):
        s = BehavioralBaselineStore(InMemoryBackend(), min_observations=2)
        s.observe("org", "agent", "tool", 5)
        s.observe("org", "agent", "tool", 5)
        s.clear()
        assert s.record_count("org", "agent", "tool") == 0
        result = s.score_anomaly("org", "agent", "tool", 5)
        assert result.has_baseline is False

    def test_module_singleton_importable(self):
        from app.security.behavioral_baseline import baseline_store
        assert baseline_store is not None

    def test_custom_backend_used(self):
        """Custom backend is called for get/save."""
        class _CountingBackend(InMemoryBackend):
            gets = saves = 0
            def get_record(self, key):
                self.gets += 1
                return super().get_record(key)
            def save_record(self, key, record):
                self.saves += 1
                return super().save_record(key, record)

        b = _CountingBackend()
        s = BehavioralBaselineStore(b)
        s.observe("org", "agent", "tool", 3)
        s.score_anomaly("org", "agent", "tool", 3)
        assert b.saves >= 1
        assert b.gets >= 1
