"""Tests for proxy.app.benchmark (Phase 3.5)."""
import pytest
from app.benchmark import (
    run_benchmark,
    ATTACK_CASES,
    CLEAN_CASES,
    CaseResult,
    CategorySummary,
    BenchmarkReport,
    _percentile,
    _detect_active_layers,
    _build_category_summaries,
)


# ---------------------------------------------------------------------------
# Dataset completeness
# ---------------------------------------------------------------------------

def test_attack_cases_count():
    """Should have at least 100 labelled attack cases."""
    assert len(ATTACK_CASES) >= 100


def test_clean_cases_count():
    """Should have at least 50 clean cases."""
    assert len(CLEAN_CASES) >= 50


def test_attack_case_format():
    """Each attack case must be a 3-tuple: (text, category, 'attack')."""
    for item in ATTACK_CASES:
        assert len(item) == 3, f"Expected 3-tuple, got {len(item)}-tuple"
        text, category, expected = item
        assert isinstance(text, str) and len(text) > 5
        assert isinstance(category, str) and len(category) > 0
        assert expected == "attack", f"Attack case has wrong label: {expected!r}"


def test_clean_case_format():
    """Each clean case must be a 2-tuple: (text, 'clean')."""
    for item in CLEAN_CASES:
        assert len(item) == 2, f"Expected 2-tuple, got {len(item)}-tuple"
        text, expected = item
        assert isinstance(text, str) and len(text) > 5
        assert expected == "clean", f"Clean case has wrong label: {expected!r}"


def test_attack_categories():
    """At least 5 distinct attack categories should be represented."""
    categories = {c for _, c, _ in ATTACK_CASES}
    assert len(categories) >= 5, f"Only {len(categories)} categories found: {categories}"


def test_no_duplicate_attack_texts():
    """Attack case texts should be unique."""
    texts = [t for t, _, _ in ATTACK_CASES]
    assert len(texts) == len(set(texts)), "Duplicate attack case texts found"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    assert _percentile([5.0], 50) == 5.0
    assert _percentile([5.0], 99) == 5.0


def test_percentile_p50():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(data, 50) == pytest.approx(3.0, abs=0.1)


def test_percentile_p99():
    data = list(range(100))
    result = _percentile(data, 99)
    assert result >= 98.0


def test_detect_active_layers_structural_always_present():
    layers = _detect_active_layers()
    assert "structural" in layers


def test_detect_active_layers_normalizer_always_present():
    layers = _detect_active_layers()
    assert "normalizer" in layers


def test_detect_active_layers_returns_list():
    layers = _detect_active_layers()
    assert isinstance(layers, list)
    assert len(layers) >= 2


# ---------------------------------------------------------------------------
# Category summaries
# ---------------------------------------------------------------------------

def test_build_category_summaries_empty():
    results = []
    summaries = _build_category_summaries(results, 0.50, 0.80)
    assert summaries == []


def test_build_category_summaries_single_category():
    results = [
        CaseResult(
            text="attack text",
            category="token_injection",
            expected="attack",
            score=0.9,
            detected=True,
            blocked=True,
            layer_scores={"structural": 0.9},
            threats=["pattern:x"],
            latency_ms=1.0,
        ),
        CaseResult(
            text="another attack",
            category="token_injection",
            expected="attack",
            score=0.3,
            detected=False,
            blocked=False,
            layer_scores={"structural": 0.3},
            threats=[],
            latency_ms=1.0,
        ),
    ]
    summaries = _build_category_summaries(results, 0.50, 0.80)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.name == "token_injection"
    assert s.total == 2
    assert s.detected == 1
    assert s.blocked == 1
    assert s.detection_rate == pytest.approx(0.5)
    assert s.block_rate == pytest.approx(0.5)


def test_build_category_summaries_excludes_clean():
    """Clean cases should not appear as attack categories."""
    results = [
        CaseResult(
            text="clean text",
            category="clean",
            expected="clean",
            score=0.1,
            detected=False,
            blocked=False,
            layer_scores={},
            threats=[],
            latency_ms=1.0,
        ),
        CaseResult(
            text="attack text",
            category="semantic_injection",
            expected="attack",
            score=0.8,
            detected=True,
            blocked=True,
            layer_scores={"structural": 0.8},
            threats=["pattern:x"],
            latency_ms=1.0,
        ),
    ]
    summaries = _build_category_summaries(results, 0.50, 0.80)
    names = {s.name for s in summaries}
    assert "clean" not in names
    assert "semantic_injection" in names


# ---------------------------------------------------------------------------
# BenchmarkReport.to_dict()
# ---------------------------------------------------------------------------

def test_benchmark_report_to_dict_keys():
    """to_dict should include all required top-level keys."""
    r = BenchmarkReport(
        run_at="2026-01-01T00:00:00",
        duration_s=1.0,
        total_attacks=10,
        total_clean=5,
        true_positives=8,
        false_negatives=2,
        true_negatives=5,
        false_positives=0,
        tpr=0.8,
        fpr=0.0,
        precision=1.0,
        f1=0.888,
        accuracy=0.933,
        latency_p50_ms=1.0,
        latency_p95_ms=5.0,
        latency_p99_ms=10.0,
        latency_mean_ms=2.0,
        categories=[],
        active_layers=["structural"],
        layer_coverage={"structural": 0.9},
        samples=[],
    )
    d = r.to_dict()
    for key in [
        "run_at", "duration_s", "total_attacks", "total_clean",
        "true_positives", "false_negatives", "true_negatives", "false_positives",
        "tpr", "fpr", "precision", "f1", "accuracy",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
        "categories", "active_layers", "layer_coverage", "samples",
    ]:
        assert key in d, f"Missing key: {key!r}"


# ---------------------------------------------------------------------------
# Full benchmark run (integration)
# ---------------------------------------------------------------------------

def test_run_benchmark_returns_report():
    """run_benchmark() should return a BenchmarkReport without raising."""
    report = run_benchmark()
    assert isinstance(report, BenchmarkReport)


def test_run_benchmark_case_counts():
    """Report case counts must match dataset sizes."""
    report = run_benchmark()
    assert report.total_attacks == len(ATTACK_CASES)
    assert report.total_clean == len(CLEAN_CASES)


def test_run_benchmark_confusion_matrix_adds_up():
    """TP + FN = total_attacks and TN + FP = total_clean."""
    report = run_benchmark()
    assert report.true_positives + report.false_negatives == report.total_attacks
    assert report.true_negatives + report.false_positives == report.total_clean


def test_run_benchmark_metrics_range():
    """All rates must be in [0, 1]."""
    report = run_benchmark()
    for attr in ("tpr", "fpr", "precision", "f1", "accuracy"):
        val = getattr(report, attr)
        assert 0.0 <= val <= 1.0, f"{attr} = {val} is out of range"


def test_run_benchmark_latency_positive():
    report = run_benchmark()
    assert report.latency_p50_ms >= 0.0
    assert report.latency_p95_ms >= report.latency_p50_ms
    assert report.latency_p99_ms >= report.latency_p95_ms


def test_run_benchmark_samples_count():
    """samples list should contain one result per case."""
    report = run_benchmark()
    assert len(report.samples) == len(ATTACK_CASES) + len(CLEAN_CASES)


def test_run_benchmark_samples_structure():
    """Each sample must have required fields."""
    report = run_benchmark()
    for s in report.samples[:5]:
        assert isinstance(s.text, str)
        assert s.expected in ("attack", "clean")
        assert 0.0 <= s.score <= 1.0
        assert isinstance(s.detected, bool)
        assert isinstance(s.blocked, bool)
        assert isinstance(s.layer_scores, dict)
        assert s.latency_ms >= 0.0


def test_run_benchmark_active_layers():
    """active_layers should always include at least structural and normalizer."""
    report = run_benchmark()
    assert "structural" in report.active_layers
    assert "normalizer" in report.active_layers


def test_run_benchmark_structural_scores_computed():
    """
    The structural layer should produce non-zero scores for token injection cases,
    even when semantic/classifier layers are not available.
    We check layer_scores directly rather than the detection threshold, because
    the structural layer weight (0.15) means it cannot reach the 0.50 threshold
    alone -- it needs semantic or classifier layers for full detection.
    """
    report = run_benchmark()
    # Find a token_injection case with a structural score > 0
    token_cases = [
        s for s in report.samples
        if s.category == "token_injection" and s.expected == "attack"
    ]
    assert len(token_cases) > 0, "No token_injection cases found in samples"

    # At least some token injection cases should have non-zero structural scores
    structural_scores = [s.layer_scores.get("structural", 0.0) for s in token_cases]
    max_structural = max(structural_scores) if structural_scores else 0.0

    # The structural scanner should catch <|im_start|> etc.
    assert max_structural > 0.0, (
        f"Structural layer produced 0.0 for all token_injection cases. "
        f"Structural scores: {structural_scores[:5]}"
    )


def test_run_benchmark_fpr_reasonable():
    """
    FPR should be low (clean messages should not trigger false positives).
    Expect FPR < 0.20 after the false-positive fixes in Phase 3.3.
    """
    report = run_benchmark()
    assert report.fpr < 0.20, (
        f"FPR={report.fpr:.1%} is too high -- check for false positive regressions.\n"
        f"FP cases: {[s.text[:80] for s in report.samples if s.expected == 'clean' and s.detected]}"
    )


def test_run_benchmark_categories_present():
    """All expected attack categories should appear in category summaries."""
    report = run_benchmark()
    category_names = {c.name for c in report.categories}
    expected = {"token_injection", "encoding_evasion", "semantic_injection",
                "privilege_escalation", "exfiltration", "indirect_injection"}
    assert expected.issubset(category_names), (
        f"Missing categories: {expected - category_names}"
    )


def test_run_benchmark_to_dict_serializable():
    """to_dict() result should be JSON-serializable."""
    import json
    report = run_benchmark()
    d = report.to_dict()
    # Should not raise
    json_str = json.dumps(d)
    assert len(json_str) > 100
