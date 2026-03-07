"""
External dataset validation for the Aegivis security scanner.

Downloads real-world prompt injection / jailbreak datasets from HuggingFace
and runs our scanner against them to measure TPR, FPR, and F1 in realistic
conditions — not our own hand-crafted cases.

Datasets:
  1. deepset/prompt-injections     — 662 samples, canonical PI benchmark
  2. jackhhao/jailbreak-classification — 1,306 samples, jailbreak vs benign
  3. leolee99/NotInject             — 339 samples, FPR stress-test (benign
                                      texts with trigger words like "ignore")

Usage:
    cd proxy
    python -m pytest tests/test_external_datasets.py -v -s
    # or directly:
    python tests/test_external_datasets.py
"""
from __future__ import annotations

import json
import sys
import time
import os
from dataclasses import dataclass, field
from pathlib import Path

# Ensure proxy/app is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)

from app.enforcement import scan_messages as enforcement_scan


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class DatasetResult:
    name: str
    total: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    errors: int = 0
    latency_ms: list[float] = field(default_factory=list)
    # Per-sample details for analysis
    false_negative_samples: list[dict] = field(default_factory=list)
    false_positive_samples: list[dict] = field(default_factory=list)

    @property
    def tpr(self) -> float:
        """True Positive Rate (Recall / Sensitivity)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate."""
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.tpr
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        s = sorted(self.latency_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]


def scan_text(text: str, threshold: float = 0.50) -> tuple[bool, float, float]:
    """
    Run the enforcement scanner (canonicalize + structural + credential) on a text string.
    Returns (detected, score, latency_ms).
    """
    t0 = time.perf_counter()
    messages = [{"role": "user", "content": text}]
    result = enforcement_scan(messages)
    detected = result.injection_score >= threshold
    latency_ms = (time.perf_counter() - t0) * 1000
    return detected, result.injection_score, latency_ms


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_deepset_prompt_injections() -> list[tuple[str, bool]]:
    """
    deepset/prompt-injections: 662 samples.
    Label: 1 = injection, 0 = benign.
    """
    print("  Downloading deepset/prompt-injections...")
    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
        samples = []
        for row in ds:
            text = row.get("text", "")
            label = row.get("label", 0)
            if text and len(text.strip()) > 5:
                samples.append((text, label == 1))
        print(f"  Loaded {len(samples)} samples from deepset/prompt-injections")
        return samples
    except Exception as e:
        print(f"  WARNING: Failed to load deepset/prompt-injections: {e}")
        return []


def load_jailbreak_classification() -> list[tuple[str, bool]]:
    """
    jackhhao/jailbreak-classification: ~1,300 samples.
    Label: 'jailbreak' = attack, 'benign' = clean.
    """
    print("  Downloading jackhhao/jailbreak-classification...")
    try:
        ds = load_dataset("jackhhao/jailbreak-classification", split="train")
        samples = []
        for row in ds:
            text = row.get("prompt", "")
            label = row.get("type", "benign")
            if text and len(text.strip()) > 5:
                is_attack = label.lower() in ("jailbreak", "injection", "attack")
                samples.append((text, is_attack))
        print(f"  Loaded {len(samples)} samples from jackhhao/jailbreak-classification")
        return samples
    except Exception as e:
        print(f"  WARNING: Failed to load jackhhao/jailbreak-classification: {e}")
        return []


def load_notinject() -> list[tuple[str, bool]]:
    """
    leolee99/NotInject: 339 samples — all benign but contain trigger words.
    This is a FALSE POSITIVE stress test.
    Label: all samples are benign (is_attack=False).
    """
    print("  Downloading leolee99/NotInject...")
    try:
        samples = []
        for split_name in ["NotInject_one", "NotInject_two", "NotInject_three"]:
            try:
                ds = load_dataset("leolee99/NotInject", split=split_name)
                for row in ds:
                    # Try common field names
                    text = row.get("text", row.get("prompt", row.get("content", row.get("instruction", ""))))
                    if text and len(text.strip()) > 5:
                        samples.append((text, False))
            except Exception:
                pass
        print(f"  Loaded {len(samples)} samples from leolee99/NotInject")
        return samples
    except Exception as e:
        print(f"  WARNING: Failed to load leolee99/NotInject: {e}")
        return []


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    samples: list[tuple[str, bool]],
    threshold: float = 0.50,
    max_fp_samples: int = 20,
    max_fn_samples: int = 20,
) -> DatasetResult:
    """Run scanner on all samples and compute metrics."""
    result = DatasetResult(name=name, total=len(samples))

    for i, (text, is_attack) in enumerate(samples):
        try:
            detected, score, latency_ms = scan_text(text, threshold)
            result.latency_ms.append(latency_ms)

            if is_attack and detected:
                result.true_positives += 1
            elif is_attack and not detected:
                result.false_negatives += 1
                if len(result.false_negative_samples) < max_fn_samples:
                    result.false_negative_samples.append({
                        "text": text[:200],
                        "score": round(score, 4),
                    })
            elif not is_attack and detected:
                result.false_positives += 1
                if len(result.false_positive_samples) < max_fp_samples:
                    result.false_positive_samples.append({
                        "text": text[:200],
                        "score": round(score, 4),
                    })
            else:
                result.true_negatives += 1

        except Exception as e:
            result.errors += 1
            if result.errors <= 3:
                print(f"  Error on sample {i}: {e}")

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(samples)}...")

    return result


def print_report(result: DatasetResult) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {result.name}")
    print(f"{'='*60}")
    print(f"  Total samples:     {result.total}")
    print(f"  True Positives:    {result.true_positives}")
    print(f"  False Positives:   {result.false_positives}")
    print(f"  True Negatives:    {result.true_negatives}")
    print(f"  False Negatives:   {result.false_negatives}")
    print(f"  Errors:            {result.errors}")
    print(f"  ---")
    print(f"  TPR (Recall):      {result.tpr:.4f}  ({result.tpr*100:.1f}%)")
    print(f"  FPR:               {result.fpr:.4f}  ({result.fpr*100:.1f}%)")
    print(f"  Precision:         {result.precision:.4f}  ({result.precision*100:.1f}%)")
    print(f"  F1 Score:          {result.f1:.4f}  ({result.f1*100:.1f}%)")
    print(f"  Avg Latency:       {result.avg_latency_ms:.2f} ms")
    print(f"  P95 Latency:       {result.p95_latency_ms:.2f} ms")

    if result.false_negative_samples:
        print(f"\n  --- Missed attacks (first {len(result.false_negative_samples)}) ---")
        for fn in result.false_negative_samples[:10]:
            print(f"    score={fn['score']:.4f}  text={fn['text'][:120]}...")

    if result.false_positive_samples:
        print(f"\n  --- False alarms (first {len(result.false_positive_samples)}) ---")
        for fp in result.false_positive_samples[:10]:
            print(f"    score={fp['score']:.4f}  text={fp['text'][:120]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  Aegivis — External Dataset Validation")
    print("  Structural scanner only (no GPU / sentence-transformers)")
    print("="*60)

    # Thresholds to test
    threshold = 0.50  # Same as production default

    all_results: list[DatasetResult] = []

    # --- Dataset 1: deepset/prompt-injections ---
    print("\n[1/3] deepset/prompt-injections")
    ds1 = load_deepset_prompt_injections()
    if ds1:
        r1 = evaluate_dataset("deepset/prompt-injections", ds1, threshold)
        print_report(r1)
        all_results.append(r1)

    # --- Dataset 2: jackhhao/jailbreak-classification ---
    print("\n[2/3] jackhhao/jailbreak-classification")
    ds2 = load_jailbreak_classification()
    if ds2:
        r2 = evaluate_dataset("jackhhao/jailbreak-classification", ds2, threshold)
        print_report(r2)
        all_results.append(r2)

    # --- Dataset 3: leolee99/NotInject ---
    print("\n[3/3] leolee99/NotInject (FPR stress test)")
    ds3 = load_notinject()
    if ds3:
        r3 = evaluate_dataset("leolee99/NotInject (FPR stress)", ds3, threshold)
        print_report(r3)
        all_results.append(r3)

    # --- Aggregate ---
    if all_results:
        print("\n" + "="*60)
        print("  AGGREGATE RESULTS")
        print("="*60)
        total_tp = sum(r.true_positives for r in all_results)
        total_fp = sum(r.false_positives for r in all_results)
        total_tn = sum(r.true_negatives for r in all_results)
        total_fn = sum(r.false_negatives for r in all_results)
        total_n  = sum(r.total for r in all_results)

        agg_tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        agg_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
        agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        agg_f1 = 2 * agg_prec * agg_tpr / (agg_prec + agg_tpr) if (agg_prec + agg_tpr) > 0 else 0

        print(f"  Total samples:     {total_n}")
        print(f"  True Positives:    {total_tp}")
        print(f"  False Positives:   {total_fp}")
        print(f"  True Negatives:    {total_tn}")
        print(f"  False Negatives:   {total_fn}")
        print(f"  ---")
        print(f"  TPR (Recall):      {agg_tpr:.4f}  ({agg_tpr*100:.1f}%)")
        print(f"  FPR:               {agg_fpr:.4f}  ({agg_fpr*100:.1f}%)")
        print(f"  Precision:         {agg_prec:.4f}  ({agg_prec*100:.1f}%)")
        print(f"  F1 Score:          {agg_f1:.4f}  ({agg_f1*100:.1f}%)")

    # Save detailed results to JSON
    output_path = Path(__file__).parent / "external_dataset_results.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": threshold,
        "scanner": "structural_only (Layer 1)",
        "datasets": [],
    }
    for r in all_results:
        report["datasets"].append({
            "name": r.name,
            "total": r.total,
            "tp": r.true_positives,
            "fp": r.false_positives,
            "tn": r.true_negatives,
            "fn": r.false_negatives,
            "tpr": round(r.tpr, 4),
            "fpr": round(r.fpr, 4),
            "precision": round(r.precision, 4),
            "f1": round(r.f1, 4),
            "avg_latency_ms": round(r.avg_latency_ms, 2),
            "p95_latency_ms": round(r.p95_latency_ms, 2),
            "false_negatives": r.false_negative_samples[:20],
            "false_positives": r.false_positive_samples[:20],
        })

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
