"""Evaluation metrics.

- exact_match / edit_similarity: lexical baselines.
- pass_at_k: execution-based, requires per-example test_code.
- uncertainty_calibration_ece: Expected Calibration Error of the
  aggregate uncertainty score against actual correctness. Central to
  RQ2 — measures whether the uncertainty signal is *useful*, not just
  reported.
"""
from __future__ import annotations

import difflib
import math
from typing import Sequence

import numpy as np


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip() == ref.strip() else 0.0


def edit_similarity(pred: str, ref: str) -> float:
    return float(difflib.SequenceMatcher(None, pred, ref).ratio())


def pass_at_k(correct_per_example: Sequence[bool], k: int) -> float:
    """Unbiased pass@k = mean of indicator(any of k samples correct).

    For the n-sample-equals-k case this reduces to the empirical pass rate.
    """
    return float(np.mean([1.0 if c else 0.0 for c in correct_per_example]))


def uncertainty_calibration_ece(
    uncertainties: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """ECE over (1 - uncertainty) as the model's confidence."""
    assert len(uncertainties) == len(correctness)
    if not uncertainties:
        return 0.0
    confidences = np.clip(1.0 - np.asarray(uncertainties, dtype=float), 0.0, 1.0)
    correct = np.asarray(correctness, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
