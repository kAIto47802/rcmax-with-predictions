from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcmax._algorithm import heavy_projection


@dataclass(frozen=True)
class ErrorMeasures:
    R_plus: float
    R_minus: float
    R_minus_K: float


def calc_error_measures(
    p: np.ndarray,  # (m, n)
    pred: np.ndarray,  # (n,)
    ref: np.ndarray,  # (n,)
    epsilon: float,
    T: float,
    K: int,
) -> ErrorMeasures:
    if K < 0:
        raise ValueError("K must be non-negative.")
    m, n = p.shape
    phi_T = heavy_projection(p, pred, epsilon, T)

    fp_jobs = np.flatnonzero((phi_T != -1) & (phi_T != ref))
    times = p[pred_fp := phi_T[fp_jobs], fp_jobs]
    R_plus = np.max(np.bincount(pred_fp, weights=times, minlength=m)) if fp_jobs.size else 0.0

    missed_jobs = np.flatnonzero((phi_T == -1) & (p[ref, np.arange(n)] > epsilon * T))
    missed_times = p[ref[missed_jobs], missed_jobs]
    R_minus = np.max(missed_times) if missed_jobs.size else 0.0

    R_minus_K = np.partition(missed_times, -K - 1)[-K - 1] if missed_jobs.size > K else 0.0

    return ErrorMeasures(R_plus=float(R_plus), R_minus=R_minus, R_minus_K=R_minus_K)
