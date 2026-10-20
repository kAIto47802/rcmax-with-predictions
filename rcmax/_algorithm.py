from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations, product

import numpy as np

from rcmax._classical.lst_two_approx import lst_two_approx
from rcmax._core import makespan, ScheduleResult
from rcmax._lp import solve_residual_lp
from rcmax._rouding import shmoys_tardos_round_residual


def heavy_projection(
    p: np.ndarray,  # (m, n)
    pred: np.ndarray,  # (n,)
    epsilon: float,
    T: float,
) -> np.ndarray:  # (n,)
    _, n = p.shape
    phi_T = np.full(n, -1, dtype=int)
    threshold = epsilon * T
    partial = pred[assigned := pred != -1]
    jobs = np.flatnonzero(assigned)
    times = p[partial, jobs]
    heavy = np.isfinite(times) & (times > threshold)
    phi_T[jobs[heavy]] = partial[heavy]
    return phi_T


def _threshold_candidates(
    p: np.ndarray,  # (m, n)
    epsilon: float,
    T: float,
) -> np.ndarray:
    threshold = epsilon * T
    values = p[np.isfinite(p) & (p > threshold)]
    return np.unique(np.r_[values, threshold])


def _critical_thresholds(
    p: np.ndarray,  # (m, n)
    epsilon: float,
    baseline: float,
) -> np.ndarray:
    low, high = baseline / 2.0, baseline
    p_eps = p / epsilon
    values = p_eps[np.isfinite(p_eps) & (low <= p_eps) & (p_eps <= high)]
    return np.unique(np.r_[values, low, high])


def _enumerate_augmentations(
    p: np.ndarray,  # (m, n)
    phi_T: np.ndarray,  # (n,)
    epsilon: float,
    T: float,
    K: int,
) -> Iterator[dict[int, int]]:
    if K < 0:
        raise ValueError("K must be non-negative.")
    _, n = p.shape
    threshold = epsilon * T
    options = {
        j: np.flatnonzero(np.isfinite(pj := p[:, j]) & (pj > threshold))
        for j in range(n)
        if phi_T[j] == -1
    }
    options = {j: machines for j, machines in options.items() if machines.size > 0}

    yield {}  # empty augmentation
    max_s = min(K, len(options))
    for s in range(1, max_s + 1):
        for subset in combinations(options, s):
            for assignment in product(*(options[j] for j in subset)):
                yield dict(zip(subset, assignment))  # type: ignore


def prediction_guided_completion(
    p: np.ndarray,
    pred: np.ndarray,
    T: float,
    epsilon: float,
    K: int,
) -> ScheduleResult | None:
    phi_T = heavy_projection(p, pred, epsilon, T)
    taus = _threshold_candidates(p, epsilon, T)

    best: ScheduleResult | None = None

    for A in _enumerate_augmentations(p, phi_T, epsilon, T, K):
        psi = phi_T.copy()
        for j, i in A.items():
            psi[j] = i

        for tau in taus:
            lp = solve_residual_lp(p, psi, tau)
            if not lp.feasible:
                continue

            sched = shmoys_tardos_round_residual(p, psi, lp, tau)
            if sched is None:
                continue

            ms = makespan(p, sched)
            if best is None or ms < best.makespan:
                best = ScheduleResult(sched, ms)
            break

    return best


def prediction_localized_search(
    p: np.ndarray,
    pred: np.ndarray,
    epsilon: float,
    K: int,
    *,
    verbose: bool = False,
) -> ScheduleResult:
    baseline = lst_two_approx(p)
    B = baseline.makespan
    T_values = _critical_thresholds(p, epsilon, B)

    best = baseline
    candidates = [baseline]

    if verbose:
        print(f"[outer] baseline makespan={B:.6g}, |T|={len(T_values)}")

    for idx, T in enumerate(T_values):
        res = prediction_guided_completion(p, pred, T, epsilon, K)
        if res is not None:
            candidates.append(res)
            if res.makespan < best.makespan:
                best = res
        if verbose:
            print(f"[outer] {idx + 1}/{len(T_values)} T={T:.6g}, current best={best.makespan:.6g}")
    return best
