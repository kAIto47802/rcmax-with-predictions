from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from rcmax._classical.lst_two_approx import greedy_schedule
from rcmax._core import makespan, ScheduleResult
from rcmax._lp import solve_residual_lp
from rcmax._rouding import shmoys_tardos_round_residual


def _large_pairs(
    p: np.ndarray,
    epsilon: float,
    T: float,
    *,
    only_feasible_for_T: bool = True,
) -> list[tuple[int, int]]:
    m, n = p.shape
    threshold = epsilon * T
    pairs: list[tuple[int, int]] = []
    for i in range(m):
        for j in range(n):
            if not np.isfinite(p[i, j]):
                continue
            if p[i, j] <= threshold:
                continue
            if only_feasible_for_T and p[i, j] > T + 1e-9:
                continue
            pairs.append((i, j))
    return pairs


def _search_large_pair_patterns(
    p: np.ndarray,
    pairs: list[tuple[int, int]],
    T: float,
    *,
    max_patterns: int | None = None,
) -> Iterator[np.ndarray]:
    m, n = p.shape

    psi = np.full(n, -1, dtype=int)
    fixed_load = np.zeros(m, dtype=float)
    emitted = 0

    pairs_sorted = sorted(pairs, key=lambda ij: p[ij[0], ij[1]], reverse=True)

    def rec(q: int) -> Iterator[np.ndarray]:
        nonlocal emitted

        if max_patterns is not None and emitted >= max_patterns:
            return

        if q == len(pairs_sorted):
            emitted += 1
            yield psi.copy()
            return

        i, j = pairs_sorted[q]

        yield from rec(q + 1)

        if max_patterns is not None and emitted >= max_patterns:
            return

        if psi[j] == -1 and fixed_load[i] + p[i, j] <= T + 1e-9:
            psi[j] = i
            fixed_load[i] += p[i, j]

            yield from rec(q + 1)

            fixed_load[i] -= p[i, j]
            psi[j] = -1

    yield from rec(0)


def mordechai_decision(
    p: np.ndarray,
    epsilon: float,
    T: float,
    *,
    max_patterns: int | None = None,
) -> ScheduleResult | None:
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0,1)")

    pairs = _large_pairs(p, epsilon, T, only_feasible_for_T=True)
    tau = epsilon * T

    best: ScheduleResult | None = None
    num_patterns = 0
    num_lps = 0
    num_feas = 0

    for psi in _search_large_pair_patterns(p, pairs, T, max_patterns=max_patterns):
        num_patterns += 1

        num_lps += 1
        lp = solve_residual_lp(p, psi, tau=tau)

        if not lp.feasible:
            continue
        if lp.value > T + 1e-7:
            continue

        num_feas += 1
        sched = shmoys_tardos_round_residual(p, psi, lp, tau=tau)
        if sched is None:
            continue

        ms = makespan(p, sched)
        cand = ScheduleResult(schedule=sched, makespan=ms)

        if best is None or cand.makespan < best.makespan:
            best = cand

    return best


def mordechai_schedule(
    p: np.ndarray,
    epsilon: float,
    *,
    iterations: int = 30,
    max_patterns_per_T: int | None = None,
    verbose: bool = False,
) -> ScheduleResult:
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0,1)")

    _, n = p.shape

    left = max(float(np.min(p[np.isfinite(p[:, j]), j])) for j in range(n))
    greedy = greedy_schedule(p)
    right = makespan(p, greedy)

    initial = mordechai_decision(
        p,
        epsilon,
        right,
        max_patterns=max_patterns_per_T,
    )

    if initial is None:
        raise RuntimeError(
            "Mordechai decision did not succeed at the greedy upper bound. "
            "If max_patterns_per_T is set, increase it or set it to None."
        )

    best = initial

    for _ in range(iterations):
        mid = (left + right) / 2.0
        dec = mordechai_decision(
            p,
            epsilon,
            mid,
            max_patterns=max_patterns_per_T,
        )

        if dec is not None:
            right = mid
            best = dec
        else:
            left = mid

        if verbose:
            print(f"Iteration: {_:2d}, T: {mid:.4f}, best: {best.makespan:.4f}")

    assert best is not None
    return best
