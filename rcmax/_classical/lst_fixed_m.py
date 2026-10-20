from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import combinations
import math

import numpy as np

from rcmax._classical.lst_two_approx import greedy_schedule
from rcmax._core import calc_loads, makespan, ScheduleResult
from rcmax._lp import solve_residual_lp
from rcmax._rouding import shmoys_tardos_round_residual


def _feasible_subsets_for_machine(
    p: np.ndarray,
    machine: int,
    available_jobs: Sequence[int],
    epsilon: float,
    T: float,
    max_count: int,
) -> Iterator[tuple[int, ...]]:
    threshold = epsilon * T
    candidates = [
        j for j in available_jobs if np.isfinite(p[machine, j]) and threshold < p[machine, j] <= T
    ]

    # Empty long assignment set for this machine is always possible.
    yield tuple()

    for s in range(1, min(max_count, len(candidates)) + 1):
        for U in combinations(candidates, s):
            load = sum(float(p[machine, j]) for j in U)
            if load <= T + 1e-9:
                yield tuple(U)


def _max_long_jobs_per_machine(epsilon: float) -> int:
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0,1)")
    return max(0, math.ceil(1.0 / epsilon) - 1)


def _enumerate_long_assignment_schedules(
    p: np.ndarray,
    epsilon: float,
    T: float,
    *,
    max_schedules: int | None = None,
) -> Iterator[np.ndarray]:
    m, n = p.shape
    max_count = _max_long_jobs_per_machine(epsilon)

    psi = np.full(n, -1, dtype=int)
    available = tuple(range(n))
    emitted = 0

    def rec(machine: int, available_jobs: tuple[int, ...]) -> Iterator[np.ndarray]:
        nonlocal emitted

        if max_schedules is not None and emitted >= max_schedules:
            return

        if machine == m:
            emitted += 1
            yield psi.copy()
            return

        for U in _feasible_subsets_for_machine(p, machine, available_jobs, epsilon, T, max_count):
            psi[list(U)] = machine

            next_available = tuple(j for j in available_jobs if j not in U)
            yield from rec(machine + 1, next_available)

            psi[list(U)] = -1

            if max_schedules is not None and emitted >= max_schedules:
                return

    yield from rec(0, available)


def lst_fixed_m_decision(
    p: np.ndarray,
    epsilon: float,
    T: float,
    *,
    max_long_schedules: int | None = None,
) -> ScheduleResult | None:
    best: ScheduleResult | None = None
    threshold = epsilon * T

    for psi in _enumerate_long_assignment_schedules(
        p, epsilon, T, max_schedules=max_long_schedules
    ):
        fixed = calc_loads(p, psi)
        if np.any(fixed > T + 1e-9):
            continue

        lp = solve_residual_lp(p, psi, tau=threshold)

        if not lp.feasible:
            continue
        if lp.value > T + 1e-7:
            continue

        schedule = shmoys_tardos_round_residual(p, psi, lp, tau=threshold)
        if schedule is None:
            continue

        cur = ScheduleResult(
            schedule=schedule,
            makespan=makespan(p, schedule),
        )

        if best is None or cur.makespan < best.makespan:
            best = cur

    return best


def lst_fixed_m_ptas(
    p: np.ndarray,
    epsilon: float,
    *,
    iterations: int = 30,
    max_long_schedules_per_T: int | None = None,
    verbose: bool = False,
) -> ScheduleResult:
    _, n = p.shape

    left = max(float(np.min(p[np.isfinite(p[:, j]), j])) for j in range(n))

    greedy = greedy_schedule(p)
    right = makespan(p, greedy)

    initial = lst_fixed_m_decision(
        p,
        epsilon,
        right,
        max_long_schedules=max_long_schedules_per_T,
    )
    if initial is None:
        raise RuntimeError(
            "LST fixed-m decision did not succeed at the greedy upper bound. "
            "If max_long_schedules_per_T is set, increase it or set it to None."
        )

    best = initial

    for i in range(iterations):
        mid = (left + right) / 2.0
        cur = lst_fixed_m_decision(
            p,
            epsilon,
            mid,
            max_long_schedules=max_long_schedules_per_T,
        )

        if cur is not None:
            right = mid
            best = cur
        else:
            left = mid
        if verbose:
            print(f"iteration {i + 1}/{iterations}: T={mid:.6g}, best={best.makespan:.6g}")

    assert best is not None
    return best
