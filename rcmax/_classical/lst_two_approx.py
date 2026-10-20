from __future__ import annotations

import numpy as np

from rcmax._core import LPSolution, makespan, ScheduleResult
from rcmax._lp import solve_lpst_feasibility
from rcmax._rouding import shmoys_tardos_round_residual


def greedy_schedule(p: np.ndarray) -> np.ndarray:
    m, n = p.shape
    loads = np.zeros(m, dtype=float)
    schedule = np.full(n, -1, dtype=int)
    order = sorted(range(n), key=lambda j: np.min(p[:, j]), reverse=True)
    for j in order:
        best_i = None
        best_val = float("inf")
        for i in range(m):
            if np.isfinite(p[i, j]):
                val = loads[i] + p[i, j]
                if val < best_val:
                    best_val = val
                    best_i = i
        if best_i is None:
            raise ValueError(f"job {j} has no finite processing time on any machine")
        schedule[j] = best_i
        loads[best_i] += p[best_i, j]
    return schedule


def lst_two_approx(
    p: np.ndarray,
    *,
    iterations: int = 40,
    tol: float = 1e-7,
) -> ScheduleResult:
    _, n = p.shape

    left = max(float(np.min(p[np.isfinite(p[:, j]), j])) for j in range(n))
    greedy = greedy_schedule(p)
    right = makespan(p, greedy)

    best_x = solve_lpst_feasibility(p, right, tol=tol).x
    best_T = right

    if best_x is None:
        raise RuntimeError("The LST feasibility LP should be feasible at the greedy upper bound.")

    for _ in range(iterations):
        mid = (left + right) / 2.0
        x = solve_lpst_feasibility(p, mid, tol=tol).x
        if x is not None:
            right = mid
            best_x = x
            best_T = mid
        else:
            left = mid

    psi_empty = np.full(n, -1, dtype=int)
    lp = LPSolution(
        feasible=True,
        value=best_T,
        x=best_x,
        residual_jobs=np.arange(n),
    )

    schedule = shmoys_tardos_round_residual(p, psi_empty, lp, tau=best_T)
    if schedule is None:
        raise RuntimeError("Shmoys--Tardos rounding failed; check numerical tolerances.")

    return ScheduleResult(
        schedule,
        makespan(p, schedule),
    )
