from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from rcmax._classical.lst_two_approx import greedy_schedule
from rcmax._core import makespan


@dataclass
class ReferenceScheduleResult:
    feasible: bool
    optimal: bool
    status: str
    schedule: np.ndarray | None
    makespan: float
    best_bound: float


def solve_rcmax_cp_sat(
    p: np.ndarray,
    *,
    time_limit_sec: float | None = None,
    num_workers: int = 16,
    log_search: bool = False,
    add_greedy_hint: bool = True,
) -> ReferenceScheduleResult:
    from ortools.sat.python import cp_model

    q, finite = _to_integer_processing_times(p)
    m, n = q.shape

    for j in range(n):
        if not np.any(finite[:, j]):
            raise ValueError(f"job {j} has no feasible machine")

    greedy = greedy_schedule(p)
    greedy_ub = math.ceil(makespan(p, greedy))
    print(f"Greedy schedule makespan: {greedy_ub}")

    model = cp_model.CpModel()

    x = {}
    for i in range(m):
        for j in range(n):
            if finite[i, j]:
                x[i, j] = model.new_bool_var(f"x_{i}_{j}")

    # A safe lower bound is the largest best processing time over jobs.
    lb = max(int(np.min(q[finite[:, j], j])) for j in range(n))
    Cmax = model.new_int_var(lb, greedy_ub, "Cmax")

    # Each job is assigned to exactly one feasible machine.
    for j in range(n):
        model.add(sum(x[i, j] for i in range(m) if (i, j) in x) == 1)

    # Machine-load constraints.
    for i in range(m):
        model.add(sum(int(q[i, j]) * x[i, j] for j in range(n) if (i, j) in x) <= Cmax)

    model.minimize(Cmax)

    if add_greedy_hint:
        for j, i_star in enumerate(greedy):
            for i in range(m):
                if (i, j) in x:
                    model.add_hint(x[i, j], 1 if i == i_star else 0)
        model.add_hint(Cmax, greedy_ub)

    solver = cp_model.CpSolver()
    if time_limit_sec is not None:
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.log_search_progress = bool(log_search)

    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)

    feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    optimal = status_code == cp_model.OPTIMAL

    if not feasible:
        try:
            best_bound = float(solver.BestObjectiveBound())
        except Exception:
            best_bound = float("inf")
        return ReferenceScheduleResult(
            feasible=False,
            optimal=False,
            status=status,
            schedule=None,
            makespan=float("inf"),
            best_bound=best_bound if np.isfinite(best_bound) else float("inf"),
        )

    schedule = np.full(n, -1, dtype=int)
    for j in range(n):
        for i in range(m):
            if (i, j) in x and solver.Value(x[i, j]) == 1:
                schedule[j] = i
                break
        if schedule[j] == -1:
            raise RuntimeError(f"solver returned no assignment for job {j}")

    best_bound = float(solver.BestObjectiveBound())

    return ReferenceScheduleResult(
        feasible=True,
        optimal=optimal,
        status=status,
        schedule=schedule,
        makespan=makespan(p, schedule),
        best_bound=best_bound if np.isfinite(best_bound) else math.inf,
    )


def _to_integer_processing_times(
    p: np.ndarray,
    *,
    scale: int = 1,
    require_integral: bool = True,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if scale <= 0 or int(scale) != scale:
        raise ValueError("scale must be a positive integer")
    scale = int(scale)

    finite = np.isfinite(p)
    q = np.zeros_like(p, dtype=np.int64)

    scaled = p[finite] * scale
    rounded = np.rint(scaled)

    if require_integral and not np.all(np.abs(scaled - rounded) <= atol):
        max_err = float(np.max(np.abs(scaled - rounded)))
        raise ValueError(
            "CP-SAT requires integer processing times. "
            f"After multiplying by scale={scale}, some coefficients are non-integral "
            f"(max error {max_err}). Increase scale or set require_integral=False."
        )

    q[finite] = rounded.astype(np.int64)
    if np.any(q[finite] <= 0):
        raise ValueError("scaled finite processing times must be positive integers")
    return q, finite
