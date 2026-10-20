from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LPSolution:
    feasible: bool
    value: float
    x: np.ndarray | None = None  # (m, num_residual_jobs)
    residual_jobs: np.ndarray | None = None  # (num_residual_jobs,)


@dataclass
class ScheduleResult:
    schedule: np.ndarray  # (n,)
    makespan: float


def makespan(
    p: np.ndarray,  # (m, n)
    schedule: np.ndarray,  # (n,)
) -> float:
    loads = calc_loads(p, schedule)
    return float(np.max(loads))


def validate_schedule(schedule: np.ndarray, m: int, n: int) -> None:
    assert schedule.shape == (n,), "Schedule must be a 1D array of length n."
    assert np.all((schedule >= -1) & (schedule < m)), "Schedule entries must be in {-1,...,m-1}."
    assert np.issubdtype(schedule.dtype, np.integer), "Schedule entries must be integers."


def calc_loads(
    p: np.ndarray,  # (m, n)
    schedule: np.ndarray,  # (n,)
) -> np.ndarray:  # (m,)
    m, _ = p.shape
    partial = schedule[assigned := schedule != -1]
    times = p[partial, np.flatnonzero(assigned)]
    loads = np.bincount(partial, weights=times, minlength=m)
    return loads.astype(float)
