from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack

from rcmax._core import calc_loads, LPSolution


def solve_residual_lp(
    p: np.ndarray,  # (m, n)
    psi: np.ndarray,  # (n,)
    tau: float,
    *,
    tol: float = 1e-9,
) -> LPSolution:
    m, _ = p.shape
    residual_jobs = np.flatnonzero(psi == -1)
    fixed = calc_loads(p, psi)
    if np.any(~np.isfinite(fixed)):
        return LPSolution(feasible=False, value=float("inf"))
    if not np.all((np.isfinite(rp := p[:, residual_jobs]) & (rp <= tau + tol)).sum(axis=0)):
        return LPSolution(feasible=False, value=float("inf"))

    nr = residual_jobs.shape[0]
    num_x = m * nr
    p_res = p[:, residual_jobs]

    c = np.zeros(num_x + 1)
    c[num_x] = 1.0

    Aeq_x = hstack([eye(nr, format="csr")] * m, format="csr")
    Aeq = hstack([Aeq_x, csr_matrix((nr, 1))], format="csr")
    beq = np.ones(nr)

    allowed = np.isfinite(p_res) & (p_res <= tau + tol)
    cols = np.flatnonzero(allowed.ravel())
    rows = cols // nr
    data = p_res.ravel()[cols]
    Aub_x = coo_matrix((data, (rows, cols)), shape=(m, num_x))
    Aub_L = -csr_matrix(np.ones((m, 1)))
    Aub = hstack([Aub_x, Aub_L], format="csr")
    bub = -fixed

    lb = np.zeros(num_x + 1)
    lb[num_x] = np.max(fixed)
    ub = np.r_[np.where(allowed.ravel(), np.inf, 0.0), np.inf]
    bounds = list(zip(lb, ub))

    res = linprog(
        c,
        A_ub=Aub,
        b_ub=bub,
        A_eq=Aeq,
        b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return LPSolution(feasible=False, value=float("inf"))
    x = res.x[:num_x].reshape((m, nr))
    x[np.abs(x) < tol] = 0.0  # Clean numerical noise
    return LPSolution(feasible=True, value=res.fun, x=x, residual_jobs=residual_jobs)


def solve_lpst_feasibility(
    p: np.ndarray,  # shape (m, n)
    T: float,
    *,
    tol: float = 1e-9,
) -> LPSolution:
    m, n = p.shape

    allowed = np.isfinite(p) & (p <= T + tol)

    if not np.all(allowed.sum(axis=0) > 0):
        return LPSolution(feasible=False, value=float("inf"))

    num_x = m * n

    c = np.zeros(num_x)

    Aeq = hstack([eye(n, format="csr")] * m, format="csr")
    beq = np.ones(n)

    cols = np.flatnonzero(allowed.ravel())
    rows = cols // n
    data = p.ravel()[cols]
    Aub = coo_matrix((data, (rows, cols)), shape=(m, num_x)).tocsr()
    bub = np.full(m, T)

    lb = np.zeros(num_x)
    ub = np.where(allowed.ravel(), np.inf, 0.0)
    bounds = list(zip(lb, ub))

    res = linprog(
        c,
        A_ub=Aub,
        b_ub=bub,
        A_eq=Aeq,
        b_eq=beq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return LPSolution(feasible=False, value=float("inf"))

    x = res.x.reshape((m, n))
    x[np.abs(x) < tol] = 0.0
    return LPSolution(feasible=True, value=res.fun, x=x)
