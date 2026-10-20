from __future__ import annotations

from collections import deque

import numpy as np

from rcmax._core import LPSolution, makespan


def shmoys_tardos_round_residual(
    p: np.ndarray,
    psi: np.ndarray,
    lp: LPSolution,
    tau: float,
    *,
    support_tol: float = 1e-10,
    job_sum_tol: float = 1e-6,
    check_bound: bool = True,
    bound_tol: float = 1e-5,
) -> np.ndarray | None:
    if not lp.feasible or lp.x is None or lp.residual_jobs is None:
        print("LP solution is not feasible or missing required fields.")
        return None

    m, _ = p.shape
    schedule = np.asarray(psi, dtype=int).copy()
    residual_jobs = lp.residual_jobs
    nr = len(residual_jobs)

    if not nr:
        return schedule

    x = np.asarray(lp.x, dtype=float).copy()
    if x.shape != (m, nr):
        return None

    # The Shmoys--Tardos construction assumes sum_i x_{ij}=1 for each job.
    # We allow tiny numerical deviations, but reject substantial violations.
    for jj in range(nr):
        s = float(np.sum(x[:, jj]))
        if not np.isfinite(s) or s <= support_tol:
            return None
        if abs(s - 1.0) > job_sum_tol:
            return None
        # Normalize only tiny numerical error; this does not change the support.
        x[:, jj] /= s

    slot_machine: list[int] = []
    edges: list[list[int]] = [[] for _ in range(nr)]

    for i in range(m):
        positive = [
            (jj, residual_jobs[jj], float(x[i, jj])) for jj in range(nr) if x[i, jj] > support_tol
        ]
        if not positive:
            continue

        positive.sort(key=lambda item: p[i, item[1]], reverse=True)

        current_slot: int | None = None
        remaining = 0.0

        def new_slot() -> int:
            slot_id = len(slot_machine)
            slot_machine.append(i)
            return slot_id

        for jj, j, mass0 in positive:
            mass = mass0
            while mass > support_tol:
                if current_slot is None or remaining <= support_tol:
                    current_slot = new_slot()
                    remaining = 1.0

                edges[jj].append(current_slot)

                use = min(mass, remaining)
                mass -= use
                remaining -= use

    # Remove duplicate edges that may arise from numerical splitting.
    edges = [sorted(set(adj)) for adj in edges]

    if any(len(adj) == 0 for adj in edges):
        return None

    match = _hopcroft_karp(edges, len(slot_machine))
    if match is None:
        return None

    for jj, slot_id in enumerate(match):
        if slot_id < 0:
            return None
        j = residual_jobs[jj]
        schedule[j] = slot_machine[slot_id]

    if np.any(schedule == -1):
        return None

    if check_bound:
        ms = makespan(p, schedule)
        target = float(lp.value + tau)
        if ms > target + bound_tol * max(1.0, target):
            return None

    return schedule


def _hopcroft_karp(edges: list[list[int]], num_right: int) -> list[int] | None:
    n_left = len(edges)
    match_left = [-1] * n_left
    match_right = [-1] * num_right
    dist = [0] * n_left

    def bfs() -> bool:
        q: deque[int] = deque()
        found_free = False
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in edges[u]:
                u2 = match_right[v]
                if u2 == -1:
                    found_free = True
                elif dist[u2] == -1:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
        return found_free

    def dfs(u: int) -> bool:
        for v in edges[u]:
            u2 = match_right[v]
            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = -1
        return False

    matching_size = 0
    while bfs():
        for u in range(n_left):
            if match_left[u] == -1 and dfs(u):
                matching_size += 1

    if matching_size != n_left:
        return None
    return match_left
