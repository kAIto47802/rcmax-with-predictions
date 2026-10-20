from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle

from experiments.prepare_data import load_instance
from rcmax import solve_rcmax_cp_sat


def main(args: Namespace) -> None:
    output_path = Path(args.output_dir) / f"rcmax_{args.instance}_reference.pkl"
    if output_path.exists():
        print(
            f"Reference result for instance '{args.instance}' already exists at '{output_path}'. "
            "Skipping computation."
        )
        return
    p = load_instance(args.instance)
    res = solve_rcmax_cp_sat(
        p,
        time_limit_sec=args.time_limit_sec,
        num_workers=args.num_workers,
        log_search=args.log_search,
        add_greedy_hint=args.add_greedy_hint,
    )
    print(
        f"[[Reference schedule]] feasible: {res.feasible}, optimal: {res.optimal}, "
        f"status: {res.status}, makespan: {res.makespan}, best_bound: {res.best_bound}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(res, f)
    output_path.with_suffix(".txt").write_text(
        f"Feasible: {res.feasible}\nOptimal: {res.optimal}\nStatus: {res.status}\n"
        f"Makespan: {res.makespan}\nBest bound: {res.best_bound}\n"
        f"Schedule: {res.schedule.tolist() if res.schedule is not None else 'None'}\n"
    )
    print(f"Saved reference result to '{output_path}'.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--instance",
        type=str,
        help="instance name (without 'rcmax_' prefix and '.dat' suffix)",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=None,
        help="CP-SAT time limit in seconds (default: no limit)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="number of CP-SAT search workers (default: 16)",
    )
    parser.add_argument(
        "--log-search",
        action="store_true",
        help="whether to print CP-SAT search logs",
    )
    parser.add_argument(
        "--no-greedy-hint",
        action="store_false",
        dest="add_greedy_hint",
        help="whether to add a greedy feasible schedule as a hint (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="directory to save results (default: 'results')",
    )
    args = parser.parse_args()
    main(args)
