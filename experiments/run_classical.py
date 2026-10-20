from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle
import time

from experiments.prepare_data import load_instance
from rcmax import lst_fixed_m_ptas, lst_two_approx, mordechai_schedule


def main(args: Namespace) -> None:
    output_path = Path(args.output_dir) / (
        f"rcmax_{args.instance}_{args.algorithm}"
        + f"_eps{args.epsilon}" * (args.algorithm == "lst_fixed_m_ptas")
        + f"_trial{args.trial_id}_pkl"
    ).replace(".", "").replace("_pkl", ".pkl")
    if output_path.exists():
        print(
            f"Result for instance '{args.instance}' with '{args.algorithm}' "
            f"already exists at '{output_path}'. Skipping computation."
        )
        return

    p = load_instance(args.instance)
    start = time.monotonic()
    if args.algorithm == "lst_two_approx":
        res = lst_two_approx(p)
    elif args.algorithm == "lst_fixed_m_ptas":
        res = lst_fixed_m_ptas(p, epsilon=args.epsilon, verbose=True)
    elif args.algorithm == "mordechai":
        res = mordechai_schedule(p, epsilon=args.epsilon, verbose=True)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    elapsed = time.monotonic() - start
    print(f"[[{args.algorithm} result]] makespan: {res.makespan}, time: {elapsed:.2f} secs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((res, elapsed), f)
    output_path.with_suffix(".txt").write_text(
        f"Makespan: {res.makespan}\nTime: {elapsed:.2f} secs\n"
        f"Schedule: {res.schedule.tolist() if res.schedule is not None else 'None'}\n"
    )
    print(f"Saved results to '{args.output_dir}'.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--instance",
        type=str,
        help="instance name (without 'rcmax_' prefix and '.npy' suffix)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["lst_two_approx", "lst_fixed_m_ptas", "mordechai"],
        help="which classical algorithm to run",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="epsilon parameter for LST fixed-m PTAS (default: 0.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="directory to save results (default: 'results')",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=0,
        help="trial ID for multiple runs on the same instance and algorithm (default: 0)",
    )
    args = parser.parse_args()
    main(args)
