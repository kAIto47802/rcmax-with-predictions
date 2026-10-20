from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import pickle
from typing import cast, Literal

import numpy as np

from experiments._plot import plot_results
from experiments.run_prediction_guided import parse_rate_or_auto


def main(args: Namespace) -> None:
    if (
        args.false_negative_rates == "auto" or args.false_positive_rates == "auto"
    ) and not args.use_errors_for_xaxis:
        raise ValueError("Auto sweep only supports using errors for x-axis.")
    fn_res = _load_pg_results(
        args, "fn", args.false_negative_rates, args.fixed_false_positive_rate
    )
    fp_res = _load_pg_results(
        args, "fp", args.false_positive_rates, args.fixed_false_negative_rate
    )
    with open(Path(args.results_dir) / args.reference_name, "rb") as f:
        ref = pickle.load(f)

    names = [rf"Ours ($K={k}$)" for k in args.search_budgets]
    fn_makespans, fn_errors, fp_makespans, fp_errors = (
        {n: v / ref.makespan for n, v in zip(names, res)} for res in [*fn_res, *fp_res]
    )
    with open(Path(args.results_dir) / args.baseline_name, "rb") as f:
        bres, _ = pickle.load(f)
    baseline_makespan = bres.makespan

    fig = plot_results(
        [fn_makespans, fp_makespans],
        r"$C_{\max}(S) / OPT$",
        xlabels=[
            r"$R_{OPT}^-(\widehat{\phi},\sigma^\star) / OPT$",
            r"$R_{OPT}^+(\widehat{\phi},\sigma^\star) / OPT$",
        ]
        if args.use_errors_for_xaxis
        else ["False Negative Rate", "False Positive Rate"],
        xs=[fn_errors, fp_errors]
        if args.use_errors_for_xaxis
        else [args.false_negative_rates, args.false_positive_rates],  # type: ignore
        axhlines=[baseline_makespan / ref.makespan] * 2,
        axhlines_labels=["LST 2-approximation"] * 2,
        axhlines_colors=["#009E73"] * 2,
        share_yaxis=True,
        legend_loc="lower right",
        legend_fontsize=16,
        xlabel_fontsize=16,
        ylabel_fontsize=16,
        figsize=tuple(args.figsize),
        x_range=(0, 1) if args.use_errors_for_xaxis else None,
        markers=dict(zip(names, ["o", "s", "^"])),
        markersizes=dict(zip(names, [7, 6, 7])),
        colors=dict(zip(names, ["#0072B2", "#D55E00", "#CC79A7"])),
        markevery=4,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seeds = f"42-{42 + args.num_seeds}" if args.num_seeds > 1 else "42"
    fixed_rates = f"fnr{args.fixed_false_negative_rate}_fpr{args.fixed_false_positive_rate}"
    plot_type = "error" if args.use_errors_for_xaxis else "rates"
    file_name = (
        f"rcmax_{args.instance}_{fixed_rates}_eps{args.epsilon}_seed{seeds}_{plot_type}"
    ).replace(".", "")
    fig.savefig(
        (Path(args.output_dir) / file_name).with_suffix(".pdf"),
        bbox_inches="tight",
    )


def _load_pg_results(
    args: Namespace,
    sweep_type: Literal["fn", "fp"],
    sweep_rates: list[float] | Literal["auto"],
    fixed_rate: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if sweep_rates == "auto":
        fixed_type = "fp" if sweep_type == "fn" else "fn"
        seeds = f"42-{42 + args.num_seeds}" if args.num_seeds > 1 else "42"
        output_path = Path(args.results_dir) / (
            f"rcmax_{args.instance}_{fixed_type}r{fixed_rate}_"
            f"eps{args.epsilon}_seed{seeds}_auto_{sweep_type}_ranges_txt"
        ).replace(".", "").replace("_txt", ".txt")
        sweep_rates = json.loads(output_path.read_text())

    data_paths = [
        [
            [
                Path(args.results_dir)
                / (
                    f"rcmax_{args.instance}_pg_results_"
                    f"fnr{s if sweep_type == 'fn' else fixed_rate}_"
                    f"fpr{s if sweep_type == 'fp' else fixed_rate}_"
                    f"eps{args.epsilon}_K{k}_seed{seed}_pkl"
                )
                .replace(".", "")
                .replace("_pkl", ".pkl")
                for seed in range(42, 42 + args.num_seeds)
            ]
            for s in sweep_rates
        ]
        for k in args.search_budgets
    ]
    data = [
        [[_load_pg_pkl_result(path, sweep_type) for path in paths] for paths in budget_paths]
        for budget_paths in data_paths
    ]
    makespans = [np.array([[r for r, _ in res] for res in results]).T for results in data]
    errors = [np.array([[e for _, e in res] for res in results]).mean(axis=1) for results in data]
    return makespans, errors


def _load_pg_pkl_result(path: Path, error_type: Literal["fn", "fp"]) -> tuple[float, float]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "rb") as f:
        res, errors, _ = pickle.load(f)
    return res.makespan, errors.R_minus if error_type == "fn" else errors.R_plus


def _normalize_rates_or_auto(
    values: list[float | Literal["auto"]],
    name: str,
    parser: ArgumentParser,
) -> list[float] | Literal["auto"]:
    if values == ["auto"]:
        return "auto"
    if "auto" in values:
        parser.error(f"{name} must be either 'auto' or a list of floats")
    return cast(list[float], values)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reference-name", type=str, help="filename of the reference result .pkl")
    parser.add_argument("--baseline-name", type=str, help="filename of the baseline result .pkl")
    parser.add_argument(
        "--instance", type=str, help="instance name (without 'rcmax_' prefix and '.npy' suffix)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="epsilon value used in the experiments"
    )
    parser.add_argument(
        "--search-budgets",
        nargs="+",
        type=int,
        default=[0, 1],
        help="list of search budgets K used in the experiments",
    )
    parser.add_argument(
        "--false-negative-rates",
        nargs="+",
        type=parse_rate_or_auto,
        help="list of false negative rates to analyze",
    )
    parser.add_argument(
        "--false-positive-rates",
        nargs="+",
        type=parse_rate_or_auto,
        help="list of false positive rates to analyze",
    )
    parser.add_argument(
        "--fixed-false-negative-rate",
        type=float,
        help="fixed false negative rate when analyzing false positive rates",
    )
    parser.add_argument(
        "--fixed-false-positive-rate",
        type=float,
        help="fixed false positive rate when analyzing false negative rates",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="number of random seeds for averaging results",
    )
    parser.add_argument(
        "--use-errors-for-xaxis",
        action="store_true",
        help="whether to use error rates instead of sweep rates for the x-axis",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="directory containing the result .pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="directory to save the output plot",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[9, 4],
        metavar=("WIDTH", "HEIGHT"),
        help="figure size in inches (default: 9 4)",
    )
    args = parser.parse_args()
    args.false_negative_rates = _normalize_rates_or_auto(
        args.false_negative_rates, "false-negative-rates", parser
    )
    args.false_positive_rates = _normalize_rates_or_auto(
        args.false_positive_rates, "false-positive-rates", parser
    )
    main(args)
