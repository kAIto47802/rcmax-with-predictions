from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle

import numpy as np

from experiments._plot import plot_results


def main(args: Namespace) -> None:
    labels = {
        **{k: rf"Ours ($K={k}$)" for k in args.search_budgets},
        "lst_two_approx": "LST 2-approximation",
        "lst_fixed_m_ptas": r"LST $(1 + \varepsilon)$-approximation",
        "mordechai": "Mordechai + LST outer",
    }

    runtimes = {
        labels[k]: np.array(
            [
                [
                    _load_pg_runtime(
                        _pg_result_path(
                            args,
                            _instance_name(args, x),
                            k,
                            seed,
                        )
                    )
                    for x in args.x_values
                ]
                for seed in range(42, 42 + args.num_seeds)
            ]
        )
        for k in args.search_budgets
    }
    xs = {labels[k]: np.array(args.x_values) for k in args.search_budgets}

    classical_specs = [
        ("lst_two_approx", None),
        ("lst_fixed_m_ptas", args.ptas_epsilon),
        ("mordechai", None),
    ]
    for algorithm, epsilon in classical_specs:
        name = labels[algorithm]
        x_values, values = _load_classical_runtimes(args, algorithm, epsilon)
        if values.size:
            runtimes[name] = values
            xs[name] = np.array(x_values)

    if not runtimes:
        raise ValueError("No runtime results were loaded.")

    plot_names = list(runtimes)
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    markers = dict(zip(plot_names, marker_cycle))
    markersizes = {name: size for name, size in zip(plot_names, [7.0, 6.0, 7.0, 6.0, 7.0, 7.0])}
    colors = dict(
        zip(
            plot_names,
            ["#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00", "#56B4E9"],
        )
    )
    zorders = {name: len(plot_names) - i for i, name in enumerate(plot_names)}

    fig = plot_results(
        [runtimes],
        "Runtime / sec",
        xlabels=[rf"${args.x_axis}$"],
        xs=[xs],
        share_yaxis=True,
        legend_loc="upper left",
        legend_fontsize=15,
        xlabel_fontsize=16,
        ylabel_fontsize=16,
        figsize=tuple(args.figsize),
        logy=args.logy,
        markers=markers,
        markersizes=markersizes,
        colors=colors,
        zorders=zorders,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        Path(args.output_dir)
        / (
            f"rcmax_{args.family}_runtime_by_{args.x_axis}_"
            f"{'n' if args.x_axis == 'm' else 'm'}{args.n if args.x_axis == 'm' else args.m}_"
            f"seed{args.data_seed}_eps{args.epsilon}_pdf"
        )
        .replace(".", "")
        .replace("_pdf", ".pdf"),
        bbox_inches="tight",
    )


def _instance_name(args: Namespace, x: int) -> str:
    m = x if args.x_axis == "m" else args.m
    n = args.n if args.x_axis == "m" else x
    return f"{args.family}_{m}x{n}_seed{args.data_seed}"


def _pg_result_path(args: Namespace, instance: str, k: int, seed: int) -> Path:
    return Path(args.results_dir) / (
        f"rcmax_{instance}_pg_results_"
        f"fnr{args.false_negative_rate}_fpr{args.false_positive_rate}_"
        f"eps{args.epsilon}_K{k}_seed{seed}_pkl"
    ).replace(".", "").replace("_pkl", ".pkl")


def _classical_result_path(
    args: Namespace,
    instance: str,
    algorithm: str,
    epsilon: float | None,
    trial_id: int,
) -> Path:
    return Path(args.results_dir) / (
        f"rcmax_{instance}_{algorithm}"
        + (f"_eps{epsilon}" if algorithm == "lst_fixed_m_ptas" else "")
        + f"_trial{trial_id}_pkl"
    ).replace(".", "").replace("_pkl", ".pkl")


def _load_pg_runtime(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "rb") as f:
        _, _, elapsed = pickle.load(f)
    return elapsed


def _load_classical_runtimes(
    args: Namespace,
    algorithm: str,
    epsilon: float | None,
) -> tuple[list[int], np.ndarray]:
    x_values: list[int] = []
    values: list[list[float]] = []
    for x in args.x_values:
        if (
            algorithm in {"lst_fixed_m_ptas", "mordechai"}
            and args.heavy_x_max is not None
            and x > args.heavy_x_max
        ):
            continue
        paths = [
            _classical_result_path(
                args,
                _instance_name(args, x),
                algorithm,
                epsilon,
                trial_id,
            )
            for trial_id in range(args.num_trials)
        ]
        existing_paths = [path for path in paths if path.exists()]
        if not existing_paths:
            print(f"Skipping missing classical results: {paths[0]} ...")
            continue
        if len(existing_paths) != len(paths):
            missing_paths = [path for path in paths if not path.exists()]
            raise FileNotFoundError(
                "Incomplete classical runtime trials. Missing files:\n"
                + "\n".join(str(path) for path in missing_paths)
            )
        x_values.append(x)
        values.append([_load_classical_runtime(path) for path in paths])
    return x_values, np.array(values).T


def _load_classical_runtime(path: Path) -> float:
    with open(path, "rb") as f:
        _, elapsed = pickle.load(f)
    return elapsed


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--family",
        type=str,
        default="U_1_100",
        help="instance family name",
    )
    parser.add_argument(
        "--x-axis",
        choices=["m", "n"],
        default="m",
        help="which parameter to use for the x-axis",
    )
    parser.add_argument(
        "--x-values",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="values of m or n used on the x-axis",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=10,
        help="fixed m value when --x-axis n",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="fixed n value when --x-axis m",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="seed used when generating instances",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="epsilon value used for prediction-guided experiments",
    )
    parser.add_argument(
        "--ptas-epsilon",
        type=float,
        default=0.01,
        help="epsilon value used for lst_fixed_m_ptas",
    )
    parser.add_argument(
        "--false-negative-rate",
        type=float,
        default=0.1,
        help="false negative rate used for prediction-guided experiments",
    )
    parser.add_argument(
        "--false-positive-rate",
        type=float,
        default=0.1,
        help="false positive rate used for prediction-guided experiments",
    )
    parser.add_argument(
        "--search-budgets",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="list of search budgets K used in the experiments",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=30,
        help="number of random seeds for averaging prediction-guided runtimes",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=30,
        help="number of classical runtime trials to average",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="use a logarithmic scale for runtime",
    )
    parser.add_argument(
        "--heavy-x-max",
        type=int,
        help="only plot lst_fixed_m_ptas and mordechai for x-axis values at most this value",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[4.5, 4],
        metavar=("WIDTH", "HEIGHT"),
        help="figure size in inches (default: 4.5 4)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="directory containing result .pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="directory to save the output plot",
    )
    main(parser.parse_args())
