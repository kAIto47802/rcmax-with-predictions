from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from concurrent.futures import as_completed, ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import time
from typing import Literal

import numpy as np

from experiments.prepare_data import load_instance
from rcmax import (
    calc_error_measures,
    ErrorMeasures,
    heavy_projection,
    prediction_localized_search,
    ReferenceScheduleResult,
    ScheduleResult,
)


def main(args: Namespace) -> None:
    p = load_instance(args.instance)
    reference_path = Path(args.reference_dir) / f"rcmax_{args.instance}_reference.pkl"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference result for instance '{args.instance}' not found at '{reference_path}'. "
            "Please run 'run_reference.py' first to generate the reference result."
        )
    with open(reference_path, "rb") as f:
        reference = pickle.load(f)
    if reference.schedule is None:
        raise ValueError(
            f"Reference result for instance '{args.instance}' does not contain a schedule."
        )

    if args.false_negative_rate == args.false_positive_rate == "auto":
        raise ValueError(
            "At least one of false_negative_rate or false_positive_rate must not be 'auto'."
        )
    jobs = []
    for s, pred_error_fnr_fpr in enumerate(
        (
            _auto_fn_fp_rates
            if "auto" in [args.false_positive_rate, args.false_negative_rate]
            else predefined_fn_fp_rates
        )(args, p, reference)
    ):
        for pred, error, fnr, fpr in pred_error_fnr_fpr:
            output_path = Path(args.output_dir) / (
                f"rcmax_{args.instance}_pg_results_fnr{fnr}_fpr{fpr}_"
                f"eps{args.epsilon}_K{args.K}_seed{42 + s}_pkl"
            ).replace(".", "").replace("_pkl", ".pkl")
            if output_path.exists():
                print(f"Prediction-guided result already exists at '{output_path}', skipping...")
                continue
            if args.num_workers > 1:
                jobs.append((pred, error, output_path))
                continue
            res, elapsed = _solve_one(p, pred, args.epsilon, args.K, verbose=True)
            _save_result(output_path, res, error, elapsed)

    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.num_workers, mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_solve_one, p, pred, args.epsilon, args.K): (error, output_path)
                for pred, error, output_path in jobs
            }
            for completed, future in enumerate(as_completed(futures), 1):
                error, output_path = futures[future]
                res, elapsed = future.result()
                _save_result(output_path, res, error, elapsed)
                print(f"[{completed}/{len(futures)}] {output_path.name}", flush=True)


def _solve_one(
    p: np.ndarray,
    pred: np.ndarray,
    epsilon: float,
    K: int,
    *,
    verbose: bool = False,
) -> tuple[ScheduleResult, float]:
    start = time.monotonic()
    res = prediction_localized_search(p, pred, epsilon, K, verbose=verbose)
    return res, time.monotonic() - start


def _save_result(
    output_path: Path,
    res: ScheduleResult,
    error: ErrorMeasures,
    elapsed: float,
) -> None:
    print(
        f"[[Prediction-guided result]] makespan: {res.makespan}, "
        f"time: {elapsed:.2f} secs, "
        f"R+: {error.R_plus:.2f}, R-: {error.R_minus:.2f}, R-_K: {error.R_minus_K:.2f}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        with open(temp_path, "wb") as f:
            pickle.dump((res, error, elapsed), f)
        temp_path.replace(output_path)
    print(f"Saved prediction-guided results to '{output_path.parent}'.")


def _generate_prediction_from_schedule(
    p: np.ndarray,  # (m, n)
    schedule: np.ndarray,  # (n,)
    epsilon: float,
    T: float,
    false_negative_rate: float,
    false_positive_rate: float,
    seed: int | None = 42,
    *,
    fp_source: Literal["all_available", "non_heavy"] = "all_available",
    fp_mode: Literal["random", "mild", "severe"] = "mild",
) -> np.ndarray:  # (n,)
    rng = np.random.default_rng(seed)
    _, n = p.shape
    threshold = epsilon * T
    heavy = heavy_projection(p, schedule, epsilon, T)

    fn_mask = (heavy_mask := heavy != -1) & (rng.random(n) < false_negative_rate)
    pred = np.where(fn_mask, -1, heavy)

    if fp_source not in {"all_available", "non_heavy"}:
        raise ValueError(f"Invalid fp_source: {fp_source}")
    fp_jobs = np.flatnonzero(~heavy_mask if fp_source == "non_heavy" else ~fn_mask)
    if fp_jobs.size == 0:
        return pred
    candidates = np.isfinite(p_fp := p[:, fp_jobs]) & (p_fp > threshold)
    candidates[schedule[fp_jobs], np.arange(fp_jobs.size)] = False
    fp_mask = candidates.any(axis=0) & (rng.random(fp_jobs.size) < false_positive_rate)
    if np.any(fp_mask):
        candidates = candidates[:, fp_mask]
        if fp_mode not in {"random", "mild", "severe"}:
            raise ValueError(f"Invalid fp_mode: {fp_mode}")
        selected = fp_jobs[fp_mask]
        chosen = (
            np.argmax(np.where(candidates, rng.random(candidates.shape), -1), axis=0)
            if fp_mode == "random"
            else np.argmin(np.where(candidates, p[:, selected], np.inf), axis=0)
            if fp_mode == "mild"
            else np.argmax(np.where(candidates, p[:, selected], -np.inf), axis=0)
        )
        pred[selected] = chosen

    return pred


def _auto_fn_fp_rates(
    args: Namespace,
    p: np.ndarray,
    reference: ReferenceScheduleResult,
) -> list[list[tuple[np.ndarray, ErrorMeasures, float, float]]]:
    assert reference.schedule is not None
    sweep_type = "fn" if args.false_negative_rate == "auto" else "fp"
    fixed_rate, fixed_type = (
        (args.false_positive_rate, "fp")
        if sweep_type == "fn"
        else (args.false_negative_rate, "fn")
    )
    assert isinstance(fixed_rate, float)
    preds_all = [
        [
            _generate_prediction_from_schedule(
                p,
                reference.schedule,
                epsilon=args.epsilon,
                T=reference.makespan,
                false_negative_rate=r if sweep_type == "fn" else fixed_rate,
                false_positive_rate=r if sweep_type == "fp" else fixed_rate,
                seed=seed,
            )
            for r in np.arange(0, 1.001, 0.001)
        ]
        for seed in range(42, 42 + args.num_seeds)
    ]
    errors_all = [
        [
            calc_error_measures(
                p,
                pred,
                reference.schedule,
                epsilon=args.epsilon,
                T=reference.makespan,
                K=args.K,
            )
            for pred in preds
        ]
        for preds in preds_all
    ]
    error_values_all = [
        np.array([e.R_minus if sweep_type == "fn" else e.R_plus for e in errors])
        for errors in errors_all
    ]
    indices_all = [
        np.unique(error_values, return_index=True)[1] for error_values in error_values_all
    ]
    indices = sorted(
        set(
            i
            for indices, error_values in zip(indices_all, error_values_all)
            for i in indices
            if error_values[i] <= reference.makespan
        )
    )
    rates = np.arange(0, 1.001, 0.001)[indices]
    output_path = Path(args.output_dir) / (
        f"rcmax_{args.instance}_{fixed_type}r{fixed_rate}_"
        f"eps{args.epsilon}_seed42-{42 + args.num_seeds}_auto_{sweep_type}_ranges_txt"
    ).replace(".", "").replace("_txt", ".txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rates.tolist()))
    print(f"Saved auto {sweep_type} rates to '{output_path}'.")
    print(f"Auto {sweep_type} rates: {rates.tolist()}")
    return [
        [
            (
                preds[i],
                errors[i],
                float(r) if sweep_type == "fn" else fixed_rate,
                float(r) if sweep_type == "fp" else fixed_rate,
            )
            for i, r in zip(indices, rates)
        ]
        for preds, errors in zip(preds_all, errors_all)
    ]


def predefined_fn_fp_rates(
    args: Namespace,
    p: np.ndarray,
    reference: ReferenceScheduleResult,
) -> list[list[tuple[np.ndarray, ErrorMeasures, float, float]]]:
    assert reference.schedule is not None
    preds_all = [
        _generate_prediction_from_schedule(
            p,
            reference.schedule,
            epsilon=args.epsilon,
            T=reference.makespan,
            false_negative_rate=args.false_negative_rate,
            false_positive_rate=args.false_positive_rate,
            seed=seed,
        )
        for seed in range(42, 42 + args.num_seeds)
    ]
    errors_all = [
        calc_error_measures(
            p,
            pred,
            reference.schedule,
            epsilon=args.epsilon,
            T=reference.makespan,
            K=args.K,
        )
        for pred in preds_all
    ]
    return [
        [(pred, error, args.false_negative_rate, args.false_positive_rate)]
        for pred, error in zip(preds_all, errors_all)
    ]


def parse_rate_or_auto(s: str) -> float | Literal["auto"]:
    if s == "auto":
        return "auto"
    try:
        value = float(s)
    except ValueError as e:
        raise ArgumentTypeError("must be a float or 'auto'") from e
    if not 0.0 <= value <= 1.0:
        raise ArgumentTypeError("rate must be in [0, 1] or 'auto'")
    return value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--instance",
        type=str,
        required=True,
        help="instance name (without 'rcmax_' prefix and '.npy' suffix)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="results",
        help="directory containing reference results (default: 'results')",
    )
    parser.add_argument(
        "--false-negative-rate",
        "-fnr",
        type=parse_rate_or_auto,
        default=0.1,
        help="false negative rate for prediction generation (default: 0.1)",
    )
    parser.add_argument(
        "--false-positive-rate",
        "-fpr",
        type=parse_rate_or_auto,
        default=0.1,
        help="false positive rate for prediction generation (default: 0.1)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="epsilon parameter for heavy job definition (default: 0.1)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="maximum number of augmentations in localized search (default: 1)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=30,
        help="number of random seeds for generating predictions (default: 5)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="number of parallel worker processes (default: 1, sequential)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="directory to save prediction-guided results (default: 'results')",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    main(args)
