from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

import numpy as np


_Family = Literal[
    "U_1_100",
    "U_10_100",
    "U_100_200",
    "U_100_120",
    "U_1000_1100",
    "job_correlated",
    "machine_correlated",
]


def _generate_instance(
    m: int,
    n: int,
    family: _Family,
    *,
    seed: int | None = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if family == "U_1_100":
        return rng.integers(1, 101, size=(m, n))
    elif family == "U_10_100":
        return rng.integers(10, 101, size=(m, n))
    elif family == "U_100_200":
        return rng.integers(100, 201, size=(m, n))
    elif family == "U_100_120":
        return rng.integers(100, 121, size=(m, n))
    elif family == "U_1000_1100":
        return rng.integers(1000, 1101, size=(m, n))
    elif family == "job_correlated":
        b = rng.integers(1, 101, size=(n,))
        d = rng.integers(1, 21, size=(m, n))
        return d + b[None, :]
    elif family == "machine_correlated":
        a = rng.integers(1, 101, size=(m,))
        c = rng.integers(1, 21, size=(m, n))
        return a[:, None] + c
    else:
        raise ValueError(f"Unknown family: {family}")


def load_instance(name: str) -> np.ndarray:
    p = np.load(Path("data") / f"rcmax_{name}.npy")
    print(f"Loaded instance '{name}' with shape {p.shape}.")
    return p


def main(args: Namespace) -> None:
    output_path = (
        Path(args.output_dir) / f"rcmax_{args.family}_{args.m}x{args.n}_seed{args.seed}.npy"
    )
    if output_path.exists():
        print(f"Instance already exists at '{output_path}'. Skipping generation.")
        return
    p = _generate_instance(args.m, args.n, args.family, seed=args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, p)
    print(f"Saved instance to '{output_path}'.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--family",
        type=str,
        choices=[
            "U_1_100",
            "U_10_100",
            "U_100_200",
            "U_100_120",
            "U_1000_1100",
            "job_correlated",
            "machine_correlated",
        ],
        help="instance family",
    )
    parser.add_argument(
        "--m",
        type=int,
        help="number of machines",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of jobs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="directory to save instances (default: 'data')",
    )
    args = parser.parse_args()
    main(args)
