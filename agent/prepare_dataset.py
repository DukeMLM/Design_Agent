# prepare_dataset.py
"""
Split the ADM dataset into the training and validation files the agent system
reads from ./dataset:

    g_training.csv, s_training.csv, g_validation.csv, s_validation.csv

ADM.zip contains data_g.csv / data_s.csv (geometry / spectrum, no header row).
Extract it into ./dataset (either ./dataset/data_g.csv or ./dataset/ADM/data_g.csv),
then run:

    python prepare_dataset.py
"""

import argparse
import os

import numpy as np
import pandas as pd


def find_source(dataset_dir: str) -> str:
    """Folder that holds data_g.csv and data_s.csv."""
    for folder in (dataset_dir, os.path.join(dataset_dir, "ADM")):
        if all(os.path.exists(os.path.join(folder, name)) for name in ("data_g.csv", "data_s.csv")):
            return folder
    raise FileNotFoundError(
        f"data_g.csv and data_s.csv were not found in {dataset_dir} or {dataset_dir}/ADM. "
        "Download ADM.zip and extract it there first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="fraction of the samples used for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    source = find_source(args.dataset_dir)
    print(f"Reading {source}/data_g.csv and data_s.csv ...")
    g = pd.read_csv(os.path.join(source, "data_g.csv"), header=None, dtype="float32").values
    s = pd.read_csv(os.path.join(source, "data_s.csv"), header=None, dtype="float32").values
    if g.shape[0] != s.shape[0]:
        raise ValueError(f"data_g.csv has {g.shape[0]} rows but data_s.csv has {s.shape[0]}")

    n = g.shape[0]
    order = np.random.default_rng(args.seed).permutation(n)
    n_val = int(round(n * args.val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]

    def write(name: str, array: np.ndarray) -> None:
        path = os.path.join(args.dataset_dir, name)
        pd.DataFrame(array).to_csv(path, header=False, index=False)
        print(f"  {path}: {array.shape[0]} rows x {array.shape[1]} columns")

    write("g_training.csv", g[train_idx])
    write("s_training.csv", s[train_idx])
    write("g_validation.csv", g[val_idx])
    write("s_validation.csv", s[val_idx])
    print(f"Done: {len(train_idx)} training and {len(val_idx)} validation samples.")


if __name__ == "__main__":
    main()
