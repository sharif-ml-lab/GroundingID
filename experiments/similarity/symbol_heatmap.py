#!/usr/bin/env python3
"""Plot the 6x6 relational similarity matrix for symbol and Grounding ID pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA = np.array(
    [
        [0.49, 0.25, 0.26, -0.22, -0.25, -0.02],
        [0.29, 0.48, 0.27, 0.22, -0.03, -0.25],
        [0.27, 0.26, 0.47, 0.01, 0.18, 0.16],
        [-0.25, 0.22, -0.01, 0.48, 0.25, -0.23],
        [-0.30, -0.03, 0.19, 0.27, 0.49, 0.20],
        [-0.03, -0.27, 0.21, -0.25, 0.24, 0.47],
    ]
)
LABELS = ["@-$", "@-#", "@-&", "$-#", "$-&", "#-&"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/grounding_alignment"),
        help="Output path without an extension; PNG and PDF are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(DATA, index=LABELS, columns=LABELS)
    plt.figure(figsize=(10, 8))
    sns.set_context("talk")
    axis = sns.heatmap(
        frame,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        annot_kws={"size": 20},
    )
    for index, annotation in enumerate(axis.texts):
        if index // len(LABELS) == index % len(LABELS):
            annotation.set_weight("bold")

    plt.title(r"$|\mathrm{Cosine\ Similarity}|$", fontsize=28, y=1.05)
    plt.xticks(rotation=45, ha="right", fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    plt.tight_layout()
    plt.savefig(args.output.with_suffix(".png"), dpi=300)
    plt.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
