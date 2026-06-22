#!/usr/bin/env python3
"""Plot pair-swap heatmaps from saved `results.csv`.

This script does not rerun model inference. It reads the existing per-query
scores written by `eval_multiobject_pair_swap_binding.py` and regenerates:

- full four-panel A-vs-B heatmap (standard/swapped x color/shape)
- filtered four-panel heatmap where:
  - color panels use only pairs correct in standard color
  - shape panels use only pairs correct in standard shape

It also writes shape-only and color-only filtered two-panel figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot pair-swap heatmaps from saved results.csv without rerunning inference.")
    ap.add_argument("--results_csv", required=True, help="Path to saved results.csv from eval_multiobject_pair_swap_binding.py")
    ap.add_argument("--out_dir", default="", help="Output directory. Default: results_csv parent")
    ap.add_argument("--prefix", default="", help="Optional filename prefix, e.g. fix5_")
    return ap.parse_args()


def load_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pair_key(row: dict) -> Tuple[str, str, str, str, str]:
    if row.get("pair_idx", "") != "":
        return ("pair_idx", row["pair_idx"], "", "", "")
    return (
        row.get("source", ""),
        row.get("sym_a", ""),
        row.get("sym_b", ""),
        str(row.get("grid_col_a", "")),
        str(row.get("grid_col_b", "")),
    )


def compute_missing_flags(rows: Sequence[dict]) -> None:
    by_pair: Dict[Tuple[str, str, str, str, str], List[dict]] = {}
    for row in rows:
        by_pair.setdefault(pair_key(row), []).append(row)

    for key, pair_rows in by_pair.items():
        std_rows = [r for r in pair_rows if str(r.get("condition", "")) == "standard"]
        if not std_rows:
            continue

        def ab_correct(task: str, row: dict) -> int:
            if f"{task}_bound_correct" in row and row[f"{task}_bound_correct"] != "":
                return int(float(row[f"{task}_bound_correct"]))
            a = float(row[f"{task}_lp_A_ab"])
            b = float(row[f"{task}_lp_B_ab"])
            pred = "A" if a >= b else "B"
            return int(pred == str(row["bound_label"]))

        pair_std_ab = int(all(int(float(r.get("bound_correct", "0"))) == 1 for r in std_rows))
        pair_std_shape = int(all(ab_correct("shape", r) == 1 for r in std_rows))
        pair_std_color = int(all(ab_correct("color", r) == 1 for r in std_rows))

        for row in pair_rows:
            row["standard_pair_bound_correct"] = row.get("standard_pair_bound_correct", str(pair_std_ab))
            row["standard_pair_shape_bound_correct"] = row.get("standard_pair_shape_bound_correct", str(pair_std_shape))
            row["standard_pair_color_bound_correct"] = row.get("standard_pair_color_bound_correct", str(pair_std_color))
            if str(row.get("condition", "")) == "swapped":
                row["eligible_standard_ab"] = row.get("eligible_standard_ab", str(pair_std_ab))
                row["eligible_standard_shape_ab"] = row.get("eligible_standard_shape_ab", str(pair_std_shape))
                row["eligible_standard_color_ab"] = row.get("eligible_standard_color_ab", str(pair_std_color))


def mean_task_matrix(rows: Sequence[dict], task: str, ab_only: bool) -> np.ndarray | None:
    if not rows:
        return None
    col_a = f"{task}_lp_A_ab" if ab_only else f"{task}_lp_A"
    col_b = f"{task}_lp_B_ab" if ab_only else f"{task}_lp_B"
    mats = []
    for r in rows:
        q = str(r["query_symbol"])
        sym_a = str(r["sym_a"])
        sym_b = str(r["sym_b"])
        if str(r.get("condition", "")) == "swapped":
            left_val = float(r[col_b])
            right_val = float(r[col_a])
        else:
            left_val = float(r[col_a])
            right_val = float(r[col_b])
        if q == sym_a:
            mats.append(np.array([[left_val, right_val], [np.nan, np.nan]], dtype=float))
        elif q == sym_b:
            mats.append(np.array([[np.nan, np.nan], [left_val, right_val]], dtype=float))
    if not mats:
        return None
    return np.nanmean(np.stack(mats, axis=0), axis=0)


def draw_four_panel(color_std, color_swap, shape_std, shape_swap, out_prefix: Path) -> None:
    mats = [color_std, color_swap, shape_std, shape_swap]
    if any(m is None for m in mats):
        return
    all_vals = np.concatenate([m.ravel() for m in mats]).astype(float)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    panels = [
        (axs[0, 0], color_std, "Standard Setting", "Color"),
        (axs[0, 1], color_swap, "Activation Swapped", "Color"),
        (axs[1, 0], shape_std, "", "Shape"),
        (axs[1, 1], shape_swap, "", "Shape"),
    ]

    image = None
    for ax, mat, title, ylabel in panels:
        image = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 1], [r"$o_{s_0}$", r"$o_{s_1}$"])
        ax.set_yticks([0, 1], [r"$s_0$", r"$s_1$"])
        ax.set_title(title, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(image, ax=axs, shrink=0.88)
    cbar.set_label("mean log-prob")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_two_panel(std_mat: np.ndarray | None, swp_mat: np.ndarray | None, out_prefix: Path, task_label: str) -> None:
    if std_mat is None or swp_mat is None:
        return
    all_vals = np.concatenate([std_mat.ravel(), swp_mat.ravel()]).astype(float)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    panels = [
        (axs[0], std_mat, "Standard Setting"),
        (axs[1], swp_mat, "Activation Swapped"),
    ]
    image = None
    for ax, mat, title in panels:
        image = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 1], [r"$o_{s_0}$", r"$o_{s_1}$"])
        ax.set_yticks([0, 1], [r"$s_0$", r"$s_1$"])
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(task_label, fontsize=14)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(image, ax=axs, shrink=0.92)
    cbar.set_label("mean log-prob")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv)
    out_dir = Path(args.out_dir) if args.out_dir else results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    rows = load_rows(results_csv)
    if not rows:
        raise SystemExit(f"No rows found in {results_csv}")
    compute_missing_flags(rows)

    std_rows = [r for r in rows if str(r.get("condition", "")) == "standard"]
    swp_rows = [r for r in rows if str(r.get("condition", "")) == "swapped"]
    std_shape_ok = [r for r in std_rows if int(float(r.get("standard_pair_shape_bound_correct", "0"))) == 1]
    std_color_ok = [r for r in std_rows if int(float(r.get("standard_pair_color_bound_correct", "0"))) == 1]
    swp_std_ok = [r for r in swp_rows if int(float(r.get("eligible_standard_ab", "0"))) == 1]
    swp_std_ok_shape = [r for r in swp_rows if int(float(r.get("eligible_standard_shape_ab", "0"))) == 1]
    swp_std_ok_color = [r for r in swp_rows if int(float(r.get("eligible_standard_color_ab", "0"))) == 1]

    draw_four_panel(
        mean_task_matrix(std_rows, "color", ab_only=True),
        mean_task_matrix(swp_rows, "color", ab_only=True),
        mean_task_matrix(std_rows, "shape", ab_only=True),
        mean_task_matrix(swp_rows, "shape", ab_only=True),
        out_dir / f"{prefix}avg_logprob_heatmap_a_vs_b",
    )
    draw_four_panel(
        mean_task_matrix(std_color_ok, "color", ab_only=True),
        mean_task_matrix(swp_std_ok_color, "color", ab_only=True),
        mean_task_matrix(std_shape_ok, "shape", ab_only=True),
        mean_task_matrix(swp_std_ok_shape, "shape", ab_only=True),
        out_dir / f"{prefix}avg_logprob_heatmap_a_vs_b_on_standard_correct",
    )
    draw_two_panel(
        mean_task_matrix(std_shape_ok, "shape", ab_only=True),
        mean_task_matrix(swp_std_ok_shape, "shape", ab_only=True),
        out_dir / f"{prefix}shape_heatmap_a_vs_b_on_standard_shape_correct",
        "Shape",
    )
    draw_two_panel(
        mean_task_matrix(std_color_ok, "color", ab_only=True),
        mean_task_matrix(swp_std_ok_color, "color", ab_only=True),
        out_dir / f"{prefix}color_heatmap_a_vs_b_on_standard_color_correct",
        "Color",
    )

    print(f"[write] {out_dir / f'{prefix}avg_logprob_heatmap_a_vs_b.pdf'}")
    print(f"[write] {out_dir / f'{prefix}avg_logprob_heatmap_a_vs_b_on_standard_correct.pdf'}")


if __name__ == "__main__":
    main()
