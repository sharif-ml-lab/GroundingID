#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===============================
# Helpers
# ===============================
def load_probe_csv(path):
    df = pd.read_csv(path)
    # unify logit margin column name
    if "logit_margin" not in df.columns:
        if "margin" in df.columns:
            df["logit_margin"] = df["margin"]
        else:
            raise ValueError(f"{os.path.basename(path)} missing columns: no 'logit_margin' or 'margin'")
    # ensure needed columns exist
    needed = {"layer", "logit_margin"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    # types
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce").astype("Int64")
    df["logit_margin"] = pd.to_numeric(df["logit_margin"], errors="coerce")

    # filter correct-only if available
    if "is_correct" in df.columns:
        df = df[df["is_correct"] == 1]

    # drop NaNs
    df = df.dropna(subset=["layer", "logit_margin"])

    # normalize symbol col if present
    if "query_symbol" in df.columns:
        df["query_symbol"] = df["query_symbol"].astype(str)
    return df

def agg_overall(df, max_layer):
    d = (df[df["layer"] <= max_layer]
           .groupby("layer", as_index=False)["logit_margin"]
           .agg(mean="mean", std="std", count="size")
           .sort_values("layer"))
    return d

def agg_by_symbol(df, max_layer):
    if "query_symbol" not in df.columns:
        d = agg_overall(df, max_layer)
        d.insert(0, "method", "All")
        return d
    d = (df[df["layer"] <= max_layer]
           .groupby(["query_symbol", "layer"], as_index=False)["logit_margin"]
           .agg(mean="mean", std="std", count="size")
           .rename(columns={"query_symbol": "method"})
           .sort_values(["method", "layer"]))
    return d

def plot_stats(df_stats, title, ylabel, outfile, max_layer):
    plt.figure(figsize=(10,6))
    ax = plt.gca()

    if "method" in df_stats.columns:
        methods = list(df_stats["method"].unique())
        colors = plt.cm.tab10.colors
        cmap = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
        for m, d in df_stats.groupby("method"):
            x = d["layer"].astype(int).to_numpy()
            y = d["mean"].to_numpy()
            s = d["std"].to_numpy()
            ax.plot(x, y, marker="o", linewidth=3, markersize=5, label=str(m), color=cmap[m])
            ax.fill_between(x, y - s, y + s, alpha=0.22, color=cmap[m])
    else:
        x = df_stats["layer"].astype(int).to_numpy()
        y = df_stats["mean"].to_numpy()
        s = df_stats["std"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=3, markersize=5, label="Overall")
        ax.fill_between(x, y - s, y + s, alpha=0.22)

    ax.set_title(title, pad=12, fontweight='bold', fontsize=16)
    ax.set_xlabel("Layer", fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    lo = int(df_stats["layer"].min())
    hi = int(min(max_layer, df_stats["layer"].max()))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)
    ax.minorticks_on()

    leg = ax.legend(frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(14)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    print(f"[saved] {outfile}")

# ===============================
# CLI
# ===============================
def main():
    ap = argparse.ArgumentParser(description="Plot logit margin (correct-only) across layers.")
    ap.add_argument("--csv", required=True, help="Path to combined probe_logitlens.csv")
    ap.add_argument("--max_layer", type=int, default=28)
    ap.add_argument("--overall_out", default="LogitMargin_Layers_overall.correct.pdf")
    ap.add_argument("--bysym_out",   default="LogitMargin_Layers_by_symbol.correct.pdf")
    ap.add_argument("--title_overall", default="Logit Margin Across Layers (Overall, correct only)")
    ap.add_argument("--title_bysym",   default="Logit Margin Across Layers (by Symbol, correct only)")
    ap.add_argument("--ylabel", default="Logit margin (target - other)")
    ap.add_argument("--write_filtered_csv", default="", help="Optional: write filtered (correct-only) CSV here")
    args = ap.parse_args()

    df = load_probe_csv(args.csv)

    if args.write_filtered_csv:
        os.makedirs(os.path.dirname(args.write_filtered_csv), exist_ok=True)
        df.to_csv(args.write_filtered_csv, index=False)
        print(f"[write] filtered correct-only -> {args.write_filtered_csv} (rows={len(df)})")

    overall = agg_overall(df, args.max_layer)
    bysym   = agg_by_symbol(df, args.max_layer)

    plot_stats(overall, args.title_overall, args.ylabel, args.overall_out, args.max_layer)
    plot_stats(bysym,   args.title_bysym,   args.ylabel, args.bysym_out,   args.max_layer)

if __name__ == "__main__":
    main()
