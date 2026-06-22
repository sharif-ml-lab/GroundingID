#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def _unify_margin_col(df):
    if "logit_margin" in df.columns:
        return df
    if "margin" in df.columns:
        df = df.copy()
        df["logit_margin"] = df["margin"]
        return df
    raise ValueError("Input CSV must contain 'logit_margin' or 'margin'.")

def _clean_types(df):
    df = df.copy()
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce").astype("Int64")
    df["logit_margin"] = pd.to_numeric(df["logit_margin"], errors="coerce")
    if "query_symbol" in df.columns:
        df["query_symbol"] = df["query_symbol"].astype(str)
    return df.dropna(subset=["layer", "logit_margin"])

def _aggregate_overall(df, max_layer):
    d = (df[df["layer"] <= max_layer]
           .groupby("layer", as_index=False)["logit_margin"]
           .agg(mean="mean", std="std", count="size")
           .sort_values("layer"))
    return d

def _aggregate_by_symbol(df, max_layer):
    if "query_symbol" not in df.columns:
        d = _aggregate_overall(df, max_layer)
        d.insert(0, "method", "All")
        return d
    d = (df[df["layer"] <= max_layer]
           .groupby(["query_symbol", "layer"], as_index=False)["logit_margin"]
           .agg(mean="mean", std="std", count="size")
           .rename(columns={"query_symbol": "method"})
           .sort_values(["method", "layer"]))
    return d

def _plot_stats(df_stats, title, ylabel, outfile, max_layer):
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
        y = d["mean"].to_numpy()
        s = d["std"].to_numpy()
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

def main():
    ap = argparse.ArgumentParser(description="Plot logit margins across layers for ALL rows and CORRECT-only.")
    ap.add_argument("--csv", required=True, help="Path to combined probe_logitlens.csv")
    ap.add_argument("--max_layer", type=int, default=28)
    ap.add_argument("--out_dir", required=True, help="Directory to place PDFs and filtered CSVs")
    ap.add_argument("--ylabel", default="Logit margin (target - other)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = _unify_margin_col(df)
    df = _clean_types(df)

    # --- ALL rows ---
    df_all = df.copy()
    df_all.to_csv(os.path.join(args.out_dir, "probe_logitlens.ALL.csv"), index=False)
    print(f"[write] {os.path.join(args.out_dir, 'probe_logitlens.ALL.csv')} rows={len(df_all)}")

    overall_all = _aggregate_overall(df_all, args.max_layer)
    bysym_all   = _aggregate_by_symbol(df_all, args.max_layer)

    _plot_stats(overall_all,
                "Logit Margin Across Layers (ALL rows)",
                args.ylabel,
                os.path.join(args.out_dir, "logitmargin_overall.ALL.pdf"),
                args.max_layer)
    _plot_stats(bysym_all,
                "Logit Margin Across Layers (by Symbol, ALL rows)",
                args.ylabel,
                os.path.join(args.out_dir, "logitmargin_bysymbol.ALL.pdf"),
                args.max_layer)

    # --- CORRECT-only (if available) ---
    if "is_correct" in df.columns:
        df_corr = df[df["is_correct"] == 1].copy()
        df_corr.to_csv(os.path.join(args.out_dir, "probe_logitlens.CORRECT.csv"), index=False)
        print(f"[write] {os.path.join(args.out_dir, 'probe_logitlens.CORRECT.csv')} rows={len(df_corr)}")

        if not df_corr.empty:
            overall_corr = _aggregate_overall(df_corr, args.max_layer)
            bysym_corr   = _aggregate_by_symbol(df_corr, args.max_layer)

            _plot_stats(overall_corr,
                        "Logit Margin Across Layers (CORRECT only)",
                        args.ylabel,
                        os.path.join(args.out_dir, "logitmargin_overall.CORRECT.pdf"),
                        args.max_layer)
            _plot_stats(bysym_corr,
                        "Logit Margin Across Layers (by Symbol, CORRECT only)",
                        args.ylabel,
                        os.path.join(args.out_dir, "logitmargin_bysymbol.CORRECT.pdf"),
                        args.max_layer)
        else:
            print("[warn] No CORRECT rows found; skipping CORRECT-only plots.")
    else:
        print("[info] Column 'is_correct' not found; skipping CORRECT-only plots.")

if __name__ == "__main__":
    main()
