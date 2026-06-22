#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from glob import glob
from typing import List, Tuple, Dict, Any, Optional
import itertools
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('dark')
# sns.set(rc={'figure.facecolor':'cornflowerblue'})
sns.set_context('notebook', font_scale=1.0, rc={'lines.linewidth': 1.5})
sns.set_palette('bright')

# ----------------------- Helpers -----------------------

def parse_layer_spec(spec: str, max_layer: int = 128):
    """
    Parse layer spec like:
      "all" -> [1..max_layer]
      "1-28" -> [1,2,...,28]
      "10-20,24,27-28" -> [10..20,24,27,28]
    Returns 1-based layer indices as ints.
    """
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(1, max_layer + 1))
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                layers.add(i)
        elif part:
            layers.add(int(part))
    out = sorted([x for x in layers if 1 <= x <= max_layer])
    if not out:
        raise ValueError("Layer spec parsed to empty set.")
    return out


def list_json_files(folder: str, recursive: bool = True):
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(glob(os.path.join(folder, pattern), recursive=recursive))


def token_order_key(token_key: str):
    """
    Keys look like '0001:The' — sort by the numeric prefix.
    """
    try:
        return int(token_key.split(":", 1)[0])
    except Exception:
        return 10**9  # send weird keys to the end


# Simple in-memory cache to avoid re-parsing the same file for the same layer set
_attention_cache: Dict[Tuple[str, Tuple[int, ...]], np.ndarray] = {}

def load_attention_values(
    json_file: str,
    layer_ids: List[int],
    layer_prefix: str = "layer",
):
    """
    For a single JSON file:
      - For each token:
          * stack all patch vectors of selected layers (NaN if missing)
          * take nanmax over patches
          * take nanmean over layers
      - Return a 1D array of per-token attentions
    Returns None if file is malformed or has no usable data.
    """
    cache_key = (json_file, tuple(layer_ids))
    if cache_key in _attention_cache:
        return _attention_cache[cache_key]

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {json_file}: {e}")
        return None

    per_token = data.get("per_token")
    if not isinstance(per_token, dict) or not per_token:
        return None

    token_keys = sorted(per_token.keys(), key=token_order_key)
    out_vals: List[float] = []

    for tk in token_keys:
        token_data = per_token.get(tk, {})
        if not isinstance(token_data, dict) or not token_data:
            out_vals.append(np.nan)
            continue

        # Collect [num_patches x num_layers]
        patch_rows = []
        for _, patch_dict in token_data.items():
            if not isinstance(patch_dict, dict):
                continue
            row = []
            for l in layer_ids:
                val = patch_dict.get(f"{layer_prefix}{l}", np.nan)
                try:
                    val = float(val)
                except Exception:
                    val = np.nan
                row.append(val)
            patch_rows.append(row)

        if not patch_rows:
            out_vals.append(np.nan)
            continue

        arr = np.array(patch_rows, dtype=float)  # [P, L]
        with np.errstate(all="ignore"):
            max_per_layer = np.nanmax(arr, axis=0)  # [L]
            mean_attention = np.nanmean(max_per_layer)  # scalar
        out_vals.append(mean_attention)

    arr_out = np.array(out_vals, dtype=float)
    if np.all(np.isnan(arr_out)):
        return None

    _attention_cache[cache_key] = arr_out
    return arr_out


def window_aggregate_with_stride(
    values: np.ndarray,
    window_size: int,
    stride: int,
    op: str = "mean"
):
    """
    Slide a window of length `window_size` across `values` with step `stride`.
    Aggregate with 'mean' or 'max' (NaN-aware). Returns (agg_values, positions).
    positions = center indices of each window (float indices).
    """
    assert window_size > 0 and stride > 0
    n = len(values)
    if n == 0 or n < window_size:
        return np.array([]), np.array([])

    agg_vals = []
    centers = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        win = values[start:end]
        if op == "max":
            agg = np.nanmax(win)
        else:
            agg = np.nanmean(win)
        agg_vals.append(agg)
        centers.append(start + (window_size - 1) / 2.0)

    return np.array(agg_vals, dtype=float), np.array(centers, dtype=float)


def process_folder(
    folder_path: str,
    layer_ids: List[int],
    window_size: int,
    stride: int,
    window_op: str,
    recursive: bool = True,
):
    """
    Process all JSON files in folder:
      - per file: per-token attention -> window aggregate
      - align by window index (pad with NaN) -> mean across files
    Returns:
      mean_curve, mean_positions, stats, ci_low, ci_high
    """
    files = list_json_files(folder_path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No JSON files found in '{folder_path}'")

    curves = []
    centers_list = []
    used_files = 0
    skipped_files = 0

    for fp in files:
        vals = load_attention_values(fp, layer_ids)
        if vals is None or len(vals) == 0 or np.all(np.isnan(vals)):
            skipped_files += 1
            continue

        agg, centers = window_aggregate_with_stride(vals, window_size, stride, window_op)
        if agg.size == 0:
            skipped_files += 1
            continue

        curves.append(agg)
        centers_list.append(centers)
        used_files += 1

    if used_files == 0:
        raise RuntimeError(f"All files in '{folder_path}' were empty or invalid.")

    max_len = max(len(c) for c in curves)
    padded = np.array([np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in curves])  # [F, W]
    mean_curve = np.nanmean(padded, axis=0)

    # 95% CI of the mean: mean ± 1.96 * (std / sqrt(n)) with NaN-aware stats
    n = np.sum(np.isfinite(padded), axis=0).astype(float)
    with np.errstate(all="ignore"):
        std = np.nanstd(padded, axis=0, ddof=1)
        se = std / np.sqrt(n)
        hw = 1.96 * se
        ci_low = mean_curve - hw
        ci_high = mean_curve + hw
        ci_low[n < 2] = np.nan
        ci_high[n < 2] = np.nan

    padded_centers = np.array([np.pad(cn, (0, max_len - len(cn)), constant_values=np.nan) for cn in centers_list])
    mean_positions = np.nanmean(padded_centers, axis=0)

    stats = {
        "total_files": len(files),
        "used_files": used_files,
        "skipped_files": skipped_files,
        "max_windows": max_len,
    }
    return mean_curve, mean_positions, stats, ci_low, ci_high


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(path: str,
             positions: np.ndarray,
             curve1: np.ndarray, curve2: np.ndarray,
             label1: str, label2: str,
             ci1_low: Optional[np.ndarray] = None, ci1_high: Optional[np.ndarray] = None,
             ci2_low: Optional[np.ndarray] = None, ci2_high: Optional[np.ndarray] = None):
    cols = ["window_index", "approx_token_pos", label1, label2]
    if ci1_low is not None and ci1_high is not None:
        cols.extend([f"{label1}_ci_low", f"{label1}_ci_high"])
    if ci2_low is not None and ci2_high is not None:
        cols.extend([f"{label2}_ci_low", f"{label2}_ci_high"])
    header = ",".join(cols) + "\n"

    idx = np.arange(len(curve1))
    data_cols = [idx, positions, curve1, curve2]
    if ci1_low is not None and ci1_high is not None:
        data_cols.extend([ci1_low, ci1_high])
    if ci2_low is not None and ci2_high is not None:
        data_cols.extend([ci2_low, ci2_high])

    data = np.column_stack(data_cols)
    np.savetxt(path, data, delimiter=",", header=header.strip(), comments="", fmt="%.6f")

# ---------- Pretty plotting (NaN-aware smoothing, CI bands, minor ticks, SVG/PDF options) ----------

def _nan_moving_average(x: np.ndarray, w: int):
    """NaN-aware centered moving average with 'same' length."""
    if w is None or w <= 1:
        return x
    w = int(w)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x).astype(float)
    vals = np.where(np.isfinite(x), x, 0.0)
    kernel = np.ones(w, dtype=float)
    num = np.convolve(vals, kernel, mode="same")
    den = np.convolve(mask, kernel, mode="same")
    out = num / np.where(den == 0.0, np.nan, den)
    return out


def plot_and_save(
    positions1: np.ndarray,
    curve1: np.ndarray,
    positions2: np.ndarray,
    curve2: np.ndarray,
    label1: str,
    label2: str,
    window_size: int,
    stride: int,
    window_op: str,
    out_png: str,
    use_positions: bool = True,
    show: bool = False,
    smooth_window: int = 0,      # optional smoothing (0 = off)
    save_svg: bool = True,       # save SVG alongside PNG/PDF
    save_pdf: bool = True,       # save PDF (vector)
    ci1_low: Optional[np.ndarray] = None,
    ci1_high: Optional[np.ndarray] = None,
    ci2_low: Optional[np.ndarray] = None,
    ci2_high: Optional[np.ndarray] = None,
):
    # Square figure + high-DPI, auto layout
    fig, ax = plt.subplots(figsize=(9.6, 8), constrained_layout=True)
    # try:
    #     ax.set_box_aspect(1.0)  # square plotting box
    # except Exception:
    #     ax.set_aspect('equal', adjustable='box')

    # X axes
    if use_positions and len(positions1) == len(curve1) and len(positions2) == len(curve2):
        x1, x2 = positions1, positions2
        ax.set_xlabel("Approximate token index (window centers)", fontweight="bold", fontsize=15)
    else:
        x1 = np.arange(len(curve1))
        x2 = np.arange(len(curve2))
        ax.set_xlabel("Generated Token Index", fontweight="bold", fontsize=15)

    # Optional smoothing (NaN-aware)
    y1 = _nan_moving_average(curve1, smooth_window)
    y2 = _nan_moving_average(curve2, smooth_window)

    # Smooth CI bounds too (visual consistency)
    if ci1_low is not None and ci1_high is not None:
        ci1l = _nan_moving_average(ci1_low, smooth_window)
        ci1h = _nan_moving_average(ci1_high, smooth_window)
    else:
        ci1l = ci1h = None
    if ci2_low is not None and ci2_high is not None:
        ci2l = _nan_moving_average(ci2_low, smooth_window)
        ci2h = _nan_moving_average(ci2_high, smooth_window)
    else:
        ci2l = ci2h = None

    # Lines + CI fills (default style; no explicit colors)
    ax.plot(x1, y1, label=label1, linewidth=2.25, alpha=0.95)
    if ci1l is not None and ci1h is not None:
        ax.fill_between(x1, ci1l, ci1h, alpha=0.18)

    ax.plot(x2, y2, label=label2, linewidth=2.25, alpha=0.95)
    if ci2l is not None and ci2h is not None:
        ax.fill_between(x2, ci2l, ci2h, alpha=0.18)

    # Axes cosmetics
    ax.set_ylabel("Attention Score",  fontweight="bold", fontsize=15)  # exact label
    ax.set_title('Text-to-Image Attention Score (MS-COCO)', fontsize=20, fontweight='bold')
    plt.rcParams['axes.titlepad'] = 15


    # No title (per request)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.2)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Auto top limit, then enforce fixed bounds
    y_all_parts = [arr[np.isfinite(arr)] for arr in [y1, y2]]
    if ci1l is not None and ci1h is not None:
        y_all_parts += [ci1l[np.isfinite(ci1l)], ci1h[np.isfinite(ci1h)]]
    if ci2l is not None and ci2h is not None:
        y_all_parts += [ci2l[np.isfinite(ci2l)], ci2h[np.isfinite(ci2h)]]
    y_all = np.concatenate([p for p in y_all_parts if p.size]) if y_all_parts else np.array([])

    if y_all.size >= 2:
        ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
        if np.isfinite(ymin) and np.isfinite(ymax):
            pad = (ymax - ymin) * 0.08 if ymax > ymin else 0.05
            ax.set_ylim(ymin - pad, ymax + pad)

    # Hard limits requested:
    # - Y from 0.01 to 0.2
    ax.set_ylim(bottom=-0.001, top=0.42)
    # - X from 0 to 30 (cap view to first 30 tokens/windows)
    ax.set_xlim(left=0, right=270)

    # Legend
    leg = ax.legend(frameon=False, handlelength=2.5, fontsize=20)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_alpha(0.95)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(12)
    # Save files
    root, _ = os.path.splitext(out_png)
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    print(f"✅ PNG saved: {out_png}")
    if save_svg:
        fig.savefig(root + ".svg", bbox_inches="tight")
        print(f"✅ SVG saved: {root + '.svg'}")
    if save_pdf:
        fig.savefig(root + ".pdf", bbox_inches="tight")
        print(f"✅ PDF saved: {root + '.pdf'}")

    if show:
        plt.show()
    plt.close(fig)


def slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text.strip())
    if len(safe) > 80:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        safe = safe[:60] + "__" + h
    return safe


def auc_of_curve(curve: np.ndarray, positions: Optional[np.ndarray] = None) -> float:
    vals = curve.astype(float)
    if positions is None or positions.size != vals.size or np.all(np.isnan(positions)):
        x = np.arange(len(vals), dtype=float)
    else:
        x = positions.astype(float)
    mask = np.isfinite(vals) & np.isfinite(x)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapz(vals[mask], x[mask]))


# ----------------------- Sweep -----------------------

def parse_semicolon_list(s: Optional[str]):
    if not s:
        return []
    s = s.replace("\n", ";")
    items = [part.strip() for part in s.split(";") if part.strip()]
    return items


def parse_comma_ints(s: Optional[str]):
    if not s:
        return []
    out = []
    for part in s.replace(" ", "").split(","):
        if part:
            out.append(int(part))
    return out


def parse_comma_strs(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [p.strip().lower() for p in s.split(",") if p.strip()]


def run_single_config(
    method1: str,
    method2: str,
    label1: str,
    label2: str,
    layer_spec: str,
    window: int,
    stride: int,
    window_op: str,
    out_dir: str,
    use_positions_x: bool,
    recursive: bool,
    show: bool,
    save_csv_flag: bool,
    smooth_window: int,
    save_svg: bool,
) -> Dict[str, Any]:
    """Run one configuration; save PNG/CSV; return summary dict."""
    layer_ids = parse_layer_spec(layer_spec, max_layer=1024)

    curve1, pos1, stats1, ci1_low, ci1_high = process_folder(
        method1, layer_ids, window, stride, window_op, recursive=recursive
    )
    curve2, pos2, stats2, ci2_low, ci2_high = process_folder(
        method2, layer_ids, window, stride, window_op, recursive=recursive
    )

    # Align to same length for plotting/CSV
    max_len = max(len(curve1), len(curve2))
    pad = lambda arr: np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

    curve1_p = pad(curve1)
    curve2_p = pad(curve2)
    pos1_p = pad(pos1) if pos1.size else np.arange(max_len, dtype=float)
    pos2_p = pad(pos2) if pos2.size else np.arange(max_len, dtype=float)

    ci1_low_p = pad(ci1_low) if ci1_low is not None else None
    ci1_high_p = pad(ci1_high) if ci1_high is not None else None
    ci2_low_p = pad(ci2_low) if ci2_low is not None else None
    ci2_high_p = pad(ci2_high) if ci2_high is not None else None

    # Filename bits
    layers_slug = slugify(layer_spec)
    base_name = f"{slugify(label1)}_vs_{slugify(label2)}__layers-{layers_slug}__w{window}_s{stride}_{window_op}"

    ensure_dir(out_dir)
    out_png = os.path.join(out_dir, base_name + ".png")
    out_pdf = os.path.join(out_dir, base_name + ".pdf")
    out_csv = os.path.join(out_dir, base_name + ".csv")
    out_json = os.path.join(out_dir, base_name + ".json")

    # Plot and save
    plot_and_save(
        pos1_p, curve1_p, pos2_p, curve2_p,
        label1, label2,
        window, stride, window_op,
        out_png=out_png,
        use_positions=use_positions_x,
        show=show,
        smooth_window=smooth_window,
        save_svg=save_svg,
        save_pdf=True,
        ci1_low=ci1_low_p, ci1_high=ci1_high_p,
        ci2_low=ci2_low_p, ci2_high=ci2_high_p,
    )

    # CSV
    if save_csv_flag:
        with np.errstate(invalid="ignore"):
            mean_positions = np.nanmean(np.vstack([pos1_p, pos2_p]), axis=0)
        save_csv(out_csv, mean_positions, curve1_p, curve2_p, label1, label2,
                 ci1_low=ci1_low_p, ci1_high=ci1_high_p,
                 ci2_low=ci2_low_p, ci2_high=ci2_high_p)
        print(f"📄 CSV saved: {out_csv}")

    # Metrics
    auc1 = auc_of_curve(curve1_p, pos1_p if use_positions_x else None)
    auc2 = auc_of_curve(curve2_p, pos2_p if use_positions_x else None)
    mean_ci_hw1 = float(np.nanmean((ci1_high_p - ci1_low_p) / 2.0)) if ci1_low_p is not None else float("nan")
    mean_ci_hw2 = float(np.nanmean((ci2_high_p - ci2_low_p) / 2.0)) if ci2_low_p is not None else float("nan")
    summary = {
        "label1": label1,
        "label2": label2,
        "method1": method1,
        "method2": method2,
        "layer_spec": layer_spec,
        "window": window,
        "stride": stride,
        "window_op": window_op,
        "positions_x": bool(use_positions_x),
        "recursive": bool(recursive),
        "png": out_png,
        "pdf": out_pdf,
        "csv": out_csv if save_csv_flag else None,
        "stats1": stats1,
        "stats2": stats2,
        "len_curve": int(max_len),
        "mean1": float(np.nanmean(curve1_p)),
        "mean2": float(np.nanmean(curve2_p)),
        "max1": float(np.nanmax(curve1_p)),
        "max2": float(np.nanmax(curve2_p)),
        "auc1": auc1,
        "auc2": auc2,
        "delta_mean": float(np.nanmean(curve1_p) - np.nanmean(curve2_p)),
        "delta_auc": float(auc1 - auc2) if np.isfinite(auc1) and np.isfinite(auc2) else float("nan"),
        "smooth_window": int(smooth_window),
        "save_svg": bool(save_svg),
        "mean_ci_halfwidth1": mean_ci_hw1,
        "mean_ci_halfwidth2": mean_ci_hw2,
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"🧾 Metadata saved: {out_json}")

    return summary


# ----------------------- Main CLI -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare attention curves between two methods; supports hyperparameter sweeps."
    )
    # Paths + labels
    parser.add_argument("--method1", required=True, help="Folder with JSONs for Method 1")
    parser.add_argument("--method2", required=True, help="Folder with JSONs for Method 2")
    parser.add_argument("--label1", default="Structured", help="Label for Method 1 curve")
    parser.add_argument("--label2", default="Baseline", help="Label for Method 2 curve")

    # Single-run params
    parser.add_argument("--layers", default="20-28",
                        help='Layer spec, e.g. "10-20", "all", or "10-20,24,27-28" (1-based).')
    parser.add_argument("--window", type=int, default=10, help="Sliding window size (tokens)")
    parser.add_argument("--stride", type=int, default=1, help="Stride between windows (tokens)")
    parser.add_argument("--window-op", choices=["mean", "max"], default="max",
                        help="Aggregation inside each window.")

    # Sweep params (lists)
    parser.add_argument("--layers-list", default=None,
                        help='Semicolon-separated list of layer specs for sweep, e.g. "10-20;all;1-28,30".')
    parser.add_argument("--windows", default=None,
                        help="Comma-separated window sizes for sweep, e.g. 5,10,15")
    parser.add_argument("--strides", default=None,
                        help="Comma-separated stride values for sweep, e.g. 1,2,5")
    parser.add_argument("--window-ops", default=None,
                        help='Comma-separated ops for sweep from {"mean","max"}, e.g. mean,max')

    # Output
    parser.add_argument("--out-dir", default="results/Row", help="Base directory to save outputs")
    parser.add_argument("--sweep-name", default="sweep", help="Subfolder under out-dir for sweep outputs")
    parser.add_argument("--save-csv", action="store_true", help="Save a CSV for each run and a master summary CSV")
    parser.add_argument("--positions-x", action="store_true",
                        help="Plot X-axis as approximate token centers; else use window index.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders for JSONs")
    parser.add_argument("--show", action="store_true", help="Also display the plot window")

    # Pretty plotting options
    parser.add_argument("--smooth", type=int, default=0,
                        help="NaN-aware moving average window (0=off). Try 5–11 for gentle smoothing.")
    parser.add_argument("--no-svg", action="store_true",
                        help="Do not save an SVG alongside the PNG/PDF.")

    args = parser.parse_args()

    layers_list = parse_semicolon_list(getattr(args, "layers_list"))
    windows_list = parse_comma_ints(args.windows)
    strides_list = parse_comma_ints(args.strides)
    ops_list = parse_comma_strs(args.window_ops)

    sweep_layers = layers_list if layers_list else [args.layers]
    sweep_windows = windows_list if windows_list else [args.window]
    sweep_strides = strides_list if strides_list else [args.stride]
    sweep_ops = ops_list if ops_list else [args.window_op]

    is_sweep = any([
        len(sweep_layers) > 1,
        len(sweep_windows) > 1,
        len(sweep_strides) > 1,
        len(sweep_ops) > 1
    ])

    out_root = os.path.join(args.out_dir, args.sweep_name) if is_sweep else args.out_dir
    ensure_dir(out_root)

    summaries: List[Dict[str, Any]] = []

    for (layer_spec, window, stride, op) in itertools.product(sweep_layers, sweep_windows, sweep_strides, sweep_ops):
        cfg_dir = os.path.join(
            out_root,
            f"layers-{slugify(layer_spec)}__w{window}_s{stride}_{op}"
        )
        ensure_dir(cfg_dir)
        print(f"\n→ Running config: layers={layer_spec} | window={window} | stride={stride} | op={op}")
        summary = run_single_config(
            method1=args.method1,
            method2=args.method2,
            label1=args.label1,
            label2=args.label2,
            layer_spec=layer_spec,
            window=window,
            stride=stride,
            window_op=op,
            out_dir=cfg_dir,
            use_positions_x=args.positions_x,
            recursive=args.recursive,
            show=args.show,
            save_csv_flag=args.save_csv,
            smooth_window=args.smooth,
            save_svg=not args.no_svg,
        )
        summaries.append(summary)

    if args.save_csv and summaries:
        summary_csv = os.path.join(out_root, "sweep_summary.csv")
        cols = [
            "layer_spec", "window", "stride", "window_op",
            "mean1", "mean2", "max1", "max2", "auc1", "auc2", "delta_mean", "delta_auc",
            "stats1.used_files", "stats1.total_files", "stats2.used_files", "stats2.total_files",
            "smooth_window", "save_svg",
            "mean_ci_halfwidth1", "mean_ci_halfwidth2",
            "png", "pdf", "csv"
        ]
        with open(summary_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for s in summaries:
                row = [
                    s["layer_spec"], str(s["window"]), str(s["stride"]), s["window_op"],
                    f"{s['mean1']:.6f}", f"{s['mean2']:.6f}",
                    f"{s['max1']:.6f}", f"{s['max2']:.6f}",
                    f"{s['auc1']:.6f}" if np.isfinite(s["auc1"]) else "nan",
                    f"{s['auc2']:.6f}" if np.isfinite(s["auc2"]) else "nan",
                    f"{s['delta_mean']:.6f}" if np.isfinite(s["delta_mean"]) else "nan",
                    f"{s['delta_auc']:.6f}" if np.isfinite(s["delta_auc"]) else "nan",
                    str(s["stats1"].get("used_files", "")), str(s["stats1"].get("total_files", "")),
                    str(s["stats2"].get("used_files", "")), str(s["stats2"].get("total_files", "")),
                    str(s.get("smooth_window", "")), str(s.get("save_svg", "")),
                    f"{s.get('mean_ci_halfwidth1', float('nan')):.6f}" if isinstance(s.get("mean_ci_halfwidth1"), (int, float)) else "nan",
                    f"{s.get('mean_ci_halfwidth2', float('nan')):.6f}" if isinstance(s.get("mean_ci_halfwidth2"), (int, float)) else "nan",
                    s.get("png", ""), s.get("pdf", ""), s.get("csv", "") or "",
                ]
                f.write(",".join(row) + "\n")
        print(f"\n📊 Sweep summary CSV: {summary_csv}")

        summary_json = os.path.join(out_root, "sweep_summary.json")
        with open(summary_json, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"🧾 Sweep summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
