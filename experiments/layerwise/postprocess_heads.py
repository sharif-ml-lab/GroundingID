#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np
import csv

def load_unweighted_mean(runs_dir: Path):
    j = runs_dir / "head_diffs_mean.json"
    if j.exists():
        M = np.array(json.loads(j.read_text()), dtype=np.float64)
        return M
    n = runs_dir / "head_diffs_all.npy"
    if not n.exists():
        raise SystemExit("Missing both head_diffs_mean.json and head_diffs_all.npy")
    HD = np.load(n)  # [N,L,H]
    return HD.mean(axis=0)

def load_weighted_mean(runs_dir: Path):
    j = runs_dir / "head_diffs_mean_weighted.json"
    if j.exists():
        return np.array(json.loads(j.read_text()), dtype=np.float64)

    # recompute if json missing
    n_hd  = runs_dir / "head_diffs_all.npy"         # [N,L,H]
    n_hvm = runs_dir / "head_vis_masses_all.npy"    # [N,L,H]
    n_lvm = runs_dir / "layer_vis_masses_all.npy"   # [N,L]
    if not (n_hd.exists() and n_hvm.exists() and n_lvm.exists()):
        raise SystemExit("Missing weighted stats AND the raw arrays to recompute them.")
    HD  = np.load(n_hd)
    HVM = np.load(n_hvm)
    LVM = np.load(n_lvm)
    # align shapes
    Lmin = min(HD.shape[1], HVM.shape[1], LVM.shape[1])
    Hmin = min(HD.shape[2], HVM.shape[2])
    HD  = HD[:, :Lmin, :Hmin]
    HVM = HVM[:, :Lmin, :Hmin]
    LVM = LVM[:, :Lmin]
    W = HD * HVM * LVM[..., None]   # [N,L,H]
    return W.mean(axis=0)

def flatten_indices(mat):
    L, H = mat.shape
    idx = []
    for l in range(L):
        for h in range(H):
            idx.append((l, h, float(mat[l, h])))
    return idx

def threshold_sets(mat):
    mu = float(mat.mean())
    sd = float(mat.std(ddof=0))
    a = mat

    above_zero_pos = np.argwhere(a > +sd)
    below_zero_neg = np.argwhere(a < -sd)

    above_mean_pos = np.argwhere(a > mu + sd)
    below_mean_neg = np.argwhere(a < mu - sd)

    return {
        "mean": mu,
        "std": sd,
        "L": mat.shape[0],
        "H": mat.shape[1],
        "above_zero_plus_sd": [(int(l), int(h), float(a[l,h])) for l,h in above_zero_pos],
        "below_zero_minus_sd": [(int(l), int(h), float(a[l,h])) for l,h in below_zero_neg],
        "above_mean_plus_sd": [(int(l), int(h), float(a[l,h])) for l,h in above_mean_pos],
        "below_mean_minus_sd": [(int(l), int(h), float(a[l,h])) for l,h in below_mean_neg],
    }

def write_csv(path, label, items):
    # items: list of (layer, head, value)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "test", "layer", "head", "value"])
        for test_name, lst in items.items():
            if not isinstance(lst, list):
                continue
            for l,h,v in lst:
                w.writerow([label, test_name, l, h, v])

def topk_by_abs(mat, k=25):
    L, H = mat.shape
    flat = [(l, h, float(mat[l, h]), abs(float(mat[l, h]))) for l in range(L) for h in range(H)]
    flat.sort(key=lambda t: t[3], reverse=True)
    return [(l, h, v) for (l, h, v, _) in flat[:k]]

def main():
    ap = argparse.ArgumentParser(description="Post-process per-head layer×head scores: thresholds vs 0±std and mean±std (unweighted & weighted).")
    ap.add_argument("--runs_dir", required=True, help="Same --out_dir you used in the main run")
    ap.add_argument("--topk", type=int, default=25, help="Show top-|value| cells per matrix")
    args = ap.parse_args()

    root = Path(args.runs_dir)
    if not root.exists():
        raise SystemExit(f"runs_dir not found: {root}")

    # ------- UNWEIGHTED -------
    M_unw = load_unweighted_mean(root)   # [L,H]
    stats_unw = threshold_sets(M_unw)

    # ------- WEIGHTED -------
    M_w = load_weighted_mean(root)       # [L,H]
    stats_w = threshold_sets(M_w)

    # Save JSON reports
    (root / "stats_unweighted.json").write_text(json.dumps(stats_unw, indent=2), encoding="utf-8")
    (root / "stats_weighted.json").write_text(json.dumps(stats_w, indent=2), encoding="utf-8")

    # Save CSV with all flagged cells
    csv_map_unw = {
        "above_zero_plus_sd": stats_unw["above_zero_plus_sd"],
        "below_zero_minus_sd": stats_unw["below_zero_minus_sd"],
        "above_mean_plus_sd": stats_unw["above_mean_plus_sd"],
        "below_mean_minus_sd": stats_unw["below_mean_minus_sd"],
    }
    write_csv(root / "significant_heads_unweighted.csv", "unweighted", csv_map_unw)

    csv_map_w = {
        "above_zero_plus_sd": stats_w["above_zero_plus_sd"],
        "below_zero_minus_sd": stats_w["below_zero_minus_sd"],
        "above_mean_plus_sd": stats_w["above_mean_plus_sd"],
        "below_mean_minus_sd": stats_w["below_mean_minus_sd"],
    }
    write_csv(root / "significant_heads_weighted.csv", "weighted", csv_map_w)

    # Top-k by absolute value (quick print + JSON)
    top_unw = topk_by_abs(M_unw, args.topk)
    top_w   = topk_by_abs(M_w,   args.topk)
    (root / "topk_unweighted.json").write_text(json.dumps(top_unw, indent=2), encoding="utf-8")
    (root / "topk_weighted.json").write_text(json.dumps(top_w,   indent=2), encoding="utf-8")

    # Console summary
    print("\n=== Unweighted ===")
    print(f"mean={stats_unw['mean']:.6f}  std={stats_unw['std']:.6f}  shape=({stats_unw['L']},{stats_unw['H']})")
    print(f"> 0+std: {len(stats_unw['above_zero_plus_sd'])} | < 0-std: {len(stats_unw['below_zero_minus_sd'])}")
    print(f"> mean+std: {len(stats_unw['above_mean_plus_sd'])} | < mean-std: {len(stats_unw['below_mean_minus_sd'])}")
    print(f"Top-{args.topk} (|value|):", top_unw[:5], "...")

    print("\n=== Weighted ===")
    print(f"mean={stats_w['mean']:.6f}  std={stats_w['std']:.6f}  shape=({stats_w['L']},{stats_w['H']})")
    print(f"> 0+std: {len(stats_w['above_zero_plus_sd'])} | < 0-std: {len(stats_w['below_zero_minus_sd'])}")
    print(f"> mean+std: {len(stats_w['above_mean_plus_sd'])} | < mean-std: {len(stats_w['below_mean_minus_sd'])}")
    print(f"Top-{args.topk} (|value|):", top_w[:5], "...")

    print(f"\n[ok] wrote:")
    print(f"  {root/'stats_unweighted.json'}")
    print(f"  {root/'stats_weighted.json'}")
    print(f"  {root/'significant_heads_unweighted.csv'}")
    print(f"  {root/'significant_heads_weighted.csv'}")
    print(f"  {root/'topk_unweighted.json'}")
    print(f"  {root/'topk_weighted.json'}")

if __name__ == "__main__":
    main()
