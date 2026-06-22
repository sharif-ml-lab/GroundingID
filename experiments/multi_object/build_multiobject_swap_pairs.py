#!/usr/bin/env python3
"""Build valid source/target pair manifests for 3-objects-per-row swap experiments.

Each manifest row picks:
- a source image
- a target image
- two symbols (rows) to swap

Validity rule:
- the four row signatures involved (src A, src B, tgt A, tgt B) must all be distinct

That makes the symbol-bound accuracy unambiguous after swapping only the objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def load_meta(path: Path):
    meta = json.loads(path.read_text())
    return list(meta.values()) if isinstance(meta, dict) else list(meta)


def shape_objects(rec: dict) -> List[dict]:
    objs = [o for o in rec.get("objects", []) if "shape" in o and "color" in o]
    def sort_key(o: dict):
        row = int(o.get("row", 10**9))
        col = int(o.get("grid_col", 10**9))
        cx = float((o.get("center_position") or [10**9, 10**9])[0])
        return (row, col, cx)
    return sorted(objs, key=sort_key)


def symbol_order(rec: dict) -> List[str]:
    syms = rec.get("symbol_arrangement")
    if syms:
        return list(syms)
    return ["@", "#", "$", "&"]


def row_map(rec: dict) -> Dict[str, List[Tuple[str, str]]]:
    syms = symbol_order(rec)

    if rec.get("row_summaries"):
        rows = {}
        for row in rec["row_summaries"]:
            items = row.get("items", [])
            ordered = sorted(items, key=lambda x: int(x.get("grid_col", 10**9)))
            rows[row["symbol"]] = [
                (str(item["color"]).lower(), str(item["shape"]).lower())
                for item in ordered
            ]
        return rows

    # Fallback from shape objects only
    objs = shape_objects(rec)
    rows_by_abs = {}
    for o in objs:
        rows_by_abs.setdefault(int(o["row"]), []).append(o)
    abs_rows = sorted(rows_by_abs)

    out: Dict[str, List[Tuple[str, str]]] = {}
    for idx, sym in enumerate(syms):
        if idx >= len(abs_rows):
            out[sym] = []
            continue
        ordered = sorted(
            rows_by_abs[abs_rows[idx]],
            key=lambda x: (int(x.get("grid_col", 10**9)), float((x.get("center_position") or [10**9])[0])),
        )
        out[sym] = [(str(o["color"]).lower(), str(o["shape"]).lower()) for o in ordered]
    return out


def row_signature(row_items: Sequence[Tuple[str, str]]) -> str:
    return "|".join(f"{color} {shape}" for color, shape in row_items)


def valid_pair(src: dict, tgt: dict, sym_a: str, sym_b: str) -> bool:
    smap = row_map(src)
    tmap = row_map(tgt)
    need = [smap.get(sym_a, []), smap.get(sym_b, []), tmap.get(sym_a, []), tmap.get(sym_b, [])]
    if any(len(items) != 3 for items in need):
        return False
    sigs = [row_signature(items) for items in need]
    return len(set(sigs)) == 4


def symbol_to_physical_index(symbols: Sequence[str], sym: str) -> int:
    return list(symbols).index(sym) + 1


def manifest_row(src: dict, tgt: dict, sym_a: str, sym_b: str) -> dict:
    syms = symbol_order(src)
    src_rows = row_map(src)
    tgt_rows = row_map(tgt)
    a_idx = symbol_to_physical_index(syms, sym_a)
    b_idx = symbol_to_physical_index(syms, sym_b)
    return {
        "source": src["filename"],
        "target": tgt["filename"],
        "sym_a": sym_a,
        "sym_b": sym_b,
        "row_pairs": f"{a_idx}->{b_idx},{b_idx}->{a_idx}",
        "src_a": row_signature(src_rows[sym_a]),
        "src_b": row_signature(src_rows[sym_b]),
        "tgt_a": row_signature(tgt_rows[sym_a]),
        "tgt_b": row_signature(tgt_rows[sym_b]),
    }


def choose_pairs(
    meta: Sequence[dict],
    max_rows: int,
    seed: int,
    require_distinct_targets: bool = False,
) -> List[dict]:
    rng = random.Random(seed)
    records = [r for r in meta if "filename" in r]
    sources = records[:]
    rng.shuffle(sources)
    symbol_pairs = [(a, b) for i, a in enumerate(symbol_order(records[0])) for b in symbol_order(records[0])[i + 1 :]]
    rng.shuffle(symbol_pairs)

    used_target_counts: Dict[str, int] = {}
    out: List[dict] = []

    for src in sources:
        local_pairs = symbol_pairs[:]
        rng.shuffle(local_pairs)
        chosen = None

        for sym_a, sym_b in local_pairs:
            candidates = [tgt for tgt in records if tgt is not src and valid_pair(src, tgt, sym_a, sym_b)]
            if not candidates:
                continue
            rng.shuffle(candidates)
            candidates.sort(key=lambda r: used_target_counts.get(r["filename"], 0))
            if require_distinct_targets:
                candidates = [c for c in candidates if used_target_counts.get(c["filename"], 0) == 0] or candidates
            tgt = candidates[0]
            chosen = manifest_row(src, tgt, sym_a, sym_b)
            used_target_counts[tgt["filename"]] = used_target_counts.get(tgt["filename"], 0) + 1
            break

        if chosen is None:
            continue
        out.append(chosen)
        if len(out) >= max_rows:
            break

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--require_distinct_targets", action="store_true")
    args = ap.parse_args()

    meta = load_meta(Path(args.metadata))
    rows = choose_pairs(
        meta,
        max_rows=args.max_rows,
        seed=args.seed,
        require_distinct_targets=args.require_distinct_targets,
    )
    if not rows:
        raise SystemExit("No valid source/target pairs found.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {out_path} (n={len(rows)})")


if __name__ == "__main__":
    main()
