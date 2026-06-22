#!/usr/bin/env python3
import json, random, argparse
from pathlib import Path

SYMS = ["@", "$", "#", "&"]
PAIR_CHOICES = [
    ("@", "$"), ("@", "#"), ("@", "&"),
    ("$", "#"), ("$", "&"),
    ("#", "&"),
]

def load_meta(meta_path: Path):
    m = json.loads(meta_path.read_text())
    return list(m.values()) if isinstance(m, dict) else m

def y_of(obj):
    if isinstance(obj, dict):
        if "row" in obj:
            try:
                return float(obj["row"])
            except Exception:
                pass
        if "center_position" in obj and isinstance(obj["center_position"], (list, tuple)) and len(obj["center_position"]) > 1:
            return float(obj["center_position"][1])
    return 0.0

def row_shapes_for_record(rec):
    """
    Return a 4-tuple list [(shape,color), ...] ordered top->bottom rows 1..4,
    robust to metadata variants. We look for objects having both 'shape' and 'color',
    then sort them by 'row' (if present) else by center_position y.
    If fewer than 4 found, return None.
    """
    objs = [o for o in rec.get("objects", []) if ("shape" in o and "color" in o)]
    if not objs:
        return None
    # Prefer explicit 'row' when present
    if any("row" in o for o in objs):
        try:
            objs = sorted(objs, key=lambda o: int(o.get("row", 0)))
        except Exception:
            objs = sorted(objs, key=y_of)
    else:
        objs = sorted(objs, key=y_of)

    # Take the first 4 rows (dataset is 4 physical rows)
    objs = objs[:4]
    if len(objs) < 4:
        return None
    return [(o["shape"].lower(), o["color"].lower()) for o in objs]

def build_index(meta):
    """
    Returns:
      items: list of (filename, [(shape,color)*4])
      by_fn: dict filename -> [(shape,color)*4]
    """
    items = []
    by_fn = {}
    for rec in meta:
        fn = rec.get("filename")
        rs = row_shapes_for_record(rec)
        if fn and rs:
            items.append((fn, rs))
            by_fn[fn] = rs
    return items, by_fn

def symbols_to_rows(sym_a, sym_b):
    # Map symbols to row indices 0..3 (top..bottom). The dataset uses the order @, $, #, &.
    s2i = {"@": 0, "$": 1, "#": 2, "&": 3}
    return s2i[sym_a], s2i[sym_b]

def differs_both(x, y):
    """x=(shape,color), y=(shape,color) -> True if both shape and color differ."""
    return x[0] != y[0] and x[1] != y[1]

def choose_pairs(meta_path, out_csv, max_rows=100, seed=1337, max_trials_per_src=500):
    rng = random.Random(seed)
    meta = load_meta(Path(meta_path))
    items, by_fn = build_index(meta)

    # Shuffle sources for variety
    rng.shuffle(items)

    # Round-robin through all 6 pairs to keep distribution balanced
    pair_cycle = PAIR_CHOICES[:]  # copy
    rng.shuffle(pair_cycle)       # start at a random pair
    pc_idx = 0

    rows = []
    used_targets = set()  # optional: avoid overusing the same target

    for (src_fn, s_rows) in items:
        # pick a pair in a round-robin manner, but we’ll allow retries/shuffles if no candidate found
        tried_pairs = set()
        cand_pair = None
        tgt_fn = None

        # Try up to all pairs in a round
        for _ in range(len(PAIR_CHOICES)):
            sym_a, sym_b = pair_cycle[pc_idx % len(pair_cycle)]
            pc_idx += 1
            if (sym_a, sym_b) in tried_pairs:
                continue
            tried_pairs.add((sym_a, sym_b))

            ia, ib = symbols_to_rows(sym_a, sym_b)

            # If source rows are missing or invalid, skip this pair
            if ia >= len(s_rows) or ib >= len(s_rows):
                continue

            # Now find a target that satisfies cross-swap mismatch constraints
            trials = 0
            rng.shuffle(items)
            for (cand_fn, t_rows) in items:
                if cand_fn == src_fn:
                    continue
                if ia >= len(t_rows) or ib >= len(t_rows):
                    continue
                # cross-swap constraints:
                # tgt[sym_a] must differ from src[sym_b] in both shape & color
                # tgt[sym_b] must differ from src[sym_a] in both shape & color
                ok1 = differs_both(t_rows[ia], s_rows[ib])
                ok2 = differs_both(t_rows[ib], s_rows[ia])
                if ok1 and ok2:
                    # lightly avoid repeating the same target too much
                    if cand_fn in used_targets and rng.random() < 0.5:
                        # Prefer a fresher target half the time
                        trials += 1
                        if trials > max_trials_per_src:
                            break
                        continue
                    cand_pair = (sym_a, sym_b)
                    tgt_fn = cand_fn
                    break
                trials += 1
                if trials > max_trials_per_src:
                    break

            if cand_pair is not None and tgt_fn is not None:
                used_targets.add(tgt_fn)
                break

        if cand_pair is None or tgt_fn is None:
            # No feasible pair/target found for this source; skip
            continue

        rows.append((src_fn, tgt_fn, cand_pair[0], cand_pair[1]))
        if len(rows) >= max_rows:
            break

    # Write CSV
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "source,target,sym_a,sym_b\n" + "\n".join(f"{a},{b},{c},{d}" for a,b,c,d in rows)
    )
    print(f"[write] {out}  (n={len(rows)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    choose_pairs(args.metadata, args.out_csv, max_rows=args.max_rows, seed=args.seed)
