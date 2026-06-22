#!/usr/bin/env python3
"""Create structure-only blank hosts from a multi-object source dataset.

The output images keep the exact row lines and left-side symbols from the
source images, but every shape object is erased to white. Metadata records keep
the structural fields and expose `slot_positions`, which are used as target
patch locations for later activation-patching experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw


def load_meta(path: Path):
    meta = json.loads(path.read_text())
    return list(meta.values()) if isinstance(meta, dict) else list(meta)


def save_meta(path: Path, rows: List[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def symbol_map(rec: dict) -> Dict[int, str]:
    rows = list(rec.get("target_rows") or [])
    syms = list(rec.get("symbol_arrangement") or rec.get("row_symbols") or ["@", "#", "$", "&"])
    return {int(r): syms[i] for i, r in enumerate(rows[: len(syms)])}


def bbox_from_obj(obj: dict) -> Tuple[float, float, float, float]:
    if "paste_position" in obj and "size" in obj:
        x, y = obj["paste_position"]
        s = float(obj["size"])
        return float(x), float(y), float(x) + s, float(y) + s
    if "center_position" in obj and "size" in obj:
        cx, cy = obj["center_position"]
        s = float(obj["size"])
        half = s / 2.0
        return float(cx) - half, float(cy) - half, float(cx) + half, float(cy) + half
    if "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) >= 4:
        x0, y0, a, b = map(float, obj["bbox"][:4])
        if a > x0 and b > y0:
            return x0, y0, a, b
        return x0, y0, x0 + a, y0 + b
    raise ValueError(f"Could not resolve bbox for object: {obj}")


def shape_objects(rec: dict) -> List[dict]:
    objs = [o for o in rec.get("objects", []) if "shape" in o and "color" in o]
    return sorted(
        objs,
        key=lambda o: (
            int(o.get("row_index_0based", 10**9)),
            int(o.get("col_index_0based", 10**9)),
            int(o.get("row", 10**9)),
            int(o.get("grid_col", 10**9)),
        ),
    )


def slot_positions(rec: dict) -> List[dict]:
    row_to_sym = symbol_map(rec)
    slots: List[dict] = []
    for obj in shape_objects(rec):
        row_val = int(obj["row"])
        grid_col = int(obj.get("grid_col", -1))
        row_idx = int(obj.get("row_index_0based", 10**9))
        col_idx = int(obj.get("col_index_0based", 10**9))
        sym = row_to_sym.get(row_val, "")
        slots.append(
            {
                "slot_id": f"{sym}:{grid_col}",
                "row": row_val,
                "row_symbol": sym,
                "row_index_0based": row_idx,
                "grid_col": grid_col,
                "col_index_0based": col_idx,
                "center_position": list(obj.get("center_position", [])),
                "paste_position": list(obj.get("paste_position", [])),
                "size": obj.get("size"),
                "patch_index": obj.get("patch_index"),
            }
        )
    return slots


def blank_row_summaries(rec: dict) -> List[dict]:
    if rec.get("row_summaries"):
        return [
            {
                "symbol": row["symbol"],
                "row": row["row"],
                "row_index_0based": row["row_index_0based"],
                "items": [],
            }
            for row in rec["row_summaries"]
        ]

    out: List[dict] = []
    row_to_sym = symbol_map(rec)
    for idx, row_val in enumerate(rec.get("target_rows") or []):
        out.append(
            {
                "symbol": row_to_sym.get(int(row_val), ""),
                "row": int(row_val),
                "row_index_0based": idx,
                "items": [],
            }
        )
    return out


def blank_record(rec: dict, out_name: str) -> dict:
    keep_keys = [
        "canvas_size_x",
        "canvas_size_y",
        "grid_size_x",
        "grid_size_y",
        "patch_size",
        "separator_lines_y",
        "target_rows",
        "symbol_arrangement",
        "row_symbols",
        "object_cols",
        "image_id",
        "description",
    ]
    out = {k: rec[k] for k in keep_keys if k in rec}
    out["filename"] = out_name
    out["source_filename"] = rec["filename"]
    out["num_objects"] = 0
    out["objects"] = []
    out["row_summaries"] = blank_row_summaries(rec)
    out["slot_positions"] = slot_positions(rec)
    return out


def erase_objects(image: Image.Image, rec: dict, erase_pad: int) -> Image.Image:
    im = image.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for obj in shape_objects(rec):
        x0, y0, x1, y1 = bbox_from_obj(obj)
        x0 = max(0, int(x0) - erase_pad)
        y0 = max(0, int(y0) - erase_pad)
        x1 = min(w - 1, int(x1) + erase_pad)
        y1 = min(h - 1, int(y1) + erase_pad)
        draw.rectangle([x0, y0, x1, y1], fill="white")
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--src_metadata", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--erase_pad", type=int, default=2)
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_meta(Path(args.src_metadata))
    out_rows: List[dict] = []

    for idx, rec in enumerate(rows):
        src_name = rec["filename"]
        src_path = src_dir / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        with Image.open(src_path) as im:
            blank = erase_objects(im.convert("RGB"), rec, erase_pad=args.erase_pad)
        blank.save(out_dir / src_name)
        out_rows.append(blank_record(rec, src_name))

        if idx == 0:
            blank.save(out_dir / "preview_blank.png")

    save_meta(out_dir / "all_samples_metadata.json", out_rows)
    (out_dir / "dataset_config.json").write_text(
        json.dumps(
            {
                "source_dir": str(src_dir),
                "source_metadata": str(Path(args.src_metadata)),
                "num_images": len(out_rows),
                "erase_pad": args.erase_pad,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[write] {out_dir / 'all_samples_metadata.json'} (n={len(out_rows)})")
    print(f"[write] {out_dir / 'preview_blank.png'}")


if __name__ == "__main__":
    main()
