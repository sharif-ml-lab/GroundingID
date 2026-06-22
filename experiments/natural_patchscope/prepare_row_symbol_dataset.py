#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a row-symbol dataset for activation patching.

Input:
  A CSV produced by find_distinct_row_coco_images.py containing:
    - image_path
    - row_categories
    - row_category_ids
    - row_ann_ids
    - row_rep_bboxes
    - row_rep_fracs

Output:
  OUTPUT_ROOT/
    meta.csv
    source/
      <rank>_<image_id>_rowsym.png

The output meta.csv contains one row per image with:
  - source_image_path
  - patch_lists (one list per row object, ordered row1..row4)
  - row_categories
  - row_symbols
  - row_rep_bboxes_448

Only the SOURCE image gets row lines and symbols.
The activation-patching target should stay a clean blank image, matching the
counting experiment design.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


DEFAULT_PATCH_SIZE = 28
DEFAULT_TARGET_SIZE = 448
DEFAULT_OVERLAP_THR = 0.20
DEFAULT_LINE_WIDTH = 3
DEFAULT_MARKER_OUTLINE_WIDTH = 2
SYMBOLS = ["@", "#", "$", "&"]
LETTER_LABELS = ["A", "B", "C", "D"]
BOX_COLORS = ["red", "lime", "cyan", "yellow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare row-symbol source images and metadata for activation patching."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV from find_distinct_row_coco_images.py",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root for prepared dataset.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Letterbox size. Default: 448.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Patch size. Default: 28.",
    )
    parser.add_argument(
        "--overlap-thr",
        type=float,
        default=DEFAULT_OVERLAP_THR,
        help="Minimum patch overlap fraction. Default: 0.20",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=DEFAULT_LINE_WIDTH,
        help="Row line width. Default: 3.",
    )
    parser.add_argument(
        "--marker-outline-width",
        type=int,
        default=DEFAULT_MARKER_OUTLINE_WIDTH,
        help="Outline width for row lines and symbols. Default: 2.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Prepare only the top K rows from the input CSV. Default: 0 (all).",
    )
    parser.add_argument(
        "--label-set",
        choices=["symbol", "letter"],
        default="symbol",
        help="Labels to draw on source images. Default: symbol",
    )
    return parser.parse_args()


def get_row_labels(label_set: str) -> list[str]:
    if label_set == "letter":
        return LETTER_LABELS
    return SYMBOLS


def load_font(patch_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = max(16, int(patch_size * 0.8))
    for name in [
        "DejaVuSansMono-Bold.ttf",
        "LiberationMono-Bold.ttf",
        "Arial Unicode.ttf",
    ]:
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def letterbox_image(image: Image.Image, target_size: int) -> tuple[Image.Image, float, int, int, int, int]:
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image size: {orig_w}x{orig_h}")

    scale = target_size / float(max(orig_w, orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, new_w, new_h


def transform_bbox_xywh(bbox: list[float], scale: float, pad_x: int, pad_y: int) -> list[float]:
    x, y, w, h = bbox
    return [x * scale + pad_x, y * scale + pad_y, w * scale, h * scale]


def bbox_to_patch_ids_xywh(
    bbox: list[float],
    target_size: int,
    patch_size: int,
    overlap_thr: float,
) -> list[int]:
    grid = target_size // patch_size
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    x1 = max(0.0, min(float(target_size), x1))
    y1 = max(0.0, min(float(target_size), y1))
    x2 = max(0.0, min(float(target_size), x2))
    y2 = max(0.0, min(float(target_size), y2))

    if x2 <= x1 or y2 <= y1:
        return []

    c_min = max(0, int(math.floor(x1 / patch_size)))
    c_max = min(grid - 1, int(math.floor((x2 - 1e-6) / patch_size)))
    r_min = max(0, int(math.floor(y1 / patch_size)))
    r_max = min(grid - 1, int(math.floor((y2 - 1e-6) / patch_size)))

    patch_area = float(patch_size * patch_size)
    patch_ids = []

    for r in range(r_min, r_max + 1):
        py1 = r * patch_size
        py2 = py1 + patch_size
        for c in range(c_min, c_max + 1):
            px1 = c * patch_size
            px2 = px1 + patch_size

            ix1 = max(px1, x1)
            iy1 = max(py1, y1)
            ix2 = min(px2, x2)
            iy2 = min(py2, y2)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih

            if inter / patch_area >= overlap_thr:
                patch_ids.append(r * grid + c + 1)

    return sorted(set(patch_ids))


def draw_row_markers(
    image: Image.Image,
    content_bbox: tuple[int, int, int, int],
    patch_size: int,
    line_width: int,
    marker_outline_width: int,
    row_labels: list[str],
) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(patch_size)
    width, height = image.size

    content_left, content_top, _, content_bottom = content_bbox
    content_height = max(1, content_bottom - content_top)
    grid_size = width // patch_size
    first_content_col = max(0, min(grid_size - 1, int(content_left // patch_size)))
    patch_x1 = first_content_col * patch_size

    for idx in range(1, 4):
        y = round(content_top + (idx * content_height / 4.0))
        if marker_outline_width > 0:
            draw.line(
                [(0, y), (width, y)],
                fill="black",
                width=line_width + (2 * marker_outline_width),
            )
        draw.line([(0, y), (width, y)], fill="white", width=line_width)

    for idx, symbol in enumerate(row_labels):
        band_y1 = content_top + (idx * content_height / 4.0)
        band_y2 = content_top + ((idx + 1) * content_height / 4.0)
        patch_y1 = round(((band_y1 + band_y2) / 2.0) - (patch_size / 2.0))
        patch_y1 = max(0, min(height - patch_size, patch_y1))

        try:
            bbox = draw.textbbox((0, 0), symbol, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(symbol, font=font)

        text_x = patch_x1 + (patch_size - text_w) / 2.0
        text_y = patch_y1 + (patch_size - text_h) / 2.0
        try:
            draw.text(
                (text_x, text_y),
                symbol,
                fill="white",
                font=font,
                stroke_width=marker_outline_width,
                stroke_fill="black",
            )
        except TypeError:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((text_x + dx, text_y + dy), symbol, fill="black", font=font)
            draw.text((text_x, text_y), symbol, fill="white", font=font)

    return image


def draw_debug_boxes(
    image: Image.Image,
    row_rep_bboxes_448: list[list[float]],
    row_categories: list[str],
    patch_lists: list[list[int]],
    patch_size: int,
    row_labels: list[str],
) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(patch_size)

    for idx, (bbox, category, patch_ids) in enumerate(zip(row_rep_bboxes_448, row_categories, patch_lists)):
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        x, y, w, h = bbox
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{row_labels[idx]} {category} ({len(patch_ids)}p)"
        text_x = x1
        text_y = max(0, y1 - max(18, patch_size))
        try:
            draw.text(
                (text_x, text_y),
                label,
                fill=color,
                font=font,
                stroke_width=2,
                stroke_fill="black",
            )
        except TypeError:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((text_x + dx, text_y + dy), label, fill="black", font=font)
            draw.text((text_x, text_y), label, fill=color, font=font)

    return image


def read_rows(input_csv: Path, top_k: int) -> list[dict]:
    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if top_k > 0:
        rows = rows[:top_k]
    return rows


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    source_root = output_root / "source"
    debug_box_root = output_root / "debug_box"
    output_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    debug_box_root.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv=input_csv, top_k=args.top_k)
    row_labels = get_row_labels(args.label_set)
    meta_rows = []

    for rank, row in enumerate(tqdm(rows, desc="Preparing row-symbol dataset", unit="img"), start=1):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            row_categories = json.loads(row["row_categories"])
            row_category_ids = json.loads(row["row_category_ids"])
            row_ann_ids = json.loads(row["row_ann_ids"])
            row_rep_bboxes = json.loads(row["row_rep_bboxes"])
            row_rep_fracs = json.loads(row["row_rep_fracs"])
        except Exception as exc:
            print(f"Skipping malformed row for {image_path}: {exc}")
            continue

        if not (
            len(row_categories)
            == len(row_category_ids)
            == len(row_ann_ids)
            == len(row_rep_bboxes)
            == len(row_rep_fracs)
            == 4
        ):
            print(f"Skipping row with non-4 row metadata: {image_path}")
            continue

        orig_image = Image.open(image_path).convert("RGB")
        letterboxed, scale, pad_x, pad_y, new_w, new_h = letterbox_image(
            orig_image, target_size=args.target_size
        )
        content_bbox = (pad_x, pad_y, pad_x + new_w, pad_y + new_h)
        source_image = draw_row_markers(
            image=letterboxed.copy(),
            content_bbox=content_bbox,
            patch_size=args.patch_size,
            line_width=args.line_width,
            marker_outline_width=args.marker_outline_width,
            row_labels=row_labels,
        )

        image_stem = image_path.stem
        image_id = row.get("image_id", image_stem)
        source_name = f"{rank:04d}_{image_id}_{image_stem}_rowsym.png"
        source_path = source_root / source_name

        row_rep_bboxes_448 = []
        patch_lists = []
        bad_row = False
        for bbox in row_rep_bboxes:
            bbox_448 = transform_bbox_xywh(
                bbox=[float(v) for v in bbox],
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            patch_ids = bbox_to_patch_ids_xywh(
                bbox=bbox_448,
                target_size=args.target_size,
                patch_size=args.patch_size,
                overlap_thr=args.overlap_thr,
            )
            if not patch_ids:
                bad_row = True
                break
            row_rep_bboxes_448.append([round(v, 2) for v in bbox_448])
            patch_lists.append(patch_ids)

        if bad_row:
            print(f"Skipping row with empty patch list after transform: {image_path}")
            continue

        source_image.save(source_path)
        debug_box_image = draw_debug_boxes(
            image=source_image.copy(),
            row_rep_bboxes_448=row_rep_bboxes_448,
            row_categories=row_categories,
            patch_lists=patch_lists,
            patch_size=args.patch_size,
            row_labels=row_labels,
        )
        debug_box_path = debug_box_root / source_name
        debug_box_image.save(debug_box_path)

        meta_rows.append(
            {
                "rank": rank,
                "split": row.get("split", ""),
                "image_id": row.get("image_id", ""),
                "file_name": row.get("file_name", image_path.name),
                "orig_image_path": str(image_path),
                "source_image_path": str(source_path),
                "debug_box_image_path": str(debug_box_path),
                "row_symbols": json.dumps(row_labels),
                "label_set": args.label_set,
                "row_categories": json.dumps(row_categories),
                "row_category_ids": json.dumps(row_category_ids),
                "row_ann_ids": json.dumps(row_ann_ids),
                "row_rep_fracs": json.dumps(row_rep_fracs),
                "row_rep_bboxes_orig": json.dumps(row_rep_bboxes),
                "row_rep_bboxes_448": json.dumps(row_rep_bboxes_448),
                "patch_lists": json.dumps(patch_lists),
                "content_bbox_448": json.dumps(list(content_bbox)),
            }
        )

    meta_path = output_root / "meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "split",
                "image_id",
                "file_name",
                "orig_image_path",
                "source_image_path",
                "debug_box_image_path",
                "row_symbols",
                "label_set",
                "row_categories",
                "row_category_ids",
                "row_ann_ids",
                "row_rep_fracs",
                "row_rep_bboxes_orig",
                "row_rep_bboxes_448",
                "patch_lists",
                "content_bbox_448",
            ],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"Saved {len(meta_rows)} prepared rows to {meta_path}")


if __name__ == "__main__":
    main()
