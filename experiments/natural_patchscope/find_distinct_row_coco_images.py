#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find raw COCO images where four different object categories can each represent
one of four horizontal rows.

Unlike the single-category counting datasets in this repo, this script reads
raw COCO annotations directly and keeps only images that can provide:
  - one representative object for row 1
  - one representative object for row 2
  - one representative object for row 3
  - one representative object for row 4
  - with all four representative category names distinct

It writes row-wise category names and rough color names into the CSV so the
selected images are usable for prompts such as:
  "Which row has the bus?"
"""

import argparse
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

from PIL import Image, ImageStat
from tqdm import tqdm


DEFAULT_ANN_CANDIDATES = [
    Path("instances_val2017.json"),
    Path("train_data/instances_train2017.json"),
]

DEFAULT_IMAGE_DIR_CANDIDATES = [
    Path("val2017"),
    Path("train2017"),
]

BASE_COLOR_PROTOTYPES = {
    "red": (220, 60, 60),
    "orange": (230, 140, 40),
    "yellow": (225, 210, 70),
    "green": (80, 170, 80),
    "cyan": (70, 180, 190),
    "blue": (70, 110, 210),
    "purple": (140, 90, 180),
    "pink": (220, 140, 180),
    "brown": (145, 95, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select raw COCO images with four distinct category labels across four rows."
    )
    parser.add_argument(
        "--ann-file",
        action="append",
        dest="ann_files",
        help="COCO instances JSON path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        dest="image_dirs",
        help="Directory containing COCO images. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default="distinct_row_coco_images.csv",
        help="Output CSV path. Default: distinct_row_coco_images.csv",
    )
    parser.add_argument(
        "--min-row-score",
        type=float,
        default=0.70,
        help="Minimum weakest-row dominance for the chosen 4-row assignment. Default: 0.70",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=300,
        help="Keep only the top K images after ranking. Default: 300",
    )
    parser.add_argument(
        "--min-box-area-frac",
        type=float,
        default=0.01,
        help="Ignore boxes smaller than this fraction of image area. Default: 0.01",
    )
    parser.add_argument(
        "--max-per-row-candidates",
        type=int,
        default=12,
        help="Limit per-row candidate objects before row assignment search. Default: 12",
    )
    return parser.parse_args()


def existing_paths(candidates: list[Path]) -> list[Path]:
    return [path for path in candidates if path.exists()]


def infer_split_from_ann_file(ann_file: Path) -> str:
    name = ann_file.name.lower()
    if "train" in name:
        return "train"
    return "val"


def sort_image_dirs_for_split(image_dirs: list[Path], split: str) -> list[Path]:
    split_tag = f"{split}2017"
    preferred = [path for path in image_dirs if split_tag in str(path).lower()]
    fallback = [path for path in image_dirs if path not in preferred]
    return preferred + fallback


def resolve_image_path(file_name: str, split: str, image_dirs: list[Path]) -> Path | None:
    ordered_dirs = sort_image_dirs_for_split(image_dirs, split=split)
    for image_dir in ordered_dirs:
        candidate = image_dir / file_name
        if candidate.exists():
            return candidate
    return None


def build_row_bands(height: float) -> list[tuple[float, float]]:
    bands = []
    total_height = max(1.0, float(height))
    for idx in range(4):
        y1 = idx * total_height / 4.0
        y2 = (idx + 1) * total_height / 4.0
        bands.append((y1, y2))
    return bands


def overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in values]


def object_row_masses_from_bbox(
    bbox: list[float],
    row_bands: list[tuple[float, float]],
) -> list[float]:
    x, y, w, h = bbox
    y1, y2 = y, y + h
    masses = [0.0, 0.0, 0.0, 0.0]
    if w <= 0 or h <= 0:
        return masses

    for row_idx, (ry1, ry2) in enumerate(row_bands):
        inter_h = overlap_1d(y1, y2, ry1, ry2)
        if inter_h <= 0:
            continue
        masses[row_idx] += inter_h * w

    return masses


def clamp_bbox_to_image(bbox: list[float], image: Image.Image) -> tuple[int, int, int, int] | None:
    x, y, w, h = bbox
    left = max(0, min(image.width, int(round(x))))
    top = max(0, min(image.height, int(round(y))))
    right = max(0, min(image.width, int(round(x + w))))
    bottom = max(0, min(image.height, int(round(y + h))))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def mean_rgb_for_bbox(image: Image.Image, bbox: list[float]) -> list[float] | None:
    clamped = clamp_bbox_to_image(bbox, image)
    if clamped is None:
        return None
    crop = image.crop(clamped)
    if crop.width <= 0 or crop.height <= 0:
        return None
    return list(ImageStat.Stat(crop).mean[:3])


def rgb_distance_normalized(rgb_a: list[float], rgb_b: list[float]) -> float:
    dist = sqrt(sum((a - b) ** 2 for a, b in zip(rgb_a, rgb_b)))
    return dist / sqrt(3.0 * (255.0**2))


def mean_rgb_to_color_name(rgb: list[float] | None) -> str:
    if rgb is None:
        return ""

    r, g, b = rgb
    brightness = (r + g + b) / 3.0
    chroma = max(rgb) - min(rgb)

    if brightness <= 45:
        return "black"
    if brightness >= 225 and chroma <= 30:
        return "white"
    if chroma <= 18:
        return "gray"

    best_name = "gray"
    best_dist = float("inf")
    for color_name, proto in BASE_COLOR_PROTOTYPES.items():
        dist = sqrt(sum((a - b) ** 2 for a, b in zip(rgb, proto)))
        if dist < best_dist:
            best_dist = dist
            best_name = color_name
    return best_name


def combo_key(objects: list[dict], combo: list[int]) -> tuple:
    fracs = [objects[obj_idx]["dominant_frac"] for obj_idx in combo]
    areas = [objects[obj_idx]["area"] for obj_idx in combo]
    return (
        min(fracs),
        sum(fracs) / 4.0,
        min(areas),
        sum(areas),
    )


def choose_distinct_row_combo(objects: list[dict], max_per_row_candidates: int) -> list[int] | None:
    row_candidates = {}
    for row_idx in range(4):
        indices = [idx for idx, obj in enumerate(objects) if obj["best_row"] == row_idx]
        indices.sort(
            key=lambda idx: (
                -objects[idx]["dominant_frac"],
                -objects[idx]["area"],
                objects[idx]["category_name"],
            )
        )
        row_candidates[row_idx] = indices[:max_per_row_candidates]
        if not row_candidates[row_idx]:
            return None

    best_combo = None
    best_key = None

    def dfs(row_idx: int, used_categories: set[int], chosen: list[int]) -> None:
        nonlocal best_combo, best_key
        if row_idx == 4:
            key = combo_key(objects, chosen)
            if best_key is None or key > best_key:
                best_key = key
                best_combo = chosen.copy()
            return

        for obj_idx in row_candidates[row_idx]:
            category_id = objects[obj_idx]["category_id"]
            if category_id in used_categories:
                continue
            used_categories.add(category_id)
            chosen.append(obj_idx)
            dfs(row_idx + 1, used_categories, chosen)
            chosen.pop()
            used_categories.remove(category_id)

    dfs(row_idx=0, used_categories=set(), chosen=[])
    return best_combo


def load_coco(ann_file: Path) -> tuple[dict[int, dict], dict[int, str], dict[int, list[dict]]]:
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {image["id"]: image for image in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    return images, categories, anns_by_image


def score_image(
    image_info: dict,
    anns: list[dict],
    categories: dict[int, str],
    max_per_row_candidates: int,
    min_box_area_frac: float,
) -> dict | None:
    width = int(image_info.get("width", 0))
    height = int(image_info.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    image_area = float(width * height)
    min_area = min_box_area_frac * image_area
    row_bands = build_row_bands(height)

    filtered = []
    for ann in anns:
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        _, _, w, h = bbox
        area = max(0.0, w) * max(0.0, h)
        if area < min_area:
            continue

        masses = object_row_masses_from_bbox(bbox=bbox, row_bands=row_bands)
        fracs = normalize(masses)
        best_row = int(max(range(4), key=lambda idx: fracs[idx]))
        dominant_frac = fracs[best_row]
        filtered.append(
            {
                "ann_id": ann["id"],
                "bbox": bbox,
                "area": area,
                "category_id": ann["category_id"],
                "category_name": categories.get(ann["category_id"], f"cat_{ann['category_id']}"),
                "row_fracs": fracs,
                "best_row": best_row,
                "dominant_frac": dominant_frac,
            }
        )

    if len(filtered) < 4:
        return None

    distinct_categories = {obj["category_id"] for obj in filtered}
    if len(distinct_categories) < 4:
        return None

    combo = choose_distinct_row_combo(filtered, max_per_row_candidates=max_per_row_candidates)
    if combo is None:
        return None

    chosen = [filtered[obj_idx] for obj_idx in combo]
    row_query_score = min(obj["dominant_frac"] for obj in chosen)
    row_query_mean = sum(obj["dominant_frac"] for obj in chosen) / 4.0

    return {
        "object_count": len(filtered),
        "distinct_category_count": len(distinct_categories),
        "row_ann_ids": [obj["ann_id"] for obj in chosen],
        "row_category_ids": [obj["category_id"] for obj in chosen],
        "row_categories": [obj["category_name"] for obj in chosen],
        "row_rep_bboxes": [obj["bbox"] for obj in chosen],
        "row_rep_fracs": [obj["dominant_frac"] for obj in chosen],
        "row_query_score": row_query_score,
        "row_query_mean": row_query_mean,
    }


def sort_key(row: dict) -> tuple:
    return (
        -float(row["row_query_score"]),
        -float(row["row_query_mean"]),
        -float(row["row_color_score"]),
        -float(row["row_color_mean"]),
        int(row["object_count"]),
    )


def build_output_row(
    split: str,
    image_info: dict,
    image_path: Path,
    result: dict,
) -> dict:
    image = Image.open(image_path).convert("RGB")

    row_rep_mean_rgbs = []
    row_color_names = []
    color_dists = []
    for bbox in result["row_rep_bboxes"]:
        mean_rgb = mean_rgb_for_bbox(image, bbox)
        row_rep_mean_rgbs.append(mean_rgb)
        row_color_names.append(mean_rgb_to_color_name(mean_rgb))

    if all(rgb is not None for rgb in row_rep_mean_rgbs):
        for i in range(4):
            for j in range(i + 1, 4):
                color_dists.append(rgb_distance_normalized(row_rep_mean_rgbs[i], row_rep_mean_rgbs[j]))

    row_color_score = min(color_dists) if color_dists else 0.0
    row_color_mean = sum(color_dists) / len(color_dists) if color_dists else 0.0

    return {
        "split": split,
        "image_id": image_info["id"],
        "file_name": image_info["file_name"],
        "image_path": str(image_path),
        "object_count": result["object_count"],
        "distinct_category_count": result["distinct_category_count"],
        "row_categories": json.dumps(result["row_categories"]),
        "row_category_ids": json.dumps(result["row_category_ids"]),
        "row_ann_ids": json.dumps(result["row_ann_ids"]),
        "row_rep_bboxes": json.dumps([[round(v, 2) for v in bbox] for bbox in result["row_rep_bboxes"]]),
        "row_rep_fracs": json.dumps([round(v, 6) for v in result["row_rep_fracs"]]),
        "row_query_score": f"{result['row_query_score']:.6f}",
        "row_query_mean": f"{result['row_query_mean']:.6f}",
        "row_rep_mean_rgbs": json.dumps(
            [None if rgb is None else [round(v, 2) for v in rgb] for rgb in row_rep_mean_rgbs]
        ),
        "row_color_names": json.dumps(row_color_names),
        "row_color_score": f"{row_color_score:.6f}",
        "row_color_mean": f"{row_color_mean:.6f}",
    }


def main() -> None:
    args = parse_args()

    ann_files = [Path(path) for path in args.ann_files] if args.ann_files else existing_paths(DEFAULT_ANN_CANDIDATES)
    image_dirs = [Path(path) for path in args.image_dirs] if args.image_dirs else existing_paths(DEFAULT_IMAGE_DIR_CANDIDATES)
    output_path = Path(args.output)

    if not ann_files:
        raise FileNotFoundError("No annotation files found. Pass --ann-file explicitly.")
    if not image_dirs:
        raise FileNotFoundError("No COCO image directories found. Pass --image-dir explicitly.")

    rows = []

    for ann_file in ann_files:
        split = infer_split_from_ann_file(ann_file)
        images, categories, anns_by_image = load_coco(ann_file)

        candidate_image_ids = []
        for image_id, anns in anns_by_image.items():
            if len(anns) < 4:
                continue
            category_ids = {ann["category_id"] for ann in anns if ann.get("iscrowd", 0) != 1}
            if len(category_ids) < 4:
                continue
            candidate_image_ids.append(image_id)

        desc = f"Selecting {split}"
        for image_id in tqdm(candidate_image_ids, desc=desc, unit="img"):
            image_info = images.get(image_id)
            if image_info is None:
                continue

            image_path = resolve_image_path(
                file_name=image_info["file_name"],
                split=split,
                image_dirs=image_dirs,
            )
            if image_path is None:
                continue

            result = score_image(
                image_info=image_info,
                anns=anns_by_image[image_id],
                categories=categories,
                max_per_row_candidates=args.max_per_row_candidates,
                min_box_area_frac=args.min_box_area_frac,
            )
            if result is None:
                continue
            if result["row_query_score"] < args.min_row_score:
                continue

            try:
                rows.append(
                    build_output_row(
                        split=split,
                        image_info=image_info,
                        image_path=image_path,
                        result=result,
                    )
                )
            except Exception as exc:
                print(f"Skipping image {image_id} ({image_path}): {exc}")

    rows.sort(key=sort_key)
    if args.top_k > 0:
        rows = rows[: args.top_k]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image_id",
                "file_name",
                "image_path",
                "object_count",
                "distinct_category_count",
                "row_categories",
                "row_category_ids",
                "row_ann_ids",
                "row_rep_bboxes",
                "row_rep_fracs",
                "row_query_score",
                "row_query_mean",
                "row_rep_mean_rgbs",
                "row_color_names",
                "row_color_score",
                "row_color_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
