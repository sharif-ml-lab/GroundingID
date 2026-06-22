#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Activation patching experiment for row-symbol identification.

Source context:
  - annotated source image with row lines and symbols on the left

Target context:
  - blank black image used as a neutral target

For each prepared image and each selected row object:
  1) cache source hidden states
  2) patch only that object's vision patches into the blank target run
  3) ask which row symbol that object belongs to
  4) compare generated answer against the expected row symbol

Outputs:
  OUTPUT_ROOT/
    predictions.csv
    summary.json
    vis/
      <image>_row1_patches.png
      ...
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_TARGET_SIZE = 448
DEFAULT_PATCH_SIZE = 28
DEFAULT_MAX_NEW_TOKENS = 4
SYMBOLS = ["@", "#", "$", "&"]
LETTER_LABELS = ["A", "B", "C", "D"]
INDEX_LABELS = ["a", "b", "c", "d"]
NOTHING_LABEL = "nothing"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run row-symbol activation patching on a prepared dataset."
    )
    parser.add_argument(
        "--meta-csv",
        required=True,
        help="Prepared dataset meta.csv from prepare_row_symbol_dataset.py",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for predictions and summary.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Qwen2.5-VL model path. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--layers-start",
        type=int,
        default=0,
        help="First layer to patch. Default: 0",
    )
    parser.add_argument(
        "--layers-end",
        type=int,
        default=27,
        help="Last layer to patch. Default: 27",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Target image size. Default: 448",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Patch size. Default: 28",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Generation length. Default: 4",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Limit number of images for debugging. Default: 0 (all).",
    )
    parser.add_argument(
        "--label-mode",
        choices=["symbol", "index"],
        default="symbol",
        help="Answer space to evaluate. Default: symbol",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["default", "grid"],
        default="default",
        help="Prompt prefix style to use before the image. Default: default",
    )
    return parser.parse_args()


def format_label_list(labels: list[str]) -> str:
    return ",".join(labels)


def format_label_set(labels: list[str]) -> str:
    return ", ".join(labels)


def get_label_token_ids(tokenizer, label: str) -> list[int]:
    token_ids = set()
    for variant in (label, f" {label}"):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(ids[0])
    if not token_ids:
        raise ValueError(f"Label must map to at least one single token: {label!r}")
    return sorted(token_ids)


def get_label_probs(tokenizer, scores: torch.Tensor, labels: list[str]) -> dict[str, float]:
    probs = torch.softmax(scores.float(), dim=-1)
    label_probs = {}
    for label in labels:
        token_ids = get_label_token_ids(tokenizer, label)
        label_probs[label] = float(probs[token_ids].sum().item())
    return label_probs


def build_user_prefix(label_mode: str, prompt_style: str, row_symbols: list[str]) -> str:
    if prompt_style == "grid":
        missing_label_word = "symbol"
        token_set = format_label_set(row_symbols)
        return (
            "The image is divided into a regular grid of patches, arranged in four horizontal rows. "
            f"Each row is labeled with exactly one symbol from the set {{{token_set}}}. "
            "Every patch in the same row must keep the same assigned symbol, and different rows may have different symbols. "
            "Your first task is to carefully understand the full pattern: "
            f"for every row in the entire image, determine which symbol {{{token_set}}} it uses and mentally assign that symbol to all patches in that row. "
            "You must preserve the symbol of each row consistently across all of its patches. "
            f"Some rows may have no objects. Also some rows may have no {missing_label_word}. "
            'If you are asked about these, say "nothing".\n'
        )

    if label_mode == "index":
        return (
            f"Scan the image using the symbols on the left as row labels({format_label_list(row_symbols)}).\n"
            "Treat the rows from top to bottom as row indices a,b,c,d.\n"
            'some rows may have no objects . also some rows may have no symbol. if you asked about these say"nothing"\n'
        )

    return (
        f"Scan the image using the symbols on the left as row labels({format_label_list(row_symbols)}).\n"
        'some rows may have no objects . also some rows may have no symbol. if you asked about these say"nothing"\n'
    )


def build_row_symbol_prompts(
    object_name: str,
    label_mode: str,
    prompt_style: str,
    row_symbols: list[str],
) -> tuple[str, str, str]:
    user_prefix = build_user_prefix(
        label_mode=label_mode,
        prompt_style=prompt_style,
        row_symbols=row_symbols,
    )

    if label_mode == "index":
        system = (
            "You are a vision-language assistant. "
            "Answer with only one of these: a, b, c, d, nothing. "
            "No extra words."
        )
        user_suffix = (
            f"what is the index of the row that \"{object_name.replace('_', ' ')}\" is in?\n"
            "Return only one of: a, b, c, d, nothing."
        )
        return system, user_prefix, user_suffix

    system = (
        "You are a vision-language assistant. "
        f"Answer with only one of these: {format_label_set(row_symbols)}, nothing. "
        "No extra words."
    )
    user_suffix = (
        f"what is the symbol of the row that \"{object_name.replace('_', ' ')}\"?\n"
        f"Return only one of: {format_label_set(row_symbols)}, nothing."
    )
    return system, user_prefix, user_suffix


def prepare_inputs(
    processor,
    image: Image.Image,
    object_name: str,
    label_mode: str,
    prompt_style: str,
    row_symbols: list[str],
) -> dict:
    system, user_prefix, user_suffix = build_row_symbol_prompts(
        object_name,
        label_mode,
        prompt_style,
        row_symbols,
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prefix},
                {"type": "image"},
                {"type": "text", "text": user_suffix},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    batch = processor(images=[image], text=[prompt], return_tensors="pt", padding=True)
    return {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def find_vision_tokens(processor, batch) -> list[int]:
    ids = batch["input_ids"][0].tolist()
    toks = processor.tokenizer.convert_ids_to_tokens(ids)

    vis = []
    i = 0
    while i < len(toks):
        if toks[i] == "<|vision_start|>":
            j = i + 1
            while j < len(toks) and toks[j] != "<|vision_end|>":
                vis.append(j)
                j += 1
            i = j
        i += 1
    return vis


def get_blocks(model):
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        if hasattr(m, "layers"):
            return m.layers

    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            if len(module) > 0 and hasattr(module[0], "self_attn"):
                return module

    raise RuntimeError("Cannot locate transformer blocks.")


@torch.no_grad()
def collect_src_hidden_states(model, batch, vis_positions) -> dict:
    out = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states

    cache = {}
    for i in range(1, len(hs)):
        layer_cache = {}
        h = hs[i][0]
        for pos in vis_positions:
            layer_cache[pos] = h[pos].detach()
        cache[i - 1] = layer_cache
    return cache


@contextmanager
def patch_layers(model, layer_cache, patch_positions, layers_start: int, layers_end: int):
    blocks = get_blocks(model)
    handles = []

    def make_hook(layer_idx):
        src = layer_cache[layer_idx]

        def hook(_, inputs):
            x = inputs[0].clone()
            for p in patch_positions:
                if p >= x.shape[1]:
                    continue
                if p in src:
                    rep = src[p].to(x.device, x.dtype)
                    scale = x[:, p].norm(dim=-1, keepdim=True) / (
                        rep.norm(dim=-1, keepdim=True) + 1e-6
                    )
                    x[:, p] = rep * scale
            return (x,) + inputs[1:]

        return hook

    for li in range(layers_start, min(layers_end + 1, len(blocks))):
        handles.append(blocks[li].register_forward_pre_hook(make_hook(li)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def patch_id_to_token_pos(patch_ids_1b: list[int], vis_positions: list[int]) -> list[int]:
    token_pos = []
    for pid in patch_ids_1b:
        idx = pid - 1
        if 0 <= idx < len(vis_positions):
            token_pos.append(vis_positions[idx])
    return sorted(set(token_pos))


def draw_patches(img_path: str, patch_ids: list[int], out_path: Path, patch_size: int, grid_size: int) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for pid in patch_ids:
        r = (pid - 1) // grid_size
        c = (pid - 1) % grid_size
        x0, y0 = c * patch_size, r * patch_size
        x1, y1 = x0 + patch_size, y0 + patch_size
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline="red", width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def normalize_answer(text: str, label_mode: str, row_symbols: list[str]) -> str:
    raw = text.strip()
    lowered = raw.lower()

    if "nothing" in lowered:
        return "nothing"

    if label_mode == "index":
        letter_match = re.search(r"\b([a-d])\b", lowered)
        if letter_match:
            return letter_match.group(1)

        digit_map = {
            "1": "a",
            "2": "b",
            "3": "c",
            "4": "d",
        }
        digit_match = re.search(r"[1-4]", raw)
        if digit_match:
            return digit_map[digit_match.group(0)]

        word_map = {
            "one": "a",
            "first": "a",
            "two": "b",
            "second": "b",
            "three": "c",
            "third": "c",
            "four": "d",
            "fourth": "d",
        }
        for word, index_label in word_map.items():
            if word in lowered:
                return index_label
        return raw

    for symbol in row_symbols:
        if re.fullmatch(r"[A-Za-z0-9]+", symbol):
            if re.search(rf"\b{re.escape(symbol.lower())}\b", lowered):
                return symbol
        elif symbol in raw:
            return symbol

    if row_symbols == SYMBOLS:
        word_map = {
            "at": "@",
            "hash": "#",
            "number sign": "#",
            "dollar": "$",
            "ampersand": "&",
        }
        for word, symbol in word_map.items():
            if word in lowered:
                return symbol

    if row_symbols == LETTER_LABELS:
        for symbol in LETTER_LABELS:
            if re.search(rf"\b{symbol.lower()}\b", lowered):
                return symbol

    return raw


def load_meta_rows(meta_csv: Path, limit_images: int) -> list[dict]:
    with open(meta_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit_images > 0:
        rows = rows[:limit_images]
    return rows


def run_one_example(
    processor,
    model,
    tokenizer,
    source_img: Image.Image,
    target_img: Image.Image,
    object_name: str,
    label_mode: str,
    prompt_style: str,
    row_symbols: list[str],
    expected_symbol: str,
    patch_ids: list[int],
    layers_start: int,
    layers_end: int,
    max_new_tokens: int,
) -> tuple[str, str, float, float, float]:
    src_batch = prepare_inputs(
        processor,
        source_img,
        object_name,
        label_mode,
        prompt_style,
        row_symbols,
    )
    tgt_batch = prepare_inputs(
        processor,
        target_img,
        object_name,
        label_mode,
        prompt_style,
        row_symbols,
    )

    vis_positions = find_vision_tokens(processor, src_batch)
    layer_cache = collect_src_hidden_states(model, src_batch, vis_positions)
    patch_positions = patch_id_to_token_pos(patch_ids, vis_positions)
    answer_labels = (INDEX_LABELS if label_mode == "index" else row_symbols) + [NOTHING_LABEL]

    with patch_layers(model, layer_cache, patch_positions, layers_start, layers_end):
        out = model.generate(
            **tgt_batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    input_len = tgt_batch["input_ids"].shape[1]
    gen_ids = out.sequences[0, input_len:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    pred = normalize_answer(raw_text, label_mode, row_symbols)
    gt_prob = 0.0
    non_gt_total_prob = 0.0
    avg_non_gt_prob = 0.0
    if out.scores:
        label_probs = get_label_probs(tokenizer, out.scores[0][0], answer_labels)
        gt_prob = label_probs.get(expected_symbol, 0.0)
        non_gt_labels = [label for label in answer_labels if label != expected_symbol]
        non_gt_total_prob = sum(label_probs.get(label, 0.0) for label in non_gt_labels)
        avg_non_gt_prob = (
            non_gt_total_prob / len(non_gt_labels)
        ) if non_gt_labels else 0.0
    return raw_text, pred, gt_prob, non_gt_total_prob, avg_non_gt_prob


def summarize(prediction_rows: list[dict], label_mode: str, prompt_style: str) -> dict:
    total = len(prediction_rows)
    correct = sum(int(row["is_correct"]) for row in prediction_rows)
    by_row = defaultdict(lambda: {"total": 0, "correct": 0})
    by_symbol = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in prediction_rows:
        row_name = f"row_{row['row_index']}"
        symbol = row["expected_symbol"]
        is_correct = int(row["is_correct"])

        by_row[row_name]["total"] += 1
        by_row[row_name]["correct"] += is_correct
        by_symbol[symbol]["total"] += 1
        by_symbol[symbol]["correct"] += is_correct

    summary = {
        "total_examples": total,
        "correct_examples": correct,
        "overall_accuracy": (correct / total) if total else 0.0,
        "label_mode": label_mode,
        "prompt_style": prompt_style,
        "mean_gt_prob": (
            sum(float(row["gt_prob"]) for row in prediction_rows) / total
        ) if total else 0.0,
        "mean_non_gt_total_prob": (
            sum(float(row["non_gt_total_prob"]) for row in prediction_rows) / total
        ) if total else 0.0,
        "mean_avg_non_gt_prob": (
            sum(float(row["avg_non_gt_prob"]) for row in prediction_rows) / total
        ) if total else 0.0,
        "by_row": {},
        "by_symbol": {},
    }

    for key, stats in by_row.items():
        total_k = stats["total"]
        correct_k = stats["correct"]
        summary["by_row"][key] = {
            "total": total_k,
            "correct": correct_k,
            "accuracy": (correct_k / total_k) if total_k else 0.0,
        }

    for key, stats in by_symbol.items():
        total_k = stats["total"]
        correct_k = stats["correct"]
        summary["by_symbol"][key] = {
            "total": total_k,
            "correct": correct_k,
            "accuracy": (correct_k / total_k) if total_k else 0.0,
        }

    return summary


def main() -> None:
    args = parse_args()

    meta_csv = Path(args.meta_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    vis_root = output_root / "vis"
    predictions_csv = output_root / "predictions.csv"
    summary_json = output_root / "summary.json"

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    tokenizer = processor.tokenizer

    rows = load_meta_rows(meta_csv=meta_csv, limit_images=args.limit_images)
    blank_target = Image.new("RGB", (args.target_size, args.target_size), color=(0, 0, 0))
    grid_size = args.target_size // args.patch_size

    prediction_rows = []
    total_examples = len(rows) * 4

    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "image_id",
                "source_image_path",
                "row_index",
                "object_name",
                "expected_symbol",
                "pred_symbol",
                "raw_text",
                "is_correct",
                "patch_count",
                "gt_prob",
                "non_gt_total_prob",
                "avg_non_gt_prob",
                "prompt_style",
            ],
        )
        writer.writeheader()

        with tqdm(total=total_examples, desc="Row-symbol patching", unit="obj") as pbar:
            for row in rows:
                source_image_path = row["source_image_path"]
                source_img = Image.open(source_image_path).convert("RGB")
                patch_lists = json.loads(row["patch_lists"])
                row_categories = json.loads(row["row_categories"])
                row_symbols = json.loads(row["row_symbols"])

                for row_idx in range(4):
                    object_name = row_categories[row_idx]
                    if args.label_mode == "index":
                        expected_symbol = INDEX_LABELS[row_idx]
                    else:
                        expected_symbol = row_symbols[row_idx]
                    patch_ids = patch_lists[row_idx]

                    vis_path = vis_root / f"{Path(source_image_path).stem}_row{row_idx + 1}.png"
                    draw_patches(
                        img_path=source_image_path,
                        patch_ids=patch_ids,
                        out_path=vis_path,
                        patch_size=args.patch_size,
                        grid_size=grid_size,
                    )

                    raw_text, pred_symbol, gt_prob, non_gt_total_prob, avg_non_gt_prob = run_one_example(
                        processor=processor,
                        model=model,
                        tokenizer=tokenizer,
                        source_img=source_img,
                        target_img=blank_target,
                        object_name=object_name,
                        label_mode=args.label_mode,
                        prompt_style=args.prompt_style,
                        row_symbols=row_symbols,
                        expected_symbol=expected_symbol,
                        patch_ids=patch_ids,
                        layers_start=args.layers_start,
                        layers_end=args.layers_end,
                        max_new_tokens=args.max_new_tokens,
                    )

                    is_correct = int(pred_symbol == expected_symbol)
                    out_row = {
                        "rank": row["rank"],
                        "image_id": row["image_id"],
                        "source_image_path": source_image_path,
                        "row_index": row_idx + 1,
                        "object_name": object_name,
                        "expected_symbol": expected_symbol,
                        "pred_symbol": pred_symbol,
                        "raw_text": raw_text,
                        "is_correct": is_correct,
                        "patch_count": len(patch_ids),
                        "gt_prob": f"{gt_prob:.8f}",
                        "non_gt_total_prob": f"{non_gt_total_prob:.8f}",
                        "avg_non_gt_prob": f"{avg_non_gt_prob:.8f}",
                        "prompt_style": args.prompt_style,
                    }
                    writer.writerow(out_row)
                    prediction_rows.append(out_row)
                    pbar.update(1)

    summary = summarize(
        prediction_rows,
        label_mode=args.label_mode,
        prompt_style=args.prompt_style,
    )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved predictions to {predictions_csv}")
    print(f"Saved summary to {summary_json}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.4f} ({summary['correct_examples']}/{summary['total_examples']})")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
