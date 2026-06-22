#!/usr/bin/env python3
"""Two-donor standard-vs-swapped activation patching on the multi-object dataset.

This implements the two-donor intervention on structure-only blank hosts:
- choose two donor objects from different source rows
- record their source-image visual activations after a full forward pass
- patch both donors into a blank target host
- evaluate:
  1) standard setting: donors patched into their own source-symbol rows
  2) swapped setting: donors patched into each other's source-symbol rows

For each setting, the script asks separate shape and color questions for the
two queried symbols and saves:
- pair metadata
- per-query log-prob scores
- swapped bound-vs-physical accuracy
- 2x2 heatmaps for shape and color
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_SYMBOLS = ["@", "#", "$", "&"]
NONE_LABEL = "none"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Two-donor standard/swapped binding evaluation on structured blank targets.")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--source_metadata", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--target_metadata", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--max_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--layers_start", type=int, default=0)
    ap.add_argument("--layers_end", type=int, default=27)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def load_meta(path: Path) -> List[dict]:
    meta = json.loads(path.read_text())
    return list(meta.values()) if isinstance(meta, dict) else list(meta)


def symbol_order(rec: dict) -> List[str]:
    syms = rec.get("symbol_arrangement") or rec.get("row_symbols")
    return list(syms) if syms else list(DEFAULT_SYMBOLS)


def row_values(rec: dict) -> List[int]:
    if rec.get("target_rows"):
        return [int(x) for x in rec["target_rows"]]
    if rec.get("row_summaries"):
        return [int(r["row"]) for r in rec["row_summaries"]]
    rows = sorted({int(o["row"]) for o in rec.get("objects", []) if "row" in o})
    return rows


def row_to_symbol(rec: dict) -> Dict[int, str]:
    rows = row_values(rec)
    syms = symbol_order(rec)
    return {rows[i]: syms[i] for i in range(min(len(rows), len(syms)))}


def symbol_to_row(rec: dict) -> Dict[str, int]:
    return {v: k for k, v in row_to_symbol(rec).items()}


def shape_objects(rec: dict) -> List[dict]:
    objs = [o for o in rec.get("objects", []) if "shape" in o and "color" in o]
    return sorted(
        objs,
        key=lambda o: (
            int(o.get("row_index_0based", 10**9)),
            int(o.get("col_index_0based", 10**9)),
            int(o.get("row", 10**9)),
            int(o.get("grid_col", 10**9)),
            float((o.get("center_position") or [10**9, 10**9])[0]),
        ),
    )


def slot_positions(rec: dict) -> List[dict]:
    return sorted(
        list(rec.get("slot_positions", [])),
        key=lambda s: (
            int(s.get("row_index_0based", 10**9)),
            int(s.get("col_index_0based", 10**9)),
            int(s.get("row", 10**9)),
            int(s.get("grid_col", 10**9)),
        ),
    )


def object_label(obj: dict) -> str:
    return f"{str(obj['color']).lower()} {str(obj['shape']).lower()}"


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
    raise ValueError(f"Could not resolve bbox for object/slot: {obj}")


def find_source_object(rec: dict, sym: str, grid_col: int) -> dict:
    row_val = symbol_to_row(rec)[sym]
    for obj in shape_objects(rec):
        if int(obj["row"]) == int(row_val) and int(obj["grid_col"]) == int(grid_col):
            return obj
    raise RuntimeError(f"Could not find source object sym={sym} grid_col={grid_col} in {rec['filename']}")


def find_target_slot(rec: dict, sym: str, grid_col: int) -> dict:
    row_val = symbol_to_row(rec)[sym]
    for slot in slot_positions(rec):
        if int(slot["row"]) == int(row_val) and int(slot["grid_col"]) == int(grid_col):
            return slot
    raise RuntimeError(f"Could not find target slot sym={sym} grid_col={grid_col} in {rec['filename']}")


def best_2d_factors(n: int) -> Tuple[int, int]:
    best = (1, n)
    gap = n
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            if abs(h - w) < gap:
                best = (h, w)
                gap = abs(h - w)
    return best


def bbox_to_patch_ids_1b(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    canvas_w: float,
    canvas_h: float,
    htok: int,
    wtok: int,
    pad: int,
) -> List[int]:
    tr = lambda yp: int(max(0, min(math.floor(yp / canvas_h * htok), htok - 1)))
    tc = lambda xp: int(max(0, min(math.floor(xp / canvas_w * wtok), wtok - 1)))
    r0, r1 = tr(y0), tr(y1)
    c0, c1 = tc(x0), tc(x1)
    r0 = max(0, min(r0, r1) - pad)
    r1 = min(htok - 1, max(r0, r1) + pad)
    c0 = max(0, min(c0, c1) - pad)
    c1 = min(wtok - 1, max(c0, c1) + pad)

    patch_ids = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            patch_ids.append(r * wtok + c + 1)
    return sorted(set(patch_ids))


def patch_id_to_token_pos(patch_ids_1b: Sequence[int], vis_positions: Sequence[int]) -> List[int]:
    token_pos = []
    for pid in patch_ids_1b:
        idx = pid - 1
        if 0 <= idx < len(vis_positions):
            token_pos.append(vis_positions[idx])
    return sorted(set(token_pos))


def get_blocks(model):
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
            if hasattr(lm, "layers"):
                return lm.layers
        if hasattr(m, "layers"):
            return m.layers
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0 and hasattr(module[0], "self_attn"):
            return module
    raise RuntimeError("Cannot locate transformer blocks.")


def format_label_list(labels: Sequence[str]) -> str:
    return ",".join(labels)


def build_scan_prefix(row_symbols: Sequence[str]) -> str:
    return (
        f"Scan the image sequentially using the symbols on the left as row labels({format_label_list(row_symbols)}).\n"
        "Each horizontal row has exactly one symbol on the left, and that symbol names the whole row.\n"
        "Some rows may have no object.\n"
        'If you are asked about an empty row, answer "none".\n'
    )


def build_attr_prompt(query_symbol: str, row_symbols: Sequence[str], task: str, vocab: Sequence[str]) -> Tuple[str, str, str]:
    choices = ", ".join(sorted(dict.fromkeys([*(str(x).lower() for x in vocab), NONE_LABEL])))
    system = (
        "You are a vision-language assistant. "
        f"Answer with only one lowercase word from: {choices}. "
        "No extra words."
    )
    prefix = build_scan_prefix(row_symbols)
    if task == "shape":
        suffix = (
            f'What is the shape of the object in row "{query_symbol}"?\n'
            f"Return only one of: {choices}."
        )
    elif task == "color":
        suffix = (
            f'What is the color of the object in row "{query_symbol}"?\n'
            f"Return only one of: {choices}."
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return system, prefix, suffix


def prepare_inputs(processor, image: Image.Image, system: str, prefix: str, suffix: str) -> Dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prefix},
                {"type": "image"},
                {"type": "text", "text": suffix},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(images=[image], text=[prompt], return_tensors="pt", padding=True)
    if batch.get("attention_mask") is None:
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    return {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def find_vision_tokens(processor, batch) -> List[int]:
    ids = batch["input_ids"][0].tolist()
    marker_pairs = [
        ("<|vision_start|>", "<|vision_end|>"),
        ("<|image_start|>", "<|image_end|>"),
    ]
    for start_tok, end_tok in marker_pairs:
        try:
            sid = processor.tokenizer.convert_tokens_to_ids(start_tok)
            eid = processor.tokenizer.convert_tokens_to_ids(end_tok)
            start_idx = ids.index(sid)
            end_idx = ids.index(eid)
            if end_idx > start_idx + 1:
                return list(range(start_idx + 1, end_idx))
        except Exception:
            pass
    toks = processor.tokenizer.convert_ids_to_tokens(ids)
    for start_tok, end_tok in marker_pairs:
        vis = []
        i = 0
        while i < len(toks):
            if toks[i] == start_tok:
                j = i + 1
                while j < len(toks) and toks[j] != end_tok:
                    vis.append(j)
                    j += 1
                if vis:
                    return vis
                i = j
            i += 1
    return []


def get_htok_wtok(batch, vis_positions: Sequence[int]) -> Tuple[int, int]:
    if "image_grid_thw" in batch and batch["image_grid_thw"].numel() >= 3:
        thw = batch["image_grid_thw"][0].tolist()
        tdim, htok, wtok = int(thw[-3]), int(thw[-2]), int(thw[-1])
        n_vis = len(vis_positions)
        if htok * wtok == n_vis:
            return htok, wtok
        if tdim * htok * wtok == n_vis:
            return htok, wtok
    return best_2d_factors(len(vis_positions))


@torch.no_grad()
def collect_src_hidden_states(model, batch, vis_positions: Sequence[int]) -> Dict[int, Dict[int, torch.Tensor]]:
    out = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states
    cache: Dict[int, Dict[int, torch.Tensor]] = {}
    for i in range(1, len(hs)):
        h = hs[i][0]
        cache[i - 1] = {pos: h[pos].detach() for pos in vis_positions}
    return cache


@contextmanager
def patch_layers(model, layer_cache, src_positions: Sequence[int], tgt_positions: Sequence[int], layers_start: int, layers_end: int):
    blocks = get_blocks(model)
    handles = []
    pairs = list(zip(tgt_positions, src_positions))

    def make_hook(layer_idx: int):
        src = layer_cache[layer_idx]

        def hook(_, inputs):
            x = inputs[0].clone()
            for tgt_p, src_p in pairs:
                if tgt_p >= x.shape[1]:
                    continue
                rep = src.get(src_p)
                if rep is None:
                    continue
                rep = rep.to(x.device, x.dtype)
                scale = x[:, tgt_p].norm(dim=-1, keepdim=True) / (rep.norm(dim=-1, keepdim=True) + 1e-6)
                x[:, tgt_p] = rep * scale
            return (x,) + inputs[1:]

        return hook

    for li in range(max(0, layers_start), min(layers_end + 1, len(blocks))):
        handles.append(blocks[li].register_forward_pre_hook(make_hook(li)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def string_logprob_with_batch(model, processor, batch, answer: str) -> float:
    tok = processor.tokenizer
    ans_ids = tok(answer, add_special_tokens=False).input_ids
    if not ans_ids:
        return float("-inf")
    ids = torch.cat(
        [batch["input_ids"], torch.tensor([ans_ids], device=batch["input_ids"].device)],
        dim=1,
    )
    am = torch.cat(
        [
            batch["attention_mask"],
            torch.ones((1, len(ans_ids)), dtype=batch["attention_mask"].dtype, device=batch["attention_mask"].device),
        ],
        dim=1,
    )
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=am,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            return_dict=True,
            use_cache=False,
        )
        logits = out.logits[0]
    t = batch["input_ids"].shape[1]
    score = 0.0
    for i, tid in enumerate(ans_ids):
        score += torch.log_softmax(logits[t - 1 + i], dim=-1)[tid].item()
    return float(score / len(ans_ids))


def logsumexp(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    m = max(vals)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def normalize_subset_logprobs(scores: Dict[str, float], keep: Sequence[str]) -> Dict[str, float]:
    z = logsumexp([scores[k] for k in keep])
    return {k: float(scores[k] - z) for k in keep}


def choose_pairs(source_rows: Sequence[dict], target_rows: Sequence[dict], max_pairs: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    target_by_source = {r.get("source_filename", r.get("filename")): r for r in target_rows}
    out: List[dict] = []

    shuffled = list(source_rows)
    rng.shuffle(shuffled)

    for rec in shuffled:
        tgt_rec = target_by_source.get(rec["filename"])
        if tgt_rec is None:
            continue

        row_sym = row_to_symbol(rec)
        objs = shape_objects(rec)
        candidates = []
        for a, b in itertools.combinations(objs, 2):
            if int(a["row"]) == int(b["row"]):
                continue
            if str(a["shape"]).lower() == str(b["shape"]).lower():
                continue
            if str(a["color"]).lower() == str(b["color"]).lower():
                continue
            sym_a = row_sym[int(a["row"])]
            sym_b = row_sym[int(b["row"])]
            candidates.append(
                {
                    "source": rec["filename"],
                    "target": tgt_rec["filename"],
                    "sym_a": sym_a,
                    "sym_b": sym_b,
                    "row_a": int(a["row"]),
                    "row_b": int(b["row"]),
                    "grid_col_a": int(a["grid_col"]),
                    "grid_col_b": int(b["grid_col"]),
                    "obj_a": object_label(a),
                    "obj_b": object_label(b),
                    "shape_a": str(a["shape"]).lower(),
                    "shape_b": str(b["shape"]).lower(),
                    "color_a": str(a["color"]).lower(),
                    "color_b": str(b["color"]).lower(),
                }
            )

        if not candidates:
            continue
        rng.shuffle(candidates)
        out.append(candidates[0])
        if len(out) >= max_pairs:
            break

    return out


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_condition_placements(pair: dict, src_rec: dict, tgt_rec: dict, condition: str) -> Tuple[List[dict], Dict[str, str], Dict[str, str]]:
    obj_a = find_source_object(src_rec, pair["sym_a"], int(pair["grid_col_a"]))
    obj_b = find_source_object(src_rec, pair["sym_b"], int(pair["grid_col_b"]))

    if condition == "standard":
        slot_a = find_target_slot(tgt_rec, pair["sym_a"], int(pair["grid_col_a"]))
        slot_b = find_target_slot(tgt_rec, pair["sym_b"], int(pair["grid_col_b"]))
        physical = {pair["sym_a"]: "A", pair["sym_b"]: "B"}
    elif condition == "swapped":
        slot_a = find_target_slot(tgt_rec, pair["sym_b"], int(pair["grid_col_a"]))
        slot_b = find_target_slot(tgt_rec, pair["sym_a"], int(pair["grid_col_b"]))
        physical = {pair["sym_a"]: "B", pair["sym_b"]: "A"}
    else:
        raise ValueError(f"Unknown condition: {condition}")

    placements = [
        {"donor_label": "A", "src_obj": obj_a, "tgt_slot": slot_a},
        {"donor_label": "B", "src_obj": obj_b, "tgt_slot": slot_b},
    ]
    bound = {pair["sym_a"]: "A", pair["sym_b"]: "B"}
    return placements, bound, physical


def score_candidates(
    model,
    processor,
    tgt_batch,
    layer_cache,
    src_positions: Sequence[int],
    tgt_positions: Sequence[int],
    layers_start: int,
    layers_end: int,
    candidates: Dict[str, str],
) -> Dict[str, float]:
    with patch_layers(
        model=model,
        layer_cache=layer_cache,
        src_positions=src_positions,
        tgt_positions=tgt_positions,
        layers_start=layers_start,
        layers_end=layers_end,
    ):
        return {name: string_logprob_with_batch(model, processor, tgt_batch, text) for name, text in candidates.items()}


def score_condition(
    model,
    processor,
    source_img: Image.Image,
    target_img: Image.Image,
    src_rec: dict,
    tgt_rec: dict,
    pair: dict,
    condition: str,
    colors: Sequence[str],
    shapes: Sequence[str],
    pad: int,
    layers_start: int,
    layers_end: int,
):
    row_symbols = symbol_order(src_rec)
    placements, bound_gold, physical_gold = build_condition_placements(pair, src_rec, tgt_rec, condition)
    query_syms = [pair["sym_a"], pair["sym_b"]]
    labels = {
        "A": {"shape": pair["shape_a"], "color": pair["color_a"]},
        "B": {"shape": pair["shape_b"], "color": pair["color_b"]},
    }

    canvas_src_w = float(src_rec.get("canvas_size_x") or src_rec.get("canvas_size") or source_img.size[0])
    canvas_src_h = float(src_rec.get("canvas_size_y") or src_rec.get("canvas_size") or source_img.size[1])
    canvas_tgt_w = float(tgt_rec.get("canvas_size_x") or tgt_rec.get("canvas_size") or target_img.size[0])
    canvas_tgt_h = float(tgt_rec.get("canvas_size_y") or tgt_rec.get("canvas_size") or target_img.size[1])

    task_results = {}
    for task, vocab in [("shape", shapes), ("color", colors)]:
        for query_sym in query_syms:
            system, prefix, suffix = build_attr_prompt(query_sym, row_symbols, task, vocab)
            src_batch = prepare_inputs(processor, source_img, system, prefix, suffix)
            tgt_batch = prepare_inputs(processor, target_img, system, prefix, suffix)

            vis_src = find_vision_tokens(processor, src_batch)
            vis_tgt = find_vision_tokens(processor, tgt_batch)
            if not vis_src or not vis_tgt:
                raise RuntimeError(f"Could not locate image tokens for task={task} query={query_sym} pair={pair}")

            htok, wtok = get_htok_wtok(src_batch, vis_src)
            htok_t, wtok_t = get_htok_wtok(tgt_batch, vis_tgt)
            if (htok, wtok) != (htok_t, wtok_t):
                raise RuntimeError(
                    f"Vision grid mismatch for {pair['source']} task={task} query={query_sym}: "
                    f"src={htok}x{wtok} tgt={htok_t}x{wtok_t}"
                )

            layer_cache = collect_src_hidden_states(model, src_batch, vis_src)
            src_concat = []
            tgt_concat = []
            for placement in placements:
                src_patch_ids = bbox_to_patch_ids_1b(
                    *bbox_from_obj(placement["src_obj"]),
                    canvas_w=canvas_src_w,
                    canvas_h=canvas_src_h,
                    htok=htok,
                    wtok=wtok,
                    pad=pad,
                )
                tgt_patch_ids = bbox_to_patch_ids_1b(
                    *bbox_from_obj(placement["tgt_slot"]),
                    canvas_w=canvas_tgt_w,
                    canvas_h=canvas_tgt_h,
                    htok=htok,
                    wtok=wtok,
                    pad=pad,
                )
                src_positions = patch_id_to_token_pos(src_patch_ids, vis_src)
                tgt_positions = patch_id_to_token_pos(tgt_patch_ids, vis_tgt)
                k = min(len(src_positions), len(tgt_positions))
                if k == 0:
                    raise RuntimeError(
                        "No overlapping patch tokens for "
                        f"{pair} condition={condition} task={task} query={query_sym}"
                    )
                src_concat.extend(src_positions[:k])
                tgt_concat.extend(tgt_positions[:k])

            candidates = {
                "A": labels["A"][task],
                "B": labels["B"][task],
                "NONE": NONE_LABEL,
            }
            scores = score_candidates(
                model=model,
                processor=processor,
                tgt_batch=tgt_batch,
                layer_cache=layer_cache,
                src_positions=src_concat,
                tgt_positions=tgt_concat,
                layers_start=layers_start,
                layers_end=layers_end,
                candidates=candidates,
            )
            task_results[(task, query_sym)] = {
                "scores": scores,
                "scores_ab": normalize_subset_logprobs(scores, ["A", "B"]),
            }

    per_query = []
    for query_sym in query_syms:
        shape_scores = task_results[("shape", query_sym)]["scores"]
        shape_scores_ab = task_results[("shape", query_sym)]["scores_ab"]
        color_scores = task_results[("color", query_sym)]["scores"]
        color_scores_ab = task_results[("color", query_sym)]["scores_ab"]

        object_scores = {
            "A": shape_scores["A"] + color_scores["A"],
            "B": shape_scores["B"] + color_scores["B"],
            "NONE": shape_scores["NONE"] + color_scores["NONE"],
        }
        object_scores_ab = normalize_subset_logprobs({"A": object_scores["A"], "B": object_scores["B"]}, ["A", "B"])
        pred_label = max(object_scores.items(), key=lambda kv: kv[1])[0]
        ab_pred_label = "A" if object_scores_ab["A"] >= object_scores_ab["B"] else "B"
        bound_label = bound_gold[query_sym]
        physical_label = physical_gold[query_sym]

        per_query.append(
            {
                "condition": condition,
                "query_symbol": query_sym,
                "pred_label": pred_label,
                "ab_pred_label": ab_pred_label,
                "bound_label": bound_label,
                "physical_label": physical_label,
                "bound_correct_full": int(pred_label == bound_label),
                "physical_correct_full": int(pred_label == physical_label),
                "bound_correct": int(ab_pred_label == bound_label),
                "physical_correct": int(ab_pred_label == physical_label),
                "none_pred": int(pred_label == "NONE"),
                "score_A": object_scores["A"],
                "score_B": object_scores["B"],
                "score_NONE": object_scores["NONE"],
                "score_A_ab": object_scores_ab["A"],
                "score_B_ab": object_scores_ab["B"],
                "margin_A_minus_B": object_scores["A"] - object_scores["B"],
                "margin_best_minus_none": max(object_scores["A"], object_scores["B"]) - object_scores["NONE"],
                "shape_lp_A": shape_scores["A"],
                "shape_lp_B": shape_scores["B"],
                "shape_lp_NONE": shape_scores["NONE"],
                "shape_pred_label": max(shape_scores.items(), key=lambda kv: kv[1])[0],
                "shape_ab_pred_label": "A" if shape_scores_ab["A"] >= shape_scores_ab["B"] else "B",
                "shape_bound_correct_full": int(max(shape_scores.items(), key=lambda kv: kv[1])[0] == bound_label),
                "shape_physical_correct_full": int(max(shape_scores.items(), key=lambda kv: kv[1])[0] == physical_label),
                "shape_bound_correct": int(("A" if shape_scores_ab["A"] >= shape_scores_ab["B"] else "B") == bound_label),
                "shape_physical_correct": int(("A" if shape_scores_ab["A"] >= shape_scores_ab["B"] else "B") == physical_label),
                "shape_lp_A_ab": shape_scores_ab["A"],
                "shape_lp_B_ab": shape_scores_ab["B"],
                "color_lp_A": color_scores["A"],
                "color_lp_B": color_scores["B"],
                "color_lp_NONE": color_scores["NONE"],
                "color_pred_label": max(color_scores.items(), key=lambda kv: kv[1])[0],
                "color_ab_pred_label": "A" if color_scores_ab["A"] >= color_scores_ab["B"] else "B",
                "color_bound_correct_full": int(max(color_scores.items(), key=lambda kv: kv[1])[0] == bound_label),
                "color_physical_correct_full": int(max(color_scores.items(), key=lambda kv: kv[1])[0] == physical_label),
                "color_bound_correct": int(("A" if color_scores_ab["A"] >= color_scores_ab["B"] else "B") == bound_label),
                "color_physical_correct": int(("A" if color_scores_ab["A"] >= color_scores_ab["B"] else "B") == physical_label),
                "color_lp_A_ab": color_scores_ab["A"],
                "color_lp_B_ab": color_scores_ab["B"],
            }
        )

    col_order = ["A", "B"] if condition == "standard" else ["B", "A"]

    color_mat = np.array(
        [
            [task_results[("color", pair["sym_a"])]["scores"][col_order[0]], task_results[("color", pair["sym_a"])]["scores"][col_order[1]]],
            [task_results[("color", pair["sym_b"])]["scores"][col_order[0]], task_results[("color", pair["sym_b"])]["scores"][col_order[1]]],
        ],
        dtype=float,
    )
    shape_mat = np.array(
        [
            [task_results[("shape", pair["sym_a"])]["scores"][col_order[0]], task_results[("shape", pair["sym_a"])]["scores"][col_order[1]]],
            [task_results[("shape", pair["sym_b"])]["scores"][col_order[0]], task_results[("shape", pair["sym_b"])]["scores"][col_order[1]]],
        ],
        dtype=float,
    )
    color_mat_ab = np.array(
        [
            [task_results[("color", pair["sym_a"])]["scores_ab"][col_order[0]], task_results[("color", pair["sym_a"])]["scores_ab"][col_order[1]]],
            [task_results[("color", pair["sym_b"])]["scores_ab"][col_order[0]], task_results[("color", pair["sym_b"])]["scores_ab"][col_order[1]]],
        ],
        dtype=float,
    )
    shape_mat_ab = np.array(
        [
            [task_results[("shape", pair["sym_a"])]["scores_ab"][col_order[0]], task_results[("shape", pair["sym_a"])]["scores_ab"][col_order[1]]],
            [task_results[("shape", pair["sym_b"])]["scores_ab"][col_order[0]], task_results[("shape", pair["sym_b"])]["scores_ab"][col_order[1]]],
        ],
        dtype=float,
    )
    return per_query, color_mat, shape_mat, color_mat_ab, shape_mat_ab


def draw_heatmaps(color_std, color_swap, shape_std, shape_swap, out_prefix: Path) -> None:
    if plt is None:
        return

    mats = [color_std, color_swap, shape_std, shape_swap]
    all_vals = np.concatenate([m.ravel() for m in mats]).astype(float)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    panels = [
        (axs[0, 0], color_std, "Standard Setting", "Color"),
        (axs[0, 1], color_swap, "Activation Swapped", "Color"),
        (axs[1, 0], shape_std, "", "Shape"),
        (axs[1, 1], shape_swap, "", "Shape"),
    ]

    image = None
    for ax, mat, title, ylabel in panels:
        image = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 1], [r"$o_{s_0}$", r"$o_{s_1}$"])
        ax.set_yticks([0, 1], [r"$s_0$", r"$s_1$"])
        ax.set_title(title, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(image, ax=axs, shrink=0.88)
    cbar.set_label("mean log-prob")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_task_pair_heatmap(std_mat: np.ndarray | None, swp_mat: np.ndarray | None, out_prefix: Path, task_label: str) -> None:
    if plt is None or std_mat is None or swp_mat is None:
        return

    all_vals = np.concatenate([std_mat.ravel(), swp_mat.ravel()]).astype(float)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    panels = [
        (axs[0], std_mat, "Standard Setting"),
        (axs[1], swp_mat, "Activation Swapped"),
    ]

    image = None
    for ax, mat, title in panels:
        image = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 1], [r"$o_{s_0}$", r"$o_{s_1}$"])
        ax.set_yticks([0, 1], [r"$s_0$", r"$s_1$"])
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(task_label, fontsize=14)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(image, ax=axs, shrink=0.92)
    cbar.set_label("mean log-prob")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def mean_task_matrix(rows: Sequence[dict], task: str, ab_only: bool) -> np.ndarray | None:
    if not rows:
        return None
    col_a = f"{task}_lp_A_ab" if ab_only else f"{task}_lp_A"
    col_b = f"{task}_lp_B_ab" if ab_only else f"{task}_lp_B"
    mats = []
    for r in rows:
        if "sym_a" not in r or "sym_b" not in r or "query_symbol" not in r:
            continue
        q = str(r["query_symbol"])
        sym_a = str(r["sym_a"])
        sym_b = str(r["sym_b"])
        if str(r.get("condition", "")) == "swapped":
            left_val = float(r[col_b])
            right_val = float(r[col_a])
        else:
            left_val = float(r[col_a])
            right_val = float(r[col_b])
        if q == sym_a:
            mats.append(np.array([[left_val, right_val], [np.nan, np.nan]], dtype=float))
        elif q == sym_b:
            mats.append(np.array([[np.nan, np.nan], [left_val, right_val]], dtype=float))
    if not mats:
        return None
    arr = np.stack(mats, axis=0)
    return np.nanmean(arr, axis=0)


def format_matrix(name: str, mat: np.ndarray | None) -> str:
    if mat is None:
        return f"{name}=NA"
    return (
        f"{name}="
        f"[[{mat[0,0]:.3f}, {mat[0,1]:.3f}], "
        f"[{mat[1,0]:.3f}, {mat[1,1]:.3f}]]"
    )


def summarize(rows: Sequence[dict], pairs_used: int) -> str:
    if not rows:
        return "No evaluated rows."

    std = [r for r in rows if r["condition"] == "standard"]
    swp = [r for r in rows if r["condition"] == "swapped"]
    std_shape_ok = [r for r in std if int(r.get("standard_pair_shape_bound_correct", 0)) == 1]
    std_color_ok = [r for r in std if int(r.get("standard_pair_color_bound_correct", 0)) == 1]
    swp_std_ok = [r for r in swp if int(r.get("eligible_standard_ab", 0)) == 1]
    swp_std_ok_full = [r for r in swp if int(r.get("eligible_standard_full", 0)) == 1]
    swp_std_ok_shape = [r for r in swp if int(r.get("eligible_standard_shape_ab", 0)) == 1]
    swp_std_ok_color = [r for r in swp if int(r.get("eligible_standard_color_ab", 0)) == 1]

    def avg(subset: Sequence[dict], key: str) -> float:
        return float(sum(float(r[key]) for r in subset)) / len(subset) if subset else 0.0

    mats = [
        format_matrix("standard_shape_ab_logprob_matrix", mean_task_matrix(std, "shape", ab_only=True)),
        format_matrix("standard_color_ab_logprob_matrix", mean_task_matrix(std, "color", ab_only=True)),
        format_matrix("swapped_shape_ab_logprob_matrix", mean_task_matrix(swp, "shape", ab_only=True)),
        format_matrix("swapped_color_ab_logprob_matrix", mean_task_matrix(swp, "color", ab_only=True)),
        format_matrix("standard_shape_ab_logprob_matrix_on_standard_shape_correct", mean_task_matrix(std_shape_ok, "shape", ab_only=True)),
        format_matrix("standard_color_ab_logprob_matrix_on_standard_color_correct", mean_task_matrix(std_color_ok, "color", ab_only=True)),
        format_matrix("swapped_shape_ab_logprob_matrix_on_standard_shape_correct", mean_task_matrix(swp_std_ok_shape, "shape", ab_only=True)),
        format_matrix("swapped_color_ab_logprob_matrix_on_standard_color_correct", mean_task_matrix(swp_std_ok_color, "color", ab_only=True)),
    ]

    return (
        f"pairs_used={pairs_used}\n"
        f"queries_total={len(rows)}\n"
        f"a_vs_b_heatmap_file=avg_logprob_heatmap_a_vs_b.png\n"
        f"a_vs_b_heatmap_pdf=avg_logprob_heatmap_a_vs_b.pdf\n"
        f"a_vs_b_filtered_heatmap_file=avg_logprob_heatmap_a_vs_b_on_standard_correct.png\n"
        f"a_vs_b_filtered_heatmap_pdf=avg_logprob_heatmap_a_vs_b_on_standard_correct.pdf\n"
        f"shape_subset_heatmap_file=shape_heatmap_a_vs_b_on_standard_shape_correct.png\n"
        f"color_subset_heatmap_file=color_heatmap_a_vs_b_on_standard_color_correct.png\n"
        f"standard_bound_acc={avg(std, 'bound_correct'):.3f}\n"
        f"standard_bound_acc_full={avg(std, 'bound_correct_full'):.3f}\n"
        f"standard_shape_bound_acc={avg(std, 'shape_bound_correct'):.3f}\n"
        f"standard_shape_bound_acc_full={avg(std, 'shape_bound_correct_full'):.3f}\n"
        f"standard_color_bound_acc={avg(std, 'color_bound_correct'):.3f}\n"
        f"standard_color_bound_acc_full={avg(std, 'color_bound_correct_full'):.3f}\n"
        f"standard_none_rate={avg(std, 'none_pred'):.3f}\n"
        f"swapped_bound_acc={avg(swp, 'bound_correct'):.3f}\n"
        f"swapped_bound_acc_full={avg(swp, 'bound_correct_full'):.3f}\n"
        f"swapped_shape_bound_acc={avg(swp, 'shape_bound_correct'):.3f}\n"
        f"swapped_shape_bound_acc_full={avg(swp, 'shape_bound_correct_full'):.3f}\n"
        f"swapped_color_bound_acc={avg(swp, 'color_bound_correct'):.3f}\n"
        f"swapped_color_bound_acc_full={avg(swp, 'color_bound_correct_full'):.3f}\n"
        f"swapped_physical_acc={avg(swp, 'physical_correct'):.3f}\n"
        f"swapped_physical_acc_full={avg(swp, 'physical_correct_full'):.3f}\n"
        f"swapped_none_rate={avg(swp, 'none_pred'):.3f}\n"
        f"swapped_mean_margin_A_minus_B={avg(swp, 'margin_A_minus_B'):.3f}\n"
        f"swapped_eval_pairs_after_standard_ab={len({int(r['pair_idx']) for r in swp_std_ok})}\n"
        f"swapped_eval_pairs_after_standard_shape_ab={len({int(r['pair_idx']) for r in swp_std_ok_shape})}\n"
        f"swapped_eval_pairs_after_standard_color_ab={len({int(r['pair_idx']) for r in swp_std_ok_color})}\n"
        f"swapped_bound_acc_on_standard_correct={avg(swp_std_ok, 'bound_correct'):.3f}\n"
        f"swapped_bound_acc_full_on_standard_correct={avg(swp_std_ok_full, 'bound_correct_full'):.3f}\n"
        f"swapped_shape_bound_acc_on_standard_shape_correct={avg(swp_std_ok_shape, 'shape_bound_correct'):.3f}\n"
        f"swapped_color_bound_acc_on_standard_color_correct={avg(swp_std_ok_color, 'color_bound_correct'):.3f}\n"
        + "\n".join(mats)
    )


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = load_meta(Path(args.source_metadata))
    target_rows = load_meta(Path(args.target_metadata))
    pairs = choose_pairs(source_rows, target_rows, max_pairs=args.max_pairs, seed=args.seed)
    write_csv(output_root / "pairs.csv", pairs)

    if args.dry_run:
        summary = f"pairs_built={len(pairs)}\ndry_run=1\n"
        (output_root / "summary.txt").write_text(summary, encoding="utf-8")
        print(summary.strip())
        print(f"[write] {output_root / 'pairs.csv'}")
        return

    if not pairs:
        raise SystemExit("No valid object pairs found.")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    source_by_name = {r["filename"]: r for r in source_rows}
    target_by_source = {r.get("source_filename", r.get("filename")): r for r in target_rows}
    colors = sorted({str(o["color"]).lower() for r in source_rows for o in shape_objects(r)})
    shapes = sorted({str(o["shape"]).lower() for r in source_rows for o in shape_objects(r)})

    rows_out: List[dict] = []
    color_std_accum = []
    color_swap_accum = []
    shape_std_accum = []
    shape_swap_accum = []
    color_std_ab_accum = []
    color_swap_ab_accum = []
    shape_std_ab_accum = []
    shape_swap_ab_accum = []

    for idx, pair in enumerate(pairs, start=1):
        src_rec = source_by_name[pair["source"]]
        tgt_rec = target_by_source[pair["source"]]
        source_path = Path(args.source_dir) / pair["source"]
        target_path = Path(args.target_dir) / pair["target"]
        if not source_path.exists() or not target_path.exists():
            continue

        source_img = Image.open(source_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        print(f"[{idx}/{len(pairs)}] {pair['source']}  {pair['sym_a']}:{pair['grid_col_a']}  <->  {pair['sym_b']}:{pair['grid_col_b']}")

        standard_rows, color_std, shape_std, color_std_ab, shape_std_ab = score_condition(
            model=model,
            processor=processor,
            source_img=source_img,
            target_img=target_img,
            src_rec=src_rec,
            tgt_rec=tgt_rec,
            pair=pair,
            condition="standard",
            colors=colors,
            shapes=shapes,
            pad=args.pad,
            layers_start=args.layers_start,
            layers_end=args.layers_end,
        )
        swapped_rows, color_swap, shape_swap, color_swap_ab, shape_swap_ab = score_condition(
            model=model,
            processor=processor,
            source_img=source_img,
            target_img=target_img,
            src_rec=src_rec,
            tgt_rec=tgt_rec,
            pair=pair,
            condition="swapped",
            colors=colors,
            shapes=shapes,
            pad=args.pad,
            layers_start=args.layers_start,
            layers_end=args.layers_end,
        )

        color_std_accum.append(color_std)
        color_swap_accum.append(color_swap)
        shape_std_accum.append(shape_std)
        shape_swap_accum.append(shape_swap)
        color_std_ab_accum.append(color_std_ab)
        color_swap_ab_accum.append(color_swap_ab)
        shape_std_ab_accum.append(shape_std_ab)
        shape_swap_ab_accum.append(shape_swap_ab)

        standard_pair_bound_correct = int(all(r["bound_correct"] == 1 for r in standard_rows))
        standard_pair_bound_correct_full = int(all(r["bound_correct_full"] == 1 for r in standard_rows))
        standard_pair_shape_bound_correct = int(all(r["shape_bound_correct"] == 1 for r in standard_rows))
        standard_pair_color_bound_correct = int(all(r["color_bound_correct"] == 1 for r in standard_rows))

        for row in standard_rows:
            rows_out.append(
                {
                    "pair_idx": idx - 1,
                    "source": pair["source"],
                    "target": pair["target"],
                    "sym_a": pair["sym_a"],
                    "sym_b": pair["sym_b"],
                    "grid_col_a": pair["grid_col_a"],
                    "grid_col_b": pair["grid_col_b"],
                    "obj_a": pair["obj_a"],
                    "obj_b": pair["obj_b"],
                    "eligible_standard_ab": 1,
                    "eligible_standard_full": 1,
                    "eligible_standard_shape_ab": 1,
                    "eligible_standard_color_ab": 1,
                    "standard_pair_bound_correct": standard_pair_bound_correct,
                    "standard_pair_bound_correct_full": standard_pair_bound_correct_full,
                    "standard_pair_shape_bound_correct": standard_pair_shape_bound_correct,
                    "standard_pair_color_bound_correct": standard_pair_color_bound_correct,
                    **row,
                }
            )

        for row in swapped_rows:
            rows_out.append(
                {
                    "pair_idx": idx - 1,
                    "source": pair["source"],
                    "target": pair["target"],
                    "sym_a": pair["sym_a"],
                    "sym_b": pair["sym_b"],
                    "grid_col_a": pair["grid_col_a"],
                    "grid_col_b": pair["grid_col_b"],
                    "obj_a": pair["obj_a"],
                    "obj_b": pair["obj_b"],
                    "eligible_standard_ab": standard_pair_bound_correct,
                    "eligible_standard_full": standard_pair_bound_correct_full,
                    "eligible_standard_shape_ab": standard_pair_shape_bound_correct,
                    "eligible_standard_color_ab": standard_pair_color_bound_correct,
                    "standard_pair_bound_correct": standard_pair_bound_correct,
                    "standard_pair_bound_correct_full": standard_pair_bound_correct_full,
                    "standard_pair_shape_bound_correct": standard_pair_shape_bound_correct,
                    "standard_pair_color_bound_correct": standard_pair_color_bound_correct,
                    **row,
                }
            )

    if not rows_out:
        raise SystemExit("No evaluation rows produced.")

    avg_color_std = np.stack(color_std_accum, axis=0).mean(axis=0)
    avg_color_swap = np.stack(color_swap_accum, axis=0).mean(axis=0)
    avg_shape_std = np.stack(shape_std_accum, axis=0).mean(axis=0)
    avg_shape_swap = np.stack(shape_swap_accum, axis=0).mean(axis=0)
    avg_color_std_ab = np.stack(color_std_ab_accum, axis=0).mean(axis=0)
    avg_color_swap_ab = np.stack(color_swap_ab_accum, axis=0).mean(axis=0)
    avg_shape_std_ab = np.stack(shape_std_ab_accum, axis=0).mean(axis=0)
    avg_shape_swap_ab = np.stack(shape_swap_ab_accum, axis=0).mean(axis=0)

    np.savez(
        str(output_root / "avg_logprob_mats.npz"),
        color_standard=avg_color_std,
        color_swapped=avg_color_swap,
        shape_standard=avg_shape_std,
        shape_swapped=avg_shape_swap,
        color_standard_ab=avg_color_std_ab,
        color_swapped_ab=avg_color_swap_ab,
        shape_standard_ab=avg_shape_std_ab,
        shape_swapped_ab=avg_shape_swap_ab,
        pairs_used=np.array([len(color_std_accum)], dtype=int),
    )
    np.savez(
        str(output_root / "avg_logprob_mats_a_vs_b.npz"),
        color_standard_ab=avg_color_std_ab,
        color_swapped_ab=avg_color_swap_ab,
        shape_standard_ab=avg_shape_std_ab,
        shape_swapped_ab=avg_shape_swap_ab,
        pairs_used=np.array([len(color_std_accum)], dtype=int),
    )

    draw_heatmaps(avg_color_std, avg_color_swap, avg_shape_std, avg_shape_swap, output_root / "avg_logprob_heatmap")
    draw_heatmaps(avg_color_std_ab, avg_color_swap_ab, avg_shape_std_ab, avg_shape_swap_ab, output_root / "avg_logprob_heatmap_ab")
    draw_heatmaps(avg_color_std_ab, avg_color_swap_ab, avg_shape_std_ab, avg_shape_swap_ab, output_root / "avg_logprob_heatmap_a_vs_b")

    std_rows = [r for r in rows_out if r["condition"] == "standard"]
    swp_rows = [r for r in rows_out if r["condition"] == "swapped"]
    std_shape_ok = [r for r in std_rows if int(r.get("standard_pair_shape_bound_correct", 0)) == 1]
    std_color_ok = [r for r in std_rows if int(r.get("standard_pair_color_bound_correct", 0)) == 1]
    swp_std_ok_shape = [r for r in swp_rows if int(r.get("eligible_standard_shape_ab", 0)) == 1]
    swp_std_ok_color = [r for r in swp_rows if int(r.get("eligible_standard_color_ab", 0)) == 1]

    std_color_ok_mat = mean_task_matrix(std_color_ok, "color", ab_only=True)
    swp_std_ok_color_mat = mean_task_matrix(swp_std_ok_color, "color", ab_only=True)
    std_shape_ok_mat = mean_task_matrix(std_shape_ok, "shape", ab_only=True)
    swp_std_ok_shape_mat = mean_task_matrix(swp_std_ok_shape, "shape", ab_only=True)

    if all(m is not None for m in (std_color_ok_mat, swp_std_ok_color_mat, std_shape_ok_mat, swp_std_ok_shape_mat)):
        draw_heatmaps(
            std_color_ok_mat,
            swp_std_ok_color_mat,
            std_shape_ok_mat,
            swp_std_ok_shape_mat,
            output_root / "avg_logprob_heatmap_a_vs_b_on_standard_correct",
        )

    draw_task_pair_heatmap(
        std_shape_ok_mat,
        swp_std_ok_shape_mat,
        output_root / "shape_heatmap_a_vs_b_on_standard_shape_correct",
        "Shape",
    )
    draw_task_pair_heatmap(
        std_color_ok_mat,
        swp_std_ok_color_mat,
        output_root / "color_heatmap_a_vs_b_on_standard_color_correct",
        "Color",
    )

    write_csv(output_root / "results.csv", rows_out)
    summary = summarize(rows_out, pairs_used=len(color_std_accum))
    (output_root / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"[write] {output_root / 'results.csv'}")
    print(f"[write] {output_root / 'summary.txt'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
