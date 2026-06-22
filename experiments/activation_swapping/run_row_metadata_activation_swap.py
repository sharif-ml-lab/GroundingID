#!/usr/bin/env python3
"""Run row-level activation swaps using image metadata.

The script maps source and target rows to image-token spans, captures source
hidden states before self-attention, and overwrites the corresponding target
states during generation.

Example:
python -u experiments/activation_swapping/run_row_metadata_activation_swap.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --metadata "$META" \
  --source "$SRC_FN" \
  --target "$TGT_FN" \
  --row_pairs "4->4" \
  --pad 1 \
  --layers all \
  --greedy \
  --prompt_file "$PROMPT" \
  --out_dir "$OUT"
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# --------------------- metadata adapters (custom schemas) ---------------------

def get_canvas_size(rec):
    cs = rec.get("canvas_size", None)
    if cs is not None: return int(cs)
    cx = int(rec.get("canvas_size_x", 0) or 0)
    cy = int(rec.get("canvas_size_y", 0) or 0)
    if cx and cy: return max(cx, cy)
    return 336

def obj_top_left_and_size(obj):
    if "position" in obj and "size" in obj:
        return obj["position"], int(obj["size"])
    cp = obj.get("center_position", None)
    sz = obj.get("size", None)
    if cp is not None and sz is not None:
        cx, cy = float(cp[0]), float(cp[1])
        half = float(sz)/2.0
        return [int(cx - half), int(cy - half)], int(sz)
    pp = obj.get("paste_position", None)
    if pp is not None and sz is not None:
        return [int(pp[0]), int(pp[1])], int(sz)
    raise KeyError("No usable bbox info on object")

def pick_obj_by_physical_row(rec, row_idx0):
    objs = rec["objects"]
    def y_of(o):
        if "center_position" in o: return float(o["center_position"][1])
        if "position" in o: return float(o["position"][1])
        if "paste_position" in o: return float(o["paste_position"][1])
        return 0.0
    objs_sorted = sorted(objs, key=y_of)
    if row_idx0<0 or row_idx0>=len(objs_sorted):
        raise IndexError("row_idx0 out of range")
    return objs_sorted[row_idx0]
import numpy as np
import torch
from PIL import Image

# --------------------- Utilities ---------------------


def best_2d_factors(n: int):
    """Finds the two factors of n that are closest to each other."""
    best = (1, n)
    gap = n
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            if abs(h - w) < gap:
                best = (h, w)
                gap = abs(h - w)
    return best


def load_meta(meta_path):
    """Loads metadata from a .json or .npy file."""
    p = Path(meta_path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=True).tolist()
    raise ValueError("Metadata must be .json or .npy")


def find_image_span(input_ids, tokenizer):
    """Finds the start and end indices of image tokens in the input_ids."""
    ids = input_ids[0].tolist()
    # Check for different special token pairs for compatibility
    pairs = [
        ("<|vision_start|>", "<|vision_end|>"),
        ("<|image_start|>", "<|image_end|>"),
    ]
    for s_name, e_name in pairs:
        try:
            s_id = tokenizer.convert_tokens_to_ids(s_name)
            e_id = tokenizer.convert_tokens_to_ids(e_name)
            s_idx = ids.index(s_id)
            e_idx = ids.index(e_id)
            if e_idx > s_idx:
                return s_idx + 1, e_idx
        except Exception:
            pass
    raise RuntimeError("Could not find image token span in input_ids")


def bbox_to_rowscols(x0, y0, x1, y1, canvas_size, H_tok, W_tok, pad):
    """Converts a bounding box to token grid row and column indices."""
    tr = lambda yp: int(np.clip(np.floor(yp / canvas_size * H_tok), 0, H_tok - 1))
    tc = lambda xp: int(np.clip(np.floor(xp / canvas_size * W_tok), 0, W_tok - 1))
    r0, r1 = tr(y0), tr(y1)
    c0, c1 = tc(x0), tc(x1)
    r0 = max(0, min(r0, r1) - pad)
    r1 = min(H_tok - 1, max(r0, r1) + pad)
    c0 = max(0, min(c0, c1) - pad)
    c1 = min(W_tok - 1, max(c0, c1) + pad)
    rows = np.arange(r0, r1 + 1)
    cols = np.arange(c0, c1 + 1)
    return rows, cols


def rowscols_to_lin(rows, cols, W_tok):
    """Converts row and column indices to a flattened 1D array of indices."""
    return np.array([r * W_tok + c for r in rows for c in cols], dtype=int)


def build_list_prompt():
    """Builds the default textual prompt for the model."""
    return (
        "Look at the image. There are four rows labeled by a number each.\n"
        "Scan based on the numbers that exist in the image.\n"
        "Write EXACTLY four lines, one per row, using only lowercase color and shape words.\n"
        "Format:\n"
        "row<label>: <color> <shape>\n"
    )


def _find_row_object(rec, row_idx0):
    """Finds the object corresponding to a given row index within a metadata record."""
    objs = rec.get("objects") or rec.get("rows") or []
    if not isinstance(objs, list) or not objs:
        return None, "no-objects"

    # Try explicit integer 'row' fields (auto-detect base 0/1)
    ints = []
    for o in objs:
        try:
            ints.append(int(o.get("row")))
        except (ValueError, TypeError):
            pass
    if ints:
        base = min(ints)  # 0 or 1
        target = row_idx0 + base
        for o in objs:
            try:
                if int(o.get("row")) == target:
                    return o, f"row-field(base={base})"
            except (ValueError, TypeError):
                pass

    # Fallback: sort by Y coordinate (top->bottom) if present
    def y_of(o):
        pos = o.get("position") or o.get("center") or [0, 0]
        return pos[1] if isinstance(pos, (list, tuple)) and len(pos) > 1 else 0

    objs_sorted = sorted(objs, key=y_of)
    if 0 <= row_idx0 < len(objs_sorted):
        return objs_sorted[row_idx0], "y-sort"
    return None, "no-match"


def _bbox_from_obj_or_fallback(obj, row_idx0, canvas_size):
    """Extracts a bounding box from an object, with fallbacks."""
    if isinstance(obj, dict):
        if "position" in obj and "size" in obj:
            x, y = obj["position"]
            sz = obj["size"]
            w, h = (sz[0], sz[1]) if isinstance(sz, (list, tuple)) and len(sz) >= 2 else (sz, sz)
            return float(x), float(y), float(x) + float(w), float(y) + float(h)
        if "bbox" in obj:
            b = obj["bbox"]
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x0, y0, a, b2 = map(float, b[:4])
                return (x0, y0, a, b2) if a > x0 and b2 > y0 else (x0, y0, x0 + a, y0 + b2)
        if all(k in obj for k in ("x", "y", "w", "h")):
            x0, y0, w, h = float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])
            return x0, y0, x0 + w, y0 + h
        if all(k in obj for k in ("left", "top", "width", "height")):
            x0, y0, w, h = (float(obj["left"]), float(obj["top"]), float(obj["width"]), float(obj["height"]))
            return x0, y0, x0 + w, y0 + h

    # Uniform fallback: 4 equal-height horizontal bands
    rows = 4
    y0 = row_idx0 * (canvas_size / rows)
    y1 = (row_idx0 + 1) * (canvas_size / rows) - 1
    return 0.0, y0, float(canvas_size - 1), y1


def idxs_for_physical_row(rec, row_idx0, canvas_size, H_tok, W_tok, pad, span_start):
    """Gets the final 1D token indices for a given physical row."""
    obj, how = _find_row_object(rec, row_idx0)
    x0, y0, x1, y1 = _bbox_from_obj_or_fallback(obj, row_idx0, canvas_size)
    rows, cols = bbox_to_rowscols(x0, y0, x1, y1, canvas_size, H_tok, W_tok, pad=pad)
    ids_local = rowscols_to_lin(rows, cols, W_tok)
    print(f"[row-pick] row{row_idx0 + 1} via {how}, bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})")
    return np.unique(ids_local + span_start)


def decode_new_only(gen_ids, batch, processor):
    """Decodes only the newly generated tokens, skipping the prompt."""
    start = batch["input_ids"].shape[1]
    return processor.tokenizer.decode(gen_ids[0][start:], skip_special_tokens=True).strip()


def _extract_id(name):
    """Extracts a sequence of 3 or more digits from a filename."""
    m = re.search(r"(\d{3,})", str(name))
    return m.group(1) if m else None


def _find_record(meta_vals, filename):
    """Finds a metadata record by filename, with fallbacks."""
    for r in meta_vals:
        if r.get("filename") == filename:
            return r
    ident = _extract_id(filename)
    if ident:
        for r in meta_vals:
            fn = str(r.get("filename", ""))
            if ident in fn:
                print(f"[meta] Resolved '{filename}' -> '{fn}' (by id {ident})")
                return r
    stem = Path(filename).stem
    for r in meta_vals:
        fn = str(r.get("filename", ""))
        if Path(fn).stem.endswith(stem):
            print(f"[meta] Resolved '{filename}' -> '{fn}' (by stem)")
            return r
    raise StopIteration(f"Could not find record for '{filename}'")


def infer_canvas_size(rec, data_dir, fallback_name=None):
    """Infers the canvas size from metadata or by reading the image file."""
    if isinstance(rec, dict) and "canvas_size" in rec:
        return int(rec["canvas_size"])
    for nm in (rec.get("filename"), fallback_name):
        if nm:
            try:
                with Image.open(Path(data_dir) / nm) as im:
                    w, h = im.size
                return int(max(w, h))
            except Exception:
                pass
    return 1024


# --------------------- Activation hooks ---------------------


class CapturePreAttn:
    """Captures hidden states before attention modules via forward pre-hooks."""
    def __init__(self, modules, when_T):
        self.handles = []
        self.cache = {}
        self.when_T = when_T
        for li, attn_module in modules:
            h = attn_module.register_forward_pre_hook(self._make_hook(li), with_kwargs=True)
            self.handles.append(h)

    def _make_hook(self, layer_idx):
        def pre_hook(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is not None and x.dim() == 3 and x.shape[1] == self.when_T:
                self.cache[layer_idx] = x[0].detach().clone()
        return pre_hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class OverwritePreAttn:
    """Overwrites hidden states before attention modules via forward pre-hooks."""
    def __init__(self, modules, when_T, idxs_tgt_1d, src_captures, idxs_src_1d):
        self.when_T = when_T
        self.idxt = torch.as_tensor(idxs_tgt_1d, dtype=torch.long)
        self.idxs = torch.as_tensor(idxs_src_1d, dtype=torch.long)
        self.src = src_captures
        self.enabled = False
        self.handles = []
        for li, attn_module in modules:
            h = attn_module.register_forward_pre_hook(self._make_hook(li), with_kwargs=True)
            self.handles.append(h)

    def _make_hook(self, layer_idx):
        def pre_hook(mod, args, kwargs):
            if not self.enabled:
                return
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is None or x.dim() != 3 or x.shape[1] != self.when_T:
                return
            if layer_idx not in self.src:
                return

            with torch.no_grad():
                k = min(self.idxt.numel(), self.idxs.numel())
                if k == 0:
                    return
                x_clone = x.clone()
                src_activations = self.src[layer_idx]
                x_clone[0, self.idxt[:k]] = src_activations[self.idxs[:k]].to(x_clone.dtype).to(x_clone.device)

            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = x_clone
                return args, kwargs
            else:
                new_args = list(args)
                new_args[0] = x_clone
                return tuple(new_args), kwargs
        return pre_hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def get_attn_modules(model):
    """Gets all self-attention modules from the model's language model layers."""
    model_layers = model.model.language_model.layers
    return [(i, model_layers[i].self_attn) for i in range(len(model_layers))]


def pick_layers(n_layers, spec):
    """Parses a layer specification string (e.g., 'all', 'mid', '10:20')."""
    if spec == "all":
        return list(range(n_layers))
    if spec == "mid":
        a = n_layers // 3
        b = (2 * n_layers) // 3
        return list(range(a, b))
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", spec or "")
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        lo = max(0, lo)
        hi = min(n_layers - 1, hi)
        return list(range(lo, hi + 1))
    # Default fallback if spec is invalid
    print(f"[Warning] Invalid layer spec '{spec}', defaulting to 'mid'.")
    return pick_layers(n_layers, "mid")


def build_batch(processor, img_path, prompt, device):
    """Builds a batch for the model from an image and a text prompt."""
    image = Image.open(img_path).convert("RGB")
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


# --------------------- Main Execution ---------------------


def main():
    """Main function to run the model intervention experiment."""
    ap = argparse.ArgumentParser(description="Run vision-language model interventions.")
    ap.add_argument("--model_dir", required=True, help="Path to the model directory.")
    ap.add_argument("--data_dir", required=True, help="Path to the image data directory.")
    ap.add_argument("--metadata", required=True, help="Path to the metadata (.json or .npy).")
    ap.add_argument("--source", required=True, help="Filename of the source image.")
    ap.add_argument("--target", required=True, help="Filename of the target image.")
    ap.add_argument("--row_pairs", default="", help="Physical row swaps, e.g., '4->2,2->4'.")
    ap.add_argument("--pairs", default="4->2,2->4", help="Label-based swaps.")
    ap.add_argument("--label_order", default="1,2,3,4", help="Order of labels for --pairs.")
    ap.add_argument("--pad", type=int, default=2, help="Padding around bounding boxes in token space.")
    ap.add_argument("--layers", default="mid", help="Layers to intervene on: 'all', 'mid', or 'start:end'.")
    ap.add_argument("--out_dir", required=True, help="Directory to save outputs.")
    ap.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    ap.add_argument("--prompt_file", default="", help="Path to a custom text prompt file.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if device == "cuda" else None
    # torch_dtype = torch.float16 if device == "cuda" else torch.float32

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # --- Model and Processor Loading ---
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        temperature=0.01,
    ).eval()

    # --- Prompt and Metadata ---
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else build_list_prompt()
    meta = load_meta(args.metadata)
    meta_vals = list(meta.values()) if isinstance(meta, dict) else meta
    rec_src = _find_record(meta_vals, args.source)
    rec_tgt = _find_record(meta_vals, args.target)

    # --- Prepare Batches and Image Grid Info ---
    dev = next(model.parameters()).device
    b_src = build_batch(processor, Path(args.data_dir) / args.source, prompt, dev)
    b_tgt = build_batch(processor, Path(args.data_dir) / args.target, prompt, dev)

    s0, e0 = find_image_span(b_src["input_ids"], processor.tokenizer)
    s1, e1 = find_image_span(b_tgt["input_ids"], processor.tokenizer)
    assert (e0 - s0) == (e1 - s1), "Image token grids must have the same size."

    N_img = e0 - s0
    H_tok, W_tok = best_2d_factors(N_img)
    print(f"[src] span=[{s0},{e0}) N_img={N_img} grid={H_tok}x{W_tok}")
    print(f"[tgt] span=[{s1},{e1}) N_img={N_img} grid={H_tok}x{W_tok}")

    cs_src = infer_canvas_size(rec_src, args.data_dir, fallback_name=args.source)
    cs_tgt = infer_canvas_size(rec_tgt, args.data_dir, fallback_name=args.target)

    # --- Layer Selection ---
    all_modules = get_attn_modules(model)
    chosen_indices = pick_layers(len(all_modules), args.layers)
    modules_for_hooks = [all_modules[i] for i in chosen_indices]
    print(f"[layers] Intervening in {len(chosen_indices)} layers: {chosen_indices}")

    # --- 1. Capture Source Activations ---
    T_src = b_src["input_ids"].shape[1]
    cap = CapturePreAttn(modules_for_hooks, when_T=T_src)
    with torch.no_grad():
        _ = model(**b_src, use_cache=False)
    cap.close()

    # --- 2. Build Swap Indices ---
    swaps = []
    if args.row_pairs.strip():
        for p in args.row_pairs.split(","):
            a, b = p.split("->")
            rs, rt = int(a) - 1, int(b) - 1
            ids_src = idxs_for_physical_row(rec_src, rs, cs_src, H_tok, W_tok, args.pad, s0)
            ids_tgt = idxs_for_physical_row(rec_tgt, rt, cs_tgt, H_tok, W_tok, args.pad, s1)
            k = min(len(ids_src), len(ids_tgt))
            swaps.append((ids_tgt[:k], ids_src[:k]))
            print(f"[pair] (physical) src row{rs + 1} -> tgt row{rt + 1} : k={k}")
    else:
        order = [int(x) for x in args.label_order.split(",")]
        lab2row = {lab - 1: i for i, lab in enumerate(order)}

        def idxs_for(rec, lab, span_start, cs):
            r0 = lab2row[lab]
            return idxs_for_physical_row(rec, r0, cs, H_tok, W_tok, args.pad, span_start)

        for p in args.pairs.split(","):
            a, b = p.split("->")
            lab_s, lab_t = int(a) - 1, int(b) - 1
            ids_src = idxs_for(rec_src, lab_s, s0, cs_src)
            ids_tgt = idxs_for(rec_tgt, lab_t, s1, cs_tgt)
            k = min(len(ids_src), len(ids_tgt))
            swaps.append((ids_tgt[:k], ids_src[:k]))
            print(f"[pair] (label) {lab_s + 1} -> {lab_t + 1} : k={k}")

    tgt_concat = np.concatenate([a for a, _ in swaps]) if swaps else np.array([], dtype=int)
    src_concat = np.concatenate([b for _, b in swaps]) if swaps else np.array([], dtype=int)

    # --- 3. Run Baseline and Intervention ---
    T_tgt = b_tgt["input_ids"].shape[1]
    ow = OverwritePreAttn(
        modules_for_hooks,
        when_T=T_tgt,
        idxs_tgt_1d=tgt_concat,
        src_captures=cap.cache,
        idxs_src_1d=src_concat,
    )

    gen_args = dict(
        max_new_tokens=64,
        do_sample=not args.greedy,
        temperature=0.7,
        top_p=1.0,
    )
    if args.greedy:
        gen_args["temperature"] = 0.0

    print("\n=== BASELINE ===")
    base_ids = model.generate(**b_tgt, **gen_args)
    base_txt = decode_new_only(base_ids, b_tgt, processor)
    print(base_txt)

    print("\n=== INTERVENED ===")
    ow.enabled = True
    int_ids = model.generate(**b_tgt, **gen_args)
    int_txt = decode_new_only(int_ids, b_tgt, processor)
    print(int_txt)
    ow.close()

    # --- Save Results ---
    (out_dir / "baseline.txt").write_text(base_txt + "\n")
    (out_dir / "intervened.txt").write_text(int_txt + "\n")
    result_summary = f"=== BASELINE ===\n{base_txt}\n\n=== INTERVENED ===\n{int_txt}\n"
    (out_dir / "result.txt").write_text(result_summary)

    np.savez(
        str(out_dir / "verify.npz"),
        layers=np.array(chosen_indices, dtype=int),
        tgt_idxs=tgt_concat,
        src_idxs=src_concat,
        T_src=np.array([T_src]),
        T_tgt=np.array([T_tgt]),
    )
    print(f"\n[verify] Wrote verification data to {out_dir / 'verify.npz'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
