#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-image pipeline: Qwen2.5-VL generation + per-token attention-to-visual

This script:
- Loads Qwen2.5-VL-7B-Instruct (local dir if exists; else HuggingFace id).
- For each job (a folder of images), and for each image:
    * Builds a chat turn: [image + question].
    * Generates an answer deterministically.
    * Extracts attention: For each generated token, records how much the last
      query position attends to visual tokens (image patches), per LAYER
      (optionally subsampled).
    * Saves JSON next to the image: <stem>.json

Notes:
- Set LAYER_STRIDE=2 or 3 to subsample layers.
- Set PATCH_STRIDE=2 or 3 to subsample visual tokens.
- Reduce MAX_NEW_TOKENS to shorten generations.
- FAST_MODE=True tries to capture attentions directly from `generate()`.
"""

import argparse
import os
import json
from typing import Tuple, List, Dict, Any, Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ==================== GLOBAL CONFIG ==================== #
MODEL_DIR       = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS  = 512
IMG_EXTS        = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
VERBOSE         = True

FAST_MODE       = True
LAYER_STRIDE    = 1
PATCH_STRIDE    = 1
MAX_PIXELS      = 512*28*28
# ======================================================= #

# ---------------- Utility functions ---------------- #
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    def process_vision_info(messages: List[dict]):
        """Minimal image loader if qwen_vl_utils is missing."""
        imgs, sizes = [], []
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "image":
                    path = part["image"]
                    img = Image.open(path).convert("RGB")
                    imgs.append(img)
                    sizes.append((img.height, img.width))
        return imgs, sizes


def load_model_and_processor(model_dir: str):
    """Load model and processor (local path if exists, else HuggingFace id)."""
    use_id = model_dir if os.path.isdir(model_dir) else "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        use_id,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(use_id)
    return model, processor


def build_messages(image_path: str, question: str):
    """Build a chat message with image and question."""
    img_part = {"type": "image", "image": image_path}
    return [{
        "role": "user",
        "content": [img_part, {"type": "text", "text": question}],
    }]


def derive_grid(processor, image_inputs, model):
    """Derive merged-patch grid geometry."""
    meta = processor.image_processor(images=image_inputs)
    thw = meta["image_grid_thw"].to("cpu").numpy().squeeze(0)
    patch_size = int(getattr(model.config.vision_config, "patch_size", 14))
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    grid_h = int(thw[1] // merge_size)
    grid_w = int(thw[2] // merge_size)
    resized_h = int(thw[1] * patch_size)
    resized_w = int(thw[2] * patch_size)
    merged_patch_px = int(patch_size * merge_size)
    return grid_h, grid_w, merged_patch_px, resized_w, resized_h


def find_image_span(processor, input_ids: torch.Tensor):
    """Locate [vision_start, vision_end) tokens in sequence."""
    tok = processor.tokenizer
    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")
    seq = input_ids[0].tolist()
    pos = seq.index(vs_id) + 1
    pos_end = seq.index(ve_id)
    return pos, pos_end


def collect_stop_ids(tokenizer) -> set:
    """Collect stop/eos tokens."""
    stop_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)
    return stop_ids


def list_images(root: str, recursive: bool = False):
    """List images in directory."""
    hits: List[str] = []
    if not recursive:
        if not os.path.isdir(root):
            return hits
        for fn in sorted(os.listdir(root)):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and fn.lower().endswith(IMG_EXTS):
                hits.append(p)
        return hits
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                hits.append(os.path.join(d, fn))
    hits.sort()
    return hits


def _subsample_indices(n: int, stride: int):
    return list(range(0, n, max(1, stride)))


def _normalize_generate_attns(attns_any):
    """Normalize attentions from generate()."""
    if attns_any is None:
        return None
    if isinstance(attns_any, (list, tuple)) and len(attns_any) > 0:
        step0 = attns_any[0]
        if isinstance(step0, (list, tuple)):
            try:
                _ = step0[0].shape
                return [list(layers) for layers in attns_any]
            except Exception:
                pass
    return None

# ---------------- Core per-image ---------------- #
def process_one_image(
    image_path: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    question: str,
    max_new_tokens: int,
    output_dir: Optional[str] = None,
):
    """Run full pipeline on one image."""
    device = next(model.parameters()).device
    tok = processor.tokenizer
    stop_ids = collect_stop_ids(tok)

    # Build input
    messages = build_messages(image_path, question)
    image_inputs, _ = process_vision_info(messages)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[chat_text], images=image_inputs, padding=True, return_tensors="pt")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Vision grid
    grid_h, grid_w, merged_patch_px, resized_w, resized_h = derive_grid(processor, image_inputs, model)
    if VERBOSE:
        print(f"    grid={grid_h}x{grid_w} resized={resized_w}x{resized_h}", flush=True)

    # Image token span
    pos, pos_end = find_image_span(processor, batch["input_ids"])
    span_len = pos_end - pos
    patch_indices = _subsample_indices(span_len, PATCH_STRIDE)

    eos_id = tok.eos_token_id
    im_end_id = None
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        pass
    terminators = [t for t in [eos_id, im_end_id] if t is not None and t != -1]
    if tok.pad_token_id is None and eos_id is not None:
        tok.pad_token_id = eos_id

    generated_token_ids: List[int] = []
    step_token_labels: List[str] = []
    per_token_patch_layers: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_step_layer_peaks: List[torch.Tensor] = []
    used_fast_path = False

    # Try fast path
    try:
        if FAST_MODE:
            with torch.no_grad():
                gen_out = model.generate(
                    **batch,
                    do_sample=False,
                    eos_token_id=(terminators if len(terminators) > 1 else terminators[0]),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
            answer = processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
            input_len = batch["input_ids"].shape[1]
            cont_ids = gen_out.sequences[0, input_len:]
            generated_token_ids = cont_ids.tolist()
            for tid in generated_token_ids:
                raw = tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if raw == "":
                    raw = tok.convert_ids_to_tokens([tid])[0]
                step_token_labels.append(raw)
            steps_layers = _normalize_generate_attns(getattr(gen_out, "attentions", None))
            if steps_layers is None:
                raise RuntimeError("Attentions not recognized.")
            num_layers = len(steps_layers[0])
            layer_indices = _subsample_indices(num_layers, LAYER_STRIDE)
            for s, layer_list in enumerate(steps_layers, start=1):
                layer_peaks = torch.zeros(len(layer_indices), dtype=torch.float32)
                token_key = f"{s:04d}:{step_token_labels[s-1]}" if s-1 < len(step_token_labels) else f"{s:04d}"
                per_token_patch_layers[token_key] = {}
                for p_global in patch_indices:
                    per_token_patch_layers[token_key][f"patch{p_global}"] = {}
                for idx_li, li in enumerate(layer_indices):
                    layer_attn = layer_list[li]
                    att_img = layer_attn[0, :, -1, pos:pos_end]
                    att_img = att_img[:, patch_indices]
                    peak = att_img.max().to(torch.float32).cpu()
                    layer_peaks[idx_li] = peak
                    head_max = att_img.max(dim=0).values.tolist()
                    for p_idx_local, val in enumerate(head_max):
                        p_global = patch_indices[p_idx_local]
                        per_token_patch_layers[token_key][f"patch{p_global}"][f"layer{li+1}"] = float(val)
                per_step_layer_peaks.append(layer_peaks)
            used_fast_path = True
    except Exception as e:
        if VERBOSE:
            print(f"    [info] FAST path unavailable ({e}); using replay.", flush=True)

    if not used_fast_path:
        return {"ok": False, "error": "Replay mode not implemented in this simplified script."}

    if len(per_step_layer_peaks) == 0:
        return {"ok": False, "error": "No decoding steps recorded."}

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = output_dir or os.path.dirname(image_path)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{stem}.json")

    json_payload = {
        "used_fast_path": used_fast_path,
        "image_path": image_path,
        "question": question,
        "generated_text": answer,
        "token_ids": generated_token_ids,
        "token_labels": step_token_labels,
        "per_token": per_token_patch_layers,
        "grid_hw": [int(grid_h), int(grid_w)],
        "resized_wh": [int(resized_w), int(resized_h)],
        "merged_patch_px": int(merged_patch_px),
        "layer_stride": int(LAYER_STRIDE),
        "patch_stride": int(PATCH_STRIDE),
        "max_new_tokens": int(max_new_tokens),
    }

    # 1. Convert the dictionary to a JSON formatted string
    json_string = json.dumps(json_payload, ensure_ascii=False, indent=2)

    # 2. Calculate the size of the JSON string in bytes (UTF-8 encoding)
    size_in_bytes = len(json_string.encode('utf-8'))

    # 3. Define the maximum allowed size (200 MB in bytes: 200 * 1024 * 1024)
    MAX_SIZE_BYTES = 200 * 1024 * 1024

    # 4. Check if the size exceeds the limit
    if size_in_bytes > MAX_SIZE_BYTES:
        size_in_mb = size_in_bytes / (1024 * 1024)
        if VERBOSE:
            print(f"    [SKIPPED] JSON size is {size_in_mb:.2f} MB (Exceeds 200MB limit). Not saving.", flush=True)
        # Return an indicator that it wasn't saved due to size
        return {"ok": False, "reason": "size_exceeded", "json_path": None}

    # 5. Save to disk if the size is acceptable
    with open(json_path, "w", encoding="utf-8") as jf:
        # Since we already converted it to a string, we use write() instead of json.dump()
        jf.write(json_string)

    if VERBOSE:
        print(f"    [SAVED] JSON -> {json_path}", flush=True)

    return {"ok": True, "json_path": json_path}

# ---------------- Job runner and main ---------------- #
def run_job(job: Dict[str, Any], model, processor):
    """Execute one job on all images in a folder."""
    images_dir = job["ROOT"]
    recursive  = bool(job.get("RECURSIVE", False))
    question   = job["QUESTION"]
    name       = job.get("NAME", os.path.basename(images_dir).strip() or "job")
    output_dir = job.get("OUTPUT_DIR")

    image_paths = list_images(images_dir, recursive=recursive)
    if not image_paths:
        print(f"[{name}] No images found in: {images_dir}")
        return

    print(f"\n=== JOB: {name} ===")
    print(f"Dir  : {images_dir}")
    print(f"Imgs : {len(image_paths)}")

    ok_cnt = 0
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{name}] [{idx}/{len(image_paths)}] START -> {img_path}", flush=True)
        try:
            rec = process_one_image(
                image_path=img_path,
                model=model,
                processor=processor,
                question=question,
                max_new_tokens=int(job.get("MAX_NEW_TOKENS", MAX_NEW_TOKENS)),
                output_dir=output_dir,
            )
            if rec.get("ok"):
                ok_cnt += 1
                print(f"[{name}] [{idx}/{len(image_paths)}] DONE -> JSON saved", flush=True)
            else:
                print(f"[{name}] [{idx}/{len(image_paths)}] FAIL -> {rec.get('error','unknown error')}", flush=True)
        except Exception as e:
            print(f"[{name}] [{idx}/{len(image_paths)}] ERROR -> {e}", flush=True)

    print(f"[{name}] Finished. OK={ok_cnt} FAIL={len(image_paths)-ok_cnt}")


def main():
    """Collect generation attention for one experimental condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DIR)
    parser.add_argument("--images", required=True, help="Folder containing condition images.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--name", default="attention")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model)
    print("Model & processor loaded.")
    run_job(
        {
            "NAME": args.name,
            "ROOT": args.images,
            "QUESTION": args.question,
            "OUTPUT_DIR": args.output_dir,
            "MAX_NEW_TOKENS": args.max_new_tokens,
            "RECURSIVE": args.recursive,
        },
        model,
        processor,
    )


if __name__ == "__main__":
    main()
