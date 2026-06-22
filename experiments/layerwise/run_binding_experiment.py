#!/usr/bin/env python3
import os, sys, json, re, math, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------------- defaults ----------------
DEF_MODEL_DIR    = os.environ.get("MODEL_DIR")
DEF_DATA_DIR     = None
DEF_META_JSON    = None
DEF_OUT_ROOT     = "attn_binding_runs"
DEF_LAYERS_SPEC  = "all"      # "all" | "mid" | "start:end" (e.g., "0:31")
DEF_PAD          = 1
DEF_GREEDY       = True
DEF_MAX_NEW_TOKENS = 16
SYMS_ORDER       = ["&","$","#","@"]      # only for stable legend ordering
SYM_TOKEN        = {"&":"AMP","$":"DOLLAR","#":"HASH","@":"AT"}

# ---------------- utils: token grid ----------------
def best_2d_factors(n: int):
    best=(1,n); gap=n
    for h in range(1,int(math.sqrt(n))+1):
        if n%h==0:
            w=n//h
            if abs(h-w)<gap:
                best=(h,w); gap=abs(h-w)
    return best

def find_image_span(input_ids, tokenizer):
    ids = input_ids[0].tolist()
    for s_name, e_name in [
        ("<|vision_start|>", "<|vision_end|>"),
        ("<|image_start|>", "<|image_end|>"),
    ]:
        try:
            s_id = tokenizer.convert_tokens_to_ids(s_name)
            e_id = tokenizer.convert_tokens_to_ids(e_name)
            s_idx = ids.index(s_id)
            e_idx = ids.index(e_id)
            if e_idx > s_idx:
                return s_idx + 1, e_idx
        except Exception:
            pass
    raise RuntimeError("Could not find image token span")

def build_batch(processor, img_path, prompt, device):
    image = Image.open(img_path).convert("RGB")
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    b = processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to(device) for k,v in b.items()}

def decode_new_only(gen_ids, batch, tokenizer):
    start = batch["input_ids"].shape[1]
    return tokenizer.decode(gen_ids[0][start:], skip_special_tokens=True).strip()

# ---------------- dataset helpers (by SYMBOL) ----------------
def _norm_shape(s):
    s = str(s).lower().strip()
    for v in ("square","circle","triangle","diamond"):
        if s == v or v in s:
            return v
    return s

def symbol_shape_map(rec):
    """
    Prefer objects[*] : {symbol -> shape}. Fallback to top-level fields if needed.
    """
    m = {}
    for o in rec.get("objects", []):
        sym = o.get("symbol")
        shp = o.get("shape")
        if sym is None or shp is None: continue
        m[str(sym)] = _norm_shape(shp)
    if len(m) >= 2:
        return m

    # fallback
    rows = rec.get("object_rows", [])
    shapes = rec.get("object_assignment", [])
    arr = rec.get("symbol_arrangement", ["&","$","#","@"])
    for i, r in enumerate(rows):
        try:
            rr = int(r)
            if i < len(shapes) and 0 <= rr < len(arr):
                m[arr[rr]] = _norm_shape(shapes[i])
        except Exception:
            pass
    return m

def symbols_present(rec):
    m = symbol_shape_map(rec)
    # keep a stable ordering for prints/plots
    return sorted(m.keys(), key=lambda s: SYMS_ORDER.index(s) if s in SYMS_ORDER else 99)

def find_all_symbol_reversed_pairs(meta):
    """
    Group by the SET of occupied symbols (2 per sample).
    Pair ANY contexts whose symbol->shape map is reversed across those two symbols.
    Arrangement/rows may differ; only symbol identity matters.
    """
    by_symset = defaultdict(list)
    for r in meta:
        sy = tuple(sorted(symbols_present(r)))
        if len(sy)==2:
            by_symset[sy].append(r)

    pairs=[]
    for symset, recs in by_symset.items():
        # bucket by signature: (shape_at_sym0, shape_at_sym1)
        sigmap = defaultdict(list)
        for r in recs:
            m = symbol_shape_map(r)
            sigmap[(m[symset[0]], m[symset[1]])].append(r)

        visited=set()
        for sig, lstA in list(sigmap.items()):
            if sig in visited: continue
            rev = (sig[1], sig[0])
            visited.add(sig)
            if rev in sigmap: visited.add(rev)
            if rev not in sigmap:
                continue
            lstB = sigmap[rev]
            if sig == rev:
                for i in range(len(lstA)):
                    for j in range(i+1, len(lstA)):
                        pairs.append((lstA[i], lstA[j], symset))
            else:
                for a in lstA:
                    for b in lstB:
                        pairs.append((a,b, symset))
    return pairs

# ---------------- patch localization by SYMBOL ----------------
def _grid_and_canvas(rec):
    # dataset-side grid/canvas (used only if we must map its coords into model grid)
    gx = int(rec.get("grid_size_x", 11))
    gy = int(rec.get("grid_size_y", 12))
    cw = int(rec.get("canvas_size_x", 308))
    ch = int(rec.get("canvas_size_y", 336))
    return gx, gy, cw, ch

def _center_patch_idx_for_symbol(rec, sym, H, W):
    """
    Try several ways to find the center patch index (0..H*W-1) for symbol 'sym'.
    """
    for o in rec.get("objects", []):
        if str(o.get("symbol")) != sym: continue

        if "patch_index" in o:
            try:
                idx = int(o["patch_index"])
                if 0 <= idx < H*W: return idx
            except Exception:
                pass

        cx=cy=None
        if "center_position" in o and isinstance(o["center_position"], (list,tuple)) and len(o["center_position"])>=2:
            cx, cy = float(o["center_position"][0]), float(o["center_position"][1])
        elif "paste_position" in o and "size" in o:
            try:
                px, py = o["paste_position"]
                sz = float(o["size"])
                cx, cy = float(px) + sz/2, float(py) + sz/2
            except Exception:
                pass
        if cx is not None and cy is not None:
            gx, gy, cw, ch = _grid_and_canvas(rec)
            if gx>0 and gy>0 and cw>0 and ch>0:
                col = int(np.clip(np.floor(cx / cw * gx), 0, gx-1))
                row = int(np.clip(np.floor(cy / ch * gy), 0, gy-1))
                # remap dataset grid to model grid proportionally
                col = int(np.clip(np.floor(col / max(1,gx) * W), 0, W-1))
                row = int(np.clip(np.floor(row / max(1,gy) * H), 0, H-1))
                return row*W + col

        if "row" in o:
            try:
                dataset_row = int(o["row"])
                row = int(np.clip(np.floor(dataset_row / max(1,int(rec.get("grid_size_y", 12))) * H), 0, H-1))
                col = W//2
                return row*W + col
            except Exception:
                pass

    # fallback: symbol_arrangement + object_rows
    arr = rec.get("symbol_arrangement", ["&","$","#","@"])
    obj_rows = rec.get("object_rows", [])
    phys_row = None
    for r in obj_rows:
        try:
            rr = int(r)
            if 0 <= rr < len(arr) and arr[rr] == sym:
                phys_row = rr
                break
        except Exception:
            pass
    if phys_row is not None:
        row = int(np.clip(np.floor((phys_row + 0.5) * (H/4.0)), 0, H-1))
        col = W//2
        return row*W + col
    return None

def patch_idxs_around(center_idx, H, W, pad=1):
    r = center_idx // W
    c = center_idx %  W
    rr = range(max(0, r-pad), min(H-1, r+pad)+1)
    cc = range(max(0, c-pad), min(W-1, c+pad)+1)
    return np.array([ri*W + ci for ri in rr for ci in cc], dtype=int)

# ---------------- intervention hooks ----------------
class CapturePreAttn:
    def __init__(self, modules, when_T):
        self.cache={}; self.when_T=when_T; self.handles=[]
        for li, attn in modules:
            self.handles.append(attn.register_forward_pre_hook(self._mk(li), with_kwargs=True))
    def _mk(self, li):
        def pre(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is not None and x.dim()==3 and x.shape[1]==self.when_T:
                self.cache[li] = x[0].detach().clone()
        return pre
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()

class OverwritePreAttn:
    def __init__(self, modules, when_T, idxs_tgt_1d, src_captures, idxs_src_1d):
        self.when_T=when_T
        self.idxt=torch.as_tensor(idxs_tgt_1d, dtype=torch.long)
        self.idxs=torch.as_tensor(idxs_src_1d, dtype=torch.long)
        self.src=src_captures
        self.enabled=False
        self.handles=[]
        for li, attn in modules:
            self.handles.append(attn.register_forward_pre_hook(self._mk(li), with_kwargs=True))
    def _mk(self, li):
        def pre(mod, args, kwargs):
            if not self.enabled or li not in self.src: return
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is None or x.dim()!=3 or x.shape[1]!=self.when_T: return
            with torch.no_grad():
                k = min(self.idxt.numel(), self.idxs.numel())
                if k==0: return
                x2=x.clone(); xs=self.src[li]
                x2[0, self.idxt[:k]] = xs[self.idxs[:k]].to(x2.dtype).to(x2.device)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"]=x2; return args, kwargs
            a=list(args); a[0]=x2; return tuple(a), kwargs
        return pre
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()

def get_attn_modules(model):
    layers = model.model.language_model.layers
    return [(i, layers[i].self_attn) for i in range(len(layers))]

def pick_layers(n_layers, spec):
    if spec == "all": return list(range(n_layers))
    if spec == "mid":
        a = n_layers//3; b = (2*n_layers)//3
        return list(range(a,b))
    m = re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', str(spec) or "")
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        lo=max(0,lo); hi=min(n_layers-1,hi)
        return list(range(lo,hi+1))
    return pick_layers(n_layers, "mid")

# ---------------- attention replay (token -> image, heads reduced) ----------------
@torch.inference_mode()
def replay_full_token2patch(model, processor, base_batch, gen_ids_full, reduce="max"):
    tok = processor.tokenizer
    pos, pos_end = find_image_span(base_batch["input_ids"], tok)
    T_prompt = base_batch["input_ids"].shape[1]
    cont_ids = gen_ids_full[0, T_prompt:].tolist()

    seq_ids = base_batch["input_ids"].clone()
    attn_mask = base_batch.get("attention_mask", None)

    if torch.cuda.is_available():
        sdpa = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    else:
        from contextlib import nullcontext
        sdpa = nullcontext()

    out_steps=[]
    with sdpa:
        dry = model(**base_batch, output_attentions=True, use_cache=False, return_dict=True)
        assert dry.attentions is not None
        L = len(dry.attentions)

        for tid in cont_ids:
            nxt = torch.tensor([[tid]], device=seq_ids.device, dtype=seq_ids.dtype)
            seq_ids = torch.cat([seq_ids, nxt], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones_like(nxt)], dim=1)

            out = model(
                input_ids=seq_ids,
                attention_mask=attn_mask,
                pixel_values=base_batch.get("pixel_values"),
                image_grid_thw=base_batch.get("image_grid_thw"),
                output_attentions=True, use_cache=False, return_dict=True,
            )
            layers=[]
            for li in range(L):
                A = out.attentions[li][0, :, -1, :]  # [heads, K]
                red = A.mean(dim=0) if reduce=="mean" else A.max(dim=0).values
                vec = red[pos:pos_end].to(torch.float32).detach().cpu().numpy()  # [Nimg]
                layers.append(vec)
            out_steps.append(layers)  # [L][Nimg]
    return out_steps  # [T][L][Nimg]

# ---------------- NEW: per-head replay to build layer×head heatmap (no head reduction) ----------------
@torch.inference_mode()
def replay_head_diff(model, processor, base_batch, gen_ids_full, idxs_prompt, idxs_other):
    """
    EXACT same score as in the notebook but per-head:
    For each layer L and head H:
        diff[L,H] = Σ_steps ( Σ_{3×3 prompt} attn  −  Σ_{3×3 other} attn )

    Additionally returns visual attention mass for weighting:
        head_vis_mass[L,H]  = Σ_steps ( Σ_{all visual tokens} attn )
        layer_vis_mass[L]   = Σ_steps ( Σ_{heads} Σ_{all visual tokens} attn )
    """
    tok = processor.tokenizer
    pos, pos_end = find_image_span(base_batch["input_ids"], tok)
    T_prompt = base_batch["input_ids"].shape[1]
    cont_ids = gen_ids_full[0, T_prompt:].tolist()

    seq_ids = base_batch["input_ids"].clone()
    attn_mask = base_batch.get("attention_mask", None)

    if torch.cuda.is_available():
        sdpa = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    else:
        from contextlib import nullcontext
        sdpa = nullcontext()

    with sdpa:
        dry = model(**base_batch, output_attentions=True, use_cache=False, return_dict=True)
        assert dry.attentions is not None
        L = len(dry.attentions)
        Hheads = dry.attentions[0].shape[1]  # [B,H,Q,K]

        sums_prompt = np.zeros((L, Hheads), dtype=np.float64)
        sums_other  = np.zeros((L, Hheads), dtype=np.float64)
        head_vis_mass  = np.zeros((L, Hheads), dtype=np.float64)
        layer_vis_mass = np.zeros((L,), dtype=np.float64)

        for tid in cont_ids:
            nxt = torch.tensor([[tid]], device=seq_ids.device, dtype=seq_ids.dtype)
            seq_ids = torch.cat([seq_ids, nxt], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones_like(nxt)], dim=1)

            out = model(
                input_ids=seq_ids,
                attention_mask=attn_mask,
                pixel_values=base_batch.get("pixel_values"),
                image_grid_thw=base_batch.get("image_grid_thw"),
                output_attentions=True, use_cache=False, return_dict=True,
            )

            for li in range(L):
                A = out.attentions[li][0, :, -1, :]          # [H, K]
                Aimg = A[:, pos:pos_end]                      # [H, Nimg]
                # notebook diff terms (3×3 vs 3×3)
                pv = Aimg[:, idxs_prompt].sum(dim=1).to(torch.float64)  # [H]
                ov = Aimg[:, idxs_other ].sum(dim=1).to(torch.float64)  # [H]
                sums_prompt[li] += pv.cpu().numpy()
                sums_other [li] += ov.cpu().numpy()

                # visual-token mass for weighting
                hv = Aimg.sum(dim=1).to(torch.float64)         # per-head mass to ALL visual tokens
                head_vis_mass[li] += hv.cpu().numpy()
                layer_vis_mass[li] += float(hv.sum().cpu().numpy())  # sum over heads

    diff = (sums_prompt - sums_other).astype(np.float32)  # [L,H]
    return diff, head_vis_mass.astype(np.float32), layer_vis_mass.astype(np.float32)

# ---------------- prompt ----------------
def one_word_shape_prompt_for_symbol(sym):
    return (
        f'Scan the image using the symbols on the left (&, $, #, @) as row labels. some rows may have no objects.if there is no object say none\n'
        f'Question: What is the SHAPE in the row labeled "{sym}"?\n'
        f'one word, lowercase'
    )

# ---------------- one pair×symbol run ----------------
def run_pair_symbol_and_dump(
    model, processor, ctx_recv, ctx_donor, symbol, data_dir, H, W, pad, modules_for_hooks,
    greedy=True, max_new_tokens=DEF_MAX_NEW_TOKENS, out_dir=None
):
    """
    donor -> receiver intervention across the two occupied symbols (cross-swap).
    Save obj_blocks_mass.json (per-layer SUM over steps of mass on 3×3).
    Also print BASE/INT answers.

    Additionally, save per-head matrices:
      - head_diff.npy : [L,H]  Σsteps(Σ3×3_prompt - Σ3×3_other)
      - head_vis_mass.npy : [L,H]  Σsteps Σvisual attn
      - layer_vis_mass.npy: [L]    Σsteps Σheads Σvisual attn
    """
    device = next(model.parameters()).device
    prompt = one_word_shape_prompt_for_symbol(symbol)
    img_recv = Path(data_dir) / ctx_recv["filename"]
    img_donor = Path(data_dir) / ctx_donor["filename"]

    bR = build_batch(processor, img_recv, prompt, device)  # receiver
    bD = build_batch(processor, img_donor, prompt, device)  # donor
    tok = processor.tokenizer
    sR,eR = find_image_span(bR["input_ids"], tok)
    sD,eD = find_image_span(bD["input_ids"], tok)
    assert (eR-sR)==(eD-sD), "image grid mismatch"

    # which two symbols?
    occR = symbols_present(ctx_recv)
    if len(occR) != 2 or symbol not in occR:
        return
    other = occR[1] if occR[0]==symbol else occR[0]

    # capture donor caches
    T_src = bD["input_ids"].shape[1]
    cap = CapturePreAttn(modules_for_hooks, when_T=T_src)
    _ = model(**bD, use_cache=False)
    cap.close()

    # absolute token indices for symbol-centered 3x3 (receiver & donor)
    def abs_idxs(rec, sym, span_start):
        c = _center_patch_idx_for_symbol(rec, sym, H, W)
        if c is None: return np.array([], dtype=int)
        return patch_idxs_around(c, H, W, pad=pad) + span_start

    tgt_sym   = abs_idxs(ctx_recv, symbol, sR)
    tgt_other = abs_idxs(ctx_recv, other,  sR)
    src_sym   = abs_idxs(ctx_donor, symbol, sD)
    src_other = abs_idxs(ctx_donor, other,  sD)

    # cross-swap: recv.sym  <= donor.other,    recv.other <= donor.sym
    idxt = np.concatenate([tgt_sym, tgt_other])
    idxs = np.concatenate([src_other[:len(tgt_sym)], src_sym[:len(tgt_other)]])
    k = min(len(idxt), len(idxs))
    if k == 0:
        return
    idxt = idxt[:k]; idxs = idxs[:k]

    T_tgt = bR["input_ids"].shape[1]
    ow = OverwritePreAttn(modules_for_hooks, when_T=T_tgt, idxs_tgt_1d=idxt, src_captures=cap.cache, idxs_src_1d=idxs)

    gen_args = dict(max_new_tokens=max_new_tokens, do_sample=not greedy, temperature=0.0 if greedy else 0.7, top_p=1.0)

    # baseline & intervened (print)
    base_ids = model.generate(**bR, **gen_args)
    base_txt = decode_new_only(base_ids, bR, tok)
    print(f"    BASE: {base_txt}")
    ow.enabled = True
    int_ids  = model.generate(**bR, **gen_args)
    ow.enabled = False
    int_txt  = decode_new_only(int_ids, bR, tok)
    print(f"    INT : {int_txt}")

    # replay intervened and collect attention with max reduction across heads
    ow.enabled = True
    steps = replay_full_token2patch(model, processor, bR, int_ids, reduce="max")    # [T][L][Nimg]
    ow.enabled = False
    ow.close()
    if not steps:
        return

    # build 3×3 neighborhoods on RECEIVER geometry (prompt=physical symbol, other=the other)
    c_prompt = _center_patch_idx_for_symbol(ctx_recv, symbol, H, W)
    c_other  = _center_patch_idx_for_symbol(ctx_recv, other,  H, W)
    if c_prompt is None or c_other is None:
        return
    idxs_prompt = patch_idxs_around(c_prompt, H, W, pad=1)
    idxs_other  = patch_idxs_around(c_other,  H, W, pad=1)

    # Layer score: per-layer sum over generation steps of attention on the 3x3 block.
    L = len(steps[0])
    sums_prompt = np.zeros(L, dtype=np.float64)
    sums_other  = np.zeros(L, dtype=np.float64)
    for st in steps:
        for li in range(L):
            vec = np.asarray(st[li], dtype=np.float64)  # [Nimg]
            sums_prompt[li] += vec[idxs_prompt].sum()
            sums_other[li]  += vec[idxs_other].sum()

    # dump exactly the structure the notebook aggregator expects
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir/"base.txt").write_text(base_txt+"\n")
        (out_dir/"int.txt").write_text(int_txt+"\n")
        layer_keys = [f"layer{i}" for i in range(L)]
        jo = {
            "per_symbol": {
                symbol: {"per_layer_sum": {k: float(v) for k,v in zip(layer_keys, sums_prompt)}},
                other:  {"per_layer_sum": {k: float(v) for k,v in zip(layer_keys, sums_other )}},
            },
            "meta": {
                "prompt_symbol": symbol,
                "other_symbol":  other,
                "grid_h": H, "grid_w": W,
                "pad_for_blocks": 1,
                "heads_reduction": "max",
                "note": "per_layer_sum = sum over steps of sum(attn on 3×3 block)",
            }
        }
        (out_dir/"obj_blocks_mass.json").write_text(json.dumps(jo, indent=2), encoding="utf-8")

    # ---- per-head matrices (same score), plus visual mass for weighting ----
    # re-enable overwrite in case the model needs the same internal pathing during replay
    ow = OverwritePreAttn(modules_for_hooks, when_T=T_tgt, idxs_tgt_1d=idxt, src_captures=cap.cache, idxs_src_1d=idxs)
    ow.enabled = True
    head_diff, head_vis_mass, layer_vis_mass = replay_head_diff(
        model, processor, bR, int_ids, idxs_prompt, idxs_other
    )  # [L,H], [L,H], [L]
    ow.enabled = False
    ow.close()

    if out_dir is not None:
        np.save(out_dir/"head_diff.npy", head_diff)
        np.save(out_dir/"head_vis_mass.npy", head_vis_mass)
        np.save(out_dir/"layer_vis_mass.npy", layer_vis_mass)
        # small JSON peeks
        (out_dir/"head_diff.json").write_text(json.dumps(head_diff.tolist(), indent=2), encoding="utf-8")
        (out_dir/"head_vis_mass.json").write_text(json.dumps(head_vis_mass.tolist(), indent=2), encoding="utf-8")
        (out_dir/"layer_vis_mass.json").write_text(json.dumps(layer_vis_mass.tolist(), indent=2), encoding="utf-8")

    # return quick diff curve + per-head matrices for global aggregation
    quick_curve = (sums_prompt - sums_other).astype(np.float32)
    return quick_curve, head_diff, head_vis_mass, layer_vis_mass

# ---------------- main ----------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Binding experiment with attention-mass scoring and layer/head heatmaps.")
    ap.add_argument("--model_dir", default=DEF_MODEL_DIR)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--out_dir",   default=DEF_OUT_ROOT)
    ap.add_argument("--layers",    default=DEF_LAYERS_SPEC)
    ap.add_argument("--pad", type=int, default=DEF_PAD)
    ap.add_argument("--greedy", action="store_true", default=DEF_GREEDY)
    ap.add_argument("--max_new_tokens", type=int, default=DEF_MAX_NEW_TOKENS)
    ap.add_argument("--limit_pairs", type=int, default=0, help="limit number of valid pairs (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    if not args.model_dir:
        print("Please provide --model_dir or export MODEL_DIR.", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # load model ONCE with eager attention
    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    if hasattr(torch.backends, "cuda"):
        try: torch.backends.cuda.enable_flash_sdp(False)
        except Exception: pass
        try: torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception: pass
        try: torch.backends.cuda.enable_math_sdp(True)
        except Exception: pass

    # load metadata
    meta = json.loads(Path(args.meta_json).read_text())
    meta = meta if isinstance(meta, list) else list(meta.values())

    # find ALL valid pairs (symbol-reversal across the two occupied symbols)
    pairs = find_all_symbol_reversed_pairs(meta)
    print(f"[pairs] total valid symbol-reversed pairs: {len(pairs)}")
    if args.limit_pairs and len(pairs) > args.limit_pairs:
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[:args.limit_pairs]
        print(f"[pairs] limiting to {len(pairs)} (seed={args.seed})")

    if not pairs:
        print("No valid pairs. Check metadata (symbol/shape fields).")
        return

    # discover model grid once from first receiver
    first_recv = Path(args.data_dir)/pairs[0][0]["filename"]
    dummy = build_batch(processor, first_recv, "hi", next(model.parameters()).device)
    s0,e0 = find_image_span(dummy["input_ids"], processor.tokenizer)
    Nimg = e0 - s0
    H,W = best_2d_factors(Nimg)
    print(f"[grid] N_img={Nimg} => grid {H}x{W}")

    # layer hooks selection once
    modules_all = get_attn_modules(model)
    chosen_idx  = pick_layers(len(modules_all), args.layers)
    modules_for_hooks = [modules_all[i] for i in chosen_idx]
    print(f"[layers] using {len(chosen_idx)} layers: {chosen_idx}")

    # run all pairs
    quick_sym_diff = defaultdict(list)   # per-symbol 1D curves (prompt - other)
    head_diffs_all = []                  # list of [L,H]
    head_vis_masses_all = []             # list of [L,H]
    layer_vis_masses_all = []            # list of [L]

    for pi, (ctxA, ctxB, symset) in enumerate(pairs, 1):
        pair_dir = out_root / f"pair_{pi:03d}"
        print(f"\n===== Pair {pi}/{len(pairs)}: {Path(ctxA['filename']).stem}  ⟵  {Path(ctxB['filename']).stem}  (syms={list(symset)}) =====")
        for sym in symset:
            row_dir = pair_dir / f"row_{SYM_TOKEN.get(sym, sym)}"
            print(f"  [Q {sym}] one-word shape")
            try:
                res = run_pair_symbol_and_dump(
                    model, processor, ctxA, ctxB, sym,
                    data_dir=args.data_dir,
                    H=H, W=W,
                    pad=args.pad,
                    modules_for_hooks=modules_for_hooks,
                    greedy=args.greedy,
                    max_new_tokens=args.max_new_tokens,
                    out_dir=row_dir
                )
                if res is None: continue
                diff_curve, head_diff, head_vis_mass, layer_vis_mass = res
                if diff_curve is not None:
                    quick_sym_diff[sym].append(diff_curve)
                if head_diff is not None:
                    head_diffs_all.append(head_diff)
                if head_vis_mass is not None:
                    head_vis_masses_all.append(head_vis_mass)
                if layer_vis_mass is not None:
                    layer_vis_masses_all.append(layer_vis_mass)
            except Exception as e:
                print(f"[warn] sym={sym} failed: {e}")

        if pi % 20 == 0:
            print(f"  processed {pi}/{len(pairs)} pairs...")

    # --- Plot 1: Per-symbol curves (unchanged logic) ---
    if quick_sym_diff:
        fig = plt.figure(figsize=(8,5)); ax=plt.gca()
        for sym in sorted(quick_sym_diff.keys(), key=lambda s: SYMS_ORDER.index(s) if s in SYMS_ORDER else 99):
            arrs = quick_sym_diff[sym]
            if not arrs: continue
            Lmin = min(len(a) for a in arrs)
            A = np.stack([a[:Lmin] for a in arrs], axis=0)
            mu=A.mean(axis=0); sd=A.std(axis=0); x=np.arange(len(mu))
            ax.plot(x, mu, label=f"{sym} (n={A.shape[0]})")
            ax.fill_between(x, mu-sd, mu+sd, alpha=0.25)
        ax.set_xlabel("layer (selected indices)")
        ax.set_ylabel("prompt 3×3 mass − other 3×3 mass (Σ over steps)")
        ax.set_title(f"Attention mass; pad={args.pad}, layers={args.layers}, greedy={args.greedy}")
        ax.legend(); fig.tight_layout()
        out_png = Path(args.out_dir)/"mean_std_per_layer.png"
        fig.savefig(out_png, dpi=200)
        Path(args.out_dir, "scores.json").write_text(
            json.dumps({k:[v.tolist() for v in vs] for k,vs in quick_sym_diff.items()}, indent=2),
            encoding="utf-8"
        )
        print(f"[plot]  {out_png}")
        print(f"[stats] {Path(args.out_dir)/'scores.json'}")

        # --- Plot 2: Averaged over all symbols (single curve) ---
        all_diff_curves = []
        for sym_curves in quick_sym_diff.values():
            all_diff_curves.extend(sym_curves)
        if all_diff_curves:
            fig_agg = plt.figure(figsize=(8, 5)); ax_agg = plt.gca()
            Lmin_agg = min(len(a) for a in all_diff_curves)
            A_agg = np.stack([a[:Lmin_agg] for a in all_diff_curves], axis=0)
            mu_agg = A_agg.mean(axis=0); sd_agg = A_agg.std(axis=0); x_agg = np.arange(len(mu_agg))
            ax_agg.plot(x_agg, mu_agg, label=f"Average over all symbols (n={A_agg.shape[0]})")
            ax_agg.fill_between(x_agg, mu_agg - sd_agg, mu_agg + sd_agg, alpha=0.25)
            ax_agg.set_xlabel("layer (selected indices)")
            ax_agg.set_ylabel("prompt 3×3 mass − other 3×3 mass (Σ over steps)")
            ax_agg.set_title(f"Aggregated Attn. Mass; pad={args.pad}, layers={args.layers}, greedy={args.greedy}")
            ax_agg.legend(); ax_agg.grid(True, linestyle='--', alpha=0.6); fig_agg.tight_layout()
            out_png_agg = Path(args.out_dir) / "mean_std_AGGREGATED.png"
            fig_agg.savefig(out_png_agg, dpi=200)
            print(f"[plot]  {out_png_agg}")

    # --- Heatmap: unweighted mean (kept) ---
    if head_diffs_all:
        Lmin = min(hd.shape[0] for hd in head_diffs_all)
        Hmin = min(hd.shape[1] for hd in head_diffs_all)
        HD = np.stack([hd[:Lmin, :Hmin] for hd in head_diffs_all], axis=0)  # [N, L, H]
        mean_HD = HD.mean(axis=0)  # [L,H]
        np.save(Path(args.out_dir)/"head_diffs_all.npy", HD)
        (Path(args.out_dir)/"head_diffs_mean.json").write_text(json.dumps(mean_HD.tolist(), indent=2), encoding="utf-8")

        plt.figure(figsize=(10, 6))
        plt.imshow(mean_HD, aspect='auto')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("head")
        plt.ylabel("layer")
        plt.title(f"Heatmap (UNWEIGHTED): prompt 3×3 − other 3×3 (Σ over steps)")
        plt.tight_layout()
        out_heat = Path(args.out_dir)/"heatmap_layers_heads.png"
        plt.savefig(out_heat, dpi=200)
        print(f"[heatmap] {out_heat}")
    else:
        print("No per-head data accumulated — heatmap not created.")

    # --- NEW: Weighted heatmap = diff * head_vis_mass * layer_vis_mass ---
    if head_diffs_all and head_vis_masses_all and layer_vis_masses_all:
        # align shapes
        Lmin = min(hd.shape[0] for hd in head_diffs_all)
        Hmin = min(hd.shape[1] for hd in head_diffs_all)
        HD  = np.stack([hd[:Lmin, :Hmin] for hd in head_diffs_all], axis=0)                 # [N,L,H]
        HVM = np.stack([hvm[:Lmin, :Hmin] for hvm in head_vis_masses_all], axis=0)         # [N,L,H]
        LVM = np.stack([lvm[:Lmin] for lvm in layer_vis_masses_all], axis=0)               # [N,L]
        # broadcast layer mass across heads
        LVM_b = LVM[..., None]  # [N,L,1]
        # weighted per-run matrices
        W = HD * HVM * LVM_b    # [N,L,H]
        mean_W = W.mean(axis=0) # [L,H]

        np.save(Path(args.out_dir)/"head_vis_masses_all.npy", HVM)
        np.save(Path(args.out_dir)/"layer_vis_masses_all.npy", LVM)
        (Path(args.out_dir)/"head_diffs_mean_weighted.json").write_text(json.dumps(mean_W.tolist(), indent=2), encoding="utf-8")

        plt.figure(figsize=(10, 6))
        plt.imshow(mean_W, aspect='auto')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("head")
        plt.ylabel("layer")
        plt.title("Heatmap (WEIGHTED): (diff) × (head visual mass) × (layer visual mass)")
        plt.tight_layout()
        out_heat_w = Path(args.out_dir)/"heatmap_layers_heads_WEIGHTED.png"
        plt.savefig(out_heat_w, dpi=200)
        print(f"[heatmap] {out_heat_w}")
    else:
        print("Not enough data for weighted heatmap (need diff + head mass + layer mass).")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
