#!/usr/bin/env python3
import argparse, json, math, re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYMS = ["@", "$", "#", "&"]

# ------------------ utilities ------------------
def best_2d_factors(n: int):
    best=(1,n); gap=n
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            if abs(h-w) < gap:
                best=(h,w); gap=abs(h-w)
    return best

def find_image_span(input_ids, tok):
    ids = input_ids[0].tolist()
    pairs = [("<|vision_start|>", "<|vision_end|>"), ("<|image_start|>", "<|image_end|>")]
    for sname, ename in pairs:
        try:
            sid = tok.convert_tokens_to_ids(sname)
            eid = tok.convert_tokens_to_ids(ename)
            s = ids.index(sid); e = ids.index(eid)
            if e > s: return s+1, e
        except Exception:
            pass
    raise RuntimeError("could not find image span")

def build_batch(processor, img_path: Path, prompt: str, device: str):
    im = Image.open(img_path).convert("RGB")
    text = processor.apply_chat_template(
        [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}],
        add_generation_prompt=False, tokenize=False
    )
    pack = processor(text=[text], images=[im], return_tensors="pt")
    return {k: v.to(device) for k,v in pack.items()}

def bbox_from_shape_obj(obj):
    sz = float(obj.get("size", 28))
    if "paste_position" in obj:
        x0, y0 = [float(t) for t in obj["paste_position"][:2]]
        return x0, y0, x0+sz, y0+sz
    cx, cy = [float(t) for t in obj.get("center_position",[0,0])[:2]]
    return cx-sz/2, cy-sz/2, cx+sz/2, cy+sz/2

def bbox_to_rowscols(x0,y0,x1,y1, canvas_x, canvas_y, H_tok, W_tok, pad):
    tr = lambda yp: int(np.clip(np.floor(yp / canvas_y * H_tok), 0, H_tok-1))
    tc = lambda xp: int(np.clip(np.floor(xp / canvas_x * W_tok), 0, W_tok-1))
    r0, r1 = tr(min(y0,y1)), tr(max(y0,y1))
    c0, c1 = tc(min(x0,x1)), tc(max(x0,x1))
    r0 = max(0, r0-pad); r1 = min(H_tok-1, r1+pad)
    c0 = max(0, c0-pad); c1 = min(W_tok-1, c1+pad)
    rows = np.arange(r0, r1+1); cols = np.arange(c0, c1+1)
    return rows, cols

def rowscols_to_lin(rows, cols, W_tok):
    return np.array([r*W_tok + c for r in rows for c in cols], dtype=int)

# Try to extract symbol-at-row from metadata (covers multiple schemas)
def symbol_for_row(rec, row_int):
    # 1) explicit "objects" with type "symbol"
    for o in rec.get("objects", []):
        if o.get("type") == "symbol" and int(o.get("row")) == int(row_int):
            val = o.get("symbol")
            if val in SYMS: return val
    # 2) arrays "symbols" + "symbol_rows"
    syms = rec.get("symbols", None); srows = rec.get("symbol_rows", None)
    if syms and srows and len(syms) == len(srows):
        for s, r in zip(syms, srows):
            if int(r) == int(row_int) and s in SYMS:
                return s
    # 3) dict mapping rows to symbol (rare)
    m = rec.get("row_to_symbol") or rec.get("row2symbol")
    if isinstance(m, dict):
        s = m.get(str(row_int)) or m.get(row_int)
        if s in SYMS: return s
    return None

def get_shape_object_for_row(rec, row_int):
    for o in rec.get("objects", []):
        if o.get("type","shape") == "shape" and int(o.get("row")) == int(row_int):
            return o
    # fallback: first object with that row
    for o in rec.get("objects", []):
        if int(o.get("row")) == int(row_int):
            return o
    return None

# Hook: capture input-to-attention hidden_states (after LN, before QKV)
class CapturePreAttn:
    def __init__(self, modules, when_T):
        self.when_T = when_T
        self.cache = {}
        self.handles=[]
        self.funcs=[]
        for li, attn in modules:
            fn = self._make(li)
            h  = attn.register_forward_pre_hook(fn, with_kwargs=True)
            self.funcs.append(fn); self.handles.append(h)

    def _make(self, li):
        def prehook(mod, args, kwargs):
            x = kwargs.get("hidden_states", None)
            if x is None and args: x = args[0]
            if x is None or x.dim()!=3: return
            if self.when_T is not None and x.shape[1] != self.when_T: return
            # store batch0 [T,D]
            self.cache[li] = x[0].detach().clone().to("cpu")
        return prehook

    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear(); self.funcs.clear()

PROMPT = "Scan the image based on the symbols on the left."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--layers", default="all")  # 'all' | 'mid' | 'lo:hi'
    ap.add_argument("--max_images", type=int, default=999999)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=(torch.float16 if device=="cuda" else torch.float32),
        device_map=("auto" if device=="cuda" else None),
        trust_remote_code=True
    ).eval()

    # choose layers
    L = len(model.model.language_model.layers)
    def pick_layers(n_layers, spec):
        if spec=="all": return list(range(n_layers))
        if spec=="mid":
            a=n_layers//3; b=(2*n_layers)//3
            return list(range(a,b))
        m = re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', spec or "")
        if m:
            lo,hi = int(m.group(1)), int(m.group(2))
            lo=max(0,lo); hi=min(n_layers-1,hi)
            return list(range(lo,hi+1))
        a=n_layers//3; b=(2*n_layers)//3
        return list(range(a,b))
    layer_ids = pick_layers(L, args.layers)
    mods = [(i, model.model.language_model.layers[i].self_attn) for i in layer_ids]

    # metadata
    meta = json.loads(Path(args.metadata).read_text())
    recs = list(meta.values()) if isinstance(meta, dict) else meta
    rec_by_name = {r["filename"]: r for r in recs}

    # collect accumulators:
    # shape-conditioned means: obj_mean__<sym>__shape_<shape>
    # shape-agnostic:          obj_mean__<sym>
    shape_kinds = ["square","circle","triangle","diamond"]
    D = None
    sums_sym_shape = { (s,sh): None for s in SYMS for sh in shape_kinds }
    cnts_sym_shape = { (s,sh): 0    for s in SYMS for sh in shape_kinds }
    sums_sym_any   = { s: None for s in SYMS }
    cnts_sym_any   = { s: 0    for s in SYMS }

    paths = sorted(Path(args.data_dir).glob(args.pattern))[:args.max_images]
    n_ok = 0

    for i, p in enumerate(paths, 1):
        rec = rec_by_name.get(p.name)
        if rec is None: continue

        # per-image prep
        b = build_batch(processor, p, PROMPT, next(model.parameters()).device)
        T = b["input_ids"].shape[1]
        cap = CapturePreAttn(mods, when_T=T)
        with torch.no_grad():
            _ = model(**b, use_cache=False)  # capture once
        cap.close()

        s, e = find_image_span(b["input_ids"], processor.tokenizer)
        N_img = e - s
        H_tok, W_tok = best_2d_factors(N_img)
        canvas_x = int(rec.get("canvas_size_x", rec.get("canvas_size", 336)))
        canvas_y = int(rec.get("canvas_size_y", rec.get("canvas_size", 336)))

        # iterate each of the 4 rows present in this image
        # we’ll need the row’s symbol AND the shape object at that row
        # rows to try: union of object rows and symbol rows if present
        object_rows = sorted({int(o.get("row")) for o in rec.get("objects", []) if o.get("row") is not None})
        for row in object_rows:
            sym = symbol_for_row(rec, row)
            if sym not in SYMS:
                continue
            obj = get_shape_object_for_row(rec, row)
            if obj is None: continue
            shape = str(obj.get("shape","")).lower()
            if shape not in shape_kinds: continue

            # locate object patches for this row
            x0,y0,x1,y1 = bbox_from_shape_obj(obj)
            rows, cols = bbox_to_rowscols(x0,y0,x1,y1, canvas_x, canvas_y, H_tok, W_tok, pad=args.pad)
            idxs = np.unique(rowscols_to_lin(rows, cols, W_tok) + s)  # absolute token positions

            # pull hidden at each chosen layer, mean over tokens for this object
            vecs = []
            for (li, _) in mods:
                h = cap.cache[li] if li in cap.cache else None
                if h is None:
                    # shouldn’t happen; skip object
                    vecs = []
                    break
                # h: [T, D]
                obj_mean = h[idxs].mean(dim=0)  # [D]
                if D is None: D = int(obj_mean.shape[0])
                vecs.append(obj_mean.unsqueeze(0))  # [1,D]
            if not vecs:
                continue
            v = torch.cat(vecs, dim=0).cpu().numpy()  # [Lsel, D]

            # accumulate (shape-conditioned)
            k = (sym, shape)
            if sums_sym_shape[k] is None:
                sums_sym_shape[k] = v.copy()
            else:
                sums_sym_shape[k] += v
            cnts_sym_shape[k] += 1

            # accumulate (shape-agnostic)
            if sums_sym_any[sym] is None:
                sums_sym_any[sym] = v.copy()
            else:
                sums_sym_any[sym] += v
            cnts_sym_any[sym] += 1

        n_ok += 1
        if (i % 20) == 0:
            print(f"[progress] {i}/{len(paths)} images processed…")

    # finalize means
    Lsel = len(layer_ids)
    def finalize(arr, cnt):
        if arr is None or cnt == 0: return None
        return (arr / max(cnt,1)).astype(np.float32)

    obj_mean_shape = {}
    for s in SYMS:
        for sh in shape_kinds:
            m = finalize(sums_sym_shape[(s,sh)], cnts_sym_shape[(s,sh)])
            if m is not None:
                obj_mean_shape[f"obj_mean__{s}__shape_{sh}"] = m

    obj_mean_any = {}
    for s in SYMS:
        m = finalize(sums_sym_any[s], cnts_sym_any[s])
        if m is not None:
            obj_mean_any[f"obj_mean__{s}"] = m

    out = {
        "layers": np.array(layer_ids, dtype=np.int32),
        "hidden_size": np.array([D if D is not None else -1], dtype=np.int32),
        "sym_keys": np.array(SYMS),
        "shape_keys": np.array(shape_kinds),
    }
    out.update(obj_mean_shape)
    out.update(obj_mean_any)

    np.savez(args.out_npz, **out)
    print(f"[done] wrote {args.out_npz}")
    # small report
    for s in SYMS:
        for sh in shape_kinds:
            c = cnts_sym_shape[(s,sh)]
            if c:
                print(f"  {s} / {sh}: n={c}")
        c = cnts_sym_any[s]
        print(f"  {s} / any-shape: n={c}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
