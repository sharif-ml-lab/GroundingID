#!/usr/bin/env python3
import argparse, json, math, re, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------------- utils ----------------
def best_2d_factors(n:int):
    best=(1,n); gap=n
    for h in range(1,int(math.sqrt(n))+1):
        if n%h==0:
            w=n//h
            if abs(h-w)<gap: best=(h,w); gap=abs(h-w)
    return best

def load_meta(p):
    p=Path(p)
    if p.suffix==".json": return json.loads(p.read_text())
    if p.suffix==".npy":  return np.load(p, allow_pickle=True).tolist()
    raise ValueError("metadata must be .json or .npy")

def find_image_span(input_ids, tok):
    ids=input_ids[0].tolist()
    for s,e in [("<|vision_start|>","<|vision_end|>"), ("<|image_start|>","<|image_end|>")]:
        try:
            sid=tok.convert_tokens_to_ids(s); eid=tok.convert_tokens_to_ids(e)
            a=ids.index(sid); b=ids.index(eid)
            if b>a: return a+1,b
        except Exception: pass
    raise RuntimeError("no image span")

def clamp_box(x0,y0,x1,y1, W,H):
    x0=max(0, min(float(x0), W-1)); x1=max(0, min(float(x1), W-1))
    y0=max(0, min(float(y0), H-1)); y1=max(0, min(float(y1), H-1))
    if x1<x0: x0,x1=x1,x0
    if y1<y0: y0,y1=y1,y0
    return x0,y0,x1,y1

def rect_from_obj(o, W,H, default_size):
    # prefer center_position + size; fallback paste_position/position + size
    if isinstance(o,dict):
        s=float(o.get("size", default_size))
        if "center_position" in o:
            cx,cy=o["center_position"]; return clamp_box(cx - s/2, cy - s/2, cx + s/2, cy + s/2, W,H)
        if "paste_position" in o:
            x,y=o["paste_position"]; return clamp_box(x, y, x+s, y+s, W,H)
        if "position" in o:
            x,y=o["position"]; return clamp_box(x, y, x+s, y+s, W,H)
    s=float(default_size); cx=W/2; cy=H/2
    return clamp_box(cx - s/2, cy - s/2, cx + s/2, cy + s/2, W,H)

def bbox_to_token_idxs(x0,y0,x1,y1, W,H, Htok,Wtok, pad, span_start):
    tr=lambda yp: int(np.clip(np.floor(yp/H*Htok), 0, Htok-1))
    tc=lambda xp: int(np.clip(np.floor(xp/W*Wtok), 0, Wtok-1))
    r0,r1=tr(y0),tr(y1); c0,c1=tc(x0),tc(x1)
    r0=max(0, min(r0,r1)-pad); r1=min(Htok-1, max(r0,r1)+pad)
    c0=max(0, min(c0,c1)-pad); c1=min(Wtok-1, max(c0,c1)+pad)
    rows=np.arange(r0,r1+1); cols=np.arange(c0,c1+1)
    return np.unique((rows[:,None]*Wtok + cols[None,:]).ravel() + span_start)

def default_prompt():
    return (
      "Look at the image. There are four rows, each labeled by a symbol on the left.\n"
      "Write EXACTLY four lines, one per row, from top to bottom, using only lowercase color and shape words.\n"
      "Format:\n"
      "row 1: <color> <shape>\n"
      "row 2: <color> <shape>\n"
      "row 3: <color> <shape>\n"
      "row 4: <color> <shape>\n"
      "Do not add extra text.\n"
    )

def grid_params(rec):
    W=int(rec.get("canvas_size_x") or rec.get("canvas_w") or rec.get("canvas_size") or 336)
    H=int(rec.get("canvas_size_y") or rec.get("canvas_h") or rec.get("canvas_size") or 336)
    gx=int(rec.get("grid_size_x") or 11)
    gy=int(rec.get("grid_size_y") or 12)
    ps=int(rec.get("patch_size") or max(1,H//gy))
    return W,H,gx,gy,ps

def row_to_symbol(rec, row_id):
    # find symbol object whose 'row' == row_id ; return the character (e.g. "@")
    for o in rec.get("objects", []):
        if ("symbol" in o or o.get("type")=="symbol") and int(o.get("row", -999))==int(row_id):
            return o.get("symbol") or o.get("text") or "?"
    return None

def find_shape_in_row(rec, row_id):
    for o in rec.get("objects", []):
        if (o.get("type")=="shape" or "shape" in o) and int(o.get("row", -999))==int(row_id):
            return o
    return None

def list_rows(rec):
    # Collect all distinct 'row' ids present among symbols (prefer) or shapes; sort numerically (e.g., 2,5,8,11)
    rows=set()
    for o in rec.get("objects", []):
        if "row" in o:
            try: rows.add(int(o["row"]))
            except: pass
    return sorted(rows)

# --------------- hooks ---------------
class CapturePreAttn:
    def __init__(self, modules, when_T):
        self.cache={}; self.h=[]; self.T=when_T
        for li,attn in modules:
            self.h.append(attn.register_forward_pre_hook(self._mk(li), with_kwargs=True))
    def _mk(self, li):
        def f(mod, args, kwargs):
            x=kwargs.get("hidden_states", args[0] if args else None)
            if x is not None and x.dim()==3 and x.shape[1]==self.T:
                self.cache[li] = x[0].detach().clone()  # [T,D]
        return f
    def close(self):
        for hh in self.h: hh.remove()
        self.h.clear()

def get_attn_modules(model):
    m = model.model.language_model.layers
    return [(i, m[i].self_attn) for i in range(len(m))]

def pick_layers(n, spec):
    if spec=="all": return list(range(n))
    if spec=="mid": return list(range(n//3,(2*n)//3))
    m=re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', str(spec) or "")
    if m:
        lo,hi=int(m.group(1)),int(m.group(2))
        return list(range(max(0,lo), min(n-1,hi)+1))
    return list(range(n//3,(2*n)//3))

def build_batch(proc, img_path, prompt, device):
    im=Image.open(img_path).convert("RGB")
    chat=proc.apply_chat_template(
        [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}],
        add_generation_prompt=True
    )
    b=proc(text=[chat], images=[im], return_tensors="pt")
    if b.get("attention_mask") is None:
        b["attention_mask"]=torch.ones_like(b["input_ids"])
    return {k:v.to(device) for k,v in b.items()}

# --------------- main ---------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir",  required=True)
    ap.add_argument("--metadata",  required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--pattern", default="sample_*_final.png", help="glob inside data_dir")
    ap.add_argument("--max_images", type=int, default=100000)
    ap.add_argument("--progress_every", type=int, default=10)
    args=ap.parse_args()

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pm_dir=out_dir/"pmeans"; pm_dir.mkdir(parents=True, exist_ok=True)

    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.float16 if device=="cuda" else torch.float32

    proc=AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, device_map="auto" if device=="cuda" else None,
        torch_dtype=dtype, trust_remote_code=True).eval()

    prompt=default_prompt()
    meta=load_meta(args.metadata)
    meta_vals=list(meta.values()) if isinstance(meta,dict) else list(meta)

    # fast index meta by filename
    by_fn={ str(r.get("filename")): r for r in meta_vals }

    # list candidate images by pattern
    files=sorted(Path(args.data_dir).glob(args.pattern))
    if args.max_images>0: files=files[:args.max_images]

    # layer modules for hook
    attn=get_attn_modules(model)
    chosen=pick_layers(len(attn), args.layers)
    mods=[attn[i] for i in chosen]
    print(f"[setup] images={len(files)} layers={len(chosen)} pad={args.pad}")

    for i,fp in enumerate(files, start=1):
        fn=fp.name
        rec=by_fn.get(fn)
        if rec is None:
            print(f"[skip] {fn}: no metadata record")
            continue

        try:
            # Build batch + capture
            b=build_batch(proc, fp, prompt, next(model.parameters()).device)
            s,e=find_image_span(b["input_ids"], proc.tokenizer)
            N=e-s; Htok,Wtok=best_2d_factors(N)

            W,H,gx,gy,ps=grid_params(rec)

            cap=CapturePreAttn(mods, when_T=b["input_ids"].shape[1])
            with torch.no_grad(): _=model(**b, use_cache=False)
            cap.close()

            # hidden size (assume same per layer)
            any_layer=next(iter(cap.cache.values()))
            D=any_layer.shape[-1]

            # rows present
            rows=list_rows(rec)

            # collect per-symbol keyed entries
            sym_means = {}   # sym -> [n_layers, D]
            obj_means = {}   # sym -> [n_layers, D]
            sym_idxs  = {}   # sym -> np.array token idxs
            obj_idxs  = {}

            for row_id in rows:
                sym=row_to_symbol(rec, row_id)
                shp=find_shape_in_row(rec, row_id)
                # symbol rect (if any)
                if sym is not None:
                    # find that symbol object entry again to get its box
                    sobj=None
                    for o in rec.get("objects", []):
                        if ("symbol" in o or o.get("type")=="symbol") and int(o.get("row", -999))==int(row_id):
                            sobj=o; break
                    if sobj is not None:
                        x0,y0,x1,y1 = rect_from_obj(sobj, W,H, ps)
                        sidx = bbox_to_token_idxs(x0,y0,x1,y1, W,H, Htok,Wtok, args.pad, s)
                        if sidx.size>0:
                            # mean per layer
                            S=[]
                            for li in chosen:
                                X=cap.cache[li]   # [T,D]
                                S.append( X[sidx].mean(dim=0).to(torch.float16).cpu().numpy() )
                            sym_means[sym]=np.stack(S, axis=0)  # [L,D]
                            sym_idxs[sym]=sidx
                # object rect (if any)
                if shp is not None and sym is not None:
                    x0,y0,x1,y1 = rect_from_obj(shp, W,H, ps)
                    oidx = bbox_to_token_idxs(x0,y0,x1,y1, W,H, Htok,Wtok, args.pad, s)
                    if oidx.size>0:
                        O=[]
                        for li in chosen:
                            X=cap.cache[li]
                            O.append( X[oidx].mean(dim=0).to(torch.float16).cpu().numpy() )
                        obj_means[sym]=np.stack(O, axis=0)  # [L,D]
                        obj_idxs[sym]=oidx

            # Save NPZ
            out_npz = pm_dir / f"pmeans_{fp.stem}.npz"
            np.savez(
                str(out_npz),
                filename=fn,
                layers=np.array(chosen, dtype=int),
                hidden_size=np.array([D], dtype=int),
                Htok=np.array([Htok], dtype=int), Wtok=np.array([Wtok], dtype=int),
                span_start=np.array([s], dtype=int), span_end=np.array([e], dtype=int),
                pad=np.array([args.pad], dtype=int),
                rows=np.array(rows, dtype=int),
                # flatten dicts
                sym_keys=np.array(list(sym_means.keys())),
                obj_keys=np.array(list(obj_means.keys())),
                **{f"sym_mean__{k}":v for k,v in sym_means.items()},
                **{f"obj_mean__{k}":v for k,v in obj_means.items()},
                **{f"sym_idxs__{k}":v for k,v in sym_idxs.items()},
                **{f"obj_idxs__{k}":v for k,v in obj_idxs.items()},
            )
            if i % max(1, args.progress_every)==0:
                print(f"[{i}/{len(files)}] saved {out_npz.name} (symbols={list(sym_means.keys())})")
        except Exception as e:
            print(f"[error] {fn}: {repr(e)}")

    print("[done]")

if __name__=="__main__":
    torch.set_grad_enabled(False)
    main()
