#!/usr/bin/env python3
import argparse, json, math, re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------- utilities ----------
def best_2d_factors(n: int):
    best=(1,n); gap=n
    for h in range(1, int(math.sqrt(n))+1):
        if n%h==0:
            w=n//h
            if abs(h-w)<gap: best=(h,w); gap=abs(h-w)
    return best

def load_meta(p):
    p = Path(p)
    if p.suffix==".json": return json.loads(p.read_text())
    if p.suffix==".npy":  return np.load(p, allow_pickle=True).tolist()
    raise ValueError("metadata must be .json or .npy")

def find_image_span(input_ids, tok):
    ids = input_ids[0].tolist()
    for s,e in [("<|vision_start|>","<|vision_end|>"), ("<|image_start|>","<|image_end|>")]:
        try:
            sid = tok.convert_tokens_to_ids(s); eid = tok.convert_tokens_to_ids(e)
            a = ids.index(sid); b = ids.index(eid)
            if b > a: return a+1, b
        except Exception: pass
    raise RuntimeError("could not find image span")

def clamp_box(x0,y0,x1,y1, W,H):
    x0=max(0, min(float(x0), W-1)); x1=max(0, min(float(x1), W-1))
    y0=max(0, min(float(y0), H-1)); y1=max(0, min(float(y1), H-1))
    if x1<x0: x0,x1=x1,x0
    if y1<y0: y0,y1=y1,y0
    return x0,y0,x1,y1

def rowscols_to_lin(rows, cols, Wtok):
    return np.array([r*Wtok + c for r in rows for c in cols], dtype=int)

def bbox_to_token_idxs(x0,y0,x1,y1, canvasW,canvasH, Htok,Wtok, pad, span_start):
    tr = lambda yp: int(np.clip(np.floor(yp/canvasH*Htok), 0, Htok-1))
    tc = lambda xp: int(np.clip(np.floor(xp/canvasW*Wtok), 0, Wtok-1))
    r0,r1 = tr(y0), tr(y1); c0,c1 = tc(x0), tc(x1)
    r0 = max(0, min(r0,r1)-pad); r1 = min(Htok-1, max(r0,r1)+pad)
    c0 = max(0, min(c0,c1)-pad); c1 = min(Wtok-1, max(c0,c1)+pad)
    rows = np.arange(r0, r1+1); cols = np.arange(c0, c1+1)
    return np.unique(rowscols_to_lin(rows, cols, Wtok) + span_start)

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

def _grid_params(rec):
    W = int(rec.get("canvas_size_x") or rec.get("canvas_w") or rec.get("canvas_size") or 336)
    H = int(rec.get("canvas_size_y") or rec.get("canvas_h") or rec.get("canvas_size") or 336)
    gx = int(rec.get("grid_size_x") or 11)
    gy = int(rec.get("grid_size_y") or 12)
    ps = int(rec.get("patch_size") or max(1, H//gy))
    return W,H,gx,gy,ps

def _row_center_y(abs_row, patch_size):
    # abs_row is absolute grid row id (2,5,8,11 etc), center at (row - 0.5) * patch_size
    return (float(abs_row) - 0.5) * float(patch_size)

def _find_shape_in_row(rec, row_id):
    objs = rec.get("objects", [])
    cands = [o for o in objs if o.get("type")=="shape" and int(o.get("row", -999))==int(row_id)]
    if cands: return cands[0]
    return None

def _typical_shape_cx(rec, default_cx):
    xs=[]
    for o in rec.get("objects", []):
        if o.get("type")=="shape" and isinstance(o.get("center_position"), (list,tuple)) and len(o["center_position"])>=1:
            xs.append(float(o["center_position"][0]))
    if xs: return float(np.median(xs))
    return float(default_cx)

def _shape_rect_from_obj(obj, W,H, default_size):
    # prefer center_position + size; fallback to paste_position+size
    if isinstance(obj, dict):
        if "center_position" in obj and "size" in obj:
            cx,cy = obj["center_position"]; s=float(obj["size"] or default_size)
            return clamp_box(cx - s/2, cy - s/2, cx + s/2, cy + s/2, W,H)
        if "paste_position" in obj and "size" in obj:
            x,y = obj["paste_position"]; s=float(obj["size"] or default_size)
            return clamp_box(x, y, x+s, y+s, W,H)
        if "position" in obj and "size" in obj:
            x,y = obj["position"]; s=float(obj["size"] or default_size)
            return clamp_box(x, y, x+s, y+s, W,H)
    s=float(default_size); cx=W/2.0; cy=H/2.0
    return clamp_box(cx - s/2, cy - s/2, cx + s/2, cy + s/2, W,H)

# ---------- hooks ----------
class CapturePreAttn:
    def __init__(self, modules, when_T):
        self.cache={}; self.handles=[]; self.when_T=when_T
        for li,attn in modules:
            self.handles.append(attn.register_forward_pre_hook(self._mk(li), with_kwargs=True))
    def _mk(self, li):
        def f(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is not None and x.dim()==3 and x.shape[1]==self.when_T:
                self.cache[li] = x[0].detach().clone()
        return f
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()

class OverwritePreAttn:
    def __init__(self, modules, when_T, idxs_tgt_1d, src_captures, idxs_src_1d):
        self.when_T = when_T
        self.idxt = torch.as_tensor(idxs_tgt_1d, dtype=torch.long)
        self.idxs = torch.as_tensor(idxs_src_1d, dtype=torch.long)
        self.src  = src_captures
        self.enabled=False; self.handles=[]
        for li,attn in modules:
            self.handles.append(attn.register_forward_pre_hook(self._mk(li), with_kwargs=True))
    def _mk(self, li):
        def f(mod, args, kwargs):
            if not self.enabled or li not in self.src: return
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is None or x.dim()!=3 or x.shape[1]!=self.when_T: return
            with torch.no_grad():
                k = min(self.idxt.numel(), self.idxs.numel())
                if k==0: return
                x2 = x.clone(); xs = self.src[li]
                x2[0, self.idxt[:k]] = xs[self.idxs[:k]].to(x2.dtype).to(x2.device)
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs); kwargs["hidden_states"]=x2; return (args,kwargs)
            else:
                args=list(args); args[0]=x2; return (tuple(args), kwargs)
        return f
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()

def get_attn_modules(model):
    m = model.model.language_model.layers
    return [(i, m[i].self_attn) for i in range(len(m))]

def pick_layers(n_layers, spec):
    if spec=="all": return list(range(n_layers))
    if spec=="mid": return list(range(n_layers//3,(2*n_layers)//3))
    m = re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', str(spec) or "")
    if m:
        lo,hi = int(m.group(1)), int(m.group(2))
        return list(range(max(0,lo), min(n_layers-1,hi)+1))
    return list(range(n_layers//3,(2*n_layers)//3))

def build_batch(proc, img_path, prompt, device):
    im = Image.open(img_path).convert("RGB")
    chat = proc.apply_chat_template(
        [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}],
        add_generation_prompt=True
    )
    b = proc(text=[chat], images=[im], return_tensors="pt")
    if b.get("attention_mask") is None:
        b["attention_mask"] = torch.ones_like(b["input_ids"])
    return {k: v.to(device) for k,v in b.items()}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--source", required=True)  # donor (image 2)
    ap.add_argument("--target", required=True)  # receiver (image 1)
    ap.add_argument("--src_row", type=int, required=True, help="absolute row id in SOURCE (e.g., 11)")
    ap.add_argument("--tgt_row", type=int, required=True, help="absolute row id in TARGET (e.g., 2/5/8/11)")
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--prompt_file", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device=="cuda" else torch.float32

    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, device_map="auto" if device=="cuda" else None,
        torch_dtype=dtype, trust_remote_code=True).eval()

    prompt = Path(args.prompt_file).read_text() if args.prompt_file else default_prompt()

    meta = load_meta(args.metadata)
    meta_vals = list(meta.values()) if isinstance(meta, dict) else list(meta)
    rec_src = next(r for r in meta_vals if r["filename"]==args.source)
    rec_tgt = next(r for r in meta_vals if r["filename"]==args.target)

    dev = next(model.parameters()).device
    b_src = build_batch(proc, Path(args.data_dir)/args.source, prompt, dev)
    b_tgt = build_batch(proc, Path(args.data_dir)/args.target, prompt, dev)

    s0,e0 = find_image_span(b_src["input_ids"], proc.tokenizer)
    s1,e1 = find_image_span(b_tgt["input_ids"], proc.tokenizer)
    assert (e0-s0)==(e1-s1), "source/target image token counts must match"
    N_img = e0 - s0
    Htok,Wtok = best_2d_factors(N_img)
    print(f"[src] span=[{s0},{e0}) N_img={N_img} grid={Htok}x{Wtok}")
    print(f"[tgt] span=[{s1},{e1}) N_img={N_img} grid={Htok}x{Wtok}")

    W0,H0,_,gy0,ps0 = _grid_params(rec_src)
    W1,H1,_,gy1,ps1 = _grid_params(rec_tgt)

    # choose layers
    attn = get_attn_modules(model)
    chosen = pick_layers(len(attn), args.layers)
    mods = [attn[i] for i in chosen]
    print(f"[layers] {len(chosen)} layers -> {chosen[:8]}{' ...' if len(chosen)>8 else ''}")

    # --- SOURCE shape in requested row ---
    src_shape = _find_shape_in_row(rec_src, args.src_row)
    if src_shape is None:
        raise SystemExit(f"[error] no SHAPE in source row {args.src_row}")
    sx0,sy0,sx1,sy1 = _shape_rect_from_obj(src_shape, W0,H0, ps0)
    print(f"[src shape] row {args.src_row} rect=({sx0:.1f},{sy0:.1f},{sx1:.1f},{sy1:.1f})")

    # --- TARGET: place at the shape slot of tgt_row (existing or synthesized) ---
    tgt_shape = _find_shape_in_row(rec_tgt, args.tgt_row)
    if tgt_shape is not None:
        tx0,ty0,tx1,ty1 = _shape_rect_from_obj(tgt_shape, W1,H1, ps1)
        print(f"[tgt shape] row {args.tgt_row} (existing) rect=({tx0:.1f},{ty0:.1f},{tx1:.1f},{ty1:.1f})")
    else:
        cx_typ = _typical_shape_cx(rec_tgt, default_cx=W1/2.0)
        sW, sH = (sx1-sx0), (sy1-sy0)  # keep source shape size
        cy = _row_center_y(args.tgt_row, ps1)
        tx0,ty0 = cx_typ - sW/2, cy - sH/2
        tx1,ty1 = tx0 + sW, ty0 + sH
        tx0,ty0,tx1,ty1 = clamp_box(tx0,ty0,tx1,ty1, W1,H1)
        print(f"[tgt shape] row {args.tgt_row} (synth @ cx={cx_typ:.1f}) rect=({tx0:.1f},{ty0:.1f},{tx1:.1f},{ty1:.1f})")

    # token idxs
    src_idxs = bbox_to_token_idxs(sx0,sy0,sx1,sy1, W0,H0, Htok,Wtok, args.pad, s0)
    tgt_idxs = bbox_to_token_idxs(tx0,ty0,tx1,ty1, W1,H1, Htok,Wtok, args.pad, s1)
    k = min(len(src_idxs), len(tgt_idxs))
    if k==0: raise SystemExit("[error] empty idx arrays; nothing to overwrite")
    src_idxs = src_idxs[:k]; tgt_idxs = tgt_idxs[:k]
    print(f"[idxs] src_n={len(src_idxs)} tgt_n={len(tgt_idxs)} -> k={k}")

    # capture source
    T_src = b_src["input_ids"].shape[1]
    cap = CapturePreAttn(mods, when_T=T_src)
    with torch.no_grad(): _ = model(**b_src, use_cache=False)
    cap.close()

    # overwrite target
    T_tgt = b_tgt["input_ids"].shape[1]
    ow = OverwritePreAttn(mods, when_T=T_tgt, idxs_tgt_1d=tgt_idxs,
                          src_captures=cap.cache, idxs_src_1d=src_idxs)

    gen = dict(max_new_tokens=64, do_sample=not args.greedy,
               temperature=0.0 if args.greedy else 0.7, top_p=1.0)

    print("\n=== SOURCE BASELINE ===")
    g_src = model.generate(**b_src, **gen)
    txt_src = proc.batch_decode(g_src, skip_special_tokens=True)[0]
    print(txt_src)

    print("\n=== TARGET BASELINE ===")
    g_base = model.generate(**b_tgt, **gen)
    txt_base = proc.batch_decode(g_base, skip_special_tokens=True)[0]
    print(txt_base)

    print("\n=== TARGET INTERVENED ===")
    ow.enabled = True
    g_int = model.generate(**b_tgt, **gen)
    txt_int = proc.batch_decode(g_int, skip_special_tokens=True)[0]
    print(txt_int)
    ow.close()

    (Path(args.out_dir)/"source.txt").write_text(txt_src+"\n")
    (Path(args.out_dir)/"baseline.txt").write_text(txt_base+"\n")
    (Path(args.out_dir)/"intervened.txt").write_text(txt_int+"\n")

    np.savez(str(Path(args.out_dir)/"verify.npz"),
             layers=np.array([i for i,_ in mods], dtype=int),
             src_idxs=src_idxs, tgt_idxs=tgt_idxs,
             src_rect=np.array([sx0,sy0,sx1,sy1]),
             tgt_rect=np.array([tx0,ty0,tx1,ty1]),
             T_src=np.array([T_src]), T_tgt=np.array([T_tgt]))
    print(f"[verify] wrote {Path(args.out_dir)/'verify.npz'}")

if __name__=="__main__":
    torch.set_grad_enabled(False)
    main()
