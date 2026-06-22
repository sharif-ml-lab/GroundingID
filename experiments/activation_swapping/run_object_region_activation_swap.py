#!/usr/bin/env python3
import argparse, json, math, os, re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYMBOLS = ["@", "$", "#", "&"]  # top→bottom

# --------------------- small utils ---------------------
def best_2d_factors(n: int):
    best=(1,n); gap=n
    for h in range(1, int(math.sqrt(n))+1):
        if n%h==0:
            w=n//h
            if abs(h-w)<gap: best=(h,w); gap=abs(h-w)
    return best

def build_list_prompt():
    return (
        "Look at the image. There are four rows labeled by symbols on the left (@, $, #, &).\n"
        "Scan based on those symbols.\n"
        "Write EXACTLY four lines, one per row, using only lowercase color and shape words.\n"
        "Format:\n"
        "row @: <color> <shape>\n"
        "row $: <color> <shape>\n"
        "row #: <color> <shape>\n"
        "row &: <color> <shape>\n"
        "Do not add any extra text.\n"
    )

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
        except Exception:
            pass
    raise RuntimeError("could not find image span")

def clamp_box(x0,y0,x1,y1, W,H):
    x0=max(0, min(x0, W-1)); x1=max(0, min(x1, W-1))
    y0=max(0, min(y0, H-1)); y1=max(0, min(y1, H-1))
    if x1<x0: x0,x1=x1,x0
    if y1<y0: y0,y1=y1,y0
    return x0,y0,x1,y1

def rowscols_to_lin(rows, cols, Wtok):
    return np.array([r*Wtok + c for r in rows for c in cols], dtype=int)

def bbox_to_token_idxs(x0,y0,x1,y1, canvasW, canvasH, Htok,Wtok, pad, span_start):
    # map (x,y) in canvas to token grid rows/cols
    tr = lambda yp: int(np.clip(np.floor(yp/canvasH*Htok), 0, Htok-1))
    tc = lambda xp: int(np.clip(np.floor(xp/canvasW*Wtok), 0, Wtok-1))
    r0,r1 = tr(y0), tr(y1); c0,c1 = tc(x0), tc(x1)
    r0 = max(0, min(r0,r1)-pad); r1 = min(Htok-1, max(r0,r1)+pad)
    c0 = max(0, min(c0,c1)-pad); c1 = min(Wtok-1, max(c0,c1)+pad)
    rows = np.arange(r0, r1+1); cols = np.arange(c0, c1+1)
    return np.unique(rowscols_to_lin(rows, cols, Wtok) + span_start)

# --------------------- metadata adapters ---------------------
def _grid_params(rec):
    # tolerate different key names
    W = int(rec.get("canvas_size_x") or rec.get("canvas_w") or rec.get("canvas_size") or 336)
    H = int(rec.get("canvas_size_y") or rec.get("canvas_h") or rec.get("canvas_size") or 336)
    gx = int(rec.get("grid_size_x") or 11)
    gy = int(rec.get("grid_size_y") or 12)
    ps = int(rec.get("patch_size") or max(1, W//gx))
    return W,H,gx,gy,ps

def _obj_rect(obj, W,H, default_size=28):
    # Prefer paste_position (top-left) if present
    if isinstance(obj, dict):
        if "paste_position" in obj and "size" in obj:
            x,y = obj["paste_position"]; s = float(obj["size"] or default_size)
            return clamp_box(float(x), float(y), float(x)+s, float(y)+s, W,H)
        if "position" in obj and "size" in obj:
            x,y = obj["position"]; s = float(obj["size"] or default_size)
            return clamp_box(float(x), float(y), float(x)+s, float(y)+s, W,H)
        if "center_position" in obj and "size" in obj:
            cx,cy = obj["center_position"]; s = float(obj["size"] or default_size)
            x0 = float(cx) - s/2; y0 = float(cy) - s/2
            return clamp_box(x0,y0,x0+s,y0+s, W,H)
        if "bbox" in obj and isinstance(obj["bbox"], (list,tuple)) and len(obj["bbox"])>=4:
            x0,y0,w,h = obj["bbox"][:4]
            return clamp_box(float(x0),float(y0),float(x0)+float(w), float(y0)+float(h), W,H)
    # fallback to center of canvas
    s = float(default_size)
    x0 = (W - s)/2; y0 = (H - s)/2
    return clamp_box(x0,y0,x0+s,y0+s, W,H)

def _symbol_rows(rec):
    sr = rec.get("symbol_rows")
    if isinstance(sr, list) and len(sr)==4:
        return list(map(int, sr))
    # fallback: 4-band split by height
    W,H,_,_,ps = _grid_params(rec)
    # try to approximate 4 bands at token rows ~ [1,4,7,10]
    return [2,5,8,11]

def _symbol_to_row(rec, sym):
    rows = _symbol_rows(rec)
    idx = SYMBOLS.index(sym)
    return rows[idx]

def _find_symbol_obj(rec, row_id):
    # find the symbol object with obj['type']=='symbol' and obj['row']==row_id
    objs = rec.get("objects", [])
    for o in objs:
        if o.get("type")=="symbol" and int(o.get("row", -999)) == int(row_id):
            return o
    return None

def _find_shape_in_row(rec, row_id):
    objs = rec.get("objects", [])
    for o in objs:
        if o.get("type")=="shape" and int(o.get("row", -999)) == int(row_id):
            return o
    # some older metas may not have 'type' filled—guess by keys
    for o in objs:
        if ("shape" in o and "color" in o) and int(o.get("row", -999)) == int(row_id):
            return o
    return None

def _resolve_row_id(rec, token, role="row"):
    """
    token can be:
      - symbol '@' '$' '#' '&'
      - '1','2','3','4' (physical order → map via symbol_rows)
      - absolute row like '2','5','8','11' (if >=5 treated as absolute)
    """
    if isinstance(token, str) and token in SYMBOLS:
        return _symbol_to_row(rec, token)
    # number-ish
    r = int(token)
    rows = _symbol_rows(rec)
    if r in rows:
        return r
    if 1 <= r <= 4 and len(rows)==4:
        return rows[r-1]
    return r  # assume absolute

# --------------------- QK/V hooking at pre-attn input ---------------------
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
    if spec=="mid": return list(range(n_layers//3, (2*n_layers)//3))
    m = re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', str(spec) or "")
    if m:
        lo,hi = int(m.group(1)), int(m.group(2))
        return list(range(max(0,lo), min(n_layers-1,hi)+1))
    return list(range(n_layers//3, (2*n_layers)//3))

def build_batch(processor, img_path, prompt, device):
    im = Image.open(img_path).convert("RGB")
    chat = processor.apply_chat_template(
        [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}],
        add_generation_prompt=True
    )
    b = processor(text=[chat], images=[im], return_tensors="pt")
    if b.get("attention_mask") is None:
        b["attention_mask"] = torch.ones_like(b["input_ids"])
    return {k: v.to(device) for k,v in b.items()}

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--source", required=True)  # donor image filename
    ap.add_argument("--target", required=True)  # receiver image filename
    # choose by symbol or by row tokens
    ap.add_argument("--src_symbol", default="")
    ap.add_argument("--tgt_symbol", default="")
    ap.add_argument("--row_pairs", default="", help="e.g. '4->1' (physical) or '11->2' (absolute) or '@->&'")
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--prompt_file", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device=="cuda" else torch.float32

    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, device_map="auto" if device=="cuda" else None,
        torch_dtype=dtype, trust_remote_code=True).eval()

    prompt = Path(args.prompt_file).read_text() if args.prompt_file else build_list_prompt()

    meta = load_meta(args.metadata)
    meta_vals = list(meta.values()) if isinstance(meta, dict) else list(meta)
    rec_src = next(r for r in meta_vals if r["filename"]==args.source)
    rec_tgt = next(r for r in meta_vals if r["filename"]==args.target)

    dev = next(model.parameters()).device
    b_src = build_batch(proc, Path(args.data_dir)/args.source, prompt, dev)
    b_tgt = build_batch(proc, Path(args.data_dir)/args.target, prompt, dev)

    s0,e0 = find_image_span(b_src["input_ids"], proc.tokenizer)
    s1,e1 = find_image_span(b_tgt["input_ids"], proc.tokenizer)
    assert (e0-s0)==(e1-s1)
    N_img = e0 - s0
    Htok,Wtok = best_2d_factors(N_img)
    print(f"[src] span=[{s0},{e0}) N_img={N_img} grid={Htok}x{Wtok}")
    print(f"[tgt] span=[{s1},{e1}) N_img={N_img} grid={Htok}x{Wtok}")

    W0,H0,_,_,ps0 = _grid_params(rec_src)
    W1,H1,_,_,ps1 = _grid_params(rec_tgt)

    # choose layers
    attn = get_attn_modules(model)
    chosen = pick_layers(len(attn), args.layers)
    mods = [attn[i] for i in chosen]
    print(f"[layers] {len(chosen)} layers -> {chosen[:8]}{' ...' if len(chosen)>8 else ''}")

    # parse which rows to use
    if args.src_symbol and args.tgt_symbol:
        rs = _resolve_row_id(rec_src, args.src_symbol)
        rt = _resolve_row_id(rec_tgt, args.tgt_symbol)
        pairs = [(rs, rt, f"{args.src_symbol}->{args.tgt_symbol}")]
    elif args.row_pairs:
        pairs=[]
        for p in [x.strip() for x in args.row_pairs.split(",") if x.strip()]:
            a,b = p.split("->")
            rs = _resolve_row_id(rec_src, a.strip())
            rt = _resolve_row_id(rec_tgt, b.strip())
            pairs.append((rs, rt, p))
    else:
        raise SystemExit("Specify --src_symbol/--tgt_symbol OR --row_pairs")

    # 1) capture SOURCE
    T_src = b_src["input_ids"].shape[1]
    cap = CapturePreAttn(mods, when_T=T_src)
    with torch.no_grad(): _ = model(**b_src, use_cache=False)
    cap.close()

    # 2) build idx lists (rects from shape in src row; rects from shape or synthesized in tgt row)
    tgt_all=[]; src_all=[]
    for rs,rt,label in pairs:
        src_shape = _find_shape_in_row(rec_src, rs)
        if src_shape is None:
            raise SystemExit(f"[error] no shape in source row {rs} (resolved).")
        sx0,sy0,sx1,sy1 = _obj_rect(src_shape, W0,H0, default_size=ps0)
        print(f"[src] {label}: shape rect=({sx0:.1f},{sy0:.1f},{sx1:.1f},{sy1:.1f})")

        tgt_shape = _find_shape_in_row(rec_tgt, rt)
        if tgt_shape is not None:
            tx0,ty0,tx1,ty1 = _obj_rect(tgt_shape, W1,H1, default_size=ps1)
            print(f"[tgt] {label}: (existing) rect=({tx0:.1f},{ty0:.1f},{tx1:.1f},{ty1:.1f})")
        else:
            # synthesize target centered at row symbol y and center x, with donor size
            sym_obj = _find_symbol_obj(rec_tgt, rt)
            if sym_obj and "center_position" in sym_obj:
                cx,cy = sym_obj["center_position"]
            else:
                # compute from patch size
                cx,cy = W1/2.0, (rt-0.5)*ps1
            sW = sx1 - sx0; sH = sy1 - sy0
            tx0 = cx - sW/2; ty0 = cy - sH/2
            tx1 = tx0 + sW;   ty1 = ty0 + sH
            tx0,ty0,tx1,ty1 = clamp_box(tx0,ty0,tx1,ty1, W1,H1)
            print(f"[tgt] {label}: (synth) rect=({tx0:.1f},{ty0:.1f},{tx1:.1f},{ty1:.1f})")

        # to token idxs
        src_idxs = bbox_to_token_idxs(sx0,sy0,sx1,sy1, W0,H0, Htok,Wtok, args.pad, s0)
        tgt_idxs = bbox_to_token_idxs(tx0,ty0,tx1,ty1, W1,H1, Htok,Wtok, args.pad, s1)
        k = min(len(src_idxs), len(tgt_idxs))
        src_all.append(src_idxs[:k]); tgt_all.append(tgt_idxs[:k])
        print(f"[idxs] {label}: src_n={len(src_idxs)} tgt_n={len(tgt_idxs)} -> k={k}")

    if not src_all or not tgt_all:
        raise SystemExit("[error] no indices built; check metadata mapping.")
    src_concat = np.concatenate(src_all); tgt_concat = np.concatenate(tgt_all)
    if src_concat.size==0 or tgt_concat.size==0:
        raise SystemExit("[error] empty idx arrays; intervention would be a no-op.")

    # 3) run 3 generations
    T_tgt = b_tgt["input_ids"].shape[1]
    ow = OverwritePreAttn(mods, when_T=T_tgt, idxs_tgt_1d=tgt_concat,
                          src_captures=cap.cache, idxs_src_1d=src_concat)
    gen_args = dict(max_new_tokens=64, do_sample=not args.greedy, temperature=0.0 if args.greedy else 0.7, top_p=1.0)

    print("\n=== SOURCE BASELINE ===")
    g_src = model.generate(**b_src, **gen_args)
    src_txt = proc.batch_decode(g_src, skip_special_tokens=True)[0]
    print(src_txt)

    print("\n=== TARGET BASELINE ===")
    g_base = model.generate(**b_tgt, **gen_args)
    base_txt = proc.batch_decode(g_base, skip_special_tokens=True)[0]
    print(base_txt)

    print("\n=== TARGET INTERVENED ===")
    ow.enabled = True
    g_int = model.generate(**b_tgt, **gen_args)
    int_txt = proc.batch_decode(g_int, skip_special_tokens=True)[0]
    print(int_txt)
    ow.close()

    # save
    (out_dir/"source.txt").write_text(src_txt+"\n")
    (out_dir/"baseline.txt").write_text(base_txt+"\n")
    (out_dir/"intervened.txt").write_text(int_txt+"\n")
    np.savez(str(out_dir/"verify.npz"),
             layers=np.array(chosen, dtype=int),
             src_idxs=src_concat, tgt_idxs=tgt_concat,
             src_rect=np.array([sx0,sy0,sx1,sy1]),
             tgt_last_rect=np.array([tx0,ty0,tx1,ty1]),
             T_src=np.array([T_src]), T_tgt=np.array([T_tgt]))
    print(f"[verify] wrote {out_dir/'verify.npz'}")

if __name__=="__main__":
    torch.set_grad_enabled(False)
    main()
