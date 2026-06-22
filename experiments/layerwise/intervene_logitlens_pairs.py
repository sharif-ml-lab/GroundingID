#!/usr/bin/env python3
import os, re, csv, json, math, argparse, gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYMBOLS = ["@", "$", "#", "&"]
SHAPES  = ["square","circle","triangle","diamond"]

def best_2d_factors(n: int):
    best=(1,n); gap=n
    for h in range(1,int(math.sqrt(n))+1):
        if n%h==0:
            w=n//h
            if abs(h-w)<gap:
                best=(h,w); gap=abs(h-w)
    return best

def load_meta(p):
    p = Path(p)
    if p.suffix==".json": return json.loads(p.read_text())
    if p.suffix==".npy":  return np.load(p, allow_pickle=True).tolist()
    raise ValueError("metadata must be .json or .npy")

def meta_list(meta):
    return list(meta.values()) if isinstance(meta, dict) else list(meta)

def find_image_span(input_ids, tok):
    ids = input_ids[0].tolist()
    for s,e in [("<|vision_start|>","<|vision_end|>"), ("<|image_start|>","<|image_end|>")]:
        try:
            sid = tok.convert_tokens_to_ids(s); eid = tok.convert_tokens_to_ids(e)
            a=ids.index(sid); b=ids.index(eid)
            if b>a: return a+1,b
        except Exception: pass
    raise RuntimeError("no image span")

def resolve_img_path(data_dir, filename):
    p0 = Path(data_dir)/filename
    if p0.exists(): return p0
    stem = Path(filename).stem
    p1 = Path(data_dir)/(stem + "_with_symbols.png")
    if p1.exists(): return p1
    m = re.search(r'(\d{3,})', filename)
    if m:
        ident = m.group(1)
        for q in Path(data_dir).glob(f"*{ident}*"):
            if q.is_file(): return q
    raise FileNotFoundError(f"cannot resolve {filename} in {data_dir}")

def y_of(o):
    pos = o.get("center_position") or o.get("position") or o.get("center") or [0,0]
    return pos[1] if isinstance(pos,(list,tuple)) and len(pos)>1 else 0

def symbol_rows(rec):
    objs = rec.get("objects") or rec.get("rows") or []
    rows={}
    for o in objs:
        if str(o.get("type","")).lower()=="symbol" and "symbol" in o:
            sym=str(o["symbol"])
            try:
                r=int(o.get("row"))
                rows[sym]=r-1 if r>1 else r
            except Exception:
                pass
    if len(rows)<4:
        syms=[o for o in objs if str(o.get("type","")).lower()=="symbol" and "symbol" in o]
        syms_sorted=sorted(syms, key=y_of)
        for i,o in enumerate(syms_sorted):
            rows[str(o["symbol"])] = i
    return rows

def _find_row_object(rec, row_idx0):
    objs = rec.get("objects") or rec.get("rows") or []
    if not isinstance(objs,list) or not objs: return None, "no-objects"
    ints=[]
    for o in objs:
        try: ints.append(int(o.get("row")))
        except Exception: pass
    if ints:
        base=min(ints); target=row_idx0+base
        for o in objs:
            try:
                if int(o.get("row"))==target:
                    return o, f"row-field(base={base})"
            except Exception: pass
    def y_of2(o):
        pos = o.get("position") or o.get("center_position") or o.get("center") or [0,0]
        return pos[1] if isinstance(pos,(list,tuple)) and len(pos)>1 else 0
    objs_sorted=sorted(objs, key=y_of2)
    if 0<=row_idx0<len(objs_sorted):
        return objs_sorted[row_idx0], "y-sort"
    return None, "no-match"

def _bbox_from_obj_or_fallback(obj, row_idx0, canvas_size):
    if isinstance(obj, dict):
        if "position" in obj and "size" in obj:
            x,y = obj["position"]; sz = obj["size"]
            if isinstance(sz,(list,tuple)) and len(sz)>=2: w,h=sz[0],sz[1]
            else: w=h=sz
            return float(x),float(y),float(x)+float(w),float(y)+float(h)
        if "center_position" in obj and "size" in obj:
            cx,cy=obj["center_position"]; sz=obj["size"]
            w=h=sz if not isinstance(sz,(list,tuple)) else sz[0]
            return float(cx-w/2), float(cy-h/2), float(cx+w/2), float(cy+h/2)
        if "bbox" in obj and isinstance(obj["bbox"], (list,tuple)) and len(obj["bbox"])>=4:
            x0,y0,a,b2 = map(float, obj["bbox"][:4])
            return (x0,y0,a,b2) if (a>x0 and b2>y0) else (x0,y0,x0+a,y0+b2)
        keys = obj.keys()
        if all(k in keys for k in ("x","y","w","h")):
            x0,y0,w,h = float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])
            return x0,y0,x0+w,y0+h
    # 4 equal bands fallback
    rows=4
    y0 = row_idx0*(canvas_size/rows)
    y1 = (row_idx0+1)*(canvas_size/rows) - 1
    return 0.0,y0,float(canvas_size-1),y1

def bbox_to_token_idxs(x0,y0,x1,y1, canvas, H,W, pad, span_start):
    tr = lambda yp: int(np.clip(np.floor(yp/canvas*H),0,H-1))
    tc = lambda xp: int(np.clip(np.floor(xp/canvas*W),0,W-1))
    r0,r1 = tr(y0),tr(y1); c0,c1 = tc(x0),tc(x1)
    r0 = max(0, min(r0,r1)-pad); r1 = min(H-1, max(r0,r1)+pad)
    c0 = max(0, min(c0,c1)-pad); c1 = min(W-1, max(c0,c1)+pad)
    rows=np.arange(r0,r1+1); cols=np.arange(c0,c1+1)
    idxs=[r*W+c for r in rows for c in cols]
    return np.unique(np.array(idxs,dtype=int)+span_start)

def row_token_idxs(rec, row_idx0, canvas, H,W, pad, span_start, log_prefix=""):
    obj, how = _find_row_object(rec, row_idx0)
    x0,y0,x1,y1=_bbox_from_obj_or_fallback(obj, row_idx0, canvas)
    ids = bbox_to_token_idxs(x0,y0,x1,y1, canvas,H,W,pad,span_start)
    print(f"   [rowAbs] row={row_idx0+1} via {how} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) tokN={len(ids)}")
    return ids

def build_batch(processor, img_path, text, device):
    im = Image.open(img_path).convert("RGB")
    chat = processor.apply_chat_template(
        [{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    b = processor(text=[chat], images=[im], return_tensors="pt")
    return {k: v.to(device) for k,v in b.items()}

def pick_layers(n, spec):
    if spec=="all": return list(range(n))
    if spec=="mid": return list(range(n//3,(2*n)//3))
    m=re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', spec or "")
    if m:
        lo,hi=int(m.group(1)),int(m.group(2))
        return list(range(max(0,lo), min(n,hi+1)))
    return list(range(n//3,(2*n)//3))

class CapturePreAttn:
    def __init__(self, modules, when_T):
        self.cache={}; self.handles=[]; self.when_T=when_T
        for li, attn in modules:
            h=attn.register_forward_pre_hook(self._mk(li), with_kwargs=True)
            self.handles.append(h)
    def _mk(self, li):
        def prehook(mod, args, kwargs):
            x = kwargs.get("hidden_states", None)
            if x is None and isinstance(args,(tuple,list)) and len(args)>0: x=args[0]
            if x is not None and x.dim()==3 and x.shape[1]==self.when_T:
                self.cache[li]=x[0].detach().clone()
        return prehook
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()

class OverwritePreAttn:
    def __init__(self, modules, when_T, idxs_tgt_1d, src_captures, idxs_src_1d):
        self.when_T=when_T
        self.idxt=torch.as_tensor(idxs_tgt_1d, dtype=torch.long)
        self.idxs=torch.as_tensor(idxs_src_1d, dtype=torch.long)
        self.src=src_captures
        self.enabled=False; self.handles=[]
        for li, attn in modules:
            h=attn.register_forward_pre_hook(self._mk(li), with_kwargs=True)
            self.handles.append(h)
    def _mk(self, li):
        def prehook(mod, args, kwargs):
            if not self.enabled or li not in self.src: return
            x = kwargs.get("hidden_states", None)
            if x is None and isinstance(args,(tuple,list)) and len(args)>0: x=args[0]
            if x is None or x.dim()!=3 or x.shape[1]!=self.when_T: return
            with torch.no_grad():
                k=min(self.idxt.numel(), self.idxs.numel(), x.shape[1], self.src[li].shape[0])
                if k==0: return
                x2=x.clone(); xs=self.src[li]
                x2[0, self.idxt[:k]] = xs[self.idxs[:k]].to(x2.dtype).to(x2.device)
            if "hidden_states" in kwargs:
                kwargs=dict(kwargs); kwargs["hidden_states"]=x2; return (args, kwargs)
            else:
                args=list(args); args[0]=x2; return (tuple(args), kwargs)
        return prehook
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()

def shape_first_subtoken_id(tokenizer, text_shape):
    # leading space to favor a single sub-token piece if applicable
    ids = tokenizer.encode(" " + text_shape, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(text_shape, add_special_tokens=False)
    # take first sub-token (logit-lens is 1-token step anyway)
    return int(ids[0]) if ids else None

def build_shape_vocab(tokenizer):
    m={}
    for s in SHAPES:
        tid = shape_first_subtoken_id(tokenizer, s)
        m[s]=tid
    return m

def gen_and_hidden(model, processor, image_path, prompt_text, device):
    im = Image.open(image_path).convert("RGB")
    chat = processor.apply_chat_template(
        [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt_text}]}],
        add_generation_prompt=True,
    )
    batch = processor(text=[chat], images=[im], return_tensors="pt")
    batch = {k: v.to(device) for k,v in batch.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type=="cuda"), dtype=torch.float16):
        out = model.generate(
            **batch,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
            output_hidden_states=True,
            use_cache=False,
        )
    gen_ids = out.sequences[0].tolist()
    new_toks = gen_ids[len(batch["input_ids"][0]):]
    pred_id = new_toks[0] if new_toks else None
    # hidden_states is a list over generation steps; take step 0
    step0 = out.hidden_states[0]   # tuple: [emb, layer1..]
    hs=[]
    for h in step0:
        hs.append(h[0,-1,:].detach().float().cpu())
    return pred_id, hs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--query_symbol", default="@")
    ap.add_argument("--max_pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--start_idx", type=int, default=0, help="for chunked runs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if device.type=="cuda" else torch.float32,
        device_map="auto" if device.type=="cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    # shape vocab (first sub-token)
    shape2id = build_shape_vocab(processor.tokenizer)
    bad = [k for k,v in shape2id.items() if v is None]
    if bad:
        print(f"[warn] shapes without token id (using first-subtoken heuristic still None): {bad}")

    meta = load_meta(args.metadata)
    L = meta_list(meta)
    rec_by_name = {str(r.get("filename") or r.get("file") or ""): r for r in L}

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    # trim by symbols and slice for chunking (manifest already filtered to A=@, B=& in your pipeline)
    total = len(rows)
    if args.start_idx>0: rows = rows[args.start_idx:]
    if args.max_pairs>0: rows = rows[:args.max_pairs]
    print(f"[info] evaluating {len(rows)} pairs; query_symbol={args.query_symbol}")

    out_csv = Path(args.out_dir)/"probe_logitlens.csv"
    w = csv.writer(open(out_csv, "w", newline=""))
    w.writerow(["source","target","row_pairs","query_symbol","layer","target_token","other_token","pred_token","margin","is_correct"])

    # prebuild prompt
    qsym = args.query_symbol
    prompt_shape = (
        "Scan the image using the symbols on the left (&, $, #, @) as row labels.\n"
        f"Answer with ONLY the shape word of the object in the \"row {qsym}\".\n"
        "One word, lowercase."
    )

    for r in rows:
        src_name=r["source"]; tgt_name=r["target"]; rp=r["row_pairs"]
        rec_s=rec_by_name.get(src_name); rec_t=rec_by_name.get(tgt_name)
        if rec_s is None or rec_t is None:
            print(f"[skip] cannot find records: {src_name} {tgt_name}")
            continue
        try:
            s_img = resolve_img_path(args.data_dir, src_name)
            t_img = resolve_img_path(args.data_dir, tgt_name)
        except Exception as e:
            print(f"[skip] {e}"); continue

        # build tiny batches to get spans and canvases
        # (we’ll not forward them fully here; only to calculate indices)
        bS = build_batch(processor, s_img, ".", device)
        bT = build_batch(processor, t_img, ".", device)
        s0,e0 = find_image_span(bS["input_ids"], processor.tokenizer)
        s1,e1 = find_image_span(bT["input_ids"], processor.tokenizer)
        if (e0-s0)!=(e1-s1):
            print(f"[skip] image span mismatch {src_name} {tgt_name}")
            del bS,bT; torch.cuda.empty_cache(); continue
        H,W = best_2d_factors(e0-s0)

        # canvases
        def infer_canvas(rec, fn):
            try:
                with Image.open(resolve_img_path(args.data_dir, fn)) as im: w,h=im.size
                return int(max(w,h))
            except Exception:
                return 1024
        cs = infer_canvas(rec_s, src_name)
        ct = infer_canvas(rec_t, tgt_name)

        # parse row pairs like "11->2,2->11"
        swaps=[]
        for p in rp.split(","):
            a,b=p.split("->")
            ra=int(a)-1; rb=int(b)-1
            ids_src = row_token_idxs(rec_t, ra, ct, H,W, args.pad, s1)
            ids_tgt = row_token_idxs(rec_s, rb, cs, H,W, args.pad, s0)
            k=min(len(ids_src),len(ids_tgt))
            if k>0: swaps.append((ids_tgt[:k], ids_src[:k]))
        if not swaps:
            print(f"[skip] no swap idxs {src_name} {tgt_name}")
            del bS,bT; torch.cuda.empty_cache(); continue
        tgt_concat = np.concatenate([a for a,_ in swaps],axis=0)
        src_concat = np.concatenate([b for _,b in swaps],axis=0)

        # hook modules
        layers = model.model.language_model.layers
        chosen = pick_layers(len(layers), args.layers)
        modules=[(i, layers[i].self_attn) for i in chosen]

        # 1) capture TARGET prefill (to get source? no — we transplant TARGET→SOURCE, so capture TARGET)
        T_t = bT["input_ids"].shape[1]
        cap_t = CapturePreAttn(modules, when_T=T_t)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type=="cuda"), dtype=torch.float16):
            _ = model(**bT, use_cache=False)
        cap_t.close()

        # 2) Overwrite SOURCE positions with captured TARGET rows, then generate 1 token
        T_s = bS["input_ids"].shape[1]
        ow = OverwritePreAttn(modules, when_T=T_s,
            idxs_tgt_1d=tgt_concat, src_captures=cap_t.cache, idxs_src_1d=src_concat)
        ow.enabled=True
        pred_id, hs_layers = gen_and_hidden(model, processor, s_img, prompt_shape, device)
        ow.enabled=False

        # tokens for target/other shapes (from TARGET rows for @ and &)
        def shape_at_row(rec, r):
            objs = rec.get("objects") or []
            shapes=[]
            for o in objs:
                if str(o.get("type","")).lower()=="shape" or "shape" in o:
                    rr = o.get("row", None)
                    try: rr=int(rr)
                    except Exception: rr=None
                    shapes.append((rr, o))
            for rr,o in shapes:
                if rr is not None and r is not None and (rr-1 if rr>1 else rr)==r:
                    return str(o.get("shape","")).lower()
            # fallback by vertical order
            cand=[]
            for _,o in shapes:
                pos = o.get("center_position") or o.get("position") or [0,0]
                y = pos[1] if isinstance(pos,(list,tuple)) and len(pos)>1 else 0
                cand.append((y,o))
            cand.sort(key=lambda x:x[0])
            if r is None or not cand: return ""
            if 0<=r<len(cand): return str(cand[r][1].get("shape","")).lower()
            return ""

        rows_sym = symbol_rows(rec_t)
        r_at  = rows_sym.get("@", None)
        r_amp = rows_sym.get("&", None)
        shape_at  = shape_at_row(rec_t, r_at)
        shape_amp = shape_at_row(rec_t, r_amp)

        # limited vocab (target vs other) depends on query
        if args.query_symbol=="@":
            target_shape = shape_at
            other_shape  = shape_amp
        else:
            target_shape = shape_amp
            other_shape  = shape_at

        tid_t = shape2id.get(target_shape, None)
        tid_o = shape2id.get(other_shape,  None)
        if tid_t is None or tid_o is None:
            print(f"[warn] skip {src_name} vs {tgt_name}: cannot tokenize shapes ({target_shape},{other_shape})")
            del bS,bT, cap_t, ow, hs_layers
            torch.cuda.empty_cache(); gc.collect()
            continue

        # LM head weight for logit lens (tied head)
        lm = model.lm_head
        W = lm.weight.detach()  # (V, d)
        # compare only the two tokens: approximate margin by projecting last hidden per layer
        # NOTE: hs_layers includes embedding + each decoder layer; index matches that
        for L, v in enumerate(hs_layers):
            v = v.to(W.dtype).to(W.device)  # match dtype/device with head
            # logits for the two ids
            log_t = torch.matmul(W[tid_t], v)
            log_o = torch.matmul(W[tid_o], v)
            margin = float((log_t - log_o).detach().cpu())
            # prediction token id (decoded just for info)
            pred_tok = int(pred_id) if pred_id is not None else -1
            # correctness relative to TARGET vs OTHER:
            is_correct = 1 if margin>0 else 0
            w.writerow([src_name, tgt_name, rp, args.query_symbol, L, tid_t, tid_o, pred_tok, f"{margin:.6f}", is_correct])

        # cleanup per-pair
        del bS,bT, cap_t, ow, hs_layers
        torch.cuda.empty_cache(); gc.collect()

    print(f"[write] {out_csv}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
