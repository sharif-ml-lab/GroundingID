import os, json, csv, random, re, subprocess
from pathlib import Path

SYM2ROW = {'!':0, '%':1, '*':2, '+':3}   # injector rows top..bottom
HOST_ROWS = [1,2,3,4]                    # 1..4

PROMPT_TMPL = """Scan the image using the symbols on the left as row labels. some rows may have no object.
i want to find the object that exists in the "{label} row" , based on this answer the following questions:
1. what is the shape of this object?
2. what is the color of this object?
if there is no object in that row say "none" for both
answer in following format :
shape
color
Do not add extra text or explanation."""

def load_meta(path):
    with open(path,'r') as f:
        m=json.load(f)
    return list(m.values()) if isinstance(m,dict) else list(m)

def parse_two_line_answer(txt):
    lines=[l.strip().lower() for l in re.split(r"\r?\n", txt) if l.strip()]
    if len(lines)>=2: return lines[0], lines[1]
    toks=re.findall(r"[a-z]+", txt.lower())
    if len(toks)>=2: return toks[0], toks[1]
    return None, None

def pick_host_rec_whose_file_exists(host_recs, host_dir):
    """Return (record, adjusted_filename) where adjusted filename exists in host_dir."""
    for r in host_recs:
        fn=r.get("filename")
        if not fn: continue
        p=Path(host_dir)/fn
        if p.exists():
            return r, fn
        # heuristic: try *_with_symbols.png if meta has bare name
        base=Path(fn).stem
        cand=f"{base}_with_symbols.png"
        p2=Path(host_dir)/cand
        if p2.exists():
            r2=dict(r); r2["filename"]=cand
            return r2, cand
    return None, None

def run_once(run_kv, model_dir, data_dir, meta_json, src_fname, tgt_fname,
             row_pairs, prompt_text, out_dir, pad=1, layers="all", greedy=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pfile=Path(out_dir)/"prompt.txt"; pfile.write_text(prompt_text)
    cmd=[
        "python","-u", run_kv,
        "--model_dir", model_dir,
        "--data_dir",  data_dir,
        "--metadata",  meta_json,
        "--source",    src_fname,
        "--target",    tgt_fname,
        "--row_pairs", row_pairs,
        "--pad",       str(pad),
        "--layers",    layers,
        "--prompt_file", str(pfile),
        "--out_dir",   out_dir,
    ]
    if greedy: cmd.append("--greedy")
    cp=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (Path(out_dir)/"run.log").write_text(cp.stdout)

    # prefer split outputs; fallback to result.txt
    base=Path(out_dir)/"baseline.txt"
    inter=Path(out_dir)/"intervened.txt"
    if base.exists() and inter.exists():
        return base.read_text().strip(), inter.read_text().strip()
    res=Path(out_dir)/"result.txt"
    if res.exists():
        t=res.read_text()
        m1=re.search(r"=== BASELINE ===\n(.*?)\n\n=== INTERVENED ===", t, re.S)
        m2=re.search(r"=== INTERVENED ===\n(.*)$", t, re.S)
        return (m1.group(1).strip() if m1 else ""), (m2.group(1).strip() if m2 else "")
    raise RuntimeError(f"Missing outputs in {out_dir}")

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--host_dir", required=True)
    ap.add_argument("--host_meta", required=True)
    ap.add_argument("--inj_dir", required=True)
    ap.add_argument("--inj_meta", required=True)
    ap.add_argument("--run_kv", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--num_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--layers", default="all")
    args=ap.parse_args()

    rng=random.Random(args.seed)
    host_recs=load_meta(args.host_meta)
    inj_recs =load_meta(args.inj_meta)

    host_fns=[r.get("filename") for r in host_recs if r.get("filename")]
    inj_fns =[r.get("filename") for r in inj_recs  if r.get("filename")]
    if not host_fns or not inj_fns:
        raise SystemExit("No filenames found in metadata.")

    out_root=Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    rows=[]

    # sample injector order (deterministic)
    order=list(range(len(inj_fns)))
    rng.shuffle(order)

    for idx,i in enumerate(order[:args.num_pairs]):
        inj_rec = inj_recs[i]
        inj_fn  = inj_rec["filename"]
        inj_path= Path(args.inj_dir)/inj_fn
        if not inj_path.exists():
            # skip injector without image on disk
            (out_root/f"pair_{idx:04d}").mkdir(parents=True, exist_ok=True)
            (out_root/f"pair_{idx:04d}"/"ERROR.txt").write_text(f"injector image missing: {inj_path}")
            continue

        # choose a host whose file exists (or rewrite to *_with_symbols.png)
        host_rec_adj, host_fn = pick_host_rec_whose_file_exists(host_recs, args.host_dir)
        if not host_rec_adj:
            (out_root/f"pair_{idx:04d}").mkdir(parents=True, exist_ok=True)
            (out_root/f"pair_{idx:04d}"/"ERROR.txt").write_text("no host image found on disk matching metadata")
            continue

        sym=rng.choice(list(SYM2ROW.keys()))
        src_row0=SYM2ROW[sym]
        host_row1=rng.choice(HOST_ROWS)  # 1..4

        pair_dir=out_root/f"pair_{idx:04d}"
        work=Path(pair_dir)/"workdir"; work.mkdir(parents=True, exist_ok=True)

        # link images into workdir
        (work/inj_fn).exists()  or os.symlink(inj_path, work/inj_fn)
        (work/host_fn).exists() or os.symlink(Path(args.host_dir)/host_fn, work/host_fn)

        # merged metadata (true injector rec + adjusted host rec)
        (work/"metadata_pair.json").write_text(json.dumps([inj_rec, host_rec_adj], indent=2))

        # prompt about injector symbol
        prompt=PROMPT_TMPL.format(label=sym)

        row_pairs=f"{src_row0+1}->{host_row1}"

        try:
            base_txt, int_txt = run_once(
                args.run_kv, args.model_dir, str(work), str(work/"metadata_pair.json"),
                inj_fn, host_fn, row_pairs, prompt, str(pair_dir),
                pad=args.pad, layers=args.layers, greedy=True
            )
        except Exception as e:
            (Path(pair_dir)/"ERROR.txt").write_text(str(e))
            continue

        b_shape,b_color=parse_two_line_answer(base_txt)
        i_shape,i_color=parse_two_line_answer(int_txt)

        rows.append({
            "pair_idx": idx,
            "injector_file": inj_fn,
            "host_file": host_fn,
            "symbol": sym,
            "injector_row_0based": src_row0,
            "host_row_1based": host_row1,
            "baseline_raw": base_txt,
            "intervened_raw": int_txt,
            "baseline_shape": b_shape, "baseline_color": b_color,
            "intervened_shape": i_shape, "intervened_color": i_color,
        })

    out_csv=out_root/"results_inject_symbols.csv"
    if rows:
        with open(out_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[done] {out_csv} rows={len(rows)}")
    else:
        print(f"[done] {out_csv} rows=0")


if __name__ == "__main__":
    main()
