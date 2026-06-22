import os, json, random, csv, re, subprocess
from pathlib import Path

SYM2ROW_INJ = {'!':0, '%':1, '*':2, '+':3}
HOST_ROWS = [1,2,3,4]

QUESTION_TMPL = """Scan the image using the symbols on the left (!, %, *, +) as row labels.
i want to find the object that exists in the "{label} row" , based on this answer the following questions:
1. what is the shape of this object?
2. what is the color of this object?
answer in following format :
shape
color
Do not add extra text or explanation."""

def load_meta(path):
    with open(path,'r') as f:
        m=json.load(f)
    return list(m.values()) if isinstance(m,dict) else list(m)

def find_rec(mvals, filename):
    # exact
    for r in mvals:
        if r.get("filename")==filename: return r
    # try numeric id (e.g., 000)
    import re
    mid = re.search(r'(\d{3,})', filename)
    if mid:
        rid=mid.group(1)
        for r in mvals:
            if rid in str(r.get("filename","")): return r
    # fallback by stem suffix
    if filename:
        stem = Path(filename).stem
        for r in mvals:
            if Path(r.get("filename","")).stem.endswith(stem): return r
    return None

def parse_two_line_answer(txt):
    lines = [l.strip().lower() for l in re.split(r'\r?\n', txt) if l.strip()]
    if len(lines)>=2: return lines[0], lines[1]
    toks = re.findall(r"[a-z]+", txt.lower())
    if len(toks)>=2: return toks[0], toks[1]
    return None, None

def run_once(run_kv, model_dir, data_dir, meta_json,
             src_fname, tgt_fname, row_pairs, prompt_text, out_dir,
             pad=1, layers="all", greedy=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pfile = Path(out_dir)/"prompt.txt"; pfile.write_text(prompt_text)
    cmd = [
        "python","-u", run_kv,
        "--model_dir", model_dir,
        "--data_dir",  data_dir,
        "--metadata",  meta_json,
        "--source",    src_fname,   # capture from INJECTOR
        "--target",    tgt_fname,   # decode on HOST
        "--row_pairs", row_pairs,
        "--pad",       str(pad),
        "--layers",    layers,
        "--prompt_file", str(pfile),
        "--out_dir",   out_dir,
    ]
    if greedy: cmd.append("--greedy")
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (Path(out_dir)/"run.log").write_text(cp.stdout)

    base_path = Path(out_dir)/"baseline.txt"
    int_path  = Path(out_dir)/"intervened.txt"
    if base_path.exists() and int_path.exists():
        return base_path.read_text().strip(), int_path.read_text().strip()

    r = Path(out_dir)/"result.txt"
    if r.exists():
        txt = r.read_text()
        m1 = re.search(r"=== BASELINE ===\n(.*?)\n\n=== INTERVENED ===", txt, re.S)
        m2 = re.search(r"=== INTERVENED ===\n(.*)$", txt, re.S)
        base = m1.group(1).strip() if m1 else ""
        inter= m2.group(1).strip() if m2 else ""
        return base, inter

    raise RuntimeError(f"Missing outputs in {out_dir}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--host_dir", required=True)
    ap.add_argument("--host_meta", required=True)
    ap.add_argument("--inj_dir", required=True)
    ap.add_argument("--inj_meta", required=True)
    ap.add_argument("--run_kv", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_pairs", type=int, default=100)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--pad", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    host_recs = load_meta(args.host_meta)
    inj_recs  = load_meta(args.inj_meta)

    host_fns = [r.get("filename") for r in host_recs if r.get("filename")]
    inj_fns  = [r.get("filename") for r in inj_recs  if r.get("filename")]
    assert host_fns and inj_fns, "No filenames found in metadata."

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(min(args.num_pairs, len(inj_fns))):
        inj_fn = inj_fns[i]
        host_fn = rng.choice(host_fns)
        sym = rng.choice(list(SYM2ROW_INJ.keys()))
        src_row0 = SYM2ROW_INJ[sym]        # 0-based injector row
        host_row1 = rng.choice([1,2,3,4])  # 1..4 host row

        inj_rec  = find_rec(inj_recs, inj_fn)
        host_rec = find_rec(host_recs, host_fn)
        if not inj_rec or not host_rec:
            pair_dir = out_root / f"pair_{i:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            (pair_dir/"ERROR.txt").write_text("could not find both records")
            continue

        pair_dir = out_root / f"pair_{i:04d}"
        work = pair_dir / "workdir"; work.mkdir(parents=True, exist_ok=True)

        inj_src  = Path(args.inj_dir)  / inj_fn
        host_src = Path(args.host_dir) / host_fn
        (work / inj_fn).exists() or os.symlink(inj_src,  work / inj_fn)
        (work / host_fn).exists() or os.symlink(host_src, work / host_fn)

        # merged metadata with both true records
        (work/"metadata_pair.json").write_text(json.dumps([inj_rec, host_rec], indent=2))

        row_pairs = f"{src_row0+1}->{host_row1}"
        prompt = QUESTION_TMPL.format(label=sym)

        try:
            base_txt, int_txt = run_once(
                args.run_kv, args.model_dir, str(work), str(work/"metadata_pair.json"),
                inj_fn, host_fn, row_pairs, prompt, str(pair_dir),
                pad=args.pad, layers=args.layers, greedy=True
            )
        except Exception as e:
            (pair_dir/"ERROR.txt").write_text(str(e))
            continue

        b_shape, b_color = parse_two_line_answer(base_txt)
        i_shape, i_color = parse_two_line_answer(int_txt)

        rows.append({
            "pair_idx": i,
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

    out_csv = Path(args.out_root)/"results_inject_symbols.csv"
    if rows:
        with open(out_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[done] {out_csv}  rows={len(rows)}")
    else:
        print(f"[done] {out_csv}  rows=0")
