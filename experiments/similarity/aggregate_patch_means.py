#!/usr/bin/env python3
import argparse, os, numpy as np
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_npz", required=True)
    args=ap.parse_args()

    files=sorted(Path(args.in_dir).glob("pmeans_*.npz"))
    if not files:
        raise SystemExit("no pmeans_*.npz files found")

    # bags: dict symbol -> list of arrays [L,D]
    sym_bag = {}
    obj_bag = {}
    layers_ref=None; D_ref=None

    for fp in files:
        v=np.load(fp, allow_pickle=True)
        layers=v["layers"]; D=int(v["hidden_size"][0])
        if layers_ref is None: layers_ref=layers
        if D_ref is None: D_ref=D
        # symbols present
        for k in v["sym_keys"]:
            arr=v[f"sym_mean__{k}"]    # [L,D]
            sym_bag.setdefault(str(k), []).append(arr)
        for k in v["obj_keys"]:
            arr=v[f"obj_mean__{k}"]
            obj_bag.setdefault(str(k), []).append(arr)

    # stack & mean
    sym_mean_all={}
    for k, lst in sym_bag.items():
        M=np.stack(lst, axis=0).astype(np.float32)  # [N,L,D]
        sym_mean_all[k]=M.mean(axis=0).astype(np.float16)   # [L,D]
    obj_mean_all={}
    for k, lst in obj_bag.items():
        M=np.stack(lst, axis=0).astype(np.float32)
        obj_mean_all[k]=M.mean(axis=0).astype(np.float16)

    # save
    out=Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out),
             layers=layers_ref,
             hidden_size=np.array([D_ref]),
             sym_keys=np.array(list(sym_mean_all.keys())),
             obj_keys=np.array(list(obj_mean_all.keys())),
             **{f"sym_mean__{k}":v for k,v in sym_mean_all.items()},
             **{f"obj_mean__{k}":v for k,v in obj_mean_all.items()},
    )
    print("[save]", out)

if __name__=="__main__":
    main()
