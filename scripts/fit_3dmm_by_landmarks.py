"""
Run landmark-only 3DMM fitting on a target mesh/offset.

Usage:
    python scripts/fit_3dmm_by_landmarks.py \
        --components pca/pca_components.npy \
        --pca_var   pca/pca_explained_variance.npy \
        --mean      pca/pca_mean.npy \
        --faces     data/face.npy \
        --target    data/targets/subject_offset.npy \
        --lm_idx    data/landmarks/vertices_indices.npy \
        --out_obj   outputs/subject_fitted.obj \
        --iters     600 --lr 0.005 --reg 1e-5 --device cuda:0
"""

import os
import argparse
import numpy as np
from src.fitting.landmark_fitter import LandmarkFitter

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", required=True)
    ap.add_argument("--pca_var",   required=True)
    ap.add_argument("--mean",      required=True)
    ap.add_argument("--faces",     required=True)
    ap.add_argument("--target",    required=True)
    ap.add_argument("--lm_idx",    required=True)  # .npy or .txt list of ints
    ap.add_argument("--out_obj",   required=True)
    ap.add_argument("--iters",     type=int, default=500)
    ap.add_argument("--lr",        type=float, default=5e-3)
    ap.add_argument("--reg",       type=float, default=1e-5)
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--weights",   default=None, help="optional npy path (M,) per-landmark weights")
    return ap.parse_args()

def load_indices(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        idx = np.load(path)
    else:
        with open(path, "r") as f:
            idx = [int(x.strip()) for x in f if x.strip()]
        idx = np.asarray(idx, dtype=np.int64)
    return idx.tolist()

def main():
    args = parse_args()
    lm_idx = load_indices(args.lm_idx)
    w = np.load(args.weights).astype(np.float32) if args.weights else None

    fitter = LandmarkFitter(
        components_path=args.components,
        pca_var_path=args.pca_var,
        mean_path=args.mean,
        face_path=args.faces,
        landmark_indices=lm_idx,
        device=args.device,
    )

    p, fitted_vertices, loss = fitter.fit(
        target_path=args.target,
        init_p=None,
        lr=args.lr,
        n_iter=args.iters,
        reg_l2=args.reg,
        weights=w,
        verbose=True,
    )

    fitter.save_mesh(args.out_obj, fitted_vertices)
    mean_err, max_err = fitter.report_landmark_errors(fitted_vertices)
    print(f"[done] mean_lm_err={mean_err:.4f}  max_lm_err={max_err:.4f}")
    print(f"[save] {args.out_obj}")

if __name__ == "__main__":
    main()
