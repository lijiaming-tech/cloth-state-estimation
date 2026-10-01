"""Build the 4434-vertex VR-Folding T-shirt template from the Zarr NOCS field.

The original template was produced by `tmpl_patchify.py` fed with an OBJ, where
`trimesh.load` silently merged 4434 vertices down to 4252. This script feeds the
(4434, 3) NOCS array straight into the same normalization and the same
FPS+Voronoi partition, so the vertex order matches `q_gt` exactly.

Normalization matches tmpl_patchify.py:274-277:  points = nocs * 0.5 - mean(nocs * 0.5)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from tmpl_patchify import split_point_cloud_with_voronoi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zarr-root",
                     help="Path to .../vr_simulation_folding_dataset_example.zarr (needs zarr)")
    src.add_argument("--nocs-npy",
                     help="Pre-exported (4434, 3) float32 NOCS array (no zarr needed)")
    ap.add_argument("--frame", default=None,
                    help="Sample name to source NOCS from (default: first sample)")
    ap.add_argument("--num-patches", type=int, default=100)
    ap.add_argument("--out", required=True, help="Output pickle path")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.nocs_npy:
        # zarr lives in a different env than torkit3d, so allow a pre-exported array.
        nocs = np.load(args.nocs_npy).astype(np.float32)
        frame = f"{args.nocs_npy} (pre-exported)"
    else:
        import zarr

        root = zarr.open(str(Path(args.zarr_root) / "Tshirt"), mode="r")
        keys = sorted(root["samples"].group_keys())
        frame = args.frame or keys[0]
        nocs = np.asarray(root["samples"][frame]["mesh/cloth_nocs_verts"][:], dtype=np.float32)

    if nocs.shape != (4434, 3):
        raise SystemExit(f"unexpected NOCS shape {nocs.shape}, expected (4434, 3)")

    points = nocs * 0.5
    points = points - points.mean(axis=0, keepdims=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    pts = torch.from_numpy(points[None]).to(device)  # (1, N, 3)
    patches, patches_idx, centers, fps_idx = split_point_cloud_with_voronoi(
        pts, n_patches=args.num_patches
    )

    # Verify the partition covers every vertex exactly once. The partitioner drops
    # empty patches, so a silent gap would misalign every downstream vertex index.
    all_idx = np.concatenate(patches_idx)
    n = points.shape[0]
    assert len(all_idx) == n, f"coverage {len(all_idx)} != {n} vertices"
    assert len(np.unique(all_idx)) == n, "duplicate vertex assignments"
    assert all_idx.min() == 0 and all_idx.max() == n - 1, "index range does not span all vertices"
    assert len(patches) == args.num_patches, (
        f"got {len(patches)} non-empty patches, expected {args.num_patches}"
    )

    data = {
        "patch_points": patches,                 # list[np.ndarray (Pi, 3)]
        "patch_index": patches_idx,              # list[np.ndarray (Pi,)]
        "centers": centers[0].cpu().numpy(),     # (num_patches, 3)
        "center_idx": fps_idx[0].cpu().numpy(),  # (num_patches,)
        "points": points,                        # (4434, 3) normalized
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data, f)

    sizes = np.array([len(p) for p in patches_idx])
    print(f"NOCS source frame : {frame}")
    print(f"vertices          : {n}  (all covered exactly once)")
    print(f"patches           : {len(patches)}  sizes min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")
    print(f"points range      : {points.min(0).round(4)} .. {points.max(0).round(4)}")
    print(f"saved             : {out}")


if __name__ == "__main__":
    main()
