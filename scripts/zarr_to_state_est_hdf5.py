"""Convert VR-Folding Zarr frames into the per-frame HDF5 that ClothStateEstDataset reads.

Requires `zarr` (the `ler` env has it; `clothdiff` does not -- run this with the
env that has zarr, then train with the env that has torkit3d).

Output layout, one file per frame:
    q       (4434, 3) float32   <- mesh/cloth_verts
    points  (30000, 3) float32  <- point_cloud/point   (stored as float16 in the zarr,
                                   widened here: half precision leaves ~3 decimal digits
                                   of positional accuracy, which is too coarse to condition on)
    faces   (8312, 3) int32     <- mesh/cloth_faces_tri  (unused by the loader, kept so a
                                   proper triangle mesh can be exported for visualisation)

Filenames are the sample names, so `sorted(os.listdir(...))` recovers chronological
order and ClothStateEstDataset's 95/5 split lands on the last two frames.
"""
import argparse
from pathlib import Path

import numpy as np
import h5py
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr-root", required=True,
                    help="Path to .../vr_simulation_folding_dataset_example.zarr")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--instance", default="Tshirt")
    ap.add_argument("--pattern", default="00068_Tshirt_000000_",
                    help="Only convert samples whose name starts with this")
    args = ap.parse_args()

    root = zarr.open(str(Path(args.zarr_root) / args.instance), mode="r")
    keys = sorted(k for k in root["samples"].group_keys() if k.startswith(args.pattern))
    if not keys:
        raise SystemExit(f"no samples matching {args.pattern!r} under {args.zarr_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, key in enumerate(keys, 1):
        s = root["samples"][key]
        q = np.asarray(s["mesh/cloth_verts"][:], dtype=np.float32)
        pts = np.asarray(s["point_cloud/point"][:], dtype=np.float32)
        faces = np.asarray(s["mesh/cloth_faces_tri"][:], dtype=np.int32)

        assert q.shape == (4434, 3), f"{key}: q shape {q.shape}"
        assert pts.shape[1] == 3, f"{key}: points shape {pts.shape}"

        path = out_dir / f"{key}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("q", data=q)
            f.create_dataset("points", data=pts)
            f.create_dataset("faces", data=faces)

        if i == 1 or i == len(keys):
            print(f"[{i}/{len(keys)}] {key}  q{q.shape} points{pts.shape} "
                  f"q_range={q.min(0).round(3)}..{q.max(0).round(3)}")

    print(f"\nwrote {len(keys)} files to {out_dir}")


if __name__ == "__main__":
    main()
