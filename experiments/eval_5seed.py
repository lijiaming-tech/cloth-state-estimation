from pathlib import Path
import csv
import hashlib
import json
import sys

REPO = Path("/home/agilex/ljm/dyf/UniClothDiff")
RUN = Path("/home/agilex/ljm/dyf/vr_folding_capacity_runs/20260926_173508")
OUT = Path(__file__).parent
sys.path[:0] = [str(REPO), str(REPO / "third_party/diffusers/src")]

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from uniclothdiff.datasets.cloth_state_est import ClothStateEstDataset
from uniclothdiff.models.transformer_state_est_v3 import TransformerStateEstV3Model
from uniclothdiff.schedulers.ddpm_state_est_scheduler import DDPM_StateEst

torch.cuda.set_device(0)
device = torch.device("cuda:0")
config = json.loads((RUN / "config.json").read_text())
data = Path(config["dataset_cfg"]["data_dir"])
names = sorted(p.name for p in data.glob("*.h5"))
assert names == config["train_files"] + config["adjacent_same_episode_holdout"]
patch = Path(config["model_cfg"]["patch_file"])
assert hashlib.sha256(patch.read_bytes()).hexdigest() == config["patch_sha256"]

dataset = ClothStateEstDataset(mode="train", **config["dataset_cfg"])
assert dataset.data_files == config["train_files"]
template = dataset.q_template.unsqueeze(0).to(device)

def read_case(index):
    name = names[index]
    with h5py.File(data / name, "r") as f:
        q = np.asarray(f["q"], dtype=np.float32)
        raw = np.asarray(f["points"], dtype=np.float32)
    generator = torch.Generator().manual_seed(9100 + index)
    indices = torch.randperm(len(raw), generator=generator)[:10000].numpy()
    pcd = torch.from_numpy(raw[indices]).unsqueeze(0).to(device)
    return name, q, pcd

cases = [
    ("train_early", *read_case(0)),
    ("train_late", *read_case(20)),
    ("adjacent_holdout", *read_case(21)),
]
train_gt = []
for name in config["train_files"]:
    with h5py.File(data / name, "r") as f:
        train_gt.append(np.asarray(f["q"], dtype=np.float32))
mean_train_pose = np.stack(train_gt).mean(axis=0)

@torch.no_grad()
def generate(model, diffusion_cfg, pcd, seed):
    schedule = DDPM_StateEst(**diffusion_cfg)
    schedule.set_timesteps(50, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(template.shape, generator=rng, device=device)
    x = x * schedule.init_noise_sigma
    for t in schedule.timesteps:
        hidden = torch.cat([schedule.scale_model_input(x, t), template], dim=-1)
        eps = model(hidden, timestep=t, encoder_hidden_states=pcd).sample
        x = schedule.step(eps, t, x, generator=rng).prev_sample
    return x[0].cpu().numpy()

records = []
train_scores = {}
for checkpoint_name in ("best_train.pt", "latest.pt"):
    ckpt = torch.load(RUN / checkpoint_name, map_location="cpu")
    step = int(ckpt["step"])
    assert ckpt["model_cfg"] == config["model_cfg"]
    assert ckpt["diffusion_cfg"] == config["diffusion_cfg"]
    model = TransformerStateEstV3Model(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion_cfg = ckpt["diffusion_cfg"]
    del ckpt

    per_seed = {seed: [] for seed in range(5)}
    print(f"\nCHECKPOINT {checkpoint_name}, step={step}", flush=True)
    for role, name, target, pcd in cases:
        frame_errors = []
        diameter = float(np.linalg.norm(np.ptp(target, axis=0)))
        baseline = float(np.linalg.norm(mean_train_pose - target, axis=1).mean())
        for seed in range(5):
            pred = generate(model, diffusion_cfg, pcd, seed)
            if not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite prediction: step={step}, {role}, seed={seed}")
            l2 = float(np.linalg.norm(pred - target, axis=1).mean())
            normalized = l2 / diameter
            pred_span = np.ptp(pred, axis=0)
            gt_span = np.ptp(target, axis=0)
            chamfer = float(0.5 * (
                cKDTree(pred).query(target)[0].mean()
                + cKDTree(target).query(pred)[0].mean()
            ))
            records.append(dict(
                checkpoint=checkpoint_name, step=step, role=role,
                frame=name, seed=seed, vertex_l2=l2,
                vertex_l2_over_gt_bbox_diagonal=normalized,
                mean_train_pose_vertex_l2=baseline,
                chamfer_mean_unsquared=chamfer,
                pred_y_span=float(pred_span[1]),
                gt_y_span=float(gt_span[1]),
            ))
            np.savez_compressed(
                OUT / f"step{step:05d}_{role}_seed{seed}.npz",
                pred=pred, q_gt=target, pcd=pcd[0].cpu().numpy()
            )
            frame_errors.append(l2)
            if role.startswith("train_"):
                per_seed[seed].append(normalized)
        print(
            f"{role}: L2={np.mean(frame_errors):.6f} ± "
            f"{np.std(frame_errors, ddof=1):.6f}; "
            f"five seeds={[round(x, 6) for x in frame_errors]}; "
            f"train-mean baseline={baseline:.6f}",
            flush=True,
        )
    train_scores[checkpoint_name] = [
        float(np.mean(per_seed[seed])) for seed in range(5)
    ]
    scores = train_scores[checkpoint_name]
    print(
        f"TRAIN SCORE, normalized L2 across two training frames: "
        f"{np.mean(scores):.3%} ± {np.std(scores, ddof=1):.3%}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()

with (OUT / "results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(OUT / "summary.json").write_text(json.dumps({
    "seeds": list(range(5)),
    "inference_steps": 50,
    "checkpoint_train_scores_by_seed": train_scores,
    "note": "Adjacent holdout is from the same episode; it is not a generalization test.",
}, indent=2))
print("\nPaired train-score difference, latest minus best, by seed:",
      [round(b-a, 6) for a, b in zip(
          train_scores["best_train.pt"], train_scores["latest.pt"])])
print("Results and reusable script:", OUT, flush=True)
