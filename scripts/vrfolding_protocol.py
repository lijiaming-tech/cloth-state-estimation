"""Shared geometry and sampling protocol for the single-episode fit experiment.

No model architecture or upstream loss changes. Heavy training imports are lazy,
so metrics and CLI help also work without the CUDA environment.
"""

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

PROTOCOL_VERSION = "vrfolding-fit-v2"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def microbatch_weights(sizes):
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Microbatches must contain samples")
    total = sum(sizes)
    return [size / total for size in sizes]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measures(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Prediction and target must have matching (V, 3) shapes")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite vertices")
    diameter = float(np.linalg.norm(np.ptp(target, axis=0)))
    if diameter <= 0:
        raise ValueError("Degenerate target bounding box")
    l2 = float(np.linalg.norm(pred - target, axis=1).mean())
    return dict(
        vertex_l2=l2, vertex_l2_over_gt_bbox_diagonal=l2 / diameter,
        gt_bbox_diagonal=diameter,
        pred_bbox_span_xyz=np.ptp(pred, axis=0).tolist(),
        gt_bbox_span_xyz=np.ptp(target, axis=0).tolist(),
        chamfer_mean_unsquared=float(0.5 * (
            cKDTree(pred).query(target)[0].mean()
            + cKDTree(target).query(pred)[0].mean())),
    )


def summarize(records, seeds):
    """Frame statistics and equal-weighted training-frame score, paired by seed."""
    seeds = list(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two distinct seeds")
    groups = {}
    for record in records:
        groups.setdefault((record["role"], record["frame"]), []).append(record)
    frames, scores = [], {seed: [] for seed in seeds}
    for (role, frame), rows in groups.items():
        if sorted(row["seed"] for row in rows) != sorted(seeds):
            raise ValueError(f"Incomplete or duplicate seed results for {frame}")
        item = dict(role=role, frame=frame, n_seeds=len(seeds))
        for metric in ("vertex_l2", "vertex_l2_over_gt_bbox_diagonal",
                       "chamfer_mean_unsquared", "pred_y_span"):
            values = [row[metric] for row in rows]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {metric}")
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_std"] = float(np.std(values, ddof=1))
        item["mean_train_pose_vertex_l2"] = rows[0]["mean_train_pose_vertex_l2"]
        item["gt_y_span"] = rows[0]["gt_y_span"]
        frames.append(item)
        if role.startswith("train_"):
            for row in rows:
                scores[row["seed"]].append(row["vertex_l2_over_gt_bbox_diagonal"])
    if not all(scores.values()):
        raise ValueError("No training frames available for checkpoint selection")
    by_seed = [float(np.mean(scores[seed])) for seed in seeds]
    return dict(frames=frames, train_score_by_seed=by_seed,
                train_fit_score=float(np.mean(by_seed)),
                train_fit_score_std=float(np.std(by_seed, ddof=1)),
                no_episode_generalization_claim=True)


def protocol(cases, seeds, steps):
    return dict(version=PROTOCOL_VERSION, seeds=list(seeds), inference_steps=steps,
                point_cloud_sampling="torch CPU randperm, seed=9100+file_index",
                cases=[dict(role=role, frame=name) for role, name, *_ in cases],
                selection="mean normalized vertex L2 on training cases across seeds")


def load_cases(data, train_files, holdout_files, template, num_points, device):
    import h5py
    import torch

    data = Path(data)
    names = train_files + holdout_files
    targets = []
    for name in train_files:
        with h5py.File(data / name, "r") as f:
            q = np.asarray(f["q"], dtype=np.float32)
        if q.shape != tuple(template.shape[1:]) or not np.isfinite(q).all():
            raise ValueError(f"Invalid template correspondence/vertices: {name}")
        targets.append(q)
    mean_pose = np.stack(targets).mean(axis=0)
    # Keep the two historic training cases; evaluate BOTH held-out frames.
    chosen = [(0, "train_early"), (len(train_files) - 1, "train_late")]
    chosen += [(len(train_files) + i, "adjacent_holdout")
               for i in range(len(holdout_files))]
    cases = []
    for index, role in chosen:
        name = names[index]
        with h5py.File(data / name, "r") as f:
            q = np.asarray(f["q"], dtype=np.float32)
            raw = np.asarray(f["points"], dtype=np.float32)
        if q.shape != mean_pose.shape or not np.isfinite(q).all():
            raise ValueError(f"Invalid target: {name}")
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < num_points:
            raise ValueError(f"Expected at least {num_points} XYZ points: {name}")
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite point cloud: {name}")
        generator = torch.Generator().manual_seed(9100 + index)
        indices = torch.randperm(len(raw), generator=generator)[:num_points].numpy()
        pcd = torch.from_numpy(raw[indices]).unsqueeze(0).to(device)
        cases.append((role, name, q, pcd))
    return cases, mean_pose


def generate(model, diffusion_cfg, pcd, template, seed, steps, scheduler_cls=None):
    import torch
    if scheduler_cls is None:
        from uniclothdiff.schedulers.ddpm_state_est_scheduler import DDPM_StateEst
        scheduler_cls = DDPM_StateEst
    if diffusion_cfg.get("prediction_type", "epsilon") != "epsilon":
        raise ValueError("This experiment uses the upstream epsilon prediction loss")
    device = template.device
    # Some model operators may use global Torch RNG. Isolate those too, per seed,
    # and leave training RNG untouched. Generator controls DDPM noise separately.
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            scheduler = scheduler_cls(**diffusion_cfg)
            scheduler.set_timesteps(steps, device=device)
            rng = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(template.shape, generator=rng, device=device,
                            dtype=template.dtype) * scheduler.init_noise_sigma
            for t in scheduler.timesteps:
                hidden = torch.cat([scheduler.scale_model_input(x, t), template], dim=-1)
                eps = model(hidden, timestep=t, encoder_hidden_states=pcd).sample
                x = scheduler.step(eps, t, x, generator=rng).prev_sample
            return x[0].cpu().numpy()
    finally:
        model.train(was_training)


def evaluate(model, diffusion_cfg, cases, mean_pose, template, seeds, steps,
             output, step, checkpoint_name="in_memory"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    for role, name, target, pcd in cases:
        baseline = measures(mean_pose, target)["vertex_l2"]
        for seed in seeds:
            pred = generate(model, diffusion_cfg, pcd, template, seed, steps)
            values = measures(pred, target)
            row = dict(checkpoint=checkpoint_name, step=step, role=role, frame=name,
                       seed=seed, **values, mean_train_pose_vertex_l2=baseline,
                       pred_y_span=values["pred_bbox_span_xyz"][1],
                       gt_y_span=values["gt_bbox_span_xyz"][1])
            records.append(row)
            np.savez_compressed(output / f'{Path(name).stem}_seed{seed}.npz',
                                pred=pred, q_gt=target, pcd=pcd[0].cpu().numpy())
    summary = summarize(records, seeds)
    summary.update(step=step, checkpoint=checkpoint_name,
                   protocol=protocol(cases, seeds, steps),
                   evaluation_seconds=time.monotonic() - started)
    (output / "results.json").write_text(json.dumps(records, indent=2))
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    for item in summary["frames"]:
        print(f'EVAL {step} {item["frame"]}: '
              f'L2={item["vertex_l2_mean"]:.6f} ± {item["vertex_l2_std"]:.6f}; '
              f'normalized={item["vertex_l2_over_gt_bbox_diagonal_mean"]:.3%}; '
              f'train-mean baseline={item["mean_train_pose_vertex_l2"]:.6f}', flush=True)
    return summary
