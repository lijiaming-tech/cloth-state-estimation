"""Train VR-Folding state estimation from scratch; measure fit on distinct poses."""

from pathlib import Path
import datetime
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time

REPO = Path('/home/agilex/ljm/dyf/UniClothDiff')
sys.path[:0] = [str(REPO), str(REPO / 'third_party/diffusers/src')]

import h5py
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from uniclothdiff.datasets.cloth_state_est import ClothStateEstDataset
from uniclothdiff.models.transformer_state_est_v3 import TransformerStateEstV3Model
from uniclothdiff.schedulers.ddpm_state_est_scheduler import DDPM_StateEst

DATA = REPO.parent / 'vr_folding_state_est_23frames'
FINAL_STEP = 20000
MICROBATCH, ACCUM = 2, 4
BASE_LR, WARMUP_UPDATES = 1e-5, 1000


def measures(pred, target):
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite generated vertices')
    diameter = float(np.linalg.norm(np.ptp(target, axis=0)))
    vertex_l2 = float(np.linalg.norm(pred - target, axis=1).mean())
    return {
        'vertex_l2': vertex_l2,
        'vertex_l2_over_gt_bbox_diagonal': vertex_l2 / diameter,
        'gt_bbox_diagonal': diameter,
        'pred_bbox_span_xyz': np.ptp(pred, axis=0).tolist(),
        'gt_bbox_span_xyz': np.ptp(target, axis=0).tolist(),
        'chamfer_mean_unsquared': float(0.5 * (
            cKDTree(pred).query(target)[0].mean()
            + cKDTree(target).query(pred)[0].mean())),
    }


def lr_at(step):
    if step <= WARMUP_UPDATES:
        return BASE_LR * step / WARMUP_UPDATES
    progress = (step - WARMUP_UPDATES) / (FINAL_STEP - WARMUP_UPDATES)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    names = sorted(p.name for p in DATA.glob('*.h5'))
    expected = [f'00068_Tshirt_000000_{i:06d}.h5' for i in range(45, 156, 5)]
    assert names == expected, 'Only the prepared 23 HDF5 frames may be used'
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    run = REPO.parent / 'vr_folding_capacity_runs' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / 'train_vrfolding_from_scratch.py')
    patch = run / 'template.pkl'
    shutil.copy2(REPO / 'assets/vr_tshirt_voronoi_template.pkl', patch)
    cfg = yaml.safe_load((REPO / 'configs/train_state_est.yaml').read_text())
    mc, dc, ds = (dict(cfg[k]) for k in ('model_cfg', 'diffusion_cfg', 'dataset_cfg'))
    for section in (mc, dc, ds):
        section.pop('type', None)
    mc['patch_file'] = str(patch)
    dc['prediction_type'] = 'epsilon'
    ds.update(data_dir=str(DATA), template_mesh_path=str(patch),
              do_camera_pose_augmentation=False)
    dataset = ClothStateEstDataset(mode='train', **ds)
    assert dataset.data_files == names[:21]
    loader = DataLoader(dataset, batch_size=MICROBATCH, shuffle=True, num_workers=0)
    model = TransformerStateEstV3Model(**mc).to(device)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-2)
    diffusion = DDPM_StateEst(**dc)
    metadata = dict(start_step=0, final_step=FINAL_STEP, microbatch=MICROBATCH,
                    accumulation=ACCUM, effective_batch=MICROBATCH*ACCUM,
                    lr_schedule='1000-update linear warmup, then cosine to zero',
                    base_lr=BASE_LR, seed=123, model_cfg=mc, diffusion_cfg=dc,
                    dataset_cfg=ds, train_files=names[:21],
                    adjacent_same_episode_holdout=names[21:],
                    no_episode_generalization_claim=True,
                    patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                    git_commit=subprocess.check_output(
                        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
    (run / 'config.json').write_text(json.dumps(metadata, indent=2))

    train_gt = []
    for name in names[:21]:
        with h5py.File(str(DATA / name), 'r') as f:
            train_gt.append(np.asarray(f['q'], dtype=np.float32))
    mean_train_pose = np.stack(train_gt).mean(axis=0)

    def read_case(index, seed):
        with h5py.File(str(DATA / names[index]), 'r') as f:
            q = np.asarray(f['q'], dtype=np.float32)
            raw_pcd = np.asarray(f['points'], dtype=np.float32)
        g = torch.Generator().manual_seed(seed)
        inds = torch.randperm(len(raw_pcd), generator=g)[:10000].numpy()
        pcd = torch.from_numpy(raw_pcd[inds]).unsqueeze(0).to(device)
        return q, pcd

    cases = [(names[i], role, *read_case(i, 9100+i))
             for i, role in ((0, 'train_early'), (20, 'train_late'), (21, 'adjacent_holdout'))]
    _, late_resampled_pcd = read_case(20, 19200)
    template = dataset.q_template.unsqueeze(0).to(device)

    @torch.no_grad()
    def generate(pcd, steps=50):
        schedule = DDPM_StateEst(**dc)
        schedule.set_timesteps(steps, device=device)
        rng = torch.Generator(device=device).manual_seed(8101)
        x = torch.randn(template.shape, generator=rng, device=device) * schedule.init_noise_sigma
        for t in schedule.timesteps:
            hidden = torch.cat([schedule.scale_model_input(x, t), template], dim=-1)
            eps = model(hidden, timestep=t, encoder_hidden_states=pcd).sample
            x = schedule.step(eps, t, x, generator=rng).prev_sample
        return x[0].cpu().numpy()

    best_train = float('inf')

    def evaluate(step):
        nonlocal best_train
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(8101)
            records = []
            outputs = {}
            for name, role, q, pcd in cases:
                pred = generate(pcd)
                outputs[role] = pred
                rec = dict(step=step, role=role, frame=name, **measures(pred, q))
                baseline = measures(mean_train_pose, q)
                rec['mean_train_pose_vertex_l2'] = baseline['vertex_l2']
                rec['mean_train_pose_l2_over_gt_bbox_diagonal'] = (
                    baseline['vertex_l2_over_gt_bbox_diagonal'])
                records.append(rec)
                np.savez_compressed(run / f'step{step:05d}_{role}.npz', pred=pred, q_gt=q,
                                    pcd=pcd[0].cpu().numpy())
                print(f'EVAL {step} {role}: vertex_L2={rec["vertex_l2"]:.6f} '
                      f'normalized={rec["vertex_l2_over_gt_bbox_diagonal"]:.3%} '
                      f'mean_train_pose_L2={rec["mean_train_pose_vertex_l2"]:.6f} '
                      f'y_span_pred/gt={rec["pred_bbox_span_xyz"][1]:.4f}/'
                      f'{rec["gt_bbox_span_xyz"][1]:.4f} '
                      f'CD={rec["chamfer_mean_unsquared"]:.6f}', flush=True)
            early_q = cases[0][2]
            late_q = cases[1][2]
            target_gap = float(np.linalg.norm(early_q - late_q, axis=1).mean())
            predicted_gap = float(np.linalg.norm(outputs['train_early'] - outputs['train_late'], axis=1).mean())
            resampling_gap = float(np.linalg.norm(
                outputs['train_late'] - generate(late_resampled_pcd), axis=1).mean())
            print(f'CONDITION {step}: early_to_late_true={target_gap:.6f} '
                  f'early_to_late_pred={predicted_gap:.6f} '
                  f'late_cloud_resampling_pred_change={resampling_gap:.6f}', flush=True)
            records.append(dict(step=step, role='condition_full_generation',
                                early_to_late_true=target_gap,
                                early_to_late_pred=predicted_gap,
                                late_cloud_resampling_pred_change=resampling_gap))
            with (run / 'metrics.jsonl').open('a') as f:
                for item in records:
                    f.write(json.dumps(item) + '\n')
            score = sum(item['vertex_l2_over_gt_bbox_diagonal'] for item in records[:2]) / 2
            if score < best_train:
                best_train = score
                temp = run / 'best_train.tmp'
                torch.save(dict(step=step, model=model.state_dict(), model_cfg=mc,
                                diffusion_cfg=dc, train_fit_score=score), temp)
                temp.replace(run / 'best_train.pt')
        model.train()

    def save(step):
        temp = run / 'latest.tmp'
        torch.save(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        model_cfg=mc, diffusion_cfg=dc, metadata=metadata), temp)
        temp.replace(run / 'latest.pt')

    print(f'RUN: {run}\nTRAIN FROM SCRATCH 0 to {FINAL_STEP}; '
          f'batch={MICROBATCH}x{ACCUM}; train poses 000045 and 000145; '
          'same-episode holdout 000150', flush=True)
    iterator = iter(loader)
    start = time.time()
    with (run / 'loss.csv').open('w', buffering=1) as log:
        log.write('step,mean_noise_mse,learning_rate\n')
        for step in range(1, FINAL_STEP + 1):
            lr = lr_at(step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(ACCUM):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = diffusion.training_losses_with_cfg(
                    model, input=batch['q_gt'],
                    model_kwargs={'pcd': batch['pcd'], 'q_temp': batch['q_temp']},
                    weight_dtype=torch.float32)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at update {step}')
                (loss / ACCUM).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            mean_loss = sum(losses) / len(losses)
            log.write(f'{step},{mean_loss:.8f},{lr:.9g}\n')
            if step % 100 == 0:
                print(f'TRAIN {step}/{FINAL_STEP}: mean_noise_MSE={mean_loss:.6f} '
                      f'lr={lr:.2g} elapsed={time.time()-start:.0f}s', flush=True)
            if step == 100 or step % 500 == 0:
                save(step)
            if step % 1000 == 0:
                evaluate(step)
    print(f'DONE: {run}', flush=True)


if __name__ == '__main__':
    main()
