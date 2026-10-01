from pathlib import Path
import datetime
import hashlib
import json
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


def geometry(pred, target):
    if not np.isfinite(pred).all():
        raise ValueError('Generated vertices contain NaN or Inf')
    vertex_l2 = float(np.linalg.norm(pred - target, axis=1).mean())
    chamfer = float(0.5 * (cKDTree(pred).query(target)[0].mean()
                           + cKDTree(target).query(pred)[0].mean()))
    return {'vertex_l2': vertex_l2, 'chamfer_mean_unsquared': chamfer}


def main():
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    steps, batch_size, lr = 2000, 2, 1e-5
    data_dir = REPO.parent / 'vr_folding_state_est_23frames'
    names = sorted(p.name for p in data_dir.glob('*.h5'))
    expected = [f'00068_Tshirt_000000_{i:06d}.h5' for i in range(45, 156, 5)]
    assert names == expected, 'Expected the prepared 23-frame dataset'

    run = REPO.parent / 'vr_folding_state_est_runs' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / 'train.py')
    patch = run / 'template.pkl'
    shutil.copy2(REPO / 'assets/vr_tshirt_voronoi_template.pkl', patch)
    cfg = yaml.safe_load((REPO / 'configs/train_state_est.yaml').read_text())
    mc, dc, ds = (dict(cfg[k]) for k in ('model_cfg', 'diffusion_cfg', 'dataset_cfg'))
    for c in (mc, dc, ds):
        c.pop('type', None)
    mc['patch_file'] = str(patch)
    ds.update(data_dir=str(data_dir), template_mesh_path=str(patch), do_camera_pose_augmentation=False)
    dc['prediction_type'] = 'epsilon'
    train = ClothStateEstDataset(mode='train', **ds)
    assert train.data_files == names[:21]
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    model = TransformerStateEstV3Model(**mc).to(device).train()
    model.requires_grad_(True)
    diffusion = DDPM_StateEst(**dc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    metadata = dict(model_cfg=mc, diffusion_cfg=dc, dataset_cfg=ds, steps=steps,
                    batch_size=batch_size, learning_rate=lr, optimizer='AdamW',
                    weight_decay=1e-2, lr_schedule='constant', seed=1,
                    train_files=names[:21], same_episode_holdout=names[21:],
                    patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                    torch_version=torch.__version__, device=torch.cuda.get_device_name(0),
                    code_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                    metric_units='original dataset coordinate units',
                    chamfer_definition='0.5*(mean nearest distance A->B + mean nearest distance B->A); unsquared')
    (run / 'config.json').write_text(json.dumps(metadata, indent=2))

    def sample(name, seed):
        with h5py.File(str(data_dir / name), 'r') as f:
            q = np.asarray(f['q'], dtype=np.float32)
            p = np.asarray(f['points'], dtype=np.float32)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(p), generator=g)[:10000].numpy()
        return q, torch.from_numpy(p[idx]).unsqueeze(0).to(device)

    template = train.q_template.unsqueeze(0).to(device)
    cases = []
    for i in (20, 21, 22):
        q, pcd = sample(names[i], 1729 + i)
        with h5py.File(str(data_dir / names[i-1]), 'r') as f:
            previous = np.asarray(f['q'], dtype=np.float32)
        cases.append((names[i], 'train' if i == 20 else 'same_episode_holdout', q, pcd, previous))
    early_q, early_pcd = sample(names[0], 1729)
    _, resampled_pcd = sample(names[21], 2718)

    def save(step):
        checkpoint = dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          model_cfg=mc, diffusion_cfg=dc, metadata=metadata,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                          numpy_rng=np.random.get_state(), python_rng=random.getstate())
        temp = run / 'latest.tmp'
        torch.save(checkpoint, temp)
        temp.replace(run / 'latest.pt')
        print(f'Checkpoint saved: {run / "latest.pt"} (step {step})', flush=True)

    @torch.no_grad()
    def evaluate(step, inference_steps, condition_probe=True):
        model.eval()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1729)
            for name, role, q, pcd, previous in cases:
                scheduler = DDPM_StateEst(**dc)
                scheduler.set_timesteps(inference_steps, device=device)
                generator = torch.Generator(device=device).manual_seed(1729)
                x = torch.randn(template.shape, generator=generator, device=device) * scheduler.init_noise_sigma
                for t in scheduler.timesteps:
                    inputs = torch.cat([scheduler.scale_model_input(x, t), template], dim=-1)
                    eps = model(inputs, timestep=t, encoder_hidden_states=pcd).sample
                    x = scheduler.step(eps, t, x, generator=generator).prev_sample
                pred = x[0].cpu().numpy()
                result = geometry(pred, q)
                baseline = geometry(previous, q)
                record = dict(kind='generation', step=step, inference_steps=inference_steps,
                              seed=1729, frame=name, role=role, **result,
                              previous_gt_oracle=baseline,
                              pred_min=pred.min(axis=0).tolist(), pred_max=pred.max(axis=0).tolist())
                with (run / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                np.savez_compressed(run / f'step{step:04d}_ddpm{inference_steps}_{Path(name).stem}.npz',
                                    pred=pred, q_gt=q, pcd=pcd[0].cpu().numpy(), previous_gt=previous)
                print(f'EVAL step={step} DDPM={inference_steps} {role} {name} '
                      f'vertex_L2={result["vertex_l2"]:.6f} CD={result["chamfer_mean_unsquared"]:.6f} '
                      f'previous_GT_L2={baseline["vertex_l2"]:.6f}', flush=True)
            if condition_probe:
                name, _, q, pcd, _ = cases[1]
                q_tensor = torch.from_numpy(q).unsqueeze(0).to(device)
                g = torch.Generator(device=device).manual_seed(4567)
                noise = torch.randn(q_tensor.shape, generator=g, device=device)
                for time_id in (999, 500):
                    t = torch.tensor([time_id], device=device)
                    xt = diffusion.add_noise(q_tensor, noise, t)
                    inputs = torch.cat([xt, template], dim=-1)
                    outputs = [model(inputs, timestep=t, encoder_hidden_states=p).sample
                               for p in (pcd, resampled_pcd, early_pcd)]
                    record = dict(kind='condition_probe', step=step, frame=name, timestep=time_id,
                                  early_frame_gt_gap=float(np.linalg.norm(q - early_q, axis=1).mean()),
                                  noise_mse=[float((y-noise).square().mean()) for y in outputs],
                                  resampling_delta=float((outputs[0]-outputs[1]).norm(dim=-1).mean()),
                                  early_frame_delta=float((outputs[0]-outputs[2]).norm(dim=-1).mean()))
                    with (run / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    print('CONDITION ' + json.dumps(record), flush=True)
        model.train()

    print(f'RUN: {run}\nTRAIN: 21 frames; HOLDOUT: 2 adjacent frames from same episode; '
          f'updates={steps}; batch={batch_size}; lr={lr}; float32', flush=True)
    print('Geometry distances use dataset units. Previous GT is an oracle baseline.', flush=True)
    iterator = iter(loader)
    total, start = 0.0, time.time()
    with (run / 'loss.csv').open('w', buffering=1) as log:
        log.write('step,noise_mse\n')
        for step in range(1, steps+1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_losses_with_cfg(model, input=batch['q_gt'],
                        model_kwargs={'pcd': batch['pcd'], 'q_temp': batch['q_temp']},
                        weight_dtype=torch.float32)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            value = loss.item()
            total += value
            log.write(f'{step},{value:.9f}\n')
            if step % 20 == 0:
                print(f'TRAIN {step}/{steps} mean_noise_MSE={total/20:.6f} '
                      f'elapsed={time.time()-start:.0f}s', flush=True)
                total = 0.0
            if step % 500 == 0:
                save(step)
                evaluate(step, 50)
    evaluate(steps, 1000, condition_probe=False)
    print(f'DONE: {run}', flush=True)


if __name__ == '__main__':
    main()
