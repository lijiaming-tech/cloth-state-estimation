"""Single-episode fit training with multi-seed evaluation and explicit resume.

The upstream model and epsilon loss are unchanged. Each invocation creates a new
run, preserving all legacy checkpoints and experiment records.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'third_party/diffusers/src')]
from vrfolding_protocol import (DEFAULT_SEEDS, evaluate, load_cases, microbatch_weights,
                               protocol, sha256)


def lr_at(step, final_step=20000, base_lr=1e-5, warmup=1000):
    if final_step <= warmup or not 0 <= step <= final_step or warmup <= 0:
        raise ValueError('Invalid learning rate schedule or update step')
    if step <= warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / (final_step - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class BatchStream:
    """Shuffled batches including the tail, with a saved permutation/cursor."""

    def __init__(self, dataset, batch_size, seed, state=None):
        import torch
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor = [], 0
        if state is not None:
            self.order, self.cursor = state['order'], state['cursor']
            if sorted(self.order) != list(range(len(dataset))):
                raise ValueError('Saved permutation does not match the dataset')
            if not 0 <= self.cursor <= len(self.order):
                raise ValueError('Invalid saved data cursor')
            self.generator.set_state(state['generator'])

    def next(self):
        import torch
        from torch.utils.data import default_collate
        if self.cursor == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
            self.cursor = 0
        indices = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(indices)
        samples = [self.dataset[index] for index in indices]
        if any(sample is None for sample in samples):
            raise ValueError(f'Dataset returned an invalid sample: indices={indices}')
        return default_collate(samples)

    def state_dict(self):
        return dict(order=list(self.order), cursor=self.cursor,
                    generator=self.generator.get_state())


def capture_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(value, path):
    import torch
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Exact optimizer-bearing checkpoint; no auto selection')
    parser.add_argument('--data-dir', type=Path, help='Relocated data, with identical file list')
    parser.add_argument('--template', type=Path, help='Relocated template; hash must match on resume')
    parser.add_argument('--output-dir', type=Path, help='New, nonexistent directory')
    parser.add_argument('--total-steps', type=int, help='Fresh-run LR horizon; resume uses saved horizon')
    parser.add_argument('--max-updates', type=int, help='Invocation budget; does not change LR horizon')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--inference-steps', type=int, default=50)
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if min(args.eval_every, args.save_every, args.inference_steps) < 1:
        parser.error('Intervals and inference steps must be positive')
    if len(set(args.eval_seeds)) != len(args.eval_seeds) or len(args.eval_seeds) < 5:
        parser.error('Checkpoint selection requires at least five distinct seeds')
    if args.max_updates is not None and args.max_updates < 1:
        parser.error('--max-updates must be positive; use the evaluator for evaluation only')
    return args


def main():
    args = parse_args()
    import numpy as np
    import torch
    import yaml
    from uniclothdiff.datasets.cloth_state_est import ClothStateEstDataset
    from uniclothdiff.models.transformer_state_est_v3 import TransformerStateEstV3Model
    from uniclothdiff.schedulers.ddpm_state_est_scheduler import DDPM_StateEst

    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    checkpoint, start_step = None, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'optimizer' not in checkpoint:
            raise ValueError('Resume requires optimizer state; best_train is evaluation-only')
        metadata = dict(checkpoint['metadata'])
        mc, dc = dict(checkpoint['model_cfg']), dict(checkpoint['diffusion_cfg'])
        ds = dict(metadata['dataset_cfg'])
        start_step, total = int(checkpoint['step']), int(metadata['final_step'])
        if args.total_steps is not None and args.total_steps != total:
            raise ValueError('Resume must preserve the saved LR horizon')
        microbatch, accum = metadata['microbatch'], metadata['accumulation']
        base_lr, warmup = metadata['base_lr'], metadata.get('warmup_updates', 1000)
        train_files = metadata['train_files']
        holdout_files = metadata['adjacent_same_episode_holdout']
        patch_source = args.template or (args.resume.parent / 'template.pkl')
        if not patch_source.is_file() and args.template is None:
            patch_source = Path(mc['patch_file'])
        if sha256(patch_source) != metadata['patch_sha256']:
            raise ValueError('Template hash differs from the trained checkpoint')
    else:
        cfg = yaml.safe_load((REPO / 'configs/train_state_est.yaml').read_text())
        mc, dc, ds = (dict(cfg[k]) for k in ('model_cfg', 'diffusion_cfg', 'dataset_cfg'))
        for section in (mc, dc, ds):
            section.pop('type', None)
        dc['prediction_type'] = 'epsilon'
        ds['do_camera_pose_augmentation'] = False
        total = args.total_steps if args.total_steps is not None else 20000
        microbatch, accum, base_lr, warmup = 2, 4, 1e-5, 1000
        names = [f'00068_Tshirt_000000_{i:06d}.h5' for i in range(45, 156, 5)]
        train_files, holdout_files = names[:21], names[21:]
        patch_source = args.template or REPO / 'assets/vr_tshirt_voronoi_template.pkl'
        metadata = {}
    lr_at(start_step, total, base_lr, warmup)
    if start_step >= total:
        raise ValueError('Training horizon already completed; use the evaluator')
    if dc.get('prediction_type', 'epsilon') != 'epsilon':
        raise ValueError('Only upstream epsilon loss is supported')
    data = (args.data_dir or Path(ds.get('data_dir', REPO.parent / 'vr_folding_state_est_23frames'))).resolve()
    expected = train_files + holdout_files
    if sorted(p.name for p in data.glob('*.h5')) != expected:
        raise ValueError('Data differs from this single-episode fit experiment')
    data_hashes = {name: sha256(data / name) for name in expected}
    if metadata.get('data_sha256', data_hashes) != data_hashes:
        raise ValueError('Data content changed since checkpoint creation')

    run = args.output_dir or (REPO.parent / 'vr_folding_capacity_runs' /
          (datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_multiseed'))
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / 'train_vrfolding_from_scratch.py')
    shutil.copy2(REPO / 'scripts/vrfolding_protocol.py', run / 'vrfolding_protocol.py')
    patch = run / 'template.pkl'
    shutil.copy2(patch_source, patch)
    mc['patch_file'] = str(patch)
    ds.update(data_dir=str(data), template_mesh_path=str(patch))
    dataset = ClothStateEstDataset(mode='train', **ds)
    if dataset.data_files != train_files:
        raise ValueError('Dataset split differs from the recorded train files')
    template = dataset.q_template.unsqueeze(0).to(device)
    cases, mean_pose = load_cases(data, train_files, holdout_files, template,
                                  ds['num_sample_points'], device)
    eval_protocol = protocol(cases, args.eval_seeds, args.inference_steps)
    model = TransformerStateEstV3Model(**mc).to(device)
    model.requires_grad_(True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
    stream = BatchStream(dataset, microbatch, 123,
                         checkpoint.get('data_stream') if checkpoint else None)
    if checkpoint and 'rng_state' in checkpoint and 'data_stream' in checkpoint:
        restore_rng(checkpoint['rng_state'])
    elif checkpoint:
        print('Legacy resume: optimizer/step restored; original RNG/data cursor were not saved.', flush=True)
    try:
        git_commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        git_commit = 'unavailable'
    metadata.update(start_step=start_step, final_step=total, microbatch=microbatch,
                    accumulation=accum, effective_batch=microbatch * accum,
                    base_lr=base_lr, warmup_updates=warmup,
                    lr_schedule='linear warmup then cosine; original horizon preserved',
                    model_cfg=mc, diffusion_cfg=dc, dataset_cfg=ds,
                    train_files=train_files, adjacent_same_episode_holdout=holdout_files,
                    patch_sha256=sha256(patch), data_sha256=data_hashes,
                    git_commit=git_commit, evaluation_protocol=eval_protocol,
                    resume_source=str(args.resume.resolve()) if args.resume else None,
                    resume_source_has_rng_and_stream=bool(checkpoint and 'rng_state' in checkpoint
                                                          and 'data_stream' in checkpoint),
                    accumulation_weighting='sample-weighted means, including tail',
                    no_episode_generalization_claim=True)
    (run / 'config.json').write_text(json.dumps(metadata, indent=2))
    del checkpoint
    diffusion = DDPM_StateEst(**dc)
    best_score = float('inf')

    def evaluate_and_select(step):
        nonlocal best_score
        summary = evaluate(model, dc, cases, mean_pose, template, args.eval_seeds,
                           args.inference_steps, run / f'eval_step{step:05d}', step)
        if summary['train_fit_score'] < best_score:
            best_score = summary['train_fit_score']
            atomic_save(dict(step=step, model=model.state_dict(), model_cfg=mc,
                             diffusion_cfg=dc, metadata=metadata,
                             train_fit_score=best_score, evaluation_protocol=eval_protocol),
                        run / 'best_train_multiseed.pt')

    def save(step):
        atomic_save(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         model_cfg=mc, diffusion_cfg=dc, metadata=metadata,
                         data_stream=stream.state_dict(), rng_state=capture_rng()), run / 'latest.pt')

    # Re-score the resumed model under THIS protocol before comparing new updates.
    if args.resume:
        evaluate_and_select(start_step)
        save(start_step)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    end_step = min(total, start_step + args.max_updates) if args.max_updates else total
    last_eval, completed = start_step if args.resume else -1, start_step
    print(f'RUN: {run}\nUPDATES {start_step} to {end_step}; LR horizon={total}; '
          f'max batch={microbatch}x{accum}; multi-seed train-fit selection', flush=True)
    start = time.monotonic()
    try:
        with (run / 'loss.csv').open('w', buffering=1) as log:
            log.write('step,mean_noise_mse,learning_rate,actual_samples\n')
            for step in range(start_step + 1, end_step + 1):
                if stop_requested:
                    save(completed)
                    print(f'Stopped at saved update {completed}', flush=True)
                    break
                # Old 21-frame loader tail had batch=1 but got batch=2 weight.
                # Retain every frame and weight each mean loss by actual samples.
                batches = [stream.next() for _ in range(accum)]
                count = sum(len(batch['q_gt']) for batch in batches)
                weights = microbatch_weights([len(batch['q_gt']) for batch in batches])
                lr = lr_at(step, total, base_lr, warmup)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                mean_loss = 0.0
                for batch, weight in zip(batches, weights):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    loss = diffusion.training_losses_with_cfg(
                        model, input=batch['q_gt'],
                        model_kwargs={'pcd': batch['pcd'], 'q_temp': batch['q_temp']},
                        weight_dtype=torch.float32)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss at update {step}')
                    (loss * weight).backward()
                    mean_loss += loss.item() * weight
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                completed = step
                log.write(f'{step},{mean_loss:.8f},{lr:.9g},{count}\n')
                if step % 100 == 0:
                    print(f'TRAIN {step}/{end_step}: noise_MSE={mean_loss:.6f} '
                          f'lr={lr:.2g} elapsed={time.monotonic()-start:.0f}s', flush=True)
                if step == 100 or step % args.save_every == 0 or step == end_step or stop_requested:
                    save(step)
                if stop_requested:
                    print(f'Stopped after complete update {step}; saved {run / "latest.pt"}', flush=True)
                    break
                if step % args.eval_every == 0:
                    evaluate_and_select(step)
                    last_eval = step
            if not stop_requested and completed != last_eval:
                evaluate_and_select(completed)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    print(f'DONE: {run}', flush=True)


if __name__ == '__main__':
    main()
