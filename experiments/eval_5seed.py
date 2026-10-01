"""Evaluate explicit checkpoints using the same protocol as the training loop.

Example: python experiments/eval_5seed.py --run /path/to/training/run
No training, automatic run selection, or checkpoint overwrites occur.
"""
import argparse
import datetime
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO), str(REPO / 'third_party/diffusers/src')]
from vrfolding_protocol import DEFAULT_SEEDS, evaluate, load_cases, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='+', default=['best_train.pt', 'latest.pt'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--output-dir', type=Path, help='New, nonexistent directory')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--inference-steps', type=int, default=50)
    parser.add_argument('--list-checkpoints', action='store_true', help='Read checkpoint steps on CPU and exit')
    args = parser.parse_args()
    if len(args.seeds) < 5 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Provide at least five distinct seeds')
    if args.inference_steps < 1:
        parser.error('Inference steps must be positive')
    import torch
    run = args.run.resolve()
    paths = [run / name for name in args.checkpoints]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.list_checkpoints:
        for path in paths:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            print(f'{path}: step={ckpt["step"]}, optimizer={"optimizer" in ckpt}, bytes={path.stat().st_size}')
            del ckpt
        return

    from uniclothdiff.models.transformer_state_est_v3 import TransformerStateEstV3Model
    import numpy as np
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    config = json.loads((run / 'config.json').read_text())
    data = args.data_dir or Path(config['dataset_cfg']['data_dir'])
    train_files, holdout = config['train_files'], config['adjacent_same_episode_holdout']
    names = train_files + holdout
    if sorted(path.name for path in data.glob('*.h5')) != names:
        raise ValueError('Data file list differs from the recorded experiment')
    if 'data_sha256' in config:
        if {name: sha256(data / name) for name in names} != config['data_sha256']:
            raise ValueError('Data content differs from the recorded experiment')
    patch = args.template or run / 'template.pkl'
    if not patch.is_file() and args.template is None:
        patch = Path(config['model_cfg']['patch_file'])
    if sha256(patch) != config['patch_sha256']:
        raise ValueError('Template hash does not match the experiment')
    with patch.open('rb') as handle:
        template = torch.tensor(pickle.load(handle)['points'], dtype=torch.float32).unsqueeze(0).to(device)
    cases, mean_pose = load_cases(data, train_files, holdout, template,
                                  config['dataset_cfg']['num_sample_points'], device)
    output = args.output_dir or (run / ('eval_offline_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index, path in enumerate(paths):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if ckpt['model_cfg'] != config['model_cfg'] or ckpt['diffusion_cfg'] != config['diffusion_cfg']:
            raise ValueError(f'Checkpoint configuration differs from run config: {path}')
        model_cfg = dict(ckpt['model_cfg'])
        model_cfg['patch_file'] = str(patch)
        model = TransformerStateEstV3Model(**model_cfg).to(device)
        model.load_state_dict(ckpt['model'], strict=True)
        step, diffusion_cfg = int(ckpt['step']), dict(ckpt['diffusion_cfg'])
        del ckpt
        model.eval()
        summary = evaluate(model, diffusion_cfg, cases, mean_pose, template,
                           args.seeds, args.inference_steps,
                           output / f'{index}_{path.stem}_step{step:05d}', step, path.name)
        summaries.append(summary)
        del model
        torch.cuda.empty_cache()
    # Same seeds across candidates permit a paired comparison; this is fit only.
    paired = []
    reference = summaries[0]
    for candidate in summaries[1:]:
        differences = np.asarray(candidate['train_score_by_seed']) - reference['train_score_by_seed']
        paired.append(dict(reference=reference['checkpoint'], candidate=candidate['checkpoint'],
                           candidate_minus_reference_by_seed=differences.tolist(),
                           difference_mean=float(differences.mean()),
                           difference_std=float(differences.std(ddof=1))))
    (output / 'comparison.json').write_text(json.dumps(dict(summaries=summaries, paired=paired), indent=2))
    print(f'Results: {output}; same-episode fit only.', flush=True)


if __name__ == '__main__':
    main()
