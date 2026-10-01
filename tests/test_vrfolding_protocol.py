"""CPU tests for evaluation bookkeeping, RNG isolation and continuation mechanics.

Uses a tiny model/scheduler for sampling tests, not the CUDA cloth model.
"""
import csv
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import vrfolding_protocol as vp
import train_vrfolding_from_scratch as trainer
from train_vrfolding_from_scratch import BatchStream, atomic_save, capture_rng, lr_at, restore_rng


class TinyModel(torch.nn.Module):
    def forward(self, hidden, timestep, encoder_hidden_states):
        # Deliberately uses GLOBAL RNG as well as the scheduler's local generator.
        return SimpleNamespace(sample=hidden[..., :3] * 0.1 + torch.rand_like(hidden[..., :3]))


class TinyScheduler:
    init_noise_sigma = 1
    def __init__(self, **kwargs):
        pass
    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps - 1, -1, -1, device=device)
    def scale_model_input(self, x, t):
        return x
    def step(self, eps, t, x, generator):
        return SimpleNamespace(prev_sample=x - eps * 0.2 +
                               torch.randn(x.shape, generator=generator) * 0.01)


class ProtocolTests(unittest.TestCase):
    def test_geometry_and_correspondence(self):
        target = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        self.assertEqual(vp.measures(target, target)['vertex_l2'], 0)
        shifted = target + [0, 0, 2]
        self.assertAlmostEqual(vp.measures(shifted, target)['vertex_l2'], 2)
        permuted = target[[1, 0, 2]]
        self.assertEqual(vp.measures(permuted, target)['chamfer_mean_unsquared'], 0)
        self.assertGreater(vp.measures(permuted, target)['vertex_l2'], 0)
        for bad in (np.full_like(target, np.nan), target[:1]):
            with self.assertRaises(ValueError):
                vp.measures(bad, target)
        with self.assertRaises(ValueError):
            vp.measures(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_historical_multiseed_scores_and_holdout_exclusion(self):
        with (REPO / 'experiments/eval_5seed_results.csv').open() as handle:
            raw = list(csv.DictReader(handle))
        history = json.loads((REPO / 'experiments/eval_5seed_summary.json').read_text())
        for name, expected in history['checkpoint_train_scores_by_seed'].items():
            rows = []
            for row in raw:
                if row['checkpoint'] != name:
                    continue
                row = dict(row)
                for key in ('vertex_l2', 'vertex_l2_over_gt_bbox_diagonal',
                            'mean_train_pose_vertex_l2', 'chamfer_mean_unsquared',
                            'pred_y_span', 'gt_y_span'):
                    row[key] = float(row[key])
                row['seed'] = int(row['seed'])
                rows.append(row)
            summary = vp.summarize(rows, list(range(5)))
            np.testing.assert_allclose(summary['train_score_by_seed'], expected)
            for row in rows:
                if row['role'] == 'adjacent_holdout':
                    row['vertex_l2_over_gt_bbox_diagonal'] = 1000
            self.assertEqual(summary['train_fit_score'], vp.summarize(rows, range(5))['train_fit_score'])
            with self.assertRaises(ValueError):
                vp.summarize(rows[:-1], range(5))

    def test_generation_reproducible_and_preserves_rng_and_mode(self):
        model = TinyModel().train()
        template = torch.zeros(1, 4, 3)
        pcd = torch.zeros(1, 8, 3)
        state = torch.get_rng_state().clone()
        first = vp.generate(model, {}, pcd, template, 0, 3, TinyScheduler)
        np.testing.assert_array_equal(state.numpy(), torch.get_rng_state().numpy())
        self.assertTrue(model.training)
        np.testing.assert_array_equal(first, vp.generate(model, {}, pcd, template, 0, 3, TinyScheduler))
        self.assertFalse(np.array_equal(first, vp.generate(model, {}, pcd, template, 1, 3, TinyScheduler)))
        class FailingScheduler(TinyScheduler):
            def step(self, *args, **kwargs):
                raise RuntimeError('sample failure')
        with self.assertRaises(RuntimeError):
            vp.generate(model, {}, pcd, template, 0, 3, FailingScheduler)
        self.assertTrue(model.training)
        np.testing.assert_array_equal(state.numpy(), torch.get_rng_state().numpy())

    def test_short_tail_gradient_matches_full_batch(self):
        batches = [torch.tensor([1., 2.]), torch.tensor([3.])]
        w = torch.tensor(0.5, requires_grad=True)
        for batch, weight in zip(batches, vp.microbatch_weights([2, 1])):
            (((w * batch - 1) ** 2).mean() * weight).backward()
        full = torch.tensor(0.5, requires_grad=True)
        ((full * torch.cat(batches) - 1) ** 2).mean().backward()
        torch.testing.assert_close(w.grad, full.grad)

    def test_stream_roundtrip_keeps_tail_and_future_epochs(self):
        dataset = [{'q_gt': torch.tensor([i])} for i in range(21)]
        stream = BatchStream(dataset, 2, 123)
        seen = [stream.next()['q_gt'].flatten().tolist() for _ in range(11)]
        self.assertEqual(sorted(sum(seen, [])), list(range(21)))
        self.assertEqual(len(seen[-1]), 1)
        state = stream.state_dict()
        resumed = BatchStream(dataset, 2, 123, state)
        for _ in range(20):
            torch.testing.assert_close(stream.next()['q_gt'], resumed.next()['q_gt'])

    def test_rng_optimizer_and_stream_checkpoint_roundtrip(self):
        dataset = [{'q_gt': torch.tensor([float(i)])} for i in range(5)]
        stream = BatchStream(dataset, 2, 123)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def update(m, opt, batches):
            opt.zero_grad()
            x = batches.next()['q_gt']
            x = x + float(np.random.uniform()) + random.random()
            (m(x).square().mean() + m(x).mean() * torch.rand(())).backward()
            opt.step()
        update(model, optimizer, stream)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'latest.pt'
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             data_stream=stream.state_dict(), rng_state=capture_rng()), path)
            self.assertFalse(path.with_suffix('.tmp').exists())
            update(model, optimizer, stream)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            checkpoint = torch.load(path, weights_only=False)
            restored = torch.nn.Linear(1, 1)
            restored.load_state_dict(checkpoint['model'])
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_opt.load_state_dict(checkpoint['optimizer'])
            restored_stream = BatchStream(dataset, 2, 123, checkpoint['data_stream'])
            restore_rng(checkpoint['rng_state'])
            update(restored, restored_opt, restored_stream)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_learning_rate_resume_preserves_horizon(self):
        self.assertEqual(lr_at(0), 0)
        self.assertEqual(lr_at(1000), 1e-5)
        self.assertAlmostEqual(lr_at(20000), 0)
        self.assertGreater(lr_at(12501), lr_at(13500))
        with self.assertRaises(ValueError):
            lr_at(20001)

    def test_fixed_point_cloud_and_all_holdouts(self):
        with tempfile.TemporaryDirectory() as folder:
            names = ['a.h5', 'b.h5', 'c.h5', 'd.h5']
            target = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            for i, name in enumerate(names):
                with h5py.File(Path(folder) / name, 'w') as handle:
                    handle['q'] = target + i
                    handle['points'] = np.arange(30, dtype=np.float32).reshape(10, 3)
            template = torch.zeros(1, 3, 3)
            cases, mean = vp.load_cases(folder, names[:2], names[2:], template, 4, 'cpu')
            again, _ = vp.load_cases(folder, names[:2], names[2:], template, 4, 'cpu')
            self.assertEqual(len(cases), 4)
            np.testing.assert_allclose(mean, target + 0.5)
            for first, second in zip(cases, again):
                torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
            with patch.object(vp, 'generate', return_value=target):
                output = Path(folder) / 'evaluation'
                summary = vp.evaluate(TinyModel(), {}, cases, mean, template, range(5), 50, output, 123)
            self.assertEqual(len(summary['frames']), 4)
            self.assertEqual(len(list(output.glob('*.npz'))), 20)
            with (output / 'results.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 20)
            with self.assertRaises(FileExistsError):
                vp.evaluate(TinyModel(), {}, cases, mean, template, range(5), 50, output, 123)

    def test_training_entrypoint_legacy_and_new_resume(self):
        """Exercise the real entrypoint with CPU tensors and substitute CUDA model."""
        class Dataset:
            def __init__(self, data_dir, template_mesh_path, **kwargs):
                self.data_files = sorted(path.name for path in Path(data_dir).glob('*.h5'))[:21]
                with open(template_mesh_path, 'rb') as handle:
                    self.q_template = torch.tensor(pickle.load(handle)['points'])
                self.data = Path(data_dir)
            def __len__(self):
                return len(self.data_files)
            def __getitem__(self, index):
                with h5py.File(self.data / self.data_files[index]) as handle:
                    q = torch.from_numpy(handle['q'][:])
                return dict(q_gt=q, pcd=torch.rand(8, 3), q_temp=self.q_template)
        class Model(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.layer = torch.nn.Linear(6, 3)
            def forward(self, hidden, **kwargs):
                return SimpleNamespace(sample=self.layer(hidden))
        class Scheduler(TinyScheduler):
            def training_losses_with_cfg(self, model, input, model_kwargs, **kwargs):
                noise = torch.randn_like(input)
                hidden = torch.cat([input + noise, model_kwargs['q_temp']], dim=-1)
                return (model(hidden, **model_kwargs).sample - noise).square().mean()
        modules = {}
        for name, key, value in [
            ('uniclothdiff.datasets.cloth_state_est', 'ClothStateEstDataset', Dataset),
            ('uniclothdiff.models.transformer_state_est_v3', 'TransformerStateEstV3Model', Model),
            ('uniclothdiff.schedulers.ddpm_state_est_scheduler', 'DDPM_StateEst', Scheduler),
        ]:
            module = ModuleType(name)
            setattr(module, key, value)
            modules[name] = module
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and isinstance(args[0], torch.device) and args[0].type == 'cuda':
                args = (torch.device('cpu'),) + args[1:]
            return original_to(tensor, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            data.mkdir()
            target = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            for i in range(45, 156, 5):
                with h5py.File(data / f'00068_Tshirt_000000_{i:06d}.h5', 'w') as handle:
                    handle['q'] = target
                    handle['points'] = np.zeros((10000, 3), dtype=np.float32)
            template = root / 'template.pkl'
            template.write_bytes(pickle.dumps(dict(points=target)))
            def run(output, extra):
                argv = ['train', '--data-dir', str(data), '--output-dir', str(output),
                        '--template', str(template), '--inference-steps', '2'] + extra
                with patch.object(sys, 'argv', argv):
                    trainer.main()
            with patch.dict(sys.modules, modules), patch.object(torch.cuda, 'set_device'), \
                    patch.object(torch.Tensor, 'to', cpu_to):
                first, full, resumed = root / 'first', root / 'full', root / 'resumed'
                run(first, ['--total-steps', '1001', '--max-updates', '2'])
                checkpoint_path = first / 'latest.pt'
                original_hash = vp.sha256(checkpoint_path)
                run(full, ['--total-steps', '1001', '--max-updates', '4'])
                # Resume must not read the current YAML at all.
                with patch('yaml.safe_load', side_effect=AssertionError('current YAML used on resume')):
                    run(resumed, ['--resume', str(checkpoint_path), '--max-updates', '2'])
                checkpoint = torch.load(resumed / 'latest.pt', weights_only=False)
                reference = torch.load(full / 'latest.pt', weights_only=False)
                self.assertEqual(checkpoint['step'], 4)
                self.assertEqual(checkpoint['metadata']['final_step'], 1001)
                for key, value in checkpoint['model'].items():
                    torch.testing.assert_close(value, reference['model'][key], rtol=0, atol=0)
                self.assertEqual(vp.sha256(checkpoint_path), original_hash)
                # Old checkpoints have optimizer but no RNG/iterator state.
                legacy = torch.load(checkpoint_path, weights_only=False)
                del legacy['rng_state'], legacy['data_stream']
                atomic_save(legacy, root / 'legacy.pt')
                run(root / 'legacy_resume', ['--resume', str(root / 'legacy.pt'), '--max-updates', '1'])
                legacy_new = torch.load(root / 'legacy_resume/latest.pt', weights_only=False)
                self.assertEqual(legacy_new['step'], 3)
                self.assertFalse(legacy_new['metadata']['resume_source_has_rng_and_stream'])


if __name__ == '__main__':
    unittest.main()
