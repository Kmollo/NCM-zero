"""Explicitly authorized, CPU-only memorization diagnostic; never a holdout test.

Uses the existing codec, optimizer, batching and full checkpoint implementation.
Only the frozen 16 training rows enter a model. No resume or automatic retry.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from .artifacts import atomic_json, digest, file_hash
from .config import load_config
from .corpus import load_bundle


def select_rows(train_rows, seed):
    groups = {}
    for row in train_rows:
        groups.setdefault(row['family'], []).append(row)
    if len(groups) != 16:
        raise ValueError('Diagnostic expects exactly 16 training families')
    rng = random.Random(seed)
    return [rng.choice(sorted(groups[family], key=lambda r: r['id']))
            for family in sorted(groups)]


def specification(config, dataset):
    from .train import experiment_spec
    cfg = load_config(config)
    if (cfg['steps'] > 1000 or cfg['max_seconds'] > 1200 or
            cfg['precision'] != 'fp32' or cfg['threads'] > 2 or
            cfg['accumulation'] != 1 or cfg['fine_drop'] != 0 or
            cfg['eval_every'] != 100 or cfg['save_every'] != 100):
        raise ValueError('Configuration exceeds the CPU diagnostic scope')
    _, splits = load_bundle(dataset)
    rows = select_rows(splits['train'], cfg['seed'])
    spec = experiment_spec(config, dataset, 'codec', 'cpu')
    spec.update(purpose='train-only-16-example-memorization',
                selected_ids=[r['id'] for r in rows], selected_rows_sha256=digest(rows),
                resource_guard_sha256=file_hash(Path(__file__).resolve().parents[1]/'scripts/guard_codec_learning_test.py'),
                evaluation_split='selected_training_rows',
                success_criterion='16/16 strict reconstructions at two consecutive distinct positive-step evaluations',
                heldout_model_evaluations=0, maximum_launches=1)
    return spec, rows


def plan(config, dataset, output):
    if Path(output).exists():
        raise ValueError('Plan already exists')
    spec, rows = specification(config, dataset)
    result = {'training_authorized': False, 'automatic_rental': False,
              'automatic_paid_api': False, 'experiment': spec, 'selected_rows': rows}
    atomic_json(output, result)
    return {'plan': str(output), 'training': False, 'selected_ids': spec['selected_ids']}


def run(config, dataset, output, approval, execute=False):
    if not execute or not approval:
        raise ValueError('Explicit execution and approved experiment are required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise ValueError('This diagnostic requires CUDA_VISIBLE_DEVICES=-1')
    spec, rows = specification(config, dataset)
    approved = json.loads(Path(approval).read_text(encoding='utf-8'))
    if (approved.get('training_authorized') is not True or
            approved.get('automatic_rental') is not False or
            approved.get('automatic_paid_api') is not False or
            approved.get('experiment') != spec or approved.get('selected_rows') != rows):
        raise ValueError('Approval must bind the exact diagnostic, sources and rows')
    evidence = json.loads((Path(dataset)/'verification.json').read_text(encoding='utf-8'))
    if not evidence.get('passed') or evidence['manifest_sha256'] != spec['dataset_manifest_sha256']:
        raise ValueError('Verified source corpus required')
    output = Path(output)
    if output.exists():
        raise ValueError('Run exists; never overwrite or automatically retry this diagnostic')

    import torch
    from . import checkpoint
    from .runtime import (batch, build, byte_loss, device_for, loss,
                          parameter_counts, seed_all, set_mode, training_module)
    from .train import batch_at, make_optimizer, metadata
    from .learning_metrics import score_reconstructions

    cfg = spec['config']
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    device = device_for('cpu', 'fp32')
    seed_all(cfg['seed'])
    started = time.monotonic()
    deadline = started + cfg['max_seconds']
    components = build('codec', cfg, device)
    module = training_module(components, 'codec')
    optimizer, scheduler = make_optimizer(module, cfg)
    meta = metadata(spec, device)
    output.mkdir(parents=True)
    atomic_json(output/'experiment.json', {'metadata': meta, 'selected_rows': rows,
                'approval_sha256': file_hash(approval), 'parameter_counts': parameter_counts(components)})
    step = cursor = consecutive = 0
    latest = output/'latest.pt'
    summary = []

    def save():
        checkpoint.save(latest, components=components, optimizer=optimizer, scheduler=scheduler,
                        metadata=meta, step=step, cursor=cursor,
                        elapsed_seconds=time.monotonic()-started, trained=step > 0)
        checkpoint.retain(latest, output/f'step-{step:06}.pt')

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU diagnostic reached its wall-time limit')

    @torch.no_grad()
    def assess():
        set_mode(components, 'codec', False)
        original = checkpoint.module_state(module)
        all_ids, all_codes, reconstruction_losses = [], [], []
        for offset in range(0, len(rows), cfg['batch_size']):
            check_time()
            chunk = rows[offset:offset+cfg['batch_size']]
            _, target = batch(chunk, cfg, device)
            codes = module.encode(target)
            logits = module.decode(codes)
            _, observed = byte_loss(logits, target, cfg['pad_weight'])
            reconstruction_losses.append((len(chunk), observed))
            all_ids.extend(logits.argmax(-1).cpu().tolist())
            all_codes.append(codes.cpu())
        current = checkpoint.module_state(module)
        for group in original:
            if any(not torch.equal(value, current[group][key]) for key, value in original[group].items()):
                raise AssertionError('Inference changed codec weights or buffers')
        report = score_reconstructions(all_ids, rows, cfg['output_length'])
        codes = torch.cat(all_codes)
        offset, usage = 0, []
        for length in cfg['levels']:
            usage.append(int(codes[:, offset:offset+length].unique().numel()))
            offset += length
        report.update(step=step, cursor=cursor, split='selected_training_rows',
                      elapsed_seconds=time.monotonic()-started,
                      checkpoint_sha256=file_hash(latest), codebook_current_unique=usage,
                      evaluation_state_unchanged=True,
                      evaluation_loss={key: sum(n*m[key] for n, m in reconstruction_losses)/len(rows)
                                       for key in ('byte_eos_ce', 'pad_ce')})
        atomic_json(output/f'reconstruction-{step:06}.json', report)
        brief = {key: report[key] for key in ('step', 'exact', 'count', 'byte_accuracy',
                 'eos_correct', 'padding_valid', 'evaluation_loss', 'elapsed_seconds')}
        summary.append(brief)
        print(json.dumps(brief), flush=True)
        return report

    status = 'step_limit'
    final_report = None
    try:
        save()
        final_report = assess()
        while step < cfg['steps']:
            check_time()
            set_mode(components, 'codec', True)
            optimizer.zero_grad(set_to_none=True)
            selected, next_cursor = batch_at(rows, cursor, cfg['batch_size'], cfg['seed'])
            prefix, target = batch(selected, cfg, device)
            value, metrics = loss(components, 'codec', cfg, prefix, target, training=True)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            check_time()
            optimizer.step()
            scheduler.step()
            step += 1
            cursor = next_cursor
            event = {'step': step, 'cursor': cursor, 'training_ids': [r['id'] for r in selected],
                     'elapsed_seconds': time.monotonic()-started, 'gradient_norm': float(norm),
                     'learning_rate': optimizer.param_groups[0]['lr'], **metrics}
            with (output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            if step % cfg['eval_every'] == 0 or step == cfg['steps']:
                save()
                final_report = assess()
                consecutive = consecutive + 1 if final_report['exact'] == len(rows) else 0
                if consecutive >= 2:
                    status = 'criterion_met'
                    break
    except (Exception, KeyboardInterrupt) as exc:
        status = 'wall_time_limit' if isinstance(exc, TimeoutError) else 'failed_or_interrupted'
        # The checkpoint remains the last complete saved boundary; partial EMA work is not saved.
        result = {'status': status, 'error': str(exc), 'last_completed_step': step,
                  'checkpoint': str(latest), 'elapsed_seconds': time.monotonic()-started,
                  'criterion_met': False, 'evaluations': summary}
        atomic_json(output/'status.json', result)
        raise
    result = {'status': status, 'step': step, 'cursor': cursor,
              'criterion_met': status == 'criterion_met', 'consecutive_perfect_evaluations': consecutive,
              'final_exact': final_report['exact'], 'count': len(rows), 'evaluations': summary,
              'checkpoint': str(latest), 'checkpoint_sha256': file_hash(latest),
              'elapsed_seconds': time.monotonic()-started, 'device': 'cpu',
              'heldout_model_evaluations': 0, 'training_stage': 'codec_only'}
    atomic_json(output/'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--approval')
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        result = plan(args.config, args.dataset, args.output)
    else:
        result = run(args.config, args.dataset, args.output, args.approval, args.execute_training)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
