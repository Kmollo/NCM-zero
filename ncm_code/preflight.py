"""CPU preparation checks. Model/EMA updates are forbidden, not merely skipped."""
import json
import random
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import checkpoint
from .artifacts import atomic_json,file_hash,source_hashes
from .config import load_config
from .corpus import load_bundle
from .runtime import (batch,build,device_for,environment,loss,parameter_counts,predict,
                      seed_all,set_mode,training_module)
from .train import batch_at,experiment_spec,make_optimizer,metadata


def snapshot(components):
    return {name:checkpoint.module_state(m) for name,m in components.items()}


def assert_weights_equal(components,before):
    for name,module in components.items():
        for key,param in module.named_parameters():
            if not torch.equal(param.detach().cpu(),before[name]['state_dict'][key]):
                raise AssertionError(f'Preflight changed model weights: {name}.{key}')


def run(config,dataset,output):
    cfg=load_config(config)
    if cfg['precision']!='fp32': raise ValueError('No-training preflight uses CPU/fp32; GPU profile is a separate future run')
    manifest,splits=load_bundle(dataset)
    if cfg['prefix_length']!=manifest['prefix_length'] or cfg['output_length']!=manifest['output_length']:
        raise ValueError('Config byte lengths must match dataset manifest')
    output=Path(output)
    if output.exists(): raise ValueError('Preflight output exists; use a new path')
    output.mkdir(parents=True)
    torch.set_num_threads(cfg['threads'])
    seed_all(cfg['seed'])
    device=device_for('cpu','fp32')
    started=time.monotonic()
    report={'training':False,'optimizer_updates':0,'ema_updates':0,'device':'cpu',
            'config':cfg,'source_hashes':source_hashes(),
            'dataset_manifest_sha256':file_hash(Path(dataset)/'manifest.json'),
            'environment':environment(),'stages':{},'limits':[
                'No optimizer update or train-mode codec forward executed.',
                'Resume serialization/RNG/data continuity checked without learned updates.',
                'Actual update equivalence, GPU memory, throughput and convergence remain unmeasured.']}
    def forbidden(*args,**kwargs):
        raise AssertionError('Optimizer updates are prohibited during preparation')
    with patch.object(torch.optim.AdamW,'step',forbidden),patch.object(torch.optim.SGD,'step',forbidden):
        for stage in ('codec','generator','control'):
            seed_all(cfg['seed'])
            components=build(stage,cfg,device)
            set_mode(components,stage,False)
            original=snapshot(components)
            rows,next_cursor=batch_at(splits['train'],0,cfg['batch_size'],cfg['seed'])
            prefix,target=batch(rows,cfg,device)
            module=training_module(components,stage)
            value,metrics=loss(components,stage,cfg,prefix,target)
            value.backward()
            grads=[p.grad for p in module.parameters() if p.requires_grad]
            if not grads or not any(g is not None and bool(g.abs().sum()>0) for g in grads):
                raise AssertionError('No nonzero gradients')
            if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
                raise AssertionError('Non-finite gradients')
            if stage=='generator':
                q=module.l0_queries.grad
                if q is None or not bool(q.abs().sum()>0): raise AssertionError('Level-zero query cannot learn')
                if any(p.grad is not None for p in components['codec'].parameters()):
                    raise AssertionError('Frozen codec received gradients')
            assert_weights_equal(components,original)
            # Upstream eval forward records usage hits. Restore those diagnostic buffers;
            # EMA, codebook weights and fine-drop counters must have stayed unchanged.
            for name,m in components.items():
                for key,buf in m.named_buffers():
                    if 'codebook_hit_' not in key and not torch.equal(buf.cpu(),original[name]['all_buffers'][key]):
                        raise AssertionError(f'Unexpected buffer mutation: {key}')
                checkpoint.restore_module(m,original[name])
                m.zero_grad(set_to_none=True)
            expected=predict(components,stage,prefix,target)
            if expected.shape!=target.shape: raise AssertionError('Generation shape mismatch')
            optimizer,scheduler=make_optimizer(module,cfg)
            spec=experiment_spec(config,dataset,stage,'cpu')
            meta=metadata(spec,device)
            artifact=output/f'{stage}-UNTRAINED.pt'
            checkpoint.save(artifact,components=components,optimizer=optimizer,scheduler=scheduler,metadata=meta,
                            step=0,cursor=next_cursor,elapsed_seconds=0.0,trained=False)
            expected_random=(random.random(),float(np.random.random()),torch.rand(4))
            next_rows,expected_cursor=batch_at(splits['train'],next_cursor,cfg['batch_size'],cfg['seed'])
            fresh=build(stage,cfg,device)
            fresh_optimizer,fresh_scheduler=make_optimizer(training_module(fresh,stage),cfg)
            step,cursor,_=checkpoint.restore(checkpoint.read(artifact),fresh,fresh_optimizer,fresh_scheduler,meta)
            observed_random=(random.random(),float(np.random.random()),torch.rand(4))
            if observed_random[:2]!=expected_random[:2] or not torch.equal(observed_random[2],expected_random[2]):
                raise AssertionError('RNG did not resume exactly')
            actual_rows,actual_cursor=batch_at(splits['train'],cursor,cfg['batch_size'],cfg['seed'])
            if actual_rows!=next_rows or actual_cursor!=expected_cursor or step!=0:
                raise AssertionError('Data cursor did not resume exactly')
            set_mode(fresh,stage,False)
            actual=predict(fresh,stage,prefix,target)
            if not torch.equal(actual,expected): raise AssertionError('Reloaded inference differs')
            assert_weights_equal(fresh,original)
            report['stages'][stage]={'passed':True,'loss_probe':metrics,'parameter_counts':parameter_counts(components),
                                    'finite_gradients':True,'weights_unchanged':True,'rng_and_cursor_restored':True,
                                    'reloaded_inference_identical':True,'untrained_checkpoint_sha256':file_hash(artifact)}
    report.update(passed=True,wall_seconds=time.monotonic()-started)
    atomic_json(output/'report.json',report)
    return report
