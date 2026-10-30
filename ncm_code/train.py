"""Bounded single-device training. Only run() may update learned state.

CLI dispatch checks explicit authorization before importing this module.
run() checks again before building a model or creating output directories.
"""
import json
import math
import random
import time
from pathlib import Path
import torch
from . import checkpoint
from .artifacts import atomic_json, digest, file_hash, source_hashes
from .config import load_config
from .corpus import load_bundle
from .runtime import (autocast,batch,build,device_for,environment,loss,
                      parameter_counts,seed_all,set_mode,training_module)
from .evaluate import evaluate


def experiment_spec(config,dataset,stage,device):
    cfg=load_config(config)
    manifest,_=load_bundle(dataset)
    if manifest.get('usage') == 'codec_only' and stage != 'codec':
        raise ValueError('Codec-only reconstruction data cannot train the generator or control model')
    if (cfg['prefix_length'],cfg['output_length'])!=(manifest['prefix_length'],manifest['output_length']):
        raise ValueError('Config and dataset byte lengths differ')
    return {'stage':stage,'device':device,'config':cfg,'config_sha256':digest(cfg),
            'dataset_manifest_sha256':file_hash(Path(dataset)/'manifest.json'),
            'source_hashes':source_hashes(),'max_steps':cfg['steps'],'max_seconds':cfg['max_seconds']}


def authorize(config,dataset,stage,device,approval,execute):
    if not execute:
        raise ValueError('Training is disabled. A future authorized run requires --execute-training and an approved experiment file.')
    spec=experiment_spec(config,dataset,stage,device)
    approved=json.loads(Path(approval).read_text(encoding='utf-8'))
    if approved.get('training_authorized') is not True or approved.get('experiment')!=spec:
        raise ValueError('Authorization must match the exact stage, config, dataset, sources and device')
    if approved.get('automatic_rental') is not False or approved.get('automatic_paid_api') is not False:
        raise ValueError('This trainer never provisions compute or paid services')
    return spec


def batch_at(rows,cursor,batch_size,seed):
    # Cursor counts consumed records. A batch never silently drops an epoch tail.
    epoch,start=divmod(cursor,len(rows))
    order=list(range(len(rows)))
    random.Random(seed+epoch).shuffle(order)
    selected=order[start:start+batch_size]
    return [rows[i] for i in selected],cursor+len(selected)


def make_optimizer(module,cfg):
    optimizer=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                               lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:0.5*(1+math.cos(math.pi*min(step,cfg['steps'])/cfg['steps'])))
    return optimizer,scheduler


def metadata(spec,device,codec_hash=None):
    return {**spec,'runtime':environment(),'execution_device':str(device),'codec_checkpoint_sha256':codec_hash}


def checked_codec(path,gate,dataset,cfg):
    if path is None or gate is None: raise ValueError('Generator training requires a trained codec and its frozen test gate report')
    state=checkpoint.read(path)
    manifest,_=load_bundle(dataset)
    report=json.loads(Path(gate).read_text(encoding='utf-8'))
    if (not state['trained'] or state['step']<1 or state['metadata']['stage']!='codec' or
        state['metadata']['config']!=cfg or state['metadata']['source_hashes']!=source_hashes()):
        raise ValueError('Codec checkpoint is untrained, incompatible or from different source code')
    if (report.get('checkpoint_sha256')!=file_hash(path) or report.get('dataset_manifest_sha256')!=file_hash(Path(dataset)/'manifest.json') or
        report.get('split')!='test' or report.get('stage')!='codec' or report.get('codec_gate_passed') is not True or
        report.get('count',0)!=manifest['splits']['test']['records'] or report.get('exact')!=report['count']):
        raise ValueError('Codec has not passed the exact-artifact test gate')
    return state['components']['codec']


def run(*,config,dataset,stage,device_name,output,approval,execute=False,resume=None,codec=None,codec_gate=None):
    spec=authorize(config,dataset,stage,device_name,approval,execute)
    cfg=spec['config']
    _,splits=load_bundle(dataset)
    verification=Path(dataset)/'verification.json'
    evidence=json.loads(verification.read_text(encoding='utf-8'))
    if not evidence.get('passed') or evidence.get('manifest_sha256')!=spec['dataset_manifest_sha256']:
        raise ValueError('Dataset needs matching successful fixture verification')
    device=device_for(device_name,cfg['precision'])
    torch.set_num_threads(cfg['threads'])
    seed_all(cfg['seed'])
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    codec_state=checked_codec(codec,codec_gate,dataset,cfg) if stage=='generator' else None
    output=Path(output)
    if output.exists(): raise ValueError('Run output already exists; resume into a new directory')
    components=build(stage,cfg,device,codec_state)
    module=training_module(components,stage)
    optimizer,scheduler=make_optimizer(module,cfg)
    meta=metadata(spec,device,file_hash(codec) if stage=='generator' else None)
    step=cursor=0
    previous_elapsed=0.0
    if resume:
        resumed=checkpoint.read(resume)
        step,cursor,previous_elapsed=checkpoint.restore(resumed,components,optimizer,scheduler,meta)
        if step>0 and not resumed['trained']:
            raise ValueError('Untrained diagnostic checkpoint cannot resume as a trained model')
        # Include recorded evaluation/failure time after the last saved update.
        status_path=Path(resume).parent/'status.json'
        if status_path.exists():
            previous_status=json.loads(status_path.read_text(encoding='utf-8'))
            if previous_status.get('checkpoint_sha256')==file_hash(resume):
                previous_elapsed=max(previous_elapsed,previous_status['elapsed_seconds'])
        if not 0<=step<cfg['steps'] or previous_elapsed>=cfg['max_seconds']:
            raise ValueError('Resumed run has already exhausted its budget')
    output.mkdir(parents=True)
    atomic_json(output/'experiment.json',{'metadata':meta,'parameter_counts':parameter_counts(components),
                                       'resume_from':str(resume) if resume else None,'training':True})
    started=time.monotonic()
    deadline=started+cfg['max_seconds']-previous_elapsed
    latest=output/'latest.pt'
    checkpoint.save(latest,components=components,optimizer=optimizer,scheduler=scheduler,metadata=meta,
                    step=step,cursor=cursor,elapsed_seconds=previous_elapsed,trained=step>0)
    checkpoint.retain(latest,output/f'step-{step:06}.pt')
    status='completed'
    try:
        while step<cfg['steps']:
            if time.monotonic()>=deadline:
                status='wall_time_limit'
                break
            set_mode(components,stage,True)
            optimizer.zero_grad(set_to_none=True)
            metrics=[]
            next_cursor=cursor
            # Accumulation weights samples equally, including a shorter epoch-tail microbatch.
            chunks=[]
            for _ in range(cfg['accumulation']):
                rows,next_cursor=batch_at(splits['train'],next_cursor,cfg['batch_size'],cfg['seed'])
                chunks.append(rows)
            total_examples=sum(map(len,chunks))
            for rows in chunks:
                if time.monotonic()>=deadline: raise TimeoutError('Time budget exhausted within accumulation')
                prefix,target=batch(rows,cfg,device)
                with autocast(cfg,device): value,observed=loss(components,stage,cfg,prefix,target,training=True)
                (value*(len(rows)/total_examples)).backward()
                metrics.append(observed)
            norm=torch.nn.utils.clip_grad_norm_(module.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            if time.monotonic()>=deadline: raise TimeoutError('Time budget exhausted before update')
            optimizer.step()
            scheduler.step()
            step+=1
            cursor=next_cursor
            elapsed=previous_elapsed+time.monotonic()-started
            event={'step':step,'cursor':cursor,'elapsed_seconds':elapsed,'microbatches':metrics,
                   'gradient_norm':float(norm),'learning_rate':optimizer.param_groups[0]['lr']}
            with (output/'metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(event,allow_nan=False)+'\n')
                stream.flush()
            # Save only complete optimizer boundaries, before evaluation.
            if step%cfg['save_every']==0 or step%cfg['eval_every']==0 or step==cfg['steps']:
                checkpoint.save(latest,components=components,optimizer=optimizer,scheduler=scheduler,metadata=meta,
                                step=step,cursor=cursor,elapsed_seconds=elapsed,trained=True)
                checkpoint.retain(latest,output/f'step-{step:06}.pt')
            if step%cfg['eval_every']==0 or step==cfg['steps']:
                evaluation=evaluate(components,stage,cfg,splits['development'],device,deadline=deadline)
                atomic_json(output/f'development-{step:06}.json',evaluation)
        # A wall-time stop may follow unsaved full steps. There is no incomplete batch here.
        checkpoint.save(latest,components=components,optimizer=optimizer,scheduler=scheduler,metadata=meta,
                        step=step,cursor=cursor,elapsed_seconds=previous_elapsed+time.monotonic()-started,trained=step>0)
    except (Exception,KeyboardInterrupt) as exc:
        status='wall_time_limit' if isinstance(exc,TimeoutError) else 'failed_or_interrupted'
        atomic_json(output/'status.json',{'status':status,'error':str(exc),'last_completed_step':step,
                                        'checkpoint_note':'Last saved complete boundary only; partial accumulation discarded.',
                                        'checkpoint_sha256':file_hash(latest),
                                        'elapsed_seconds':previous_elapsed+time.monotonic()-started})
        raise
    result={'status':status,'step':step,'cursor':cursor,'checkpoint':str(latest),
            'checkpoint_sha256':file_hash(latest),
            'elapsed_seconds':previous_elapsed+time.monotonic()-started,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None}
    atomic_json(output/'status.json',result)
    return result
