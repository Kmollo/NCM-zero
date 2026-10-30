"""Explicit preparation, evaluation and training commands; no training default."""
import argparse
import json
import sys
from pathlib import Path
from .artifacts import atomic_json, digest, file_hash, source_hashes
from .config import load_config
from .corpus import load_bundle


def parser():
    p=argparse.ArgumentParser(description='Zero NCM preparation and bounded experiments. No default training.')
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare-data',help='Author and freeze a small independent synthetic corpus')
    prep.add_argument('--output',required=True)
    prep.add_argument('--config',required=True)
    verify=sub.add_parser('verify-data',help='Check hashes/splits and execute authored fixtures in WASI')
    verify.add_argument('--dataset',required=True)
    doctor=sub.add_parser('doctor',help='Inspect imports/runtime; performs no model work')
    doctor.add_argument('--output')
    compare=sub.add_parser('compare-checkpoints',help='Compare resumed/uninterrupted state without updates')
    compare.add_argument('--left',required=True)
    compare.add_argument('--right',required=True)
    compare.add_argument('--atol',type=float,default=0.0)
    compare.add_argument('--rtol',type=float,default=0.0)
    compare.add_argument('--output',required=True)
    pre=sub.add_parser('preflight',help='CPU gradients/checkpoint checks, with optimizer updates blocked')
    plan=sub.add_parser('plan',help='Write an unapproved exact experiment specification')
    train=sub.add_parser('train',help='Disabled unless a matching experiment is explicitly authorized')
    for command in (pre,plan,train):
        command.add_argument('--config',required=True)
        command.add_argument('--dataset',required=True)
        command.add_argument('--output',required=True)
    for command in (plan,train):
        command.add_argument('--stage',choices=['codec','generator','control'],required=True)
        command.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    train.add_argument('--execute-training',action='store_true')
    train.add_argument('--approval')
    train.add_argument('--resume')
    train.add_argument('--codec')
    train.add_argument('--codec-gate')
    ev=sub.add_parser('evaluate',help='Read a checkpoint; no updates; final test requires an explicit switch')
    ev.add_argument('--checkpoint',required=True)
    ev.add_argument('--dataset',required=True)
    ev.add_argument('--split',choices=['development','test'],default='development')
    ev.add_argument('--final-test',action='store_true')
    ev.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ev.add_argument('--functional',action='store_true')
    ev.add_argument('--output',required=True)
    return p


def dispatch(args):
    # Refuse before torch import or any model/directory creation.
    if args.command=='train' and (not args.execute_training or not args.approval):
        raise ValueError('Training not authorized: requires --execute-training and --approval. Preparation does not train.')
    if args.command=='prepare-data':
        from .corpus import prepare
        cfg=load_config(args.config)
        return prepare(args.output,cfg['prefix_length'],cfg['output_length'],cfg['seed'])
    if args.command=='verify-data':
        from .corpus import verify_bundle
        return verify_bundle(args.dataset)
    if args.command=='doctor':
        from .runtime import environment
        from .model import upstream
        upstream()
        report={'environment':environment(),'upstream_verified':True,'training':False}
        if args.output: atomic_json(args.output,report)
        return report
    if args.command=='compare-checkpoints':
        from .checkpoint import compare
        if Path(args.output).exists():raise ValueError('Comparison report already exists')
        report=compare(args.left,args.right,args.atol,args.rtol)
        report.update(left_sha256=file_hash(args.left),right_sha256=file_hash(args.right))
        atomic_json(args.output,report)
        return report
    if args.command=='plan':
        # Plan stays dependency-free and never includes an authorization.
        cfg=load_config(args.config)
        manifest,_=load_bundle(args.dataset)
        if (cfg['prefix_length'],cfg['output_length'])!=(manifest['prefix_length'],manifest['output_length']):
            raise ValueError('Config and dataset byte lengths differ')
        spec={'stage':args.stage,'device':args.device,'config':cfg,'config_sha256':digest(cfg),
              'dataset_manifest_sha256':file_hash(Path(args.dataset)/'manifest.json'),
              'source_hashes':source_hashes(),'max_steps':cfg['steps'],'max_seconds':cfg['max_seconds']}
        report={'training_authorized':False,'automatic_rental':False,'automatic_paid_api':False,
                'experiment':spec,'note':'User authorization must precede changing this flag or running training.'}
        if Path(args.output).exists(): raise ValueError('Plan exists; choose a new destination')
        atomic_json(args.output,report)
        return report
    if args.command=='preflight':
        from .preflight import run
        return run(args.config,args.dataset,args.output)
    if args.command=='train':
        from .train import run
        return run(config=args.config,dataset=args.dataset,stage=args.stage,device_name=args.device,
                   output=args.output,approval=args.approval,execute=args.execute_training,
                   resume=args.resume,codec=args.codec,codec_gate=args.codec_gate)
    if args.command=='evaluate':
        if args.split=='test' and not args.final_test: raise ValueError('Test evaluation requires --final-test; never use it for selection')
        if Path(args.output).exists(): raise ValueError('Evaluation output exists')
        from . import checkpoint
        from .runtime import build,device_for,seed_all
        from .evaluate import evaluate
        payload=checkpoint.read(args.checkpoint)
        cfg=payload['metadata']['config']
        stage=payload['metadata']['stage']
        manifest,splits=load_bundle(args.dataset)
        data_hash=file_hash(Path(args.dataset)/'manifest.json')
        if data_hash!=payload['metadata']['dataset_manifest_sha256'] or source_hashes()!=payload['metadata']['source_hashes']:
            raise ValueError('Checkpoint source/data provenance mismatch')
        device=device_for(args.device,cfg['precision'])
        seed_all(cfg['seed'])
        components=build(stage,cfg,device,payload['components'].get('codec'))
        for name,module in components.items(): checkpoint.restore_module(module,payload['components'][name])
        report=evaluate(components,stage,cfg,splits[args.split],device,functional=args.functional)
        report.update(split=args.split,checkpoint_sha256=file_hash(args.checkpoint),dataset_manifest_sha256=data_hash,
                      trained_checkpoint=payload['trained'],training=False)
        report['codec_gate_passed']=bool(stage=='codec' and args.split=='test' and payload['trained'] and report['count']>0 and report['exact']==report['count'])
        atomic_json(args.output,report)
        return report
    raise ValueError('Unknown command')


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    try:
        report=dispatch(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        p.exit(2,f'Zero: {exc}\n')
    print(json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False))
    if report.get('passed') is False:
        raise SystemExit(1)
    return 0
