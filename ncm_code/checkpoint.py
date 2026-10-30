"""Complete local checkpoints, including nonpersistent upstream EMA counters."""
import os
import math
import random
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch


def module_state(module):
    return {'state_dict': {k:v.detach().cpu().clone() for k,v in module.state_dict().items()},
            'all_buffers': {k:v.detach().cpu().clone() for k,v in module.named_buffers()}}


def restore_module(module,state):
    module.load_state_dict(state['state_dict'],strict=True)
    buffers=dict(module.named_buffers())
    if set(buffers)!=set(state['all_buffers']): raise ValueError('Checkpoint buffer inventory mismatch')
    with torch.no_grad():
        for name,value in state['all_buffers'].items():
            if value.shape!=buffers[name].shape or value.dtype!=buffers[name].dtype:
                raise ValueError(f'Checkpoint buffer shape/type mismatch: {name}')
            buffers[name].copy_(value.to(buffers[name].device))


def rng_state():
    state=np.random.get_state()
    return {'python':random.getstate(), 'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda'])!=torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs')
        torch.cuda.set_rng_state_all(state['cuda'])


def save(path, *, components, optimizer, scheduler, metadata, step, cursor, elapsed_seconds, trained):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':1,'components':{k:module_state(v) for k,v in components.items()},
             'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'rng':rng_state(),
             'metadata':metadata,'step':step,'cursor':cursor,'elapsed_seconds':elapsed_seconds,
             'trained':trained,'precision_scaler':None}
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def read(path):
    result=torch.load(path,map_location='cpu',weights_only=True)
    if result.get('schema_version')!=1: raise ValueError('Unknown checkpoint schema')
    return result


def retain(source,destination):
    """Atomically retain a complete step checkpoint for a later resume comparison."""
    destination=Path(destination)
    fd,temp=tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent)
    try:
        with os.fdopen(fd,'wb') as writer,Path(source).open('rb') as reader:
            shutil.copyfileobj(reader,writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temp,destination)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def restore(payload,components,optimizer,scheduler,metadata):
    if payload['metadata']!=metadata: raise ValueError('Resume config/data/source/runtime/stage mismatch')
    if (type(payload['step']) is not int or payload['step']<0 or type(payload['cursor']) is not int or payload['cursor']<0 or
        type(payload['elapsed_seconds']) not in (int,float) or not math.isfinite(payload['elapsed_seconds']) or payload['elapsed_seconds']<0):
        raise ValueError('Invalid checkpoint progress or time')
    if set(payload['components'])!=set(components): raise ValueError('Checkpoint component mismatch')
    for name,module in components.items(): restore_module(module,payload['components'][name])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return payload['step'],payload['cursor'],payload['elapsed_seconds']


def compare(left,right,atol=0.0,rtol=0.0):
    """Compare learned-resume evidence; only elapsed wall time may differ."""
    if not all(math.isfinite(v) and v>=0 for v in (atol,rtol)):
        raise ValueError('Comparison tolerances must be finite and nonnegative')
    differences=[]
    tensors=0
    def walk(a,b,path):
        nonlocal tensors
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            tensors+=1
            same=a.shape==b.shape and a.dtype==b.dtype
            if same:
                same=bool(torch.allclose(a,b,atol=atol,rtol=rtol)) if a.is_floating_point() else torch.equal(a,b)
            if not same:differences.append(path)
        elif type(a)!=type(b):differences.append(path)
        elif isinstance(a,dict):
            if set(a)!=set(b):differences.append(path+'.keys')
            for key in a.keys() & b.keys():walk(a[key],b[key],f'{path}.{key}')
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):differences.append(path+'.length')
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,f'{path}[{i}]')
        elif a!=b:differences.append(path)
    a,b=read(left),read(right)
    a.pop('elapsed_seconds');b.pop('elapsed_seconds')
    walk(a,b,'checkpoint')
    return {'passed':not differences,'tensors_compared':tensors,'difference_count':len(differences),
            'differences':differences[:100],'atol':atol,'rtol':rtol,'ignored':['elapsed_seconds'],
            'training':False}
