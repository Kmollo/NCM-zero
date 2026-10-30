"""Strict artifact evaluation, with optional WASI functional checks."""
import ast
import json
import math
import secrets
import time
from .artifacts import digest
from .data import decode, PAD
from .runtime import batch, predict, set_mode, autocast
import torch


def tool_payload(text):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value: raise ValueError('Duplicate JSON member')
            value[key]=item
        return value
    def invalid_constant(value):
        raise ValueError(f'Nonstandard JSON constant: {value}')
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid_constant)


def score_ids(ids, row):
    result={'id':row['id'],'kind':row['kind'],'family':row['family'],
            'expected_sha256':digest(row['completion']),'exact':False,'valid_output':False,
            'parses':False,'functional_pass':None}
    try:
        text=decode(ids)
    except (ValueError, UnicodeError) as exc:
        result['error']=str(exc)
        return result
    result.update(valid_output=True,output=text,output_sha256=digest(text),exact=text==row['completion'])
    try:
        if row['kind']=='python': ast.parse(text)
        else: tool_payload(text)
        result['parses']=True
    except (SyntaxError, ValueError): pass
    if row['kind']=='tool_json':
        # Canonical JSON preserves scalar types (true is not integer 1).
        try: result['functional_pass']=digest(tool_payload(text))==digest(row['verification'])
        except ValueError: result['functional_pass']=False
    return result


def functional_check(result,row):
    if row['kind']!='python': return
    if not result['parses']:
        result['functional_pass']=False
        return
    from zero_baseline.grading import _observed_run
    marker='ZERO_TRAINING_EVAL_'+secrets.token_hex(16)
    source=('import sys\nsys.path.insert(0,"/workspace")\nimport app\nsolve=app.solve\n'
            +row['verification']+'\nprint('+repr(marker)+',flush=True)\n')
    observed=_observed_run(source,{'app.py':result['output']},marker,'stdout',10)
    # The grading helper's private marker observes completion, not just exit 0.
    result['functional_observation']=observed
    result['functional_pass']=observed['passed']


@torch.no_grad()
def evaluate(components, stage, cfg, rows, device, functional=False, deadline=None):
    set_mode(components,stage,False)
    outcomes=[]
    started=time.monotonic()
    for offset in range(0,len(rows),cfg['batch_size']):
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Evaluation exceeded experiment wall-time budget')
        chunk=rows[offset:offset+cfg['batch_size']]
        prefix,target=batch(chunk,cfg,device)
        with autocast(cfg,device):
            generated=predict(components,stage,prefix,target).cpu().tolist()
        for ids,row in zip(generated,chunk):
            outcome=score_ids(ids,row)
            if functional: functional_check(outcome,row)
            outcome['artifact_bytes']=len(row['completion'].encode())
            outcomes.append(outcome)
    buckets={}
    for item in outcomes:
        length=item['artifact_bytes']
        bucket=f"{item['kind']}:{'1-64' if length<=64 else '65-128' if length<=128 else '129+'}"
        entry=buckets.setdefault(bucket,{'count':0,'exact':0,'valid_output':0})
        for key,value in (('count',1),('exact',item['exact']),('valid_output',item['valid_output'])):
            entry[key]+=int(value)
    report={'stage':stage,'count':len(outcomes),'exact':sum(x['exact'] for x in outcomes),
            'valid_outputs':sum(x['valid_output'] for x in outcomes),
            'parses':sum(x['parses'] for x in outcomes),
            'functional_checked':sum(x['functional_pass'] is not None for x in outcomes),
            'functional_passed':sum(x['functional_pass'] is True for x in outcomes),
            'by_type_and_length':buckets,'cases':outcomes,'wall_seconds':time.monotonic()-started}
    if stage=='codec':
        bits=sum(cfg['levels'])*math.ceil(math.log2(cfg['codebook_size']))
        report['representation']={'latent_positions':sum(cfg['levels']),
                                  'fixed_width_code_bits_per_artifact':bits,
                                  'mean_source_bits':sum(x['artifact_bytes']*8 for x in outcomes)/len(outcomes),
                                  'excludes_model_codebooks_and_container_overhead':True}
    return report
