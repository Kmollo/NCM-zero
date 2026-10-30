"""Model execution and losses. No optimizer updates in this module."""
import contextlib
import importlib.metadata
import math
import platform
import random
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import encode, PAD, EOS, VOCAB
from .model import build_tokenizer, build_generator, generator_logits, generate_ids


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment():
    packages = ('torch','numpy','hydra-core','omegaconf','antlr4-python3-runtime','PyYAML')
    return {'python': sys.version.split()[0], 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in packages},
            'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}


def device_for(name, precision):
    if name not in ('cpu', 'cuda'):
        raise ValueError('Only explicit cpu or cuda device is supported')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if precision == 'bf16' and (name != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('This profile requires a BF16-capable CUDA device')
    return torch.device(name)


def autocast(cfg, device):
    return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if cfg['precision']=='bf16' else contextlib.nullcontext())


class ByteControl(nn.Module):
    """Small causal byte Transformer control; separately accounts all parameters."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg['generator_dim']
        self.embedding = nn.Embedding(VOCAB, d)
        self.position = nn.Embedding(cfg['prefix_length']+cfg['output_length'], d)
        layer = nn.TransformerEncoderLayer(d, cfg['heads'], 4*d, dropout=0.0,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.stack = nn.TransformerEncoder(layer, cfg['generator_layers'], enable_nested_tensor=False)
        # TransformerEncoder clones layers; initialize each matrix independently.
        for block in self.stack.layers:
            for p in block.parameters():
                if p.ndim > 1: nn.init.xavier_uniform_(p)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, prefix, previous):
        ids = torch.cat([prefix, previous], dim=1)
        h = self.embedding(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        causal = torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).triu(1)
        h = self.stack(h, mask=causal, src_key_padding_mask=ids.eq(PAD))
        return self.head(h[:,prefix.shape[1]:])

    @torch.no_grad()
    def generate(self, prefix):
        previous = torch.full((prefix.shape[0],1),EOS,device=prefix.device,dtype=torch.long)
        done = torch.zeros(prefix.shape[0],device=prefix.device,dtype=torch.bool)
        outputs=[]
        for _ in range(self.cfg['output_length']):
            nxt=self(prefix,previous)[:,-1].argmax(-1)
            nxt=torch.where(done,PAD,nxt)
            outputs.append(nxt)
            done |= nxt.eq(EOS)
            previous=torch.cat([previous,nxt[:,None]],dim=1)
            if done.all():
                outputs.extend([torch.full_like(nxt,PAD)]*(self.cfg['output_length']-len(outputs)))
                break
        return torch.stack(outputs,dim=1)


def build(stage, cfg, device, codec_state=None):
    if stage == 'control': return {'model': ByteControl(cfg).to(device)}
    codec = build_tokenizer(cfg).to(device)
    if codec_state is not None:
        from .checkpoint import restore_module
        restore_module(codec,codec_state)
    if stage == 'codec': return {'codec':codec}
    if stage == 'generator': return {'codec':codec, 'model':build_generator(cfg,codec).to(device)}
    raise ValueError('Unknown stage')


def training_module(components, stage):
    return components['codec' if stage=='codec' else 'model']


def set_mode(components, stage, training):
    for model in components.values(): model.eval()
    training_module(components,stage).train(training)


def batch(rows, cfg, device):
    return (torch.tensor([encode(r['prompt'],cfg['prefix_length']) for r in rows],device=device),
            torch.tensor([encode(r['completion'],cfg['output_length']) for r in rows],device=device))


def byte_loss(logits, target, pad_weight):
    if logits.shape != (*target.shape,VOCAB): raise ValueError('Byte logits shape mismatch')
    ce=F.cross_entropy(logits.float().flatten(0,1),target.flatten(),reduction='none').reshape(target.shape)
    active=target.ne(PAD)
    # Each artifact contributes equally; padding cannot dominate by count.
    active_loss=((ce*active).sum(1)/active.sum(1).clamp_min(1)).mean()
    padding=~active
    pad_loss=((ce*padding).sum(1)/padding.sum(1).clamp_min(1)).mean()
    return active_loss+pad_weight*pad_loss, {'byte_eos_ce':float(active_loss.detach()),'pad_ce':float(pad_loss.detach())}


def level_loss(logits, codes, levels):
    if logits.shape[:2] != codes.shape or codes.shape[1] != sum(levels):
        raise ValueError('Latent level/target alignment mismatch')
    offset=0
    losses=[]
    for length in levels:
        losses.append(F.cross_entropy(logits[:,offset:offset+length].float().flatten(0,1),
                                      codes[:,offset:offset+length].flatten()))
        offset+=length
    return torch.stack(losses).mean(), [float(x.detach()) for x in losses]


def loss(components, stage, cfg, prefix, target, training=False):
    if stage=='codec':
        logits,vq,util,global_util,_=components['codec'](target)
        reconstruction,metrics=byte_loss(logits,target,cfg['pad_weight'])
        result=reconstruction+cfg['vq_weight']*vq
        metrics.update(vq_loss=float(vq.detach()),codebook_utilization=list(util),global_utilization=global_util)
    elif stage=='generator':
        logits,codes=generator_logits(components['model'],components['codec'],prefix,target,
                                     cfg['conditioning_noise'] if training else 0.0)
        result,levels=level_loss(logits,codes,cfg['levels'])
        metrics={'level_cross_entropy':levels}
    else:
        previous=torch.cat([torch.full_like(target[:,:1],EOS),target[:,:-1]],dim=1)
        result,metrics=byte_loss(components['model'](prefix,previous),target,cfg['pad_weight'])
    if not bool(torch.isfinite(result)):
        raise FloatingPointError('Non-finite loss; refusing optimizer update')
    metrics['loss']=float(result.detach())
    return result, metrics


@torch.no_grad()
def predict(components,stage,prefix,target):
    if stage=='codec':
        codec=components['codec']
        return codec.decode(codec.encode(target)).argmax(-1)
    if stage=='generator':
        return generate_ids(components['model'],components['codec'],prefix)
    return components['model'].generate(prefix)


def parameter_counts(components):
    return {name:{'total':sum(p.numel() for p in module.parameters()),
                  'trainable':sum(p.numel() for p in module.parameters() if p.requires_grad)}
            for name,module in components.items()}
