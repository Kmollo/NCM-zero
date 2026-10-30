"""Strict, dependency-free experiment configuration."""
import json
import math
from pathlib import Path

DEFAULTS = {"max_seconds": 21600, "gradient_clip": 1.0, "vq_weight": 1.0,
            "pad_weight": 0.1, "threads": 2}
POSITIVE_INTS = ("prefix_length", "output_length", "codebook_size", "tokenizer_dim",
                 "generator_dim", "generator_layers", "heads", "batch_size",
                 "accumulation", "steps", "save_every", "eval_every", "threads")
FIELDS = set(POSITIVE_INTS) | set(DEFAULTS) | {"levels", "fine_drop", "conditioning_noise",
          "learning_rate", "weight_decay", "precision", "seed"}


def validate(cfg):
    cfg = {**DEFAULTS, **cfg}
    if set(cfg) != FIELDS:
        raise ValueError(f"Configuration keys: missing={FIELDS-set(cfg)}, unknown={set(cfg)-FIELDS}")
    for key in POSITIVE_INTS:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    levels = cfg['levels']
    if (not isinstance(levels, list) or len(levels) < 2 or
        any(type(x) is not int or x < 1 for x in levels) or
        levels != sorted(set(levels)) or levels[-1] != cfg['output_length']):
        raise ValueError('levels must be strictly increasing positive integers ending at output_length')
    for key in ('tokenizer_dim', 'generator_dim'):
        if cfg[key] % cfg['heads']:
            raise ValueError(f'{key} must be divisible by heads')
    for key in ('max_seconds', 'gradient_clip', 'vq_weight', 'pad_weight',
                'learning_rate', 'weight_decay', 'fine_drop', 'conditioning_noise'):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if value < 0 or (key in ('max_seconds', 'gradient_clip', 'learning_rate') and value == 0):
            raise ValueError(f'invalid {key}')
    if not 0 < cfg['pad_weight'] <= 1 or not 0 <= cfg['fine_drop'] < 1 or not 0 <= cfg['conditioning_noise'] <= 1:
        raise ValueError('invalid padding/noise/drop probability')
    if cfg['precision'] not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    return cfg


def load_config(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8')))
