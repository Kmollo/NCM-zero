"""Idea 1 bits measurement (docs/alien-search/README.md): NCM's teacher-forced bits per 64-token chunk, per level,
against the AR control's bits on the same chunks. Works for any NCM (no reveal needed).

Same held-out windows as code_scale/reveal_bits.py (2,048 windows, seed 21). Also reports, per level, the bits of
a code-frequency prior (no prefix at all) fitted on 8,192 training chunks, to show how much the prefix helps.

  CODE_SCALE_LEVELS=... CODE_SCALE_ENC=... CODE_SCALE_CODEC=... CODE_SCALE_NCM=... python -m code_scale.level_bits
Writes runs/code-scale-v1/<NCM>/level-bits.json.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from code_scale import data
from code_scale.evaluate import load_models
from code_scale.models import ENC, SHAPE, ncm_logits
from code_scale.plan_probe import baseline_bits, encode
from code_scale.train import CODEC_DIR, DEV, RUNS, windows

P, T, LEVELS = SHAPE["prefix_len"], SHAPE["chunk"], SHAPE["levels"]
BOUNDS = np.cumsum([0] + LEVELS)
LN2 = math.log(2)
N_WIN = 64 if "--quick" in sys.argv else 2048


@torch.no_grad()
def main():
    tok, eos, pad = data.ids()
    codec, ncm, ar = load_models(tok.get_vocab_size(), pad)
    torch.manual_seed(0)
    w = windows(data.stream("val"), N_WIN, P + T, np.random.default_rng(21))
    codes_all = encode(codec, w[:, P:])
    bits = torch.zeros(len(LEVELS), device=DEV)
    ar_bits = 0.0
    for i in range(0, N_WIN, 64):
        prefix, codes, win = w[i:i + 64, :P], codes_all[i:i + 64], w[i:i + 64]
        with torch.autocast("cuda", dtype=torch.float16):
            lg = ar(win[:, :-1])[:, -T:]
            logits = ncm_logits(ncm, codec, prefix, codes, pad, corrupt_rate=0.0)
        ar_bits += (F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), win[:, -T:].reshape(-1),
                                    reduction="sum") / LN2).item()
        nll = -F.log_softmax(logits.float(), -1).gather(-1, codes[..., None])[..., 0] / LN2
        for l in range(len(LEVELS)):
            bits[l] += nll[:, BOUNDS[l]:BOUNDS[l + 1]].sum()
    train_codes = encode(codec, windows(data.stream("train"), 8192, T, np.random.default_rng(5)))
    prior = [round(b * n, 1) for b, n in zip(baseline_bits(train_codes, codes_all), LEVELS)]
    per_level = [round(v / N_WIN, 1) for v in bits.tolist()]
    ar_chunk = round(ar_bits / N_WIN, 1)
    report = {"ncm": os.environ.get("CODE_SCALE_NCM", "ncm"), "codec": CODEC_DIR, "encoder": ENC,
              "windows": N_WIN, "ar_bits_per_chunk": ar_chunk,
              "ncm_bits_per_chunk": round(sum(per_level), 1), "ncm_bits_per_level": per_level,
              "extra_bits_vs_ar": round(sum(per_level) - ar_chunk, 1),
              "prior_bits_per_level": prior, "prior_bits_per_chunk": round(sum(prior), 1)}
    out = RUNS / report["ncm"] / "level-bits.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
