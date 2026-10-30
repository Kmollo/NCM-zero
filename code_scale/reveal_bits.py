"""Path C cause measurement (docs/ncm-next-plan.md, "Path C"): how many of NCM's extra bits come from
positions in a level being guessed blind to each other?

The codes are computed from the text, so a perfect code model needs no more bits per chunk than the text
itself; the AR control's cross-entropy on the same 64 tokens is our estimate of that. For a reveal-trained
NCM, teacher-forced bits per chunk are measured with
  1 pass : every position guessed with nothing of its own level shown (= upstream NCM),
  k pass : positions split per level into k random groups; group p is scored with groups < p shown
           (a valid k-step factorisation, so its sum is an achievable code length).
extra = bits_1pass - AR bits; removed = bits_1pass - bits_3pass. Rule (fixed before results):
the cause is blind guessing if removed >= 0.5 * extra, otherwise the codes themselves.

  CODE_SCALE_LEVELS=... CODE_SCALE_REVEAL=0,1,2,3,4,5,6 CODE_SCALE_NCM=ncm-rv7 python -m code_scale.reveal_bits
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
from code_scale.models import SHAPE, ncm_logits
from code_scale.plan_probe import encode
from code_scale.train import DEV, RUNS, windows

P, T, LEVELS = SHAPE["prefix_len"], SHAPE["chunk"], SHAPE["levels"]
BOUNDS = np.cumsum([0] + LEVELS)
LN2 = math.log(2)
N_WIN = 64 if "--quick" in sys.argv else 2048
PASSES = (1, 3, 8)


def groups(B, k):
    """(B, N) group index per position: a random split of each level into k near-equal groups."""
    g = torch.empty(B, BOUNDS[-1], dtype=torch.long, device=DEV)
    for l, n in enumerate(LEVELS):
        rank = torch.rand(B, n, device=DEV).argsort(1).argsort(1)
        g[:, BOUNDS[l]:BOUNDS[l + 1]] = rank * k // n
    return g


@torch.no_grad()
def main():
    tok, eos, pad = data.ids()
    codec, ncm, ar = load_models(tok.get_vocab_size(), pad)
    assert getattr(ncm, "reveal", None) is not None and ncm.reveal.reveal == list(range(len(LEVELS))), \
        "needs a model trained with reveals on every level"
    torch.manual_seed(0)
    w = windows(data.stream("val"), N_WIN, P + T, np.random.default_rng(21))
    codes_all = encode(codec, w[:, P:])
    bits = {k: torch.zeros(len(LEVELS), device=DEV) for k in PASSES}
    ar_bits = 0.0
    for i in range(0, N_WIN, 64):
        prefix, codes, win = w[i:i + 64, :P], codes_all[i:i + 64], w[i:i + 64]
        B = len(prefix)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = ar(win[:, :-1])[:, -T:]
        ar_bits += (F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), win[:, -T:].reshape(-1),
                                    reduction="sum") / LN2).item()
        for k in PASSES:
            g = groups(B, k)
            for p in range(k):
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = ncm_logits(ncm, codec, prefix, codes, pad, corrupt_rate=0.0, shown=g < p)
                nll = -F.log_softmax(logits.float(), -1).gather(-1, codes[..., None])[..., 0] / LN2
                nll = nll * (g == p)
                for l in range(len(LEVELS)):
                    bits[k][l] += nll[:, BOUNDS[l]:BOUNDS[l + 1]].sum()
    per_level = {k: [round(v / N_WIN, 1) for v in bits[k].tolist()] for k in PASSES}
    total = {k: round(sum(per_level[k]), 1) for k in PASSES}
    ar_chunk = round(ar_bits / N_WIN, 1)
    extra, removed = total[1] - ar_chunk, total[1] - total[3]
    report = {"windows": N_WIN, "ar_bits_per_chunk": ar_chunk,
              "ncm_bits_per_chunk": {f"{k}_pass": total[k] for k in PASSES},
              "ncm_bits_per_level": {f"{k}_pass": per_level[k] for k in PASSES},
              "extra_bits_1pass_vs_ar": round(extra, 1), "removed_by_3_passes": round(removed, 1),
              "removed_share": round(removed / extra, 3) if extra > 0 else None,
              "cause": "blind guessing within a level" if extra > 0 and removed >= 0.5 * extra else "the codes themselves",
              "rule": "cause = blind guessing if 3 passes remove >= half of the extra bits over AR (fixed before results)"}
    out = RUNS / os.environ.get("CODE_SCALE_NCM", "ncm") / "reveal-bits.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
