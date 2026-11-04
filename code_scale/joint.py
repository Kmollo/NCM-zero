"""Joint training of tokenizer and generator (docs/joint-training-plan.md).

The tokenizer's code choice is a hard nearest-code pick, and upstream builds every residual from z.detach(), so
no signal about how guessable the codes are can reach the encoder. soft_codes() re-runs the same level-by-level
choice WITH gradient to z and turns the distances into soft scores q = softmax(-distance / tau). argmax q is
exactly the chosen code (checked by tests/test_joint.py). guess_term() then asks the encoder to put its soft
scores where the generator already expects them: sum_c q(c) * -log p_model(c), with p_model held fixed.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from code_scale.models import SHAPE

LEVELS = SHAPE["levels"]
BOUNDS = np.cumsum([0] + LEVELS)


def soft_codes(vq, z, taus):
    """Same choice as vq.z_to_idx (fp32, no stochastic encode), but differentiable w.r.t. z.

    Returns codes (B, N) and log q (B, N, V) where V is the largest codebook (smaller ones padded with -inf).
    Codebook vectors and the quant_resi path are held fixed; only z (the encoder) and level 0's learned
    down-sampling receive gradient.
    """
    with torch.autocast("cuda", enabled=False):
        z = z.float()
        B, L, C = z.shape
        V = max(vq.vqvae_vocab_sizes)
        z_rest = z
        codes, logq = [], []
        n = len(vq.levels)
        for i, level in enumerate(vq.levels):
            emb = vq.embeddings[i].weight.detach().float()
            flat = vq._down_for_quant(z_rest, i, level) if i != n - 1 else z_rest
            d = vq._distances(flat.reshape(-1, C), emb)
            idx = d.argmin(1).view(B, level)
            lq = F.log_softmax(-d / taus[i], dim=1).view(B, level, -1)
            if lq.shape[-1] < V:
                lq = F.pad(lq, (0, V - lq.shape[-1]), value=float("-inf"))
            codes.append(idx)
            logq.append(lq)
            with torch.no_grad():
                q = vq.embeddings[i](idx).float()
                q = vq._up_to_full(q, i, L) if i != n - 1 else q
                q = vq._apply_quant_resi(q, i)
            z_rest = z_rest - q
        return torch.cat(codes, 1), torch.cat(logq, 1)


def guess_term(logq, gen_logits):
    """Mean over code positions of sum_c q(c) * -log p_model(c) (nats); p_model is held fixed."""
    logp = F.log_softmax(gen_logits.detach().float(), -1)
    q = logq.exp()
    return -(q * logp.masked_fill(q == 0, 0.0)).sum(-1).mean()


@torch.no_grad()
def near_tie_taus(codec, chunks, bs=128):
    """tau per level = median gap between the second-nearest and nearest code distance (fp32)."""
    vq = codec.vq
    gaps = [[] for _ in LEVELS]
    for s in range(0, len(chunks), bs):
        z = codec._encode_pre_quant(chunks[s:s + bs]).float()
        B, L, C = z.shape
        z_rest = z.clone()
        for i, level in enumerate(vq.levels):
            emb = vq.embeddings[i].weight.float()
            flat = vq._down_for_quant(z_rest, i, level) if i != len(LEVELS) - 1 else z_rest
            d = vq._distances(flat.reshape(-1, C), emb)
            top2 = d.topk(2, dim=1, largest=False).values
            gaps[i].append((top2[:, 1] - top2[:, 0]).cpu())
            idx = d.argmin(1).view(B, level)
            q = vq.embeddings[i](idx).float()
            q = vq._up_to_full(q, i, L) if i != len(LEVELS) - 1 else q
            z_rest = z_rest - vq._apply_quant_resi(q, i)
    return [float(torch.cat(g).median()) for g in gaps]


def load_taus(path):
    return json.loads(open(path).read())["taus"]


def main():
    """python -m code_scale.joint taus  -> runs/code-scale-v1/joint-taus.json from codec-l7 (1,024 held-out chunks)"""
    import sys
    from code_scale import data
    from code_scale.train import CODEC_DIR, RUNS, load_codec, windows
    assert sys.argv[1:] == ["taus"]
    codec = load_codec(data.ids()[0].get_vocab_size())
    x = windows(data.stream("val"), 1024, SHAPE["chunk"], np.random.default_rng(7))
    taus = near_tie_taus(codec, x)
    out = RUNS / "joint-taus.json"
    out.write_text(json.dumps({"codec": CODEC_DIR, "chunks": 1024, "seed": 7, "taus": taus,
                               "rule": "median gap between 2nd-nearest and nearest code distance, per level"}, indent=2))
    print(out.read_text())


if __name__ == "__main__":
    main()
