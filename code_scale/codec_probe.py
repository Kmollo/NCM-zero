"""Read-only probe of the trained code-scale-v1 tokenizer: where is information lost?"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from code_scale import data
from code_scale.models import SHAPE
import sys

from code_scale.models import build_codec
from code_scale.train import DEV, RUNS, windows


@torch.no_grad()
def main():
    tok, _, _ = data.ids()
    name = sys.argv[1] if len(sys.argv) > 1 else "codec"
    codec = build_codec(tok.get_vocab_size()).to(DEV)
    codec.load_state_dict(torch.load(RUNS / name / "final.pt", map_location=DEV, weights_only=False)["model"])
    codec.eval()
    vq = codec.vq
    x = windows(data.stream("val"), 1024, SHAPE["chunk"], np.random.default_rng(123))
    acc = lambda logits: (logits.argmax(-1) == x).float().mean().item()
    out = {}

    z = codec.encode_z(x).float()                         # continuous, pre-quantization
    out["decoder_on_continuous_z"] = acc(codec.decoder(z))

    idx, zhats = vq.z_to_idx(z, return_zhats=True)
    zz = z.pow(2).sum()
    out["per_level"] = []
    offset, prev_err = 0, 1.0
    for i, (lvl, zh) in enumerate(zip(vq.levels, zhats)):
        codes = idx[:, offset:offset + lvl].reshape(-1)
        counts = torch.bincount(codes, minlength=vq.vqvae_vocab_sizes[i]).float()
        p = counts[counts > 0] / counts.sum()
        err = ((z - zh).pow(2).sum() / zz).item()
        ema = getattr(vq, f"ema_cluster_size_{i}")
        out["per_level"].append({
            "level": i, "positions": lvl,
            "codes_used_on_val": int((counts > 0).sum()), "codebook": vq.vqvae_vocab_sizes[i],
            "code_perplexity": float(torch.exp(-(p * p.log()).sum())),
            "relative_residual_error_after": round(err, 4),
            "error_removed_by_level": round(prev_err - err, 4),
            "decode_accuracy_using_levels_0_to_here": acc(codec.decoder(zh)),
            "ema_cluster_size_mean": float(ema.mean()),
            "codes_below_restart_threshold": int((ema < vq.restart_threshold).sum()),
            "expected_vectors_per_step_per_code": 64 * lvl / vq.vqvae_vocab_sizes[i],
        })
        prev_err, offset = err, offset + lvl

    # How much does the finest residual matter? Replace only the last level with exact z.
    out["decode_with_all_coarse_levels_plus_exact_finest_residual"] = acc(codec.decoder(z))
    # Token-identity capacity: unique tokens per finest code (collision rate).
    fine = idx[:, -SHAPE["chunk"]:].reshape(-1).cpu().numpy()
    toks = x.reshape(-1).cpu().numpy()
    pairs = {}
    for c, t in zip(fine, toks):
        pairs.setdefault(c, set()).add(t)
    out["finest_code_mean_distinct_tokens"] = float(np.mean([len(v) for v in pairs.values()]))
    out["val_unique_tokens"] = int(len(set(toks.tolist())))
    # Scale check: codebook vector norms vs the residual they quantize, per level.
    out["z_rms"] = z.pow(2).mean().sqrt().item()
    out["codebook_rms"] = [vq.embeddings[i].weight.pow(2).mean().sqrt().item() for i in range(len(vq.levels))]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
