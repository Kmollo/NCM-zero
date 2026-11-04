"""Second CPU-only read check of codec-l7: PCA dimensionality of per-level quantizer inputs
(FSQ feasibility), full swap to second-nearest at levels 5/6 (RDOQ worst case), level-0
down-conv scale, codebook usage on the sample."""
import os, sys, json, time
os.environ["CODE_SCALE_LEVELS"] = "4,8,16,32,48,64,64"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np, torch
torch.set_num_threads(8)
from code_scale.models import build_codec, SHAPE
from code_scale import data

t0 = time.time()
tok, eos, pad = data.ids()
codec = build_codec(tok.get_vocab_size())
codec.load_state_dict(torch.load("runs/code-scale-v1/codec-l7/final.pt", map_location="cpu", weights_only=False)["model"])
codec.eval()
val = data.stream("val")
T, LEVELS = SHAPE["chunk"], SHAPE["levels"]; B = np.cumsum([0] + LEVELS)
rng = np.random.default_rng(7)
n = 256
starts = rng.integers(0, len(val) - T - 1, size=n)
x = torch.from_numpy(np.stack([val[s:s + T].astype(np.int64) for s in starts]))
out = {}
with torch.no_grad():
    z = codec._encode_pre_quant(x)
    idx = codec.vq.z_to_idx(z)
    vq = codec.vq
    z_rest = z.clone(); pca = []; usage = []
    for i, level in enumerate(LEVELS):
        emb = vq.embeddings[i].weight
        flat = (vq._down_for_quant(z_rest, i, level) if i != len(LEVELS) - 1 else z_rest).reshape(-1, z.shape[-1])
        c = flat - flat.mean(0)
        ev = torch.linalg.svdvals(c).pow(2); ev = ev / ev.sum()
        cum = ev.cumsum(0)
        pca.append({"level": i, "var_top4": round(cum[3].item(), 3), "var_top8": round(cum[7].item(), 3),
                    "var_top16": round(cum[15].item(), 3), "var_top32": round(cum[31].item(), 3),
                    "dims_for_90pct": int((cum < 0.9).sum().item()) + 1})
        used = idx[:, B[i]:B[i + 1]].unique().numel()
        usage.append({"level": i, "distinct_codes_in_sample": used, "positions": int(n * level)})
        ind = idx[:, B[i]:B[i + 1]]
        q = emb[ind]
        quant = vq._up_to_full(q, i, T) if i != len(LEVELS) - 1 else q
        z_rest = z_rest - vq._apply_quant_resi(quant, i)
    out["PCA_quantizer_inputs"] = pca
    out["codebook_usage_sample"] = usage
    # level-0 learned down conv scale vs init (1/16 identity)
    w = vq.learned_down_l1.weight  # (D, D, 16)
    D = w.shape[0]
    init = torch.zeros_like(w); init[torch.arange(D), torch.arange(D), :] = 1.0 / 16
    out["level0_down_conv"] = {"weight_fro_norm": round(w.norm().item(), 2), "init_fro_norm": round(init.norm().item(), 2),
                               "ratio": round((w.norm() / init.norm()).item(), 1),
                               "codebook0_mean_norm": round(vq.embeddings[0].weight.norm(dim=1).mean().item(), 1),
                               "codebook5_mean_norm": round(vq.embeddings[5].weight.norm(dim=1).mean().item(), 1),
                               "up_conv_fro_norm": round(vq.learned_up_l1.weight.norm().item(), 2)}
    # full swap of a level to its second-nearest code (worst case for RDOQ with m=2)
    base = codec.decode(idx).argmax(-1)
    z_rest = z.clone(); full = []
    for i, level in enumerate(LEVELS):
        emb = vq.embeddings[i].weight
        flat = (vq._down_for_quant(z_rest, i, level) if i != len(LEVELS) - 1 else z_rest).reshape(-1, z.shape[-1])
        d = vq._distances(flat, emb)
        second = d.topk(2, largest=False).indices[:, 1].view(-1, level)
        idx2 = idx.clone(); idx2[:, B[i]:B[i + 1]] = second
        rec2 = codec.decode(idx2).argmax(-1)
        full.append({"level": i, "token_acc_vs_truth": round((rec2 == x).float().mean().item(), 4),
                     "tokens_unchanged_vs_argmin": round((rec2 == base).float().mean().item(), 4),
                     "chunks_unchanged_vs_argmin": round((rec2 == base).all(1).float().mean().item(), 3)})
        ind = idx[:, B[i]:B[i + 1]]
        q = emb[ind]
        quant = vq._up_to_full(q, i, T) if i != len(LEVELS) - 1 else q
        z_rest = z_rest - vq._apply_quant_resi(quant, i)
    out["full_level_swap_to_second_nearest"] = full
    out["base_token_acc"] = round((base == x).float().mean().item(), 4)
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
