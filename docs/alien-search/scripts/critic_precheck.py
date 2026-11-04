"""CPU-only read check of codec-l7 (no training, no GPU). Reports:
 A. token accuracy when decoding from levels 0..k only (how much fine levels carry)
 B. energy removed per level and nearest/second-nearest distance gap statistics
 C. shift stability per level (slot-wise, and equivariant for full-resolution levels)
 D. one-shot near-tie swap: fraction of tokens unchanged when 10% of level-k codes are
    replaced by their second-nearest code (RDOQ pre-test proxy)
"""
import os, sys, json, time
os.environ["CODE_SCALE_LEVELS"] = "4,8,16,32,48,64,64"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from code_scale.models import build_codec, SHAPE
from code_scale import data

t0 = time.time()
tok, eos, pad = data.ids()
V = tok.get_vocab_size()
codec = build_codec(V)
state = torch.load("runs/code-scale-v1/codec-l7/final.pt", map_location="cpu", weights_only=False)
codec.load_state_dict(state["model"]); codec.eval()
val = data.stream("val")
T = SHAPE["chunk"]; LEVELS = SHAPE["levels"]; B = np.cumsum([0] + LEVELS)
rng = np.random.default_rng(123)
n = 192
starts = rng.integers(0, len(val) - T - 1, size=n)
x = torch.from_numpy(np.stack([val[s:s + T].astype(np.int64) for s in starts]))
xs = torch.from_numpy(np.stack([val[s + 1:s + 1 + T].astype(np.int64) for s in starts]))
out = {}
with torch.no_grad():
    z = codec._encode_pre_quant(x)
    idx, zhats = codec.vq.z_to_idx(z, return_zhats=True)
    # A. truncated decode
    acc = []
    for k, zh in enumerate(zhats):
        rec = codec.decoder(zh).argmax(-1)
        acc.append({"levels_0_to": k, "token_acc": round((rec == x).float().mean().item(), 4),
                    "chunk_exact": round((rec == x).all(1).float().mean().item(), 4)})
    out["A_truncated_decode"] = acc
    # B. energy + gaps, replaying the residual loop
    vq = codec.vq
    z_rest = z.clone(); stats = []
    for i, level in enumerate(LEVELS):
        emb = vq.embeddings[i].weight
        flat = (vq._down_for_quant(z_rest, i, level) if i != len(LEVELS) - 1 else z_rest).reshape(-1, z.shape[-1])
        d = vq._distances(flat, emb)
        srt, order = d.sort(1)
        gap = (srt[:, 1] - srt[:, 0])
        within_gap = (d <= (srt[:, :1] + gap[:, None])).sum(1).float()  # codes within one gap of the best
        ind = order[:, 0].view(-1, level)
        q = emb[ind]
        quant = vq._up_to_full(q, i, T) if i != len(LEVELS) - 1 else q
        quant = vq._apply_quant_resi(quant, i)
        e_in = z_rest.pow(2).sum(-1).mean().item()
        new_rest = z_rest - quant
        e_out = new_rest.pow(2).sum(-1).mean().item()
        stats.append({"level": i, "E_rest_in": round(e_in, 2), "E_rest_out": round(e_out, 2),
                      "frac_energy_removed": round(1 - e_out / e_in, 3),
                      "nearest_d2_median": round(srt[:, 0].median().item(), 2),
                      "gap_1st_2nd_median": round(gap.median().item(), 3),
                      "gap_over_nearest_median": round((gap / srt[:, 0].clamp(min=1e-6)).median().item(), 4),
                      "codes_within_one_gap_median": round(within_gap.median().item(), 1),
                      "codes_within_one_gap_mean": round(within_gap.mean().item(), 1)})
        z_rest = new_rest
    out["B_residual_stats"] = stats
    # C. shift stability per level
    idx_s = codec.encode(xs)
    stab = []
    for l in range(len(LEVELS)):
        a, b = idx[:, B[l]:B[l + 1]], idx_s[:, B[l]:B[l + 1]]
        row = {"level": l, "slotwise_changed": round((a != b).float().mean().item(), 3)}
        if LEVELS[l] == T:  # equivariant: code[t+1] before vs code[t] after shift
            row["equivariant_changed"] = round((a[:, 1:] != b[:, :-1]).float().mean().item(), 3)
        stab.append(row)
    out["C_shift_stability"] = stab
    # D. near-tie swap at each level: 10% of positions -> second nearest code, decode, compare
    base_rec = codec.decode(idx).argmax(-1)
    swaps = []
    g = torch.Generator().manual_seed(0)
    z_rest = z.clone()
    for i, level in enumerate(LEVELS):
        emb = vq.embeddings[i].weight
        flat = (vq._down_for_quant(z_rest, i, level) if i != len(LEVELS) - 1 else z_rest).reshape(-1, z.shape[-1])
        d = vq._distances(flat, emb)
        second = d.topk(2, largest=False).indices[:, 1].view(-1, level)
        mask = torch.rand(second.shape, generator=g) < 0.10
        idx2 = idx.clone()
        seg = idx2[:, B[i]:B[i + 1]]
        idx2[:, B[i]:B[i + 1]] = torch.where(mask, second, seg)
        rec2 = codec.decode(idx2).argmax(-1)
        swaps.append({"level": i, "positions_swapped": int(mask.sum()),
                      "token_acc_vs_truth": round((rec2 == x).float().mean().item(), 4),
                      "chunks_unchanged_vs_argmin_decode": round((rec2 == base_rec).all(1).float().mean().item(), 3),
                      "tokens_unchanged_vs_argmin_decode": round((rec2 == base_rec).float().mean().item(), 4)})
        ind = idx[:, B[i]:B[i + 1]]
        q = emb[ind]
        quant = vq._up_to_full(q, i, T) if i != len(LEVELS) - 1 else q
        z_rest = z_rest - vq._apply_quant_resi(quant, i)
    out["D_second_nearest_swap_10pct"] = swaps
    out["base_token_acc"] = round((base_rec == x).float().mean().item(), 4)
    out["base_chunk_exact"] = round((base_rec == x).all(1).float().mean().item(), 4)
out["n_chunks"] = n
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
