"""Phase A2 (docs/ncm-next-plan.md): is NCM's coarse "plan" predictable from the prefix at all?

A2.1  bits of help per level: NCM teacher-forced cross-entropy vs a blind code-frequency guess.
A2.2  probe: can the AR control's reading of the prefix predict the level 0-2 codes?
A2.3  stability: share of level 0-2 codes that change when one identifier is renamed or the
      window shifts by one token.
All cross-entropies are bits per code on held-out windows. Read-only for every model.

  CODE_SCALE_LEVELS=4,8,16,32,48,64,64 CODE_SCALE_CODEC=codec-l7 CODE_SCALE_NCM=ncm-l7 \
    python -m code_scale.plan_probe            # all three checks
  CODE_SCALE_LEVELS=4,8,16,32,64 python -m code_scale.plan_probe --gain-only   # old buggy run
  CODE_SCALE_LEVELS=... CODE_SCALE_CODEC=codec-pp05 python -m code_scale.plan_probe --probe-only
      # a new tokenizer, no NCM yet: baseline + probes + stability -> runs/.../<codec>/plan-probe.json
"""
import json
import keyword
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from code_scale import data
from code_scale.evaluate import load_models
from code_scale.models import SHAPE, build_ar, ncm_logits
from code_scale.train import DEV, RUNS, load_codec, windows

P, T, LEVELS, V = SHAPE["prefix_len"], SHAPE["chunk"], SHAPE["levels"], SHAPE["codebook"]
BOUNDS = np.cumsum([0] + LEVELS)
LN2 = math.log(2)
PROBE_ONLY = "--probe-only" in sys.argv
OUT = (RUNS / os.environ.get("CODE_SCALE_CODEC", "codec") if PROBE_ONLY
       else RUNS / os.environ.get("CODE_SCALE_NCM", "ncm")) / "plan-probe.json"
FEATS = RUNS / "ar-prefix-feats-v21-t22.pt"  # AR features of the fixed probe windows; same for every tokenizer


@torch.no_grad()
def encode(codec, chunks, bs=256):
    return torch.cat([codec.encode(chunks[i:i + bs]) for i in range(0, len(chunks), bs)])


def baseline_bits(train_codes, val_codes):
    """Per-level CE (bits) of val codes under add-one-smoothed train code frequencies."""
    out = []
    for l in range(len(LEVELS)):
        tr = train_codes[:, BOUNDS[l]:BOUNDS[l + 1]].reshape(-1)
        va = val_codes[:, BOUNDS[l]:BOUNDS[l + 1]].reshape(-1)
        vs = SHAPE["codebooks"][l]
        p = (torch.bincount(tr, minlength=vs).float() + 1) / (tr.numel() + vs)
        out.append((-torch.log2(p[va])).mean().item())
    return out


@torch.no_grad()
def ncm_bits(ncm, codec, prefix, codes, pad, bs=64):
    total = torch.zeros(len(LEVELS), device=DEV)
    for i in range(0, len(prefix), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ncm_logits(ncm, codec, prefix[i:i + bs], codes[i:i + bs], pad, corrupt_rate=0.0)
        nll = -F.log_softmax(logits.float(), -1).gather(-1, codes[i:i + bs, :, None])[..., 0] / LN2
        for l in range(len(LEVELS)):
            total[l] += nll[:, BOUNDS[l]:BOUNDS[l + 1]].mean(1).sum()
    return (total / len(prefix)).tolist()


@torch.no_grad()
def ar_features(ar, prefix, bs=128):
    """Last-position and mean hidden state of the AR control over the prefix (768-d)."""
    feats = []
    for i in range(0, len(prefix), bs):
        idx = prefix[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            x = ar.wte(idx) + ar.wpe(torch.arange(idx.shape[1], device=DEV))
            for block in ar.blocks:
                x = block(x)
            h = ar.ln(x).float()
        feats.append(torch.cat([h[:, -1], h.mean(1)], -1))
    return torch.cat(feats)


class Probe(nn.Module):
    def __init__(self, d_in, positions, hidden):
        super().__init__()
        self.pos = nn.Embedding(positions, 64)
        self.net = (nn.Linear(d_in + 64, V) if hidden == 0 else
                    nn.Sequential(nn.Linear(d_in + 64, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, V)))

    def forward(self, f):  # f: (B, d) -> (B, positions, V)
        B, n = f.shape[0], self.pos.num_embeddings
        x = torch.cat([f[:, None].expand(B, n, -1), self.pos.weight[None].expand(B, n, -1)], -1)
        return self.net(x)


def train_probe(ftr, ytr, fva, yva, hidden, epochs=12, bs=512):
    """Train on 90% of ftr, pick the epoch by the other 10%, report val bits/code."""
    torch.manual_seed(0)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-5
    ftr, fva = (ftr - mu) / sd, (fva - mu) / sd
    cut = int(0.9 * len(ftr))
    probe = Probe(ftr.shape[1], ytr.shape[1], hidden).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.05)
    bits = lambda f, y: torch.cat([F.cross_entropy(probe(f[i:i + 2048]).reshape(-1, V), y[i:i + 2048].reshape(-1),
                                                   reduction="none") for i in range(0, len(f), 2048)]).mean().item() / LN2
    best_dev, best_val = float("inf"), None
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(cut, device=DEV)
        for i in range(0, cut, bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(probe(ftr[j]).reshape(-1, V), ytr[j].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            dev = bits(ftr[cut:], ytr[cut:])
            if dev < best_dev:
                best_dev, best_val = dev, bits(fva, yva)
    return best_val


def identifier_tokens(tok):
    ids = {}
    for t, i in tok.get_vocab().items():
        s = tok.decode([i])
        core = s.lstrip(" ")
        if len(core) >= 3 and core.isidentifier() and not keyword.iskeyword(core):
            ids[i] = s.startswith(" ")
    return ids


@torch.no_grad()
def stability(tok, codec, stream, n=1024):
    rng = np.random.default_rng(5)
    starts = rng.integers(0, len(stream) - T - 1, size=4 * n)
    ident = identifier_tokens(tok)
    by_space = {sp: [i for i, s in ident.items() if s == sp] for sp in (True, False)}
    base, renamed, shifted = [], [], []
    for s in starts:
        chunk = stream[s:s + T].astype(np.int64)
        cands = [t for t in set(chunk.tolist()) if t in ident]
        if not cands:
            continue
        old = max(cands, key=lambda t: (chunk == t).sum())
        new = int(rng.choice([t for t in by_space[ident[old]] if t != old]))
        base.append(chunk)
        renamed.append(np.where(chunk == old, new, chunk))
        shifted.append(stream[s + 1:s + 1 + T].astype(np.int64))
        if len(base) == n:
            break
    to = lambda a: torch.from_numpy(np.stack(a)).to(DEV)
    cb, cr, cs = encode(codec, to(base)), encode(codec, to(renamed)), encode(codec, to(shifted))
    perm = torch.randperm(len(cb), generator=torch.Generator().manual_seed(1)).to(DEV)
    out = []
    for l in range(3):
        sl = slice(BOUNDS[l], BOUNDS[l + 1])
        out.append({"level": l,
                    "changed_by_one_identifier_rename": round((cb[:, sl] != cr[:, sl]).float().mean().item(), 4),
                    "changed_by_one_token_shift": round((cb[:, sl] != cs[:, sl]).float().mean().item(), 4),
                    "changed_vs_unrelated_chunk": round((cb[:, sl] != cb[perm, sl]).float().mean().item(), 4)})
    return out, len(base)


def main():
    gain_only = "--gain-only" in sys.argv
    tok, eos, pad = data.ids()
    train, val = data.stream("train"), data.stream("val")
    if PROBE_ONLY:
        codec, ncm = load_codec(tok.get_vocab_size()), None
        ar = build_ar(tok.get_vocab_size()).to(DEV)
        ar.load_state_dict(torch.load(RUNS / "ar/final.pt", map_location=DEV, weights_only=False)["model"])
        ar.eval()
    else:
        codec, ncm, ar = load_models(tok.get_vocab_size(), pad)
    report = {"levels": LEVELS, "codec": os.environ.get("CODE_SCALE_CODEC", "codec"),
              "ncm": os.environ.get("CODE_SCALE_NCM", "ncm"), "started": time.strftime("%Y-%m-%d %H:%M")}
    save = lambda: OUT.write_text(json.dumps(report, indent=2))

    wv = windows(val, 20480, P + T, np.random.default_rng(21))
    wt = windows(train, 51200, P + T, np.random.default_rng(22))
    cv, ct = encode(codec, wv[:, P:]), encode(codec, wt[:, P:])
    base = baseline_bits(ct, cv)
    model = ncm_bits(ncm, codec, wv[:, :P], cv, pad) if ncm is not None else [float("nan")] * len(LEVELS)
    report["A2_1_bits_per_code"] = [
        {"level": l, "codes": LEVELS[l], "uniform": round(math.log2(SHAPE["codebooks"][l]), 3), "blind_guess": round(base[l], 3),
         "ncm": round(model[l], 3), "ncm_gain": round(base[l] - model[l], 3),
         "ncm_sees_true_earlier_levels": l > 0} for l in range(len(LEVELS))]
    if ncm is None:
        for row in report["A2_1_bits_per_code"]:
            row["ncm"] = row["ncm_gain"] = None
    else:  # what NCM must predict per 64-token chunk, teacher-forced (lower = easier); added 2026-10-29
        report["ncm_total_bits_per_chunk"] = round(sum(LEVELS[l] * model[l] for l in range(len(LEVELS))), 1)
        report["codebooks"] = SHAPE["codebooks"]
    print(json.dumps(report["A2_1_bits_per_code"]), flush=True)
    save()
    if gain_only:
        return

    if FEATS.exists():
        fv, ft = (t.to(DEV).float() for t in torch.load(FEATS, map_location=DEV, weights_only=True))
    else:
        fv, ft = ar_features(ar, wv[:, :P]), ar_features(ar, wt[:, :P])
        torch.save((fv.half().cpu(), ft.half().cpu()), FEATS)
        fv, ft = fv.half().float(), ft.half().float()  # identical inputs whether cached or not
    report["A2_2_probe_bits_per_code"] = []
    for l in range(3):
        sl = slice(BOUNDS[l], BOUNDS[l + 1])
        row = {"level": l, "blind_guess": round(base[l], 3), "ncm": round(model[l], 3) if ncm is not None else None}
        for name, hidden in (("linear", 0), ("mlp", 512)):
            row[f"probe_{name}"] = round(train_probe(ft, ct[:, sl], fv, cv[:, sl], hidden), 3)
            row[f"probe_{name}_gain"] = round(base[l] - row[f"probe_{name}"], 3)
        report["A2_2_probe_bits_per_code"].append(row)
        print(json.dumps(row), flush=True)
        save()

    report["A2_3_stability"], report["A2_3_chunks"] = stability(tok, codec, val)
    print(json.dumps(report["A2_3_stability"]), flush=True)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M")
    save()
    print("PLAN_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
