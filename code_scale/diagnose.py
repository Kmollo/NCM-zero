"""Phase A diagnostics for a trained NCM (docs/ncm-next-plan.md). Read-only: trains nothing.

A1  decoding sweep: same held-out windows, several ways of picking codes (and AR for reference),
    plus a repetition score and HumanEval syntax-valid rate per setting.
A2  where it breaks: teacher-forced accuracy per level, and generation with the TRUE codes
    forced for levels 0..k (k = -1 means normal generation, k = last means codec oracle).
A3  the same generation on training windows (generalization vs structure).

Run with the layout the model was trained with, e.g.
  CODE_SCALE_LEVELS=4,8,16,32,48,64,64 CODE_SCALE_CODEC=codec-l7 CODE_SCALE_NCM=ncm-l7 \
  CODE_SCALE_EVAL=diag-l7 python -m code_scale.diagnose
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from code_scale import data
from code_scale.evaluate import OUT, cut, load_models, similarity
from code_scale.models import SHAPE, ncm_logits
from code_scale.train import DEV, windows

P, T, LEVELS = SHAPE["prefix_len"], SHAPE["chunk"], SHAPE["levels"]
SETTINGS = {  # name -> (temperature, top_k, top_p)
    "greedy": (0.0, 0, 1.0),
    "t0.8_p0.95": (0.8, 0, 0.95),          # what our benchmark used
    "t1.0_full": (1.0, 0, 1.0),            # upstream eval configs
    "t1.0_k50_p0.9": (1.0, 50, 0.9),       # upstream generate() defaults
}


def rep4(ids):
    """Share of 4-grams in one output that repeat an earlier 4-gram (0 = no loops)."""
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def ncm_gen(ncm, codec, ctx, setting, forced=None, k=-1):
    """NCM chunk; if forced is given, levels 0..k use the true codes instead of the model's pick."""
    temp, top_k, top_p = SETTINGS[setting]
    original = ncm._sample_level
    if forced is not None and k >= 0:
        bounds = np.cumsum([0] + LEVELS)
        calls = {"level": 0}

        def sample(logits, *a, **kw):
            lvl = calls["level"]
            calls["level"] += 1
            if lvl <= k:
                return forced[:, bounds[lvl]:bounds[lvl + 1]]
            return original(logits, *a, **kw)
        ncm._sample_level = sample
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            codes = ncm.generate(None, ctx, temperature=temp, top_k=top_k, top_p=top_p)
    finally:
        ncm._sample_level = original
    return codec.decode(codes).argmax(-1)


@torch.no_grad()
def ar_gen(ar, ctx, setting):
    temp, _, top_p = SETTINGS[setting]
    with torch.autocast("cuda", dtype=torch.float16):
        return ar.generate(ctx, T, temperature=temp, top_p=top_p)[:, -T:]


def score(tok, gen, ref):
    texts = [tok.decode(g.tolist()) for g in gen]
    refs = [tok.decode(r.tolist()) for r in ref]
    return {"char_similarity": round(float(np.mean([similarity(a, b) for a, b in zip(texts, refs)])), 4),
            "token_match": round((gen == ref).float().mean().item(), 4),
            "repetition": round(float(np.mean([rep4(g.tolist()) for g in gen])), 4)}


def batched(fn, prefix, bs=32):
    return torch.cat([fn(prefix[i:i + bs], i) for i in range(0, len(prefix), bs)])


def humaneval_syntax(tok, eos, val, fn, n=5):
    tasks = [json.loads(l) for l in Path("evals/code-bench-v1/tasks/humaneval.jsonl").read_text(encoding="utf-8").splitlines()]
    filler = torch.from_numpy(val[:P].astype(np.int64)).to(DEV)
    ok, rep, example = [], [], None
    torch.manual_seed(0)
    for task in tasks:
        ids = torch.tensor(tok.encode(task["prompt"]).ids, device=DEV)
        seq = torch.cat([filler, torch.tensor([eos], device=DEV), ids])[None].expand(n, -1)
        start = seq.shape[1]
        for _ in range(2):
            seq = torch.cat([seq, fn(seq[:, -P:].contiguous())], dim=1)
        for row in seq[:, start:]:
            completion = cut(tok.decode(row.tolist()), "humaneval")
            rep.append(rep4(row.tolist()))
            try:
                compile(task["prompt"] + completion, "<candidate>", "exec")
                ok.append(1)
            except (SyntaxError, ValueError):
                ok.append(0)
            example = example or {"task_id": task["task_id"], "completion": completion[:300]}
    return {"syntax_valid_rate": round(float(np.mean(ok)), 4), "repetition": round(float(np.mean(rep)), 4),
            "samples": len(ok), "example": example}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok, eos, pad = data.ids()
    val, train = data.stream("val"), data.stream("train")
    codec, ncm, ar = load_models(tok.get_vocab_size(), pad)
    report = {"levels": LEVELS, "started": time.strftime("%Y-%m-%d %H:%M")}
    save = lambda: (OUT / "report.json").write_text(json.dumps(report, indent=2))

    w = windows(val, 256, P + T, np.random.default_rng(7))  # same windows as code_scale.evaluate
    prefix, ref = w[:, :P], w[:, P:]
    report["reference_repetition"] = round(float(np.mean([rep4(r.tolist()) for r in ref])), 4)

    # A1: decoding sweep on held-out continuations
    report["A1_continuation"] = {}
    for s in SETTINGS:
        torch.manual_seed(0)
        report["A1_continuation"][f"ncm_{s}"] = score(tok, batched(lambda c, i: ncm_gen(ncm, codec, c, s), prefix), ref)
        if s != "t1.0_k50_p0.9":
            torch.manual_seed(0)
            report["A1_continuation"][f"ar_{s}"] = score(tok, batched(lambda c, i: ar_gen(ar, c, s), prefix), ref)
        print(json.dumps({s: report["A1_continuation"].get(f"ncm_{s}")}), flush=True)
        save()

    # A2: per-level teacher-forced accuracy (true earlier levels, no corruption)
    codes = torch.cat([codec.encode(ref[i:i + 64]) for i in range(0, 256, 64)])
    picks = []
    for i in range(0, 256, 32):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            picks.append(ncm_logits(ncm, codec, prefix[i:i + 32], codes[i:i + 32], pad, corrupt_rate=0.0).argmax(-1))
    picks = torch.cat(picks)
    hits = (picks == codes).float()
    bounds = np.cumsum([0] + LEVELS)
    report["A2_teacher_forced_level_accuracy"] = [
        {"level": l, "codes": LEVELS[l], "accuracy": round(hits[:, bounds[l]:bounds[l + 1]].mean().item(), 4)}
        for l in range(len(LEVELS))]
    with torch.no_grad():
        tf_tokens = torch.cat([codec.decode(picks[i:i + 64]).argmax(-1) for i in range(0, 256, 64)])
    report["A2_teacher_forced_all_levels_decoded"] = score(tok, tf_tokens, ref)
    print(json.dumps(report["A2_teacher_forced_level_accuracy"]), flush=True)
    save()

    # A2: generation with true codes forced for levels 0..k (greedy for the rest)
    report["A2_forced_levels"] = []
    for k in range(-1, len(LEVELS)):
        gen = batched(lambda c, i: ncm_gen(ncm, codec, c, "greedy", codes[i:i + 32], k), prefix)
        report["A2_forced_levels"].append({"true_levels_0_to": k, **score(tok, gen, ref)})
        print(json.dumps(report["A2_forced_levels"][-1]), flush=True)
        save()

    # A3: the same on training windows
    wt = windows(train, 256, P + T, np.random.default_rng(11))
    pt, rt = wt[:, :P], wt[:, P:]
    report["A3_train_windows"] = {
        "ncm_greedy": score(tok, batched(lambda c, i: ncm_gen(ncm, codec, c, "greedy"), pt), rt),
        "ar_greedy": score(tok, batched(lambda c, i: ar_gen(ar, c, "greedy"), pt), rt)}
    print(json.dumps(report["A3_train_windows"]), flush=True)
    save()

    # A1: HumanEval syntax-valid rate per decoding setting (no sandbox scoring needed)
    report["A1_humaneval"] = {}
    for s in SETTINGS:
        report["A1_humaneval"][f"ncm_{s}"] = humaneval_syntax(tok, eos, val, lambda c: ncm_gen(ncm, codec, c, s))
        print(json.dumps({s: report["A1_humaneval"][f"ncm_{s}"]["syntax_valid_rate"]}), flush=True)
        save()
    report["A1_humaneval"]["ar_t0.8_p0.95"] = humaneval_syntax(tok, eos, val, lambda c: ar_gen(ar, c, "t0.8_p0.95"))
    report["finished"] = time.strftime("%Y-%m-%d %H:%M")
    save()
    print("DIAGNOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
