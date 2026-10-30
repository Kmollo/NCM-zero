"""Step 6 evaluation: continuation quality, benchmark samples, generation speed.

python -m code_scale.evaluate   -> runs/code-scale-v1/eval/ (report.json + samples for code_bench)
"""
import difflib
import json
import time
from pathlib import Path

import numpy as np
import torch

from code_scale import data
from code_scale.models import ATTN, SHAPE, build_ar, build_ncm, install_generator, is_block_causal
from code_scale.train import DEV, RUNS, load_codec, windows

P, T = SHAPE["prefix_len"], SHAPE["chunk"]
import os
NCM_DIR = os.environ.get("CODE_SCALE_NCM", "ncm")
OUT = RUNS / os.environ.get("CODE_SCALE_EVAL", "eval")
STOPS = {"humaneval": ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\nassert "],
         "easy": ["\nif __name__", "\nprint(", "\nassert ", "\n\n\n\n"]}


def load_models(vocab, pad):
    codec = load_codec(vocab)
    ncm = build_ncm(codec, pad).to(DEV)
    ncm.load_state_dict(torch.load(RUNS / NCM_DIR / "final.pt", map_location=DEV, weights_only=False)["model"])
    if is_block_causal(ncm) != (ATTN == "block_causal"):  # the mask buffer comes from the checkpoint
        raise ValueError(f"{NCM_DIR} was trained with a different attention pattern than CODE_SCALE_ATTN={ATTN}")
    install_generator(ncm)
    ar = build_ar(vocab).to(DEV)
    ar.load_state_dict(torch.load(RUNS / "ar/final.pt", map_location=DEV, weights_only=False)["model"])
    return codec, ncm.eval(), ar.eval()


@torch.no_grad()
def ncm_chunk(ncm, codec, ctx, temperature):
    kw = dict(temperature=0.0, top_k=0, top_p=1.0) if temperature == 0 else dict(temperature=temperature, top_k=0, top_p=0.95)
    with torch.autocast("cuda", dtype=torch.float16):
        codes = ncm.generate(None, ctx, **kw)
    return codec.decode(codes).argmax(-1)


@torch.no_grad()
def ar_chunk(ar, ctx, temperature):
    with torch.autocast("cuda", dtype=torch.float16):
        return ar.generate(ctx, T, temperature=temperature, top_p=0.95)[:, -T:]


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def continuation(tok, codec, ncm, ar, val, n=256):
    rng = np.random.default_rng(7)
    w = windows(val, n, P + T, rng)
    prefix, ref = w[:, :P], w[:, P:]
    outs = {"ncm": [], "ar": [], "codec_oracle": []}
    for i in range(0, n, 32):
        outs["ncm"].append(ncm_chunk(ncm, codec, prefix[i:i + 32], 0.0))
        outs["ar"].append(ar_chunk(ar, prefix[i:i + 32], 0.0))
        outs["codec_oracle"].append(codec.decode(codec.encode(ref[i:i + 32])).argmax(-1))
    ref_text = [tok.decode(r.tolist()) for r in ref]
    result = {}
    for name, chunks in outs.items():
        gen = torch.cat(chunks)
        texts = [tok.decode(g.tolist()) for g in gen]
        result[name] = {"token_match": (gen == ref).float().mean().item(),
                        "char_similarity": float(np.mean([similarity(a, b) for a, b in zip(texts, ref_text)]))}
    result["examples"] = [{"prefix_tail": tok.decode(prefix[j, -48:].tolist()), "reference": ref_text[j],
                           "ncm": tok.decode(torch.cat(outs["ncm"])[j].tolist()),
                           "ar": tok.decode(torch.cat(outs["ar"])[j].tolist())} for j in range(4)]
    return result


def cut(text, suite):
    text = text.split("<|endoftext|>")[0]
    for s in STOPS[suite]:
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text.rstrip() + "\n"


def has_code(completion):
    """True if the completion has at least one line that is not blank and not a comment.
    A blank (or comment-only) completion after a docstring prompt compiles, so plain
    syntax-valid counts it as a success (amendment 2026-10-29)."""
    return any(l.strip() and not l.strip().startswith("#") for l in completion.splitlines())


def bench(tok, eos, codec, ncm, ar, val, suite, n, temperature=0.8, chunks=2, limit=None):
    tasks = [json.loads(l) for l in Path(f"evals/code-bench-v1/tasks/{suite}.jsonl").read_text(encoding="utf-8").splitlines()][:limit]
    filler = torch.from_numpy(val[:P].astype(np.int64)).to(DEV)
    files = {m: (OUT / f"{m}-{suite}-samples.jsonl").open("w", encoding="utf-8") for m in ("ncm", "ar")}
    syntax = {"ncm": [], "ar": []}
    with_code = {"ncm": [], "ar": []}
    torch.manual_seed(0)
    for task in tasks:
        ids = torch.tensor(tok.encode(task["prompt"]).ids, device=DEV)
        start = torch.cat([filler, torch.tensor([eos], device=DEV), ids])  # in-distribution: a new file begins
        for model in ("ncm", "ar"):
            seq = start[None].expand(n, -1)
            t0 = time.perf_counter()
            for _ in range(chunks):
                ctx = seq[:, -P:].contiguous()
                new = ncm_chunk(ncm, codec, ctx, temperature) if model == "ncm" else ar_chunk(ar, ctx, temperature)
                seq = torch.cat([seq, new], dim=1)
            latency = (time.perf_counter() - t0) / n
            for row in seq[:, start.shape[0]:]:
                completion = cut(tok.decode(row.tolist()), suite)
                try:
                    compile(task["prompt"] + completion, "<candidate>", "exec")
                    syntax[model].append(1)
                except (SyntaxError, ValueError):
                    syntax[model].append(0)
                with_code[model].append(syntax[model][-1] * has_code(completion))
                files[model].write(json.dumps({"task_id": task["task_id"], "completion": completion,
                                               "latency_s": latency, "num_tokens": chunks * T}) + "\n")
    for f in files.values():
        f.close()
    return {m: {"syntax_valid_rate": float(np.mean(v)), "syntax_valid_with_code_rate": float(np.mean(with_code[m])),
                "samples": len(v)} for m, v in syntax.items()}


def speed(codec, ncm, ar, val):
    rng = np.random.default_rng(9)
    result = {}
    for bs in (1, 32):
        ctx = windows(val, bs, P, rng)
        for name, fn in (("ncm", lambda: ncm_chunk(ncm, codec, ctx, 0.8)), ("ar", lambda: ar_chunk(ar, ctx, 0.8))):
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 5
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            s = (time.perf_counter() - t0) / reps
            result[f"{name}_b{bs}_tokens_per_sec"] = bs * T / s
    result["ncm_speedup_b1"] = result["ncm_b1_tokens_per_sec"] / result["ar_b1_tokens_per_sec"]
    result["ncm_speedup_b32"] = result["ncm_b32_tokens_per_sec"] / result["ar_b32_tokens_per_sec"]
    return result


def main():
    import sys
    quick = "--quick" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    tok, eos, pad = data.ids()
    val = data.stream("val")
    codec, ncm, ar = load_models(tok.get_vocab_size(), pad)
    report = {"speed": speed(codec, ncm, ar, val)}
    print(json.dumps(report), flush=True)
    report["continuation"] = continuation(tok, codec, ncm, ar, val, n=32 if quick else 256)
    print(json.dumps({k: v for k, v in report["continuation"].items() if k != "examples"}), flush=True)
    report["bench"] = {"easy": bench(tok, eos, codec, ncm, ar, val, "easy", n=2 if quick else 10, limit=3 if quick else None),
                       "humaneval": bench(tok, eos, codec, ncm, ar, val, "humaneval", n=2 if quick else 5, limit=3 if quick else None)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["bench"]), flush=True)


if __name__ == "__main__":
    main()
