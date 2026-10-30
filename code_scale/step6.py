"""Amended step-6 test (docs/code-scale-v1-plan.md, amendment 2026-10-27).

Both models decode with the same sampling (temperature 0.8, top_p 0.95), mean of 3 seeds on the
same 256 held-out windows; greedy and repetition are reported but do not decide. Benchmark
samples (HumanEval, easy) are written for zero_baseline.code_bench scoring. Speed is measured
separately by code_scale.speed under the fixed protocol.

CODE_SCALE_LEVELS=... CODE_SCALE_CODEC=... CODE_SCALE_NCM=... CODE_SCALE_EVAL=... python -m code_scale.step6
"""
import json
import sys
import time

import numpy as np
import torch

from code_scale import data
from code_scale.diagnose import ar_gen, batched, ncm_gen, rep4, score
from code_scale.evaluate import OUT, bench, load_models
from code_scale.models import SHAPE
from code_scale.train import windows

P, T = SHAPE["prefix_len"], SHAPE["chunk"]
QUICK = "--quick" in sys.argv  # code-path check only: 32 windows, 1 seed, 3 tasks per suite
SEEDS = (0,) if QUICK else (0, 1, 2)
N = 32 if QUICK else 256


def mean_scores(rows):
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok, eos, pad = data.ids()
    val = data.stream("val")
    codec, ncm, ar = load_models(tok.get_vocab_size(), pad)
    report = {"levels": SHAPE["levels"], "started": time.strftime("%Y-%m-%d %H:%M"),
              "protocol": "t0.8 top_p0.95, mean of seeds 0-2, 256 val windows (rng 7)"}
    save = lambda: (OUT / "step6.json").write_text(json.dumps(report, indent=2))

    w = windows(val, 256, P + T, np.random.default_rng(7))[:N]  # same windows as every earlier run
    prefix, ref = w[:, :P], w[:, P:]
    with torch.no_grad():
        oracle = torch.cat([codec.decode(codec.encode(ref[i:i + 64])).argmax(-1) for i in range(0, N, 64)])
    cont = {"reference_repetition": round(float(np.mean([rep4(r.tolist()) for r in ref])), 4),
            "codec_oracle": score(tok, oracle, ref)}
    for name, fn in (("ncm", lambda c, s: ncm_gen(ncm, codec, c, s)), ("ar", lambda c, s: ar_gen(ar, c, s))):
        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            per_seed.append(score(tok, batched(lambda c, i: fn(c, "t0.8_p0.95"), prefix), ref))
        cont[f"{name}_sampled"] = {**mean_scores(per_seed), "per_seed": per_seed}
        torch.manual_seed(0)
        cont[f"{name}_greedy"] = score(tok, batched(lambda c, i: fn(c, "greedy"), prefix), ref)
        print(json.dumps({name: cont[f"{name}_sampled"]}), flush=True)
    report["continuation"] = cont
    save()

    lim = 3 if QUICK else None
    report["bench"] = {"easy": bench(tok, eos, codec, ncm, ar, val, "easy", n=2 if QUICK else 10, limit=lim),
                       "humaneval": bench(tok, eos, codec, ncm, ar, val, "humaneval", n=2 if QUICK else 5, limit=lim)}
    print(json.dumps(report["bench"]), flush=True)

    ratio = lambda a, b: round(a / b, 3) if b else None
    s_n, s_a = cont["ncm_sampled"]["char_similarity"], cont["ar_sampled"]["char_similarity"]
    he = report["bench"]["humaneval"]
    ez = report["bench"]["easy"]
    report["gate_quality"] = {
        "similarity_ratio": ratio(s_n, s_a),
        "humaneval_syntax_ratio": ratio(he["ncm"]["syntax_valid_with_code_rate"], he["ar"]["syntax_valid_with_code_rate"]),
        "easy_syntax_ratio": ratio(ez["ncm"]["syntax_valid_with_code_rate"], ez["ar"]["syntax_valid_with_code_rate"]),
        "old_humaneval_syntax_ratio_counts_blank": ratio(he["ncm"]["syntax_valid_rate"], he["ar"]["syntax_valid_rate"]),
        "old_easy_syntax_ratio_counts_blank": ratio(ez["ncm"]["syntax_valid_rate"], ez["ar"]["syntax_valid_rate"]),
        "rule": "similarity_ratio, humaneval_syntax_ratio and easy_syntax_ratio each >= 0.8; syntax ratios use "
                "syntax-valid completions that contain code (amendment 2026-10-29); speed judged from speed.json"}
    g = report["gate_quality"]
    g["pass"] = all(g[k] is not None and g[k] >= 0.8 for k in ("similarity_ratio", "humaneval_syntax_ratio", "easy_syntax_ratio"))
    report["finished"] = time.strftime("%Y-%m-%d %H:%M")
    save()
    print(json.dumps(report["gate_quality"]), flush=True)
    print("STEP6_DONE", flush=True)


if __name__ == "__main__":
    main()
