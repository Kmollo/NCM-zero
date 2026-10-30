"""Apply the Path B continue/stop rule (docs/ncm-next-plan.md, "Path B", fixed 2026-10-29 before results).

Reads whatever the Path B run has produced so far and writes runs/code-scale-v1/path-b-verdict.md
(first line = verdict) and path-b-verdict.json. Safe to run at any time.

python -m code_scale.decide
"""
import json
from pathlib import Path

R = Path("runs/code-scale-v1")
RUNS = [  # name, what changed, ncm dir, eval dir, tokenizer
    ("Run 1", "levels see the levels above them (block_causal)", "ncm-bc-l7", "eval-bc-l7", "codec-l7"),
    ("Run 2", "smaller plan codebooks (64/128/256/512)", "ncm-sc", "eval-sc", "codec-sc"),
]
# Baseline ncm-l7 (both runs change one thing vs it). Similarity: Phase A, t0.8 (0.239 / 0.331).
# Syntax with code: code_scale/rescore_syntax.py on eval-l7. Bits: ncm-l7/plan-probe.json.
BASE = {"similarity_ratio": 0.72, "humaneval_syntax_ratio": 0.097, "easy_syntax_ratio": 0.101, "bits": 1127.3}
ORDER = ["NO PROGRESS", "WEAK", "PROGRESS", "PASS"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def category(g):
    s, h, e = g["similarity_ratio"], g["humaneval_syntax_ratio"] or 0, g["easy_syntax_ratio"] or 0
    if s >= 0.8 and h >= 0.8 and e >= 0.8:
        return "PASS"
    if s >= 0.78 or h >= 0.40 or e >= 0.40:
        return "PROGRESS"
    if s < 0.75 and h < 0.25 and e < 0.25:
        return "NO PROGRESS"
    return "WEAK"


VERDICT = {
    "PASS": "CONTINUE: NCM reached the step-6 quality bar against GPT at this scale.",
    "PROGRESS": "CONTINUE: clear progress. Next: combine the changes and add memory within a level.",
    "WEAK": "UNCLEAR: small gains only. At most one more night (memory within a level), then stop if no clear progress.",
    "NO PROGRESS": "STOP recommended for the quick fixes: neither change moved quality. One bigger idea is untested "
                   "(positions within a level seeing each other); try it only as a last bet.",
}


def main():
    rows, out = [], {"runs": {}}
    for name, change, n, e, c in RUNS:
        r = {"change": change}
        cs = load(R / c / "summary.json")
        if cs and c != "codec-l7":
            v = cs["val_reconstruction"]
            r["tokenizer"] = {"token_accuracy": v["token_accuracy"], "chunk_exact": v["chunk_exact"],
                              "passes": v["token_accuracy"] >= 0.99 and v["chunk_exact"] >= 0.5}
        s6, sp, pp = load(R / e / "step6.json"), load(R / e / "speed.json"), load(R / n / "plan-probe.json")
        if s6 and "gate_quality" in s6:
            g = s6["gate_quality"]
            r["gate"] = {k: g[k] for k in ("similarity_ratio", "humaneval_syntax_ratio", "easy_syntax_ratio")}
            r["category"] = category(g)
            c6 = s6["continuation"]
            r["similarity"] = {"ncm": c6["ncm_sampled"]["char_similarity"], "gpt": c6["ar_sampled"]["char_similarity"]}
            b, rs = s6["bench"], load(R / e / "syntax-rescore.json") or {}
            r["with_code"] = {f"{s}_{m}": b[s][m].get("syntax_valid_with_code_rate", rs.get(f"{m}_{s}", {}).get("syntax_valid_with_code_rate"))
                              for s in ("humaneval", "easy") for m in ("ncm", "ar")}
            if g.get("humaneval_syntax_ratio") is None or "syntax_valid_with_code_rate" not in b["humaneval"]["ncm"]:
                g = {**g, "humaneval_syntax_ratio": rs.get("humaneval_ratio_with_code"),
                     "easy_syntax_ratio": rs.get("easy_ratio_with_code")}  # step6.json from before the amendment
                r["gate"] = {k: g[k] for k in ("similarity_ratio", "humaneval_syntax_ratio", "easy_syntax_ratio")}
                r["category"] = category(g)
        if sp:
            r["speedup_b1"], r["speedup_b32"] = sp.get("ncm_speedup_b1"), sp.get("ncm_speedup_b32")
        if pp and "ncm_total_bits_per_chunk" in pp:
            r["bits_per_chunk"] = pp["ncm_total_bits_per_chunk"]
        out["runs"][name] = r
    done = [r["category"] for r in out["runs"].values() if "category" in r]
    log = (R / "pipeline-path-b.log").read_text() if (R / "pipeline-path-b.log").exists() else ""
    finished = "PATH_B_DONE" in log
    if done:
        best = max(done, key=ORDER.index)
        out["verdict"] = best
        head = VERDICT[best] if finished or best in ("PASS", "PROGRESS") else "IN PROGRESS (so far: " + VERDICT[best] + ")"
    else:
        out["verdict"] = None
        head = "NO RESULT: no run finished its step-6 test" + (" (see runs/code-scale-v1/pipeline-path-b.log)." if finished else " yet.")

    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    num = lambda x, f="{:.2f}": "—" if x is None else f.format(x)
    lines = [head, "", "Rule (fixed before results): PASS = all three ratios >= 0.8; PROGRESS = similarity >= 0.78 "
             "or a with-code syntax ratio >= 0.40; NO PROGRESS = similarity < 0.75 and both syntax ratios < 0.25; "
             "otherwise WEAK. Baseline (ncm-l7): similarity 0.72, HumanEval 0.10, easy 0.10, 1,127 bits per chunk.", ""]
    for name, r in out["runs"].items():
        lines.append(f"## {name}: {r['change']}  ->  {r.get('category', 'not finished')}")
        if "tokenizer" in r:
            t = r["tokenizer"]
            lines.append(f"- Tokenizer: {pct(t['token_accuracy'])} tokens, {pct(t['chunk_exact'])} exact chunks "
                         f"({'passes' if t['passes'] else 'FAILS'} the 99% / 50% gate)")
        if "gate" in r:
            g, w = r["gate"], r["with_code"]
            lines += [f"- Similarity to the real code: NCM {r['similarity']['ncm']:.3f} vs GPT {r['similarity']['gpt']:.3f} "
                      f"= {num(g['similarity_ratio'])} of GPT (baseline 0.72, need 0.8)",
                      f"- Valid Python with code, HumanEval: NCM {pct(w['humaneval_ncm'])} vs GPT {pct(w['humaneval_ar'])} "
                      f"= {num(g['humaneval_syntax_ratio'])} (baseline 0.10, need 0.8)",
                      f"- Valid Python with code, easy tasks: NCM {pct(w['easy_ncm'])} vs GPT {pct(w['easy_ar'])} "
                      f"= {num(g['easy_syntax_ratio'])} (baseline 0.10, need 0.8)"]
        if "bits_per_chunk" in r:
            lines.append(f"- Bits NCM must predict per chunk: {r['bits_per_chunk']:,.0f} (baseline 1,127; lower is easier)")
        if "speedup_b1" in r:
            lines.append(f"- Speed vs GPT: {num(r['speedup_b1'])}x one at a time, {num(r['speedup_b32'])}x 32 at once"
                         + (" (Run 1 uses a simple, about 2x slower generator; speed does not decide tonight)" if name == "Run 1" else ""))
        lines.append("")
    (R / "path-b-verdict.md").write_text("\n".join(lines), encoding="utf-8")
    (R / "path-b-verdict.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
