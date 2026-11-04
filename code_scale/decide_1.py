"""Idea 1 rules (docs/ncm-next-plan.md, "Idea 1", fixed 2026-11-03 before results).

python -m code_scale.decide_1 pick   -> prints the qualifying pre-test arms, chosen first (empty: none qualify)
python -m code_scale.decide_1        -> writes runs/code-scale-v1/idea-1-verdict.md (first line = verdict) and .json
"""
import json
import os
import sys

from code_scale.decide import R, category, load

ARMS = ["lookup-group4", "lookup-layer"]  # order = tie-break (upstream recipe first)
# IDEA1_TAG=4 -> idea 1b (4-block tokenizer decoder): ncm-lk4, eval-lk4, idea-1b-verdict.md, pipeline-idea-1b.log
TAG = os.environ.get("IDEA1_TAG", "")
N, E, BASE_N = f"ncm-lk{TAG}", f"eval-lk{TAG}", "ncm-l7"
NAME = "idea-1b" if TAG else "idea-1"


def pretest():
    out = {}
    for arm in ARMS:
        p = R / f"pre1-{arm}" / "log.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if '"token_accuracy"' in l] if p.exists() else []
        if rows:
            out[arm] = {"step": rows[-1]["step"], "token_accuracy": rows[-1]["token_accuracy"],
                        "chunk_exact": rows[-1]["chunk_exact"]}
    return out


def pick():
    pt = pretest()
    ok = [a for a in ARMS if a in pt and pt[a]["step"] >= 999 and pt[a]["token_accuracy"] >= 0.85]
    if len(ok) == 2 and pt[ok[1]]["token_accuracy"] > pt[ok[0]]["token_accuracy"] + 0.005:
        ok.reverse()
    return ok


def bits_rule(base, new):
    need = 0.3 * base["extra_bits_vs_ar"]
    drop_0_4 = sum(base["ncm_bits_per_level"][:5]) - sum(new["ncm_bits_per_level"][:5])
    c1 = new["extra_bits_vs_ar"] <= 0.7 * base["extra_bits_vs_ar"]
    c2 = drop_0_4 >= 0.5 * need
    return c1 and c2, {"extra_base": base["extra_bits_vs_ar"], "extra_new": new["extra_bits_vs_ar"],
                       "extra_limit": round(0.7 * base["extra_bits_vs_ar"], 1), "drop_levels_0_4": round(drop_0_4, 1),
                       "drop_levels_0_4_needed": round(0.5 * need, 1), "c1_extra": c1, "c2_levels_0_4": c2}


def main():
    if sys.argv[1:] == ["pick"]:
        print(" ".join(pick()))
        return
    log = (R / f"pipeline-{NAME}.log").read_text() if (R / f"pipeline-{NAME}.log").exists() else ""
    finished = ("IDEA_1_DONE" in log or "IDEA_1B_DONE" in log)
    pt, base, new = pretest(), load(R / BASE_N / "level-bits.json"), load(R / N / "level-bits.json")
    s6, ns = load(R / E / "step6.json"), load(R / N / "summary.json")
    out = {"pretest": pt, "picked": pick()}
    lines = ["## Pre-test (1,000 steps; codec-l7 had 94.0% tokens at step 1,000; qualify >= 85%)"]
    lines += [f"- {a}: {100 * v['token_accuracy']:.1f}% tokens, {100 * v['chunk_exact']:.1f}% exact chunks (step {v['step']})"
              for a, v in pt.items()]
    bits_pass, cat = None, None
    if ns:
        cs = load(R / ns["codec"] / "summary.json") or {}
        rec = cs.get("val_reconstruction", {})
        out["codec"] = {"dir": ns["codec"], "encoder": cs.get("encoder"), **rec}
        lines += ["", f"## Tokenizer {ns['codec']} ({cs.get('encoder')})",
                  f"- Reconstruction: {100 * rec.get('token_accuracy', 0):.2f}% tokens, "
                  f"{100 * rec.get('chunk_exact', 0):.1f}% exact chunks (gate 99% / 50%; codec-l7 99.72% / 88.1%)"]
    if base and new:
        bits_pass, d = bits_rule(base, new)
        out["bits"] = {**d, "pass": bits_pass, "base_per_level": base["ncm_bits_per_level"],
                       "new_per_level": new["ncm_bits_per_level"], "ar_base": base["ar_bits_per_chunk"],
                       "ar_new": new["ar_bits_per_chunk"], "prior_base": base["prior_bits_per_level"],
                       "prior_new": new["prior_bits_per_level"]}
        f = lambda v: ", ".join(f"{x:.0f}" for x in v)
        lines += ["", "## Bits per 64-token chunk (teacher-forced, 2,048 held-out windows)",
                  f"- GPT: {new['ar_bits_per_chunk']:.1f}",
                  f"- NCM old tokenizer (ncm-l7): {base['ncm_bits_per_chunk']:.1f} (extra {d['extra_base']:.1f}); "
                  f"per level {f(base['ncm_bits_per_level'])}",
                  f"- NCM lookup tokenizer (ncm-lk): {new['ncm_bits_per_chunk']:.1f} (extra {d['extra_new']:.1f}); "
                  f"per level {f(new['ncm_bits_per_level'])}",
                  f"- Code-frequency prior per level, old: {f(base['prior_bits_per_level'])}; "
                  f"new: {f(new['prior_bits_per_level'])}",
                  f"- Rule 1, extra <= {d['extra_limit']:.1f}: {'yes' if d['c1_extra'] else 'no'}",
                  f"- Rule 2, levels 0-4 drop >= {d['drop_levels_0_4_needed']:.1f} (got {d['drop_levels_0_4']:.1f}): "
                  f"{'yes' if d['c2_levels_0_4'] else 'no'}",
                  f"- Bits rule: {'PASS' if bits_pass else 'FAIL'}"]
    if s6 and "gate_quality" in s6:
        g, c6, b = s6["gate_quality"], s6["continuation"], s6["bench"]
        cat = category(g)
        w = {f"{s}_{m}": b[s][m]["syntax_valid_with_code_rate"] for s in ("humaneval", "easy") for m in ("ncm", "ar")}
        out.update(category=cat, with_code=w,
                   gate={k: g[k] for k in ("similarity_ratio", "humaneval_syntax_ratio", "easy_syntax_ratio")})
        pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
        num = lambda x: "-" if x is None else f"{x:.2f}"
        lines += ["", f"## Quality (step-6 test): {cat}",
                  f"- Similarity: NCM {c6['ncm_sampled']['char_similarity']:.3f} vs GPT "
                  f"{c6['ar_sampled']['char_similarity']:.3f} = {num(g['similarity_ratio'])} of GPT "
                  "(old tokenizer 0.72; progress needs 0.78)",
                  f"- Valid Python with code, HumanEval: NCM {pct(w['humaneval_ncm'])} vs GPT {pct(w['humaneval_ar'])} = "
                  f"{num(g['humaneval_syntax_ratio'])} (old 0.10; progress needs 0.40)",
                  f"- Valid Python with code, easy tasks: NCM {pct(w['easy_ncm'])} vs GPT {pct(w['easy_ar'])} = "
                  f"{num(g['easy_syntax_ratio'])} (old 0.10; progress needs 0.40)"]
    if pt and not out["picked"] and not TAG:
        head = "STOP: no lookup tokenizer reached 85% tokens at step 1,000."
    elif bits_pass or cat in ("PASS", "PROGRESS"):
        head = "CONTINUE: the lookup tokenizer " + ("passed the bits rule" if bits_pass else f"reached {cat} on quality") + "."
    elif bits_pass is False and cat is not None:
        head = ("STOP: the lookup tokenizer did not pass the bits rule or reach PROGRESS; "
                "the tokenizer recipe was not the cause.")
    elif "STOPPED" in log:
        head = "STOP: " + [l for l in log.splitlines() if "STOPPED" in l][-1].split("STOPPED", 1)[1].strip(": ")
    else:
        head = "NO RESULT yet" + (" (see runs/code-scale-v1/pipeline-" + NAME + ".log)." if finished else ".")
    if not finished and head.startswith(("CONTINUE", "STOP: the lookup")):
        head = "IN PROGRESS (so far: " + head + ")"
    out["verdict"] = head
    text = [head, "", "Rule (fixed before results): CONTINUE if the bits rule passes (extra bits over GPT <= 70% of "
            "ncm-l7's, with at least half of the required drop in levels 0-4, tokenizer gate held) or step-6 quality "
            "is PROGRESS/PASS; otherwise STOP.", ""] + lines + [""]
    (R / f"{NAME}-verdict.md").write_text("\n".join(text), encoding="utf-8")
    (R / f"{NAME}-verdict.json").write_text(json.dumps(out, indent=2))
    print("\n".join(text))


if __name__ == "__main__":
    main()
