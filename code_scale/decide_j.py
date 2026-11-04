"""Joint-training rules (docs/joint-training-plan.md, fixed before results).

python -m code_scale.decide_j pick  -> prints the picked lambda (empty: none qualifies)
python -m code_scale.decide_j       -> writes runs/code-scale-v1/joint-verdict.md (first line = verdict) and .json
JT_TAG="-dry" points everything at dry-run folders. JT_SET=jt2 JT_LAMS=0,0.3,1.0 selects the strong-push check
(folders codec-jt2-*, verdict joint2-verdict.md).
"""
import json
import os
import sys

from code_scale.decide import R, category, load
from code_scale.decide_1 import bits_rule

TAG = os.environ.get("JT_TAG", "")
SET = os.environ.get("JT_SET", "jt")
LAMS = os.environ.get("JT_LAMS", "0,0.01,0.03,0.1").split(",")
NAME = "joint" if SET == "jt" else "joint" + SET[2:]
N, E, C, BASE_N = f"ncm-{SET}{TAG}", f"eval-{SET}{TAG}", f"codec-{SET}{TAG}", "ncm-l7"


def pretest():
    out = {}
    for lam in LAMS:
        cs = load(R / f"codec-{SET}-pre-{lam}{TAG}" / "summary.json")
        lb = load(R / f"ncm-{SET}-pre-{lam}{TAG}" / "level-bits.json")
        if cs and lb:
            rec = cs["val_reconstruction"]
            out[lam] = {"token_accuracy": rec["token_accuracy"], "chunk_exact": rec["chunk_exact"],
                        "bits": lb["ncm_bits_per_chunk"], "per_level": lb["ncm_bits_per_level"],
                        "codes_changed": cs.get("codes_changed_vs_start")}
    return out


def pick():
    if TAG and os.environ.get("JT_DRY_PICK"):  # dry runs only: force a pick so later steps get exercised
        return os.environ["JT_DRY_PICK"]
    pt = pretest()
    if "0" not in pt:
        return None
    ok = [l for l in LAMS[1:] if l in pt and pt[l]["token_accuracy"] >= 0.995 and pt[l]["chunk_exact"] >= 0.70]
    if not ok:
        return None
    best = min(ok, key=lambda l: pt[l]["bits"])
    return best if pt[best]["bits"] <= 0.95 * pt["0"]["bits"] else None


def main():
    if sys.argv[1:] == ["pick"]:
        print(pick() or "")
        return
    log = (R / f"pipeline-{NAME}{TAG}.log").read_text() if (R / f"pipeline-{NAME}{TAG}.log").exists() else ""
    finished = "JOINT_DONE" in log
    pt, lam = pretest(), pick()
    out = {"pretest": pt, "picked_lambda": lam}
    f = lambda v: ", ".join(f"{x:.0f}" for x in v)
    lines = [f"## Pre-test (300 joint steps from codec-l7 + ncm-l7, lambdas {', '.join(LAMS)}; joint generator's bits on held-out chunks)"]
    for l, v in pt.items():
        lines.append(f"- lambda {l}: {100 * v['token_accuracy']:.2f}% tokens, {100 * v['chunk_exact']:.1f}% exact chunks, "
                     f"{v['bits']:.1f} bits per chunk (per level {f(v['per_level'])}); codes changed per level "
                     f"{v['codes_changed']}")
    lines.append(f"- Pick (>= 99.5% tokens, >= 70% exact, bits <= 95% of lambda 0): {lam or 'none qualifies'}")
    bits_pass, cat = None, None
    cs = load(R / C / "summary.json")
    if cs:
        rec = cs["val_reconstruction"]
        out["codec"] = {"lam": cs.get("lam"), **rec, "codes_changed": cs.get("codes_changed_vs_start")}
        lines += ["", f"## Tokenizer {C} (1,500 joint steps, lambda {cs.get('lam')})",
                  f"- Reconstruction: {100 * rec['token_accuracy']:.2f}% tokens, {100 * rec['chunk_exact']:.1f}% exact "
                  "chunks (gate 99% / 50%; codec-l7 99.72% / 88.1%)",
                  f"- Codes changed vs codec-l7, per level: {cs.get('codes_changed_vs_start')}"]
    base, new = load(R / BASE_N / "level-bits.json"), load(R / N / "level-bits.json")
    if base and new:
        bits_pass, d = bits_rule(base, new)
        out["bits"] = {**d, "pass": bits_pass, "base_per_level": base["ncm_bits_per_level"],
                       "new_per_level": new["ncm_bits_per_level"]}
        lines += ["", "## Bits per 64-token chunk (fresh generator, 1,800 steps, teacher-forced, 2,048 held-out windows)",
                  f"- GPT: {new['ar_bits_per_chunk']:.1f}",
                  f"- NCM old tokenizer (ncm-l7): {base['ncm_bits_per_chunk']:.1f} (extra {d['extra_base']:.1f}); "
                  f"per level {f(base['ncm_bits_per_level'])}",
                  f"- NCM joint tokenizer ({N}): {new['ncm_bits_per_chunk']:.1f} (extra {d['extra_new']:.1f}); "
                  f"per level {f(new['ncm_bits_per_level'])}",
                  f"- Rule 1, extra <= {d['extra_limit']:.1f}: {'yes' if d['c1_extra'] else 'no'}",
                  f"- Rule 2, levels 0-4 drop >= {d['drop_levels_0_4_needed']:.1f} (got {d['drop_levels_0_4']:.1f}): "
                  f"{'yes' if d['c2_levels_0_4'] else 'no'}",
                  f"- Bits rule: {'PASS' if bits_pass else 'FAIL'}"]
    s6 = load(R / E / "step6.json")
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
                  f"{c6['ar_sampled']['char_similarity']:.3f} = {num(g['similarity_ratio'])} of GPT (old 0.72; progress 0.78)",
                  f"- Valid Python with code, HumanEval: NCM {pct(w['humaneval_ncm'])} vs GPT {pct(w['humaneval_ar'])} = "
                  f"{num(g['humaneval_syntax_ratio'])} (old 0.10; progress 0.40)",
                  f"- Valid Python with code, easy tasks: NCM {pct(w['easy_ncm'])} vs GPT {pct(w['easy_ar'])} = "
                  f"{num(g['easy_syntax_ratio'])} (old 0.10; progress 0.40)"]
    if bits_pass or cat in ("PASS", "PROGRESS"):
        head = "CONTINUE: joint training " + ("passed the bits rule" if bits_pass else f"reached {cat} on quality") + "."
    elif bits_pass is False and cat is not None:
        head = "STOP: joint training did not pass the bits rule or reach PROGRESS."
    elif "STOPPED" in log:
        head = "STOP: " + [l for l in log.splitlines() if "STOPPED" in l][-1].split("STOPPED", 1)[1].strip(": ")
    else:
        head = "NO RESULT yet."
    if not finished and head.startswith(("CONTINUE", "STOP: joint")):
        head = "IN PROGRESS (so far: " + head + ")"
    out["verdict"] = head
    text = [head, "", "Rule (fixed before results): CONTINUE if the bits rule passes (extra bits over GPT <= 611.7, "
            "levels 0-4 drop >= 131.1, tokenizer gate held) or step-6 quality is PROGRESS/PASS; otherwise STOP.", ""]
    (R / f"{NAME}-verdict{TAG}.md").write_text("\n".join(text + lines + [""]), encoding="utf-8")
    (R / f"{NAME}-verdict{TAG}.json").write_text(json.dumps(out, indent=2))
    print("\n".join(text + lines))


if __name__ == "__main__":
    main()
