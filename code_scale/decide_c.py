"""Apply the Path C continue/stop rule (docs/ncm-next-plan.md, "Path C", fixed 2026-10-29 before results).

Path C is the last bet: CONTINUE only on PASS or PROGRESS (same thresholds as Path B); anything else is STOP.
Writes runs/code-scale-v1/path-c-verdict.md (first line = verdict) and path-c-verdict.json.

python -m code_scale.decide_c
"""
import json

from code_scale.decide import BASE, R, category, load

N, E = "ncm-rv7", "eval-rv7"
VERDICT = {
    "PASS": "CONTINUE: NCM reached the step-6 quality bar against GPT at this scale.",
    "PROGRESS": "CONTINUE: letting positions within a level see each other gave clear progress.",
    "OTHER": "STOP: the last bet did not reach the progress bar; stop NCM for code at this scale.",
}


def main():
    s6, sp, pp = load(R / E / "step6.json"), load(R / E / "speed.json"), load(R / N / "plan-probe.json")
    log = (R / "pipeline-path-c.log").read_text() if (R / "pipeline-path-c.log").exists() else ""
    finished = "PATH_C_DONE" in log
    out = {"run": N, "change": "positions in every level see codes already chosen in their level; 3 passes per level"}
    lines = []
    rb = load(R / N / "reveal-bits.json")
    if rb:
        b = rb["ncm_bits_per_chunk"]
        out["cause"] = {k: rb[k] for k in ("ar_bits_per_chunk", "ncm_bits_per_chunk", "extra_bits_1pass_vs_ar",
                                           "removed_by_3_passes", "removed_share", "cause")}
        share = rb["removed_share"]
        lines += [f"- Cause measurement: GPT needs {rb['ar_bits_per_chunk']:,.0f} bits per chunk; NCM needs "
                  f"{b['1_pass']:,.0f} guessing blind, {b['3_pass']:,.0f} with 3 passes, {b['8_pass']:,.0f} with 8 passes",
                  f"- 3 passes remove {rb['removed_by_3_passes']:,.0f} of the {rb['extra_bits_1pass_vs_ar']:,.0f} extra bits"
                  + (f" ({100 * share:.0f}%)" if share is not None else "")
                  + f" -> cause: {rb['cause']} (rule: at least half means blind guessing)"]
    if s6 and "gate_quality" in s6:
        g, c6, b = s6["gate_quality"], s6["continuation"], s6["bench"]
        cat = category(g)
        key = cat if cat in ("PASS", "PROGRESS") else "OTHER"
        out.update(category=cat, verdict="CONTINUE" if key != "OTHER" else "STOP",
                   gate={k: g[k] for k in ("similarity_ratio", "humaneval_syntax_ratio", "easy_syntax_ratio")})
        w = {f"{s}_{m}": b[s][m]["syntax_valid_with_code_rate"] for s in ("humaneval", "easy") for m in ("ncm", "ar")}
        head = VERDICT[key]
        pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
        num = lambda x: "—" if x is None else f"{x:.2f}"
        lines += [f"- Category: {cat}",
                 f"- Similarity to the real code: NCM {c6['ncm_sampled']['char_similarity']:.3f} vs GPT "
                 f"{c6['ar_sampled']['char_similarity']:.3f} = {num(g['similarity_ratio'])} of GPT (baseline 0.72; continue needs 0.78)",
                 f"- Valid Python with code, HumanEval: NCM {pct(w['humaneval_ncm'])} vs GPT {pct(w['humaneval_ar'])} "
                 f"= {num(g['humaneval_syntax_ratio'])} (baseline 0.10; continue needs 0.40)",
                 f"- Valid Python with code, easy tasks: NCM {pct(w['easy_ncm'])} vs GPT {pct(w['easy_ar'])} "
                 f"= {num(g['easy_syntax_ratio'])} (baseline 0.10; continue needs 0.40)"]
        out["with_code"] = w
    else:
        out["verdict"] = None
        head = "NO RESULT: the step-6 test did not finish" + (" (see runs/code-scale-v1/pipeline-path-c.log)." if finished else " yet.")
    if pp and "ncm_total_bits_per_chunk" in pp:
        out["bits_per_chunk_nothing_shown"] = pp["ncm_total_bits_per_chunk"]
        lines.append(f"- Bits per chunk with nothing shown (first pass): {pp['ncm_total_bits_per_chunk']:,.0f} "
                     f"(baseline {BASE['bits']:,.0f})")
    if sp:
        out["speedup_b1"], out["speedup_b32"] = sp.get("ncm_speedup_b1"), sp.get("ncm_speedup_b32")
        lines.append(f"- Speed vs GPT: {sp.get('ncm_speedup_b1')}x one at a time, {sp.get('ncm_speedup_b32')}x 32 at once "
                     "(21 model calls per chunk with a simple full-forward decoder; reported, does not decide)")
    if not finished and out.get("verdict"):
        head = "IN PROGRESS (so far: " + head + ")"
    text = [head, "", "Rule (fixed before results, last bet): CONTINUE if PASS (all three ratios >= 0.8) or PROGRESS "
            "(similarity >= 0.78 or a with-code syntax ratio >= 0.40); otherwise STOP.", "",
            "## Path C: positions within every level see each other (3 passes per level)"] + lines + [""]
    (R / "path-c-verdict.md").write_text("\n".join(text), encoding="utf-8")
    (R / "path-c-verdict.json").write_text(json.dumps(out, indent=2))
    print("\n".join(text))


if __name__ == "__main__":
    main()
