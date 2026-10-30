"""Rescore saved benchmark samples with the amended syntax rule (2026-10-29).

Old rule: prompt + completion compiles. A blank or comment-only completion after a docstring
prompt compiles, so it counted as valid. New rule also requires a line of code (evaluate.has_code).

python -m code_scale.rescore_syntax eval-l7 eval-pp05   -> prints and writes <eval>/syntax-rescore.json
"""
import json
import sys
from pathlib import Path

from code_scale.evaluate import has_code

RUNS = Path("runs/code-scale-v1")


def rescore(eval_dir):
    out = {}
    for suite in ("humaneval", "easy"):
        prompts = {json.loads(l)["task_id"]: json.loads(l)["prompt"] for l in
                   Path(f"evals/code-bench-v1/tasks/{suite}.jsonl").read_text(encoding="utf-8").splitlines()}
        for m in ("ncm", "ar"):
            f = RUNS / eval_dir / f"{m}-{suite}-samples.jsonl"
            if not f.exists():
                continue
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            valid = with_code = blank = 0
            for r in rows:
                ok = True
                try:
                    compile(prompts[r["task_id"]] + r["completion"], "<candidate>", "exec")
                except (SyntaxError, ValueError):
                    ok = False
                code = has_code(r["completion"])
                valid += ok
                with_code += ok and code
                blank += not code
            n = len(rows)
            out[f"{m}_{suite}"] = {"samples": n, "syntax_valid_rate": round(valid / n, 4),
                                   "syntax_valid_with_code_rate": round(with_code / n, 4),
                                   "no_code_rate": round(blank / n, 4)}
    for suite in ("humaneval", "easy"):
        a, b = out.get(f"ncm_{suite}"), out.get(f"ar_{suite}")
        if a and b:
            out[f"{suite}_ratio_with_code"] = (round(a["syntax_valid_with_code_rate"] / b["syntax_valid_with_code_rate"], 3)
                                               if b["syntax_valid_with_code_rate"] else None)
    (RUNS / eval_dir / "syntax-rescore.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d, json.dumps(rescore(d), indent=1))
