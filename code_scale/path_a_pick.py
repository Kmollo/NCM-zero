"""Path A gate (docs/ncm-next-plan.md, pass lines fixed before results).

A new tokenizer passes if it reconstructs (>= 99% tokens, >= 50% exact chunks on 1,024 held-out
chunks) AND its level-0 codes are guessable (MLP probe gain >= 1.0 bit over the blind guess).
Among passing tokenizers the one with the largest level-0 probe gain is used for NCM.
Prints the chosen run name, or NONE. Writes runs/code-scale-v1/path-a-decision.json.

python -m code_scale.path_a_pick codec-pp05 codec-pp01
"""
import json
import sys

from code_scale.train import RUNS


def check(name):
    row = {"codec": name}
    try:
        rec = json.loads((RUNS / name / "summary.json").read_text())["val_reconstruction"]
        probe = json.loads((RUNS / name / "plan-probe.json").read_text())
        l0 = next(r for r in probe["A2_2_probe_bits_per_code"] if r["level"] == 0)
    except (FileNotFoundError, KeyError, StopIteration, json.JSONDecodeError) as e:
        return {**row, "pass": False, "error": f"missing result: {e!r}"}
    stab = {r["level"]: r for r in probe.get("A2_3_stability", [])}
    row.update(token_accuracy=rec["token_accuracy"], chunk_exact=rec["chunk_exact"],
               level0_probe_mlp_gain=l0["probe_mlp_gain"], level0_probe_linear_gain=l0["probe_linear_gain"],
               level0_shift_changed=stab.get(0, {}).get("changed_by_one_token_shift"),
               codes_used_l0_l2=rec.get("codes_used_l0_l2"))
    row["reconstruction_pass"] = rec["token_accuracy"] >= 0.99 and rec["chunk_exact"] >= 0.50 and rec["chunks"] >= 512
    row["predictability_pass"] = l0["probe_mlp_gain"] >= 1.0
    row["pass"] = row["reconstruction_pass"] and row["predictability_pass"]
    return row


def main():
    rows = [check(n) for n in sys.argv[1:]]
    passing = sorted((r for r in rows if r["pass"]),
                     key=lambda r: (r["level0_probe_mlp_gain"], r["chunk_exact"]), reverse=True)
    chosen = passing[0]["codec"] if passing else "NONE"
    (RUNS / "path-a-decision.json").write_text(json.dumps({"candidates": rows, "chosen": chosen}, indent=2))
    print(chosen)


if __name__ == "__main__":
    main()
