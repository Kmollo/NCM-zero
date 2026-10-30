#!/bin/bash
# Path C run, night of 2026-10-29 (docs/ncm-next-plan.md, "Path C"). Needs no internet and no Claude.
# One change vs ncm-l7: positions of every level (0-6) see the codes already chosen in their own level
# (random reveals in training, 3 confidence-ordered passes per level at generation; amended 00:20 from levels 0-3). Same tokenizer
# codec-l7, block_diagonal attention, 1800 steps. Then step-6 test -> speed -> scoring -> bits -> verdict.
# Progress: tail runs/code-scale-v1/pipeline-path-c.log ; tail runs/code-scale-v1/logs/ncm-rv7.log
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=runs/code-scale-v1
L=$R/logs
LOG=$R/pipeline-path-c.log
mkdir -p $L
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CODE_SCALE_CODEC=codec-l7 CODE_SCALE_ATTN=block_diagonal CODE_SCALE_REVEAL=0,1,2,3,4,5,6 CODE_SCALE_PASSES=3
unset CODE_SCALE_CODEBOOKS
N=ncm-rv7
E=eval-rv7
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }

check_meta() {
  $PY - "$@" <<'EOF'
import json, sys
d, pairs = sys.argv[1], sys.argv[2:]
s = json.load(open(f"runs/code-scale-v1/{d}/summary.json"))
bad = [f"{k}: planned {v}, got {json.dumps(s.get(k))}" for k, v in (p.split("=", 1) for p in pairs)
       if json.dumps(s.get(k), separators=(",", ":")) != v]
print("; ".join(bad) if bad else "settings OK")
sys.exit(1 if bad else 0)
EOF
}

note "START"
if [ -d $R/$N ] && [ ! -f $R/$N/summary.json ]; then
  mv $R/$N $R/$N-partial-$(date +%H%M); note "$N: moved an unfinished folder aside"
fi
if [ -f $R/$N/summary.json ]; then
  note "$N already trained, skipping training"
else
  note "$N: NCM training (reveal all levels, 3 passes, tokenizer codec-l7), 1800 steps"
  $PY -m code_scale.train ncm --steps 1800 --out $N > $L/$N.log 2>&1 || note "$N: training FAILED (see $L/$N.log)"
fi
if [ -f $R/$N/summary.json ] && M=$(check_meta $N attn_pattern='"block_diagonal"' codec='"codec-l7"' reveal="[0,1,2,3,4,5,6]" passes=3 steps=1800); then
  note "$N: $M"
  export CODE_SCALE_NCM=$N CODE_SCALE_EVAL=$E
  note "$N: cause measurement (bits with 1 / 3 / 8 passes vs GPT)"
  $PY -m code_scale.reveal_bits > $L/reveal-bits-$N.log 2>&1 || note "$N: cause measurement FAILED (see $L/reveal-bits-$N.log)"
  $PY -m code_scale.decide_c > /dev/null 2>> $L/decide-c.log
  note "$E: step-6 test"
  $PY -m code_scale.step6 > $L/step6-$E.log 2>&1 || note "$E: step6 FAILED (see $L/step6-$E.log)"
  $PY -m code_scale.decide_c > /dev/null 2>> $L/decide-c.log
  note "$E: speed test"
  $PY -m code_scale.speed > $L/speed-$E.log 2>&1 || note "$E: speed FAILED (see $L/speed-$E.log)"
  note "$E: scoring"
  for s in easy humaneval; do
    k=1,10; [ $s = humaneval ] && k=1,5
    for m in ncm ar; do
      $PY -m zero_baseline.code_bench score --suite $s --samples $R/$E/$m-$s-samples.jsonl \
        --out $R/$E/$m-$s-score.json --k $k --workers 6 > $L/score-$E-$m-$s.log 2>&1 || note "$E: score $m $s FAILED"
    done
  done
  $PY -m code_scale.rescore_syntax $E > $L/rescore-$E.log 2>&1 || note "$E: rescore FAILED"
  note "$E: bits per chunk"
  $PY -m code_scale.plan_probe --gain-only > $L/bits-$N.log 2>&1 || note "$N: bits per chunk FAILED"
else
  note "$N: STOPPED (${M:-no summary})"
fi
note "PATH_C_DONE"
$PY -m code_scale.decide_c > $L/decide-c.log 2>&1 || note "decide FAILED (see $L/decide-c.log)"
note "verdict: $(head -1 $R/path-c-verdict.md 2>/dev/null)"
