#!/bin/bash
# Path A overnight run (docs/ncm-next-plan.md). Needs no internet and no Claude.
# Started at 22:30 by the Windows scheduled task "NCM Path A" (via code_scale/keep_awake_run.py).
#   two tokenizers with the plan loss -> probe each -> pick -> NCM -> step-6 test -> speed -> scoring
# Progress: bash code_scale/progress_a.sh --watch
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=runs/code-scale-v1
L=$R/logs
LOG=$R/pipeline-path-a.log
mkdir -p $L
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64 PYTHONUNBUFFERED=1
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
note "START"

for PAIR in 05:0.5 01:0.1; do  # run name : plan weight (fixed 2026-10-29: "0.$W" gave 0.05 / 0.01)
  W=${PAIR%%:*}; WEIGHT=${PAIR#*:}  # note: codec-pp05/pp01 from 2026-10-29 hold weights 0.05/0.01; rename before rerunning
  C=codec-pp$W
  if [ -f $R/$C/summary.json ]; then
    note "$C already trained, skipping"
  else
    note "$C: tokenizer training (plan weight $WEIGHT), 5000 steps"
    if ! $PY -m code_scale.train codec --steps 5000 --restart-fix --plan-weight $WEIGHT --out $C > $L/$C.log 2>&1; then
      note "$C: training FAILED (see $L/$C.log)"; continue
    fi
  fi
  note "$C: plan probe"
  CODE_SCALE_CODEC=$C $PY -m code_scale.plan_probe --probe-only > $L/plan-probe-$C.log 2>&1 \
    || note "$C: plan probe FAILED (see $L/plan-probe-$C.log)"
done

CHOSEN=$($PY -m code_scale.path_a_pick codec-pp05 codec-pp01 2> $L/path-a-pick.log | tail -1)
note "decision: $CHOSEN (details: $R/path-a-decision.json)"
if [ "$CHOSEN" != "codec-pp05" ] && [ "$CHOSEN" != "codec-pp01" ]; then
  note "STOPPED: no tokenizer passed both gates; NCM not trained (plan outcome: map the tradeoff / backups)"
  note "PATH_A_DONE"; exit 0
fi

N=ncm-${CHOSEN#codec-}
E=eval-${CHOSEN#codec-}
export CODE_SCALE_CODEC=$CHOSEN CODE_SCALE_NCM=$N CODE_SCALE_EVAL=$E
note "$N: NCM training, 1800 steps"
if ! $PY -m code_scale.train ncm --steps 1800 --out $N > $L/$N.log 2>&1; then
  note "STOPPED: NCM training failed (see $L/$N.log)"; note "PATH_A_DONE"; exit 2
fi
note "$E: plan probe with NCM (bits of help)"
$PY -m code_scale.plan_probe --gain-only > $L/plan-probe-$N.log 2>&1 || note "NCM gain check FAILED"
note "$E: amended step-6 test"
$PY -m code_scale.step6 > $L/step6-$E.log 2>&1 || note "step6 FAILED (see $L/step6-$E.log)"
note "$E: speed test"
$PY -m code_scale.speed > $L/speed-$E.log 2>&1 || note "speed FAILED (see $L/speed-$E.log)"
note "$E: scoring"
for s in easy humaneval; do
  k=1,10; [ $s = humaneval ] && k=1,5
  for m in ncm ar; do
    $PY -m zero_baseline.code_bench score --suite $s --samples $R/$E/$m-$s-samples.jsonl \
      --out $R/$E/$m-$s-score.json --k $k --workers 6 > $L/score-$E-$m-$s.log 2>&1 || note "score $m $s FAILED"
  done
done
note "PATH_A_DONE"
