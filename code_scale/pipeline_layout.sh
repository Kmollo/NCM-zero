#!/bin/bash
# Layout experiment: restart fix + two full-resolution levels [4,8,16,32,48,64,64].
# Waits for pipeline_fix.sh to end (gate stop or done), then: tokenizer -> gate -> NCM -> evaluate -> score.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
L=runs/code-scale-v1/logs
until grep -qE "STOPPED|PIPELINE_FIX_DONE|exit=" runs/code-scale-v1-pipeline-fix.log 2>/dev/null; do sleep 60; done
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64
$PY -m code_scale.train codec --steps 5000 --restart-fix --out codec-l7 > $L/codec-l7.log 2>&1
export CODE_SCALE_CODEC=codec-l7 CODE_SCALE_NCM=ncm-l7 CODE_SCALE_EVAL=eval-l7
if ! $PY -m code_scale.train ncm --steps 1800 --out ncm-l7 > $L/ncm-l7.log 2>&1; then
  echo "STOPPED: tokenizer gate or NCM training failed (see $L/ncm-l7.log)"; exit 2
fi
$PY -m code_scale.evaluate > $L/evaluate-l7.log 2>&1
for s in easy humaneval; do
  k=1,10; [ $s = humaneval ] && k=1,5
  for m in ncm ar; do
    $PY -m zero_baseline.code_bench score --suite $s --samples runs/code-scale-v1/eval-l7/$m-$s-samples.jsonl --out runs/code-scale-v1/eval-l7/$m-$s-score.json --k $k --workers 6 > $L/score-l7-$m-$s.log 2>&1
  done
done
echo PIPELINE_L7_DONE
