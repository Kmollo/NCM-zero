#!/bin/bash
# Idea 1 pre-test (docs/alien-search/README.md): 1,000 steps of each lookup tokenizer on codec-l7's 5,000-step schedule.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
L=runs/code-scale-v1/logs
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CODE_SCALE_CODEBOOKS
for arm in lookup-group4 lookup-layer; do
  CODE_SCALE_ENC=$arm $PY -m code_scale.train codec --steps 5000 --stop-at 1000 --restart-fix --out pre1-$arm > $L/pre1-$arm.log 2>&1
done
echo PRETEST_1_DONE >> $L/pre1-done.txt
