#!/bin/bash
# Idea 1 night (docs/ncm-next-plan.md, "Idea 1"): upstream's lookup tokenizer encoder. Needs no internet and no Claude.
# Waits while runs/code-scale-v1/HOLD holds a future time (delete it to start now). Then: codec (pre-test pick)
# -> gate (fallback to the other qualifying arm) -> NCM -> bits per level -> step-6 test -> scoring -> verdict.
# Progress: tail runs/code-scale-v1/pipeline-idea-1.log
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=runs/code-scale-v1
L=$R/logs
LOG=$R/pipeline-idea-1.log
mkdir -p $L
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CODE_SCALE_ATTN=block_diagonal
unset CODE_SCALE_CODEBOOKS CODE_SCALE_REVEAL
N=ncm-lk
E=eval-lk
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
finish() {
  note "IDEA_1_DONE"
  $PY -m code_scale.decide_1 > $L/decide-1.log 2>&1 || note "decide FAILED (see $L/decide-1.log)"
  note "verdict: $(head -1 $R/idea-1-verdict.md 2>/dev/null)"
  exit 0
}

note "START"
ARMS=$($PY -m code_scale.decide_1 pick)
note "pre-test qualifying arms (chosen first): ${ARMS:-none}"
if [ -z "$ARMS" ]; then note "STOPPED: no lookup tokenizer reached 85% tokens at step 1,000"; finish; fi
$PY -c "from code_scale.train import wait_for_hold_release as w; w()" > /dev/null
note "hold released"

C=""
for arm in $ARMS; do
  export CODE_SCALE_ENC=$arm
  D=codec-lk-$arm
  if [ ! -f $R/$D/summary.json ]; then
    if [ -d $R/$D ]; then mv $R/$D $R/$D-partial-$(date +%H%M); note "$D: moved an unfinished folder aside"; fi
    note "$D: tokenizer training, 5000 steps"
    $PY -m code_scale.train codec --steps 5000 --restart-fix --out $D > $L/$D.log 2>&1 || note "$D: training FAILED"
  fi
  if $PY -c "from code_scale.train import require_codec_quality as q; q('$R/$D/summary.json')" 2>/dev/null; then
    note "$D: gate passed"
    C=$D
    break
  fi
  note "$D: gate FAILED"
done
if [ -z "$C" ]; then note "STOPPED: no lookup tokenizer passed the reconstruction gate"; finish; fi

export CODE_SCALE_CODEC=$C CODE_SCALE_NCM=$N CODE_SCALE_EVAL=$E
if [ -d $R/$N ] && [ ! -f $R/$N/summary.json ]; then
  mv $R/$N $R/$N-partial-$(date +%H%M); note "$N: moved an unfinished folder aside"
fi
if [ ! -f $R/$N/summary.json ]; then
  note "$N: NCM training on $C, 1800 steps"
  $PY -m code_scale.train ncm --steps 1800 --out $N > $L/$N.log 2>&1 || note "$N: training FAILED (see $L/$N.log)"
fi
if ! $PY -m code_scale.check_ncm_1 $R/$N/summary.json $C $CODE_SCALE_ENC 2>/dev/null; then
  note "STOPPED: $N missing or trained with other settings"; finish
fi
note "$N: settings OK; bits per level"
$PY -m code_scale.level_bits > $L/level-bits-$N.log 2>&1 || note "$N: bits FAILED (see $L/level-bits-$N.log)"
$PY -m code_scale.decide_1 > /dev/null 2>> $L/decide-1.log
note "$E: step-6 test"
$PY -m code_scale.step6 > $L/step6-$E.log 2>&1 || note "$E: step6 FAILED (see $L/step6-$E.log)"
note "$E: scoring"
for s in easy humaneval; do
  k=1,10; [ $s = humaneval ] && k=1,5
  for m in ncm ar; do
    $PY -m zero_baseline.code_bench score --suite $s --samples $R/$E/$m-$s-samples.jsonl \
      --out $R/$E/$m-$s-score.json --k $k --workers 6 > $L/score-$E-$m-$s.log 2>&1 || note "$E: score $m $s FAILED"
  done
done
$PY -m code_scale.rescore_syntax $E > $L/rescore-$E.log 2>&1 || note "$E: rescore FAILED"
finish
