#!/bin/bash
# Joint training (docs/joint-training-plan.md). Needs no internet and no Claude; safe if the wifi goes off.
# Pre-test (4 lambdas x 300 joint steps + bits) -> pick -> joint 1,500 steps -> freeze tokenizer -> gate ->
# fresh generator 1,800 steps -> bits per level -> step-6 test -> scoring -> verdict (runs/code-scale-v1/joint-verdict.md).
# Progress: tail runs/code-scale-v1/pipeline-joint.log
# Dry run only: JT_TAG=-dry PRE_STEPS=.. JT_STEPS=.. NCM_STEPS=.. CODE_SCALE_TEST_BATCH=.. LB_ARGS=--quick S6_ARGS=--quick SCORE_ARGS=--allow-partial
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
R=runs/code-scale-v1
L=$R/logs
TAG=${JT_TAG:-}
SET=${JT_SET:-jt}
LAMS=${JT_LAMS:-0,0.01,0.03,0.1}
CLR=${CODEC_LR:-1e-4}
NAME=joint; [ "$SET" != jt ] && NAME=joint${SET#jt}
PRE=${PRE_STEPS:-300}
JS=${JT_STEPS:-1500}
NS=${NCM_STEPS:-1800}
LOG=$R/pipeline-$NAME$TAG.log
mkdir -p $L
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CODE_SCALE_ATTN=block_diagonal CODE_SCALE_ENC=attn JT_TAG=$TAG JT_SET=$SET JT_LAMS=$LAMS
unset CODE_SCALE_CODEBOOKS CODE_SCALE_REVEAL CODE_SCALE_DEC CODE_SCALE_CODEC CODE_SCALE_NCM CODE_SCALE_EVAL
C=codec-$SET$TAG
N=ncm-$SET$TAG
E=eval-$SET$TAG
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
finish() {
  note "JOINT_DONE"
  $PY -m code_scale.decide_j > $L/decide-$NAME$TAG.log 2>&1 || note "decide FAILED (see $L/decide-$NAME$TAG.log)"
  note "verdict: $(head -1 $R/$NAME-verdict$TAG.md 2>/dev/null)"
  exit 0
}

note "START"
if [ ! -f $R/joint-taus.json ]; then note "STOPPED: runs/code-scale-v1/joint-taus.json missing"; finish; fi

for lam in ${LAMS//,/ }; do
  PC=codec-$SET-pre-$lam$TAG
  PN=ncm-$SET-pre-$lam$TAG
  if [ ! -f $R/$PC/summary.json ]; then
    rm -rf $R/$PC $R/$PN
    note "pre-test lambda $lam: $PRE joint steps, tokenizer lr $CLR"
    $PY -m code_scale.train_joint --steps $PRE --lam $lam --codec-lr $CLR --out-codec $PC --out-ncm $PN > $L/$PC.log 2>&1 \
      || note "pre-test lambda $lam: FAILED (see $L/$PC.log)"
  fi
  if [ -f $R/$PC/summary.json ] && [ ! -f $R/$PN/level-bits.json ]; then
    CODE_SCALE_CODEC=$PC CODE_SCALE_NCM=$PN $PY -m code_scale.level_bits $LB_ARGS > $L/level-bits-$PN.log 2>&1 \
      || note "pre-test lambda $lam: bits FAILED (see $L/level-bits-$PN.log)"
  fi
done
LAM=$($PY -m code_scale.decide_j pick)
note "pre-test pick: lambda ${LAM:-none}"
if [ -z "$LAM" ]; then note "STOPPED: no lambda kept the rebuild and cut bits by 5% vs lambda 0"; finish; fi
if [ -n "$MAIN_WAIT_HOLD" ]; then
  note "main run waits for runs/code-scale-v1/HOLD ($(cat $R/HOLD 2>/dev/null)); delete it to start now"
  $PY -c "from code_scale.train import wait_for_hold_release as w; w()" > /dev/null
  note "hold released"
fi

if [ ! -f $R/$C/summary.json ]; then
  rm -rf $R/$C $R/ncm-$SET-joint$TAG
  note "$C: joint training, lambda $LAM, tokenizer lr $CLR, $JS steps"
  $PY -m code_scale.train_joint --steps $JS --lam $LAM --codec-lr $CLR --out-codec $C --out-ncm ncm-$SET-joint$TAG > $L/$C.log 2>&1 \
    || note "$C: joint training FAILED (see $L/$C.log)"
fi
if ! $PY -c "from code_scale.train import require_codec_quality as q; q('$R/$C/summary.json')" 2>/dev/null; then
  note "STOPPED: $C failed the reconstruction gate"; finish
fi
note "$C: gate passed"

export CODE_SCALE_CODEC=$C CODE_SCALE_NCM=$N CODE_SCALE_EVAL=$E
if [ -d $R/$N ] && [ ! -f $R/$N/summary.json ]; then mv $R/$N $R/$N-partial-$(date +%H%M); note "$N: moved an unfinished folder aside"; fi
if [ ! -f $R/$N/summary.json ]; then
  note "$N: fresh generator on $C, $NS steps"
  $PY -m code_scale.train ncm --steps $NS --out $N > $L/$N.log 2>&1 || note "$N: training FAILED (see $L/$N.log)"
fi
if ! $PY -c "
import json, sys
s = json.load(open('$R/$N/summary.json'))
ok = (s.get('codec') == '$C' and s.get('steps') == $NS and s.get('batch') == ${EXPECT_BATCH:-32} and s.get('reveal') == []
      and s.get('encoder') == 'attn' and s.get('dec_layers') == 2 and s.get('attn_pattern') == 'block_diagonal')
sys.exit(0 if ok else 1)" 2>/dev/null; then
  note "STOPPED: $N missing or trained with other settings"; finish
fi
note "$N: settings OK; bits per level"
$PY -m code_scale.level_bits $LB_ARGS > $L/level-bits-$N.log 2>&1 || note "$N: bits FAILED (see $L/level-bits-$N.log)"
$PY -m code_scale.decide_j > /dev/null 2>> $L/decide-$NAME$TAG.log
note "$E: step-6 test"
$PY -m code_scale.step6 $S6_ARGS > $L/step6-$E.log 2>&1 || note "$E: step6 FAILED (see $L/step6-$E.log)"
note "$E: scoring"
for s in easy humaneval; do
  k=1,10; [ $s = humaneval ] && k=1,5
  for m in ncm ar; do
    $PY -m zero_baseline.code_bench score --suite $s --samples $R/$E/$m-$s-samples.jsonl \
      --out $R/$E/$m-$s-score.json --k $k --workers 6 $SCORE_ARGS > $L/score-$E-$m-$s.log 2>&1 || note "$E: score $m $s FAILED"
  done
done
$PY -m code_scale.rescore_syntax $E > $L/rescore-$E.log 2>&1 || note "$E: rescore FAILED"
finish
