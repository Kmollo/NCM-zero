# code-scale-v1: NCM vs autoregressive on real Python (local smoke scale)

Written 2026-10-23 ~01:35 local, before any code-scale model trained.

## Question
At matched model size, data and training compute, does NCM (pinned upstream
VQVAE tokenizer + NextConceptTransformer) produce Python continuations of
similar quality to a plain GPT-style AR model while generating faster?

## Setup (fixed)
- Data: `data/code-pretrain-v1` (permissive-license Python, repo-split,
  decontaminated against HumanEval/MBPP/project tasks). Byte-level BPE, 8,192 vocab.
- Window: 192-token prefix → 64-token continuation.
- NCM tokenizer: VQVAE, 64 tokens → 5 levels [4,8,16,32,64], 2,048 codes/level.
- NCM generator: 6 layers, 384 wide, ~12.5M trainable params.
- AR control: GPT, 7 layers, 384 wide, ~15.7M params (tied embeddings).
- Both: AdamW, fp16 on the local Quadro T1000; same windows, same number of
  optimizer steps (batch 32). Measured cost per step is equal within 2%
  (gpu-smoke.json). The NCM tokenizer's own training is reported separately as
  extra NCM cost.

## Measurements
1. Tokenizer gate: on held-out val chunks, token accuracy and whole-chunk exact
   reconstruction. Target: ≥99% token accuracy. Below that, NCM output is capped
   by the tokenizer; reported as a limitation, not hidden.
2. Continuation quality on 256 held-out val windows (greedy): token match with
   the reference, plus character similarity of decoded text.
3. HumanEval and easy suite (code-bench-v1): n samples per task at temperature
   0.8: syntax-valid rate and pass@1/pass@k via the frozen scorer.
4. Generation speed at batch 1 and batch 32 (AR uses a KV cache).

## Decision rule for "NCM is worth continuing" (fixed before results)
NCM passes step 6 if all hold:
- continuation similarity ≥ 80% of AR's, and
- HumanEval/easy syntax-valid rate ≥ 80% of AR's (pass@k reported, but at this
  scale both are expected near zero, so it can't decide alone), and
- NCM generates ≥ 2× more tokens/second than AR at the same batch size.
Otherwise: report where it falls short; step 7 is not attempted.

## Limits
~12–16M params and ~12M training tokens per model on a laptop GPU. This is a
smoke-scale signal for whether a rented run is justified, not a capability claim.

## Amendments (2026-10-26)
- Tokenizer gate for NCM training (`code_scale/train.py: require_codec_quality`):
  ≥ 99% token accuracy and ≥ 50% exact 64-token chunks. The exact-chunk part
  was relaxed from 99% by the user's decision; 99% exact chunks requires about
  99.98% per-token accuracy, a near-lossless standard.
- Queued layout experiment `l7`: restart fix plus levels [4, 8, 16, 32, 48, 64, 64]
  (two full-resolution levels, ~3.1× code capacity per token vs 1.6×).
  Same steps, data and evaluation; the AR control is unchanged.

## Predictions recorded before the l7 NCM result (2026-10-26, 2:30 pm EDT)
Written before any NCM generator has trained on a passing tokenizer.
- **User:** NCM will write better code than the matched AR model (higher
  continuation similarity and syntax-valid rate).
- **Claude:** NCM will trail AR on code quality but beat it on single-sample
  generation speed.
Both are checked against the step-6 measurements above, unchanged.

## Amendment (2026-10-27, after Phase A diagnostics, before any further result)
Decoding protocol for measurement 2 (continuation quality), for all runs from now on:
- Both models decode with the **same** sampling setting: temperature 0.8, top_p 0.95,
  no top_k (the setting the benchmark already uses). Mean over 3 seeds on the same
  256 held-out windows.
- Greedy results are still reported, but no longer decide the gate.
- A repetition score (share of repeated 4-grams) is reported for both models and for
  the reference text.
- Reason: Phase A showed greedy decoding makes **both** models loop (repetition ~0.67
  for NCM and AR vs 0.07 in real code), so greedy similarity mostly measured looping.
  The upstream authors evaluate their released text models with sampling.
- Decision thresholds are unchanged (≥ 80% of AR on similarity and syntax-valid,
  ≥ 2× tokens/second).
- Speed (measurement 4) is re-measured under a fixed protocol: laptop idle and on AC
  power, both models in one session, warm-up, then the median of 3 repeats of 10
  generations at batch 1 and batch 32, with GPU clocks and temperature logged.
