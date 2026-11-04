# NCM next plan (written 2026-10-27, after the l7 step-6 fail)

## Where we are
- Tokenizer: solved for this test. `codec-l7`: 99.72% tokens, 88.1% exact chunks;
  decoding the reference codes gives 0.996 similarity.
- Generator: fails step 6. Output collapses into repeated fragments
  (`_name_name_name…`). Similarity 0.136 vs AR 0.282; valid Python 3.4% vs 68.7%.
- Speed: not trustworthy (machine state changed between runs).

## Facts that shape the plan
- Our continuation test decodes NCM greedily (always the top code). Upstream evaluates
  its released models with full sampling (`temperature 1.0, top_p 1.0`) and offers
  classifier-free guidance (`cfg_scale`). The benchmark used `0.8 / 0.95`.
- Upstream's smallest released generator is 12 layers × 768 wide (~100M params).
  Ours is 6 × 384 (12.9M), trained 1,800 steps.
- We have not checked our NCM harness against an official checkpoint.

## Phase A: find the cause (1 evening, no training, ~2 h GPU)
| Step | Question | How |
|---|---|---|
| A1 | Is it the decoding? | Re-evaluate `ncm-l7` with greedy, full sampling (upstream default), 0.8/0.95, and CFG 1.0. Add a repetition metric (share of output in repeated n-grams). |
| A2 | Which level breaks? | Per-level teacher-forced accuracy on held-out data, plus mixed decoding: generate levels 0..k, use the true codes after k. The level where similarity drops is the culprit. |
| A3 | Generalization or structure? | Same generation on training windows. Looping there too means a structural/decoding problem, not overfitting. |
| A4 | Is our harness right? | Run the official TinyStories checkpoint through our generation code. Coherent stories = harness OK. **Needs a download OK.** |

**Gate A:** we can name the cause: harness bug, decoding, a specific level, or model too small.

## Phase A2: is the plan predictable at all? (added 2026-10-28, before any result)
Phase A found the generator cannot predict the coarse levels (level-0 accuracy 1.6%).
Research review: accuracy is the wrong readout (64-way chance is 1.56%), and the
small-codebook experiment changes three things at once. Measure information first.

All numbers are cross-entropy in **bits per code** on ~20k held-out windows.
"Gain" = blind-guess baseline CE minus model CE. Baseline = per-level code frequencies
counted on training windows (add-one smoothed).

| Check | What | Cost |
|---|---|---|
| A2.1 | NCM teacher-forced CE per level vs baseline, for `ncm-l7` **and** the old buggy-restart run (`codec` + `ncm`) | ~25 min GPU |
| A2.2 | Probe: linear and 2-layer MLP from the AR control's prefix hidden states (last position + mean) to each level 0–2 code; train ~50k train windows, test on val | ~35 min |
| A2.3 | Code stability: rename one identifier / shift the window 1 token; share of level 0–2 codes that change | ~10 min |

Correction (2026-10-28, before running): NCM's teacher-forced CE at levels 1–2 also
sees the true earlier levels, while the probe sees only the prefix, so only level 0 is
a like-for-like comparison. The decision lines below apply to **level 0**; levels 1–2
are reported as context.

Decision lines (fixed now):
- **Path A, codes are the problem:** probe gain < 0.5 bits **and** NCM gain < 0.5 bits.
  Nothing can read the plan from the context, so the tokenizer must change.
- **Path B, generator is the problem:** probe gain exceeds NCM gain by ≥ 1 bit.
  The information is there and NCM misses it.
- **Path C, partial:** anything else (both gain ≥ 0.5 bits, within 1 bit of each other).
  NCM gets what this scale can get; test longer training first.
- Codebook size as the lever: if the old buggy run's gain per level is within 0.5 bits
  of l7's, shrinking codebooks is deprioritized.

Path A (tokenizer): add a predictability term (CRT-style: a small prefix-conditioned
predictor of coarse codes trained jointly, its loss added to the tokenizer loss), so
coarse codes become guessable; same reconstruction gate plus level-0 gain ≥ 1 bit.
Alternatives: align coarse latents to the AR model's hidden states; chunk on line
boundaries.
Path B (generator): `l0_ar` on the same tokenizer, then iterative refinement within
coarse levels, then a larger generator.
Path C: 3× steps for both NCM and AR (fair), re-measure gain; then Path A.

## Phase A2 result (2026-10-28): Path A
Level 0, bits of help over the blind guess (8.24 bits): NCM 0.043, linear probe on AR
prefix features −0.002, MLP probe 0.126. All < 0.5, so by the line above the coarse
codes carry almost no prefix-predictable information: the tokenizer must change.
(Levels 1–2, stability and old-run numbers: `runs/code-scale-v1/ncm-l7/plan-probe.json`,
`runs/code-scale-v1/ncm/plan-probe.json`.)

## Path A plan: make the plan codes guessable (scheduled for 2026-10-29 00:00)
Idea: give the tokenizer a second objective. Besides reconstructing the chunk, its
coarse latent must be predictable from the prefix (contrastive predictive coding; cf.
CRT "When Worse is Better", https://arxiv.org/abs/2412.16326).

Method
1. Train the tokenizer on prefix + chunk windows (192 + 64), not chunks alone.
2. A small plan predictor reads the prefix and predicts the chunk's level-0 latent
   (the quantizer input at level 0; exact location verified in `upstream/ncm/models/quant.py`
   before building).
3. Contrastive (InfoNCE) loss: the prediction must pick its own chunk's latent out of the
   batch. Gradients reach the encoder, so it learns to put predictable content in level 0.
   A constant latent cannot win, which prevents collapse.
4. The generator is unchanged.

Fairness: version 1 feeds the predictor the frozen AR control's prefix hidden states
(fast proof of concept; NCM then borrows from AR's training). If it works, version 2
uses its own small prefix encoder trained from scratch, and only that version counts
for the step-6 decision.

Steps and pass lines (fixed now)
| Step | What | Time | Pass line |
|---|---|---|---|
| 1 | Build predictor + loss; unit checks | 1–2 h | checks pass |
| 2 | Tokenizer, 5,000 steps, two loss weights (0.1, 0.5) | ~1.5 h each | ≥ 99% tokens, ≥ 50% exact chunks |
| 3 | `plan_probe` on the new tokenizer (probe only, no NCM) | ~15 min | level-0 MLP probe gain ≥ 1.0 bit (now 0.13) |
| 4 | NCM on the best passing tokenizer | ~2.5 h | — |
| 5 | Amended step-6 test vs AR | ~45 min | ≥ 80% of AR on similarity and syntax-valid |

Outcomes
- Predictable and exact: the reconstruction/generation tradeoff is beaten; write up, PR upstream.
- Predictable but not exact: map the tradeoff across loss weights (itself a result).
- Not predictable: (a) align coarse latents to AR features of the chunk (VA-VAE style,
  https://arxiv.org/abs/2501.01423); (b) cut chunks at line boundaries.

Watch for: collapse to few codes (log per-level code use), ~30% slower tokenizer steps.

Run details (fixed 2026-10-28 19:30, before any Path A result)
- Runs offline from 22:30 (internet off ~23:00): Windows scheduled task "NCM Path A" →
  `code_scale/keep_awake_run.py` (asks Windows not to sleep while it runs) →
  `code_scale/pipeline_path_a.sh`. Progress: `bash code_scale/progress_a.sh --watch`.
- Code: `code_scale/plan_loss.py` (predictor + InfoNCE, unit checks in `tests/test_plan_loss.py`);
  `train.py codec --plan-weight W`. Target = level-0 quantizer input
  `vq._down_for_quant(z, 0, 4)` with its gradient (upstream detaches z inside the quantizer,
  so this is a separate path into the encoder); predictor = LayerNorm → 512 MLP over the AR
  control's last + mean prefix states; learned temperature.
- Measured cost: the AR read of 64 × 192 prefixes takes 1.2 s on this GPU (2.1 s/step vs
  0.87). So the plan loss uses **32 of the 64 windows** per step (InfoNCE chance 1/32);
  reconstruction still uses all 64. ~1.6 s/step, ~2.2 h per tokenizer.
- Order: weight 0.5 (`codec-pp05`), then 0.1 (`codec-pp01`), each followed by
  `plan_probe --probe-only` (blind guess, linear + MLP probes, stability).
- Pick (`code_scale/path_a_pick.py`): pass = reconstruction gate **and** level-0 MLP probe
  gain ≥ 1.0 bit; among passing, the largest level-0 gain (ties: more exact chunks).
  None passing → stop, no NCM.
- Then NCM 1,800 steps (`ncm-pp..`), NCM bits of help (`plan_probe --gain-only`), the amended
  step-6 test (`code_scale/step6.py`: t 0.8 / top_p 0.95, seeds 0–2, 256 windows; HumanEval
  and easy samples), fixed-protocol speed (`code_scale/speed.py`), sandbox scoring.

### A2 detail results (2026-10-28) and what they add
Bits of help over the blind guess (levels 0–2 of the 7-level l7 layout [4,8,16,32,48,64,64]):

| Level | Probe (prefix only, MLP) | NCM teacher-forced |
|---|---:|---:|
| 0 (4 codes) | 0.13 | 0.04 |
| 1 (8 codes) | 1.22 | 1.14 (also sees true level 0) |
| 2 (16 codes) | 0.94 | 1.77 (also sees true levels 0–1) |

Share of codes that change under small edits (1,024 held-out chunks):

| Level | Rename one identifier | Shift window by 1 token | Unrelated chunk |
|---|---:|---:|---:|
| 0 | 19.6% | **100%** | 99.4% |
| 1 | 22.8% | 50.5% | 99.7% |
| 2 | 19.2% | 66.0% | 99.7% |

Old buggy-restart run (5 levels, ~185–352 codes in use) for comparison: level-0 blind
guess 6.90 bits, NCM gain 0.42; level 1 gain 0.81; level 2 gain 1.37. Within 0.5 bits of
l7 at levels 0–2, so by the line above, **shrinking codebooks is deprioritized**. The
buggy codes were easier only because they held less information (6.9 vs 8.2 bits to
guess at level 0), which matches the maintainer's reconstruction/generation tradeoff.

Reading: level 0 behaves like a positional fingerprint of the exact tokens, not a
concept; levels 1–2 are only ~1 bit (of ~10) predictable. Additions to Path A:
- Track **shift stability**: share of level-0 codes changed by a 1-token shift
  (now 100%). Success target: ≤ 50%. Report it for every new tokenizer.
- Optional extra loss if the contrastive term alone does not lower it: make the
  level-0 latent of a window and of the same window shifted by 1–2 tokens agree.
- The line-boundary chunking backup moves up: if the plan depends on where a chunk
  starts, starting chunks at natural boundaries may help directly.

## NCM-native track: stop judging NCM as a GPT
NCM is currently set up and tested like an LLM (64-token continuation, scored token by
token). Its own strengths are not used. Items, with what they depend on:

| Item | What | Depends on |
|---|---|---|
| N1 Prefix reuse (speed) | Compute the 192-token prefix once per chunk and reuse it for all levels (prefix states do not depend on levels); `query_level0` currently disables upstream's cache | nothing; can run any time |
| N2 Fair task test | Score both models on solving tasks (tests run in the sandbox) within the **same time budget**, so NCM's speed and many samples count | N1, a working plan |
| N3 Bigger chunks | 128- or 256-token chunks: NCM's passes stay fixed while AR's steps grow | Path A result |
| N4 Many plans | Sample several different plans, fill each in, keep what passes tests | Path A (guessable plans) |
| N5 Repair | Keep an almost-right plan, resample only detail levels | Path A |
| N6 Compressed context | Upstream `ctx_encoder`: read past code as concepts, not raw tokens, for long files | later |

Fairness rule for this track: any NCM-native test must be a real task both models
can attempt under the same budget; no test designed only so NCM wins.

## Phase B: fix it (1–3 nights, depending on A)
- Decoding: adopt the best setting, re-run the frozen evaluation. No retraining.
- Fine levels collapse: add refinement within a level (predict, re-mask the least
  confident codes, re-predict, 2–4 passes), or upstream's autoregressive level 0 (`l0_ar`).
- Model too small or undertrained: larger generator and/or more steps, with the AR
  control retrained at the matched size and steps.

**Gate B:** the unchanged step-6 rule (≥ 80% of AR on similarity and valid Python,
≥ 2× faster).

## Phase C: clean speed measurement (30 min, any time)
Laptop idle and plugged in, warm-up runs, 3 repeats, batch 1 and 32, both models in
the same session.

## Phase D: only if Gate B passes
1. Step 7: sample-and-test loop (NCM's speed spent on more samples).
2. Scaling trend locally: ~5M, 15M, 30M for both models. Does the gap shrink?
3. Rented run ($100–300): **separate user approval.**

## Stop rule
If Phase B's fixes still leave NCM below 80% of AR, write up both findings (the
tokenizer restart bug and the generator collapse) and stop the NCM-for-code bet.

## Housekeeping
- Add resume-from-checkpoint to `code_scale/train.py` (sleep/interruptions cost 7 h last night).
- Turn on keep-awake for overnight runs.
- Optional: LinkedIn post on the restart bug (needs only existing results).

## Path B (fixed 2026-10-29 ~13:00, before any Path B result)

Why: Path A made level 0 slightly guessable and nothing changed. A second-opinion review
(Fable, 2026-10-29) found NCM must predict ~1,100 bits per 64-token chunk (teacher-forced,
`ncm-l7` 1,127, `ncm-pp05` 1,104) against ~285 for the AR control, mostly at levels 2–4, and that
positions within a level are sampled independently, which matches NCM's syntax errors
(unmatched brackets, indentation, unterminated strings). It also found that the plan loss on
levels 1–3 would get no gradient (the quantizer builds those residuals from `z.detach()`), so
"push levels 0–3" was dropped.

**Measurement amendment (before results).** A blank or comment-only completion after a docstring
prompt compiles, so it counted as syntax-valid: 61% of AR's HumanEval samples have no code.
The step-6 syntax ratios now use syntax-valid completions that contain a line of code
(`code_scale/evaluate.py: has_code`). Rescored baselines (`code_scale/rescore_syntax.py`):

| Run | HumanEval with code, NCM / AR | ratio | Easy with code, NCM / AR | ratio |
|---|---|---:|---|---:|
| ncm-l7 | 0.7% / 7.6% | 0.10 | 1.1% / 10.5% | 0.10 |
| ncm-pp05 | 1.0% / 7.6% | 0.13 | 1.2% / 10.5% | 0.12 |

**Two runs, each one change against `ncm-l7`** (`code_scale/pipeline_b.sh`, start 15:00):

1. `ncm-bc-l7`: `attn_pattern: block_causal` (level k attends to levels < k), same tokenizer
   `codec-l7`, 1,800 steps. Upstream `generate()` only shows each level the prefix and itself, so
   block_causal models generate with `models.full_generate` (the training forward pass, level by
   level). Checked: on a block_diagonal model it matches upstream generate (exact in fp32 unit
   test, 98.9% of codes in fp16 on `ncm-pp05`); it is ~2x slower, so run 1's speed does not decide.
2. `codec-sc` + `ncm-sc`: codebooks 64/128/256/512 at levels 0–3, 2,048 at levels 4–6, restart
   fix, no plan loss, 5,000 steps; NCM 1,800 steps with normal attention. Same reconstruction gate
   (≥ 99% tokens, ≥ 50% exact, ≥ 512 chunks); if it fails, run 2 stops.

Each run: amended step 6 (t 0.8 / top_p 0.95, 3 seeds, 256 windows), speed, scoring, rescore,
then teacher-forced bits per chunk (reported, not deciding). Settings saved by `train.py` are
checked against this plan before evaluation.

**Continue/stop rule** (`code_scale/decide.py`, best of the finished runs):
- PASS: similarity, HumanEval-with-code and easy-with-code ratios all ≥ 0.8 → continue.
- PROGRESS: similarity ratio ≥ 0.78, or either with-code syntax ratio ≥ 0.40 → continue
  (combine the changes, then memory within a level).
- NO PROGRESS: similarity ratio < 0.75 and both with-code syntax ratios < 0.25 → stop the quick
  fixes; only the larger within-level change remains as a last bet.
- Otherwise WEAK → at most one more night.
Seed noise for reference: `ncm-pp05` similarity per seed 0.234–0.238, AR 0.320–0.341 (ratio ±~0.02).

## Path C (fixed 2026-10-29 ~23:55, before any Path C result): last bet

**Why.** Path B ruled out the two quick explanations: letting levels see the levels above them
(block_causal) and smaller coarse codebooks both scored NO PROGRESS. What remains is that NCM
samples all positions of a level at once, each blind to what its neighbours chose. NCM's errors
(unmatched brackets, broken indents) are coordination errors, and the Phase A forcing test showed
coherent coarse levels lift similarity from 0.24 to 0.68. Non-autoregressive translation and
masked-diffusion code models show the same problem and fix it with a few confidence-ordered
passes instead of one (Mask-Predict / MaskGIT style; DiffuCoder loses quality as steps drop).

**One change vs ncm-l7** (tokenizer codec-l7, block_diagonal attention, 1800 steps, same data and
seed): for levels 0-3, each position also gets the codec embedding of codes already chosen in its
own level (plus a learned shown/hidden flag). Training reveals a random fraction r ~ U(0,1) of
true codes per level (r = 0 a quarter of the time); the loss skips revealed positions. Generation
decodes each of levels 0-3 in 3 passes (keep the most confident third, show it, repeat); levels
4-6 stay one pass. 15 model calls per chunk instead of 7 (GPT: 64 steps).

**Measurement.** Exactly Path B: step-6 test (t 0.8 / top_p 0.95, 3 seeds, 256 windows), syntax
ratios with the amended valid-with-code rule, speed, bits per chunk with nothing shown.

**Rule (last bet).** CONTINUE if PASS (all three ratios >= 0.8) or PROGRESS (similarity ratio
>= 0.78, or a with-code syntax ratio >= 0.40). Anything else: STOP NCM for code at this scale.
No "one more night". Speed is reported, not decided (the decoder is a simple full forward per
pass). Code: code_scale/pipeline_c.sh, code_scale/decide_c.py; verdict in
runs/code-scale-v1/path-c-verdict.md. Runs offline.

**Amendment (2026-10-30 ~00:20, before any Path C result; the levels 0-3 run was stopped after
~25 minutes of training, no evaluation).** Levels 0-3 hold only ~485 of NCM's ~1,115 bits per
chunk; levels 4-6 hold ~630. A run that reveals only levels 0-3 could fail without telling us
why. So: reveals on **all seven levels** (run ncm-rv7, eval-rv7; 3 passes per level = 21 model
calls per chunk), and a **cause measurement** first (code_scale/reveal_bits.py, 2,048 held-out
windows): teacher-forced bits per chunk with 1 pass (blind), 3 and 8 passes (random split per
level, each group scored with the earlier groups shown), against the AR control's bits on the
same 64 tokens. The codes are computed from the text, so a perfect code model needs no more than
the text's bits. Cause rule: if 3 passes remove at least half of NCM's extra bits over AR, the
cause is blind guessing within a level; otherwise the codes themselves (tokenizer), which no
sampler can fix. The quality rule above is unchanged.

## Idea 1 (fixed 2026-11-03 ~12:55, before any idea-1 result): upstream's lookup tokenizer encoder

**Why.** Path C showed the extra bits are in the codes themselves. The Alien search (docs/alien-search/README.md)
found that every code_scale tokenizer deviates from upstream's released text tokenizers: ours encodes with
2 bidirectional attention layers plus positions; upstream's (lm1b, tinystories, xsum, owt) use a token-lookup
encoder (`enc_layers=0`), and lm1b also drops positions. One change: the encoder (`CODE_SCALE_ENC`,
code_scale/models.py). Everything else as codec-l7 / ncm-l7 (levels [4,8,16,32,48,64,64], codebook 2,048,
restart fix, codec 5,000 steps, NCM 1,800 steps, block_diagonal attention, no reveals).
Two arms: `lookup-group4` (lm1b exactly: GroupNorm(4) over the chunk) and `lookup-layer` (LayerNorm per
position, so each position's pre-quantization input is its own token alone).

**Pre-test (code_scale/pretest_1.sh, run now).** Each arm trains 1,000 steps of codec-l7's 5,000-step
schedule. codec-l7 had 94.0% token accuracy at step 1,000. An arm qualifies if its token accuracy at its
last eval (step 999, 256 chunks) is >= 85%. Tonight's arm = the qualifying arm with the higher token
accuracy (tie within 0.5 points: lookup-group4, upstream's recipe). Neither qualifies: STOP idea 1
(the lookup encoder cannot reconstruct code at this size), no night run.

**Night (code_scale/pipeline_1.sh, waits for runs/code-scale-v1/HOLD).** Codec 5,000 steps (codec-lk) ->
reconstruction gate (>= 99% tokens, >= 50% exact chunks; train.py enforces it). If the gate fails and the
other arm qualified, that arm is trained instead. Then NCM 1,800 steps (ncm-lk) -> bits per level ->
step-6 test (eval-lk) -> scoring -> verdict (code_scale/decide_1.py -> runs/code-scale-v1/idea-1-verdict.md).

**Bits measurement.** code_scale/level_bits.py: teacher-forced bits per 64-token chunk per level, 2,048
held-out windows (same seed as reveal_bits), vs the AR control's bits on the same tokens. The baseline is
**ncm-l7 measured by the same script** (measured before tonight's run; stored as
runs/code-scale-v1/ncm-l7/level-bits.json). extra = NCM bits - AR bits.

**Bits rule.** PASS only if all hold:
1. extra(ncm-lk) <= 0.7 x extra(ncm-l7) (at least 30% of the extra bits removed);
2. levels 0-4 together drop by at least half of that required 30% (so cheapening only levels 5-6, which the
   decoder mostly ignores, cannot pass);
3. the tokenizer gate held.

**Verdict.** CONTINUE if the bits rule passes (next: ideas 2/3 on top of this tokenizer) or the step-6
category is PROGRESS/PASS (same thresholds as Path C). Otherwise STOP NCM for code at this scale: the
tokenizer recipe was not the cause. Gate failure on both arms: STOP. No extra nights. Similarity ratio and
syntax ratios are reported in either case. Runs offline.

**Pre-test and baseline (2026-11-03 13:20, before the night run).** Pre-test at step 999: lookup-group4 91.7%
tokens, lookup-layer 91.9% (codec-l7: 94.0% at step 1,000). Both qualify; the difference is inside the 0.5-point
tie, so tonight's arm is lookup-group4, with lookup-layer as the gate fallback. Baseline (ncm-l7, level_bits.py):
GPT 250.5, NCM 1,124.4 bits per chunk, extra 873.9; per level 32.7, 67.2, 144.6, 238.8, 270.5, 120.5, 250.1.
So the bits rule needs extra <= 611.7 and a levels 0-4 drop >= 131.1. Night starts at 19:30 (user's choice).

## Idea 1b (fixed 2026-11-03 ~21:45, before any idea-1b result): lookup encoder plus upstream's 4-block decoder

**Why.** Idea 1 stopped on the reconstruction gate (lookup tokenizers flat near 96.5% tokens with our 2-block
decoder). Upstream lm1b pairs the lookup encoder with a 4-block decoder (`dec_layers: 4`); the user asked to
try that. One change vs codec-lk-lookup-group4: `CODE_SCALE_DEC=4` (width stays 256, so codebook vectors and
the NCM generator are unchanged; tokenizer 9.27M -> 10.85M parameters). Single arm, no fallback, no pre-test.
Names: codec-lk4, ncm-lk4, eval-lk4; code_scale/pipeline_1b.sh; verdict runs/code-scale-v1/idea-1b-verdict.md.

**Rule.** Exactly idea 1's: gate (>= 99% tokens, >= 50% exact chunks); bits rule vs the same ncm-l7 baseline
(extra <= 611.7 and a levels 0-4 drop >= 131.1); CONTINUE if the bits rule passes or step-6 quality is
PROGRESS/PASS, otherwise STOP. Gate failure: STOP, and this closes the tokenizer-recipe question at this size.
No further nights on this line.
