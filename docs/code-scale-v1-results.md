# code-scale-v1 results: NCM vs autoregressive on real Python (local smoke scale)

Run 2026-10-23, 01:50–06:34 local, Quadro T1000. Plan and decision rule:
[code-scale-v1-plan.md](code-scale-v1-plan.md). Raw outputs: `runs/code-scale-v1/`.

## Outcome: NCM did not pass the step-6 gate; step 7 was not attempted

| Measurement | NCM | AR control | Gate |
|---|---:|---:|---|
| Params (trainable) | 12.5M (+10.5M tokenizer) | 15.7M | — |
| Training (1,800 steps × 32 windows) | 93 min | 93 min | matched |
| Val continuation char similarity | 0.169 | 0.282 | NCM ≥ 80% of AR: **fail (60%)** |
| Val continuation token match | 3.1% | 7.4% | — |
| HumanEval syntax-valid | 2.3% | 66.5% | NCM ≥ 80% of AR: **fail** |
| Easy-tier syntax-valid | 4.5% | 12.6% | **fail** |
| pass@1 / pass@k (HumanEval, easy) | 0 | 0 | expected at this scale |
| Generation speed, batch 1 | 244 tok/s | 38 tok/s | **6.3× faster: pass** |
| Generation speed, batch 32 | 625 tok/s | 843 tok/s | 0.74×: fail |

## Why: the quantized tokenizer is the bottleneck

- NCM tokenizer after 5,000 steps (67 min): 65.7% token accuracy, 0/1,024
  exact 64-token chunks. Target was ≥ 99%.
- Even a perfect generator is capped by it: decoding the *reference* codes
  ("codec oracle") reaches only 0.61 char similarity. NCM's generated output
  often collapses into repeated filler (e.g. lines of `#`).
- The later fixed-subset diagnostic ([codec-diagnostic-v2-results.md](codec-diagnostic-v2-results.md))
  agrees: the same encoder/decoder without quantization reconstructs 100% of
  training chunks, but quantized variants stay at 35–42% token accuracy.

## Likely structural cause (hypothesis, untested)

The released NCM tokenizers give each token much more code capacity than this
experiment did, using several near-full-resolution levels:

| Config | Tokens × bits/token | Code positions × bits/code | Code/token bit ratio |
|---|---|---|---:|
| Upstream LM1B | 128 × 14.9 | 641 × 13 | ~4.3× |
| Upstream XSum | 48 × 15.6 | 200 × 12 | ~3.2× |
| code-scale-v1 | 64 × 13 | 124 × 11 | ~1.6× |

This run had a single full-length level ([4, 8, 16, 32, 64]); upstream ends
with several (LM1B …92, 108, 128; XSum …32, 40, 48). The next codec test should
match upstream's level layout before changing the objective.

## Speed

NCM's batch-1 speed advantage is real (one sample at a time, 6.3×). At batch 32
the KV-cached AR is faster, because upstream generation with `query_level0`
cannot use a KV cache and re-runs the prefix at every level.

## Limits

One seed, ~12M training tokens per model, ~15M parameters. The AR control was
far from good (0 pass@k) but learned code structure (66.5% syntax-valid on
HumanEval completions). This is a smoke-scale signal, not a capability claim.

## Follow-up (2026-10-26): the tokenizer had a dead-code restart bug

Probing the trained tokenizer (`code_scale/codec_probe.py`) showed only 185–352
of 2,048 codes per level in use, with dead codes ~7× larger than the vectors
they quantize. Coarse levels 0–2 removed only 8% of the latent error, and 47%
stayed uncaptured after all five levels. Cause: upstream `_restart_dead_codes`
stores a restarted vector as an EMA *sum* while setting the EMA *count* to
`restart_threshold`, which inflates the code ~1/threshold. The code then never
wins and is restarted forever. Reported upstream:
https://github.com/HazyResearch/ncm/issues/16

Fix (`code_scale/models.py: fixed_restart`): `ema_w[dead] = v * threshold`.
Matched rerun (same init, batches and LR schedule, only the fix changed):

| Step | Original token accuracy | Fixed token accuracy |
|---:|---:|---:|
| 500 | 20.7% | 73.2% |
| 1,000 | 39.4% | 82.9% |
| 1,500 | 49.8% | 87.5% (2.5% exact chunks) |

The original reached only 65.7% after 5,000 steps. Earlier Zero codec runs also
used `restart_threshold=0.1` (the historical blend codec has 48/64 level-0 codes
at the threshold), so earlier "quantization hurts" conclusions should be
rechecked with the fix. The NCM step-6 result above was obtained with the
buggy tokenizer and is not a fair test of NCM.

Full 5,000-step fixed tokenizer (`codec-fix`, original layout): **95.5% token
accuracy, 17.3% exact chunks** on 1,024 held-out chunks (original: 65.7% / 0%).
Below the NCM training gate (≥ 99% tokens, ≥ 50% exact chunks), so NCM was not
retrained on it. Next: layout `l7` [4, 8, 16, 32, 48, 64, 64] with the fix.

Layout `l7` run (restart fix + [4, 8, 16, 32, 48, 64, 64]) was stopped by the user
at step ~1,000 of 5,000 to free the laptop. Partial held-out results (256 chunks):
step 500 → 83.8% tokens / 0.8% exact; step 1,000 → **94.0% / 7.4%**, versus
82.9% / 0.4% for the one-full-level fixed layout at the same step. Not resumable
from this run (no resume support); rerun with `bash code_scale/pipeline_layout.sh`
after removing its wait line, or ask Claude to restart it.

### Layout `l7` full run (2026-10-26, 20:17–21:28 local): tokenizer gate passed

Restart fix + levels [4, 8, 16, 32, 48, 64, 64], 5,000 steps, 70 min, 10.9M params.
Final held-out check on 1,024 chunks: **99.72% token accuracy, 88.1% exact
64-token chunks** (902/1,024). Gate (≥ 99% / ≥ 50%): **pass**. NCM generator
training on it started automatically.

| Step | Token accuracy | Exact chunks (256-chunk check) |
|---:|---:|---:|
| 1,000 | 94.0% | 7.4% |
| 2,000 | 98.7% | 55.9% |
| 3,000 | 99.6% | 82.4% |
| 4,000 | 99.8% | 87.1% |
| 4,500 | 99.85% | 91.8% |

| Tokenizer | Token accuracy | Exact chunks |
|---|---:|---:|
| Original layout, upstream restart (overnight run) | 65.7% | 0.0% |
| Original layout, restart fix (`codec-fix`) | 95.5% | 17.3% |
| **l7 layout, restart fix (`codec-l7`)** | **99.72%** | **88.1%** |

Informal mid-run guesses for the final exact-chunk rate: user 95% (later 94.5%),
Claude 90–94%; both were high.

### Layout `l7` NCM step-6 result (finished 2026-10-27 morning): fail

NCM generator trained 1,800 steps on `codec-l7` (final loss 3.97). Wall time was
9.7 h instead of ~2.3 h; the laptop most likely slept overnight. Same AR control,
same evaluation (`runs/code-scale-v1/eval-l7/`).

| Measurement | NCM (l7) | NCM (original run) | AR control | Gate |
|---|---:|---:|---:|---|
| Codec oracle char similarity | **0.996** | 0.61 | — | — |
| Val continuation char similarity | 0.136 | 0.169 | 0.282 | ≥ 80% of AR: **fail (48%)** |
| Val continuation token match | 3.1% | 3.1% | 7.4% | — |
| HumanEval syntax-valid | 3.4% | 2.3% | 68.7% | **fail** |
| Easy-tier syntax-valid | 4.1% | 4.5% | 13.8% | **fail** |
| pass@1 / pass@k | 0 | 0 | 0 | — |
| Speed, batch 1 (this run) | 102 tok/s | — | 98 tok/s | 1.05×: **fail** |
| Speed, batch 32 (this run) | 147 tok/s | — | 308 tok/s | 0.48×: fail |

Step 6 fails on all three conditions, so step 7 is not attempted (plan stop rule).

**What changed:** the tokenizer is no longer the limit. Decoding the reference codes
now reproduces the reference almost exactly (0.996), yet NCM's own output barely
moved. The bottleneck is now the generator.

**Failure mode:** greedy NCM output collapses into repeated fragments
(`_name_name_name…`, `LICENSE-LICENSE-…`, runs of tabs/newlines). This is the
classic failure of predicting many positions in parallel: each position picks its
own most likely code, and nothing within a level enforces a coherent sequence.

**Speed caveat:** absolute speeds differ a lot from the original run (AR 38 → 98
tok/s at batch 1; NCM 244 → 102), so machine state (clocks, throttling, after
sleep) is affecting the measurement. Within this run, NCM was not faster. NCM now
runs 7 levels instead of 5.

**Recorded predictions:** user (NCM writes better code): not supported. Claude
(NCM trails on quality, wins single-sample speed): quality part supported, speed
part not supported in this run.

Next questions (not run): does NCM's own per-level prediction accuracy collapse
at the fine levels; does sampling or iterative refinement within a level remove
the repetition; and a controlled speed re-measurement.

## Phase A diagnostics (2026-10-27): the generator cannot predict the coarse levels

Scripts: `code_scale/diagnose.py` (`runs/code-scale-v1/diag-l7/report.json`),
`code_scale/upstream_check.py` (`runs/code-scale-v1/upstream-check.json`).
Paused by the user before AR's HumanEval syntax rate under the sweep and before the
fixed-protocol speed test.

**A1, decoding.** Held-out continuations, 256 windows (reference repetition 0.071):

| Setting | NCM similarity | NCM repetition | AR similarity | AR repetition |
|---|---:|---:|---:|---:|
| greedy | 0.135 | 0.676 | 0.282 | 0.674 |
| t 0.8, top_p 0.95 | 0.239 | 0.009 | 0.331 | 0.129 |
| t 1.0, full | 0.226 | 0.002 | 0.296 | 0.026 |
| t 1.0, top_k 50, top_p 0.9 | 0.245 | 0.017 | — | — |

Greedy looping is not specific to NCM: AR loops just as much. Under the same sampling
(t 0.8) NCM reaches 72% of AR's similarity (48% under greedy). NCM HumanEval
syntax-valid: greedy 12.2%, t 0.8 2.2%, t 1.0 1.8%, top_k 50 3.3% (AR at t 0.8: 68.7%
in the step-6 run). fp32 generation gives the same results as fp16 for this model.

**A2, where it breaks.** Teacher-forced accuracy per level (true earlier levels):

| Level | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Codes | 4 | 8 | 16 | 32 | 48 | 64 | 64 |
| Accuracy | 1.6% | 4.9% | 6.7% | 20.3% | 29.9% | 74.0% | 47.3% |

Greedy generation with the true codes forced for levels 0..k:

| True levels | none | 0 | 0–1 | 0–2 | 0–3 | 0–4 | 0–5 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Similarity | 0.135 | 0.188 | 0.307 | 0.433 | 0.675 | 0.825 | 0.947 | 0.997 |
| Repetition | 0.676 | 0.376 | 0.164 | 0.058 | 0.025 | 0.037 | 0.062 | 0.070 |

Given the true coarse "plan", the generator fills in detail well, and the looping
disappears. It cannot predict the plan itself (1.6% at level 0, near chance for a
2,048-way choice).

**A3, generalization.** Training windows give the same picture (NCM 0.135, AR 0.289
greedy), so this is structural, not overfitting.

**A4, harness.** The official TinyStories NCM (88.6M generator) run through our code
path gives output identical to upstream's path (decode difference 0.0; repetition
0.45%, ~104 words per story). Under fp16 autocast that model overflows (11 of 11
levels non-finite), so fp16 generation is unsafe for upstream-scale models. The
released LM1B configs use strong token-level repetition penalties
(`frequency_penalty 11.5`, `adjacency_penalty 10,4,0.5`); TinyStories uses none.

**Diagnosis.** The coarse codes are learned only for reconstruction, with 2,048
options each, so they behave like arbitrary IDs that the generator cannot predict.
Next experiment (not started): small codebooks for the coarse levels, then
autoregressive level 0 (`l0_ar`) as the following single change.

## Path A (2026-10-29): plan loss made level 0 guessable by the gate, step 6 still fails

Overnight offline run, 22:30–05:42 (`code_scale/pipeline_path_a.sh`, plan and pass lines in
[ncm-next-plan.md](ncm-next-plan.md), "Path A plan"). Raw outputs: `runs/code-scale-v1/codec-pp05/`,
`codec-pp01/`, `ncm-pp05/`, `eval-pp05/`, `path-a-decision.json`.

**Deviation from the plan (script bug):** the pipeline passed `--plan-weight 0.$W` with W = 05 / 01,
so the two tokenizers trained with plan-loss weights **0.05 and 0.01**, not the planned 0.5 and 0.1
(confirmed in each `summary.json`). Everything else ran as fixed. The planned weights are untested.
Run names below keep the directory names (`pp05` = weight 0.05).

### Tokenizers (1,024 held-out chunks; probe on 20,480 val windows)

| | l7 (no plan loss) | pp05 (weight 0.05) | pp01 (weight 0.01) | Pass line |
|---|---:|---:|---:|---|
| Token accuracy | 99.72% | 99.77% | 99.77% | ≥ 99% |
| Exact 64-token chunks | 88.1% | 89.0% | 89.0% | ≥ 50% |
| Level-0 blind guess (bits) | 8.24 | 8.83 | 8.88 | — |
| Level-0 MLP probe gain (bits) | 0.13 | **1.05** | 0.37 | ≥ 1.0 |
| Level-0 linear probe gain | −0.00 | 0.87 | 0.21 | — |
| Level-1 / level-2 MLP gain | 1.22 / 0.94 | 1.25 / 0.97 | 1.24 / 0.96 | — |
| Level-0 codes changed by 1-token shift | 100% | 98.6% | 99.9% | target ≤ 50% |
| Level-0 codes used (of 2,048) | — | 510 | 680 | — |
| Plan guesser top-1 in batch (chance 3%) | — | 45% | 34% | — |

- Reconstruction held: both pass, slightly above l7. The plan loss did not cost exactness.
- pp05 passes both gates (level-0 gain 1.05 bits, just over 1.0); pp01 fails predictability.
  `path_a_pick` chose pp05.
- Caveat: the codes also became more varied (blind guess 8.83 vs 8.24 bits), so the probe's
  absolute uncertainty fell only from 8.11 to 7.79 bits. Levels 1–2 did not change (the loss
  only targets level 0). Shift stability barely moved (target not met).

### NCM on pp05 (1,800 steps, final loss 3.86; l7: 3.97)

Bits of help over the blind guess, NCM teacher-forced: level 0 **0.82** (l7: 0.04), level 1 1.57
(1.14), level 2 1.98 (1.77). NCM now uses most of what the probe finds at level 0.

### Amended step-6 test (t 0.8 / top_p 0.95, mean of 3 seeds, 256 windows)

| Measurement | NCM (pp05) | AR control | Ratio | Gate |
|---|---:|---:|---:|---|
| Codec oracle similarity | 0.997 | — | — | — |
| Continuation similarity | 0.236 | 0.331 | 0.715 | ≥ 0.8: **fail** |
| HumanEval syntax-valid (820 samples) | 2.9% | 68.7% | 0.043 | **fail** |
| Easy syntax-valid (660 samples) | 5.3% | 13.8% | 0.385 | **fail** |
| Repetition (reference 0.071) | 0.013 | 0.129 | — | — |
| pass@1 / pass@k | 0 | 0 | — | — |
| Speed, batch 1 (fixed protocol) | 295 tok/s | 93 tok/s | **3.18×** | ≥ 2×: pass |
| Speed, batch 32 | 430 tok/s | 909 tok/s | 0.47× | fail |

Greedy (reported only): NCM 0.130, AR 0.282. Step 6 fails; quality is unchanged from l7 under
the same sampling (l7: 0.239 vs 0.331, ratio 0.72).

**Reading.** The predictability objective works mechanically: it made level 0 about 1 bit
guessable without hurting reconstruction, and NCM picked that up (0.04 → 0.82 bits). But it did
not move generation quality. The Phase A forcing test already showed why this was likely too
small: giving NCM the *true* level 0 only lifted similarity 0.135 → 0.188; the large gains came
from levels 0–3 together (0.675). One bit of ~9 at one level is not enough. The fixed-protocol
speed test confirms NCM's single-sample speed advantage (3.2×; 3.9× on l7 in the evening check).

**Not tested yet (next options, not started):** the planned weights 0.5 / 0.1 (the evening smoke
run at 0.5 trained normally); the plan loss on levels 0–3, not just 0; the shift-agreement loss
(level 0 still changes on a 1-token shift); version 2 with its own prefix encoder.

## Path B (2026-10-29): neither quick fix moved quality; verdict STOP for the quick fixes

Plan: docs/ncm-next-plan.md, "Path B" (rule fixed before results). Two runs, each one change vs
ncm-l7, same 1800 NCM steps, same step-6 test (t 0.8 / top_p 0.95, 3 seeds, 256 windows).
Syntax ratios use the amended rule: compiles **and** contains a line of code. Raw output:
runs/code-scale-v1/path-b-verdict.md, eval-bc-l7/, eval-sc/.

| | Similarity (NCM / AR) | Ratio | HumanEval valid with code (NCM / AR) | Ratio | Easy valid with code (NCM / AR) | Ratio | Bits per chunk | Category |
|---|---|---:|---|---:|---|---:|---:|---|
| Baseline ncm-l7 | 0.239 / 0.331 | 0.72 | — | 0.10 | — | 0.10 | 1,127 | — |
| Run 1: block_causal on codec-l7 | 0.235 / 0.331 | 0.71 | 0.9% / 7.6% | 0.11 | 0.8% / 10.5% | 0.07 | 1,115 | NO PROGRESS |
| Run 2: codebooks 64/128/256/512 (codec-sc) | 0.240 / 0.331 | 0.73 | 0.0% / 7.6% | 0.00 | 1.1% / 10.5% | 0.10 | 1,008 | NO PROGRESS |

Rule: PASS = all three ratios ≥ 0.8; PROGRESS = similarity ≥ 0.78 or a with-code syntax ratio
≥ 0.40; NO PROGRESS = similarity < 0.75 and both syntax ratios < 0.25. Both runs are NO PROGRESS,
so by the plan the verdict is **STOP for the quick fixes**.

Other measurements:
- codec-sc passed the tokenizer gate: 99.65% tokens, 84.5% exact chunks (codec oracle similarity 0.996).
- Bits per code, blind guess → NCM. Run 1 levels 0–3: 8.24→8.20, 9.54→8.41, 10.83→9.07,
  10.73→7.52. Run 2 levels 0–3: 5.95→5.60, 6.96→5.88, 7.92→6.46, 8.82→6.39. Seeing the levels
  above did not make the coarse levels guessable (run 1 level 0 gain 0.04 bits, as before).
  Smaller codebooks cut ~120 bits per chunk mainly by making each coarse code cheaper, but
  NCM's gain over the blind guess stayed about the same, and quality did not move.
- Speed, batch 1 / 32 vs AR: run 1 2.39× / 0.24× (simple full-forward generator, ~2× slower;
  not judged); run 2 4.89× / 0.47×.
- Greedy similarity (reported only): run 1 0.146, run 2 0.151, AR 0.282.

**Reading.** The gap is not explained by levels being blind to the levels above them, nor by the
coarse codebooks being too large. NCM still generates valid Python with code about a tenth as
often as AR. The one structural idea left untested is letting positions within a level see each
other during generation (closest to AR); per the plan it is a last bet, not a reason to continue.

Note: the pipeline's own verdict line was written just before `PATH_B_DONE`, so it reads
"IN PROGRESS"; rerunning `python -m code_scale.decide` afterwards gives the final STOP verdict.

## Path C (2026-10-30): positions seeing each other removed 1% of the extra bits; verdict STOP

Plan: docs/ncm-next-plan.md, "Path C" plus the 00:20 amendment (both fixed before results). One
change vs ncm-l7: in every level (0–6), positions see the codes already chosen in their own level
(random reveals in training, 3 confidence-ordered passes per level at generation). Same tokenizer
codec-l7, block_diagonal attention, 1800 steps (final loss 4.00). Raw output:
runs/code-scale-v1/path-c-verdict.md, ncm-rv7/reveal-bits.json, eval-rv7/.

**Cause measurement** (2,048 val windows, teacher-forced bits per 64-token chunk):

| | GPT (AR) | NCM 1 pass (blind) | NCM 3 passes | NCM 8 passes |
|---|---:|---:|---:|---:|
| Bits per chunk | 250.5 | 1,190.7 | 1,185.5 | 1,183.9 |

Per level, 1 pass → 3 passes: 33.1→33.2, 67.9→67.0, 147.2→147.3, 245.7→245.4, 285.0→281.5,
127.7→127.3, 284.1→283.8. 3 passes remove 5.2 of the 940.2 extra bits (0.6%). Rule: the cause is
blind guessing only if 3 passes remove at least half. So the cause is **the codes themselves**:
knowing the other codes in a level barely helps predict a code, so the extra bits do not come from
guessing positions independently.

**Quality** (same step-6 test as Path B: t 0.8 / top_p 0.95, 3 seeds, 256 windows; syntax counts
only completions that compile and contain code):

| | Similarity (NCM / AR) | Ratio | HumanEval valid with code (NCM / AR) | Ratio | Easy valid with code (NCM / AR) | Ratio | Bits per chunk | Category |
|---|---|---:|---|---:|---|---:|---:|---|
| Baseline ncm-l7 | 0.239 / 0.331 | 0.72 | — | 0.10 | — | 0.10 | 1,127 | — |
| Path C: ncm-rv7 | 0.236 / 0.331 | 0.71 | 0.6% / 6.0% | 0.10 | 1.1% / 9.2% | 0.12 | 1,193 | NO PROGRESS |

Rule (last bet): CONTINUE if PASS (all three ratios ≥ 0.8) or PROGRESS (similarity ≥ 0.78 or a
with-code syntax ratio ≥ 0.40); otherwise STOP. Path C is NO PROGRESS, so the verdict is **STOP:
stop NCM for code at this scale**.

Other measurements: speed vs AR, batch 1 / 32: 0.59× / 0.08×. The decoder makes 21 model calls per
chunk with a simple full-forward pass; speed is reported but does not decide. Bits per chunk with
nothing shown is 1,193 vs the baseline's 1,127. Some model capacity went to the reveal inputs, and
the result did not improve.

**Reading.** All three structural fixes left quality where it was: seeing the levels above
(Path B, run 1), smaller coarse codebooks (Path B, run 2), and seeing the codes within a level
(Path C). The cause measurement locates the gap: NCM needs about 4.7× the bits GPT needs for the
same text, and nearly all of that excess is in the codes. Each code is hard to predict from the
prefix even when every other code in its level is known. Under the plan's rules, NCM work on code
at this scale stops here.

## Idea 1 (2026-11-03): upstream's lookup tokenizer encoder cannot reconstruct code at this size; verdict STOP

Plan and rule: docs/ncm-next-plan.md, "Idea 1" (fixed before results). Pipeline code_scale/pipeline_1.sh,
started 19:30, stopped by rule at 21:24. Verdict file runs/code-scale-v1/idea-1-verdict.md.

| Tokenizer (5,000 steps, same levels, decoder and schedule) | Tokens rebuilt | Exact chunks | Gate (99% / 50%) |
|---|---|---|---|
| codec-l7 (2 attention layers + positions, ours so far) | 99.72% | 88.1% | pass |
| codec-lk-lookup-group4 (upstream lm1b encoder) | 96.61% | 20.7% | fail |
| codec-lk-lookup-layer (lookup, per-position LayerNorm) | 95.85% | 17.7% | fail |

- Pre-test at step 1,000: 91.7% and 91.9% vs codec-l7's 94.0%, both above the 85% bar. The gap then widened:
  the lookup tokenizers flattened near 96.5% from step 3,000 on, while codec-l7 went 98.7% -> 99.7%.
- Both arms failed the gate, so no NCM was trained and the bits rule was never measured. Rule: STOP.
- Reading: with only a token lookup in front of the quantizer, the small 2-layer decoder cannot rebuild exact
  code from the codes. The attention encoder we added is what made reconstruction near-perfect, so upstream's
  text recipe does not carry over to code at this size. Untested difference: upstream lm1b also uses a larger
  decoder (4 layers, width 384, vs our 2 layers, width 256); that would be a new experiment, not part of this rule.
- Baseline measured for this rule (ncm-l7, code_scale/level_bits.py): GPT 250.5, NCM 1,124.4 bits per chunk;
  level 0 costs 32.7 bits vs 33.2 for a code-frequency guess with no context at all.

## Idea 1b (2026-11-03): upstream's 4-block decoder does not rescue the lookup tokenizer; verdict STOP

Plan: docs/ncm-next-plan.md, "Idea 1b". Pipeline code_scale/pipeline_1b.sh, 21:38 -> stopped by rule at 22:54.

| Tokenizer (5,000 steps) | Tokens rebuilt | Exact chunks | Gate |
|---|---|---|---|
| codec-l7 (attention encoder, 2-block decoder) | 99.72% | 88.1% | pass |
| codec-lk-lookup-group4 (lookup, 2-block decoder) | 96.61% | 20.7% | fail |
| codec-lk4 (lookup, 4-block decoder as lm1b) | 96.59% | 22.9% | fail |

- The 4-block decoder tracked the 2-block run almost exactly (step 1,500: 93.4% vs 93.9%; step 3,000: 96.4% vs
  96.4%). Decoder size is not the limit; the encoder is. A position that sees only its own token cannot be
  quantized into codes that rebuild exact code, at this size, whatever decoder follows.
- So at this scale there is a trade: the attention encoder rebuilds well but its codes cost ~874 extra bits; the
  lookup encoder cannot pass the gate. The tokenizer-recipe question is closed as planned.
- The whole pipeline after the tokenizer was dry-run offline beforehand (internet blocked, throwaway folders),
  so a pass would have run to the end without internet.

## Joint training (2026-11-03/04): a gentle guess push does nothing; verdict STOP at the pre-test

Plan: docs/joint-training-plan.md (fixed before results). Pipeline code_scale/pipeline_j.sh, 23:39 -> stopped by
rule at 01:35 (pre-test). 300 joint steps from codec-l7 + ncm-l7 per arm, tokenizer lr 1e-4, generator lr 3e-4.

| lambda | Tokens rebuilt | Exact chunks | Joint generator bits per chunk (held-out) |
|---|---|---|---|
| 0 (control) | 99.65% | 83.6% | 1,301.0 |
| 0.01 | 99.65% | 83.8% | 1,304.2 |
| 0.03 | 99.64% | 83.6% | 1,305.1 |
| 0.1 | 99.66% | 84.3% | 1,297.8 |

- Pick rule needed <= 95% of the control (<= 1,236 bits): none qualifies. Rule: STOP, no main run.
- The push did not even lower its own target: the training-log guess term (steps 200-299) is 7.818, 7.820,
  7.809, 7.818 nats for the four arms. At this learning rate and length the encoder barely moved, so this
  tests "a gentle push", not joint training as such.
- Code churn without any push: in 300 steps 59% of level-0 codes and 57% of level-6 codes changed vs codec-l7
  (the codebook keeps moving), and the joint generator lost ground (1,124 -> 1,301 bits) chasing them.
- Rebuild held in every arm (99.6-99.7% tokens), so joint training itself is safe for the tokenizer.

## Strong-push check (2026-11-04): the push moves its own target but not the held-out bits; verdict STOP

Plan: docs/joint-training-plan.md, "Strong-push check" (fixed before results). 08:38 -> stopped by rule at 10:04.
300 joint steps per arm from codec-l7 + ncm-l7, tokenizer lr 5e-4 in every arm.

| lambda | Tokens rebuilt | Exact chunks | Held-out bits per chunk | Training guess term (nats, steps 200-299) |
|---|---|---|---|---|
| 0 (control) | 99.10% | 65.9% | 1,345.8 | 7.769 |
| 0.3 | 99.08% | 65.6% | 1,320.5 | 7.679 |
| 1.0 | 98.33% | 43.5% | 1,420.6 | 7.411 |

- No arm qualifies (needed >= 99.5% tokens, >= 70% exact, bits <= 1,278.5). Rule: STOP joint training, and with it
  NCM for code at this scale.
- Unlike the gentle pre-test, the push now works on its own target (7.77 -> 7.68 -> 7.41 nats). But on held-out
  chunks the best arm (0.3) is only 1.9% below the control, and at 1.0 the bits get worse and the rebuild breaks
  (codes changed: 70% at level 0, 76% at level 6). The faster learning rate alone already costs rebuild quality
  (control 99.10% / 65.9% vs codec-l7's 99.72% / 88.1%).
- Reading: making the soft scores agree with the generator during training does not turn into codes a fresh look
  at held-out code finds easier; pushing harder trades away the rebuild first. At this size, the tokenizer cannot be
  made both exact and guessable by this route.
