# Critic review of the 23 proposals (read-only; two CPU checks on codec-l7, ~30 s each)

Scope: brief.md, control.md, fable_1..5.md checked against `code_scale/*.py`, `upstream/ncm/models/*.py`,
`upstream/ncm/config/eval/*.yaml` (pinned revision c702742), and `runs/code-scale-v1/*`. Two read-only CPU
scripts were run on `codec-l7/final.pt` (`../scripts/critic_precheck.py`, `critic_precheck2.py`); nothing was
trained, no GPU used, no file outside this one written.

## 0. Verdict in brief

Top 3 to test, in order:
1. **Lookup / position-free tokenizer encoder** (G3: `enc_layers=0`, `enc_positional=False`; arm A per-position
   LayerNorm as fable_4 P1, arm B GroupNorm as upstream lm1b) - **30%**. Cheapest night, strongest mechanism,
   and it *is* upstream's own text-tokenizer recipe, which we silently abandoned (contextual 2-block encoder).
2. **Entropy-constrained RVQ with a prefix+coarser-level prior** (G1, hard-argmin ECVQ, warm-started from the
   best gate-passing codec, lambda on the near-tie levels 0-2, 4, 6 first) - **25%**. Optimises the measured
   quantity; the "prior CE is a go/no-go bound" argument in control.md is wrong (see claim C12), so expect the
   generator to land above the prior.
3. **Propose-and-verify re-encoding (RDOQ)** with the existing generator (G7, fable_5 P3, extended to levels
   {0,1,2,5,6}) - **25%** for the bits test, but a hollow pass: my swap test shows level-6 and level-0..2 codes
   are nearly ignored by the decoder, so draining them lowers bits without touching the levels that decide quality.
   Run its 20-minute pre-test before anything else; it costs nothing and bounds the slack in the test.

"Return to upstream's own text recipe" deserves to be first **for the encoder** (item 1). Not the whole recipe:
12 levels / 8,192 codes / dec_layers 4 is a different experiment, and the lm1b configs use `fine_drop.dual` and
`code_corruption` keys the pinned `quant.py` does not implement (claim C4).

The pass test itself is weak: a 6-level codec (drop level 6) would plausibly cut ~280 of the 940 bits by
construction (level 6 = 284 bits for +1.1% tokens / +39 pts exact chunks on my sample). Run `[4,8,16,32,48,64]`
as a control arm, and require any "pass" to show at least half its bit reduction at levels 0-4.

## 1. Claims checked against the code

| # | Claim (who) | Verdict | Where |
|---|---|---|---|
| C1 | Residuals for every level are built from `z.detach()`; a loss on distances / soft assignments at any level gives the encoder no gradient (all agents) | **TRUE for all 7 levels, including level 0.** `z_no_grad = z.detach(); z_rest = z_no_grad.clone()`; level 0's STE `q = flat + (q - flat).detach()` passes gradient only to `learned_down_l1`'s weights, not to z. The encoder is trained only through the final STE `z_hat = z + (z_hat - z).detach()` (decoder loss) and the commitment term. | `upstream/ncm/models/quant.py:451-453, 473, 504-505, 519, 531` |
| C2 | "Level 0 already has a gradient path through `learned_down_l1`" (fable_1 P1 step 1) | **FALSE as stated.** Path reaches the conv weights only. Path A got gradient by recomputing `_down_for_quant(z, 0, 4)` on the undetached z outside the quantizer. | `code_scale/plan_loss.py:27-29`; `docs/ncm-next-plan.md:117` |
| C3 | GroupNorm statistics pool over (channels/8 x all 64 positions), so every latent depends on the whole chunk; `pre_quant_norm_type='layer'` exists as a per-position alternative (control, fable_1-5) | **TRUE.** `nn.GroupNorm(8, 256, affine=False)` applied to `z.transpose(1,2)` (B,C,L). LayerNorm option at `vqvae.py:62-64`. `build_codec` does not pass `pre_quant_norm_type`; it must be added. | `upstream/ncm/models/vqvae.py:59-67, 80-86`; `code_scale/models.py:64-78` |
| C4 | Our tokenizer deviates from upstream's released text recipes (lead) | **TRUE on the point that matters; overstated elsewhere.** Upstream `VQVAE` default `enc_layers=0` ("encoder = wte+wpe lookup", `vqvae.py:24, 41`); *none* of lm1b/tinystories/xsum/owt sets `enc_layers`, so all four released text tokenizers are lookup encoders. Ours: `enc_layers=2` bidirectional attention (`models.py:35, 68`). `enc_positional: false` only in lm1b-*.yaml (`lm1b-small.yaml:65`); tinystories/xsum/owt keep the default True. `tie_heads_to_codebook: true` in tinystories/xsum/owt (`tinystories.yaml:43`, `xsum.yaml:42`, `owt.yaml:64-66`), but lm1b uses `shared_output_head: true, shared_head_per_level_bias: true` exactly as we do (`lm1b-small.yaml:45-46`). `query_level0`: lm1b/xsum true, tinystories/owt false. `learned_l1_cascade`: false in lm1b (as ours), true elsewhere (xsum with cascade norms). EMA decay 0.99 in lm1b/xsum/owt (as ours), 0.95 only in tinystories. Codec dropout: ours 0.0, upstream 0.1-0.3 (+ `pre_quant_dropout 0.25` in xsum). lm1b's `fine_drop: {dual, w_full, w_trunc}` and `code_corruption` are **not read by the pinned `quant.py`** (only prob/min_keep/prob_start/anneal_steps, `quant.py:263-268`), so the pinned code cannot reproduce the lm1b tokenizer recipe as written. |
| C5 | `enc_attn_window`, `enc_conv_layers/kernel`, `enc_positional`, `dec_positional` are existing upstream switches (control-3, fable_1-3, fable_2-2, fable_3-2, fable_4-P1) | **TRUE.** With `enc_layers=0, enc_conv_layers=0, enc_positional=False, dropout=0` the encoder returns exactly `wte(idx)` (fable_4's claim). | `upstream/ncm/models/vae.py:45-51, 64, 115-120, 137-145, 166-179`; `vqvae.py:24-31` |
| C6 | Generator uses one 384x2048 head shared by all levels; upstream's `tie_heads_to_codebook` (distance head against the frozen codebook, per-level 384->256 proj) is unused here | **TRUE.** `_TiedCodebookHead` is the exact inverse form of `z_to_idx`'s argmin. Param count is about equal to the shared head (7 x 98k vs 787k), not zero. | `code_scale/models.py:96`; `upstream/ncm/models/next_concept.py:9-34, 216-237` |
| C7 | STE at level 0 only; upstream comment "at higher levels it diverges within ~100 steps" | **TRUE.** | `quant.py:503-505` |
| C8 | `idx_to_ncm_input` hands level k the area-pooled cumulative z_hat of levels < k; a prior built on that tensor sees what the generator sees (control-1, fable_1-1, fable_3-1, fable_5-P2) | **TRUE.** | `quant.py:671-694` |
| C9 | Per-level bits 33/68/147/246/285/128/284; levels 0-2 = 248 < 280 needed, so no coarse-only fix can pass (fable_1/2/4/5) | **TRUE.** 1-pass bits; 3-pass removes 5.2 (0.6%). | `runs/code-scale-v1/ncm-rv7/reveal-bits.json` |
| C10 | Blind-guess entropies 8.2/9.5/10.8/10.7/10.2/10.6/10.1; probe gain 0.13 at L0; shift change 100%/51%/66%; rename 20% | **TRUE.** Path A (`codec-pp05`) got the probe to 1.05 bits and shift change 98.6% (not 100%). | `ncm-l7/plan-probe.json`; `codec-pp05/plan-probe.json` |
| C11 | Path A's InfoNCE is capped at log2(32) = 5 bits (plan_batch 32) | **TRUE** in spirit. | `code_scale/train.py:102, 165-167` |
| C12 | "Prior CE <= 700 is a go/no-go since NCM has strictly more context than the prior" (control-1) | **FALSE.** The proposed prior reads the frozen 16M AR's prefix features (`ar_prefix_features`); the 13M generator reads raw `wte(prefix)` through its own 6 layers. The prior's CE is an optimistic bound, not a guarantee (fable_3's "achievable code length" has the same gap: the receiver would need the AR). | `code_scale/plan_loss.py:48-56`; `code_scale/models.py:277-286` |
| C13 | Decoder is trained with `fine_drop` prob 0.5, min_keep 1 (fable_1, fable_5) | **TRUE.** Half the steps the decoder sees levels 0..K for K uniform in 0..5. | `models.py:77`; `quant.py:457-462, 525-530` |
| C14 | Restart fix (`fixed_restart`) was on for codec-l7; threshold 0.1 | **TRUE** (`docs/code-scale-v1-results.md:115`); `restart_threshold` is one float, per-level values (fable_1 step 6) need a code change. | `models.py:40-58, 74`; `quant.py:276, 499` |
| C15 | Mixed forcing: true levels 0-3 -> 0.675, 0-4 -> 0.825, 0-5 -> 0.947 | **TRUE.** | `diag-l7/report.json` |
| C16 | Timings: codec 5,000 steps ~1.1 h (no AR read) / ~2 h (with AR read); NCM 1,800 steps ~2.2 h; bits ~0.2-0.5 h | **TRUE** (`codec-sc` 3,986 s, `codec-pp05` 7,156 s, `ncm-rv7` 7,831 s). `codec-l7` 22,913 s and `ncm-l7` 13,628 s were interrupted nights, not clean timings. | `runs/code-scale-v1/*/summary.json` |
| C17 | "Six of seven levels quantize quantization error, which is noise-like; the nearest centroid explains a small part of the residual energy" (fable_2 sec 3, fable_5 reading 1) | **HALF TRUE** (measured, below): levels 0-4 each remove 3-20% of the residual energy, but level 5 removes 55% and level 6 32% of what remains. The finest levels carry the content; the coarse levels barely reduce the residual. |
| C18 | "Nearest and second-nearest centroids are almost equidistant in 256-d" (fable_3, fable_5-P3, control-1) | **TRUE at levels 0, 1, 2, 4, 6** (median gap 3%, 7%, 11%, 17%, 8% of the nearest squared distance); **FALSE at levels 3 and 5** (38%, 42%). |
| C19 | FSQ with 4 dims per level keeps enough information (control-4, fable_2-1) | **DOUBTFUL** (measured): per-level quantizer inputs need ~50 PCA dims (level 0: 107) for 90% variance; top 4 dims hold 12-17%. A jointly learned projection can reshape z, but a 4-d bottleneck x 7 levels for an 8,192-token vocabulary is the proposal most likely to fail the gate. |
| C20 | Level-0 latent/codebook scale (new finding) | **UNCLEAR effect, real fact:** `learned_down_l1` weight norm is 45x its init; level-0 codebook vectors have mean norm 430 vs 9 at level 5; level-0 nearest distances ~2e5 vs ~25-50 elsewhere. This is the inflation `vqvae.py:59` warns about, here through the learned down conv (no norm after it; upstream xsum uses `learned_l1_cascade_norm_final`). It may be part of why level 0 is a pure fingerprint (relative gap 3%). Cheap fix candidates: cascade with final norm (upstream flag) or freezing the down conv. |

## 2. CPU measurements on codec-l7 (192-256 held-out chunks, seeds 123/7; scripts in ../scripts/)

Truncated decode (`decoder(z_hats[k])`): levels 0..0: 8.3% tokens; 0..1: 22.0%; 0..2: 35.7%; 0..3: 63.2%;
0..4: 82.5%; 0..5: **98.7% / 52.6% exact**; 0..6: 99.85% / 91.7% exact. So level 6 (284 bits, 24% of NCM's
total) buys +1.1% tokens; fable_1's own Step-0 threshold (levels 0-4 >= 95% => fine levels are nuisance) is
**not met** (82.5%): level 5 carries real content, level 6 is close to nuisance.

Energy removed per level (fraction of residual energy): 0.03, 0.04, 0.06, 0.20, 0.19, 0.55, 0.32; 17% of the
GroupNorm'd energy remains after all 7 levels.

Shift stability, all 7 levels (slot-wise changed): 1.00, 0.52, 0.66, 0.78, 0.83, 0.999, 0.99. Equivariant
comparison (code[t+1] before vs code[t] after) at the 64-position levels: level 5 **0.33**, level 6 0.82. Level 5
is already two-thirds shift-equivariant; level 6 is a hash.

Second-nearest swaps (decode with other levels' argmin codes kept, no re-encoding; a harsher setting than
ECVQ/RDOQ, which re-encode the finer levels):
- 10% of positions swapped: level 6 -> **0 tokens change** (100% chunks identical); levels 0, 1 -> 0 tokens;
  level 2 -> 0.01%; level 3 -> 0.07%; level 4 -> 0.08%; level 5 -> 0.3% (82% chunks identical).
- 100% of a level swapped: level 6 -> 0.29% tokens, 85.5% chunks identical; level 0 -> 0.1% / 94%; level 1 ->
  0.07% / 95%; level 2 -> 0.14% / 91%; level 3 -> 1.0% / 53%; level 4 -> 0.7% / 68%; level 5 -> 3.8% / 14%.
Reading: the decoder barely uses the exact code identity at levels 0-2 and 6 once level 5 is present. Those
four levels cost the generator 532 bits. Any method that re-picks codes among near-ties (G1, G7) has large
slack there; and the pass test can be met by cleaning levels that do not decide quality.

## 3. The proposals grouped (22 distinct items; control-3 has sub-variants 3a/b/c)

### G1. Rate-distortion tokenizer with a learned conditional prior (6): control-1, fable_3-1 (hard-argmin ECVQ); fable_1-1, fable_2-3, fable_5-P2 (soft-assignment gradient, rate term on all levels); fable_4-P2 (the real generator as entropy model, alternating joint fine-tune)
- Strongest version: fable_3-1/control-1's hard ECVQ (`argmin d + lambda * (-log q)`), warm-started from a
  gate-passing codec, prior = MLP on [prefix features, area-pooled z_hat_<k, level/position], lambda_k calibrated
  from the measured 1st-2nd gap per level (control's rule), ramped from 0, EMA/restart/quant_resi untouched.
  Two improvements over every written version: (i) build the prior on the *generator's own* prefix pathway
  (`codec.encoder.wte(prefix)` + a 2-block causal stack), or on cached `ncm-l7` teacher-forced logits, instead
  of frozen AR features, so the early readout is not an optimistic bound (C12); (ii) start lambda > 0 only on
  the near-tie levels 0-2, 4, 6 (C18, section 2) and keep lambda_3 = lambda_5 = 0 where switches cost
  distortion. The soft-assignment variants (fable_1/2/5) add a gradient into the encoder but also the
  "sit on cell boundaries" failure mode and more code; they are the second night, not the first.
- Fairness/cheat: codes depend on (chunk, prefix) but **decode from codes alone**, so no language modelling
  moves into the decoder; the generator pays all bits. Clean. Risks: collapse (gate + `codes_used` catch it),
  EMA codebooks dragged by rate-chosen assignments, prior memorising training windows (measure on held-out).
  fable_4-P2's co-training must be followed by a from-scratch generator for the number that counts (it says so).
- New vs tried: new. Path A was contrastive, level 0 only, no assignment change. B2 cut marginal entropy.
- One night: yes with warm start (2 x 0.9 h codec + 2.2 h NCM + 0.3 h). AR-feature caching adds 0.5 h if used.
- Pre-test (< 30 min, existing checkpoints): fit the proposed prior on `ncm-l7` codes with `plan_probe`'s
  cached features (`ar-prefix-feats-v21-t22.pt`) and report its held-out CE per level before touching the codec
  (control's "early readout", but as a 15-minute probe); and the swap test above (already done: slack exists).

### G2. Prefix-conditioned codec: encoder and/or decoder read the prefix (4): control-2, fable_1-2, fable_2-3 (ctx tokens), fable_3-4 (DPCM predictor feeding the decoder)
- Strongest version: control-2 (last 64 prefix tokens, segment embedding, mask "prefix attends to prefix only").
- Fairness: the measured bits remain a valid description length of the chunk given the prefix, *if* the gate is
  measured with the generation-time prefix only and the decoder's parameters/latency are counted as part of
  NCM. But the question changes: a 2-block full-attention decoder over [prefix | codes] is a small conditional
  LM, generation quality becomes decoder-limited, and the codes stop being a tokenizer. fable_4 and fable_5
  reject this on exactly these grounds; I agree for night 1. If run, add a leakage metric: token accuracy of
  `decode(random codes, prefix)` (how much the decoder predicts from the prefix alone) and report NCM bits +
  that number together.
- New: yes (untried; `ctx_encoder` is generator-side). One night: codec from scratch (no warm start) ~2.5 h +
  NCM 2.2 h: tight. fable_3-4 is two nights plus a day of code.
- Pre-test: none on existing checkpoints.

### G3. Local / position-free / lookup encoder (5): control-3 (3a window 8 + conv, 3b `enc_layers=0`, 3c causal window), fable_1-3, fable_2-2 (window 4), fable_3-2 (window 4 + LayerNorm), fable_4-P1 (`enc_layers=0`, no wpe, LayerNorm)
- Strongest version: fable_4-P1 = upstream's lm1b encoder with per-position norm. Codes become a fixed map of
  (token, generator's known input): `code_k[i] = argmin ||LN(wte[tok_i]) - zhat_<k,i - e||`, so levels 5-6 are
  (nearly) a product quantization of the token given the coarse plan, and the coarse levels are quantized bags of
  token embeddings. This is the only group whose argument explains why a 13M model can *compute* the map.
  Windowed-attention variants (control-3a, fable_2-2) keep a learned hash over 9-17 tokens; a smaller change
  with a weaker argument. Run lookup first; windowed attention is the fallback if the lookup fails the gate.
- Fairness/cheat: none (information only enters the decoder through codes; gate measures loss).
- New vs tried: new here (all code_scale codecs used `enc_layers=2`; the older `ncm_code/model.py` line used
  `enc_layers=1` with LayerNorm in a different experiment). The plan listed a shift-agreement *loss* and
  line-boundary chunking, never run; this is the architectural version.
- One night: the cheapest (~0.8 h codec + 2.2 h NCM + 0.2 h). Two codec arms (LayerNorm vs GroupNorm-4 as lm1b)
  fit in one night with one NCM.
- Gate risk is the main risk (~30%): a lookup + 2-block decoder must reach 99%/50%; two full-resolution levels
  of 2,048 give 22 bits per position for 13-bit tokens, so capacity is fine; the 5-level `codec-fix` (one
  full-res level, contextual encoder) got 95.5%, which shows why l7 is needed.
- Pre-test (15 min GPU, not run by me): train the lookup codec for 1,000 steps and compare token accuracy with
  `codec-l7/log.jsonl` at the same step (83.8% at 500, 94.0% at 1,000). Below 85% at step 1,000: stop.
- Also test: 7-level `plan_probe --probe-only` shift stability per level with the equivariant comparison
  (section 2 shows level 5 already at 0.33; lookup should push levels 3-6 down and level 0 well below 1.0).

### G4. Quantizer geometry (3): FSQ 4-d [8,8,8,4] (control-4, fable_2-1); non-residual Gaussian pyramid with PQ halves at levels 5-6 (fable_5-P1)
- FSQ as specified is likely to fail the gate (C19); if tried, 8+ dims per level and only after G3. It also
  needs `idx_to_ncm_input`, `Reveal`, heads and `SHAPE` plumbing: not a one-night change with margin.
  Probability of the 30% pass: ~15%.
- Gaussian pyramid: a real idea (levels stop coding quantization error), but my energy numbers say the current
  fine levels are *not* noise (C17), and with the contextual encoder kept, each level still quantizes a hash of
  the whole chunk. Touches six upstream methods. ~20%, second night at best.

### G5. Generator head (3): tied/geometry-aware head (fable_1-4; fable_5-P1 optional; control-4's factorised heads), token-factored head on lookup codes (fable_4-P3)
- fable_1-4 is upstream's `tie_heads_to_codebook` + `tie_heads_learn_scale` + `tie_heads_bias` (owt.yaml) with
  soft targets added; one flag flip, one NCM run (2.2 h), no gate risk. Alone it will not cut 280 bits (~8%),
  but it is a free A/B arm beside any night's NCM and a diagnostic of the "arbitrary indexing" hypothesis.
- fable_4-P3 is only meaningful on G3's codes; second night, clever, moderate implementation risk.

### G6. Align latents to the frozen AR (2): fable_2-4 (per-position hidden-state regression), fable_3-3 (pooled bands + syntax skeleton)
- Quality-motivated (VA-VAE / SpeechTokenizer analogue; listed as a backup in `ncm-next-plan.md:106`). Coarse
  bands hold 248 bits (C9), so fable_3-3 cannot pass alone (its author says so). fable_2-4 reshapes all 64
  positions but the author would not run it alone. ~15% each; later.

### G7. Propose-and-verify re-encoding with the trained generator, RDOQ (1): fable_5-P3
- Mechanism: among the m nearest codes, pick the one the generator likes; accept only if `decode` is unchanged;
  retrain the generator on the chosen codes (EM). No tokenizer retraining.
- My swap test says the slack is real and larger than the author assumed: level 6 and levels 0-2 are nearly
  decoder-invariant to the code choice; level 5 is not. So run it on levels {0,1,2,5,6} (level 5 with a
  distortion penalty), not 5-6 only.
- Fairness: the bits remain a valid description length (every accepted chunk decodes exactly; fallback chunks
  are scored with argmin codes). The subtle point: the code targets now depend on the generator, so the number is
  "bits of a generator-chosen code" rather than "bits of the tokenizer's code"; state it. Hollow w.r.t. quality:
  draining level 6 cannot move similarity or syntax validity (level 6 buys 1.1% tokens).
- New: yes. One night: 1,200 NCM steps at 3 forwards/step ~5 h plus 0.3 h pre-test: tight but feasible.
- Pre-test (20 min GPU, existing `codec-l7` + `ncm-l7`): one-shot bits with m=8 at levels {0,1,2,5,6}, acceptance
  rate, gate. Decision line (fix it before running): continue only if the one-shot gain >= 120 bits.

### Rejected ideas the agents agree on (and I concur)
MaskGIT/diffusion/MBR/speculative samplers (fact 5 and Path C), coarse-only fixes (C9), smaller codebooks or
dropping levels as a *fix* (information removal; but see section 4 for dropping level 6 as a *control*).

## 4. The pass test is gameable; add a control and a condition

Level 6 costs 284 of NCM's 1,191 bits and is nearly ignored by the decoder (section 2). A 6-level codec
`[4,8,16,32,48,64]` would likely land near 99.0-99.5% tokens / 55-65% exact and, if NCM's other levels stayed
where they are, at ~907 total bits = extra ~656 <= 660: a pass by deletion. The brief lists "level layouts"
as tried, but a two-full-resolution-level layout minus the last level has not been run. Recommendation:
- run `codec-l6` + NCM as a control arm (1.1 h + 2.2 h); if it passes, the bar is too low;
- amend the pass rule before results: extra bits <= 660 **and** at least 140 of the 280 removed bits come
  from levels 0-4, **and** the step-6 quality ratios are reported next to the bits.

## 5. Top 3, with calibrated probabilities of passing the 30% test as written

Base rates: four previous paths (A, B1, B2, C) all failed to move the metric that mattered; new-code one-night
runs here have had roughly a one-in-three chance of an interruption or bug (two of the clean-timing runs
above were interrupted nights). Agents' self-reported confidences (35-50% for their lead items) are inflated.

1. **G3 lookup encoder (fable_4-P1 / control-3b / upstream lm1b): 30%.** P(gate) ~0.7, P(bits pass | gate)
   ~0.45. Reason: cheapest, the only argument that explains learnability, removes an undocumented deviation
   from upstream, and every other group compounds on it. Arms: LayerNorm vs GroupNorm(4); plus the tied head as
   the generator arm if time allows. Yes, this is "return to upstream's recipe", restricted to the encoder.
2. **G1 hard ECVQ with a conditional prior on the generator's own prefix pathway: 25%.** P(implemented
   cleanly) ~0.7, P(gate) ~0.85, P(bits pass | both) ~0.4. Warm-start from the night-1 codec if it passed the
   gate, otherwise from codec-l7. Do the 15-minute prior probe first.
3. **G7 RDOQ on levels {0,1,2,5,6}: 25%** for the bits number; near 0% for quality. Do its 20-minute pre-test
   first in any case: it measures how much of the 940 bits is pure near-tie slack, which calibrates 1 and 2.

Not top-3 but recommended as free arms: tied head (`tie_heads_to_codebook`, one NCM run); 6-level control.
Not recommended for night 1: G2 (moves LM into the decoder), G4 FSQ (C19), G6 (quality play, cannot pass).

## 6. Did the Alien-inspired agents beat the control?

Materially in two places, marginally elsewhere. (1) fable_5-P3 (RDOQ, from the speculative-decoding atom 139)
is a lever nobody else proposed: no retraining, a 20-minute falsifier on existing checkpoints, and my swap test
says its premise (near-tie codes that decode identically) holds at levels 0-2 and 6. (2) fable_4-P1 pushes the
"local encoder" idea to its limit (`enc_layers=0`, no wpe, per-position norm) with the product-quantization /
"fixed map of (token, known input)" argument; the control reached the same configuration only as fallback 3b
without the argument. The "where are the bits" arithmetic (levels 0-2 = 248 < 280) appears in fable_1/2/4/5 and
not in control.md; it kills every coarse-only idea in one line. fable_1's Step-0 and fable_2's energy pre-checks
are good falsifiers (run here: they cap fable_1's own proposal, since levels 0-4 give 82.5%, not 95%). On the main
idea (ECVQ / rate-distortion tokenizer) the control is as good or better: tighter lambda calibration, warm
start, cleaner fairness language, fewer moving parts; its one error (prior CE as a bound, C12) is shared by
fable_3. The Alien set also carried more noise: 4-d FSQ (likely gate failure), prefix-conditioned decoders that
two of the Alien agents themselves reject, AR distillation that cannot pass, and a two-night DPCM. Net: the
combinations supplied one genuinely new testable lever and sharper diagnostics; they did not change the
leading idea, and the leading idea is upstream's own recipe, which no agent (control or Alien) identified as
such.
