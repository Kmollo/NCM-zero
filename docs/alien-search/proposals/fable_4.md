# Proposals from combos_4 (Fable, batch 4)

## Reading of the facts that drives every proposal

The chain rule settles what the 940 extra bits are. The codes are a deterministic function of the
chunk, so H(all 236 codes | prefix) <= H(chunk | prefix) ~= 250 bits, and fact 5 says the codes of a
level are nearly independent given the prefix and the coarser levels (reveals remove 5 bits). So the
parallel-within-level factorisation is NOT paying a correlation penalty: the 940 bits are an
**estimation gap**: the generator cannot compute the map text -> codes. Fact 6 says why: the map is
chaotic (a one-token shift changes 100% of level-0 codes), because the encoder is 2 bidirectional
attention layers over the chunk plus absolute positions, GroupNorm statistics pooled over all 64
positions, and residual-of-residual subtraction (fine code at position i = nearest centroid of
"token i's latent minus the interpolated coarse reconstruction", where the coarse reconstruction
itself depends on which centroids won). The AR control pays 250 bits because next-token prediction is
a smooth, local, learnable function. Any fix that passes the bits test must make the per-code map a
**simple function of (a few tokens, what the generator already knows)**, and it must do so at
levels 2-6, which hold 1,090 of the 1,190 bits. Fixes aimed at level 0 (Path A, "semantic" coarse
codes) cannot move the total by 280 bits even if they work perfectly: levels 0-1 together cost 101.

Where the generator's input comes from matters: `idx_to_ncm_input` hands level k the continuous
area-pooled `z_hat` of levels < k. So anything that is a fixed function of (token_i, z_hat_{<k, i})
is, in principle, learnable by the generator; anything that is a function of all 64 tokens through
attention is not.

---

### P1. Token-local, position-free encoder: codes become a product quantisation of tokens plus a bag-of-tokens plan
- Atoms used: row 7 (60, 193, 207), row 15 (81, 124, 263), row 20 (57, 102, 167). The live ones:
  193 (the attention mask as a routing device that enforces item independence; here taken to the
  limit: no token-to-token mixing in the encoder at all), 81 (restrict the encoder's capacity so it
  cannot build a hash), 124 (audio RVQ tokenizers: let the first levels carry a coarse summary and
  the residual levels carry exact detail), 102 (segmentation that respects boundaries instead of
  arbitrary fragmentation: fact 6 is the shift-fertility problem in code form).
- Mechanism: with `enc_layers=0`, no absolute position in the encoder and a per-position pre-quant
  norm, the pre-quant latent is `z_i = LN(wte[tok_i])`: a lookup. Then
  (a) fine levels: code_{k,i} = argmin_c |LN(wte[tok_i]) - z_hat_{<k,i} - e_{k,c}|, where z_hat_{<k,i}
  is exactly the generator's input for that level. The code is a fixed map of (token, known input);
  the generator only has to predict the token distribution and push it through the map. Levels 5-6
  (64 positions each, 412 bits now) become the two codebooks of a product quantisation of 8,192 token
  embeddings: given level 5, level 6 is almost deterministic. (b) coarse levels: code = nearest centroid
  of the mean embedding of a 16/8/4/2-token window (level 0 through the learned stride-16 conv, which is
  linear). A one-token shift moves the mean slightly, so codes change only when the mean crosses a
  cell boundary: fact 6 (100% change) becomes "some" change. (c) nothing in the code depends on the
  rest of the chunk except through the generator's own input (Phi convs and linear interpolation mix
  neighbours' coarse codes, but those are given). Facts 4 and 7: the per-code cost stops being a hash
  cost; the probe failure in fact 7 was a failure to predict a hash, not a failure to read the prefix.
- Exact change (all in `code_scale/models.py: build_codec`, one env flag `CODE_SCALE_ENC=lookup`):
  `enc_layers=0`, `enc_positional=False`, `pre_quant_norm_type='layer'` (GroupNorm over (C/8 x L)
  pools statistics over the whole chunk and would smuggle a global dependence back in; the upstream
  VQVAE already accepts `'layer'`), keep `dec_layers=2`, `dec_positional=True`, the l7 layout, 2,048
  codebooks, the restart fix, `fine_drop` as is (it keeps coarse levels decodable on their own, i.e. a
  real plan). `Encoder.forward` with these settings returns `wte(idx)` (checked in `upstream/ncm/models/vae.py`).
  Generator unchanged. Report shift stability from `plan_probe` as before.
- Why it's new: every `code_scale` codec so far used `enc_layers=2` with absolute positions; the
  layout, codebook-size, restart and plan-loss experiments all kept the contextual encoder. Upstream's
  original architecture was a lookup encoder (comment in vqvae.py), but with `wpe` (positional), which
  is exactly the shift fingerprint of fact 6; the position-free, per-token-normalised version has not
  been run anywhere here.
- One-night test: codec 5,000 steps (lookup encoder is cheaper than 0.87 s/step; ~50 min), gate check,
  NCM 1,800 steps on it (~95 min), teacher-forced bits per level on the same 2,048 held-out windows
  (`reveal_bits.py` 1-pass path or `plan_probe --gain-only`; ~10 min). Expected: levels 5-6 fall from
  412 to <= 80 bits (PQ second codebook is near-deterministic; first codebook given 48-position level 4
  is nearly a token-identity residual); the entropy moves to levels 3-4, where it is a learnable
  token-pair/window cost. Point estimate 520-600 bits (= 250 text bits + ~100-150 bits of
  sibling redundancy that only appears once codes are meaningful, see P3 + ~150 estimation gap).
  Pass line <= 660. Reconstruction gate: expected >= 99.8% tokens (two 2,048-codebooks at full
  resolution cover 8,192 token embeddings exactly; the decoder only has to invert a lookup).
- Cheat risk: none of the information kind (codes still determine the chunk; the gate measures it).
  The honest cost is conceptual: coarse codes become quantised bags of token embeddings, not
  "concepts"; the fine levels become tokens-given-a-plan. If bits drop but quality does not, that is
  the finding: NCM-for-code works only when its fine levels are token-like. A second effect to watch:
  within-level correlation becomes non-zero (siblings constrained by their parent), so Path C's
  reveal machinery, useless on hash codes (fact 5), becomes useful; measure it with the 3-pass
  `reveal_bits` on a reveal-trained generator if time allows (second night).
- Confidence it passes the 30% test: 45%. The chain-rule argument is solid; the uncertainty is whether
  a 6x384 generator trained 1,800 steps closes the estimation gap enough on the token-resolving levels 3-4.
- GPU cost on a Quadro T1000: ~2.5 h (codec 0.8 h, NCM 1.6 h, measurement 0.2 h).

### P2. Rate-distortion fine-tuning: the generator's cross-entropy is the rate term, differentiable through a local soft assignment, on all 236 codes
- Atoms used: row 2 (32, 81, 228), row 4 (33, 188, 224), row 8 (33, 60, 151), row 20 (12, 188, 244).
  81 (information bottleneck: minimise the code rate under a conditional entropy model while keeping
  the decoder happy), 188/60 (Gumbel-softmax / straight-through / Voronoi relaxation to get a gradient
  through the argmin), 33 (composite loss), 32 (the encoder absorbs the generator's calibrated
  distribution, not its argmax), 228 (rate budget allocated per level by the entropy model instead of
  uniformly).
- Mechanism: fact 4's number IS the generator's teacher-forced cross-entropy. Learned image/video
  codecs (Balle et al.) get low rate by training the encoder against the entropy model that will
  code it; here the entropy model is the prefix-conditioned generator. Minimising
  rec + vq + lambda * sum_k E_p_k[-log q_gen,k(c | prefix, c_<k)] moves a latent that sits near a
  Voronoi boundary into the cell the generator already favours and leaves cells the decoder needs
  apart. It attacks the estimation gap directly where it is measured (all levels, fact 4), including
  the fine levels (the plan-loss doc noted levels 1-3 get no gradient because residuals are built from
  z.detach(); here the residual target is recomputed as z - z_hat_{<k}.detach(), which carries z's
  gradient). Fact 5 says a code's cost does not depend on its siblings, so a per-code rate term is the
  right granularity.
- Exact change: new stage `train.py joint`, warm-started from `codec-l7/final.pt` and `ncm-l7/final.pt`
  (or from P1's pair). Per step on prefix+chunk windows (batch 32):
  1. `z = codec._encode_pre_quant(chunk)`; run the quantiser loop once with grad bookkeeping: for each
     level k, `r_k = codec.vq._down_for_quant(z - z_hat_{<k}.detach(), k, n_k)` (level 0 uses the
     learned down conv as now), hard codes `c_k = argmin |r_k - E_k|`, `z_hat` accumulated with the
     upstream up/Phi path (no grad). Decoder loss `rec` on the usual STE `z_hat`, plus upstream `vq_loss`.
  2. Generator teacher-forced on the hard codes (`ncm_logits`, corrupt_rate 0) -> `log q_k` per position.
  3. Local soft assignment: `p_k = softmax(-d_k^2 / tau_k)` restricted to the M = 16 nearest codes per
     position (mass outside is zero), `tau_k` = running mean of (second-nearest - nearest) distance at
     that level so that p has ~2-3 bits of entropy. Rate_k = mean over positions of
     `-sum_c p_k(c) * log q_k(c).detach()`.
  4. Alternate: even steps update the codec with `rec + vq_loss + lambda * sum_k rate_k` (generator
     frozen); odd steps update the generator with its normal CE on the current hard codes (codec frozen).
  Two lambdas (0.05 and 0.2, with rate in nats per code and rec in nats per token), 1,000 steps each.
  Log codes-in-use per level (collapse watch already in `codec_eval`).
- Why it's new: Path A was an InfoNCE between frozen-AR prefix features and the level-0 *latent*,
  weight 0.05, one level, no gradient path at levels >= 1, and it never touched the generator. P2
  optimises the measured quantity (all-level code CE under the actual generator), with a gradient
  through the discrete choice. Nothing in the tried list trains the tokenizer against an entropy model.
- One-night test: two joint fine-tunes of 1,000 steps (~3.5 s/step: codec fwd/bwd + generator
  fwd/bwd at batch 32; ~1 h each), gate on both, then for the better one retrain a fresh generator
  from scratch for 1,800 steps on the frozen final codes (~1.6 h) and measure bits. The from-scratch
  number is the one that counts (the co-trained generator could have tracked moving codes). Expected:
  on `codec-l7` the chaotic encoder leaves little slack: 940 -> 700-800 (fails or borderline); on P1's
  codec the rate term mostly tidies coarse levels and sibling structure: P1's number minus 50-100.
- Cheat risk: (1) collapse to few codes per level lowers rate and breaks reconstruction: caught by the
  gate and the codes-in-use log. (2) The encoder could chase the generator's biases (everything maps
  to the generator's favourite cell): also a reconstruction failure, caught by the gate. (3) Soft
  temperature too low gives a one-hot p and no gradient (no cheat, but a null result): the tau rule
  and the logged entropy of p guard it. (4) The co-trained generator's bits are not comparable to the
  baseline's: hence the from-scratch generator in the test.
- Confidence it passes the 30% test: 30% from `codec-l7` alone; 55% as a second night on top of P1.
  The direction is right by construction; the magnitude on a hash-like encoder is the doubt.
- GPU cost on a Quadro T1000: ~3.7 h (2 x 1 h joint + 1.6 h fresh generator + measurement).

### P3. The generator drafts tokens and the frozen quantiser verifies: a token-factored output head for full-resolution levels (on P1's codes)
- Atoms used: row 3 (111, 139, 228), row 10 (139, 168, 199), row 16 (139, 156, 185), row 1 (0, 32, 59).
  139 (propose-and-verify: a cheap drafter proposes tokens, an authoritative deterministic map
  verifies them; here the map is the frozen encoder + quantiser), 32 (distil the AR control's
  calibrated token distributions into the generator), 185 (parallel decoding given global context),
  60 (Voronoi partition as the output parametrisation).
- Mechanism: after P1, a full-resolution code is `c(tok_i, u_i) = argmin_c |LN(wte[tok_i]) - u_i - e_c|`
  with `u_i` the generator's own input at that position. A free 2,048-way softmax head has to learn
  this map separately for every (token, input-cell) pair from ~3.7M training positions; the AR control
  only has to learn p(token | context). Give the generator a token head (8,192-way, tied to the codec
  `wte` it already uses for the prefix) and define the code logits as the push-forward of the token
  distribution through the known map: `q(c) = sum_tok p(tok) 1[c(tok, u_i) = c]` over the top-K = 64
  tokens, plus a small free-head background for the remainder. The fine-level estimation gap collapses
  to the token-prediction gap, which the AR control shows is small at this budget (fact 4: 250 bits).
  Fact 5 (no within-level correlation to exploit) is unchanged by this head; it does not pretend otherwise.
- Exact change: in `code_scale/models.py` a `TokenFactoredHead` for levels whose resolution is 64
  (levels 5 and 6 in l7): (1) `tok_logits = h @ codec.encoder.wte.weight.T` (shared, frozen table);
  (2) top-K tokens per position; (3) codes for those tokens: distances of `LN(wte[tok]) - u_i` to the
  level's codebook (K x 2,048 x 256 per position; ~70 GFLOP per level per batch of 32, well under a
  second on the T1000); (4) `logits_c = log(softmax(tok_logits)_topK scattered by argmin + eps + free_head_c)`.
  Train with the usual code CE (the gradient reaches the token head through the scatter) plus an
  auxiliary token CE with weight 0.5 on the same positions (the minimal version, if the push-forward is
  too fiddly for one night, is this auxiliary token head alone). `u_i` is already computed by
  `idx_to_ncm_input`; the Phi conv is applied after selection in `quant.py`, so the argmin on raw
  codebook distances is the exact encoder rule.
- Why it's new: nothing tried touches the output head or uses the frozen quantiser inside the
  generator; Path B/C changed attention and decoding order only. It is not "predict tokens instead of
  codes": the measured quantity is still the code cross-entropy, the decoder is unchanged, and coarse
  levels keep the free head.
- One-night test: requires P1's codec and its 1,800-step generator as the comparison. Train the P3
  generator 1,800 steps on the same codec, same seed and windows (~1.8 h with the extra head), measure
  per-level bits. Expected: levels 5-6 to within 20 bits of their true conditional entropy (<= 60 total),
  level 4 unchanged; total = P1's number minus 40-120 bits. Gate is the codec's, unchanged.
- Cheat risk: the head could put all mass on the background and ignore the token path (null result,
  not a cheat; logged as the share of mass through the push-forward). No information is discarded: the
  codes are still the codec's codes. A subtle one: if K is too small the push-forward misses the true
  token and the CE spikes; the eps background bounds it, and K = 64 covers > 99% of AR top-p mass on code.
- Confidence it passes the 30% test: as an add-on to P1, 55% combined; alone on `codec-l7` it is
  useless (a hash code is not a function of one token), so 0% there.
- GPU cost on a Quadro T1000: ~2 h on top of P1's run.

---

## Rejected after checking against the facts (so they are not re-proposed)

- Consensus / view-invariance codes (row 1 atom 0, rows 5/11 atoms 65/66/156/199: make the coarse
  codes agree across AR-sampled alternative continuations or shifted windows). This forces level-0
  codes toward a function of the prefix, which pushes the chunk's information into the fine levels,
  where the hash cost is paid in full. Total bits do not fall; only the level-0 share (33 bits) moves.
  It is Path A with a different positive pair.
- Semantic distillation of coarse levels to the AR control's mid-layer features (atoms 32, 46, 207,
  124; VA-VAE / SpeechTokenizer analogue; listed in the plan doc as untried). Coarse levels hold
  101 of 1,190 bits; even a perfect result cannot deliver the 280-bit cut. Worth doing for quality
  after P1, not for this test.
- Prefix-conditioned tokenizer (encoder and decoder cross-attend to the frozen AR prefix states, atoms
  268/263/46). The bound H(codes | prefix) <= H(chunk | prefix) is the same with or without
  conditioning; the only thing conditioning can add is a simpler code map, and the encoder it needs
  (cross-attention) reintroduces context mixing. Not worth a night before P1/P2 have been run.
- Constrained/speculative/MBR decoding, classifier guidance (atoms 57, 61, 0, 139 as decoders): they
  change samples, not teacher-forced bits. Fail the test's definition.
- Non-uniform codebook pruning by sensitivity (atom 228 alone): B2 showed fewer bits by construction
  (1,008) without learnability; pruning the fine codebooks loses reconstruction (gate).
- Causal (within-chunk) encoder so fine codes depend on tokens <= i: sum over positions of
  H(tokens <= i) blows past 250 bits; parallel prediction of running states is the wrong object.
- Haar / critically sampled pyramid to remove sibling redundancy (siblings' residuals sum to the
  parent's quantisation error): a real cost once codes are meaningful (my estimate ~100-150 bits on
  P1's codes), but it is a prediction, not a measured fact; run `reveal_bits` 3-pass on P1 first and
  propose it only if 3 passes remove >= 100 bits.
