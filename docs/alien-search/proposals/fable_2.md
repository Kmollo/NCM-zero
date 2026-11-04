# Proposals (fable_2): make NCM's codes predictable from the prefix

Source: brief.md, combos_2.json, code_scale/{models,plan_loss,train,reveal_bits}.py, upstream/ncm/models/{vqvae,quant,vae}.py,
runs/code-scale-v1/ncm-l7/plan-probe.json and the codec/NCM logs (for timings).

## Reading of the facts before any idea

1. **Where the bits are.** Per level: 33, 68, 147, 246, 285, 128, 284. Levels 0-2 hold 248 of 1,191 bits.
   The pass bar is -282 bits. So **no change confined to the coarse "plan" levels can pass**, even if it made
   them perfectly predictable. Path A (level 0 = 33 bits) was arithmetically unable to pass this test. Every
   proposal below has to change the geometry of levels 3-6 (815 bits).
2. **The codes behave like hashes of the exact chunk.** Four independent facts say so: (a) the code-frequency
   prior at every level is 8.2-10.8 bits, i.e. near-uniform over 2,048 codes (plan-probe.json, blind_guess);
   (b) a 1-token shift changes 100 % of level-0 codes (fact 6); (c) knowing all other same-level codes removes
   0.6 % of the extra bits (fact 5), so codes carry no mutual information a model can use; (d) AR prefix
   features predict level-0 codes 0.1 bits better than the prior (fact 7). A deterministic function of the text
   whose outputs look uniform, flip under tiny edits and are mutually uninformative is a hash.
3. **Why the tokenizer hashes (code-level cause).** z is GroupNorm'd 256-d (||z||^2 ~ 256 per position). Each
   level places 2,048 centroids in 256-d (`quant.py` `_distances`/argmin): the nearest centroid explains a
   small part of the residual energy, the residual handed to the next level is nearly as large as the input and
   points in a direction set by *which cell z fell into*, i.e. it is quantization error. Six of seven levels
   quantize quantization error, which is noise-like by construction. On top of that the encoder is a
   bidirectional 2-layer transformer with absolute `wpe`, so z at position t depends on all 64 tokens and on
   the chunk-boundary offset (fact 6). Information-theoretically H(codes | prefix) <= 250 bits (fact 4); the 940
   extra bits are a *learnability* gap: a 13M generator cannot learn p(code | prefix) when the code is a
   near-random partition of the plausible continuations.
4. **Cheap pre-check (CPU, ~10 min, no training; run before spending a night).** On codec-l7 and 1,024 val
   chunks, inside `MultiscaleResidualVectorQuantizer.z_to_idx`: per level k log
   `E||quantized_k||^2 / E||z_rest_k||^2` (energy removed) and the cosine between `z_rest_k` of a chunk and of
   the same chunk shifted by one token (at the matching slots). Prediction of the hash hypothesis: energy
   removed < 25 % at levels 1-6, cosine < 0.3 after level 0. If energy removed is > 60 % per level the
   hypothesis is wrong and Proposal 1 should be dropped in favour of Proposal 3 alone.

The 20 combos were used as follows: combo 13 (33 composite loss / 141 soft labels / 188 Gumbel-STE) and combo 1
(189 STE) shaped Proposal 3; combos 3/4 (124 audio semantic-vs-acoustic tokenization) and 11 (32 hidden-state
distillation) shaped Proposal 4; combos 7/12 (193 attention masks), 9 (201 positional encoding) and 19 (81
information bottleneck, 100 compositional sub-word encoders) shaped Proposal 2. Atoms 156/0/9/139
(diffusion, MBR, flow matching, speculative decoding) are sampler-side and were rejected: fact 5 and Path C show the
sampler is not the cause. Atom 45 (edit operations) is discussed under Proposal 3 (the rate loss learns the
"copy" code it describes without hand-building it).

---

### Proposal 1: Low-dimensional FSQ levels (de-hash the quantizer geometry)
- Atoms used: 188 (combo 13), 122 (combos 6, 9), 81 (combo 19). Mostly derived from the facts; the atoms only confirm the tool.
- Mechanism: replace each level's 2,048-centroid nearest-neighbour search in 256-d with finite scalar
  quantization in a learned 4-d projection ([8,8,8,4] = 2,048 codes). Cells become large axis-aligned boxes in a
  4-d space the encoder is forced to make informative, instead of tiny Voronoi cells in 256-d. Consequences tied to
  the facts: a small text edit moves the 4-d point a little and changes at most one digit by one (fact 6 should
  fall from 100 % to well under 50 % at level 0); the residual passed to the next level is bounded in the quantized
  directions (rounding error <= half a bin) and the next level projects onto *other* directions, so levels become a
  progressive, PCA-like decomposition of z rather than nested quantization error (fact 5 reverses: neighbouring
  and coarser codes become informative); and the plausible continuations given a prefix land in a few adjacent
  boxes instead of being scattered over unrelated code IDs, which is what lets a 13M generator learn
  p(code | prefix) with few bits (fact 4, the learnability gap). Capacity is unchanged (11 bits per code), so the
  information budget for reconstruction is identical.
- Exact change (`upstream/ncm/models/quant.py`, new config `fsq: {enabled: true, digits: [8,8,8,4]}` passed from
  `code_scale/models.py` `build_codec`):
  - per level k: `down_k = nn.Linear(256, 4)`, `up_k = nn.Linear(4, 256)`.
  - forward/`z_to_idx`: `u = tanh(down_k(flat_input)) * (L_i/2 - 1e-3)` per digit, `d = u + (round(u) - u).detach()`
    (STE on every level; FSQ has no codebook to diverge, unlike the VQ STE note in `quant.py`), integer digits
    shifted to 0..L_i-1, `index = mixed_radix(digits)`, `quantized = up_k(d)`. Keep `_up_to_full`, `Phi`
    (`_apply_quant_resi`), the residual subtraction, fine_drop and the final `z_hat = z + (z_hat - z).detach()`.
    Drop EMA, restarts, `commitment_cost` on FSQ levels (keep `mse(z_hat, z_no_grad)` as the only vq term).
  - keep `self.embeddings[k]` as a non-trainable `nn.Embedding(2048, 256)` buffer refreshed each step from
    `up_k(all 2048 digit combinations)`, so `idx_to_ncm_input`, `idx_to_z_q`, `_TiedCodebookHead`, `Reveal` and
    the generator need no change. `SHAPE["codebook"]` stays 2048; bits are measured over the same 2,048-way head.
  - log per level: digit utilisation histogram and marginal code entropy (replaces `codes_used_l0_l2`).
- Why it's new vs "already tried": Path B2 shrank the *number* of codes in 256-d (cells still tiny relative to
  z's spread, still hashes; bits fell only because information fell). Level layouts and the restart fix do not
  touch cell geometry. Nothing so far changed the dimensionality of the space the argmin runs in.
- One-night test (~4.5 h): codec-fsq 5,000 steps (~70 min, same schedule as codec-l7; EMA removal makes it
  slightly faster). Gate: `codec_eval` >= 99 % tokens, >= 50 % exact. Immediately (10 min, no NCM yet): the
  pre-check metrics of section 4 plus `plan_probe` stability; require 1-token-shift change at level 0 <= 50 %
  and per-level marginal entropy >= 8 bits (proof the codes still carry information). Then NCM 1,800 steps
  (~2.2 h, `CODE_SCALE_CODEC=codec-fsq`), then teacher-forced bits per chunk (the 1-pass path of
  `reveal_bits.py` or `plan_probe --gain-only`, ~30 min). Expected: levels 0-2 248 -> ~150, levels 3-6
  943 -> ~500; total ~900, extra ~650 (pass <= 660 is at the edge: this is a 30-40 % cut, not a sure thing).
- Cheat risk: FSQ can waste codes (digits saturating at the tanh bounds, effective codebook << 2,048), which
  lowers bits by carrying less information. Caught by (a) the reconstruction gate, (b) the per-level marginal
  entropy floor (>= 8 bits), and (c) reporting the conditional-minus-marginal gain per level: a pass must come
  from the conditional bits falling, not the marginal ones (the Path B2 signature).
- Confidence it passes the 30 % test: 35 %. The mechanism addresses all four hash facts, but a 30 % cut from
  geometry alone with a 2-layer bidirectional encoder still in place is uncertain.
- GPU cost on a Quadro T1000: ~4.5 h (codec 1.2 h, checks 0.3 h, NCM 2.2 h, bits 0.5 h).

### Proposal 2: Local, position-free encoder (codes as n-gram features, shift-equivariant by construction)
- Atoms used: 193 (combos 7, 12), 185 (combos 5, 12, 14, 18), 201 (combo 9), 100 (combo 19).
- Mechanism: today the code at slot t is a function of all 64 tokens and the absolute offset (bidirectional
  attention + `wpe`), which is why a 1-token shift rewrites every level-0 code (fact 6) and why same-level
  neighbours tell the generator nothing (fact 5). With windowed attention (|i-j| <= 4) and no absolute position,
  z_t is a function of tokens t-4..t+4 only. Fine-level codes then are local n-gram clusters: shifting the window
  moves the codes one slot instead of changing them (equivariance); predicting the code at slot t given the
  prefix and coarser codes is predicting a local n-gram cluster, the kind of thing an LM is good at, and codes
  at neighbouring slots become strongly informative about each other. Coarse levels pool 16/8/4 local features,
  so a 1-token shift changes 1/16 of their input instead of the whole thing. Reconstruction is not at risk: the
  decoder stays global (`dec_positional=True`) and 64 fine codes carrying local identity are enough.
- Exact change (`code_scale/models.py` `build_codec`, all flags already exist in `upstream/ncm/models/vae.py`):
  `enc_attn_window=4`, `enc_positional=False`, optionally `enc_conv_layers=2, enc_conv_kernel=3` (local n-gram
  features before attention). Level-0 learned downsampler (`learned_l1_sampling`, stride 16) is unchanged.
  New metric in `plan_probe`: *equivariant* shift stability (compare code[t] after a 1-token shift with code[t+1]
  before), since slot-wise comparison would wrongly count a moved code as changed at fine levels.
- Why it's new vs "already tried": no run changed the encoder's receptive field or positional scheme; the
  "line-boundary chunking" idea in the plan attacks the same symptom from the data side and was not run.
- One-night test (~4.5 h, or ~1.2 h extra if run as arm B/C next to Proposal 1): codec-local 5,000 steps,
  gate, pre-check + equivariant stability (require level-0 shift change <= 50 %), NCM 1,800 steps, bits. A direct
  mechanism test is free: re-run `reveal_bits.py` on this NCM with reveals; if `removed_share` rises from 0.006 to
  >= 0.2, the codes have become mutually informative as claimed, whatever the total says. Expected: levels 3-6
  943 -> ~650, levels 0-2 ~200; extra ~800 alone. Recommended night: train codec-fsq, codec-local and
  codec-fsq+local (3 x 70 min), pick the gate-passing one with the best pre-check numbers, train one NCM on it.
- Cheat risk: a window of 4 with no position could make the encoder unable to encode token order inside a
  window; the decoder then reconstructs wrong orders and the gate fails (visible as exact-chunk < 50 % with token
  accuracy still high). The information budget is otherwise untouched (same 236 x 11 bits).
- Confidence it passes the 30 % test: 20 % alone (the residual-of-residual hash from section 3 is untouched),
  ~40 % combined with Proposal 1.
- GPU cost on a Quadro T1000: ~4.5 h alone; +1.2 h as an extra arm beside Proposal 1.

### Proposal 3: Rate-distortion tokenizer: the code's cost under a conditional prior as a training term on all seven levels, through soft assignments, with a prefix-conditioned codec
- Atoms used: 33 + 141 + 188 (combo 13), 189 (combo 1), 66 (combos 1, 10, 14, 16, 19), 45 (combos 1, 8).
- Mechanism: the measured quantity is the NLL of the codes under a model that sees the prefix and the coarser
  levels. Train the tokenizer to minimise exactly that, as learned compression does (distortion + lambda x rate
  under a learned conditional prior). Among the many code assignments that reconstruct the chunk equally well
  (fact 4: 940 bits of slack with no information to lose), the encoder is pushed to the one a prefix-reading
  prior finds cheap: smooth, shift-stable, and defaulting to a "nothing new here" code wherever the prefix and
  coarser levels already determine the content (this is the "copy" operation of atom 45, learned rather than
  hand-built). The gradient reaches the encoder through a soft assignment over codes (atom 141: the hard code
  treated as a probabilistic label; atoms 188/189: STE/Gumbel-style relaxation), so unlike Path A it acts on the
  codes themselves and on all levels, where the bits are. Conditioning the encoder on the prefix is what lets it
  *act* on the signal: without the prefix it can only lower marginal entropy (the Path B2 trap); with it, it can
  make codes cheap relative to what the prefix implies (atom 66: the generator sees the prefix with the chunk
  occluded; the encoder should encode only what the occlusion hides).
- Exact change (`code_scale/train.py` codec stage, `code_scale/plan_loss.py`, small hook in `quant.py`):
  - windows of P+T (the planner branch already does this). Prefix features: `ar_prefix_features` (frozen AR,
    768-d, already implemented, 1.2 s / 64 windows) -> `Linear(768, 2*256)` -> two context vectors prepended to
    the encoder input sequence and to the decoder input (`Encoder.forward` / `Decoder.forward` take an optional
    `ctx` tensor; strip the 2 slots before `pre_quant_norm`). `codec_eval` and `load_codec` consumers pass the
    prefix too; `VQVAE.encode/decode` get a `ctx` argument (the generator already embeds the prefix and can pass
    the same features at generation).
  - `RatePrior`: per level k a 2-layer transformer over the `levels[k]` positions (d=256, ~1M params total)
    whose inputs are the AR prefix features (as a prepended token) plus, for k >= 1, the coarser-level
    conditioning the real generator gets: `z_hat` after levels < k area-pooled to `levels[k]` (exactly what
    `idx_to_ncm_input` builds), detached. Output: logits over 2,048 per position.
  - `quant.py` returns per level `dist_k` (B*level x 2048) and the hard `idx_k`. Soft assignment
    `p_enc = softmax(-dist_k / tau)` (tau learned, init = median nearest-vs-second-nearest gap); with Proposal 1's
    FSQ, `p_enc` = product of per-digit softmaxes over distances to bin centres (cheap, 2,048 entries).
  - losses: `rate_fit_k = CE(logits_k, idx_k)` (trains the prior; encoder detached) and
    `rate_enc_k = sum_c p_enc(c) * (-log_softmax(logits_k.detach()))[c]` (trains the encoder; prior detached).
    `loss = rec + vq + sum_k rate_fit_k + lambda * sum_k rate_enc_k`, lambda ramped 0 -> 0.05 between steps 1,000
    and 2,000 so the codec first reaches reconstruction; `plan_batch=32` as in Path A for the AR read.
  - log per level: `rate_fit_k` in bits (a live estimate of the generator's bits per code), marginal code entropy.
- Why it's new vs "already tried": Path A was InfoNCE between prefix features and the *continuous* level-0
  latent, weight 0.05, level 0 only (33 bits, 3 % of the total; could not pass this test even if perfect). This
  is (a) all seven levels, (b) the measured quantity itself (code NLL under a prior that sees the coarser levels,
  as the generator does) rather than a contrastive proxy, (c) gradient through the quantizer's soft assignment,
  not only the pre-quant latent, (d) a prefix-conditioned encoder/decoder, which no run has had.
- One-night test (~5 h): codec-rd 5,000 steps at ~1.6 s/step (AR read + prior) ~2.2 h, or warm-start from
  codec-l7 (or codec-fsq from Proposal 1) for 2,000 steps ~1 h; gate; NCM 1,800 steps 2.2 h; bits 0.5 h.
  Decision numbers available *before* the NCM run: the prior's own `rate_fit` bits per chunk at the end of codec
  training. If the prior is below ~900 bits per chunk the NCM run is worth doing (the prior has the generator's
  inputs and less capacity, so the generator should do at least as well). Expected: extra bits 940 -> 550-650
  stacked on Proposal 1; 940 -> 700-800 on the plain VQ codec (the rate gradient is weak in 256-d where the soft
  assignment is almost one-hot).
- Cheat risk: the rate term rewards marginal collapse (fewer codes used), which is Path B2 with extra steps.
  Caught by (a) the gate, (b) requiring the conditional bits (`rate_fit`, NCM bits) to fall at least 2x more
  than the marginal (blind-guess) bits per level, (c) `rec_loss` within +0.02 of the baseline codec at the same
  step. A second risk is the prefix-conditioned decoder "reconstructing from the prefix" instead of the codes;
  it cannot (a 2-block decoder is no language model), and exact-chunk accuracy would collapse if it tried.
- Confidence it passes the 30 % test: 40 % stacked on Proposal 1, 25 % on the VQ codec. It optimises the
  pass metric directly; the uncertainty is whether a 1M-param prior trained for 5,000 steps is a faithful enough
  stand-in for the generator.
- GPU cost on a Quadro T1000: ~5 h (3.5 h with warm-start).

### Proposal 4: Distil the AR control's chunk hidden states into the pre-quant latent (an LM-aligned "semantic" tokenizer, Mimi/SpeechTokenizer style)
- Atoms used: 124 (combos 3, 4), 32 (combo 11), 50 and 270 (combos 10, 11, 15), 263 (combo 4: frozen backbone,
  trained bridge).
- Mechanism: in neural audio codecs, RVQ tokens became LM-predictable only after the first codebook was
  distilled from a self-supervised speech model (SpeechTokenizer, Mimi): the latent stops being an arbitrary
  reconstruction-optimal embedding and becomes a smooth function in the coordinates an LM uses. Analogue here:
  add a loss pulling `z_t` (pre-quant, all 64 positions) toward the frozen AR control's hidden state after
  reading chunk token t (teacher-forced over prefix + chunk). The AR state is smooth in content and
  depends mostly on the left context, so z becomes a trajectory in LM coordinates; the prefix-conditional
  distribution over such trajectories is concentrated (it is the AR's own uncertainty), and once quantized into
  large cells (needs Proposal 1's geometry; in 256-d Voronoi cells it would still hash) the code distribution
  given the prefix concentrates with it. This targets fine levels too because it reshapes z at full
  resolution, not only a pooled plan (facts 4-5), and LM states change smoothly under a 1-token shift (fact 6).
- Exact change (`code_scale/train.py` codec stage): windows of P+T; frozen AR forward as in
  `ar_prefix_features` but keep `h = ar.ln(x)[:, P:P+T]` (positions that have read chunk token t), 384-d;
  `loss_distill = 1 - cos(Linear(256->384)(z), h.detach())` averaged over positions, weight 0.5 ramped over the
  first 1,000 steps; `plan_batch=32` windows carry the loss (AR read 1.2 s / 64). Everything else unchanged;
  best run on top of the Proposal 1 quantizer.
- Why it's new vs "already tried": Path A tied the level-0 pooled latent to *prefix* features contrastively
  (prefix -> plan). This ties every position's latent to the LM's reading of the *chunk itself* by regression,
  a target that exists for all 64 positions and all levels. No run has used the AR as a teacher for z.
- One-night test (~4.7 h): codec-distil 5,000 steps (~2 h with the AR read), gate (the distillation target
  under-weights the current token identity, so watch exact-chunk; if the gate fails at weight 0.5, 0.2 is the
  one allowed retry), pre-check metrics, NCM 1,800 steps, bits. Expected: 940 -> 650-750 stacked on
  Proposal 1; alone (VQ geometry) little change, and I would not run it alone.
- Cheat risk: low; the loss adds a target, it does not remove information, and the gate guards
  reconstruction. The failure mode is the opposite (reconstruction loss from fighting the teacher).
- Confidence it passes the 30 % test: 20 %. The audio precedent is strong, but the code analogue is untested and
  the AR teacher is itself small (16M, 250 bits/chunk).
- GPU cost on a Quadro T1000: ~4.7 h.

---

## Rejected on the way (so they are not re-proposed)
- Any coarse-level-only fix (semantic level 0-2, bigger "plan" codebooks, level-0 AR): bounded by 248 bits.
- Smaller codebooks, dropping fine levels, fewer levels: reduce bits by removing information (Path B2).
- Fully causal encoder with the finest level tied to token identity: rebuilds the AR model inside the tokenizer.
- Sampler-side ideas from the combos (diffusion/MaskGIT passes, MBR, speculative decoding): Path C measured that
  within-level factorisation costs 5 bits of 940; the sampler is not the problem.
- Hand-built edit/copy codes at the finest level (atom 45): what Proposal 3's rate term learns on its own.
- Line-boundary chunking: plausible for fact 6, but changes the data pipeline and does not address facts 4-5.

## Recommended order for the nights
Night 1: Proposal 1 with Proposal 2 as extra codec arms (3 codecs, one NCM). Night 2: Proposal 3 warm-started
from the best Night-1 codec. Proposal 4 only if Night 1 shows the geometry change moved shift-stability but not
the bits. Before any night: the 10-minute CPU pre-check in section 4.

## Summary
1. Low-dimensional FSQ levels (de-hash the quantizer): 35 %.
2. Local, position-free encoder: 20 % alone, ~40 % with 1.
3. Rate-distortion tokenizer with a conditional prior on all levels + prefix-conditioned codec: 40 % stacked on 1, 25 % alone.
4. AR-state distillation into the latent (Mimi-style): 20 %, only as a component.
Genuinely useful combos: 13 (composite loss + soft labels + STE -> the soft-assignment rate loss of 3) and 3/4/11 (audio semantic/acoustic split + hidden-state distillation -> 4); combos 7/12/9/19 nudged 2 (local masks, positions, bottleneck). Proposal 1 (FSQ) and the "where are the bits" arithmetic I would have reached from the facts alone; the diffusion/MBR/flow-matching atoms that dominate the list were not useful here.
