# Proposals (fable_1): making NCM's codes predictable from the prefix

Written after reading the brief, combos_1.json, code_scale/{models,plan_loss,train,reveal_bits}.py,
upstream/ncm/models/{vqvae,quant,vae}.py, docs/ncm-next-plan.md and the run summaries
(runs/code-scale-v1/*/summary.json, ncm-rv7/reveal-bits.json, diag-l7/report.json).

## Two framing facts that every proposal below is built on

**F-A. The 940 extra bits are a learnability gap, not an entropy gap.** The codes are a deterministic
function of the chunk, so H(codes | prefix) = H(chunk | prefix) <= ~250 bits (what the 16M AR model
achieves). Any gap above that is the 13M generator failing to *compute* the code map, not the codes
holding more information. Facts 5 and 7 say the map is hash-like: a code depends on the exact tokens
(fact 6: 100% of level-0 codes change on a 1-token shift; rename one identifier -> 20% change), so a
parallel predictor would have to enumerate plausible 16-token spans and hash each one. Nothing in the
tokenizer's objective (reconstruction + commitment; EMA k-means codebooks with dead-code restart at
threshold 0.1) ever asks the map to be smooth or low-rate. Quite the opposite: k-means spreads centroids
over whatever residual variance the encoder emits, and restart re-seeds any code that goes quiet.

**F-B. The bits live in the fine levels, so coarse-only fixes cannot pass the test.** Per level
(ncm-rv7/reveal-bits.json): 33 / 68 / 147 / 246 / 285 / 128 / 284. Levels 0-2 hold 248 bits in total;
the pass line needs -280. Levels 3-6 hold 942. Fine levels sit at 48-64 positions (about one per
token) and are the later stages of a *residual* quantizer: they quantize `z - sum(coarse)`, i.e. the
continuous 256-d leftover of a 2-layer bidirectional encoder after GroupNorm. That leftover carries far
more degrees of freedom than token identity; fine_drop (p=0.5, min_keep=1) trains the decoder to work
from truncated levels, so most token identity is already fixed by levels 0-4. Fine codes are therefore
mostly *nuisance*: high entropy (2-6 bits/position under the generator) with little information about
x. This is also why Path B2's small coarse codebooks cut only 180 bits: they touched the wrong levels.

A 30-minute pre-check (no training) that decides how much upside the fine levels hold is Step 0 of
Proposal 1 below.

I reject at the outset (see the end): MaskGIT/diffusion samplers (rows 1, 17; fact 5), SpeechTokenizer-
style semantic distillation of the coarse levels (rows 3/6/10/12/16/19, atom 32; it can touch at most
248 bits and pushes token identity down into the fine levels, raising their bits), speculative decoding
(atom 139; inference only), and smaller codebooks (tried).

---

### 1. Rate-distortion tokenizer: a conditional entropy model as a training loss on every level

- **Atoms used:** row 2 (111, 188, 228), row 8 (26, 66, 122). 228 (decouple a global sparsity budget
  from per-layer allocation, by sensitivity) -> per-level rate budgets lambda_l; 188 (continuous
  relaxation / straight-through to get gradients through discrete choices) -> soft assignments carry
  the rate gradient to the encoder at levels >= 1, where upstream has none; 26 (ELBO: a rate/KL term
  is the half of the objective the tokenizer is missing); 111 (dictionary features should be few and
  monosemantic rather than a dense hash).
- **Mechanism.** Add to the tokenizer's loss the cross-entropy that a *parallel, NCM-shaped* predictor
  pays for each code given (prefix features, true coarser levels, position). The tokenizer then
  minimises reconstruction + lambda_l * bits_l, i.e. exactly the quantity the test measures, subject to
  the gate. Two things happen: (i) nuisance variance that costs bits but buys no reconstruction is
  pushed out of the fine levels (fact F-B: 942 bits with little information); (ii) where a code must
  carry information, the encoder places it in cells the predictor can reach from its inputs, which is
  the "smooth map" fact 7 asked for. Fact 7 is the existence proof: a 0.05-weight InfoNCE term on
  level 0 alone moved its predictability by 1 bit/code without hurting reconstruction. Learned image
  and audio compression (hyperprior entropy models) show the general effect: without a rate term the
  latent's entropy balloons; with one it collapses to what the decoder needs.
- **Exact change** (code_scale, new `--rate-weight` path in `train.py`, new module `rate_loss.py`):
  1. *Gradient path.* In a wrapper around `codec.vq.forward` (do not edit upstream), recompute, for
     each level l >= 1, the quantizer input with the encoder gradient kept:
     `z_rest_l^g = z - sg(sum_{m<l} quantized_m)` (same values as upstream's detached `z_rest`, different
     graph), then `d_l = ||down_l(z_rest_l^g) - E_l||^2` (B, n_l, 2048). Level 0 already has a gradient
     path through `learned_down_l1`. Codebook EMA updates and the decoder path are untouched.
  2. *Soft assignment.* `q_l = softmax(-d_l / tau_l)`, tau_l = running median of the nearest-neighbour
     distance at level l (no new hyper-parameter; keeps q peaked but differentiable). No straight-through
     at levels >= 1, so the upstream warning ("ST at higher levels diverges") does not apply.
  3. *Entropy model (proxy generator), per level:* an MLP with the same inputs the real NCM generator
     gets at that position: `codec.vq.idx_to_ncm_input(codes)` slice for level l (detached, 256-d, the
     coarser levels' reconstruction at this position), a learned position embedding, and the prefix
     feature. Prefix feature = frozen AR last+mean hidden state (plan_loss.ar_prefix_features) for
     levels 0-2 only (where the prefix matters; 1.2 s per 64 windows, as Path A measured), nothing for
     levels 3-6 (the generator's prefix gain there is small and fact 5 says within-level context is
     irrelevant, so a per-position MLP is a fair proxy). Head: `logits_lk = -||W_l h - E_l[k]||^2 / s_l`
     tied to the (detached) codebook, see Proposal 4 for why.
  4. *Rate term:* `R_l = mean over positions of sum_k q_l[k] * (-log p_proxy(k))`, i.e. the proxy's
     cross-entropy under the soft assignment; gradient reaches the encoder through q and the proxy
     through p. Loss = rec + vq_loss + sum_l lambda_l * R_l / ln 2.
  5. *Budgets (atom 228):* lambda = [0.02, 0.02, 0.05, 0.1, 0.1, 0.1, 0.1] (coarse levels protected:
     fact 3 says quality lives there; fine levels pay). Ramp lambda linearly from 0 at step 1000 to full
     at step 2500 of 5000 so the codebooks form first.
  6. *Stop fighting the rate term:* set `restart_threshold` to 0 for levels >= 3 after step 1000 (the
     restart re-seeds dead codes *inside* the dense "null residual" region and splits it, which is how
     a near-zero residual becomes an 11-bit code). The existing `fixed_restart` hook is the place to do
     this per level.
  7. Everything else as codec-l7 (levels 4,8,16,32,48,64,64, codebook 2048, fine_drop on).
- **Why it is new vs "already tried".** Path A was a contrastive (InfoNCE) term on level 0 only, from
  the prefix only, with the quantizer's detached residuals making levels >= 1 unreachable (the Path B
  notes say exactly this). This proposal (a) is a proper rate term (cross-entropy of a code predictor,
  the measured quantity itself), (b) reaches all seven levels through un-detached soft assignments,
  (c) conditions on the coarser levels like the real generator, (d) budgets per level, and (e) removes
  the dead-code restart that actively creates the fine-level entropy. Path B2 cut capacity by shrinking
  codebooks; this cuts *entropy* while leaving capacity for the hard chunks.
- **One-night test.**
  - *Step 0 (30 min, no training, decides the upside):* with codec-l7, decode from truncated levels
    (`vq.idx_to_zhats` -> `decoder(z_hats[k])`) on 1,024 val chunks and report token accuracy for
    k = 2, 3, 4, 5, 6. If levels 0-4 alone give >= 95% tokens, levels 5-6 (412 bits) are nearly pure
    nuisance and this proposal has >= 300 bits of headroom from them alone; if they give < 85%, cap the
    expectation and raise lambda on levels 3-4 instead.
  - Train codec-rd 5,000 steps (batch 64; ~2.5-3 h with the AR prefix read on a 32-window sub-batch for
    levels 0-2, as Path A ran). Gate from `codec_eval` at every 500 steps; abort if token accuracy at
    step 3,000 < 98.5%.
  - Train the *standard* NCM generator on codec-rd, 1,800 steps (~2.2 h), unchanged architecture, then
    measure teacher-forced bits with nothing shown (reveal_bits logic with PASSES=(1,); 10 min).
  - Expected: levels 3-6 from 942 to 450-600, levels 0-2 roughly unchanged -> total ~700-850, extra
    ~450-600 (baseline 940; pass <= 660). Report per-level bits and `codes_used` per level.
- **Cheat risk.** The direct cheat is a tokenizer that makes codes cheap by dropping information; the
  reconstruction gate (>= 99% tokens, >= 50% exact chunks on 1,024 chunks, decoded from codes alone)
  is the catch and is measured on the same codec that the generator is then trained on. A second,
  subtler risk is "teaching to the test": the bits must be measured with a *freshly, normally trained*
  NCM generator, never with the proxy. Third, soft assignments could let the encoder sit on cell
  boundaries (q spread) so that the hard argmin code at inference differs from what the decoder was
  trained on; `codec_eval` uses `encode` (hard argmin) so the gate catches this too, and tau tied to
  the NN-distance median keeps q peaked.
- **Confidence it passes the 30% test: 45%.** It optimises the measured quantity directly and the
  fine-level nuisance is large, but the proxy may not transfer to the real generator and the
  rate/restart interplay with EMA codebooks is untested here.
- **GPU cost on the T1000:** ~5.5-6 h (0.5 + 3 + 2.2 + 0.2).

---

### 2. Prefix-conditional ("P-frame") tokenizer: codes carry only what the prefix cannot predict

- **Atoms used:** row 15 (69, 236, 272), row 1 atom 122 (conditional VQ-VAE). 236 is the key one:
  video codecs separate the static, predictable background from transient motion and code only the
  residual against a prediction from the previous frame. The 192-token prefix is the previous frame;
  the chunk is a P-frame. 272: project the context into the tokenizer's token space with a light
  module rather than retraining anything big. 69: the VQ-VAE as a bridge to a sequence model.
- **Mechanism.** The brief's fact 7 states the root structural problem: *the tokenizer conditions on
  nothing*, so every token's identity must be in the codes, including the ~60% of Python tokens that a
  192-token prefix nearly determines (copied identifiers, closing brackets, indentation, `self`,
  `return`). Each such token currently costs full fine-level nuisance bits. If encoder and decoder both
  see the prefix tail, the codes only have to carry `chunk - prediction(prefix)`; predictable positions
  can take a constant "null" code that the generator predicts at ~0 bits. The generator sees the same
  prefix at inference, so nothing is hidden from it. This is a genuinely different mechanism from
  Proposal 1 (which keeps the information but makes the map smooth): it removes information from the
  codes that the decoder can recover elsewhere, at every level, and it composes with Proposal 1. On its
  own, without a rate term, it would do little: free capacity means the encoder keeps encoding
  everything anyway. So run it *on top of* Proposal 1.
- **Exact change.**
  1. Train on prefix+chunk windows (train.py already does this in the plan-weight branch).
  2. *Decoder conditioning:* feed the last 64 prefix tokens through the codec's own `encoder.wte`
     (+ a learned "context" segment vector), concatenate them in front of the 64 `z_q` positions, run
     the existing 2 decoder blocks over the 128 positions with full attention, read logits at the last
     64. A thin wrapper around upstream `Decoder`; `codec.decode(idx)` becomes `decode(idx, prefix)`.
  3. *Encoder conditioning:* same for the encoder: 64 prefix embeddings prepended, codes taken from the
     chunk's 64 positions before `_down_for_quant`. (Needed so the encoder knows what is predictable; a
     cheaper variant conditions only the decoder and lets the rate term from Proposal 1 do the rest.)
  4. Rate term exactly as Proposal 1; the proxy already has the prefix for levels 0-2.
  5. `codec_eval`, `load_codec`, `ncm_logits`/`full_generate` and `evaluate.py` pass the prefix to
     `decode`. `require_codec_quality` is unchanged in its thresholds.
- **Why it is new vs "already tried".** Nothing tried changed what the tokenizer sees. Path A fed the
  prefix to a side predictor; the codec's encoder and decoder stayed unconditional. The upstream
  `ctx_encoder` option (N6 in the plan) is a *generator*-side compressed context, not tokenizer
  conditioning.
- **One-night test.** Same pipeline as Proposal 1 (codec 5,000 steps ~3 h with the 128-position
  decoder, NCM 1,800 steps ~2.2 h, bits 10 min). Gate: decode(prefix, codes) >= 99% tokens, >= 50%
  exact. Also report, as a diagnostic, the fraction of fine-level positions whose code equals the
  level's most frequent code (expected to rise from ~0 to 30-50%). Expected extra bits: 350-550
  (together with Proposal 1), i.e. total ~600-800.
- **Cheat risk.** (a) The decoder could become an in-chunk language model that guesses easy tokens
  from the prefix and ignores the codes; that is the intended trade and the gate decides whether it
  went too far (the gate is on codes+prefix -> exact chunk). (b) Fairness: the AR control also sees
  the 192-token prefix, so the comparison stays like for like; state this in the write-up. (c) A
  decoder that *copies* prefix tokens into the chunk when codes are wrong would show up as a quality
  regression, not a bits gain; keep the step-6 quality test as a second gate.
- **Confidence it passes the 30% test: 40% alone (55% combined with Proposal 1).** The information
  argument is solid; the risk is engineering (a conditional decoder in a 2-layer codec) and that the
  2-layer decoder exploits far less prefix redundancy than the 7-layer AR model does.
- **GPU cost on the T1000:** ~6 h.

---

### 3. Shift-equivariant local encoder: one code per local pattern, at any offset

- **Atoms used:** row 14 (66, 124, 201), row 7 atom 193. 201 (replace absolute positions by relative,
  distance-aware ones so the representation of a relation does not depend on where it sits) is the
  direct fix for fact 6; 124 (audio tokenizers: causal/convolutional encoders with RVQ, which are
  shift-equivariant by construction and are the ancestors of NCM's quantizer); 193 (attention masks as
  structural priors: a windowed mask makes the encoder local).
- **Mechanism.** The codec encoder uses an absolute `wpe` over the 64 chunk positions plus full
  bidirectional attention, so "def foo(" at chunk offset 0 and at offset 17 get unrelated codes: the
  generator must learn 64 different code maps, one per absolute position, from 1,800 x 32 windows.
  Fact 6 (100% of level-0 codes change on a 1-token shift) is the symptom. With no absolute position
  in the encoder, a windowed attention (|i-j| <= 4) or conv stack, the fine code at position j
  becomes a fixed function of tokens j-4..j+4 wherever the chunk starts: the same local pattern
  always gets the same code, so every training window teaches all 64 positions at once (a ~64x
  increase in effective training signal for the code map), and the generator, which already uses
  RoPE, can reuse what it learns across positions. This attacks the learnability gap of F-A directly
  and mostly at the fine levels (F-B), where the function becomes a local n-gram map instead of a
  hash of the whole chunk.
- **Exact change** (config-level in `build_codec`, no upstream edits): `enc_positional=False`,
  `enc_attn_window=4` (upstream `SelfAttention` supports it), optionally `enc_conv_layers=2,
  enc_conv_kernel=5`; `dec_positional=False` (upstream's own default; the decoder reads positions from
  the sequence order of z_q). Level 0's `learned_down_l1` is a stride-16 conv aligned to chunk bins,
  so level 0 stays equivariant only modulo 16; keep it (fact 3: coarse levels decide quality) and
  expect the level-0 shift-change rate to fall from 100% toward the ~1/16-content-change floor rather
  than to zero. Optional "encode in context": prepend the last 8 prefix tokens to the encoder input
  and discard their positions before downsampling, so left-boundary codes are not boundary artefacts.
- **Why it is new vs "already tried".** The tried level layouts and codebook sizes all kept the
  absolute-position, full-attention encoder; shift stability was only *measured* (A2.3) and listed as a
  target, never engineered. The plan's optional "agreement loss between a window and its 1-2 token
  shift" is a soft version; equivariance by construction is exact and costs nothing.
- **One-night test.** codec-eq 5,000 steps (~1.2 h; it is the cheapest arm), gate as usual, then the
  A2.3 shift test (10 min): level-0/1/2 code change under a 1-token shift, target <= 50% at level 0 and
  <= 25% at levels 1-2. If the gate and the shift target hold, NCM 1,800 steps (~2.2 h) and bits.
  Expected extra bits: 650-800 alone (the nuisance remains, but it becomes a consistent local
  function); 30% chance of crossing 660 alone, clearly higher stacked under Proposal 1.
- **Cheat risk.** Low: this changes no objective and removes no information. The one way it can look
  good falsely is if the windowed encoder *fails* the gate and the fine levels degrade into a
  low-entropy, lossy code; the gate catches that. Also check that `codes_used` at levels 0-2 does not
  collapse (train.py already logs it).
- **Confidence it passes the 30% test: 30%.** Mechanism is sound and the fix is exact, but it does
  not remove nuisance variance, only makes it consistent; it is the natural partner of Proposal 1.
- **GPU cost on the T1000:** ~3.5 h (1.2 + 2.2 + 0.2).

---

### 4. Geometry-aware generator head with soft (distance-based) targets

- **Atoms used:** row 4 (8, 60, 223), row 10/12 atom 188. 8 (Optimal Transport: score by the
  structural distance between elements rather than exact point-wise overlap) -> a near-miss code should
  cost less than a far miss; 60 (continuous relaxation of discrete tokens, Voronoi projection) -> the
  tokenizer's soft assignment is a natural target; 223 (small per-level low-rank adapters on a shared
  core) -> per-level 384->256 projections instead of per-level 2048-way heads.
- **Mechanism.** The generator's output head is a single 384x2048 matrix shared by all seven levels
  (`shared_output_head`, `shared_head_per_level_bias`), even though the seven codebooks are unrelated:
  code 37 at level 0 and code 37 at level 5 share one output vector, disambiguated only by a bias and
  the level encoding. Within a level, the index is an arbitrary labelling of 2048 centroids in a
  256-d space, and cross-entropy treats neighbouring centroids as unrelated classes, so an
  "approximately right residual" earns nothing. At the fine levels (F-B) the residual *direction* is
  partly predictable even when the exact cell is a hash; a head that scores codes by distance to a
  predicted vector turns that into mass spread over the right neighbourhood, which is fewer bits than
  a confident wrong class. Training on the tokenizer's soft assignment `q(k|z) = softmax(-d/tau)`
  instead of the hard one-hot gives the same geometric partial credit on the target side.
- **Exact change** (generator only; `build_ncm` + a new head module): replace `lm_heads` by, per
  level l, `h -> W_l h` (384->256, 7 x 0.1M params, well under the 5.5M that seven full heads would
  cost) and `logits_lk = -||W_l h - E_l[k]||^2 / s_l + b_lk` with `E_l` the frozen codec codebook
  (`codec.vq.embeddings[l].weight`), `s_l` a learned per-level temperature, `b_lk` a per-level bias.
  Loss: cross-entropy against `q_l` computed once per batch from the frozen codec's distances at
  tau = median NN distance (precompute in `ncm_logits`'s no_grad block; `z_to_idx` already exposes
  the distances), with the hard-code CE still reported as the bits metric. `_MaskedHead` and
  `install_generator` are unaffected.
- **Why it is new vs "already tried".** Every tried change kept the shared linear head and hard
  targets. The brief's tried list is about attention patterns, codebook sizes, reveals and a tokenizer
  loss; none touched how the generator scores codes.
- **One-night test.** NCM 1,800 steps on codec-l7 (~2.2 h), bits with nothing shown (10 min). Fast,
  tokenizer unchanged, so it also isolates the "arbitrary indexing" hypothesis: if the head alone cuts
  >= 100 bits, the hypothesis is real and the head should be used in Proposal 1's proxy and every later
  generator; if it cuts < 30 bits, the code map is a hash even in latent space and only Proposals 1-3
  can help. Expected extra bits: 750-850 alone (pass unlikely alone).
- **Cheat risk.** None on information (the tokenizer is untouched); the gate is already passed by
  codec-l7. The only measurement trap is reporting the soft-target loss as the bits; the bits must stay
  the hard-code cross-entropy on held-out chunks.
- **Confidence it passes the 30% test: 20% alone.** It is the cheapest and most diagnostic arm, and
  its value is mostly as a component of Proposal 1 (the proxy head) and as a hypothesis test.
- **GPU cost on the T1000:** ~2.5 h.

---

## Rejected from the combos (with the arithmetic)

- *Semantic distillation of the coarse levels to the frozen AR's in-chunk states* (atom 32 in six
  rows; SpeechTokenizer's recipe). Attractive for quality (smooth, rename-robust coarse codes), but
  levels 0-2 hold 248 bits total and the test needs -280; worse, moving token identity out of level 0
  pushes it into levels 3-6, raising the bits that dominate. Worth revisiting only after Proposal 1
  has drained the fine levels.
- *Order-agnostic diffusion / iterative refinement / bidirectional masking* (rows 1, 17, atoms 156,
  185, 66): fact 5 rules out within-level coordination as the cause (0.6% of the extra bits).
- *Speculative decoding* (atom 139): changes latency, not the code distribution.
- *Formal grammars / ASTs as the code alphabet* (atoms 22, 253): a syntax-aware chunking would help
  quality, but it is not a tokenizer change that keeps the 64-token residual-VQ setup comparable to the
  baseline bits, and the brief's "line-boundary chunking" backup already covers the cheap version.

## Suggested order for one GPU
Night 1: Step 0 diagnostic (30 min) -> Proposal 1 codec (with Proposal 4's head inside the proxy) ->
standard NCM -> bits. Night 2, if night 1 is >= 15% but < 30%: Proposal 1 + 2 (conditional tokenizer).
Proposal 3 is a 1.2 h codec run that can be slotted in any afternoon and stacked under either.
