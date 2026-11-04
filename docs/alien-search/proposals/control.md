# Proposals: make the tokenizer's codes predictable from the prefix

Written from the brief plus a read of `code_scale/models.py`, `plan_loss.py`, `train.py`, `reveal_bits.py`
and upstream `vqvae.py` / `quant.py` / `vae.py`. No training run, no external sources.

## The diagnosis the proposals rest on

- The per-level **blind-guess** (marginal) entropies measured in `runs/code-scale-v1/ncm-l7/plan-probe.json`
  are 8.2 / 9.5 / 10.8 / 10.7 / 10.2 / 10.6 / 10.1 bits per code. Summed over 236 positions that is
  ~2,430 bits of marginal entropy per chunk, while the joint entropy of the code tuple is at most the text's
  (~250 bits given the prefix, a few hundred unconditionally). The codes are therefore hugely **redundant with
  each other**, but the redundancy is laid out like a hash: level 5 has a 10.6-bit marginal yet NCM predicts it
  to 1.9 bits (it is almost a deterministic function of levels 0-4), while level 6 stays at 3.9 bits.
- Why hash-like: the encoder is 2 blocks of **full bidirectional attention over the 64 tokens**, plus an
  **absolute position table** (`enc_positional=True`) and a **GroupNorm over (channels x all 64 positions)**
  (`pre_quant_groupnorm=8`, `nn.GroupNorm` on the (B, C, L) layout). So every code depends on every token
  and on the exact window offset. The residual VQ then slices this latent into 2,048 Voronoi cells per level
  with randomly initialised EMA codebooks: cell labels carry no structure, and nothing in the loss asks the
  encoder to make a cell choice that a prefix reader could anticipate. Facts 5-7 follow directly: reveals
  within a level add nothing because the other codes are related to the target only through the global
  scramble (fact 5); a 1-token shift changes every pooled latent (fact 6); the probe finds ~0 bits (fact 7).
- Consequence for a fix: the gap (940 bits) is **modelling error, not entropy**. Two levers can shrink it:
  (a) change *which* cell the encoder picks when several reconstruct equally well, so the choice is the one a
  prefix-aware prior expects (proposals 1 and 2), or (b) change the *geometry* of the text-to-code map so the
  13M generator can learn it from 57k chunks: local, shift-equivariant, structured labels (proposals 3, 4).
  Path A (InfoNCE on the level-0 latent) touched neither: it was a contrastive proxy, capped at log2(32) = 5
  bits, on 33 of the 1,190 bits, with no gradient path to levels 1-6 (the quantizer works on `z.detach()`).

Costs below use the clean-machine timings in `runs/code-scale-v1/*/summary.json`: codec 5,000 steps ~2 h
(codec-pp05: 7,156 s), NCM 1,800 steps ~2.2-3.8 h (ncm-rv7: 7,831 s; ncm-l7: 13,628 s), bits measurement
(`reveal_bits` 1-pass or `plan_probe --gain-only`) ~0.4 h.

---

### 1. Entropy-constrained residual VQ with a prefix-conditioned prior (rate-distortion tokenizer)

- **Mechanism.** Today the quantizer picks `argmin_i d_i` (squared distance) at every level. Replace it with
  the ECVQ rule `argmin_i [ d_i + lambda * (-log q_i) ]`, where `q` is a small *learned prior* over the
  2,048 codes that sees exactly what NCM will see: the prefix and the coarser levels already chosen. The
  encoder's latent does not have to change for the codes to become predictable: whenever two cells are
  nearly equidistant (the common case given ~2,430 marginal bits for ~300 joint bits of content), the
  cheaper one under the prior wins, the small extra residual is absorbed by the finer levels, and the fine
  levels face the same pressure, so the "null"/expected code becomes the default wherever reconstruction
  does not need otherwise. This attacks facts 4-5 head on: the measured quantity (bits of the codes given
  prefix and earlier levels) *is* the quantity being minimised, subject to the reconstruction term. The
  commitment loss `mse(z_hat.detach(), z)` additionally pulls the encoder toward the cheap cells, so the
  latent geometry itself drifts toward predictable codes over training. Codes become a function of
  (chunk, prefix); the **decoder still sees only codes**, so nothing about generation changes.
- **Exact change** (all in a subclass of `MultiscaleResidualVectorQuantizer` used by `build_codec`, plus
  `train.py codec`):
  1. `vq.forward(z, prior_logits=None)` and `z_to_idx(z, prior_logits=None)`: at level `i`, if
     `prior_logits[i]` (shape B x level_i x 2048) is given, `cost = distances + lambda * (-log_softmax(prior_logits[i]))`
     and `indices = argmin(cost)`. EMA/codebook updates, residual subtraction, `quant_resi`, fine-drop: unchanged.
  2. Prior network `CodePrior` (new file `code_scale/code_prior.py`, ~0.5M params): input per (level i,
     position j) = concat[ frozen AR prefix features (`plan_loss.ar_prefix_features`, 768-d, as Path A v1),
     `F.interpolate(z_hat_{<i}, size=level_i, mode="area")[:, j]` (the same tensor `idx_to_ncm_input`
     feeds NCM; zeros for level 0), learned level+position embedding ] -> LayerNorm -> 512 -> GELU -> 2048
     logits. Trained jointly with CE on the chosen indices (inputs detached). Its validation CE per level is
     the **early readout**: it is a lower-capacity stand-in for NCM's own cross-entropy.
  3. Loss = rec + vq_loss + beta * prior_CE (beta = 1, the prior gets its own gradient only; the encoder is
     affected only via which indices are chosen and the commitment term). `lambda` is the single knob:
     scale it so that at step 0, `lambda * 11 bits * ln 2` equals the median gap between the best and
     second-best distance at level 3 (log this gap in the first 20 steps and set lambda from it; sweep x1
     and x4). Optionally anneal lambda from 0 over the first 300 steps so codebooks are not dragged while
     the prior is still uniform.
  4. `codec.encode(x, prefix)`; NCM training (`train.py ncm`, `ncm_logits`, `reveal_bits`, `plan_probe`,
     `evaluate`) passes the window's prefix to `encode`. The AR prefix read costs ~1.2 s per 64 prefixes
     (Path A measurement), so apply ECVQ to 32 of the 64 windows per step as Path A did, or cache AR
     features (`runs/code-scale-v1/ar-prefix-feats-*.pt` already exists for this purpose).
- **Why new.** Path A was InfoNCE between prefix features and the *level-0 latent* (bounded by 5 bits per
  code, level 0 only, no gradient to other levels, no change to the assignment rule). Path B2 shrank
  codebooks uniformly, which reduces *marginal* entropy at every context equally (and lost information:
  1,008 bits for less reconstruction headroom). ECVQ reduces *conditional* entropy, keeps the full 2,048
  codes available for genuinely new content, acts on all seven levels, and needs no gradient through the
  argmin. Nothing in the tried list changes the encoder's cell choice.
- **One-night test.** Warm start from `codec-l7/final.pt` (keeps reconstruction, saves ~1.5 h): fine-tune
  2,000 steps with ECVQ, lambda = x1 and x4 in two runs (~0.9 h each incl. the AR prefix read). Gate:
  `codec_eval` >= 99% tokens, >= 50% exact chunks. Go/no-go before NCM: prior CE summed over the chunk on
  val <= 700 bits (it must be, since NCM has strictly more context than the prior). Then NCM 1,800 steps on
  the better run (~2.5 h), bits with `reveal_bits` 1-pass. Expected: 480-650 extra bits over AR (the
  prior's own CE plus NCM's usual ~10% slack). Pass line <= 660.
- **Cheat risk.** With lambda too large the encoder always picks the prior's favourite, codes carry nothing,
  reconstruction collapses: the gate catches it (watch `codes_used_l0_l2` and `chunk_exact` every 500
  steps; abort the lambda if token accuracy drops below 99.3% mid-run). A subtler one: codes depend on the
  prefix, so evaluate reconstruction with the *same* `encode(x, prefix)` on held-out windows, decode with
  codes only; if someone decodes with the prefix that is proposal 2, not this one. There is no way for the
  prefix to leak into the decoder here.
- **Confidence it passes the 30% test: 50%.** It optimises the measured quantity directly and ECVQ is the
  textbook way to buy rate with negligible distortion; the risks are optimisation (EMA codebooks being
  dragged by rate-chosen assignments, the lambda scale) and the prior being too weak at levels 2-4.
- **GPU cost (T1000):** 2 x 0.9 h codec + 2.5 h NCM + 0.4 h bits = ~4.7 h.

---

### 2. Conditional codec: encoder and decoder both read the prefix, codes carry only the innovation

- **Mechanism.** Fact 7's "codes depend only on the chunk" means the tokenizer must spend codes on content
  the prefix already determines (closing indentation, the enclosing class/function names, repeated
  identifiers, import aliases). That content is predictable in principle, but it reaches the generator only
  through the hash-like code map, so NCM pays for it twice over. Give the codec the prefix: the encoder's
  attention from chunk positions can look at the prefix, the decoder reconstructs the chunk from
  (prefix, codes). Content the decoder can read off the prefix no longer needs a non-null code, so the
  codes' *content* shrinks toward H(chunk | prefix) ~ 250 bits and so does the number of distinct code
  patterns the generator must learn. The modelling-error ratio (1,190 / ~400 ~ 3x today) then applies to a
  smaller base. It also directly removes the "fingerprint" component of facts 6-7: a code's meaning becomes
  "what differs from what the prefix implies", which is what the generator is asked to produce anyway.
- **Exact change** (`build_codec`, `VQVAE.forward/encode/decode`, `train.py codec`, and the NCM decode path):
  1. Codec prefix length `Pc = 64` tokens (the last 64 of the 192; local context gives most of the
     redundancy and keeps the encoder/decoder attention at 128 positions, ~2x the FLOPs of today's 64).
  2. Encoder: input `[prefix tokens | chunk tokens]` through the shared `wte`, a learned 2-slot segment
     embedding (prefix/chunk) added to `wpe`, the 2 encoder blocks over 128 positions with the mask
     "chunk attends to prefix and chunk; prefix attends to prefix only" (so prefix states do not depend on
     the chunk). Take the 64 chunk positions -> pre-quant norm -> VQ unchanged (7 levels, 2,048).
  3. Decoder: input `[wte(prefix) + wpe | z_q + wpe]` with the same mask, 2 blocks, output head on the 64
     chunk positions only. Reconstruction CE on the chunk only.
  4. `codec.encode(x, prefix)` / `codec.decode(idx, prefix)`; `evaluate.py` / step-6 decoding passes the
     generation prefix to `decode`. `idx_to_ncm_input` and the generator are unchanged.
- **Why new.** No tried item lets the tokenizer see outside the chunk. Path A fed prefix *features* to a
  side loss, not to the encoder or decoder; the codes still had to carry the whole chunk. B1/B2/C changed
  the generator or codebook size only.
- **One-night test.** Train the conditional codec 5,000 steps from scratch (~2.5 h; a warm start is not
  possible because the decoder input changes). Gate: `decode(encode(x, p), p)` on 1,024 held-out windows,
  >= 99% tokens, >= 50% exact. Secondary readouts that cost nothing: the blind-guess entropy per level
  (`plan_probe --probe-only`) should fall well below 8.2/9.5/10.8 bits at levels 0-2, and the share of
  positions whose level 4-6 code is the most frequent ("null") code should rise. Then NCM 1,800 steps
  (~2.5 h), bits as usual. Expected: 650-800 extra bits; pass needs <= 660, so this is a coin flip on its own
  and a strong partner for proposal 1 (ECVQ on a conditional codec).
- **Cheat risk.** The decoder could learn to ignore the codes and emit prefix-conditioned guesses; that
  fails the exact-chunk gate (50%) immediately, because new content in the chunk is not guessable. The
  reconstruction gate must be measured with the *generation-time* prefix, never with the chunk itself in
  the decoder's context; the mask in step 2 guarantees prefix states cannot see the chunk, so decoding
  cannot peek. Also report the gate on the plain `encode/decode` path for the old codec so the two numbers
  are compared like for like.
- **Confidence: 30%.** The direction is sound (conditional entropy coding) and it compounds with 1, but the
  bits saved depend on how much of the 940-bit error sits on prefix-redundant content, which we have not
  measured; the decoder's extra context also slows the codec ~2x.
- **GPU cost:** 2.5 h codec + 2.5 h NCM + 0.4 h = ~5.4 h.

---

### 3. Local, shift-equivariant encoder (no absolute positions, no cross-position norm, windowed context)

- **Mechanism.** Facts 5 and 6 say the code at (level k, span j) is a function of all 64 tokens and the
  exact window offset. Three components make it so: the encoder `wpe`, the GroupNorm over all positions,
  and full attention. Remove them and each code becomes a function of the tokens in its own span plus a
  few neighbours, invariant to where the chunk starts. Then the coarse-to-fine map is *local and shared
  across positions*: "coarse code at span j" -> "fine codes inside span j" is the same small function
  everywhere, learnable from 57k chunks (the generator already has RoPE and a conv post-upsample, both
  built for local structure). And the prefix predicts the first spans directly because they are no longer
  entangled with the chunk's tail. The decoder keeps its `wpe` (it needs to place tokens), so
  reconstruction capacity is unchanged: 7 x 11 bits per position is far more than the ~13 bits a token
  needs.
- **Exact change** (`build_codec` only; everything is an existing upstream switch except the norm layout):
  `enc_positional=False`; `pre_quant_norm_type='layer'` (per-position `LayerNorm(affine=False)`, already
  supported in `VQVAE.__init__`); `enc_conv_layers=2, enc_conv_kernel=3` (relative order inside a span
  without a position table); `enc_attn_window=8` (upstream `SelfAttention._window_mask`, |i-j| <= 8) with
  `enc_layers=2`. Variant 3b (second night if 3a fails the gate or is flat): `enc_layers=0`, conv-only
  encoder, strictly local. Variant 3c: a causal window (mask `0 <= i-j <= 16`) so code j depends on tokens
  <= its span, which also makes the within-level left-to-right reveal order of Path C meaningful.
- **Why new.** Path C revealed codes within a level but on the *global* codes, where neighbours carry no
  local information (hence 0.6%). B1 changed the generator's attention, not the encoder's. No tried item
  changed the encoder's receptive field, positions or normalisation. The "level layouts" tried kept the
  same global encoder.
- **One-night test.** Codec 5,000 steps (~2 h; slightly faster than today, attention is windowed). Gate as
  usual. Cheap readouts before NCM: `plan_probe --probe-only` shift stability (level-0 codes changed by a
  1-token shift: today 100%; expect <= 60% since a shift moves 1 of 16 pooled tokens), and the probe's
  level-0 gain (expect >= 0.5 bits). Then NCM 1,800 steps, bits. Expected: 700-850 extra bits (largest
  drop at levels 2-4 where the coarse-to-fine map now has local structure). This is the proposal most
  likely to help *quality* even if bits only reach the 20-25% range, because syntax errors are local
  coordination failures.
- **Cheat risk.** None by construction: the code count and codebooks are unchanged, information can only be
  lost through worse reconstruction, which the gate measures. Watch one failure mode: without a position
  table the encoder may lose the within-span order of repeated tokens (`a, a` vs `a a,`); the conv layers
  are there for that, and `chunk_exact` will show it.
- **Confidence: 30%.** Locality is the standard reason AR tokens are easy to model and these codes are not,
  but the pooled coarse latents of 16 arbitrary tokens remain high-entropy objects, so the coarse levels
  may stay near-uniform and only levels 3-6 improve.
- **GPU cost:** 2 h codec + 2.5 h NCM + 0.4 h = ~4.9 h.

---

### 4. Structured code labels: residual finite-scalar quantization with factorised generator heads

- **Mechanism.** A 2,048-way EMA codebook gives the generator nominal labels: being "almost right" in
  latent space earns nothing, every miss costs the full surprise, and the labels drift during codec
  training. Replace each level's codebook with FSQ: project the 256-d latent to 4 dims, `tanh`, round each
  dim to [8, 8, 8, 4] bins (= 2,048 codes, so `SHAPE["codebook"]` and the shared head are untouched),
  project back. The code index is now a 4-digit mixed-radix number of a smooth low-dimensional coordinate.
  Nearby latents share most digits, so the generator can be partly right; a 1-token shift moves one or two
  digits instead of relabelling the cell (fact 6); and because the per-level map is a fixed, known
  function of a bounded coordinate, the coarse-to-fine residual structure is smooth rather than hashed
  (facts 4-5). With a factorised head (four small softmaxes per position, code length = sum of digit CEs,
  a valid prefix code), the generator's 2,048-way problem becomes four 4-to-8-way problems over ordered
  bins.
- **Exact change.** New `FSQLevel` module replacing `vq.embeddings[i]` + EMA buffers in a subclass of the
  quantizer: `down: Linear(256, 4)`, `up: Linear(4, 256)`, bins `[8, 8, 8, 4]`, straight-through
  rounding, `idx = digits -> int` and `idx -> digits -> up` for `idx_to_z_q`/`idx_to_ncm_input`. Delete the
  EMA update, restart and commitment terms for FSQ levels (keep `mse(z_hat, z_no_grad)` on the sum).
  Generator: `shared_output_head` replaced by per-level heads of 8+8+8+4 = 28 logits; `ncm_loss` sums the
  four digit CEs; `_sample_level` samples digits; keep a 2,048-way joint head in a second run as the
  like-for-like measurement. Levels 0-6 all FSQ (a mixed variant with FSQ at levels 0-3 only is the
  fallback if the fine levels lose reconstruction).
- **Why new.** B2 made codebooks *smaller*; this keeps 2,048 codes per level but gives them geometry. No
  tried item changed the quantizer family or the label structure.
- **One-night test.** Codec 5,000 steps (~2 h; FSQ is cheaper than EMA VQ). Gate as usual (FSQ at 4 dims
  may be the gate's tightest spot; if `chunk_exact` < 50%, raise to 5 dims [8,8,4,4,2] in the second
  run). Readouts: shift stability (expect a large drop in *digits* changed), blind-guess entropy per
  digit. NCM 1,800 steps with factorised heads, bits = summed digit CE. Expected: 850-1,000 extra bits;
  this proposal alone is unlikely to reach 660, but it is the natural quantizer for proposals 1-3 (ECVQ on
  FSQ is a per-digit bias; locality plus smooth labels compounds).
- **Cheat risk.** Factorised bits assume the digits of one code are conditionally independent given the
  context; that is a valid (upper-bound) code length, so it cannot under-count. The 4-d bottleneck could
  lose information: the gate catches it. Do not compare factorised bits against the 2,048-way baseline
  without also reporting the joint-head run.
- **Confidence: 20%.** The structure helps the generator's optimisation, but the entropy of what the coarse
  levels encode is unchanged, so the gain is mostly at the fine levels where NCM already does best.
- **GPU cost:** 2 h codec + 2.5 h NCM + 0.4 h = ~4.9 h (plus 2.5 h for the joint-head control run).

---

## Order and combination

Run 1 first: it is the only one whose early readout (the prior's CE) says before NCM training whether the
night will pass, and it is the only one that optimises the measured quantity itself. If 1 passes the gate
but lands at 660-800 bits, the next night is 1 + 2 (ECVQ on a conditional codec): the two mechanisms are
orthogonal (which cell vs. what content). 3 is the one to run if the quality metrics, not the bits, are the
blocker. 4 is a quantizer swap to adopt only underneath 1-3.

## Summary
1. Entropy-constrained residual VQ with a prefix-conditioned prior: 50%
2. Conditional codec (encoder and decoder read the prefix): 30%
3. Local, shift-equivariant encoder (no wpe, LayerNorm, windowed/conv): 30%
4. Residual FSQ with factorised generator heads: 20%
Recommended first night: proposal 1 (~4.7 h), go/no-go from the prior's CE before NCM training.
