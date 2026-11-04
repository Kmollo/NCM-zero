# Proposals (fable_3): making NCM's codes predictable from the prefix

Source material: `../brief.md`, its 20-row combination list (20 rows), `code_scale/models.py`,
`code_scale/train.py`, `code_scale/plan_loss.py`, `code_scale/reveal_bits.py`, `code_scale/plan_probe.py`,
`upstream/ncm/models/{quant,vqvae,vae,next_concept}.py`, `docs/ncm-next-plan.md`, `docs/code-scale-v1-results.md`.
Atom texts were treated as reference data only.

## The diagnosis the proposals rest on

The codes are a deterministic function of the 64-token chunk. Under NCM's own factorisation (levels in order,
positions within a level independent given the earlier levels) the ideal code length is
H(text | prefix) + sum over levels of the within-level multi-information. Fact 5 measured the second term as
about 5 bits (3 and 8 passes change nothing). So the ideal is roughly 255 bits, the generator pays 1,191, and
the 940 extra bits are not entropy the codes carry; they are KL, i.e. conditionals a 13M model cannot learn.

Why they are unlearnable, from the code read:

- Each code is `argmin_c ||r - e_c||^2` over a near-uniformly used 2,048-entry codebook in 256 dimensions
  (`quant.py: forward/z_to_idx`). In 256-d the nearest and second-nearest centroids are almost equidistant, so the
  assignment is decided by tiny perturbations. Fact 6 (100% of level-0 codes flip under a one-token shift) is
  this, amplified by the encoder's absolute position table (`Encoder.wpe`, `enc_positional=True`), two layers of
  full bidirectional attention, and a GroupNorm whose statistics mix all 64 positions (`pre_quant_groupnorm=8`).
- Level k quantises the *quantisation error* of levels < k (`z_rest -= quantized`). The fine codes therefore
  index noise created by the coarse codebook boundaries, not anything about the text. L6 alone costs 284 bits,
  more than AR spends on the whole chunk; revealing neighbours does not help (fact 5) because that noise is
  decorrelated by construction.
- Predicting such a code from the prefix means pushing p(text | prefix) through a hash. The AR model never has
  to do this: its "code" is the token.

A fix must therefore do one of two things, and ideally both: (A) let the encoder choose, among the many
near-equidistant codes that reconstruct equally well, the one a prefix-conditioned model can predict (this turns
boundary noise into predictability without losing information); (B) make the text-to-code map smooth and local
so the push-forward of the language distribution is learnable. Path B2 (small codebooks) did neither: it made
every code cheaper uniformly, which is why bits fell by 120 while quality did not move. Path A (InfoNCE on the
level-0 latent, capped at log2(32) = 5 bits of signal by construction) touched 33 of the 1,191 bits.

Rejected on these grounds before writing: dropping level 6 (a layout change; it saves bits only by removing
information and the gate would likely fail), more MaskGIT passes (fact 5), any loss on levels 1-3 through the
upstream quantiser's `z.detach()` path (gets no gradient; the proposals below route around it explicitly).

Common measurement for every proposal (fits the existing scripts):
codec 5,000 steps -> `codec_eval` gate (>= 99% tokens, >= 50% exact on 1,024 held-out chunks) ->
`plan_probe --probe-only` (15 min: blind guess, probe gain, shift stability) -> NCM 1,800 steps on the same
windows/seed -> `plan_probe --gain-only` for `ncm_total_bits_per_chunk` on held-out windows; AR = 250.5 on the
same windows. Pass: NCM total <= 910.5 (extra <= 660). The reconstruction gate is always measured with the
real decoder on `decode(encode(x))`.

---

### 1. Entropy-constrained residual VQ with a prefix-conditioned code prior (ECVQ-NCM)

- Atoms used: 26 (variational inference: the rate term is a KL to a *conditional* prior; rows 9, 13),
  81 (information bottleneck; rows 2, 7, 13), 122 (conditional VQ-VAE; row 1), 102 ("marginalising over equally
  valid tokenisations", i.e. the downstream model pays for the tokenizer's arbitrary pick among equivalent
  segmentations; rows 12, 15, 20), 188 (soft assignment relaxation, optional v2 only; rows 5, 16).
- Mechanism: Classical entropy-constrained VQ (Chou, Lookabaugh, Gray 1989) chooses the code by
  distortion + lambda * rate instead of distortion alone. Here the rate is conditional: -log q_k(c | prefix,
  codes of levels < k), with q_k a small prior that sees exactly what the NCM generator sees for level k (prefix
  features and the area-pooled coarse reconstruction `idx_to_ncm_input` gives). Ties to the facts: (fact 6 and
  the 256-d geometry) most residuals sit within a sliver of several centroids, so the encoder can switch to a
  prior-favoured code at almost no distortion; (residual structure) any distortion introduced at level k is
  re-encoded by levels k+1..6, so coarse-level switches are nearly free for reconstruction and the gate should
  hold with lambda_6 = 0; (fact 5) the within-level term stays tiny, so a per-position prior is the right form and
  its cross-entropy is an achievable code length for a generator of NCM's shape; (fact 7) at level 0 the prior
  knows little, so the gain there is small (it degenerates to "prefer frequent codes", the Path B2 effect); the
  bulk of the 940 bits sits at levels 2-4 and 6, where the prior conditions on coarser levels and can be sharp.
  Rough size: if ~20 codes are within one typical distance gap and the generator's conditional has ~200
  effective codes (7.7 bits at L3), the best of 20 costs about 7.7 - log2(20) = 3.4 bits, saving ~4 bits/code at
  L3 (130 bits), ~3 at L4 (140), ~4 at L2 (60). The commitment term `MSE(z_hat.detach(), z)` then pulls z toward
  the chosen code, so the encoder itself drifts toward predictable latents (co-adaptation, as in learned
  compression where encoder and entropy model are trained together).
- Exact change (code_scale only, upstream untouched):
  1. `code_scale/ecvq.py`: `CodePrior(nn.Module)` with one head per level k: input = [prefix features
     (frozen AR last+mean, 768-d, exactly `plan_loss.ar_prefix_features`), area-pooled z_hat_<k at resolution
     L_k (256-d, zeros for k = 0), learned position embedding (64-d)] -> LayerNorm -> Linear(1088, 512) -> GELU ->
     Linear(512, 2048). About 0.6M params per level.
  2. `ecvq_forward(vq, z, prior_fn, lam)` and `ecvq_z_to_idx(...)`: copies of the upstream loops in
     `quant.py` with one line changed after `distances`: `scores = distances.view(B, L_k, V) +
     lam[k] * (-log_softmax(prior_fn(k, pooled_zhat), -1))`; `indices = scores.argmin(-1)`. EMA updates,
     restart fix, quant_resi and fine_drop unchanged. Installed like `fixed_restart` (monkeypatch on `codec.vq`).
  3. `lam[k] = alpha * gap_k`, where gap_k = median over a warm-up batch of (second-nearest - nearest squared
     distance) at level k; alpha in {0.5, 2}; lam = 0 for steps 0-500, linear ramp to full over steps 500-1,000;
     lam[6] = 0 in the first run (no cascade below L6 to absorb its distortion).
  4. `train.py codec --ecvq-alpha A`: windows are prefix+chunk (as in the Path A branch); AR prefix features
     cached once for 100k windows (31 min, 154 MB fp16) and cycled; loss = rec + vq_loss + CE(prior logits,
     chosen indices.detach()) summed over levels (trains the prior only; the encoder is moved by reconstruction
     and commitment, v1 has no straight-through term). Log the prior's per-level CE on 2,048 val windows every
     500 steps (the early read, see test).
  5. `codec.encode(x, prefix_feats)` path (used by `train.py ncm`, `plan_probe`, `evaluate`): `ecvq_z_to_idx`
     with the prior, level by level. NCM training computes AR features online for its 32 prefixes (+0.6 s/step).
     The generator, `idx_to_ncm_input`, the decoder and generation are unchanged: codes now depend on
     (chunk, prefix) but decode from codes alone.
- Why it is new vs "already tried": Path A changed the encoder's *latent* at level 0 through a contrastive loss
  whose signal is capped at 5 bits; this changes the *assignment rule* at every level using the actual code
  cross-entropy under the generator's own conditioning. Path B2 shrank codebooks (uniform rate cut, no context);
  this keeps 2,048 codes and spends them non-uniformly per context. Nothing in the repo conditions the encode
  step on the prefix.
- One-night test: cached AR features (31 min) -> codec alpha = 2 (5,000 steps, ~1.6 h at ~1.1 s/step) -> gate.
  Early read at the end of codec training: the prior's total CE on held-out codes is itself an achievable code
  length. If it is <= 910 go on; if > 1,050 stop and run alpha = 0.5 instead. Then NCM 1,800 steps (~2.6 h with
  online AR features) -> `plan_probe --gain-only`. Expected: NCM total 760-900 bits (extra 510-650), gains at
  L2-L5, L6 unchanged at ~284 (a follow-up sets lam[6] > 0 and checks the gate). Also record shift stability
  (expect a drop from 100% because near-boundary points now land on the prior-favoured side, which is stable
  across the shift) and per-level codes used (collapse watch).
- Cheat risk: (i) collapse to few codes per level raises predictability and ruins reconstruction: the gate
  catches it, and `codes_used_l0_l2` is logged every 500 steps; (ii) the prior memorises training windows: bits
  are measured on held-out windows with a generator that never saw the prior's weights; (iii) prefix information
  smuggled into the codes: impossible, the decoder never sees the prefix and the gate is decode-from-codes;
  (iv) distortion hidden by a lenient metric: the gate is the real decoder on held-out chunks. The one honest
  loss of information is a slightly larger quantisation error at coarse levels, which the fine levels re-encode.
- Confidence it passes the 30% test: 45%. The lever acts on every level that holds the bits, the geometry says
  switches are cheap, and the prior's CE gives an early abort; the risk is that the real gain per code is 2 bits
  rather than 4 and the run lands at ~700 extra.
- GPU cost on a Quadro T1000: ~5 h for one alpha (6.5 h if the second alpha's codec is also trained).

---

### 2. Shift-equivariant local encoder (no absolute positions, windowed attention, per-position norm)

- Atoms used: 201 (replace absolute coordinates with distance-aware ones; rows 4, 19), 193 (attention masks as
  structural priors; rows 4, 11), 101/102 (arbitrary segmentation boundaries fragment structure; rows 7, 12, 19).
- Mechanism: Fact 6 says level-0 codes are a fingerprint of (token, absolute position) pairs. Three components
  of the current encoder make them so: `wpe` over 64 positions, full bidirectional attention, and a GroupNorm
  whose statistics span the chunk. Removing all three makes z_i = f(x_{i-w..i+w}) exactly: a local,
  translation-equivariant feature. The level-0 input is then a pooled bag of local features over 16 tokens, so a
  one-token shift changes 2 of the 16 constituents instead of all of them, and an identifier rename changes only
  the windows that contain it. The push-forward of p(text | prefix) through a pooled local feature is a
  coarse-grained language model (which 16-gram neighbourhood comes next), which is learnable; through the
  current global map it is not. A second, measurable consequence: the fine codes become local functions of the
  text, so the within-level dependence that fact 5 found absent reappears, and the k-pass code lengths that
  `reveal_bits.py` already computes become meaningful (a valid factorisation). Reconstruction should hold:
  upstream's original architecture is a pure lookup encoder (`enc_layers=0`), the decoder keeps `dec_positional=
  True` and full attention, and 3 fine levels x 11 bits per token is ample capacity for 13-bit tokens.
- Exact change: in `models.build_codec`, behind `CODE_SCALE_ENC=local4`: `enc_positional=False`,
  `enc_attn_window=4` (upstream `SelfAttention._window_mask`, two layers give a 17-token receptive field),
  `pre_quant_norm_type='layer'` (per-position LayerNorm, affine-free; `vqvae.py` supports it). Everything else
  as codec-l7 (`learned_l1_sampling` stride-16 conv is already shift-equivariant on the 16-grid). The NCM
  generator's prefix path `codec.encoder.wte(prefix)` is unaffected. No new training code.
- Why it is new vs "already tried": the plan lists a shift-agreement *loss* as an untried option; this removes
  the cause architecturally (equivariance by construction, no extra loss, no extra compute). Level layouts and
  codebook sizes are untouched.
- One-night test: codec 5,000 steps (~75 min) -> gate -> `plan_probe --probe-only` (15 min). First read: level-0
  codes changed by a one-token shift, target <= 50% (now 100%), and the MLP probe gain at L0-L2 (now 0.13 / 1.22
  / 0.94 bits). If stability is still > 80% stop here (the pooled-vector geometry flipped anyway) and keep the
  codec only as a base for proposal 1. Otherwise NCM 1,800 steps (~2.3 h) -> `plan_probe --gain-only`. Expected:
  1-pass NCM total ~1,000-1,080 (extra 750-830): coarse levels roughly halve, L3-L4 improve modestly, L6 unchanged,
  so alone it probably does not pass; its job is to make proposal 1's switches and the 3-pass factorisation
  effective. Report 1-pass bits as the headline (the brief's metric) and 3-pass bits as secondary.
- Cheat risk: none from the change itself (no information is removed, only the encoder's access to position).
  The gate catches a weaker encoder. A subtle trap: a very small window could make the fine codes almost the
  token identity, which lowers NCM bits only if the generator can predict tokens in parallel; the measured bits
  are honest about that either way.
- Confidence it passes the 30% test: 30% alone; ~55% as the base codec for proposal 1 (the two changes act on
  the same near-boundary noise from opposite sides: fewer boundary crossings, and a prior to decide the rest).
- GPU cost on a Quadro T1000: ~4 h (1.5 h if stopped after the probe).

---

### 3. Coarse bands aligned to a language model's reading of the chunk, plus a syntax-skeleton head

- Atoms used: 32 (distil hidden states, not outputs; rows 4, 15, 17), 46 (lightweight linear bridges between
  independently trained representation spaces; row 10), 127 (anisotropy: whiten before aligning; row 8),
  253 (explicit structural representation of code, AST/indentation/bracket structure; rows 16, 19),
  44 (probing as the readout; row 20).
- Mechanism: The VQ spends level 0's 11 bits on the largest-variance directions of the pooled embedding, which
  for BPE code are dominated by rare identifiers, not by what the prefix can predict. Facts 3 and 7: the probe
  reads only 0.13 bits of level 0 from the prefix, and forcing true levels 0-3 lifts similarity 0.135 -> 0.675, so
  a coarse band that *means* something is what the generator is missing. The frozen AR control's hidden states
  over the chunk are a smooth, prefix-predictable summary of the same text (the AR predicts its own future states
  far better than it predicts individual tokens, because pooling over 16 positions averages out token noise).
  Regressing the pooled pre-quant latent at scales 4, 8, 16 onto whitened pooled AR states through a linear bridge
  makes the coarse Voronoi cells partition a language-shaped space, so p(code | prefix) is the push-forward of a
  smooth map (VA-VAE / REPA's finding in images: aligning VAE latents to a foundation model's features makes them
  far easier to generate at equal reconstruction). The skeleton head is the explicit, Python-specific version of
  the same pressure: indentation level, token class (keyword / identifier / number / string / bracket / newline /
  comment) and bracket depth per 16-token window are shift-stable and are exactly what the AR gets right (68%
  syntax-valid). Gradient reaches the encoder directly through pooled z, so the `z.detach()` obstacle at levels
  1-3 does not arise (the bands are read off z, not off the residual chain).
- Exact change: `train.py codec --align-weight W [--skeleton-weight S]`, windows prefix+chunk:
  1. Frozen AR forward on the 256-token window; take the last 64 hidden states after `ar.ln` (B, 64, 384);
     standardise per channel with running mean/std and project onto a fixed PCA basis computed from a 4k-window
     warm-up (whitening); detach.
  2. For s in {4, 8, 16}: t_s = area-pool(h, s); u_s = area-pool(z, s), with u_4 = `vq.learned_down_l1(z)` so it
     is exactly what level 0 quantises; bridge W_s = Linear(256, 384); loss_s = mean(1 - cos(W_s u_s, t_s)).
     align = loss_4 + loss_8 + loss_16, weight W in {0.5, 0.1}.
  3. Optional skeleton head on u_4 and u_8: per window, an 8-bin indentation histogram, a 10-class token-class
     histogram and max bracket depth (clipped at 4), computed from the BPE tokens (decode each id once; the
     `plan_probe.identifier_tokens` pattern), BCE/MSE loss with weight S = 0.2.
  4. Reconstruction, vq_loss, codebooks and the generator are unchanged.
- Why it is new vs "already tried": Path A aligned the level-0 latent to *prefix* features contrastively (what
  the prefix knows now); this distils the AR's reading of the *chunk* into three coarse bands by regression,
  whitened, through a bridge that leaves the latent free to keep reconstruction detail. The plan listed a VA-VAE
  option as a backup and never ran it; the skeleton target was not in any plan.
- One-night test: codec with W = 0.5 (online AR on 256 tokens, ~2.1 s/step, ~2.9 h; or 25 min of cached pooled
  features for 50k windows and ~1.5 h) -> gate -> `plan_probe --probe-only`: pass line for continuing is an MLP
  probe gain >= 2.0 bits at L0 and >= 2.0 at L1-L2 (now 0.13 / 1.22 / 0.94) and shift-change <= 60%. Then NCM
  (~2.3 h) -> bits. Expected: NCM total 950-1,050 (extra 700-800): L0-L2 fall from 248 to ~120, L3-L4 drop
  somewhat because they condition on a meaningful plan, L6 unchanged. Alone it is likely a near miss; stacked on
  proposal 1 it should move the coarse levels that ECVQ cannot (fact 7: the prior is weak at level 0).
- Cheat risk: the alignment can push detail out of the coarse bands into the fine levels (bits move, total may
  not fall): the test measures the total, so it is honest; collapse of coarse latents onto the AR subspace
  reduces reconstruction, caught by the gate; the skeleton features are computed from the chunk only and add
  pressure, they remove nothing. The AR's features enter only the tokenizer's training loss, never inference, so
  no capacity is borrowed at test time.
- Confidence it passes the 30% test: 25% alone (the fine-level residual noise, ~700 of the 940 bits, is
  untouched); the plan-quality effect (forcing test) may matter more than the bits.
- GPU cost on a Quadro T1000: ~5-5.5 h (with cached features ~4.5 h).

---

### 4. Predictive (closed-loop) residual coding: quantise the innovation, not the residual

- Atoms used: 122 (conditional VAE; row 1), 26 (conditional prior / ELBO; rows 9, 13), 23 (the AR-vs-NAR
  trade-off: AR models carry context forward step by step, NAR models assume independence; row 14),
  139 (propose-and-verify: a cheap draft, then code only the correction; rows 11, 17), 60/188 (continuous
  relaxation, for the predictor's regression target; rows 5, 10).
- Mechanism: In the current RVQ, level k codes `down_k(z - sum_{j<k} q_j)`: the reconstruction error of the levels
  above, which the generator must predict from scratch at every level. In predictive coding (DPCM, every modern
  neural video codec) the encoder subtracts a prediction that the *decoder side can also compute* and codes only
  the innovation. Here the prediction P_k is a small network of (prefix, codes of levels < k), i.e. exactly the
  generator's conditioning; the code becomes `argmin ||down_k(z_rest) - P_k - e_c||`, and the reconstruction adds
  P_k back: `q_k = up_k(e_c + P_k)`. Why this attacks the 940 bits: the generator's conditional over innovation
  codes is simple and nearly stationary (most mass on "small correction" codes, modulated by how uncertain the
  context is), so its KL is small even though the codes still carry H(text | prefix); the codebook (EMA) adapts
  to the innovation distribution, so its index is a smooth function of a small, centred quantity instead of a hash
  of a full residual. Facts: the gain is at the fine levels (3-6, ~940 of 1,191 bits), where P_k conditions on
  coarser codes and can be good; at level 0 P_0 knows only the prefix and the gain is small, as in fact 7. The
  decoder now also receives prefix information through P_k, which is legitimate conditional coding (the AR
  control conditions on the prefix too) and is measured by the same gate.
- Exact change (invasive; the codec and the generator both become prefix-conditioned):
  1. `LatentPredictor` per level k: input [prefix features, area-pooled z_hat_<k at L_k, position] -> MLP ->
     (B, L_k, 256). For fairness at inference, the prefix features come from an *own* 2-layer causal encoder over
     the 192 prefix tokens (384-d, last + mean), trained jointly through the MSE below; the frozen AR is not used
     because it would run at generation time.
  2. Quantiser loop (copy of upstream's into `code_scale/dpcm.py`): `target = down_k(z_rest) - rho * P_k.detach()`;
     `indices = argmin ||target - E_k||`; `q_k = quant_resi(up_k(E_k[indices] + rho * P_k))`; MSE(P_k,
     down_k(z_rest).detach()) added to the loss; rho ramps 0 -> 1 over steps 500-1,500 so codebooks form before
     the target moves. lam-style ramping is the known cure for the "STE diverges within 100 steps" note in
     `quant.py`.
  3. `idx_to_ncm_input(idx, prefix)`, `idx_to_z_q(idx, prefix)`, `decode(idx, prefix)`, `encode(x, prefix)`:
     rebuild the cumulative with P_k level by level (sequential, cheap). `models._inputs`, `ncm_logits`,
     `full_generate` and `evaluate` pass the prefix through; the frozen predictor lives inside the codec.
  4. The NCM generator architecture is unchanged; it predicts innovation indices.
- Why it is new vs "already tried": nothing in the repo or upstream changes what the quantiser targets; all
  previous paths changed what the generator sees or how it samples. It is the complement of proposal 1 (which
  changes which code is chosen, not what is quantised).
- One-night test (realistically two nights plus a day of coding): night 1 = codec with rho ramp (~2 h) ->
  gate -> the predictor's own MSE as a fraction of residual variance per level (if P_k explains < 20% of the
  variance at levels 3-5, stop: the innovation is the signal and nothing changes); night 2 = NCM (~2.5 h) ->
  bits with prefix-conditioned encode. Expected if it trains: NCM total 650-900 (extra 400-650), with L6 finally
  moving because P_6 conditions on all other levels. Expected if the ramp destabilises codebooks: gate fails
  (reconstruction < 99%) and the run is a clear negative.
- Cheat risk: the decoder can now copy from the prefix through P_k (e.g. identifiers that recur), which lowers
  code bits legitimately; it is the same conditioning the AR control enjoys and the gate still demands the chunk
  back from (codes, prefix). Collapse (encoder makes z predictable by carrying less) is caught by the gate.
  A wrong implementation that leaks the chunk into P_k (e.g. pooling z_hat of the *current* level) would make
  bits vanish: guard with a unit test that P_k's inputs are prefix + levels < k only, mirroring `reveal_bits`'s
  factorisation check.
- Confidence it passes the 30% test: 35% on the second night (highest ceiling, highest implementation risk).
- GPU cost on a Quadro T1000: ~5 h per night, two nights.

---

## Suggested order

Night 1: proposal 2 (flags only, 1.5-4 h) while coding proposal 1. Night 2: proposal 1 on the better of
codec-l7's config or the local codec from night 1. Night 3 (only if night 2 lands between 660 and 760 extra):
proposal 3's alignment on top of 1, since ECVQ is weak exactly where alignment is strong (level 0, fact 7).
Proposal 4 is the follow-up if night 2 shows that most remaining bits are at levels 5-6.
