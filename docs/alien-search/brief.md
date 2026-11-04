# Brief: why NCM loses to GPT on Python code, and what a fix must do

## Setup
- NCM = Hazy Research's Next-Concept Model. A tokenizer (multiscale residual VQ-VAE) turns each 64-token
  chunk of Python (BPE vocab 8,192) into codes on 7 levels of [4, 8, 16, 32, 48, 64, 64] positions
  (236 codes per chunk), codebook 2,048 per level. The tokenizer conditions on nothing: codes depend only on
  the chunk itself. A generator (6 layers x 384, ~13M params) sees a 192-token prefix and predicts the
  codes level by level, coarse to fine; within a level all positions are predicted in parallel.
  The decoder turns codes back into tokens.
- Control: a GPT-style AR model (7 x 384, ~16M params), same data, same 1,800 steps x 32 windows.
- Hardware: one laptop GPU (Quadro T1000, 4 GB). One training night ~= 2-6 hours.

## Measured facts
1. Tokenizer is near-perfect: 99.72% token accuracy, 88.1% exact chunks; decoding the true codes gives
   0.996 character similarity.
2. Quality: NCM reaches ~0.72 of AR's continuation similarity and ~0.10 of AR's rate of valid Python with code.
3. Teacher-forced level accuracy (true earlier levels): L0 1.6%, L1 4.9%, L2 6.7%, L3 20.3%, L4 29.9%,
   L5 74.0%, L6 47.3%. If the true codes for levels 0-3 are given, greedy similarity goes 0.135 -> 0.675.
4. Bits per 64-token chunk on held-out data (teacher-forced): **AR 250.5**; NCM **1,190.7**.
   Per level (bits): 33, 68, 147, 246, 285, 128, 284.
   The codes are a deterministic function of the text, so a perfect code model would need <= the text's
   bits. NCM wastes ~940 bits.
5. Letting positions within a level see the other true codes of that level removes only 5 of the 940 extra
   bits (0.6%). So the problem is NOT parallel/blind guessing within a level. Each code is hard to
   predict from the prefix even when all other codes in its level are known.
6. Level-0 codes change for 98-100% of chunks if the chunk window shifts by one token (codes are not
   shift-stable).
7. A small MLP probe on AR's prefix features predicts level-0 codes only ~0.1 bits better than the code
   frequency prior (8.24 bits blind); after adding a predictability loss (InfoNCE from prefix features to
   level-0 latent, weight 0.05) it reached 1.05 bits gain without hurting reconstruction, but quality did
   not move.

## Already tried (do not propose again unless the twist is clearly new)
- Predictability (InfoNCE "plan") loss on level 0 only, weight 0.05 (Path A).
- Generator levels attend to all earlier levels (block-causal) (Path B1).
- Small coarse codebooks 64/128/256/512 (Path B2): fewer bits per chunk (1,008) but no quality gain.
- Within-level reveals / MaskGIT-style multi-pass decoding (Path C).
- Fixed an upstream dead-code restart bug; tried several level layouts.

## What the fix must do
Make the tokenizer's codes **predictable from the prefix** (cut NCM's extra bits over AR) while keeping
reconstruction good (>= 99% tokens, >= 50% exact chunks), at ~13M generator params on a T1000.

## The test every proposal must define
One night or less on the T1000: train the modified tokenizer (and generator if needed) briefly and measure
**NCM extra bits over AR per chunk** (baseline 940) plus the reconstruction gate. Pass = extra bits drop
>= 30% (to <= 660) with the gate still passing.
