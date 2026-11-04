"""Model builders for the code-scale experiment.

NCM = pinned upstream VQVAE code tokenizer + NextConceptTransformer generator,
configured like the released XSum/LM1B checkpoints but scaled down.
AR = a plain GPT-style decoder over the same BPE tokens, the matched control.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ncm_code.model import upstream

# One fixed experiment shape. Changing any of these is a new experiment.
SHAPE = {
    "prefix_len": 192,          # BPE tokens of context
    "chunk": 64,                # BPE tokens generated per NCM call
    # CODE_SCALE_LEVELS="4,8,16,32,48,64,64" selects another layout (a new experiment).
    "levels": [int(v) for v in os.environ.get("CODE_SCALE_LEVELS", "4,8,16,32,64").split(",")],
    "codebook": 2048,           # largest codebook (and the size of NCM's shared output head)
}
# CODE_SCALE_CODEBOOKS="64,128,256,512,2048,2048,2048" gives each level its own codebook size.
SHAPE["codebooks"] = ([int(v) for v in os.environ["CODE_SCALE_CODEBOOKS"].split(",")]
                      if os.environ.get("CODE_SCALE_CODEBOOKS") else [SHAPE["codebook"]] * len(SHAPE["levels"]))
assert len(SHAPE["codebooks"]) == len(SHAPE["levels"]) and max(SHAPE["codebooks"]) == SHAPE["codebook"]
# CODE_SCALE_ATTN=block_causal lets level k attend to levels < k (training); default is upstream's block_diagonal.
ATTN = os.environ.get("CODE_SCALE_ATTN", "block_diagonal")
# Path C: CODE_SCALE_REVEAL="0,1,2,3" lets positions of these levels see codes already chosen in their own
# level (trained with random reveals, generated in CODE_SCALE_PASSES confidence-ordered passes per level).
REVEAL = [int(v) for v in os.environ["CODE_SCALE_REVEAL"].split(",")] if os.environ.get("CODE_SCALE_REVEAL") else []
PASSES = int(os.environ.get("CODE_SCALE_PASSES", "3"))
# CODE_SCALE_DEC=4 gives the tokenizer decoder 4 blocks (upstream lm1b), idea 1b; default 2 as all codecs so far.
CODEC = {"embed_dim": 256, "n_layers": 4, "enc_layers": 2, "dec_layers": int(os.environ.get("CODE_SCALE_DEC", "2"))}
# Idea 1 (docs/alien-search): CODE_SCALE_ENC picks the tokenizer encoder.
#   attn          : 2 bidirectional attention layers + positions, GroupNorm(8) (all code_scale codecs so far)
#   lookup-group4 : upstream lm1b recipe: code input = token embedding only, no positions, GroupNorm(4)
#   lookup-layer  : same lookup, but LayerNorm per position (each position's input is its token alone)
ENCODERS = {
    "attn": {"enc_layers": 2, "enc_positional": True, "pre_quant_groupnorm": 8, "pre_quant_norm_type": "group"},
    "lookup-group4": {"enc_layers": 0, "enc_positional": False, "pre_quant_groupnorm": 4, "pre_quant_norm_type": "group"},
    "lookup-layer": {"enc_layers": 0, "enc_positional": False, "pre_quant_groupnorm": 8, "pre_quant_norm_type": "layer"},
}
ENC = os.environ.get("CODE_SCALE_ENC", "attn")
assert ENC in ENCODERS, f"CODE_SCALE_ENC must be one of {list(ENCODERS)}"
NCM_GEN = {"n_embd": 384, "n_layer": 6, "n_head": 6, "dropout": 0.1}
AR_GEN = {"n_embd": 384, "n_layer": 7, "n_head": 6, "dropout": 0.1}


def fixed_restart(vq):
    """Restart dead codes AT the sampled input vector.

    Upstream sets ema_w = v and ema_cs = threshold, so the code becomes
    v / threshold (~10x too large at 0.1): it never wins, stays dead and is
    re-restarted every step. Scaling ema_w by the threshold keeps code == v.
    """
    @torch.no_grad()
    def restart(flat_input, ema_cs, ema_w, emb_i):
        dead = ema_cs < vq.restart_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0 or flat_input.shape[0] == 0:
            return
        src = torch.randint(0, flat_input.shape[0], (n_dead,), device=flat_input.device)
        new_vecs = flat_input[src].to(emb_i.weight.dtype)
        emb_i.weight.data[dead] = new_vecs
        ema_w[dead] = new_vecs * vq.restart_threshold
        ema_cs[dead] = vq.restart_threshold
    vq._restart_dead_codes = restart


def build_codec(vocab_size, dropout=0.0, restart_fix=False):
    VQVAE, _ = upstream()
    levels = SHAPE["levels"]
    codec = VQVAE(
        in_vocab_size=vocab_size, context_length=SHAPE["chunk"],
        embed_dim=CODEC["embed_dim"], vqvae_vocab_size=list(SHAPE["codebooks"]),
        n_layers=CODEC["n_layers"], compression_factor=1, dropout=dropout,
        enc_layers=ENCODERS[ENC]["enc_layers"], dec_layers=CODEC["dec_layers"],
        use_qk_norm=True, pre_quant_groupnorm=ENCODERS[ENC]["pre_quant_groupnorm"], pre_quant_dropout=0.0,
        pre_quant_norm_type=ENCODERS[ENC]["pre_quant_norm_type"],
        enc_positional=ENCODERS[ENC]["enc_positional"], dec_positional=True,
        vector_quantizer_config=OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.99, "epsilon": 1e-5, "commitment_cost": 0.25,
            "restart_threshold": 0.1, "learned_l1_sampling": True,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
            "levels": {"use_manual_levels": True, "manual_levels": levels},
            "aux": {"fine_drop": {"prob": 0.5, "min_keep": 1}},
        }),
    )
    if restart_fix:
        fixed_restart(codec.vq)
    return codec


def build_ncm(codec, pad_id):
    _, NextConceptTransformer = upstream()
    codec.eval().requires_grad_(False)
    n = len(SHAPE["levels"])
    ncm = NextConceptTransformer(codec, OmegaConf.create({
        **NCM_GEN, "bias": True, "prefix_len": SHAPE["prefix_len"],
        "vocab_size": [SHAPE["codebook"]] * n, "attn_pattern": ATTN,
        "use_level_encoding": True, "use_positional_encoding": True,
        "query_level0": True, "predict_level0": False, "raw_token_prefix": True,
        "prefix_pad_token_id": pad_id, "use_rope": True, "rope_base": 10000,
        "use_qk_norm": True, "use_flex_attention": False,
        "shared_output_head": True, "shared_head_per_level_bias": True,
        "post_upsample_conv": {"enabled": True, "kernel_size": 3},
        "ctx_encoder": {"enabled": False},
    }))
    mask_small_codebooks(ncm)
    if REVEAL:
        ncm.reveal = Reveal(codec.vq.embed_dim, SHAPE["levels"], REVEAL)
    return ncm


def _bounds(levels):
    b = [0]
    for l in levels:
        b.append(b[-1] + l)
    return b


class Reveal(nn.Module):
    """Adds the codes already chosen in a position's own level to its input (levels in REVEAL only).

    Revealed position: scale * codec_embedding(code) + flag[1]. Hidden position of a REVEAL level: flag[0].
    Other levels get nothing (identical to the baseline). With block_diagonal attention a level's
    positions see each other, so revealed codes reach the hidden positions of the same level only.
    """

    def __init__(self, dim, levels, reveal):
        super().__init__()
        self.levels, self.reveal = list(levels), list(reveal)
        self.flag = nn.Parameter(torch.zeros(2, dim))
        self.scale = nn.Parameter(torch.ones(1))
        b = _bounds(levels)
        active = torch.zeros(b[-1], dtype=torch.bool)
        for k in reveal:
            active[b[k]:b[k + 1]] = True
        self.register_buffer("active", active, persistent=False)

    def forward(self, vq, codes, shown):  # codes, shown: (B, N); returns (B, N, dim)
        b = _bounds(self.levels)
        with torch.no_grad():
            emb = torch.zeros(*codes.shape, self.flag.shape[1], device=codes.device)
            for k in self.reveal:
                emb[:, b[k]:b[k + 1]] = vq.embeddings[k](codes[:, b[k]:b[k + 1]]).float()
        shown = shown & self.active
        hidden = ~shown & self.active
        return shown[..., None] * (self.scale * emb + self.flag[1]) + hidden[..., None] * self.flag[0]


def random_shown(codes, p_none=0.25):
    """Training reveals: per sample and REVEAL level, a ratio r ~ U(0,1) (0 with prob p_none) of true codes."""
    b, shown = _bounds(SHAPE["levels"]), torch.zeros_like(codes, dtype=torch.bool)
    B, dev = codes.shape[0], codes.device
    for k in REVEAL:
        r = torch.rand(B, 1, device=dev) * (torch.rand(B, 1, device=dev) > p_none)
        shown[:, b[k]:b[k + 1]] = torch.rand(B, b[k + 1] - b[k], device=dev) < r
    return shown


def _inputs(ncm, prefix_vectors, level_inputs, codes, shown):
    query = ncm.build_l0_query_input(prefix_vectors.shape[0], level_inputs.dtype)
    target = torch.cat([query, level_inputs], dim=1)
    if getattr(ncm, "reveal", None) is not None:
        target = target + ncm.reveal(ncm.vqvae_local[0].vq, codes, shown).to(target.dtype)
    return torch.cat([prefix_vectors.to(level_inputs.dtype), target], dim=1)


@torch.no_grad()
def reveal_generate(ncm, idx, prefix_tokens, temperature=1.0, top_k=50, top_p=0.9):
    """Coarse to fine like upstream, but each REVEAL level is decoded in PASSES passes: sample the hidden
    positions, keep the most confident ones (model probability of the sampled code), show them, repeat;
    the last pass keeps everything (MaskGIT-style). Other levels: one pass."""
    assert idx is None
    vq = ncm.vqvae_local[0].vq
    levels, sizes = vq.levels, vq.vqvae_vocab_sizes
    B, dev = prefix_tokens.shape[0], prefix_tokens.device
    b = _bounds(levels)
    codes = torch.zeros(B, b[-1], dtype=torch.long, device=dev)
    shown = torch.zeros(B, b[-1], dtype=torch.bool, device=dev)
    prefix = prefix_tokens[:, :ncm.prefix_len]
    prefix_vectors = ncm.vqvae_local[0].encoder.wte(prefix)
    pad_mask = prefix.eq(ncm.prefix_pad_id) if ncm.prefix_pad_id is not None else None
    for k in range(len(levels)):
        n, multi = levels[k], k in ncm.reveal.reveal
        passes = PASSES if multi else 1
        for p in range(1, passes + 1):
            level_inputs = vq.idx_to_ncm_input(codes)
            logits = ncm(_inputs(ncm, prefix_vectors, level_inputs, codes, shown),
                         prefix_pad_mask=pad_mask)[:, b[k]:b[k + 1], :sizes[k]]
            cur, sh = codes[:, b[k]:b[k + 1]], shown[:, b[k]:b[k + 1]]
            new = torch.where(sh, cur, ncm._sample_level(logits, temperature, top_k, top_p))
            keep_n = n if p == passes else math.ceil(n * p / passes)
            conf = F.log_softmax(logits.float(), -1).gather(-1, new[..., None])[..., 0].masked_fill(sh, float("inf"))
            keep = torch.zeros_like(sh).scatter_(1, conf.topk(keep_n, dim=1).indices, True)
            codes[:, b[k]:b[k + 1]] = torch.where(keep, new, cur)
            if multi:
                shown[:, b[k]:b[k + 1]] = keep
    return codes


class _MaskedHead:
    """Shared output head view that rules out codes beyond this level's codebook (not a module: no weights)."""

    def __init__(self, view, size):
        self.view, self.size = view, size

    def __call__(self, x):
        out = self.view(x)
        keep = torch.arange(out.shape[-1], device=out.device) < self.size
        return out.masked_fill(~keep, -1e4)


def mask_small_codebooks(ncm):
    sizes = ncm.vqvae_local[0].vq.vqvae_vocab_sizes
    if all(v == SHAPE["codebook"] for v in sizes):
        return
    assert ncm.lm_heads.per_level_bias, "masking needs per-level head views"
    for i, v in enumerate(sizes):
        if v < SHAPE["codebook"] and not isinstance(ncm.lm_heads._views[i], _MaskedHead):
            ncm.lm_heads._views[i] = _MaskedHead(ncm.lm_heads._views[i], v)


def is_block_causal(ncm):
    """Read the pattern from the attention-mask buffer (it is loaded with the checkpoint)."""
    P, l0 = ncm.prefix_len, ncm.levels[0]
    return bool(torch.isfinite(ncm.attn_mask_for_masking[0, 0, P + l0, P]))


@torch.no_grad()
def full_generate(ncm, idx, prefix_tokens, temperature=1.0, top_k=50, top_p=0.9):
    """Level-by-level generation through the training forward pass.

    Upstream's generate() shows each level only the prefix and itself, which matches block_diagonal
    training but not block_causal. Here every level is decoded with the exact training mask; codes of
    later levels are still zero but cannot affect earlier levels (level k only attends to levels <= k).
    Sampling goes through ncm._sample_level so diagnose.py's forcing hook still works.
    """
    assert idx is None
    vq = ncm.vqvae_local[0].vq
    levels, sizes = vq.levels, vq.vqvae_vocab_sizes
    B, dev = prefix_tokens.shape[0], prefix_tokens.device
    bounds = [0]
    for l in levels:
        bounds.append(bounds[-1] + l)
    codes = torch.zeros(B, bounds[-1], dtype=torch.long, device=dev)
    prefix = prefix_tokens[:, :ncm.prefix_len]
    prefix_vectors = ncm.vqvae_local[0].encoder.wte(prefix)
    pad_mask = prefix.eq(ncm.prefix_pad_id) if ncm.prefix_pad_id is not None else None
    for k in range(len(levels)):
        level_inputs = vq.idx_to_ncm_input(codes)
        query = ncm.build_l0_query_input(B, level_inputs.dtype)
        inputs = torch.cat([prefix_vectors.to(level_inputs.dtype), query, level_inputs], dim=1)
        logits = ncm(inputs, prefix_pad_mask=pad_mask)[:, bounds[k]:bounds[k + 1], :sizes[k]]
        codes[:, bounds[k]:bounds[k + 1]] = ncm._sample_level(logits, temperature, top_k, top_p)
    return codes


def install_generator(ncm):
    """After loading a checkpoint: block_causal models must generate with full_generate."""
    if getattr(ncm, "reveal", None) is not None:
        assert not is_block_causal(ncm), "reveal decoding assumes block_diagonal"
        ncm.generate = lambda idx, prefix_tokens, **kw: reveal_generate(ncm, idx, prefix_tokens, **kw)
    elif is_block_causal(ncm):
        ncm.generate = lambda idx, prefix_tokens, **kw: full_generate(ncm, idx, prefix_tokens, **kw)
    return ncm


def corrupt(codes, levels, codebooks, max_rate=0.3):
    """Upstream-style code corruption of conditioning codes; level 0 is skipped.
    codebooks: one size per level (an int means the same size for every level)."""
    if isinstance(codebooks, int):
        codebooks = [codebooks] * len(levels)
    out = codes.clone()
    rate = torch.rand(codes.shape[0], 1, device=codes.device) * max_rate
    start = levels[0]
    for size, vs in zip(levels[1:-1], codebooks[1:-1]):
        chunk = out[:, start:start + size]
        mask = torch.rand(chunk.shape, device=codes.device) < rate
        chunk.copy_(torch.where(mask, torch.randint_like(chunk, vs), chunk))
        start += size
    return out


def ncm_logits(model, codec, prefix, codes, pad_id, corrupt_rate=0.3, shown=None):
    """Teacher-forced logits for all levels; prefix embedded by the frozen codec.
    shown: (B, N) bool, true codes revealed to their own level (reveal models only; None = none shown)."""
    with torch.no_grad():
        cond = corrupt(codes, SHAPE["levels"], SHAPE["codebooks"], corrupt_rate) if corrupt_rate else codes
        level_inputs = codec.vq.idx_to_ncm_input(cond)
        prefix_vectors = codec.encoder.wte(prefix)
    if shown is None:
        shown = torch.zeros_like(codes, dtype=torch.bool)
    return model(_inputs(model, prefix_vectors, level_inputs, codes, shown), prefix_pad_mask=prefix.eq(pad_id))


def ncm_loss(logits, codes, ignore=None):
    """Mean cross-entropy; positions in `ignore` (revealed codes, trivially copyable) are left out."""
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), codes.reshape(-1), reduction="none")
    if ignore is None:
        return ce.mean()
    keep = (~ignore).reshape(-1).float()
    return (ce * keep).sum() / keep.sum().clamp(min=1)


class _Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h, self.dropout = h, dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    """Plain decoder-only Transformer; the matched autoregressive control."""

    def __init__(self, vocab_size, context, n_embd, n_layer, n_head, dropout):
        super().__init__()
        self.context = context
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(context, n_embd)
        self.blocks = nn.ModuleList(_Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.wte.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    def _cached(self, idx, cache, start):
        """Forward with a per-layer KV cache (inference only)."""
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(start, start + T, device=idx.device))
        for i, block in enumerate(self.blocks):
            B, _, C = x.shape
            q, k, v = block.qkv(block.ln1(x)).split(C, dim=2)
            q, k, v = (t.view(B, T, block.h, C // block.h).transpose(1, 2) for t in (q, k, v))
            if cache[i] is not None:
                k = torch.cat([cache[i][0], k], dim=2)
                v = torch.cat([cache[i][1], v], dim=2)
            cache[i] = (k, v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
            x = x + block.proj(y.transpose(1, 2).reshape(B, T, C))
            x = x + block.mlp(block.ln2(x))
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, steps, temperature=0.0, top_p=1.0):
        assert idx.shape[1] + steps <= self.context
        cache = [None] * len(self.blocks)
        logits = self._cached(idx, cache, 0)[:, -1].float()
        for step in range(steps):
            if step:
                logits = self._cached(idx[:, -1:], cache, idx.shape[1] - 1)[:, -1].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, -1)
                if top_p < 1.0:
                    sp, si = probs.sort(-1, descending=True)
                    keep = sp.cumsum(-1) - sp <= top_p
                    probs = torch.zeros_like(probs).scatter_(-1, si, sp * keep)
                nxt = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def build_ar(vocab_size):
    return GPT(vocab_size, SHAPE["prefix_len"] + SHAPE["chunk"], **AR_GEN)


def count(module, trainable=True):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable)
