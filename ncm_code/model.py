"""Thin training adapter over the pinned upstream NCM modules."""
import json
from pathlib import Path
import subprocess
import sys

import torch
from omegaconf import OmegaConf

from .data import PAD, VOCAB
from .config import load_config

ROOT = Path(__file__).resolve().parents[1]


def upstream():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    path = ROOT / lock["path"]
    if not (path / "models" / "next_concept.py").exists():
        raise ValueError("Missing official NCM source. Run: python scripts/fetch_upstream.py")
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["revision"]:
        raise ValueError("Upstream revision differs from upstream.lock.json")
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Upstream has modified tracked files; restore or explicitly version changes before training")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer
    return VQVAE, NextConceptTransformer


def build_tokenizer(cfg):
    VQVAE, _ = upstream()
    return VQVAE(in_vocab_size=VOCAB, context_length=cfg["output_length"], embed_dim=cfg["tokenizer_dim"],
                 vqvae_vocab_size=[cfg["codebook_size"]] * len(cfg["levels"]), n_layers=cfg["heads"],
                 compression_factor=1, dropout=0.0, enc_layers=1, dec_layers=1, pre_quant_groupnorm=1,
                 pre_quant_norm_type="layer", vector_quantizer_config=OmegaConf.create({
                     "clss": "multiscale_residual_vector_quantizer", "decay": 0.95,
                     "commitment_cost": 0.25, "restart_threshold": 0.1,
                     "levels": {"use_manual_levels": True, "manual_levels": cfg["levels"]},
                     "aux": {"fine_drop": {"prob": cfg["fine_drop"], "min_keep": 1}}}))


def build_generator(cfg, tokenizer):
    _, NextConceptTransformer = upstream()
    tokenizer.eval().requires_grad_(False)
    return NextConceptTransformer(tokenizer, OmegaConf.create({
        "n_embd": cfg["generator_dim"], "n_layer": cfg["generator_layers"], "n_head": cfg["heads"],
        "dropout": 0.0, "bias": True, "prefix_len": cfg["prefix_length"],
        "vocab_size": [cfg["codebook_size"]] * len(cfg["levels"]), "attn_pattern": "block_diagonal",
        "use_level_encoding": True, "query_level0": True, "raw_token_prefix": True,
        "prefix_pad_token_id": PAD, "use_rope": False, "use_positional_encoding": True}))


def generator_logits(model, tokenizer, prefix, target, noise=0.0):
    with torch.no_grad():
        codes = tokenizer.encode(target)
        conditioning = codes.clone()
        if noise:
            mask = torch.rand(conditioning.shape, device=conditioning.device) < noise
            conditioning = torch.where(mask, torch.randint(model.vocab_sizes[0], conditioning.shape, device=conditioning.device), conditioning)
        inputs = tokenizer.vq.idx_to_ncm_input(conditioning)
        prefix_vectors = tokenizer.encoder.wte(prefix)
    # Keep the learned level-zero query outside no_grad so it actually learns.
    inputs = torch.cat([prefix_vectors, model.build_l0_query_input(prefix.shape[0]), inputs], dim=1)
    return model(inputs, prefix_pad_mask=prefix.eq(PAD)), codes


@torch.no_grad()
def generate_ids(model, tokenizer, prefix):
    model.eval()
    tokenizer.eval()
    codes = model.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0)
    return tokenizer.decode(codes).argmax(-1)
