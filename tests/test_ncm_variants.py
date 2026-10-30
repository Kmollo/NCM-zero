"""Checks for the 2026-10-29 variants: per-level codebook sizes, block_causal generation, has_code."""
import os

os.environ.setdefault("CODE_SCALE_LEVELS", "4,8,16,32,48,64,64")

import pytest
import torch

from code_scale import models
from code_scale.evaluate import has_code

SMALL = [64, 128, 256, 512, 2048, 2048, 2048]


@pytest.fixture
def variant(monkeypatch):
    def make(codebooks, attn):
        monkeypatch.setitem(models.SHAPE, "codebooks", codebooks)
        monkeypatch.setattr(models, "ATTN", attn)
        torch.manual_seed(0)
        codec = models.build_codec(512, restart_fix=True).eval()
        ncm = models.build_ncm(codec, pad_id=0).eval()
        return codec, ncm
    return make


def bounds():
    b = [0]
    for l in models.SHAPE["levels"]:
        b.append(b[-1] + l)
    return b


def test_corrupt_stays_inside_each_codebook():
    levels = models.SHAPE["levels"]
    codes = torch.cat([torch.randint(0, v, (64, l)) for l, v in zip(levels, SMALL)], dim=1)
    out = models.corrupt(codes, levels, SMALL, max_rate=1.0)
    b = bounds()
    for i, v in enumerate(SMALL):
        assert int(out[:, b[i]:b[i + 1]].max()) < v
    assert torch.equal(out[:, :levels[0]], codes[:, :levels[0]])  # level 0 is never corrupted


def test_small_codebooks_masked_in_training_logits(variant):
    codec, ncm = variant(SMALL, "block_diagonal")
    prefix = torch.randint(1, 512, (2, models.SHAPE["prefix_len"]))
    codes = codec.encode(torch.randint(1, 512, (2, 64)))
    logits = models.ncm_logits(ncm, codec, prefix, codes, pad_id=0, corrupt_rate=0.0)
    b = bounds()
    for i, v in enumerate(SMALL):
        part = logits[:, b[i]:b[i + 1]]
        assert int(part.argmax(-1).max()) < v
        if v < 2048:
            assert float(part[..., v:].max()) <= -1e4 + 1


def test_full_generate_matches_upstream_for_block_diagonal(variant):
    codec, ncm = variant([2048] * 7, "block_diagonal")
    assert not models.is_block_causal(ncm)
    prefix = torch.randint(1, 512, (3, models.SHAPE["prefix_len"]))
    with torch.no_grad():
        ref = ncm.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0)
        new = models.full_generate(ncm, None, prefix, temperature=0.0, top_k=0, top_p=1.0)
    assert (ref == new).float().mean().item() > 0.99


def test_block_causal_generation_ignores_later_levels(variant):
    codec, ncm = variant(SMALL, "block_causal")
    assert models.is_block_causal(ncm)
    models.install_generator(ncm)
    prefix = torch.randint(1, 512, (2, models.SHAPE["prefix_len"]))
    codes = ncm.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0)
    b = bounds()
    for i, v in enumerate(SMALL):
        assert int(codes[:, b[i]:b[i + 1]].max()) < v
    # changing level-3+ codes must not change the logits of levels 0-2
    other = codes.clone()
    other[:, b[3]:] = 1
    with torch.no_grad():
        a = models.ncm_logits(ncm, codec, prefix, codes, 0, corrupt_rate=0.0)
        c = models.ncm_logits(ncm, codec, prefix, other, 0, corrupt_rate=0.0)
    assert torch.allclose(a[:, :b[3]], c[:, :b[3]], atol=1e-4)
    assert not torch.allclose(a[:, b[3]:], c[:, b[3]:], atol=1e-4)


def test_has_code():
    assert not has_code("\n")
    assert not has_code("    # todo\n\n")
    assert has_code("    return x + 1\n")
    assert has_code("    pass\n")


@pytest.fixture
def reveal_model(monkeypatch):
    monkeypatch.setattr(models, "REVEAL", [0, 1, 2, 3])
    monkeypatch.setitem(models.SHAPE, "codebooks", [2048] * 7)
    monkeypatch.setattr(models, "ATTN", "block_diagonal")
    torch.manual_seed(0)
    codec = models.build_codec(512, restart_fix=True).eval()
    ncm = models.build_ncm(codec, pad_id=0).eval()
    with torch.no_grad():  # make reveals visible to the network (trained values are not zero)
        ncm.reveal.flag.normal_(0, 0.5)
    models.install_generator(ncm)
    return codec, ncm


def test_reveal_only_reaches_its_own_level(reveal_model):
    codec, ncm = reveal_model
    b = bounds()
    prefix = torch.randint(1, 512, (2, models.SHAPE["prefix_len"]))
    codes = codec.encode(torch.randint(1, 512, (2, 64)))
    none = torch.zeros_like(codes, dtype=torch.bool)
    some = none.clone()
    some[:, b[1]:b[1] + 4] = True  # reveal half of level 1
    with torch.no_grad():
        a = models.ncm_logits(ncm, codec, prefix, codes, 0, corrupt_rate=0.0, shown=none)
        c = models.ncm_logits(ncm, codec, prefix, codes, 0, corrupt_rate=0.0, shown=some)
    assert torch.allclose(a[:, :b[1]], c[:, :b[1]], atol=1e-4)
    assert torch.allclose(a[:, b[2]:], c[:, b[2]:], atol=1e-4)
    assert not torch.allclose(a[:, b[1] + 4:b[2]], c[:, b[1] + 4:b[2]], atol=1e-4)  # hidden half of level 1 sees it


def test_random_shown_and_loss_ignore(reveal_model):
    b = bounds()
    codes = torch.randint(0, 2048, (64, b[-1]))
    shown = models.random_shown(codes)
    assert not shown[:, b[4]:].any() and shown[:, :b[4]].any()
    logits = torch.zeros(64, b[-1], 2048)
    logits.scatter_(-1, codes[..., None], -50.0)  # every revealed position would be a huge loss
    full = models.ncm_loss(logits, codes)
    part = models.ncm_loss(logits.masked_fill(shown[..., None], 0.0), codes, ignore=shown)
    assert torch.isclose(part, full, rtol=1e-3)


def test_reveal_generate_one_pass_matches_full_generate(reveal_model, monkeypatch):
    codec, ncm = reveal_model
    with torch.no_grad():
        ncm.reveal.flag.zero_()  # hidden flag 0: one pass is then exactly the baseline decoder
    monkeypatch.setattr(models, "PASSES", 1)
    prefix = torch.randint(1, 512, (3, models.SHAPE["prefix_len"]))
    with torch.no_grad():
        ref = models.full_generate(ncm, None, prefix, temperature=0.0, top_k=0, top_p=1.0)
        new = ncm.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0)
    assert (ref == new).float().mean().item() > 0.99


def test_reveal_generate_three_passes(reveal_model, monkeypatch):
    codec, ncm = reveal_model
    monkeypatch.setattr(models, "PASSES", 3)
    calls = []
    orig = ncm.forward
    monkeypatch.setattr(ncm, "forward", lambda *a, **k: calls.append(1) or orig(*a, **k))
    prefix = torch.randint(1, 512, (2, models.SHAPE["prefix_len"]))
    with torch.no_grad():
        codes = ncm.generate(None, prefix, temperature=0.8, top_k=0, top_p=0.95)
    assert len(calls) == 4 * 3 + 3
    assert codes.shape == (2, bounds()[-1]) and int(codes.max()) < 2048
