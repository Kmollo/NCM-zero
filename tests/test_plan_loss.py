"""Unit checks for the Path A plan-predictability loss (code_scale/plan_loss.py)."""
import math
import os

os.environ.setdefault("CODE_SCALE_LEVELS", "4,8,16,32,48,64,64")

import torch

from code_scale.models import build_codec
from code_scale.plan_loss import PlanPredictor, info_nce, level0_latent


def test_random_predictions_score_chance():
    torch.manual_seed(0)
    B = 64
    loss, acc = info_nce(torch.randn(B, 4, 256), torch.randn(B, 4, 256), torch.tensor(0.0))
    assert abs(loss.item() - math.log(B)) < 0.3
    assert acc.item() < 0.1


def test_perfect_predictions_win():
    torch.manual_seed(0)
    t = torch.randn(64, 4, 256)
    loss, acc = info_nce(t.clone(), t, torch.tensor(2.3))
    assert acc.item() == 1.0
    assert loss.item() < 0.1


def test_constant_latent_cannot_win():
    torch.manual_seed(0)
    t = torch.ones(64, 4, 256)
    loss, _ = info_nce(torch.randn(64, 4, 256), t, torch.tensor(2.3))
    assert loss.item() >= math.log(64) - 1e-3  # never better than chance


def test_level0_latent_matches_quantizer_and_carries_gradient():
    torch.manual_seed(0)
    codec = build_codec(512, restart_fix=True).eval()
    x = torch.randint(0, 512, (8, 64))
    z = codec._encode_pre_quant(x)
    lat = level0_latent(codec, z)
    assert lat.shape == (8, 4, codec.vq.embed_dim)
    # same level-0 codes as the tokenizer's own encode()
    emb = codec.vq.embeddings[0].weight
    picks = codec.vq._distances(lat.reshape(-1, lat.shape[-1]), emb).argmin(1).view(8, 4)
    assert torch.equal(picks, codec.encode(x)[:, :4])
    # the loss reaches the encoder
    planner = PlanPredictor(768, 4, codec.vq.embed_dim)
    loss, _ = info_nce(planner(torch.randn(8, 768)), lat, planner.log_scale)
    loss.backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in codec.encoder.parameters() if p.grad is not None)
    assert enc_grad > 0
    assert planner.net[1].weight.grad.abs().sum().item() > 0
