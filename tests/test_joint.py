"""Joint training: the differentiable code choice must pick exactly the tokenizer's own codes."""
import torch

from code_scale.joint import guess_term, soft_codes
from code_scale.models import SHAPE, build_codec


def _codec():
    torch.manual_seed(0)
    return build_codec(64, restart_fix=True).eval()


def test_soft_codes_match_encode_and_reach_encoder():
    codec = _codec()
    x = torch.randint(0, 64, (4, SHAPE["chunk"]))
    taus = [1.0] * len(SHAPE["levels"])
    z = codec._encode_pre_quant(x)
    codes, logq = soft_codes(codec.vq, z, taus)
    assert torch.equal(codes, codec.encode(x))
    assert torch.equal(logq.argmax(-1), codes)
    assert torch.allclose(logq.exp().sum(-1), torch.ones(codes.shape), atol=1e-4)
    gen_logits = torch.randn(*codes.shape, logq.shape[-1])
    guess_term(logq, gen_logits).backward()
    grads = [p.grad for p in codec.encoder.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0
    assert codec.vq.embeddings[0].weight.grad is None  # codebook vectors stay fixed


def test_guess_term_prefers_codes_the_model_expects():
    logq = torch.log(torch.tensor([[[0.9, 0.1], [0.1, 0.9]]]))
    gen = torch.tensor([[[5.0, -5.0], [5.0, -5.0]]])  # model expects code 0 everywhere
    agree = guess_term(logq[:, :1], gen[:, :1])
    disagree = guess_term(logq[:, 1:], gen[:, 1:])
    assert agree < disagree
