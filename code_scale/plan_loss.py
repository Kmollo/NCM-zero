"""Path A (docs/ncm-next-plan.md): make the tokenizer's level-0 "plan" guessable from the prefix.

A small predictor reads the prefix (version 1: the frozen AR control's prefix hidden states,
last position + mean, 768-d) and predicts the chunk's level-0 latent, i.e. the quantizer input
at level 0 (`vq._down_for_quant(z, 0, levels[0])`, upstream/ncm/models/quant.py). An InfoNCE
loss makes each prediction pick its own chunk's latent out of the batch. The target keeps its
gradient, so the encoder learns to put prefix-predictable content into level 0. A constant
latent cannot win the contrastive game, which prevents collapse.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanPredictor(nn.Module):
    def __init__(self, d_in, positions, d_latent, hidden=512):
        super().__init__()
        self.positions, self.d_latent = positions, d_latent
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, positions * d_latent))
        self.log_scale = nn.Parameter(torch.tensor(2.3))  # learned temperature, starts at 1/0.1

    def forward(self, feats):  # (B, d_in) -> (B, positions, d_latent)
        return self.net(feats).view(-1, self.positions, self.d_latent)


def level0_latent(codec, z):
    """The level-0 quantizer input for pre-quant latents z (B, L, C); keeps the gradient."""
    return codec.vq._down_for_quant(z, 0, codec.vq.levels[0])


def info_nce(pred, target, log_scale):
    """Per level-0 position, each prediction must pick its own chunk's latent among the batch.

    Returns (loss, top-1 accuracy). Chance accuracy is 1 / batch.
    """
    B, n, _ = pred.shape
    p = F.normalize(pred.float(), dim=-1).transpose(0, 1)    # (n, B, C)
    t = F.normalize(target.float(), dim=-1).transpose(0, 1)  # (n, B, C)
    logits = torch.bmm(p, t.transpose(1, 2)) * log_scale.clamp(max=4.6).exp()  # (n, B, B)
    labels = torch.arange(B, device=pred.device).expand(n, B)
    loss = 0.5 * (F.cross_entropy(logits.reshape(n * B, B), labels.reshape(-1))
                  + F.cross_entropy(logits.transpose(1, 2).reshape(n * B, B), labels.reshape(-1)))
    acc = (logits.argmax(-1) == labels).float().mean()
    return loss, acc


@torch.no_grad()
def ar_prefix_features(ar, prefix):
    """Frozen AR control's reading of the prefix: last-position + mean hidden state (2 x 384)."""
    with torch.autocast("cuda", dtype=torch.float16):
        x = ar.wte(prefix) + ar.wpe(torch.arange(prefix.shape[1], device=prefix.device))
        for block in ar.blocks:
            x = block(x)
        h = ar.ln(x).float()
    return torch.cat([h[:, -1], h.mean(1)], -1)
