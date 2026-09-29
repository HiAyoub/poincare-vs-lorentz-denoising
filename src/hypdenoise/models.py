"""U-Net denoisers with a Euclidean, Poincare or Lorentz encoder and a Euclidean decoder.

All variants share the same topology, channel widths (32-64-128-256-512), decoder and
residual head. Only the encoder geometry changes. The hyperbolic encoders use exactly the
same hypll layers; switching Poincare <-> Lorentz only changes the manifold object.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypll import nn as hnn
from hypll.manifolds.lorentz import Lorentz
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor


def _norm(c: int) -> nn.GroupNorm:
    g = min(8, c)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


# ------------------------------------------------------------------------------------------
# Euclidean blocks
# ------------------------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool, self.conv = nn.MaxPool2d(2), DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(self.reduce(x))
        if x.shape[-2:] != skip.shape[-2:]:
            dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ------------------------------------------------------------------------------------------
# Hyperbolic blocks (manifold-agnostic: work with PoincareBall and Lorentz)
# ------------------------------------------------------------------------------------------
class HyperbolicDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, manifold, k=3):
        super().__init__()
        self.hconv1 = hnn.HConvolution2d(in_ch, out_ch, k, manifold=manifold, padding=k // 2)
        self.hrelu = hnn.HReLU(manifold=manifold)
        self.hconv2 = hnn.HConvolution2d(out_ch, out_ch, k, manifold=manifold, padding=k // 2)

    def forward(self, x):
        return self.hconv2(self.hrelu(self.hconv1(x)))


class HyperbolicDown(nn.Module):
    def __init__(self, in_ch, out_ch, manifold, k=3):
        super().__init__()
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv = HyperbolicDoubleConv(in_ch, out_ch, manifold, k)

    def forward(self, x):
        return self.conv(self.pool(x))


# ------------------------------------------------------------------------------------------
# Encoders: all return a list of Euclidean feature maps [level0, ..., bottleneck]
# ------------------------------------------------------------------------------------------
class EuclideanEncoder(nn.Module):
    def __init__(self, widths, in_ch=3):
        super().__init__()
        self.inc = DoubleConv(in_ch, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])

    def forward(self, x):
        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        return feats


class HyperbolicEncoder(nn.Module):
    """Image -> expmap0 -> hyperbolic conv blocks; every level is logmap0'ed as a skip.

    lift_scale: hypll's Poincare ball has conformal factor 2 at the origin, so the Poincare
    tangent vector u corresponds to the Lorentz tangent vector 2u. With lift_scale = 1
    (Poincare) and 2 (Lorentz), both encoders compute the same function at identical weights.
    """

    def __init__(self, widths, manifold, lift_scale: float, in_ch=3, k=3):
        super().__init__()
        self.manifold, self.lift_scale = manifold, lift_scale
        self.inc = HyperbolicDoubleConv(in_ch, widths[0], manifold, k)
        self.downs = nn.ModuleList(
            [HyperbolicDown(widths[i], widths[i + 1], manifold, k) for i in range(len(widths) - 1)]
        )

    def _log(self, h):
        return self.manifold.logmap(None, h).tensor / self.lift_scale

    def forward(self, x):
        h = self.manifold.expmap(TangentTensor(data=self.lift_scale * x, man_dim=1, manifold=self.manifold))
        h = self.inc(h)
        feats = [self._log(h)]
        for d in self.downs:
            h = d(h)
            feats.append(self._log(h))
        return feats


# ------------------------------------------------------------------------------------------
# Full denoiser
# ------------------------------------------------------------------------------------------
class UNetDenoiser(nn.Module):
    """Residual denoiser: returns noisy - head(decoder(encoder(noisy)))."""

    def __init__(self, encoder: nn.Module, widths, out_ch=3):
        super().__init__()
        self.encoder = encoder
        depth = len(widths) - 1
        self.ups = nn.ModuleList(
            [Up(widths[depth - i], widths[depth - i - 1], widths[depth - i - 1]) for i in range(depth)]
        )
        self.head = nn.Conv2d(widths[0], out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def manifold(self):
        return getattr(self.encoder, "manifold", None)

    def curvature(self):
        m = self.manifold
        return None if m is None else float(m.c().detach().cpu())

    def forward(self, x):
        feats = self.encoder(x)
        h = feats[-1]
        for i, up in enumerate(self.ups):
            h = up(h, feats[-i - 2])
        return x - self.head(h)


def build_model(cfg: dict) -> UNetDenoiser:
    m = cfg["model"]
    widths = [m["base_channels"] * 2**i for i in range(m["depth"] + 1)]
    kind = m["encoder"]
    if kind == "euclidean":
        enc = EuclideanEncoder(widths)
    elif kind in ("poincare", "lorentz"):
        c = Curvature(value=m["curvature_init"], requires_grad=m["trainable_curvature"])
        if kind == "poincare":
            enc = HyperbolicEncoder(widths, PoincareBall(c=c), lift_scale=1.0)
        else:
            enc = HyperbolicEncoder(widths, Lorentz(c=c), lift_scale=2.0)
    else:
        raise ValueError(f"Unknown encoder '{kind}'")
    return UNetDenoiser(enc, widths)


def count_params(model: nn.Module) -> int:
    return sum((p.tensor if hasattr(p, "tensor") else p).numel() for p in model.parameters())
