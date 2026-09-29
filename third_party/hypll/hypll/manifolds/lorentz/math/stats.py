from typing import Optional, Union

import torch

from hypll.manifolds.poincare_ball.math.stats import frechet_mean as poincare_frechet_mean

from .diffgeom import dist, lorentz_to_poincare, poincare_to_lorentz, time_coord


def midpoint(
    x: torch.Tensor,
    c: torch.Tensor,
    man_dim: int = -1,
    batch_dim: Union[int, list[int]] = 0,
    w: Optional[torch.Tensor] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Lorentzian centroid: the normalised (weighted) sum of the ambient points,

        m = sum_i w_i x_i / (sqrt(c) * sqrt(-<sum_i w_i x_i, sum_i w_i x_i>_L)).

    This is the isometric image of the Einstein / gyro-midpoint computed by the
    Poincare ball's ``midpoint``.
    """
    x0 = time_coord(x, c, dim=man_dim)
    if w is None:
        s = x.sum(dim=batch_dim, keepdim=True)
        s0 = x0.sum(dim=batch_dim, keepdim=True)
    else:
        s = (w * x).sum(dim=batch_dim, keepdim=True)
        s0 = (w * x0).sum(dim=batch_dim, keepdim=True)

    minkowski_norm_sq = (s0.pow(2) - s.pow(2).sum(dim=man_dim, keepdim=True)).clamp_min(1e-15)
    midpoint = s / (c.sqrt() * minkowski_norm_sq.sqrt())
    if not keepdim:
        midpoint = midpoint.squeeze(dim=batch_dim)

    return midpoint


def frechet_mean(
    x: torch.Tensor,
    c: torch.Tensor,
    vec_dim: int = -1,
    batch_dim: Union[int, list[int]] = 0,
    keepdim: bool = False,
) -> torch.Tensor:
    """Frechet mean, computed by mapping to the Poincare ball (isometry), reusing the
    differentiable solver of the Poincare ball, and mapping back.
    """
    p = lorentz_to_poincare(x, c, dim=vec_dim)
    mean_p = poincare_frechet_mean(x=p, c=c, vec_dim=vec_dim, batch_dim=batch_dim, keepdim=keepdim)
    if isinstance(batch_dim, int):
        batch_dim = [batch_dim]
    vec_dim_pos = vec_dim if vec_dim >= 0 else x.dim() + vec_dim
    batch_dim_pos = [bd if bd >= 0 else x.dim() + bd for bd in batch_dim]
    out_vec_dim = (
        vec_dim_pos if keepdim else vec_dim_pos - sum(bd < vec_dim_pos for bd in batch_dim_pos)
    )
    return poincare_to_lorentz(mean_p, c, dim=out_vec_dim)


def frechet_variance(
    x: torch.Tensor,
    c: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    vec_dim: int = -1,
    batch_dim: Union[int, list[int]] = 0,
    keepdim: bool = False,
) -> torch.Tensor:
    """Mean squared geodesic distance to mu (the Frechet mean if mu is None)."""
    if isinstance(batch_dim, int):
        batch_dim = [batch_dim]

    if mu is None:
        mu = frechet_mean(x=x, c=c, vec_dim=vec_dim, batch_dim=batch_dim, keepdim=True)

    if x.dim() != mu.dim():
        for bd in sorted(batch_dim):
            mu = mu.unsqueeze(bd)

    distance = dist(x=x, y=mu, c=c, dim=vec_dim, keepdim=keepdim)
    distance = distance.pow(2)
    return distance.mean(dim=batch_dim, keepdim=keepdim)
