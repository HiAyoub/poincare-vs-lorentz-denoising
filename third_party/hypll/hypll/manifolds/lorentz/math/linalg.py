from typing import Optional

import torch

from .diffgeom import max_t, time_coord


def lorentz_hyperplane_dists(
    x: torch.Tensor,
    z: torch.Tensor,
    r: Optional[torch.Tensor],
    c: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    The HNN++ hyperplane operation v_k(x), written in Lorentz (spatial) coordinates.

    With the same parameters (z, r), this is exactly the quantity computed by
    ``poincare_hyperplane_dists`` on the isometric image of x in the Poincare ball.
    Using lambda_x = 1 + sqrt(c) x_0 and lambda_x * p = x_s, the Poincare formula becomes

        v_k(x) = 2 ||z_k|| / sqrt(c) * asinh( sqrt(c) ( <x_s, z_k>/||z_k|| cosh(2 sqrt(c) r_k)
                                                        - x_0 sinh(2 sqrt(c) r_k) ) ),

    which is linear in the ambient coordinates of x and has no singularity at the
    boundary of the ball.

    Parameters
    ----------
    x : tensor
        spatial coordinates of the input points
    z : tensor
        hyperplane orientations, shape (in_features, out_features)
    r : tensor
        hyperplane offsets, shape (out_features,)
    c : tensor
        curvature

    Returns
    -------
    tensor
        v_k(x) for every hyperplane k, along dimension ``dim``
    """
    dim_shifted_x = x.movedim(source=dim, destination=-1)

    c_sqrt = c.sqrt()
    z_norm = z.norm(dim=0).clamp_min(1e-15)
    xz = torch.matmul(dim_shifted_x, z) / z_norm

    if r is None:
        arg = c_sqrt * xz
    else:
        x0 = time_coord(dim_shifted_x, c, dim=-1)
        two_csqrt_r = 2.0 * c_sqrt * r
        arg = c_sqrt * (xz * two_csqrt_r.cosh() - x0 * two_csqrt_r.sinh())

    dim_shifted_output = 2 * z_norm / c_sqrt * torch.asinh(arg)
    return dim_shifted_output.movedim(source=-1, destination=dim)


def lorentz_fully_connected(
    x: torch.Tensor,
    z: torch.Tensor,
    bias: Optional[torch.Tensor],
    c: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    The HNN++ fully connected layer in Lorentz coordinates.

    The Poincare version outputs w / (1 + sqrt(1 + c ||w||^2)) with
    w_k = sinh(sqrt(c) v_k) / sqrt(c). That point is the image of the hyperboloid point
    with spatial coordinates w, so in the Lorentz model the output is simply w.
    """
    c_sqrt = c.sqrt()
    v = lorentz_hyperplane_dists(x=x, z=z, r=bias, c=c, dim=dim)
    t = (c_sqrt * v).clamp(min=-max_t(v.dtype), max=max_t(v.dtype))
    return t.sinh() / c_sqrt
