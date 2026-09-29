"""Differential geometry of the Lorentz (hyperboloid) model of hyperbolic space.

Conventions
-----------
The hyperboloid of curvature ``-c`` (``c > 0``) is

    H^n_c = { (x_0, x_s) in R^{n+1} : -x_0^2 + ||x_s||^2 = -1/c,  x_0 > 0 }.

Points are stored by their **spatial coordinates** ``x_s`` in R^n only. The time
coordinate is implicit, ``x_0 = sqrt(1/c + ||x_s||^2)``, and is recomputed when
needed. This keeps the number of channels equal to the dimension of the manifold,
exactly as for the Poincare ball, so every hypll layer works unchanged.

A tangent vector ``u`` at ``x`` is likewise stored by its spatial part ``u_s``. Its
time part is implied by the tangency condition <x, u>_L = 0, i.e.
``u_0 = <x_s, u_s> / x_0``. At the origin ``o = (1/sqrt(c), 0)`` we have ``u_0 = 0``,
so tangent vectors at the origin are ordinary Euclidean vectors whose Euclidean norm
equals their Riemannian norm.

Note: the hypll Poincare ball uses a conformal factor of 2 at the origin, so under the
standard isometry a Poincare tangent vector ``u`` at the origin corresponds to the
Lorentz tangent vector ``2u``.

All functions take a ``dim`` argument giving the manifold dimension. Inputs may have a
different number of dimensions as long as they broadcast; ``dim`` refers to the
broadcasted shape.
"""

import torch

# Maximum value of sqrt(c) * d(o, x) allowed per dtype. Keeps sinh/cosh and x_0^2 finite.
_MAX_T = {
    torch.float16: 5.0,
    torch.bfloat16: 40.0,
    torch.float32: 40.0,
    torch.float64: 300.0,
}


def max_t(dtype: torch.dtype) -> float:
    return _MAX_T.get(dtype, 40.0)


def _neg_dim(dim: int, *tensors: torch.Tensor) -> int:
    """Converts ``dim`` (w.r.t. the broadcasted shape) to a negative index that is
    valid for every tensor that has that dimension."""
    if dim < 0:
        return dim
    return dim - max(t.dim() for t in tensors)


def time_coord(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Implicit time coordinate x_0 = sqrt(1/c + ||x_s||^2) (keepdim)."""
    return (1 / c + x.pow(2).sum(dim=dim, keepdim=True)).sqrt()


def tangent_time_coord(
    x: torch.Tensor, u: torch.Tensor, c: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Implicit time coordinate u_0 = <x_s, u_s> / x_0 of a tangent vector at x (keepdim)."""
    d = _neg_dim(dim, x, u)
    return (x * u).sum(dim=d, keepdim=True) / time_coord(x, c, d)


def project(x: torch.Tensor, c: torch.Tensor, dim: int = -1, eps: float = -1.0) -> torch.Tensor:
    """Radially clamps points so that sqrt(c) * d(o, x) <= eps (dtype default if eps < 0).

    Every x_s in R^n is a valid point on the hyperboloid, so this only guards against
    overflow; it is a no-op for all points at a reasonable distance from the origin.
    """
    t_max = eps if eps >= 0 else max_t(x.dtype)
    c_sqrt = c.sqrt()
    max_norm = torch.sinh(torch.as_tensor(t_max, dtype=x.dtype, device=x.device)) / c_sqrt
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    return torch.where(norm > max_norm, x / norm * max_norm, x)


def expmap0(v: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exponential map at the origin: x_s = sinh(sqrt(c)|v|) v / (sqrt(c)|v|)."""
    t = v.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    t_clamped = t.clamp_max(max_t(v.dtype))
    return torch.sinh(t_clamped) / t * v


def logmap0(y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Logarithmic map at the origin: v = asinh(sqrt(c)|y_s|) y_s / (sqrt(c)|y_s|)."""
    t = y.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    return torch.asinh(t) / t * y


def minkowski_sqnorm_diff(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1, keepdim: bool = True
) -> torch.Tensor:
    """<x - y, x - y>_L = ||x_s - y_s||^2 - (x_0 - y_0)^2 >= 0, computed stably.

    Uses x_0 - y_0 = <x_s - y_s, x_s + y_s> / (x_0 + y_0) to avoid cancellation.
    """
    d = _neg_dim(dim, x, y)
    delta = x - y
    total = x + y
    time_sum = time_coord(x, c, d) + time_coord(y, c, d)
    time_diff = (delta * total).sum(dim=d, keepdim=True) / time_sum
    out = (delta.pow(2).sum(dim=d, keepdim=True) - time_diff.pow(2)).clamp_min(0)
    return out if keepdim else out.squeeze(d)


def dist(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    """Geodesic distance d(x, y) = 2/sqrt(c) * asinh(sqrt(c) * ||x - y||_L / 2).

    Equivalent to acosh(-c <x, y>_L) / sqrt(c), but accurate for nearby points.
    """
    c_sqrt = c.sqrt()
    sq = minkowski_sqnorm_diff(x, y, c, dim=dim, keepdim=keepdim)
    return 2 / c_sqrt * torch.asinh(c_sqrt * sq.clamp_min(1e-30).sqrt() / 2)


def inner(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Riemannian inner product of tangent vectors u, v at x: <u_s, v_s> - u_0 v_0."""
    d = _neg_dim(dim, x, u, v)
    u0 = tangent_time_coord(x, u, c, d)
    v0 = tangent_time_coord(x, v, c, d)
    out = (u * v).sum(dim=d, keepdim=True) - u0 * v0
    return out if keepdim else out.squeeze(d)


def expmap(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exponential map at x: cosh(sqrt(c)|v|) x + sinh(sqrt(c)|v|) v / (sqrt(c)|v|)."""
    d = _neg_dim(dim, x, v)
    v_norm = inner(x, v, v, c, dim=d, keepdim=True).clamp_min(1e-30).sqrt()
    t = c.sqrt() * v_norm
    t_clamped = t.clamp_max(max_t(v.dtype))
    return torch.cosh(t_clamped) * x + torch.sinh(t_clamped) / t * v


def logmap(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Logarithmic map at x: d(x, y) / sinh(sqrt(c) d) * sqrt(c) * (y - cosh(sqrt(c) d) x)."""
    d_ = _neg_dim(dim, x, y)
    distance = dist(x, y, c, dim=d_, keepdim=True)
    t = (c.sqrt() * distance).clamp_min(1e-15)
    # alpha = -c <x, y>_L = cosh(t); y - alpha x is the unnormalised tangent direction
    direction = y - torch.cosh(t) * x
    return t / torch.sinh(t) * direction


def transp(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Parallel transport of v from x to y along the geodesic:

    P(v) = v + c <y, v>_L / (1 - c <x, y>_L) * (x + y).
    """
    d = _neg_dim(dim, x, y, v)
    x0 = time_coord(x, c, d)
    y0 = time_coord(y, c, d)
    v0 = (x * v).sum(dim=d, keepdim=True) / x0
    yv = (y * v).sum(dim=d, keepdim=True) - y0 * v0  # <y, v>_L
    alpha = c * (x0 * y0 - (x * y).sum(dim=d, keepdim=True))  # -c <x, y>_L >= 1
    return v + c * yv / (1 + alpha) * (x + y)


def euc_to_tangent(x: torch.Tensor, u: torch.Tensor, c: torch.Tensor, dim: int = -1):
    """Converts a Euclidean gradient w.r.t. the spatial coordinates into the Riemannian
    gradient. The metric in spatial coordinates is G = I - x_s x_s^T / x_0^2, whose
    inverse gives grad = u + c <x_s, u> x_s.
    """
    d = _neg_dim(dim, x, u)
    return u + c * (x * u).sum(dim=d, keepdim=True) * x


def cdist(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Pairwise distances between x of shape (..., N, D) and y of shape (..., M, D)."""
    return dist(x.unsqueeze(-2), y.unsqueeze(-3), c, dim=-1)


def lorentz_to_poincare(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Isometry from the hyperboloid to the Poincare ball: p = x_s / (1 + sqrt(c) x_0)."""
    return x / (1 + c.sqrt() * time_coord(x, c, dim))


def poincare_to_lorentz(p: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Isometry from the Poincare ball to the hyperboloid: x_s = 2 p / (1 - c ||p||^2)."""
    return 2 * p / (1 - c * p.pow(2).sum(dim=dim, keepdim=True)).clamp_min(1e-15)


def to_ambient(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Returns the full (n+1)-dimensional ambient coordinates (x_0, x_s)."""
    return torch.cat([time_coord(x, c, dim), x], dim=dim)


def from_ambient(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drops the time coordinate of ambient coordinates (x_0, x_s)."""
    return x.narrow(dim, 1, x.size(dim) - 1)
