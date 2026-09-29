"""Tests for the Lorentz manifold.

Most tests check that the Lorentz manifold is the exact isometric counterpart of the
PoincareBall: mapping the Lorentz result to the ball must reproduce the Poincare result.
Everything runs in float64 so tolerances can be tight.
"""

import pytest
import torch

from hypll.manifolds.lorentz import Curvature, Lorentz
from hypll.manifolds.lorentz.math.diffgeom import (
    lorentz_to_poincare,
    poincare_to_lorentz,
    time_coord,
)
from hypll.manifolds.poincare_ball import PoincareBall
from hypll.nn import HConvolution2d, HMaxPool2d, HReLU
from hypll.tensors import ManifoldTensor, TangentTensor

DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-7


def _manifolds(c_value: float = 0.5):
    # Share one Curvature so both manifolds always have exactly the same c.
    curvature = Curvature(value=torch.tensor(c_value, dtype=DTYPE))
    return Lorentz(c=curvature), PoincareBall(c=curvature)


def _rand_points(manifold: Lorentz, *shape, scale: float = 1.0, man_dim: int = -1):
    torch.manual_seed(0)
    v = torch.randn(*shape, dtype=DTYPE) * scale
    return manifold.expmap(TangentTensor(data=v, manifold=manifold, man_dim=man_dim))


def _to_ball(lorentz: Lorentz, poincare: PoincareBall, x: ManifoldTensor) -> ManifoldTensor:
    return lorentz.to_poincare_ball(x, target=poincare)


# ---------------------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------------------
def test_points_lie_on_hyperboloid() -> None:
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 16, 5, scale=2.0)
    amb = lorentz.to_ambient(x)
    c = lorentz.c()
    minkowski = -amb[:, 0] ** 2 + amb[:, 1:].pow(2).sum(-1)
    assert torch.allclose(minkowski, -1 / c * torch.ones_like(minkowski), rtol=1e-10)
    assert (amb[:, 0] > 0).all()


def test_ambient_roundtrip() -> None:
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 8, 4)
    back = lorentz.from_ambient(lorentz.to_ambient(x))
    assert torch.allclose(back.tensor, x.tensor)


def test_expmap0_logmap0_inverse() -> None:
    lorentz, _ = _manifolds()
    torch.manual_seed(1)
    v = torch.randn(32, 6, dtype=DTYPE) * 3
    x = lorentz.expmap(TangentTensor(data=v, manifold=lorentz))
    v_back = lorentz.logmap(None, x)
    assert torch.allclose(v_back.tensor, v, atol=ATOL, rtol=RTOL)


def test_dist_to_origin_equals_tangent_norm() -> None:
    lorentz, _ = _manifolds()
    torch.manual_seed(2)
    v = torch.randn(32, 6, dtype=DTYPE)
    x = lorentz.expmap(TangentTensor(data=v, manifold=lorentz))
    origin = ManifoldTensor(data=torch.zeros_like(v), manifold=lorentz)
    assert torch.allclose(lorentz.dist(origin, x), v.norm(dim=-1), atol=ATOL, rtol=RTOL)


def test_dist_small_distances_are_accurate() -> None:
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 10, 3, scale=2.0)
    eps = 1e-7
    torch.manual_seed(3)
    u = torch.randn(10, 3, dtype=DTYPE)
    u = (
        u
        / lorentz.inner(
            TangentTensor(u, manifold_points=x, manifold=lorentz),
            TangentTensor(u, manifold_points=x, manifold=lorentz),
            keepdim=True,
        ).sqrt()
    )
    y = lorentz.expmap(TangentTensor(data=eps * u, manifold_points=x, manifold=lorentz))
    assert torch.allclose(lorentz.dist(x, y), torch.full((10,), eps, dtype=DTYPE), rtol=1e-5)


def test_logmap_expmap_inverse_at_point() -> None:
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 16, 4, scale=1.5)
    torch.manual_seed(4)
    y = lorentz.expmap(TangentTensor(torch.randn(16, 4, dtype=DTYPE), manifold=lorentz))
    v = lorentz.logmap(x, y)
    y_back = lorentz.expmap(v)
    assert torch.allclose(y_back.tensor, y.tensor, atol=1e-8, rtol=1e-7)
    # |log_x(y)| = d(x, y)
    assert torch.allclose(lorentz.inner(v, v).sqrt(), lorentz.dist(x, y), atol=1e-8, rtol=1e-7)


def test_transp_preserves_inner_product() -> None:
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 8, 5, scale=1.0)
    torch.manual_seed(5)
    y = lorentz.expmap(TangentTensor(torch.randn(8, 5, dtype=DTYPE), manifold=lorentz))
    u = TangentTensor(torch.randn(8, 5, dtype=DTYPE), manifold_points=x, manifold=lorentz)
    w = TangentTensor(torch.randn(8, 5, dtype=DTYPE), manifold_points=x, manifold=lorentz)
    pu, pw = lorentz.transp(u, y), lorentz.transp(w, y)
    assert torch.allclose(lorentz.inner(pu, pw), lorentz.inner(u, w), atol=1e-8, rtol=1e-7)


def test_transp_of_log_is_minus_log() -> None:
    # Transporting log_x(y) from x to y gives -log_y(x).
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 8, 3)
    torch.manual_seed(6)
    y = lorentz.expmap(TangentTensor(torch.randn(8, 3, dtype=DTYPE), manifold=lorentz))
    transported = lorentz.transp(lorentz.logmap(x, y), y)
    assert torch.allclose(transported.tensor, -lorentz.logmap(y, x).tensor, atol=1e-8, rtol=1e-7)


def test_euc_to_tangent_is_riemannian_gradient() -> None:
    # Defining property: <grad_R f, v>_x = df(v) = <grad_E f, v>_euclidean for all tangent v.
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 8, 4, scale=1.5)
    torch.manual_seed(7)
    grad_e = torch.randn(8, 4, dtype=DTYPE)
    v = torch.randn(8, 4, dtype=DTYPE)
    grad_r = lorentz.euc_to_tangent(x, ManifoldTensor(grad_e, manifold=lorentz))
    lhs = lorentz.inner(grad_r, TangentTensor(v, manifold_points=x, manifold=lorentz))
    rhs = (grad_e * v).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-8, rtol=1e-7)


def test_project_clamps_only_far_points() -> None:
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 8, 4)
    assert torch.equal(lorentz.project(x).tensor, x.tensor)
    far = ManifoldTensor(torch.full((2, 4), 1e150, dtype=DTYPE), manifold=lorentz)
    assert torch.isfinite(lorentz.to_ambient(lorentz.project(far))).all()


# ---------------------------------------------------------------------------------------
# Isometry with the Poincare ball
# ---------------------------------------------------------------------------------------
def test_isometry_roundtrip() -> None:
    c = torch.tensor(0.7, dtype=DTYPE)
    torch.manual_seed(8)
    x = torch.randn(16, 5, dtype=DTYPE) * 3
    assert torch.allclose(poincare_to_lorentz(lorentz_to_poincare(x, c), c), x, rtol=1e-9)


def test_expmap0_matches_poincare() -> None:
    # Poincare tangent u at the origin <-> Lorentz tangent 2u.
    lorentz, poincare = _manifolds()
    torch.manual_seed(9)
    u = torch.randn(32, 6, dtype=DTYPE)
    xp = poincare.expmap(TangentTensor(u, manifold=poincare))
    xl = lorentz.expmap(TangentTensor(2 * u, manifold=lorentz))
    assert torch.allclose(_to_ball(lorentz, poincare, xl).tensor, xp.tensor, atol=ATOL)


def test_logmap0_matches_poincare() -> None:
    lorentz, poincare = _manifolds()
    x = _rand_points(lorentz, 32, 6)
    vl = lorentz.logmap(None, x).tensor
    vp = poincare.logmap(None, _to_ball(lorentz, poincare, x)).tensor
    assert torch.allclose(vl, 2 * vp, atol=ATOL, rtol=RTOL)


def test_dist_matches_poincare() -> None:
    lorentz, poincare = _manifolds()
    x = _rand_points(lorentz, 32, 6)
    torch.manual_seed(10)
    y = lorentz.expmap(TangentTensor(torch.randn(32, 6, dtype=DTYPE), manifold=lorentz))
    d_l = lorentz.dist(x, y)
    d_p = poincare.dist(_to_ball(lorentz, poincare, x), _to_ball(lorentz, poincare, y))
    assert torch.allclose(d_l, d_p, atol=1e-8, rtol=1e-7)


def test_cdist_matches_dist() -> None:
    lorentz, _ = _manifolds()
    x = _rand_points(lorentz, 2, 5, 3)
    y = _rand_points(lorentz, 2, 4, 3, scale=0.5)
    cd = lorentz.cdist(x, y)
    assert cd.shape == (2, 5, 4)
    ref = lorentz.dist(
        ManifoldTensor(x.tensor.unsqueeze(2), manifold=lorentz),
        ManifoldTensor(y.tensor.unsqueeze(1), manifold=lorentz),
    )
    assert torch.allclose(cd, ref)


@pytest.mark.parametrize("use_bias", [True, False])
def test_fully_connected_matches_poincare(use_bias: bool) -> None:
    lorentz, poincare = _manifolds()
    torch.manual_seed(11)
    # Points in (batch, features, positions) layout with man_dim=1, as produced by unfold.
    u = torch.randn(4, 12, 7, dtype=DTYPE) * 0.5
    xp = poincare.expmap(TangentTensor(u, manifold=poincare, man_dim=1))
    xl = lorentz.expmap(TangentTensor(2 * u, manifold=lorentz, man_dim=1))
    z_p, b_p = poincare.construct_dl_parameters(12, 5, bias=use_bias)
    z_p.tensor.data = torch.randn(12, 5, dtype=DTYPE) * 0.2
    if use_bias:
        b_p.data = torch.randn(5, dtype=DTYPE) * 0.2
    out_p = poincare.fully_connected(xp, z_p, b_p)
    out_l = lorentz.fully_connected(xl, z_p, b_p)
    assert out_l.man_dim == 1
    assert torch.allclose(_to_ball(lorentz, poincare, out_l).tensor, out_p.tensor, atol=1e-8)


def test_midpoint_matches_poincare() -> None:
    lorentz, poincare = _manifolds()
    x = _rand_points(lorentz, 10, 3, 4)
    torch.manual_seed(12)
    w = torch.rand(10, 3, 1, dtype=DTYPE)
    for weights in (None, w):
        m_l = lorentz.midpoint(x, batch_dim=0, w=weights)
        m_p = poincare.midpoint(_to_ball(lorentz, poincare, x), batch_dim=0, w=weights)
        assert m_l.shape == (3, 4)
        assert torch.allclose(_to_ball(lorentz, poincare, m_l).tensor, m_p.tensor, atol=1e-8)


def test_frechet_mean_and_variance_match_poincare() -> None:
    lorentz, poincare = _manifolds()
    x = _rand_points(lorentz, 20, 3, scale=0.7)
    m_l = lorentz.frechet_mean(x, batch_dim=0)
    m_p = poincare.frechet_mean(_to_ball(lorentz, poincare, x), batch_dim=0)
    assert torch.allclose(_to_ball(lorentz, poincare, m_l).tensor, m_p.tensor, atol=1e-7)
    var_l = lorentz.frechet_variance(x, mu=m_l, batch_dim=0)
    var_p = poincare.frechet_variance(_to_ball(lorentz, poincare, x), mu=m_p, batch_dim=0)
    assert torch.allclose(var_l, var_p, atol=1e-7, rtol=1e-6)


def test_cat_and_flatten_match_poincare() -> None:
    lorentz, poincare = _manifolds()
    torch.manual_seed(13)
    u1 = torch.randn(2, 3, 4, dtype=DTYPE) * 0.5
    u2 = torch.randn(2, 5, 4, dtype=DTYPE) * 0.5
    l1 = lorentz.expmap(TangentTensor(2 * u1, manifold=lorentz, man_dim=1))
    l2 = lorentz.expmap(TangentTensor(2 * u2, manifold=lorentz, man_dim=1))
    p1 = poincare.expmap(TangentTensor(u1, manifold=poincare, man_dim=1))
    p2 = poincare.expmap(TangentTensor(u2, manifold=poincare, man_dim=1))

    cat_l = lorentz.cat([l1, l2], dim=1)
    cat_p = poincare.cat([p1, p2], dim=1)
    assert cat_l.shape == (2, 8, 4)
    assert torch.allclose(_to_ball(lorentz, poincare, cat_l).tensor, cat_p.tensor, atol=1e-8)

    flat_l = lorentz.flatten(l1, start_dim=1, end_dim=-1)
    flat_p = poincare.flatten(p1, start_dim=1, end_dim=-1)
    assert flat_l.shape == (2, 12) and flat_l.man_dim == 1
    assert torch.allclose(_to_ball(lorentz, poincare, flat_l).tensor, flat_p.tensor, atol=1e-8)


# ---------------------------------------------------------------------------------------
# hypll layers work unchanged and give the same function as on the Poincare ball
# ---------------------------------------------------------------------------------------
def test_hconv_hrelu_hmaxpool_match_poincare() -> None:
    lorentz, poincare = _manifolds()
    torch.manual_seed(14)

    conv_p = HConvolution2d(3, 8, kernel_size=3, manifold=poincare, padding=1)
    conv_l = HConvolution2d(3, 8, kernel_size=3, manifold=lorentz, padding=1)
    # Identical float64 parameters for both layers.
    weights = torch.randn(27, 8, dtype=DTYPE) * 0.2
    bias = torch.randn(8, dtype=DTYPE) * 0.1
    for conv in (conv_p, conv_l):
        conv.weights.tensor.data = weights.clone()
        conv.bias.data = bias.clone()
    relu_p, relu_l = HReLU(manifold=poincare), HReLU(manifold=lorentz)
    pool_p = HMaxPool2d(kernel_size=2, manifold=poincare, stride=2)
    pool_l = HMaxPool2d(kernel_size=2, manifold=lorentz, stride=2)

    img = torch.rand(2, 3, 8, 8, dtype=DTYPE)
    xp = poincare.expmap(TangentTensor(img, manifold=poincare, man_dim=1))
    xl = lorentz.expmap(TangentTensor(2 * img, manifold=lorentz, man_dim=1))

    out_p = pool_p(relu_p(conv_p(xp)))
    out_l = pool_l(relu_l(conv_l(xl)))
    assert out_l.shape == out_p.shape == (2, 8, 4, 4)
    assert torch.allclose(_to_ball(lorentz, poincare, out_l).tensor, out_p.tensor, atol=1e-8)


def test_hconv_backward_is_finite() -> None:
    lorentz, _ = _manifolds()
    conv = HConvolution2d(3, 4, kernel_size=3, manifold=lorentz, padding=1)
    img = torch.rand(2, 3, 16, 16, requires_grad=True)
    x = lorentz.expmap(TangentTensor(img, manifold=lorentz, man_dim=1))
    loss = lorentz.logmap(None, conv(x)).tensor.pow(2).mean()
    loss.backward()
    assert torch.isfinite(img.grad).all()
    assert torch.isfinite(conv.weights.tensor.grad).all()


def test_trainable_curvature_gets_gradient() -> None:
    curvature = Curvature(value=0.5, requires_grad=True)
    lorentz = Lorentz(c=curvature)
    conv = HConvolution2d(3, 4, kernel_size=3, manifold=lorentz, padding=1)
    x = lorentz.expmap(TangentTensor(torch.rand(1, 3, 8, 8), manifold=lorentz, man_dim=1))
    lorentz.logmap(None, conv(x)).tensor.sum().backward()
    assert curvature.value.grad is not None and torch.isfinite(curvature.value.grad)


def test_riemannian_adam_on_lorentz_embedding() -> None:
    from hypll.nn import HEmbedding
    from hypll.optim import RiemannianAdam

    lorentz, _ = _manifolds()
    emb = HEmbedding(num_embeddings=10, embedding_dim=3, manifold=lorentz)
    opt = RiemannianAdam(emb.parameters(), lr=1e-2)
    target = ManifoldTensor(torch.zeros(1, 3), manifold=lorentz)
    idx = torch.arange(10)
    loss_before = lorentz.dist(emb(idx), target).pow(2).mean().item()
    for _ in range(20):
        opt.zero_grad()
        loss = lorentz.dist(emb(idx), target).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(emb.weight.tensor).all()
    assert loss.item() < loss_before


def test_time_coordinate_positive_and_consistent() -> None:
    c = torch.tensor(2.0, dtype=DTYPE)
    x = torch.randn(5, 3, dtype=DTYPE)
    x0 = time_coord(x, c)
    assert (x0 > 0).all()
    assert torch.allclose(
        -(x0.squeeze(-1) ** 2) + x.pow(2).sum(-1), -1 / c * torch.ones(5, dtype=DTYPE)
    )
