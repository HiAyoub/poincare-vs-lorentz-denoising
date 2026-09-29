import functools
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, empty, eye, no_grad
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold
from torch.nn.init import normal_, zeros_

from hypll.manifolds.base import Manifold
from hypll.manifolds.euclidean import Euclidean
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor
from hypll.utils.math import beta_func
from hypll.utils.tensor_utils import (
    check_dims_with_broadcasting,
    check_if_man_dims_match,
    check_tangent_tensor_positions,
)

from .math.diffgeom import (
    cdist,
    dist,
    euc_to_tangent,
    expmap,
    expmap0,
    from_ambient,
    inner,
    logmap,
    logmap0,
    lorentz_to_poincare,
    poincare_to_lorentz,
    project,
    to_ambient,
    transp,
)
from .math.linalg import lorentz_fully_connected, lorentz_hyperplane_dists
from .math.stats import frechet_mean, frechet_variance, midpoint


class Lorentz(Manifold):
    """Class representing the Lorentz (hyperboloid) model of hyperbolic space.

    The hyperboloid of curvature -c is {(x_0, x_s) : -x_0^2 + ||x_s||^2 = -1/c, x_0 > 0}.
    Points are stored by their spatial coordinates x_s only, and the time coordinate
    x_0 = sqrt(1/c + ||x_s||^2) is implicit. As a result, a manifold tensor with n
    channels represents points of n-dimensional hyperbolic space, just as for the
    PoincareBall, and every hypll layer can be used with this manifold unchanged.
    Use ``to_ambient`` to obtain the full (n+1)-dimensional coordinates.

    All neural network operations (hyperplane distances, fully connected layers,
    beta-concatenation) are the exact Lorentz counterparts of the HNN++ operations of
    the PoincareBall: with identical parameters they compute the same function, up to the
    isometry between the two models. The practical differences are numerical: points are
    not confined to a ball of radius 1/sqrt(c), so no clipping near the boundary is needed.

    Tangent vectors at the origin are Euclidean vectors whose norm equals their
    Riemannian norm. The PoincareBall uses a conformal factor of 2 at the origin, so a
    Poincare tangent vector u at the origin corresponds to the Lorentz tangent vector 2u.

    Attributes:
        c:
            Curvature of the manifold.

    """

    def __init__(self, c: Curvature):
        """Initializes an instance of the Lorentz manifold.

        Examples:
            >>> from hypll.manifolds.lorentz import Curvature, Lorentz
            >>> curvature = Curvature(value=1.0)
            >>> manifold = Lorentz(c=curvature)

        """
        super(Lorentz, self).__init__()
        self.c = c

    def project(self, x: ManifoldTensor, eps: float = -1.0) -> ManifoldTensor:
        """Guards against overflow by radially clamping points so that
        sqrt(c) * d(o, x) <= eps (a dtype-dependent default when eps < 0)."""
        new_tensor = project(x=x.tensor, c=self.c(), dim=x.man_dim, eps=eps)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def expmap(self, v: TangentTensor) -> ManifoldTensor:
        dim = v.broadcasted_man_dim
        if v.manifold_points is None:
            new_tensor = expmap0(v=v.tensor, c=self.c(), dim=dim)
        else:
            new_tensor = expmap(x=v.manifold_points.tensor, v=v.tensor, c=self.c(), dim=dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=dim)

    def logmap(self, x: Optional[ManifoldTensor], y: ManifoldTensor) -> TangentTensor:
        if x is None:
            dim = y.man_dim
            new_tensor = logmap0(y=y.tensor, c=self.c(), dim=y.man_dim)
        else:
            dim = check_dims_with_broadcasting(x, y)
            new_tensor = logmap(x=x.tensor, y=y.tensor, c=self.c(), dim=dim)
        return TangentTensor(data=new_tensor, manifold_points=x, manifold=self, man_dim=dim)

    def transp(self, v: TangentTensor, y: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(v, y)
        if v.manifold_points is None:
            # Tangent vectors at the origin: the origin has spatial coordinates 0.
            x = torch.zeros_like(y.tensor)
        else:
            x = v.manifold_points.tensor
        tangent_vectors = transp(x=x, y=y.tensor, v=v.tensor, c=self.c(), dim=dim)
        return TangentTensor(
            data=tangent_vectors,
            manifold_points=y,
            manifold=self,
            man_dim=dim,
        )

    def dist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        dim = check_dims_with_broadcasting(x, y)
        return dist(x=x.tensor, y=y.tensor, c=self.c(), dim=dim)

    def inner(
        self, u: TangentTensor, v: TangentTensor, keepdim: bool = False, safe_mode: bool = True
    ) -> Tensor:
        dim = check_dims_with_broadcasting(u, v)
        if safe_mode:
            check_tangent_tensor_positions(u, v)

        if u.manifold_points is None:
            # At the origin the metric is Euclidean.
            return (u.tensor * v.tensor).sum(dim=dim, keepdim=keepdim)

        return inner(
            x=u.manifold_points.tensor, u=u.tensor, v=v.tensor, c=self.c(), dim=dim, keepdim=keepdim
        )

    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor) -> TangentTensor:
        dim = check_dims_with_broadcasting(x, u)
        tangent_vectors = euc_to_tangent(x=x.tensor, u=u.tensor, c=self.c(), dim=x.man_dim)
        return TangentTensor(
            data=tangent_vectors,
            manifold_points=x,
            manifold=self,
            man_dim=dim,
        )

    def hyperplane_dists(self, x: ManifoldTensor, z: ManifoldTensor, r: Optional[Tensor]) -> Tensor:
        if x.man_dim != 1 or z.man_dim != 0:
            raise ValueError(
                f"Expected the manifold dimension of the inputs to be 1 and the manifold "
                f"dimension of the hyperplane orientations to be 0, but got {x.man_dim} and "
                f"{z.man_dim}, respectively"
            )
        return lorentz_hyperplane_dists(x=x.tensor, z=z.tensor, r=r, c=self.c(), dim=x.man_dim)

    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]
    ) -> ManifoldTensor:
        if z.man_dim != 0:
            raise ValueError(
                f"Expected the manifold dimension of the hyperplane orientations to be 0, but got "
                f"{z.man_dim} instead"
            )
        new_tensor = lorentz_fully_connected(
            x=x.tensor, z=z.tensor, bias=bias, c=self.c(), dim=x.man_dim
        )
        new_tensor = ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)
        return self.project(new_tensor)

    def frechet_mean(
        self,
        x: ManifoldTensor,
        batch_dim: Union[int, list[int]] = 0,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        output_man_dim = x.man_dim - sum(bd < x.man_dim for bd in batch_dim)
        new_tensor = frechet_mean(
            x=x.tensor, c=self.c(), vec_dim=x.man_dim, batch_dim=batch_dim, keepdim=keepdim
        )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=output_man_dim)

    def midpoint(
        self,
        x: ManifoldTensor,
        batch_dim: int = 0,
        w: Optional[Tensor] = None,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]

        if x.man_dim in batch_dim:
            raise ValueError(
                f"Tried to aggregate over dimensions {batch_dim}, but input has manifold "
                f"dimension {x.man_dim} and cannot aggregate over this dimension"
            )

        # Output manifold dimension is shifted left for each batch dim that disappears
        man_dim_shift = sum(bd < x.man_dim for bd in batch_dim)
        new_man_dim = x.man_dim - man_dim_shift if not keepdim else x.man_dim

        new_tensor = midpoint(
            x=x.tensor, c=self.c(), man_dim=x.man_dim, batch_dim=batch_dim, w=w, keepdim=keepdim
        )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=new_man_dim)

    def frechet_variance(
        self,
        x: ManifoldTensor,
        mu: Optional[ManifoldTensor] = None,
        batch_dim: Union[int, list[int]] = -1,
        keepdim: bool = False,
    ) -> Tensor:
        if mu is not None:
            mu = mu.tensor

        return frechet_variance(
            x=x.tensor,
            c=self.c(),
            mu=mu,
            vec_dim=x.man_dim,
            batch_dim=batch_dim,
            keepdim=keepdim,
        )

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[Parameter]]:
        # Same parameterisation as the PoincareBall (HNN++): Euclidean hyperplane
        # orientations and scalar offsets. This keeps models interchangeable.
        weight = ManifoldParameter(
            data=empty(in_features, out_features),
            manifold=Euclidean(),
            man_dim=0,
        )

        if bias:
            b = Parameter(data=empty(out_features))
        else:
            b = None

        return weight, b

    def reset_parameters(self, weight: ManifoldParameter, bias: Optional[Parameter]) -> None:
        in_features, out_features = weight.size()
        if in_features <= out_features:
            with no_grad():
                weight.tensor.copy_(1 / 2 * eye(in_features, out_features))
        else:
            normal_(
                weight.tensor,
                mean=0,
                std=(2 * in_features * out_features) ** -0.5,
            )
        if bias is not None:
            zeros_(bias)

    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> ManifoldTensor:
        # Beta-concatenation of the patches in the tangent space at the origin, as for the
        # PoincareBall. Zero padding in the tangent space corresponds to padding with the
        # origin of the manifold.
        in_channels = input.size(1)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kernel_vol = kernel_size[0] * kernel_size[1]

        beta_ni = beta_func(in_channels / 2, 1 / 2)
        beta_n = beta_func(in_channels * kernel_vol / 2, 1 / 2)

        input = self.logmap(x=None, y=input)
        input.tensor = input.tensor * beta_n / beta_ni
        new_tensor = unfold(
            input=input.tensor,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        new_tensor = TangentTensor(data=new_tensor, manifold_points=None, manifold=self, man_dim=1)
        return self.expmap(new_tensor)

    def flatten(self, x: ManifoldTensor, start_dim: int = 1, end_dim: int = -1) -> ManifoldTensor:
        """Flattens a manifold tensor by reshaping it. If start_dim or end_dim are passed,
        only dimensions starting with start_dim and ending with end_dim are flattend.

        If the manifold dimension of the input tensor is among the dimensions which
        are flattened, applies beta-concatenation to the points on the manifold.
        Otherwise simply flattens the tensor using torch.flatten.

        Updates the manifold dimension if necessary.

        """
        start_dim = x.dim() + start_dim if start_dim < 0 else start_dim
        end_dim = x.dim() + end_dim if end_dim < 0 else end_dim

        # Get the range of dimensions to flatten.
        dimensions_to_flatten = x.shape[start_dim + 1 : end_dim + 1]

        if start_dim <= x.man_dim and end_dim >= x.man_dim:
            # Use beta concatenation to flatten the manifold dimension of the tensor.
            tangents = self.logmap(None, x)
            n_i = x.shape[x.man_dim]
            n = n_i * functools.reduce(lambda a, b: a * b, dimensions_to_flatten, 1)
            beta_n = beta_func(n / 2, 0.5)
            beta_n_i = beta_func(n_i / 2, 0.5)
            tangents.tensor = torch.flatten(
                input=tangents.tensor,
                start_dim=start_dim,
                end_dim=end_dim,
            )
            tangents.tensor = tangents.tensor * beta_n / beta_n_i
            tangents.man_dim = start_dim
            return self.expmap(tangents)
        else:
            flattened = torch.flatten(
                input=x.tensor,
                start_dim=start_dim,
                end_dim=end_dim,
            )
            man_dim = x.man_dim if end_dim > x.man_dim else x.man_dim - len(dimensions_to_flatten)
            return ManifoldTensor(data=flattened, manifold=x.manifold, man_dim=man_dim)

    def cdist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        return cdist(x=x.tensor, y=y.tensor, c=self.c())

    def cat(
        self,
        manifold_tensors: Union[Tuple[ManifoldTensor, ...], List[ManifoldTensor]],
        dim: int = 0,
    ) -> ManifoldTensor:
        check_if_man_dims_match(manifold_tensors)
        if dim == manifold_tensors[0].man_dim:
            tangent_tensors = [self.logmap(None, t) for t in manifold_tensors]
            ns = torch.tensor([t.shape[t.man_dim] for t in manifold_tensors])
            n = ns.sum()
            beta_ns = beta_func(ns / 2, 0.5)
            beta_n = beta_func(n / 2, 0.5)
            cat = torch.cat(
                [(t.tensor * beta_n) / beta_n_i for (t, beta_n_i) in zip(tangent_tensors, beta_ns)],
                dim=dim,
            )
            new_tensor = TangentTensor(data=cat, manifold=self, man_dim=dim)
            return self.expmap(new_tensor)
        else:
            cat = torch.cat([t.tensor for t in manifold_tensors], dim=dim)
            man_dim = manifold_tensors[0].man_dim
            return ManifoldTensor(data=cat, manifold=self, man_dim=man_dim)

    # ------------------------------------------------------------------------------------
    # Lorentz-specific helpers
    # ------------------------------------------------------------------------------------
    def to_ambient(self, x: ManifoldTensor) -> Tensor:
        """Returns the (n+1)-dimensional ambient coordinates (x_0, x_s) of the points."""
        return to_ambient(x.tensor, c=self.c(), dim=x.man_dim)

    def from_ambient(self, x: Tensor, man_dim: int = -1) -> ManifoldTensor:
        """Creates a manifold tensor from ambient coordinates (x_0, x_s) by dropping x_0."""
        return ManifoldTensor(data=from_ambient(x, dim=man_dim), manifold=self, man_dim=man_dim)

    def to_poincare_ball(self, x: ManifoldTensor, target: Manifold) -> ManifoldTensor:
        """Maps points to the Poincare ball ``target`` (with the same curvature) using the
        standard isometry p = x_s / (1 + sqrt(c) x_0)."""
        new_tensor = lorentz_to_poincare(x.tensor, c=self.c(), dim=x.man_dim)
        return ManifoldTensor(data=new_tensor, manifold=target, man_dim=x.man_dim)

    def from_poincare_ball(self, p: ManifoldTensor) -> ManifoldTensor:
        """Maps points of a Poincare ball (with the same curvature) to this manifold using
        the standard isometry x_s = 2p / (1 - c ||p||^2)."""
        new_tensor = poincare_to_lorentz(p.tensor, c=self.c(), dim=p.man_dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=p.man_dim)
