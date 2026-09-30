"""Fast CPU checks of the three model variants (run: pytest tests -q)."""
import copy

import pytest
import torch

from hypdenoise.models import build_model, count_params
from hypdenoise.utils import load_config


def _cfg(encoder):
    cfg = load_config("configs/base.yaml")
    cfg["model"]["encoder"] = encoder
    cfg["model"]["base_channels"] = 4  # tiny for speed
    return cfg


@pytest.mark.parametrize("encoder", ["euclidean", "poincare", "lorentz"])
def test_forward_shape_and_identity_at_init(encoder):
    model = build_model(_cfg(encoder)).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # zero-initialised head -> the untrained denoiser is the identity
    assert torch.allclose(y, x)


def test_hyperbolic_models_have_same_parameter_count():
    assert count_params(build_model(_cfg("poincare"))) == count_params(build_model(_cfg("lorentz")))


def test_poincare_and_lorentz_compute_the_same_function():
    """Core claim of the paper setup: at identical weights, the Lorentz model is the
    isometric counterpart of the Poincare model, so the outputs match (float64)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        p = build_model(_cfg("poincare")).eval()
        torch.nn.init.normal_(p.head.weight, std=0.1)  # non-trivial head so the encoder matters
        l = build_model(_cfg("lorentz")).eval()
        l.load_state_dict(copy.deepcopy(p.state_dict()))
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            fp, fl = p.encoder(x), l.encoder(x)
            for a, b in zip(fp, fl):
                assert torch.allclose(a, b, atol=1e-8), (a - b).abs().max()
            assert torch.allclose(p(x), l(x), atol=1e-8)
    finally:
        torch.set_default_dtype(old)
