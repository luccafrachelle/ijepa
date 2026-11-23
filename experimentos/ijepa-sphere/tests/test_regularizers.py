import importlib.util
import os

import pytest
import torch

from ijepa_sphere._official_regs import SIGReg, SUSReg
from ijepa_sphere.regularizers import Regularizer, build_regularizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(ROOT, "third_party", "SPHEREJEPA", "losses")


def _load(name):
    path = os.path.join(OFFICIAL, name + ".py")
    if not os.path.exists(path):
        pytest.skip("third_party/SPHEREJEPA not present")
    spec = importlib.util.spec_from_file_location("official_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _uniform_sphere(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)  # Regularizer normalizes for susreg


def _concentrated(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros(d); mu[0] = 1.0
    return mu + 0.1 * torch.randn(n, d, generator=g)


def test_identical_to_official_classes():
    sp, lj = _load("spherejepa"), _load("lejepa")
    z = torch.randn(256, 256)
    for ours, theirs in [(SUSReg(dim=256, knots=151), sp.SUSReg(dim=256, knots=151)),
                         (SIGReg(knots=51), lj.SIGReg(knots=51))]:
        assert torch.allclose(ours.phi, theirs.phi) and torch.allclose(ours.weights, theirs.weights)
        for _ in range(3):  # same generator sequence step by step
            assert torch.equal(ours(z), theirs(z))


def test_wrapper_matches_official_loss_modules():
    """Regularizer == the Lreg computed inside the official SPHEREJEPA / LEJEPA modules."""
    sp, lj = _load("spherejepa"), _load("lejepa")

    class P:  # params namespace
        lambda_reg = 0.05; projection_dim = 256; seed = 0; global_views = 2; num_projections = 1024

    zs = [torch.randn(64, 256) for _ in range(2)]
    P.knots = 151
    _, m = sp.SPHEREJEPA(P)(zs)
    assert m["train/Lreg"] == pytest.approx(Regularizer("susreg")(torch.stack(zs)).item(), rel=1e-5)
    P.knots = 51
    _, m = lj.LEJEPA(P)(zs)
    assert m["train/Lreg"] == pytest.approx(Regularizer("sigreg")(torch.stack(zs)).item(), rel=1e-5)


def test_defaults():
    s, g = Regularizer("susreg"), Regularizer("sigreg")
    assert len(s.stat.t) == 151 and len(g.stat.t) == 51
    assert s.stat.num_projections == g.stat.num_projections == 1024
    assert float(s.stat.t[-1]) == 5.0 and float(g.stat.t[-1]) == 5.0
    assert build_regularizer("none") is None and build_regularizer(None) is None
    with pytest.raises(ValueError):
        Regularizer("susreg", dim=512)


def test_susreg_uniform_small_concentrated_large():
    reg = Regularizer("susreg")
    u = reg(_uniform_sphere(1024, 256, 1)).item()
    c = reg(_concentrated(1024, 256, 2)).item()
    u4 = Regularizer("susreg")(_uniform_sphere(4096, 256, 3)).item()
    print(f"\nSUSReg (x B): uniform sphere B=1024 {u:.4f}, B=4096 {u4:.4f}, concentrated {c:.4f}")
    # the statistic is multiplied by B, so under the null it is O(1) (not growing with B)
    assert u < 2.0 and u4 < 2.0 and c > 100 * u


def test_sigreg_gaussian_small_concentrated_large():
    reg = Regularizer("sigreg")
    u = reg(torch.randn(1024, 256, generator=torch.Generator().manual_seed(1))).item()
    c = reg(_concentrated(1024, 256, 2)).item()
    u4 = Regularizer("sigreg")(torch.randn(4096, 256, generator=torch.Generator().manual_seed(3))).item()
    print(f"\nSIGReg (x N): N(0,I) N=1024 {u:.4f}, N=4096 {u4:.4f}, concentrated {c:.4f}")
    assert u < 2.0 and u4 < 2.0 and c > 100 * u


def test_susreg_is_scale_invariant_sigreg_is_not():
    z = torch.randn(256, 256)
    s1, s2 = Regularizer("susreg"), Regularizer("susreg")
    assert torch.allclose(s1(z), s2(10 * z), rtol=1e-4)
    g1, g2 = Regularizer("sigreg"), Regularizer("sigreg")
    assert not torch.allclose(g1(z), g2(10 * z))


def test_fp32_under_autocast_and_grad():
    z = torch.randn(128, 256, requires_grad=True)
    reg = Regularizer("susreg")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = reg(z.to(torch.bfloat16))
    assert out.dtype == torch.float32
    reg2 = Regularizer("sigreg")
    reg2(z).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
