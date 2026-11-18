"""Distribution regularizers: SUSReg (sphere, SPHERE-JEPA) and SIGReg (Gaussian, LeJEPA).

The statistic classes are the official ones, unchanged (`_official_regs.py`).
`Regularizer` reproduces exactly how the official loss modules call them:

  susreg:  Lreg = SUSReg(F.normalize(z, dim=-1)) * B     (losses/spherejepa.py: SPHEREJEPA.forward)
  sigreg:  Lreg = SIGReg(z)                               (losses/lejepa.py: LEJEPA.forward;
                                                           the factor N is inside SIGReg)

With several views (z of shape [V, B, D]) the per-view values are averaged,
as in the official modules. Defaults: num_projections=1024, t_max=5, seed=0,
knots=151 for susreg and 51 for sigreg (configs/imagenet100_tinyvit_*.yaml).

Precision: the official runs are fp32 (Lightning default). The statistic uses
cos/sin of values up to ~t_max*sqrt(d)*|proj|, which bf16 cannot resolve, so
`Regularizer.forward` casts z to float32 and disables autocast. The math is
otherwise unchanged.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ijepa_sphere._official_regs import SIGReg, SUSReg

DEFAULT_KNOTS = {"susreg": 151, "sigreg": 51}


class Regularizer(nn.Module):
    """Callable regularizer: `reg(z) -> scalar` (fp32).

    Args:
        kind: 'susreg' or 'sigreg'.
        dim: embedding dim (must equal z.shape[-1]); SUSReg uses the exact
             sphere CF only for dim <= 256 (else the official code silently
             switches to the Gaussian approximation) -> we refuse dim > 256.
        num_projections, knots, t_max, seed: passed to the official class.
    Input z: [B, D] or [V, B, D] (raw projector output, NOT normalized).
    """

    def __init__(self, kind, dim=256, num_projections=1024, knots=None, t_max=5, seed=0):
        super().__init__()
        kind = kind.lower()
        if knots is None:
            knots = DEFAULT_KNOTS[kind]
        self.kind, self.dim = kind, dim
        if kind == "susreg":
            if dim > 256:
                raise ValueError("SUSReg with dim > 256 uses the Gaussian approximation; use dim <= 256")
            self.stat = SUSReg(num_projections=num_projections, dim=dim, t_max=t_max, knots=knots, seed=seed)
        elif kind == "sigreg":
            self.stat = SIGReg(num_projections=num_projections, t_max=t_max, knots=knots, seed=seed)
        else:
            raise ValueError(f"unknown regularizer {kind!r}")

    def forward(self, z):
        if z.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {tuple(z.shape)}")
        views = z.unsqueeze(0) if z.dim() == 2 else z
        with torch.autocast(device_type=z.device.type, enabled=False):
            views = views.float()
            if self.kind == "susreg":
                views = F.normalize(views, dim=-1)
                B = views.shape[1]
                return torch.stack([self.stat(zi) for zi in views]).mean() * B
            return torch.stack([self.stat(zi) for zi in views]).mean()


def build_regularizer(kind, **kwargs):
    """Returns None for kind in (None, 'none'), else a `Regularizer`."""
    if kind is None or str(kind).lower() == "none":
        return None
    return Regularizer(kind, **kwargs)
