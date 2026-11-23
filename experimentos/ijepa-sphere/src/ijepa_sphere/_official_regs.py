# Copied verbatim from darktanuki/SPHEREJEPA @ 9a55566:
#   - SUSReg from losses/spherejepa.py
#   - SIGReg from losses/lejepa.py
# No license file in the upstream repo (academic use). Do not edit these classes;
# wrappers live in ijepa_sphere.regularizers.
import numpy as np
import torch
from torch import nn

class SUSReg(torch.nn.Module):
    """
    """

    def __init__(self, num_projections=1024, dim=256, t_max=5, knots=51, seed=0):
        super().__init__()
        self.seed = seed
        self.num_projections = num_projections
        self.dim = dim
        use_gaussian_approx = self.dim>256

        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        window = torch.exp(-t.square()/2)

        g = torch.Generator()
        g.manual_seed(self.seed)
        self._generator = g

        phi_np  = self._phi_exact_numpy(t.cpu().numpy().astype(np.float64), self.dim)
        phi_opt = torch.from_numpy(phi_np.astype(np.float32))

        self.register_buffer("_seed", torch.tensor(self.seed, dtype=torch.int64))
        self.register_buffer("t", t)
        self.register_buffer("phi", window if use_gaussian_approx else phi_opt)
        self.register_buffer("weights", weights * window)

    @staticmethod
    def _phi_exact_numpy(t, d, s_small= 0.5, series_terms = 6):
        """
        Stable exact CF of V = sqrt(d) * <x,y>, x,y ~ Unif(S^{d-1}).

        Uses:
          - exact Taylor expansion for small s = t*sqrt(d)
          - log-domain Bessel evaluation for larger s

        This avoids numerical underflow while remaining exact.
        """
        import numpy as np
        from scipy.special import jv, gammaln

        t = np.asarray(t, dtype=np.float64)
        v = d / 2.0 - 1.0
        s = t * np.sqrt(d)

        out = np.empty_like(s, dtype=np.float64)
        out[s == 0] = 1.0

        nz_idx = np.where(s != 0)[0]
        snz = s[nz_idx]

        small = np.abs(snz) < s_small
        large = ~small

        # --- Small-s: exact Taylor series ---
        # phi(s) = sum_{k>=0} (-1)^k / (k! (d/2)_k) * (s^2/4)^k
        if np.any(small):
            idx = nz_idx[small]
            ss = s[idx]
            x = (ss * ss) / 4.0
            a0 = d / 2.0

            phi_s = np.ones_like(ss)
            term = np.ones_like(ss)
            fact = 1.0
            poch = 1.0
            for k in range(1, series_terms):
                fact *= k
                poch *= (a0 + (k - 1))
                term *= (-x)
                phi_s += term / (fact * poch)
            out[idx] = phi_s

        # --- Large-s: log-domain exact Bessel formula ---
        if np.any(large):
            idx = nz_idx[large]
            sl = s[idx]
            J = jv(v, sl)
            absJ = np.maximum(np.abs(J), 1e-300)
            log_abs = gammaln(d / 2.0) + v * np.log(2.0 / np.abs(sl)) + np.log(absJ)
            out[idx] = np.sign(J) * np.exp(log_abs)

        return out
        
    def forward(self, proj):
        if self._generator.device != proj.device:
            self._generator = torch.Generator(device=proj.device).manual_seed(int(self._seed)) #type:ignore 
        A = torch.randn(
            proj.size(-1),
            self.num_projections,
            device=proj.device,
            dtype=proj.dtype,
            generator=self._generator
        )
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t * (self.dim ** 0.5)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) 
        return statistic.mean()

class SIGReg(torch.nn.Module):
    def __init__(self, num_projections=1024,  t_max=5, knots=51, seed=0):
        super().__init__()
        self.num_projections = num_projections
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self._seed = seed
        g = torch.Generator()
        g.manual_seed(self._seed)
        self._generator = g

    def forward(self, proj):
        if self._generator.device != proj.device:
            self._generator = torch.Generator(device=proj.device).manual_seed(int(self._seed))

        #proj : (B,d)
        A = torch.randn(proj.size(-1), self.num_projections, device=proj.device, dtype=proj.dtype, generator=self._generator) #(d,nb_proj)
        A = A.div_(A.norm(p=2, dim=0)) #(d,nb_proj)
        x_t = (proj @ A).unsqueeze(-1) * self.t #(B, nb_proj, T)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square() #(nb_proj, T)
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
    
