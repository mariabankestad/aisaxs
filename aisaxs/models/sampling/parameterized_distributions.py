

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

SQRT2_ = math.sqrt(2)
class BaseRadiusSampler(nn.Module):
    """Maps u∈(0,1) -> r∈(0,1). Subclass this to implement new families."""
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def parameters_summary(self) -> str:
        """Short text summary for logging."""
        return self.__class__.__name__


class PWLCDFSampler(BaseRadiusSampler):
    """
    Piecewise-linear CDF on [0,1]. Defines bin masses via softmax logits.
    Inverse-CDF sampling: u -> r. Initializes exactly uniform.
    """
    def __init__(self, K=32, device="cuda", dtype=torch.float32):
        super().__init__()
        self.K = int(K)
        self.logits = nn.Parameter(torch.zeros(self.K, device=device, dtype=dtype))  # 0 -> uniform
        edges = torch.linspace(0., 1., self.K+1, device=device, dtype=dtype)
        self.register_buffer("edges", edges)
        self.register_buffer("widths", edges[1:] - edges[:-1])  # constant 1/K

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clamp(1e-8, 1-1e-8)
        w = F.softmax(self.logits, dim=0)                                  # bin masses, sum=1
        c = torch.cat([torch.zeros(1, device=u.device, dtype=u.dtype),
                       torch.cumsum(w, dim=0)], dim=0)                     # CDF knots [K+1]

        # find bin j: c[j] <= u < c[j+1]
        j = torch.clamp(torch.searchsorted(c, u, right=True) - 1, 0, self.K-1)

        c0   = c[j]                      # left CDF value
        mass = w[j]                      # bin mass
        x0   = self.edges[j]             # left edge in r-space
        dx   = self.widths[j]            # bin width (1/K)
        t    = (u - c0) / (mass + 1e-12) # fractional mass inside bin
        r    = x0 + t * dx               # linear within bin
        return r.clamp(1e-8, 1-1e-8)
    def parameters_summary(self) -> str:
        w = torch.softmax(self.logits, dim=0)
        H = -(w * (w.clamp_min(1e-12)).log()).sum()
        Hn = H / torch.log(torch.tensor(float(len(w)), device=w.device))
        return f"PWL(K={self.K}, Hn={float(Hn):.2f})"
    


class KumarSampler(BaseRadiusSampler):
    """Kumaraswamy(α,β) reparameterized via u∈(0,1)."""
    def __init__(self, alpha=0.7, beta=7.0, device="cuda", dtype=torch.float32):
        super().__init__()
        self.register_buffer("_one", torch.ones((), device=device, dtype=dtype))
        def inv_softplus(x): 
            t = torch.tensor(x, device=device, dtype=dtype)
            return torch.log(torch.expm1(t))
        self.alpha_raw = nn.Parameter(inv_softplus(alpha))
        self.beta_raw  = nn.Parameter(inv_softplus(beta))
    def _ab(self):
        a = F.softplus(self.alpha_raw) + 1e-6
        b = F.softplus(self.beta_raw)  + 1e-6
        return a, b
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clamp(1e-8, 1-1e-8)
        a, b = self._ab()
        # Kumar inverse CDF: r = (1 - (1-u)^{1/β})^{1/α}
        r = (self._one - (self._one - u).pow(self._one / b)).pow(self._one / a)
        return r.clamp(1e-8, 1-1e-8)
    def parameters_summary(self) -> str:
        a, b = self._ab()
        return f"Kumar(α={float(a):.3f}, β={float(b):.3f})"
    

class RQSSampler(BaseRadiusSampler):
    """
    Monotone RQS T:[0,1]->[0,1] with exact identity initialization.
    - widths/heights are softmax-normalized; zeros -> uniform exactly
    - slopes softplus(raw) with raw set s.t. slopes == 1.0 at init
    """
    def __init__(self, K=16, device="cuda", dtype=torch.float32):
        super().__init__()
        self.K = int(K)
        self.register_buffer("_zero", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("_one",  torch.tensor(1.0, device=device, dtype=dtype))

        self.widths_raw  = nn.Parameter(torch.zeros(K,   device=device, dtype=dtype))  # softmax -> uniform
        self.heights_raw = nn.Parameter(torch.zeros(K,   device=device, dtype=dtype))  # softmax -> uniform

        # choose raw so softplus(raw) == 1.0 exactly
        def inv_softplus(y):
            y = torch.as_tensor(y, device=device, dtype=dtype)
            return torch.log(torch.expm1(y))
        self.slopes_raw  = nn.Parameter(inv_softplus(1.0) * torch.ones(K+1, device=device, dtype=dtype))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clamp(1e-8, 1-1e-8)
        # uniform at init (softmax(0)=1/K)
        w = F.softmax(self.widths_raw,  dim=0)           # [K], sum=1
        h = F.softmax(self.heights_raw, dim=0)           # [K], sum=1
        s = F.softplus(self.slopes_raw)                  # [K+1], =1.0 at init

        xk = torch.cat([self._zero[None], torch.cumsum(w, dim=0)], dim=0)  # [K+1]
        yk = torch.cat([self._zero[None], torch.cumsum(h, dim=0)], dim=0)  # [K+1]

        j = torch.clamp(torch.searchsorted(xk, u, right=True) - 1, 0, self.K-1)

        x0, x1 = xk[j],   xk[j+1]
        y0, y1 = yk[j],   yk[j+1]
        s0, s1 = s[j],    s[j+1]

        dx = x1 - x0
        dy = y1 - y0
        t  = (u - x0) / (dx + 1e-12)
        dj = dy / (dx + 1e-12)

        # Durkan et al. (2019) monotone RQS forward (stable)
        t1  = 1.0 - t
        num = s0 * t * t + dj * t * t1
        den = s0 * t * t + s1 * t1 * t1 + dj * t * t1
        r   = y0 + dy * (num / (den + 1e-12))

        # exact endpoints
        r = torch.where(u <= xk[0] + 1e-12, yk[0], r)
        r = torch.where(u >= xk[-1] - 1e-12, yk[-1], r)
        return r.clamp(1e-8, 1-1e-8)
    def parameters_summary(self) -> str:
        mean_slope = F.softplus(self.slopes_raw).mean().item()
        return f"RQS(K={self.K}, ⟨s⟩={mean_slope:.2f})"
    




class LogNormalSampler(BaseRadiusSampler):
    """
    LogNormal on physical radius r_phys, clamped to [trunc_A, trunc_B] (Å),
    then mapped to r01 in (0,1) using a *fixed* ML normalisation interval
    [ml_A, ml_B].

    - The *distribution parameters* (mean_A, std_A) live in [trunc_A, trunc_B]
      and [s_min, s_max].
    - The *ML model* always sees
          r01 = (r_phys - ml_A) / (ml_B - ml_A),
      with (ml_A, ml_B) typically = (100, 500) to match training.

    User-facing (physical) parameters:
      - mean_A:  physical mean radius in [trunc_A, trunc_B] (Å)
      - std_A:   physical std in [s_min, s_max] (Å)
    """

    def __init__(
        self,
        *,
        trunc_A: float = 120.0,       # physical support for the distribution
        trunc_B: float = 400.0,
        ml_A: float = 100.0,          # fixed range used for ML normalisation
        ml_B: float = 500.0,
        init_mean_A: float = 160.0,   # physical mean (Å), inside [trunc_A,trunc_B]
        init_std_A: float = 30.0,     # physical std (Å)
        s_min: float = 1.0,           # minimal allowed std (Å)
        s_max: float = 200.0,         # maximal allowed std (Å)
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert trunc_B > trunc_A, "trunc_B must be > trunc_A"
        assert ml_B > ml_A, "ml_B must be > ml_A"
        assert s_max > s_min >= 0.0, "Need 0 ≤ s_min < s_max"
        assert trunc_A >= ml_A and trunc_B <= ml_B, \
            "trunc interval must be inside ML interval to avoid r01 saturating at 0/1."
        self.register_buffer("_trunc_A", torch.as_tensor(trunc_A, device=device, dtype=dtype))
        self.register_buffer("_trunc_B", torch.as_tensor(trunc_B, device=device, dtype=dtype))

        # --- ML normalisation interval (fixed, used for mapping to r01) ---
        self.register_buffer("_ml_A", torch.as_tensor(ml_A, device=device, dtype=dtype))
        self.register_buffer("_ml_B", torch.as_tensor(ml_B, device=device, dtype=dtype))

        # std bounds in physical space
        self.register_buffer("_s_min", torch.as_tensor(s_min, device=device, dtype=dtype))
        self.register_buffer("_s_max", torch.as_tensor(s_max, device=device, dtype=dtype))

        def inv_sigmoid(y):
            y = torch.as_tensor(y, device=device, dtype=dtype)
            eps = 1e-6
            y = y.clamp(eps, 1.0 - eps)
            return torch.log(y / (1.0 - y))

        # ---- init raw_mean so that mean_A ≈ init_mean_A in [trunc_A, trunc_B] ----
        mean_frac0 = (init_mean_A - trunc_A) / (trunc_B - trunc_A)
        mean_frac0 = float(max(1e-3, min(1 - 1e-3, mean_frac0)))
        self.mean_raw = nn.Parameter(inv_sigmoid(mean_frac0))

        # ---- init raw_std so that std_A ≈ init_std_A in [s_min, s_max] ----
        t0 = (init_std_A - s_min) / (s_max - s_min)
        t0 = float(max(1e-3, min(1 - 1e-3, t0)))
        self.std_raw = nn.Parameter(inv_sigmoid(t0))

    # Convenience accessors
    @property
    def trunc_A(self):
        return self._trunc_A

    @property
    def trunc_B(self):
        return self._trunc_B

    @property
    def ml_A(self):
        return self._ml_A

    @property
    def ml_B(self):
        return self._ml_B

    # ---- internal helpers ----
    def _mean_std(self):
        """
        raw → physical (mean_A ∈ [trunc_A,trunc_B], std_A ∈ [s_min,s_max]).
        """
        A, B = self.trunc_A, self.trunc_B
        s_min, s_max = self._s_min, self._s_max

        mean_frac = torch.sigmoid(self.mean_raw)          # (0,1)
        mean_A = A + (B - A) * mean_frac                  # (A,B)

        t = torch.sigmoid(self.std_raw)                   # (0,1)
        std_A = s_min + (s_max - s_min) * t               # [s_min,s_max]

        return mean_A, std_A

    def _mu_sigma(self):
        """
        Convert physical (mean_A, std_A) to LogNormal parameters (mu, sigma).
        """
        mean_A, std_A = self._mean_std()

        eps = 1e-12
        ratio2 = (std_A / (mean_A + eps))**2
        sigma2 = torch.log(1.0 + ratio2)          # σ² = log(1 + (s/m)²)
        sigma = torch.sqrt(sigma2 + eps)
        mu = torch.log(mean_A + eps) - 0.5 * sigma2

        return mu, sigma

    @staticmethod
    def _phi_inv(u: torch.Tensor) -> torch.Tensor:
        #two = torch.tensor(2.0, device=u.device, dtype=u.dtype)
        return SQRT2_ * torch.erfinv(2.0 * u - 1.0)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        #two = torch.tensor(2.0, device=x.device, dtype=x.dtype)
        return 0.5 * (1.0 + torch.erf(x / SQRT2_))

    def _lognormal_cdf(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # LogNormal CDF: Phi((ln x - mu)/sigma)
        eps = 1e-12
        z = (torch.log(x.clamp_min(eps)) - mu) / (sigma + eps)
        return self._phi(z)

    def _sample_r_phys(self, u: torch.Tensor) -> torch.Tensor:
        """
        Correct conditioned LogNormal sampling on [trunc_A, trunc_B] with fixed (mu, sigma).
        """
        u = u.clamp(1e-8, 1.0 - 1e-8)
        mu, sigma = self._mu_sigma()

        A = self.trunc_A
        B = self.trunc_B

        # CDF bounds under the *same* mu,sigma
        Fa = self._lognormal_cdf(A, mu, sigma)
        Fb = self._lognormal_cdf(B, mu, sigma)

        span = (Fb - Fa).clamp_min(1e-12)

        # renormalize u into [Fa, Fb]
        u2 = (Fa + u * span).clamp(1e-8, 1.0 - 1e-8)

        # inverse CDF of lognormal
        z = self._phi_inv(u2)
        r_phys = torch.exp(mu + sigma * z)

        # numerical guard only
        return r_phys.clamp(A, B)

    # ---- main mapping: u∈(0,1) → r01∈(0,1) for the ML model ----
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: tensor in (0,1)
        returns r01 ∈ (0,1) to be injected into the *normalized* radius column
        expected by the ML model.

        Distribution support is [trunc_A,trunc_B], but normalisation uses
        the fixed ML range [ml_A, ml_B].
        """
        r_phys = self._sample_r_phys(u)

        # Map to [0,1] using the *ML* interval [ml_A, ml_B]
        denom = (self.ml_B - self.ml_A)
        r01 = ((r_phys - self.ml_A) / denom).clamp(1e-8, 1.0 - 1e-8)
        return r01

    # ---- summaries / exports (no grad) ----
    @torch.no_grad()
    def parameters_summary(self) -> str:
        mean_A, std_A = self._mean_std()
        return (
            f"LogNormal(clamp=[{float(self.trunc_A):.0f},{float(self.trunc_B):.0f}]Å, "
            f"mean={float(mean_A):.1f}Å, std={float(std_A):.1f}Å, "
            f"ML_norm=[{float(self.ml_A):.0f},{float(self.ml_B):.0f}]Å)"
        )

    @torch.no_grad()
    def truncated_mean_std_phys(self, n_est=4096, device=None, dtype=None):
        """
        Monte Carlo estimate of mean/std of the *effective* physical distribution
        r_phys ∈ [trunc_A,trunc_B] that is used in forward().
        """
        if device is None:
            device = self.trunc_A.device
        if dtype is None:
            dtype = self.trunc_A.dtype

        u = torch.linspace(0.5/n_est, 1.0 - 0.5/n_est,
                           n_est, device=device, dtype=dtype)
        r_phys = self._sample_r_phys(u)
        mean_phys = r_phys.mean()
        std_phys  = r_phys.std(unbiased=False)
        return mean_phys, std_phys

    @torch.no_grad()
    def physical_radius_params(self) -> dict:
        """
        Export:
        - mean_A, std_A: target physical parameters (based on [trunc_A,trunc_B])
        - trunc_A, trunc_B: effective support where mass lives
        - ml_A, ml_B: fixed ML normalisation interval
        """
        mean_A, std_A = self._mean_std()
        return {
            "radius_mean_A":   mean_A.detach().cpu(),
            "radius_std_A":    std_A.detach().cpu(),
            "radius_trunc_min": self.trunc_A.detach().cpu(),
            "radius_trunc_max": self.trunc_B.detach().cpu(),
            "radius_ml_min":    self.ml_A.detach().cpu(),
            "radius_ml_max":    self.ml_B.detach().cpu(),
        }


import torch
import torch.nn as nn
import math

SQRT2_ = 2.0 ** 0.5

def _inv_sigmoid_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))

class LogNormalSamplerBatched(nn.Module):
    """
    Radius distribution: LogNormal with parameters defined via (mean_A, cv).
      std_A = cv * mean_A

    - mean_A constrained to [mean_lo, mean_hi]
    - cv constrained to [cv_lo, cv_hi]
    - trunc_A/B define the *integration support* (conditioning), not learnable bounds
    - output is r01 in (0,1) using ML normalization interval [ml_A, ml_B]
    """

    def __init__(
        self,
        *,
        R: int,
        trunc_A: float,
        trunc_B: float,
        ml_A: float,
        ml_B: float,
        mean_lo: float,
        mean_hi: float,
        cv_lo: float = 0.10,
        cv_hi: float = 0.35,
        init_mean_A: float = None,
        init_cv: float = 0.20,
        learn_mean: bool = True,
        learn_cv: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.R = int(R)
        dev = torch.device(device)

        assert trunc_B > trunc_A
        assert ml_B > ml_A
        assert trunc_A >= ml_A and trunc_B <= ml_B, "trunc must lie within ML interval"
        assert mean_hi > mean_lo
        assert cv_hi > cv_lo > 0.0
        assert trunc_A <= mean_lo and mean_hi <= trunc_B, "mean bounds should lie inside truncation"

        self.register_buffer("_trunc_A", torch.tensor(trunc_A, device=dev, dtype=dtype))
        self.register_buffer("_trunc_B", torch.tensor(trunc_B, device=dev, dtype=dtype))
        self.register_buffer("_ml_A", torch.tensor(ml_A, device=dev, dtype=dtype))
        self.register_buffer("_ml_B", torch.tensor(ml_B, device=dev, dtype=dtype))

        self.register_buffer("_mean_lo", torch.tensor(mean_lo, device=dev, dtype=dtype))
        self.register_buffer("_mean_hi", torch.tensor(mean_hi, device=dev, dtype=dtype))
        self.register_buffer("_cv_lo",   torch.tensor(cv_lo,   device=dev, dtype=dtype))
        self.register_buffer("_cv_hi",   torch.tensor(cv_hi,   device=dev, dtype=dtype))

        if init_mean_A is None:
            init_mean_A = float(0.5 * (mean_lo + mean_hi))

        init_mean_A = float(max(mean_lo + 1e-8, min(mean_hi - 1e-8, init_mean_A)))
        init_cv     = float(max(cv_lo   + 1e-8, min(cv_hi   - 1e-8, init_cv)))

        # Map init to raw in R^1 via sigmoid boxing
        mean_frac0 = (init_mean_A - mean_lo) / (mean_hi - mean_lo)
        cv_frac0   = (init_cv     - cv_lo)   / (cv_hi   - cv_lo)

        mean_raw0 = _inv_sigmoid_clamped(torch.tensor(mean_frac0, device=dev, dtype=dtype)).expand(self.R).contiguous()
        cv_raw0   = _inv_sigmoid_clamped(torch.tensor(cv_frac0,   device=dev, dtype=dtype)).expand(self.R).contiguous()

        if learn_mean:
            self.mean_raw = nn.Parameter(mean_raw0)
        else:
            self.register_buffer("mean_raw", mean_raw0)

        if learn_cv:
            self.cv_raw = nn.Parameter(cv_raw0)
        else:
            self.register_buffer("cv_raw", cv_raw0)

    @property
    def trunc_A(self): return self._trunc_A
    @property
    def trunc_B(self): return self._trunc_B
    @property
    def ml_A(self): return self._ml_A
    @property
    def ml_B(self): return self._ml_B

    def mean_cv(self):
        # mean_A in [mean_lo, mean_hi], cv in [cv_lo, cv_hi]
        mean_frac = torch.sigmoid(self.mean_raw)                 # [R]
        cv_frac   = torch.sigmoid(self.cv_raw)                   # [R]
        mean_A = self._mean_lo + (self._mean_hi - self._mean_lo) * mean_frac
        cv     = self._cv_lo   + (self._cv_hi   - self._cv_lo)   * cv_frac
        return mean_A, cv

    def _mu_sigma(self):
        mean_A, cv = self.mean_cv()
        std_A = cv * mean_A

        eps = 1e-12
        ratio2 = (std_A / (mean_A + eps))**2
        sigma2 = torch.log1p(ratio2)                 # [R]
        sigma  = torch.sqrt(sigma2 + eps)
        mu     = torch.log(mean_A + eps) - 0.5*sigma2
        return mu, sigma

    @staticmethod
    def _phi_inv(u: torch.Tensor) -> torch.Tensor:
        return SQRT2_ * torch.erfinv(2.0 * u - 1.0)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / SQRT2_))

    def _lognormal_cdf(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        z = (torch.log(x.clamp_min(eps)) - mu) / (sigma + eps)
        return self._phi(z)

    def _sample_r_phys(self, u: torch.Tensor) -> torch.Tensor:
        # u: [R,K] in (0,1)
        u = u.clamp(1e-8, 1.0 - 1e-8)
        mu, sigma = self._mu_sigma()                 # [R], [R]
        mu = mu[:, None]
        sigma = sigma[:, None]

        A = self.trunc_A
        B = self.trunc_B

        Fa = self._lognormal_cdf(A, mu, sigma)       # [R,1]
        Fb = self._lognormal_cdf(B, mu, sigma)       # [R,1]
        span = (Fb - Fa).clamp_min(1e-12)

        u2 = (Fa + u * span).clamp(1e-8, 1.0 - 1e-8)
        z = self._phi_inv(u2)
        r = torch.exp(mu + sigma * z)
        return r.clamp(A, B)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [K] or [R,K]
        if u.ndim == 1:
            u = u[None, :].expand(self.R, -1)
        elif u.ndim != 2 or u.shape[0] != self.R:
            raise ValueError(f"Expected u shape [R,K] (or [K]), got {tuple(u.shape)}")

        r_phys = self._sample_r_phys(u)
        r01 = (r_phys - self.ml_A) / (self.ml_B - self.ml_A)
        return r01.clamp(1e-8, 1.0 - 1e-8)

    @torch.no_grad()
    def physical_radius_params(self) -> dict:
        mean_A, cv = self.mean_cv()
        std_A = cv * mean_A
        R = self.R
        dev = mean_A.device
        dt  = mean_A.dtype
        return {
            "radius_mean_A": mean_A.detach(),
            "radius_cv": cv.detach(),
            "radius_std_A": std_A.detach(),
            "radius_trunc_min": self.trunc_A.to(dev, dt).expand(R).detach(),
            "radius_trunc_max": self.trunc_B.to(dev, dt).expand(R).detach(),
            "radius_ml_min": self.ml_A.to(dev, dt).expand(R).detach(),
            "radius_ml_max": self.ml_B.to(dev, dt).expand(R).detach(),
        }
