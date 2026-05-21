from torch.quasirandom import SobolEngine
import math
import torch
import torch.nn as nn
from .parameterized_distributions import BaseRadiusSampler
from typing import Dict, Any


def make_X_from_specs_physical(
    specs: Dict[str, Dict[str, Any]],
    r_col: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Bygg en basrad X[0] i FYSISK skala från specs["..."]["init"].

    - d = max(col) + 1
    - base[j] = specs[name]["init"] i fysisk skala
    - radiuskolumnen r_col får ett rimligt initvärde om den inte satts.
    """
    d = max(int(s["col"]) for s in specs.values()) + 1
    base = torch.zeros(d, device=device, dtype=dtype)

    # Fyll allt från init-värden
    for name, s in specs.items():
        j = int(s["col"])
        init_phys = float(s.get("init", s["lo"]))
        base[j] = init_phys

    # Om radiuskolumnen inte kom från specs: sätt något vettigt
    if base[r_col].abs() < 1e-12:
        # t.ex. mitten av [0,1] eller mitten av [A,B] om du vill vara fysisk
        base[r_col] = 0.0

    X = base.unsqueeze(0)  # shape (1, d)
    return X

class QMCFullCurve(nn.Module):
    """
    QMC over radius via sampler(u)->r in (0,1).
    Replaces column r_col in base_row; calls model; returns ln E[I].
    """
    def __init__(self, specs, r_col: int, sampler: BaseRadiusSampler,
                 n_s=512, seed=12345, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device, self.dtype = torch.device(device), dtype
        X = make_X_from_specs_physical(specs=specs, r_col = r_col, device = device, dtype = dtype)
        _, d = X.shape
        self.d, self.r_col =  d, int(r_col)
        assert 0 <= self.r_col < self.d
        # base row: every column fixed across q except r_col
        mask = torch.ones(d, dtype=torch.bool, device=self.device); mask[self.r_col] = False
        self.register_buffer("base_row", X[0].clone())
        e_r = torch.zeros(d, device=self.device, dtype=self.dtype); e_r[self.r_col] = 1.0
        self.register_buffer("e_r", e_r)
        self.register_buffer("LOGE10", torch.log(torch.tensor(10.0, device=self.device, dtype=self.dtype)))
        self.sampler = sampler.to(self.device)
        self.n_s   = int(n_s)
        self.sobol = SobolEngine(dimension=1, scramble=True, seed=seed)
    @torch.no_grad()
    def _u_sobol_antithetic(self):
        u = self.sobol.draw(self.n_s).squeeze(1).to(self.device, dtype=self.dtype)
        return torch.cat([u, 1.0 - u], dim=0).clamp(1e-12, 1-1e-12)
    def forward(self, curve_predictor, boxed=None):
        u = self._u_sobol_antithetic()
        r = self.sampler(u)
        n = r.shape[0]

        base = self.base_row if boxed is None else boxed.apply_to_base(self.base_row, normalize=True)
        theta = base.unsqueeze(0).expand(n, self.d) + r.unsqueeze(1) * self.e_r.unsqueeze(0)
        y = curve_predictor(theta)  
        ln_mean = torch.logsumexp(self.LOGE10 * y - math.log(n), dim=0) 
        return ln_mean, y


def _leggauss_torch(K: int, device, dtype):
    """
    Gauss–Legendre nodes/weights on [-1,1] computed via Golub–Welsch,
    implemented entirely in torch.
    Returns:
        x: (K,) nodes in [-1,1]
        w: (K,) weights (sum w = 2)
    """
    if K <= 0:
        raise ValueError("K must be positive for Gauss–Legendre.")

    k = torch.arange(1, K, device=device, dtype=dtype)
    beta = k / torch.sqrt(4.0 * k * k - 1.0)          # (K-1,)

    # Jacobi matrix for Legendre polynomials
    T = torch.zeros(K, K, device=device, dtype=dtype)
    T.diagonal(1).copy_(beta)
    T.diagonal(-1).copy_(beta)

    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(T)
    x = eigvals                                   # nodes in [-1,1]
    w = 2.0 * eigvecs[0, :]**2                    # weights on [-1,1]

    return x, w

def make_nodes_and_weights(K: int, rule: str, device, dtype):
    """
    Return u ∈ [0,1], w ≥0, sum w = 1 for a given quadrature rule,
    implemented entirely in torch.
    """
    rule = rule.lower()

    if rule == "legendre":
        # Gauss–Legendre on [-1,1] → [0,1]
        x, w = _leggauss_torch(K, device=device, dtype=dtype)  # x∈[-1,1], w sum=2
        u = (x + 1.0) * 0.5                                    # map to [0,1]
        w = w * 0.5                                            # rescale weights so sum(w)=1

    elif rule == "clenshaw-curtis":
        # Classic CC nodes/weights on [0,1]
        if K < 2:
            raise ValueError("Clenshaw–Curtis requires K >= 2.")
        k = torch.arange(K, device=device, dtype=dtype)        # 0..K-1
        u = 0.5 * (1.0 - torch.cos(math.pi * k / (K - 1)))     # [0,1]
        w = torch.ones(K, device=device, dtype=dtype)
        w[0] = 0.5
        w[-1] = 0.5
        w = w / w.sum()                                        # normalize to sum=1

    elif rule == "midpoint":
        # Midpoint rule on [0,1]
        k = torch.arange(K, device=device, dtype=dtype)        # 0..K-1
        u = (k + 0.5) / K                                      # midpoints in (0,1)
        w = torch.full((K,), 1.0 / K, device=device, dtype=dtype)

    elif rule == "simpson":
        # Simpson’s rule on [0,1]: K must be odd (even number of intervals)
        if K % 2 == 0:
            raise ValueError("Simpson requires K to be odd (even number of intervals).")
        u = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
        w = torch.ones(K, device=device, dtype=dtype)
        # pattern: 1, 4, 2, 4, 2, ..., 4, 1
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w = w / w.sum()  
        eps = 1e-3                                      # normalize to sum=1
        u = eps + (1.0 - 2.0 * eps) * u
    else:
        raise ValueError(f"Unknown quadrature rule: {rule}")

    return u, w

def smooth_linear_I(I: torch.Tensor, sigma_pts: float = 1.5) -> torch.Tensor:
    """
    Simple Gaussian smoothing over q-index (assuming uniform q-grid).
    I: (n_q,)
    """
    n_q = I.shape[0]
    radius = int(3 * sigma_pts)
    if radius == 0:
        return I

    xs = torch.arange(-radius, radius + 1, device=I.device, dtype=I.dtype)
    kernel = torch.exp(-0.5 * (xs / sigma_pts) ** 2)
    kernel = kernel / kernel.sum()

    I_ = I.view(1, 1, n_q)                     # [1,1,L]
    k = kernel.view(1, 1, -1)                  # [1,1,K]
    I_pad = torch.nn.functional.pad(I_, (radius, radius), mode="replicate")
    I_smooth = torch.nn.functional.conv1d(I_pad, k)[0, 0, :]  # (n_q,)
    return I_smooth


class QuadFullCurve(nn.Module):
    """
    Deterministic quadrature over u∈[0,1] with fixed nodes/weights (independent of φ).
    Uses sampler(u)->r to map to radius; injects r into r_col; calls model; returns:
      ln_mean: ln E_r[I(q)]   and   y: per-node log10 I(q).
    """
    def __init__(self, specs, r_col: int, sampler: BaseRadiusSampler,
                 K=32, rule="simpson", device="cuda", dtype=torch.float32):
        super().__init__()
        self.device, self.dtype = torch.device(device), dtype
        X = make_X_from_specs_physical(specs=specs, r_col = r_col, device = device, dtype = dtype)
        _ , d = X.shape
        self.d, self.r_col = d, int(r_col)
        assert 0 <= self.r_col < self.d

        # assert all non-radius columns are constant across q
        mask = torch.ones(d, dtype=torch.bool, device=self.device); mask[self.r_col] = False
        self.register_buffer("base_row", X[0].clone())
        e_r = torch.zeros(d, device=self.device, dtype=self.dtype); e_r[self.r_col] = 1.0
        self.register_buffer("e_r", e_r)
        self.register_buffer("LOGE10", torch.log(torch.tensor(10.0, device=self.device, dtype=self.dtype)))

        self.sampler = sampler.to(self.device)
        self.K = int(K)

        # ---- fixed nodes/weights on [0,1] (independent of sampler params) ----
        rule = rule.lower()
        rule = rule.lower()
        K = self.K
        device, dtype = self.device, self.dtype

        u, w = make_nodes_and_weights(K, rule, device, dtype)

        self.register_buffer("u_nodes", u)   # (K,)
        self.register_buffer("w_nodes", w)   # (K,)

    def forward(self, curve_predictor, boxed=None):
        # map fixed u->r via chosen sampler (differentiable in sampler params)
        r = self.sampler(self.u_nodes)                                       # (K,)
        base = self.base_row if boxed is None else boxed.apply_to_base(self.base_row, normalize=True)
        theta = base.unsqueeze(0).expand(self.K, self.d) + (r).unsqueeze(1) * self.e_r.unsqueeze(0)
        y = curve_predictor(theta )

        # ln E[I] with weights: ln sum_i w_i * 10^{y_i}
        lnw = torch.log(self.w_nodes + 1e-38).unsqueeze(1)                   # (K,1)
        ln_mean = torch.logsumexp(self.LOGE10 * y + lnw, dim=0)             
        return ln_mean, y, theta

def make_integrator(method: str, specs, r_col: int, sampler: BaseRadiusSampler,
                    device="cuda", dtype=torch.float32, **kwargs):
    """
    method: "qmc" or "quad"
    kwargs for qmc:  n_s, seed
    kwargs for quad: K, rule ("legendre"|"clenshaw-curtis"|"midpoint"|"simpson)
    """
    m = method.lower()
    if m == "qmc":
        return QMCFullCurve(specs, r_col=r_col, sampler=sampler,
                            n_s=kwargs.get("n_s", 128),
                            seed=kwargs.get("seed", 12345),
                            device=device, dtype=dtype)
    elif m == "quad":
        return QuadFullCurve(specs, r_col=r_col, sampler=sampler,
                             K=kwargs.get("K", 256),
                             rule=kwargs.get("rule", "legendre"),
                             device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown integration method: {method}")
    


class QuadFullCurveBatched(nn.Module):
    """
    Batched deterministic quadrature over u∈[0,1] with fixed nodes/weights.

    Inputs:
      - base_row: [R, d] physical row (or already normalized, depending on your pipeline)
      - sampler(u_nodes): returns r: [R, K] (normalized radius column, as in your original)
      - curve_predictor(theta_flat): theta_flat [R*K, d] -> y_flat [R*K, n_q]

    Output:
      - ln_mean: [R, n_q]  where ln_mean[r] = ln Σ_k w_k * 10^{y[r,k]}
      - y:       [R, K, n_q]   per-node log10 I(q)
      - theta:   [R, K, d]
    """
    def __init__(
        self,
        *,
        d: int,
        r_col: int,
        sampler: nn.Module,
        K: int = 32,
        rule: str = "simpson",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        self.d = int(d)
        self.r_col = int(r_col)
        assert 0 <= self.r_col < self.d

        self.sampler = sampler.to(self.device)

        # unit vector selecting r_col
        e_r = torch.zeros(self.d, device=self.device, dtype=self.dtype)
        e_r[self.r_col] = 1.0
        self.register_buffer("e_r", e_r)

        self.register_buffer("LOGE10", torch.log(torch.tensor(10.0, device=self.device, dtype=self.dtype)))

        self.K = int(K)
        u, w = make_nodes_and_weights(self.K, rule.lower(), self.device, self.dtype)
        self.register_buffer("u_nodes", u)   # [K]
        self.register_buffer("w_nodes", w)   # [K], sum=1

        # precompute ln(w) once
        self.register_buffer("lnw", torch.log(self.w_nodes.clamp_min(1e-38)))  # [K]

    def forward(self, curve_predictor: nn.Module, base_row: torch.Tensor):
        """
        base_row: [R, d]
        """
        base_row = base_row.to(device=self.device, dtype=self.dtype)
        if base_row.ndim != 2 or base_row.shape[1] != self.d:
            raise ValueError(f"base_row must be [R,{self.d}], got {tuple(base_row.shape)}")

        R = base_row.shape[0]
        K = self.K

        # r: [R, K]
        r = self.sampler(self.u_nodes)  # expects batched sampler
        if r.ndim != 2 or r.shape != (R, K):
            raise ValueError(f"sampler(u_nodes) must return [R,K]=[{R},{K}], got {tuple(r.shape)}")
        r = r.to(dtype=self.dtype)

        # theta: [R, K, d] = base_row[:,None,:] + r[:,:,None] * e_r[None,None,:]
        theta = base_row[:, None, :].expand(R, K, self.d) + r[:, :, None] * self.e_r[None, None, :]

        # run model in one go: [R*K, d] -> [R*K, n_q]
        theta_flat = theta.reshape(R * K, self.d)
        y_flat = curve_predictor(theta_flat)  # log10 I(q)
        if y_flat.ndim != 2:
            raise ValueError(f"curve_predictor must return [N,n_q], got {tuple(y_flat.shape)}")

        n_q = y_flat.shape[1]
        y = y_flat.reshape(R, K, n_q)

        # ln mean: logsumexp over K with weights
        # ln Σ_k w_k * 10^{y} = logsumexp( ln w_k + ln(10)*y )
        lnw = self.lnw[None, :, None]                 # [1,K,1]
        ln_mean = torch.logsumexp(self.LOGE10 * y + lnw, dim=1)  # [R,n_q]

        return ln_mean, y, theta
