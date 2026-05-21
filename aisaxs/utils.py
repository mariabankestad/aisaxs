import torch 
import numpy as np
import os
def h_poly(t):
    t2 = t * t
    t3 = t2 * t
    return [
        2*t3 - 3*t2 + 1,       # h00
        t3 - 2*t2 + t,         # h10
        -2*t3 + 3*t2,          # h01
        t3 - t2                # h11
    ]

def cubic_interp(x, y, xs):
    """
    x: [N]       -- shared interpolation nodes
    y: [B, N]    -- batch of function values
    xs: [M]      -- shared query points
    returns: [B, M] interpolated values
    """
    B, N = y.shape
    M = xs.shape[0]
    
    dx = x[1:] - x[:-1]                      # [N-1]
    m = (y[:, 1:] - y[:, :-1]) / dx         # [B, N-1]
    m = torch.cat([m[:, [0]], (m[:, 1:] + m[:, :-1])/2, m[:, [-1]]], dim=1)  # [B, N]

    idx = torch.searchsorted(x[1:], xs)         # [M]
    idx = torch.clamp(idx, 0, N - 2)

    x0 = x[idx]                              # [M]
    x1 = x[idx + 1]                          # [M]
    dxs = x1 - x0                            # [M]
    t = (xs - x0) / dxs                      # [M]

    h = h_poly(t)                            # list of 4 tensors [M]

    # Gather y and m values at idx per batch
    idx_b = idx.unsqueeze(0).expand(B, M)           # [B, M]
    y0 = torch.gather(y, 1, idx_b)                       # [B, M]
    y1 = torch.gather(y, 1, idx_b + 1)                   # [B, M]
    m0 = torch.gather(m, 1, idx_b)
    m1 = torch.gather(m, 1, idx_b + 1)

    dxs = dxs.unsqueeze(0)  # [1, M] for broadcasting

    out = (
        h[0] * y0 +
        h[1] * m0 * dxs +
        h[2] * y1 +
        h[3] * m1 * dxs
    )  # [B, M]
    return out

def keep_mask_robust(
    L,
    *,
    alpha=1.2,
    p_best=0.1,
    Kmin_frac=0.05,
    Kmax_frac=0.30,
):
    """
    Robust keep rule:
      L_star = median of best p_best fraction
      keep if L <= alpha * L_star
      then enforce Kmin/Kmax (fractions of N) by keeping best-loss points.

    Returns:
      mask_keep: (N,) bool
      info: dict with thresholds and counts
    """
    L = np.asarray(L).ravel()
    N = L.size
    order = np.argsort(L)

    # robust reference: median of best p_best
    k = int(np.ceil(p_best * N))
    k = max(k, 1)
    L_star = float(np.mean(L[order[:k]]))

    # initial threshold mask
    mask = (L <= alpha * L_star)

    # enforce Kmin/Kmax as fractions of N
    Kmin = int(np.ceil(Kmin_frac * N))
    Kmax = int(np.ceil(Kmax_frac * N))
    Kmin = max(Kmin, 1)
    Kmax = max(Kmax, Kmin)

    kept = np.where(mask)[0]
    if kept.size < Kmin:
        kept = order[:Kmin]
    elif kept.size > Kmax:
        kept = order[:Kmax]
    else:
        # sort kept by loss (optional, but stable)
        kept = kept[np.argsort(L[kept])]

    mask_keep = np.zeros(N, dtype=bool)
    mask_keep[kept] = True

    info = {
        "N": N,
        "alpha": float(alpha),
        "p_best": float(p_best),
        "k_best": int(k),
        "L_star": L_star,
        "threshold": float(alpha * L_star),
        "Kmin": int(Kmin),
        "Kmax": int(Kmax),
        "kept": int(mask_keep.sum()),
    }
    return mask_keep, info

def load_results_from_dir(save_dir, map_location="cpu"):
    files = sorted(
        f for f in os.listdir(save_dir)
        if f.endswith(".pt")
    )

    results = []
    for f in files:
        path = os.path.join(save_dir, f)
        res = torch.load(path, map_location=map_location)
        results.append(res)
    flat_results = []
    for batch in results:
        flat_results.extend(batch)
    return flat_results



def q2_area(q, I):
    dx = q[1:] - q[:-1]
    y  = (q**2) * I
    return torch.sum(0.5 * (y[...,1:] + y[...,:-1]) * dx, dim=-1)

def normalize_q2(q, I, mask=None, eps=1e-12):
    if mask is not None:
        qm, Im = q[mask], I[..., mask]
    else:
        qm, Im = q, I
    Qw = q2_area(qm, Im).clamp_min(eps)
    return I / Qw[..., None], Qw


def _lognormal_density_q(q, mean, std, eps=1e-12):
    """
    Lognormal density over q, parameterized by mean/std in q-space.
    """
    q = np.asarray(q, dtype=float)
    mean = float(mean)
    std = float(std)

    if mean <= 0:
        raise ValueError(f"mean must be > 0, got {mean}")
    if std <= 0:
        raise ValueError(f"std must be > 0, got {std}")

    s2 = np.log(1.0 + (std**2) / (mean**2))
    mu = np.log(mean) - 0.5 * s2
    sigma = np.sqrt(s2)

    q_safe = np.clip(q, eps, None)
    return np.exp(-0.5 * ((np.log(q_safe) - mu) / sigma) ** 2) / (
        q_safe * sigma * np.sqrt(2.0 * np.pi)
    )


def direct_lengthscale_curve_from_params(
    *,
    w_q,
    m_q,
    s_q,
    q_min=2 * np.pi / 250.0,
    q_max=0.2,
    n_q=500,
    normalize="area",
):
    """
    Build a simple equivalent-length-scale curve directly from model parameters,
    without simulating any density field.

    This uses the shell-weighted spectral mass
        M(q) ~ q^2 * G(q)
    where G(q) is the two-component lognormal mixture.

    Then it replots the same quantity against
        ell = 2*pi / q

    Parameters
    ----------
    w_q, m_q, s_q : array-like, shape [2]
        Mixture weights, q means, and q stds.
    q_min, q_max : float
        q-range in Å^-1.
    n_q : int
        Number of q points.
    normalize : {"area", "max", None}
        Optional normalization of the shell-weighted curve before plotting.

    Returns
    -------
    dict with keys:
        q
        ell
        G_q
        M_q
        M_ell
    """
    w_q = np.asarray(w_q, dtype=float)
    m_q = np.asarray(m_q, dtype=float)
    s_q = np.asarray(s_q, dtype=float)

    if w_q.shape != (2,) or m_q.shape != (2,) or s_q.shape != (2,):
        raise ValueError("w_q, m_q, and s_q must all have shape (2,)")

    q = np.linspace(q_min, q_max, n_q, dtype=float)

    G_q = (
        w_q[0] * _lognormal_density_q(q, m_q[0], s_q[0])
        + w_q[1] * _lognormal_density_q(q, m_q[1], s_q[1])
    )

    # 3D shell-weighted spectral mass
    M_q = q**2 * G_q

    if normalize == "area":
        area = np.trapz(M_q, q)
        if area > 0:
            M_q = M_q / area
    elif normalize == "max":
        m = np.max(M_q)
        if m > 0:
            M_q = M_q / m
    elif normalize is not None:
        raise ValueError("normalize must be one of {'area', 'max', None}")

    ell = 2.0 * np.pi / q

    # simple transformed curve: same y-values, just use ell on x-axis
    order = np.argsort(ell)
    ell = ell[order]
    M_ell = M_q[order]

    return {
        "q": q,
        "ell": ell,
        "G_q": G_q,
        "M_q": M_q,
        "M_ell": M_ell,
    }