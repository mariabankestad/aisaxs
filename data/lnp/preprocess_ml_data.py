#!/usr/bin/env python3
"""
make_training_data.py

- Reads YAML with Parameter_ranges, parameter_path, simulated_data_path, name
- Builds input tensor in the order of Parameter_ranges keys
- Interpolates log10 intensities to a uniform q-grid (Hermite or monotone cubic)
- Saves into a folder called <name>/ :
    input_data.pt
    output_data.pt
    train.pt, val.pt, test.pt
    splits.pt
"""

import sys
import yaml
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

DTYPE  = torch.float64
DEVICE = "cpu"  # set "cuda" if you want tensors on GPU

def assert_columns(df: pd.DataFrame, cols, where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}")

import torch

def h_poly(t: torch.Tensor):
    """Cubic Hermite basis functions (h00, h10, h01, h11)."""
    t2 = t * t
    t3 = t2 * t
    return 2*t3 - 3*t2 + 1, t3 - 2*t2 + t, -2*t3 + 3*t2, t3 - t2


def _prepare_interp(x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor):
    """
    Common prep: checks, secant slopes, interval indices, and normalized t.
    x:  [N] (strictly increasing), y: [B,N], xs: [M]
    Returns: (B,N, dx, sec, idx, t, dxs)
    """
    assert x.ndim == 1 and y.ndim == 2 and xs.ndim == 1, "Shapes must be x:[N], y:[B,N], xs:[M]"
    B, N = y.shape
    msg = "x must be strictly increasing (required by searchsorted and Hermite)."
    dx = x[1:] - x[:-1]
    if not torch.all(dx > 0):
        raise ValueError(msg)

    # Per-interval secant slopes (broadcast B×(N-1))
    sec = (y[:, 1:] - y[:, :-1]) / dx.unsqueeze(0)

    # Interval indices for xs; use right=True then -1 (safer at boundaries)
    idx = torch.searchsorted(x, xs, right=True) - 1
    idx = idx.clamp(0, N - 2)  # in [0, N-2]

    x0, x1 = x[idx], x[idx + 1]
    dxs = x1 - x0
    t = (xs - x0) / (dxs + 1e-12)  # normalized local coord
    return B, N, dx, sec, idx, t, dxs


def cubic_interp_hermite(x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """
    Natural cubic Hermite with finite-difference/averaged tangents.
    x:[N], y:[B,N], xs:[M] -> ys:[B,M]
    """
    B, N, dx, sec, idx, t, dxs = _prepare_interp(x, y, xs)

    # Endpoint slopes = first/last secant; interior = average of neighboring secants
    m = torch.empty(B, N, dtype=y.dtype, device=y.device)
    m[:, 0]  = sec[:, 0]
    m[:, -1] = sec[:, -1]
    if N > 2:
        m[:, 1:-1] = 0.5 * (sec[:, :-1] + sec[:, 1:])

    # Hermite basis
    h00, h10, h01, h11 = h_poly(t)                  # [M] each

    # Gather per-query interval values
    idx_b = idx.unsqueeze(0).expand(B, -1)          # [B,M]
    y0  = torch.gather(y, 1, idx_b)                 # [B,M]
    y1  = torch.gather(y, 1, idx_b + 1)
    m0  = torch.gather(m, 1, idx_b)
    m1  = torch.gather(m, 1, idx_b + 1)

    dxs = dxs.unsqueeze(0)                           # [1,M]
    return h00*y0 + h10*m0*dxs + h01*y1 + h11*m1*dxs


import torch

def cubic_interp_monotone(x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """
    Monotone cubic Hermite with safe endpoint linear extrapolation.
    x: [N] increasing, y: [B,N], xs: [M] -> ys: [B,M]
    """
    x = x.flatten()
    B, N = y.shape
    xs = xs.flatten()
    assert torch.all(x[1:] > x[:-1]), "x must be strictly increasing"

    # --- Secants
    dx  = x[1:] - x[:-1]                  # [N-1]
    sec = (y[:,1:] - y[:,:-1]) / dx       # [B,N-1]

    # --- Initial slopes m: endpoint=secant, interior=avg(secants)
    m = torch.zeros_like(y)
    m[:,0]  = sec[:,0]
    m[:,-1] = sec[:,-1]
    if N > 2:
        m[:,1:-1] = 0.5*(sec[:,:-1] + sec[:,1:])

    # --- Zero slopes where flat or sign change around node i
    # flat intervals -> zero adjacent slopes
    flat = sec.abs() < 1e-30                      # [B,N-1]
    m[:, :-1] = torch.where(flat, torch.zeros_like(m[:, :-1]), m[:, :-1])
    m[:, 1: ] = torch.where(flat, torch.zeros_like(m[:, 1: ]), m[:, 1: ])

    # sign change of neighboring secants around interior nodes -> m_i=0
    if N > 2:
        sign_change = (sec[:,:-1] * sec[:,1:]) <= 0   # [B,N-2]
        m[:,1:-1] = torch.where(sign_change, torch.zeros_like(m[:,1:-1]), m[:,1:-1])

    # --- Hyman/Fritsch–Carlson limiter per interval
    eps = 1e-30
    alpha = m[:,:-1] / (sec + eps)               # [B,N-1]
    beta  = m[:,1: ] / (sec + eps)               # [B,N-1]

    # No overshoot: if alpha<0 or beta<0, zero corresponding slope
    m_left  = torch.where(alpha < 0, torch.zeros_like(m[:,:-1]), m[:,:-1])
    m_right = torch.where(beta  < 0, torch.zeros_like(m[:,1: ]), m[:,1: ])

    # Hyman bound: scale so alpha^2 + beta^2 <= 9
    alpha = m_left/(sec+eps)
    beta  = m_right/(sec+eps)
    s2 = alpha.square() + beta.square()
    mask = s2 > 9.0
    tau = 3.0 / torch.sqrt(s2 + eps)
    m_left  = torch.where(mask, alpha*tau*sec, m_left)
    m_right = torch.where(mask,  beta*tau*sec, m_right)

    # Stitch back m (average interior contributions)
    if N > 2:
        m_new = torch.empty_like(m)
        m_new[:,0]  = m_left[:,0]
        m_new[:,-1] = m_right[:,-1]
        m_new[:,1:-1] = 0.5*(m_left[:,1:] + m_right[:,:-1])
        m = m_new
    else:
        m[:,0]  = m_left[:,0]
        m[:,-1] = m_right[:,-1]

    # --- Locate intervals for xs
    # idx such that x[idx] <= xs < x[idx+1], clamp to valid range
    idx = torch.searchsorted(x, xs, right=True) - 1     # [M]
    idx = idx.clamp(0, N-2)
    x0 = x[idx]                                         # [M]
    x1 = x[idx+1]
    dxs = (xs - x0)
    h = (xs - x0) / (x1 - x0)
    h = h.clamp(0.0, 1.0)                               # keep inside for interpolation

    # Hermite basis
    h00 = (1 + 2*h) * (1 - h)**2
    h10 = h * (1 - h)**2
    h01 = h**2 * (3 - 2*h)
    h11 = h**2 * (h - 1)

    # Gather per interval
    idx_b = idx.unsqueeze(0).expand(B, -1)              # [B,M]
    y0 = torch.gather(y, 1, idx_b)
    y1 = torch.gather(y, 1, idx_b + 1)
    m0 = torch.gather(m, 1, idx_b)
    m1 = torch.gather(m, 1, idx_b + 1)
    dx_interval = (x1 - x0).unsqueeze(0)                # [1,M]

    ys_interp = h00*y0 + h10*m0*dx_interval + h01*y1 + h11*m1*dx_interval  # [B,M]

    # --- Safe endpoint linear extrapolation
    left_mask  = xs <= x[0]
    right_mask = xs >= x[-1]
    ys = ys_interp.clone()

    if left_mask.any():
        # use node 0 with slope m[:,0]
        dl = (xs[left_mask] - x[0]).unsqueeze(0)        # [1,ML]
        ys[:, left_mask] = y[:, [0]] + m[:, [0]] * dl

    if right_mask.any():
        dr = (xs[right_mask] - x[-1]).unsqueeze(0)      # [1,MR]
        ys[:, right_mask] = y[:, [-1]] + m[:, [-1]] * dr

    return ys


def main(cfg_path: str, method: str = "hermite"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    name = cfg["name"]
    outdir = Path(name)
    outdir.mkdir(exist_ok=True)

    # ---- inputs ----
    param_ranges = cfg["Parameter_ranges"]
    keys = list(param_ranges.keys())
    df_par = pd.read_csv(cfg["parameter_path"])
    assert_columns(df_par, keys, cfg["parameter_path"])
    mins = torch.tensor([param_ranges[k]["min"] for k in keys], dtype=DTYPE, device=DEVICE)
    maxs = torch.tensor([param_ranges[k]["max"] for k in keys], dtype=DTYPE, device=DEVICE)
    X_raw = torch.tensor(df_par[keys].to_numpy(), dtype=DTYPE, device=DEVICE)

    torch.save({"names": keys, "mins": mins, "maxs": maxs, "data": X_raw},
               outdir / "input_data.pt")

    # ---- outputs ----
    df_sim = pd.read_csv(cfg["simulated_data_path"])
    q_col = cfg.get("q_column", "q_sim")
    n_q   = int(cfg.get("n_q", 500))
    assert_columns(df_sim, [q_col], cfg["simulated_data_path"])
    q = torch.tensor(df_sim[q_col].to_numpy(), dtype=DTYPE, device=DEVICE)
    Y = torch.tensor(df_sim.drop(columns=[q_col]).to_numpy().T, dtype=DTYPE, device=DEVICE)
    q_new = torch.linspace(0.002, q.max()-5e-4, n_q, dtype=DTYPE, device=DEVICE)
    if method == "hermite":
        Y_new = cubic_interp_hermite(q, Y, q_new)
    else:
        Y_new = cubic_interp_monotone(q, Y, q_new)
    Y_new = torch.log10(Y_new)

    torch.save({"q": q_new, "Ilog10": Y_new}, outdir / "output_data.pt")

    # ---- splits ----
    X = (X_raw - mins.unsqueeze(0)) / (maxs.unsqueeze(0) - mins.unsqueeze(0) + 1e-12)

    N = X.size(0)
    indices = torch.arange(N)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.02, random_state=0)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.02, random_state=0)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train, Y_val, Y_test = Y_new[train_idx], Y_new[val_idx], Y_new[test_idx]
    y_mean, y_std = Y_train.mean(), Y_train.std() + 1e-12

    Y_train_s, Y_val_s, Y_test_s = (Y_train-y_mean)/y_std, (Y_val-y_mean)/y_std, (Y_test-y_mean)/y_std
    torch.save({"x": X_train, "y": Y_train_s, "idx": torch.as_tensor(train_idx)}, outdir / "train.pt")
    torch.save({"x": X_val,   "y": Y_val_s,   "idx": torch.as_tensor(val_idx)},   outdir / "val.pt")
    torch.save({"x": X_test,  "y": Y_test_s,  "idx": torch.as_tensor(test_idx)},  outdir / "test.pt")
    torch.save({
        "train_idx": torch.as_tensor(train_idx),
        "val_idx":   torch.as_tensor(val_idx),
        "test_idx":  torch.as_tensor(test_idx),
        "x_min": mins, "x_max": maxs,
        "y_mean": y_mean, "y_std": y_std,
        "q": q_new, "interp_method": method,
    }, outdir / "splits.pt")

    print(f"[OK] Saved files into folder: {outdir}/")
    for f in ["input_data.pt","output_data.pt","train.pt","val.pt","test.pt","splits.pt"]:
        print("  -", f)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python make_training_data.py <config.yaml> [hermite|monotone]")
        sys.exit(1)
    cfg_path = sys.argv[1]
    method   = sys.argv[2] if len(sys.argv) == 3 else "monotone"
    main(cfg_path, method)
