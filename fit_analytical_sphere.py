#!/usr/bin/env python3
"""
Fit experimental SAXS data with a polydisperse homogeneous sphere model.
Suitable for homogeneous nanoparticles (e.g. gold particles) and as a baseline
for the heterogeneous-interior GRF model used elsewhere in this framework.

The analytical form factor is:
    P(q, r) = [3(sin(qr) - qr·cos(qr)) / (qr)^3]^2
    I(q, r) = r^6 · P(q, r)

Polydispersity is incorporated via the same differentiable log-normal
quadrature used by the GRF surrogate path, but no neural network is involved:
the form factor is evaluated analytically at each quadrature radius.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from scipy.io import loadmat

from aisaxs.models.sampling.integrator import (
    make_nodes_and_weights,
)
from aisaxs.models.sampling.loss_functions import (
    corr_loss_batched,
    cdf_loss_batched,
)
from aisaxs.models.sampling.parameterized_distributions import LogNormalSamplerBatched
from aisaxs.models.sampling.utils import normalize_q2
from data.lnp.preprocess_ml_data import cubic_interp_monotone


# ---------------------------------------------------------------------------
# Analytical sphere form factor wrapped as a "curve predictor"
# ---------------------------------------------------------------------------

class AnalyticalSpherePredictor(nn.Module):
    """
    Drop-in replacement for the neural surrogate that returns log10 I(q)
    for a homogeneous sphere of radius r.

    Interface:  forward(theta)  ->  log10(I)
        theta : [N, d]   — only column ``r_col`` is used (normalized radius)
        output: [N, n_q]

    The radius column is assumed to be in the ML-normalized range [0, 1],
    corresponding to physical radii in [ml_A, ml_B] Angstrom.
    """

    def __init__(
        self,
        q: torch.Tensor,
        r_col: int = 1,
        ml_A: float = 100.0,
        ml_B: float = 500.0,
    ):
        super().__init__()
        self.r_col = r_col
        self.ml_A = ml_A
        self.ml_B = ml_B
        self.register_buffer("q", q.flatten())

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        r01 = theta[:, self.r_col]                          # [N]
        r = self.ml_A + r01 * (self.ml_B - self.ml_A)       # physical radius (A)

        qr = self.q[None, :] * r[:, None]                   # [N, n_q]

        # Stable sphere form factor amplitude: 3 [sin(x)-x cos(x)] / x^3
        # Taylor expansion near x=0:  f(x) ≈ 1 - x^2/10
        small = qr.abs() < 1e-4
        safe_qr = torch.where(small, torch.ones_like(qr), qr)

        f = 3.0 * (torch.sin(safe_qr) - safe_qr * torch.cos(safe_qr)) / safe_qr**3
        f = torch.where(small, 1.0 - qr**2 / 10.0, f)

        P = f * f                                            # form factor

        # Volume-squared weighting: I ∝ V^2 P ∝ r^6 P
        I = r[:, None] ** 6 * P

        return torch.log10(I.clamp_min(1e-38))


# ---------------------------------------------------------------------------
# Batched integrator that works with the sphere predictor
# ---------------------------------------------------------------------------

class SphereQuadBatched(nn.Module):
    """
    Batched Gauss–Legendre quadrature over the radius distribution.

    Identical maths to QuadFullCurveBatched but simplified: the only
    parameter column that varies across quadrature nodes is the radius.
    """

    def __init__(
        self,
        *,
        sphere_predictor: AnalyticalSpherePredictor,
        sampler: nn.Module,
        K: int = 32,
        rule: str = "legendre",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.sphere = sphere_predictor
        self.sampler = sampler
        self.d = sphere_predictor.r_col + 1        # minimal theta width
        self.r_col = sphere_predictor.r_col

        dev = torch.device(device)
        self.register_buffer(
            "LOGE10",
            torch.log(torch.tensor(10.0, device=dev, dtype=dtype)),
        )
        u, w = make_nodes_and_weights(K, rule.lower(), dev, dtype)
        self.register_buffer("u_nodes", u)
        self.register_buffer("w_nodes", w)
        self.register_buffer("lnw", torch.log(w.clamp_min(1e-38)))
        self.K = K

    def forward(self, *_ignored):
        """
        Returns (ln_mean, y, None) to match the QuadFullCurveBatched API.
        ln_mean: [R, n_q]   —  ln E_r[I(q)]
        y:       [R, K, n_q]
        """
        r = self.sampler(self.u_nodes)              # [R, K] normalized radii
        R, K = r.shape

        # Build a minimal theta tensor with only the radius column filled
        theta = torch.zeros(R * K, self.d, device=r.device, dtype=r.dtype)
        theta[:, self.r_col] = r.reshape(-1)

        y_flat = self.sphere(theta)                 # [R*K, n_q]
        n_q = y_flat.shape[1]
        y = y_flat.reshape(R, K, n_q)

        lnw = self.lnw[None, :, None]               # [1, K, 1]
        ln_mean = torch.logsumexp(self.LOGE10 * y + lnw, dim=1)   # [R, n_q]

        return ln_mean, y, None


# ---------------------------------------------------------------------------
# Data loading (reused from fit_experimental_curve.py)
# ---------------------------------------------------------------------------

def load_experimental_curve(
    mat_path: str,
    q_target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    data = loadmat(mat_path)
    q_exp = torch.from_numpy(data["q_data"].flatten()).to(q_target)
    I_exp = torch.from_numpy(data["I_data"].flatten()).to(q_target)
    I_interp = cubic_interp_monotone(q_exp, I_exp.unsqueeze(0), q_target).flatten()
    I_norm, _ = normalize_q2(q_target, I_interp, mask=mask)
    return I_norm


def compute_logQw_masked(q_masked: torch.Tensor, logI_masked: torch.Tensor) -> torch.Tensor:
    """
    Compute log(∫_{q_masked} q^2 I dq) via trapezoidal rule in log-space.
    logI_masked: [..., n_masked]
    Returns: [...] scalar per batch entry.
    """
    q64 = q_masked.to(torch.float64)
    logq = torch.log(q64)
    logw = torch.log(0.5 * (q64[1:] - q64[:-1]))
    logI64 = logI_masked.to(torch.float64)
    z0 = 2 * logq[:-1] + logI64[..., :-1]
    z1 = 2 * logq[1:]  + logI64[...,  1:]
    seg = logw + torch.logsumexp(torch.stack([z0, z1], dim=-1), dim=-1)
    return torch.logsumexp(seg, dim=-1)


def normalize_logI_masked(q_full: torch.Tensor, logI: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Normalize logI using ∫_{mask} q^2 I dq as the normalization constant.
    Applies the shift to the full q curve.
    logI: [..., n_q]
    Returns: [..., n_q]
    """
    logQw = compute_logQw_masked(q_full[mask], logI[..., mask]).to(logI.dtype)
    return logI - logQw.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Main fitting routine
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit experimental SAXS with a polydisperse homogeneous sphere."
    )
    p.add_argument("--experimental-mat", default="data/lnp/experimental_curve.mat")
    p.add_argument("--output-data",      default="data/lnp/setup_shell_shift/output_data.pt")
    p.add_argument("--save-dir",         default="results/curve_fitting/analytical_sphere")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--dtype",            default="float32", choices=["float32", "float64"])
    p.add_argument("--batch-size",       type=int, default=256)
    p.add_argument("--num-restarts",     type=int, default=4096)
    p.add_argument("--total-steps",      type=int, default=1000)
    p.add_argument("--warmup-steps",     type=int, default=200)
    p.add_argument("--lr",               type=float, default=1e-2)
    p.add_argument("--grad-clip",        type=float, default=5.0)
    p.add_argument("--K-quad",           type=int, default=32,
                   help="Number of quadrature nodes for radius integration.")
    p.add_argument("--q-max",            type=float, default=0.02,
                   help="Fit and evaluate only q <= q_max (Angstrom^-1). "
                        "Full curves are still saved. Default: 0.02 A^-1.")
    p.add_argument("--log-every",        type=int, default=200)
    return p.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main():
    args = parse_args()
    dev = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ---- load q grid and experimental data ----
    output_data = torch.load(args.output_data, map_location="cpu")
    q = output_data["q"].to(dev, dtype=dtype)
    n_q = q.numel()

    # ---- q mask: fit and evaluate only low-q region ----
    q_mask = q <= args.q_max
    print(f"[INFO] Fitting q <= {args.q_max:.4f} A^-1  "
          f"({q_mask.sum().item()}/{n_q} points); full curves will be saved")

    # Normalize experiment over the fitted q range only
    y_exp = load_experimental_curve(args.experimental_mat, q, mask=q_mask).to(dev, dtype=dtype)
    y_log = y_exp.log()

    # ---- sphere predictor ----
    ml_A, ml_B = 100.0, 500.0
    r_col = 0  # sphere model uses a single column
    sphere = AnalyticalSpherePredictor(q, r_col=r_col, ml_A=ml_A, ml_B=ml_B)
    sphere = sphere.to(dev, dtype=dtype)

    # ---- batched fitting ----
    R = args.num_restarts
    B = args.batch_size
    K = args.K_quad
    num_batches = (R + B - 1) // B

    mse_fn = nn.MSELoss(reduction="none")

    all_results = []

    from torch.quasirandom import SobolEngine
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    for b in range(num_batches):
        start = b * B
        end = min(R, (b + 1) * B)
        Rb = end - start

        sampler = LogNormalSamplerBatched(
            R=Rb,
            trunc_A=105.0,
            trunc_B=495.0,
            ml_A=ml_A,
            ml_B=ml_B,
            mean_lo=150.0,
            mean_hi=350.0,
            cv_lo=0.05,
            cv_hi=0.50,
            init_mean_A=None,  # will be overridden by Sobol init below
            init_cv=0.20,
            learn_mean=True,
            learn_cv=True,
            device=args.device,
            dtype=dtype,
        )

        integrator = SphereQuadBatched(
            sphere_predictor=sphere,
            sampler=sampler,
            K=K,
            rule="legendre",
            device=args.device,
            dtype=dtype,
        )

        # Sobol initialization for mean and cv
        sob = SobolEngine(dimension=2, scramble=True)
        U = sob.draw(Rb).to(dev, dtype=dtype)

        # Override sampler raw params with Sobol inits
        with torch.no_grad():
            # mean in [150, 350], cv in [0.05, 0.50]
            margin = 0.05
            mean_frac = margin + (1.0 - 2.0 * margin) * U[:, 0]
            cv_frac = margin + (1.0 - 2.0 * margin) * U[:, 1]
            eps = 1e-6
            sampler.mean_raw.data.copy_(
                torch.log(mean_frac.clamp(eps, 1.0 - eps) / (1.0 - mean_frac.clamp(eps, 1.0 - eps)))
            )
            sampler.cv_raw.data.copy_(
                torch.log(cv_frac.clamp(eps, 1.0 - eps) / (1.0 - cv_frac.clamp(eps, 1.0 - eps)))
            )

        opt = AdamW(sampler.parameters(), lr=args.lr)
        sched = CosineAnnealingLR(opt, T_max=args.total_steps, eta_min=args.lr * 0.1)

        # ---- training loop ----
        for step in range(args.total_steps):
            opt.zero_grad(set_to_none=True)

            ln_mean, _, _ = integrator()
            pred_l = normalize_logI_masked(q, ln_mean, q_mask)   # [Rb, n_q]
            pred_r = pred_l.exp()

            # Loss on masked q-points only
            Lmse_l = mse_fn(pred_l[:, q_mask], y_log[q_mask].unsqueeze(0).expand(Rb, -1)).mean(dim=1)
            Lmse_r = mse_fn(
                pred_r[:, q_mask] / 1e5,
                y_exp[q_mask].unsqueeze(0).expand(Rb, -1) / 1e5,
            ).mean(dim=1)
            Lcorr = corr_loss_batched(pred_l[:, q_mask], y_log[q_mask])
            Lcdf = cdf_loss_batched(pred_l[:, q_mask], y_log[q_mask])

            loss_r = 10.0 * Lcorr + 100.0 * Lcdf + 0.1 * Lmse_l + Lmse_r
            loss = loss_r.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=args.grad_clip)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                print(f"  batch {b+1}/{num_batches}  step {step:4d}/{args.total_steps}"
                      f"  loss={loss.item():.6f}")

        # ---- final eval (full q-range, for plotting) ----
        with torch.no_grad():
            ln_mean, _, _ = integrator()
            pred_l = normalize_logI_masked(q, ln_mean, q_mask)
            pred_r = pred_l.exp()

            # Recompute loss on the fitted q range for reporting
            Lmse = mse_fn(pred_l[:, q_mask], y_log[q_mask].unsqueeze(0).expand(Rb, -1)).mean(dim=1)
            rmse_log = Lmse.sqrt()

            phys = sampler.physical_radius_params()

        for i in range(Rb):
            all_results.append({
                "restart": start + i,
                "rmse_log": float(rmse_log[i].cpu()),
                "loss": float(loss_r[i].cpu()),
                "radius_mean_A": float(phys["radius_mean_A"][i].cpu()),
                "radius_std_A": float(phys["radius_std_A"][i].cpu()),
                "radius_cv": float(phys["radius_cv"][i].cpu()),
                "curve_log_norm": pred_l[i].detach().cpu(),
                "curve_norm": pred_r[i].detach().cpu(),
            })

        # Save batch
        batch_path = os.path.join(args.save_dir, f"batch_{b:04d}.pt")
        torch.save(all_results[-Rb:], batch_path)

        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- save all results + summary ----
    torch.save(all_results, os.path.join(args.save_dir, "all_results.pt"))

    # Summary: best fit
    losses = [r["rmse_log"] for r in all_results]
    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    best = all_results[best_idx]
    print(f"\n{'='*60}")
    print(f"Best fit (low-q RMSE_log = {best['rmse_log']:.4f}):")
    print(f"  mean radius = {best['radius_mean_A']:.1f} A")
    print(f"  std radius  = {best['radius_std_A']:.1f} A")
    print(f"  cv          = {best['radius_cv']:.3f}")
    print(f"{'='*60}")

    # Save experimental data alongside for easy plotting
    torch.save({
        "q": q.cpu(),
        "y_exp": y_exp.cpu(),
        "y_log": y_log.cpu(),
        "q_mask": q_mask.cpu(),
        "best_result": best,
        "all_results_summary": [{
            "restart": r["restart"],
            "rmse_log": r["rmse_log"],
            "radius_mean_A": r["radius_mean_A"],
            "radius_std_A": r["radius_std_A"],
            "radius_cv": r["radius_cv"],
        } for r in all_results],
    }, os.path.join(args.save_dir, "summary.pt"))


if __name__ == "__main__":
    main()
