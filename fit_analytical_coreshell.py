#!/usr/bin/env python3
"""
Fit experimental SAXS data with a polydisperse core-shell sphere model.
Suitable for nanoparticles with a uniform shell of distinct electron density
(e.g. coated gold particles), and as an intermediate baseline between the
homogeneous sphere and the heterogeneous-interior GRF model.

The analytical form factor is:
    f(x) = 3(sin(x) - x cos(x)) / x^3        (normalised sphere amplitude)
    F(q)  = R^3 f(qR) + r_c^3 (c-1) f(q r_c)
    I(q)  = F(q)^2

where R is the outer radius, r_c = R - d is the core radius, c = Δρ_core/Δρ_shell
is the contrast ratio (Δρ_shell ≡ 1 since the Porod-invariant normalisation removes
the overall intensity scale), and d is the shell thickness.

Polydispersity: R is drawn from a truncated log-normal (same quadrature as the GRF
surrogate); d and c are scalars optimised per restart, not varied across quadrature nodes.

Companion to `fit_analytical_sphere.py` (homogeneous baseline) and
`fit_experimental_curve.py` (heterogeneous-interior GRF model).
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from scipy.io import loadmat

from aisaxs.models.sampling.integrator import make_nodes_and_weights
from aisaxs.models.sampling.parameterized_distributions import LogNormalSamplerBatched
from aisaxs.models.sampling.utils import normalize_q2
from data.lnp.preprocess_ml_data import cubic_interp_monotone


# ---------------------------------------------------------------------------
# Core-shell form factor
# ---------------------------------------------------------------------------

class AnalyticalCoreShellPredictor(nn.Module):
    """
    Analytical core-shell form factor.

    Interface: forward(r01, d_A, contrast_ratio) -> log10(I)
        r01:            [N]  normalised outer radius in [0,1]
        d_A:            [N]  shell thickness (Å)
        contrast_ratio: [N]  Δρ_core / Δρ_shell  (Δρ_shell ≡ 1)
        output:         [N, n_q]
    """

    def __init__(
        self,
        q: torch.Tensor,
        ml_A: float = 100.0,
        ml_B: float = 500.0,
    ):
        super().__init__()
        self.ml_A = ml_A
        self.ml_B = ml_B
        self.register_buffer("q", q.flatten())

    @staticmethod
    def _sphere_amp(x: torch.Tensor) -> torch.Tensor:
        """Stable f(x) = 3[sin(x) - x cos(x)] / x^3, f(0) = 1."""
        small = x.abs() < 1e-4
        safe = torch.where(small, torch.ones_like(x), x)
        f = 3.0 * (torch.sin(safe) - safe * torch.cos(safe)) / safe ** 3
        # Taylor: f(x) ≈ 1 - x^2/10 + x^4/280
        f_taylor = 1.0 - x ** 2 / 10.0 + x ** 4 / 280.0
        return torch.where(small, f_taylor, f)

    def forward(
        self,
        r01: torch.Tensor,
        d_A: torch.Tensor,
        contrast_ratio: torch.Tensor,
    ) -> torch.Tensor:
        # Physical outer radius [N]
        R = self.ml_A + r01 * (self.ml_B - self.ml_A)
        # Core radius — clamp to at least 1 Å to keep it positive
        r_c = (R - d_A).clamp_min(1.0)

        # q×radius products  [N, n_q]
        qR  = self.q[None, :] * R[:, None]
        qrc = self.q[None, :] * r_c[:, None]

        f_R  = self._sphere_amp(qR)           # [N, n_q]
        f_rc = self._sphere_amp(qrc)          # [N, n_q]

        # Volume factors (4π/3 cancels under Porod normalisation)
        R3   = R ** 3                         # [N]
        rc3  = r_c ** 3                       # [N]

        # Form factor amplitude  F = R^3 f(qR) + r_c^3 (c-1) f(q r_c)
        c_minus_1 = (contrast_ratio - 1.0)[:, None]   # [N, 1]
        F = R3[:, None] * f_R + rc3[:, None] * c_minus_1 * f_rc

        I = F * F
        return torch.log10(I.clamp_min(1e-38))


# ---------------------------------------------------------------------------
# Per-restart learnable shell parameters (d and contrast ratio)
# ---------------------------------------------------------------------------

class CoreShellParamsBatched(nn.Module):
    """
    Per-restart learnable d (shell thickness) and contrast_ratio (Δρ_c/Δρ_s),
    each constrained to a physical range via tanh reparameterisation.
    """

    def __init__(
        self,
        R: int,
        d_lo: float = 20.0,
        d_hi: float = 100.0,
        contrast_lo: float = 0.5,
        contrast_hi: float = 2.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        dev = torch.device(device)
        self.d_lo = d_lo
        self.d_hi = d_hi
        self.d_mid = 0.5 * (d_lo + d_hi)
        self.d_span = 0.5 * (d_hi - d_lo)
        self.c_lo = contrast_lo
        self.c_hi = contrast_hi
        self.c_mid = 0.5 * (contrast_lo + contrast_hi)
        self.c_span = 0.5 * (contrast_hi - contrast_lo)

        self.raw_d        = nn.Parameter(torch.zeros(R, device=dev, dtype=dtype))
        self.raw_contrast = nn.Parameter(torch.zeros(R, device=dev, dtype=dtype))

    def d(self) -> torch.Tensor:
        return self.d_mid + self.d_span * torch.tanh(self.raw_d)

    def contrast(self) -> torch.Tensor:
        return self.c_mid + self.c_span * torch.tanh(self.raw_contrast)

    def set_from_sobol(self, u_d: torch.Tensor, u_contrast: torch.Tensor) -> None:
        """Initialise from Sobol fractions u ∈ (0,1)."""
        eps = 1e-6
        u_d       = u_d.clamp(eps, 1.0 - eps)
        u_contrast = u_contrast.clamp(eps, 1.0 - eps)
        # invert tanh: raw = atanh((u * span*2 - span) / span)
        #            = atanh(2u - 1)
        self.raw_d.data.copy_(torch.atanh(2.0 * u_d - 1.0))
        self.raw_contrast.data.copy_(torch.atanh(2.0 * u_contrast - 1.0))

    def physical_values(self) -> dict[str, torch.Tensor]:
        return {"d_A": self.d().detach(), "contrast_ratio": self.contrast().detach()}


# ---------------------------------------------------------------------------
# Batched quadrature integrator for core-shell
# ---------------------------------------------------------------------------

class CoreShellQuadBatched(nn.Module):
    """
    Gauss-Legendre quadrature over the outer radius distribution.
    d and contrast_ratio are per-restart scalars, broadcast over K nodes.
    """

    def __init__(
        self,
        *,
        predictor: AnalyticalCoreShellPredictor,
        sampler: nn.Module,
        cs_params: CoreShellParamsBatched,
        K: int = 32,
        rule: str = "legendre",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.predictor = predictor
        self.sampler   = sampler
        self.cs_params = cs_params
        self.K         = K

        dev = torch.device(device)
        self.register_buffer(
            "LOGE10",
            torch.log(torch.tensor(10.0, device=dev, dtype=dtype)),
        )
        u, w = make_nodes_and_weights(K, rule.lower(), dev, dtype)
        self.register_buffer("u_nodes", u)
        self.register_buffer("lnw", torch.log(w.clamp_min(1e-38)))

    def forward(self, *_ignored):
        """
        Returns (ln_mean, y, None) — same API as SphereQuadBatched.
        ln_mean: [R, n_q]
        y:       [R, K, n_q]
        """
        r01 = self.sampler(self.u_nodes)     # [R, K]  normalised radii
        R_batch, K = r01.shape

        # Broadcast d and contrast over K nodes  →  [R*K]
        d   = self.cs_params.d().unsqueeze(1).expand(R_batch, K).reshape(-1)
        c   = self.cs_params.contrast().unsqueeze(1).expand(R_batch, K).reshape(-1)
        r_f = r01.reshape(-1)                # [R*K]

        y_flat = self.predictor(r_f, d, c)   # [R*K, n_q]
        n_q    = y_flat.shape[1]
        y      = y_flat.reshape(R_batch, K, n_q)

        lnw    = self.lnw[None, :, None]     # [1, K, 1]
        ln_mean = torch.logsumexp(self.LOGE10 * y + lnw, dim=1)  # [R, n_q]

        return ln_mean, y, None


# ---------------------------------------------------------------------------
# Data loading (shared with fit_analytical_sphere.py)
# ---------------------------------------------------------------------------

def load_experimental_curve(
    mat_path: str,
    q_target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    data     = loadmat(mat_path)
    q_exp    = torch.from_numpy(data["q_data"].flatten()).to(q_target)
    I_exp    = torch.from_numpy(data["I_data"].flatten()).to(q_target)
    I_interp = cubic_interp_monotone(q_exp, I_exp.unsqueeze(0), q_target).flatten()
    I_norm, _ = normalize_q2(q_target, I_interp, mask=mask)
    return I_norm


def compute_logQw_masked(q_masked: torch.Tensor, logI_masked: torch.Tensor) -> torch.Tensor:
    q64   = q_masked.to(torch.float64)
    logq  = torch.log(q64)
    logw  = torch.log(0.5 * (q64[1:] - q64[:-1]))
    logI64 = logI_masked.to(torch.float64)
    z0    = 2 * logq[:-1] + logI64[..., :-1]
    z1    = 2 * logq[1:]  + logI64[...,  1:]
    seg   = logw + torch.logsumexp(torch.stack([z0, z1], dim=-1), dim=-1)
    return torch.logsumexp(seg, dim=-1)


def normalize_logI_masked(
    q_full: torch.Tensor,
    logI: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    logQw = compute_logQw_masked(q_full[mask], logI[..., mask]).to(logI.dtype)
    return logI - logQw.unsqueeze(-1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit experimental SAXS with a polydisperse core-shell sphere."
    )
    p.add_argument("--experimental-mat", default="data/lnp/experimental_curve.mat")
    p.add_argument("--output-data",      default="data/lnp/setup_shell_shift/output_data.pt")
    p.add_argument("--save-dir",         default="results/curve_fitting/analytical_coreshell")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--dtype",            default="float32", choices=["float32", "float64"])
    p.add_argument("--batch-size",       type=int,   default=256)
    p.add_argument("--num-restarts",     type=int,   default=4096)
    p.add_argument("--total-steps",      type=int,   default=1000)
    p.add_argument("--warmup-steps",     type=int,   default=200)
    p.add_argument("--lr",               type=float, default=1e-2)
    p.add_argument("--grad-clip",        type=float, default=5.0)
    p.add_argument("--K-quad",           type=int,   default=32)
    p.add_argument("--q-max",            type=float, default=0.02,
                   help="Fit only q <= q_max (Å⁻¹). Full curves are still saved.")
    p.add_argument("--log-every",        type=int,   default=200)
    # radius distribution bounds
    p.add_argument("--ml-A",             type=float, default=100.0)
    p.add_argument("--ml-B",             type=float, default=500.0)
    p.add_argument("--mean-lo",          type=float, default=150.0)
    p.add_argument("--mean-hi",          type=float, default=350.0)
    p.add_argument("--cv-lo",            type=float, default=0.05)
    p.add_argument("--cv-hi",            type=float, default=0.50)
    # shell parameter bounds
    p.add_argument("--d-lo",             type=float, default=20.0,
                   help="Lower bound for shell thickness d (Å).")
    p.add_argument("--d-hi",             type=float, default=100.0,
                   help="Upper bound for shell thickness d (Å).")
    p.add_argument("--contrast-lo",      type=float, default=0.5,
                   help="Lower bound for Δρ_core/Δρ_shell.")
    p.add_argument("--contrast-hi",      type=float, default=2.0,
                   help="Upper bound for Δρ_core/Δρ_shell.")
    return p.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    dev   = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ---- q grid and experimental data ----
    output_data = torch.load(args.output_data, map_location="cpu")
    q           = output_data["q"].to(dev, dtype=dtype)
    n_q         = q.numel()

    q_mask = q <= args.q_max
    print(
        f"[INFO] Fitting q <= {args.q_max:.4f} Å⁻¹  "
        f"({q_mask.sum().item()}/{n_q} points); full curves will be saved"
    )

    y_exp = load_experimental_curve(args.experimental_mat, q, mask=q_mask).to(dev, dtype=dtype)
    y_log = y_exp.log()

    # ---- predictor and quadrature setup ----
    predictor = AnalyticalCoreShellPredictor(q, ml_A=args.ml_A, ml_B=args.ml_B)
    predictor = predictor.to(dev, dtype=dtype)

    R  = args.num_restarts
    B  = args.batch_size
    K  = args.K_quad
    mse_fn = nn.MSELoss(reduction="none")

    all_results = []

    from torch.quasirandom import SobolEngine
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    num_batches = (R + B - 1) // B

    for b in range(num_batches):
        start = b * B
        end   = min(R, (b + 1) * B)
        Rb    = end - start

        # ---- radius distribution sampler ----
        sampler = LogNormalSamplerBatched(
            R=Rb,
            trunc_A=args.ml_A + 5.0,
            trunc_B=args.ml_B - 5.0,
            ml_A=args.ml_A,
            ml_B=args.ml_B,
            mean_lo=args.mean_lo,
            mean_hi=args.mean_hi,
            cv_lo=args.cv_lo,
            cv_hi=args.cv_hi,
            init_mean_A=None,
            init_cv=0.20,
            learn_mean=True,
            learn_cv=True,
            device=args.device,
            dtype=dtype,
        )

        # ---- shell parameters ----
        cs_params = CoreShellParamsBatched(
            R=Rb,
            d_lo=args.d_lo,
            d_hi=args.d_hi,
            contrast_lo=args.contrast_lo,
            contrast_hi=args.contrast_hi,
            device=args.device,
            dtype=dtype,
        )

        # ---- integrator ----
        integrator = CoreShellQuadBatched(
            predictor=predictor,
            sampler=sampler,
            cs_params=cs_params,
            K=K,
            rule="legendre",
            device=args.device,
            dtype=dtype,
        )

        # ---- Sobol initialisation (4 dims: mean, cv, d, contrast) ----
        sob = SobolEngine(dimension=4, scramble=True)
        U   = sob.draw(Rb).to(dev, dtype=dtype)
        margin = 0.05
        U_m = margin + (1.0 - 2.0 * margin) * U

        with torch.no_grad():
            eps = 1e-6
            sampler.mean_raw.data.copy_(
                torch.log(U_m[:, 0].clamp(eps, 1 - eps) / (1 - U_m[:, 0].clamp(eps, 1 - eps)))
            )
            sampler.cv_raw.data.copy_(
                torch.log(U_m[:, 1].clamp(eps, 1 - eps) / (1 - U_m[:, 1].clamp(eps, 1 - eps)))
            )
            cs_params.set_from_sobol(U_m[:, 2], U_m[:, 3])

        opt = AdamW(
            list(sampler.parameters()) + list(cs_params.parameters()),
            lr=args.lr,
        )
        sched = CosineAnnealingLR(opt, T_max=args.total_steps, eta_min=args.lr * 0.1)

        # ---- optimisation loop ----
        for step in range(args.total_steps):
            opt.zero_grad(set_to_none=True)

            ln_mean, _, _ = integrator()
            pred_l = normalize_logI_masked(q, ln_mean, q_mask)  # [Rb, n_q]
            pred_r = pred_l.exp()

            Lmse_l = mse_fn(
                pred_l[:, q_mask],
                y_log[q_mask].unsqueeze(0).expand(Rb, -1),
            ).mean(dim=1)
            Lmse_r = mse_fn(
                pred_r[:, q_mask] / 1e5,
                y_exp[q_mask].unsqueeze(0).expand(Rb, -1) / 1e5,
            ).mean(dim=1)

            loss = (0.1 * Lmse_l + Lmse_r).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(sampler.parameters()) + list(cs_params.parameters()),
                max_norm=args.grad_clip,
            )
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                print(
                    f"  batch {b+1}/{num_batches}  step {step:4d}/{args.total_steps}"
                    f"  loss={loss.item():.6f}"
                )

        # ---- final evaluation (full q range) ----
        with torch.no_grad():
            ln_mean, _, _ = integrator()
            pred_l = normalize_logI_masked(q, ln_mean, q_mask)
            pred_r = pred_l.exp()

            Lmse = mse_fn(
                pred_l[:, q_mask],
                y_log[q_mask].unsqueeze(0).expand(Rb, -1),
            ).mean(dim=1)
            rmse_log = Lmse.sqrt()

            phys_r = sampler.physical_radius_params()
            phys_s = cs_params.physical_values()

        for i in range(Rb):
            all_results.append({
                "restart":         start + i,
                "rmse_log":        float(rmse_log[i].cpu()),
                "radius_mean_A":   float(phys_r["radius_mean_A"][i].cpu()),
                "radius_std_A":    float(phys_r["radius_std_A"][i].cpu()),
                "radius_cv":       float(phys_r["radius_cv"][i].cpu()),
                "d_A":             float(phys_s["d_A"][i].cpu()),
                "contrast_ratio":  float(phys_s["contrast_ratio"][i].cpu()),
                "curve_log_norm":  pred_l[i].detach().cpu(),
                "curve_norm":      pred_r[i].detach().cpu(),
            })

        batch_path = os.path.join(args.save_dir, f"batch_{b:04d}.pt")
        torch.save(all_results[-Rb:], batch_path)

        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- save all results + summary ----
    torch.save(all_results, os.path.join(args.save_dir, "all_results.pt"))

    losses   = [r["rmse_log"] for r in all_results]
    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    best     = all_results[best_idx]
    print(f"\n{'='*60}")
    print(f"Best fit (low-q RMSE_log = {best['rmse_log']:.4f}):")
    print(f"  mean radius   = {best['radius_mean_A']:.1f} Å")
    print(f"  std  radius   = {best['radius_std_A']:.1f} Å")
    print(f"  cv            = {best['radius_cv']:.3f}")
    print(f"  shell thick d = {best['d_A']:.1f} Å")
    print(f"  contrast c    = {best['contrast_ratio']:.3f}")
    print(f"{'='*60}")

    torch.save({
        "q":           q.cpu(),
        "y_exp":       y_exp.cpu(),
        "y_log":       y_log.cpu(),
        "q_mask":      q_mask.cpu(),
        "best_result": best,
        "all_results_summary": [{
            "restart":        r["restart"],
            "rmse_log":       r["rmse_log"],
            "radius_mean_A":  r["radius_mean_A"],
            "radius_std_A":   r["radius_std_A"],
            "radius_cv":      r["radius_cv"],
            "d_A":            r["d_A"],
            "contrast_ratio": r["contrast_ratio"],
        } for r in all_results],
    }, os.path.join(args.save_dir, "summary.pt"))


if __name__ == "__main__":
    main()
