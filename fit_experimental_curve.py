#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from scipy.io import loadmat

from aisaxs.models.mlps.dtc_model import CurvePredictor, DCTModel, make_dct2_ortho_uniform
from aisaxs.models.sampling.curve_fitting import fit_with_random_inits_batched
from aisaxs.models.sampling.integrator import QuadFullCurveBatched
from aisaxs.models.sampling.parameterized_distributions import LogNormalSamplerBatched
from aisaxs.models.sampling.parameters import UnifiedParamsBatched
from aisaxs.models.sampling.utils import make_specs, normalize_q2
from data.lnp.preprocess_ml_data import cubic_interp_monotone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an experimental SAXS curve with the heterogeneous-interior (GRF + neural surrogate) forward model, using multi-start gradient-based optimisation."
    )
    parser.add_argument(
        "--experimental-mat",
        type=str,
        default="data/lnp/experimental_curve.mat",
        help="Path to experimental .mat file.",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="data/lnp/setup_shell_shift/output_data.pt",
        help="Path to output_data.pt.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/lnp/setup_shell_shift/metadata.pt",
        help="Path to metadata.pt (normalization stats and q grid).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="saved_models/model_tot.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/lnp/config.yaml",
        help="Path to YAML config with parameter ranges.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/curve_fitting/fitted_curves_experimental",
        help="Directory to save fitting results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Torch dtype.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for restarts.",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=16 * 256,
        help="Total number of random restarts.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1000,
        help="Total optimization steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Warmup steps.",
    )
    parser.add_argument(
        "--lr-sampler",
        type=float,
        default=1e-2,
        help="Learning rate for sampler parameters.",
    )
    parser.add_argument(
        "--lr-params",
        type=float,
        default=1e-2,
        help="Learning rate for model parameters.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Logging frequency.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--save-batch-every",
        type=int,
        default=1,
        help="Save every N batches.",
    )
    parser.add_argument(
        "--dct-size",
        type=int,
        default=300,
        help="DCT basis size.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=1025,
        help="Hidden width of DCTModel.",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=0.200,
        help="Fit only q <= q_max (Å⁻¹). Default: 0.200 (full range). "
             "Pass e.g. 0.02, 0.05, 0.10 to study information content.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_experimental_curve(experimental_mat_path: str, q_target: torch.Tensor) -> torch.Tensor:
    experimental_data = loadmat(experimental_mat_path)
    q_exp = torch.from_numpy(experimental_data["q_data"].flatten())
    I_exp = torch.from_numpy(experimental_data["I_data"].flatten())

    I_experimental = cubic_interp_monotone(
        q_exp,
        I_exp.unsqueeze(0),
        q_target,
    ).flatten()

    I_exp_norm, _ = normalize_q2(q_target, I_experimental)
    return I_exp_norm


def build_curve_predictor(
    model_path: str,
    y_mean: float,
    y_std: float,
    dct_size: int,
    hidden: int,
    device: str,
) -> CurvePredictor:
    Phi_dct = torch.from_numpy(make_dct2_ortho_uniform(dct_size, dct_size)).float()
    model = DCTModel(Phi_dct, 9, hidden=hidden).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    curve_predictor = CurvePredictor(y_mean, y_std, model)
    return curve_predictor.to(device)


def main() -> None:
    args = parse_args()

    device = args.device
    dtype = get_torch_dtype(args.dtype)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    output_data = torch.load(args.output_data, map_location="cpu")
    metadata = torch.load(args.metadata, map_location="cpu")

    q = output_data["q"]
    y_mean = metadata["y_mean"].item()
    y_std = metadata["y_std"].item()

    q_mask = (q <= args.q_max)
    print(
        f"[INFO] Fitting q <= {args.q_max:.4f} Å⁻¹  "
        f"({q_mask.sum().item()}/{q.numel()} points)"
    )

    y_exp = load_experimental_curve(args.experimental_mat, q).to(device, dtype=dtype)

    curve_predictor = build_curve_predictor(
        model_path=args.model_path,
        y_mean=y_mean,
        y_std=y_std,
        dct_size=args.dct_size,
        hidden=args.hidden,
        device=device,
    )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    ranges = cfg["Parameter_ranges"]

    overrides = {
        "GRF_shift": {"lo": -0.98, "hi": 0.98, "init": -0.1, "learn": True},
        "w_q1": {"lo": 0.01, "hi": 0.99, "init": 0.7, "learn": True},
        "m_q1": {"lo": 0.05, "hi": 0.15, "init": 0.10, "learn": True},
        "m_q2": {"lo": 0.05, "hi": 0.15, "init": 0.11, "learn": True},
        "s_q1": {"lo": 0.01, "hi": 0.15, "init": 0.015, "learn": True},
        "s_q2": {"lo": 0.01, "hi": 0.15, "init": 0.10, "learn": True},
        "shell_thickness": {"lo": 50.0, "hi": 65.0, "init": 60.0, "learn": True},
        "relative_rho_shell": {"lo": 0.7, "hi": 1.3, "init": 0.90, "learn": True},
    }

    specs = make_specs(
        ranges,
        default_learn=True,
        overrides=overrides,
    )

    r_col = 1
    d = max(int(spec["col"]) for spec in specs.values()) + 1
    batch_size = args.batch_size
    num_restarts = args.num_restarts

    result = fit_with_random_inits_batched(
        num_restarts=num_restarts,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        q=q,
        y_exp=y_exp,
        curve_predictor=curve_predictor,
        base_specs=specs,
        make_boxed=lambda R: UnifiedParamsBatched(
            specs,
            R=batch_size,
            r_col=r_col,
            device=device,
            dtype=dtype,
        ),
        sampler_factory=lambda R: LogNormalSamplerBatched(
            R=batch_size,
            trunc_A=105.0,
            trunc_B=495.0,
            ml_A=100.0,
            ml_B=500.0,
            mean_lo=150.0,
            mean_hi=270.0,
            cv_lo=0.10,
            cv_hi=0.35,
            init_mean_A=205.32,
            init_cv=0.20,
            learn_mean=True,
            learn_cv=True,
        ),
        integrator_factory=lambda sampler: QuadFullCurveBatched(
            d=d,
            r_col=r_col,
            sampler=sampler,
            K=32,
            rule="legendre",
            device=device,
            dtype=dtype,
        ),
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        lr_sampler=args.lr_sampler,
        lr_params=args.lr_params,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        save_batch_every=args.save_batch_every,
        save_curves=True,
        q_mask=q_mask.to(device, dtype=torch.bool),
    )

    return result


if __name__ == "__main__":
    main()