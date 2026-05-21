#!/usr/bin/env python3
from __future__ import annotations

import json
import os

import numpy as np
import torch
import yaml
from scipy.io import loadmat

from aisaxs.models.mlps.dtc_model import (
    CurvePredictor,
    DCTModel,
    make_dct2_ortho_uniform,
)
from aisaxs.models.sampling.curve_fitting import fit_with_random_inits_batched
from aisaxs.models.sampling.integrator import QuadFullCurveBatched
from aisaxs.models.sampling.parameterized_distributions import LogNormalSamplerBatched
from aisaxs.models.sampling.parameters import UnifiedParamsBatched
from aisaxs.models.sampling.utils import make_specs, normalize_q2
from data.lnp.preprocess_ml_data import cubic_interp_monotone


def load_synth_cases(prefix: str, k: int):
    cases = []
    for i in range(1, k + 1):
        npz = f"{prefix}{i:03d}.npz"
        jsn = f"{prefix}{i:03d}.json"

        d = np.load(npz, allow_pickle=False)
        q = d["q"]
        I_poly = d["I_poly"]

        meta = None
        if os.path.exists(jsn):
            with open(jsn, "r") as f:
                meta = json.load(f)

        cases.append(
            {
                "npz": npz,
                "q": q,
                "I_poly": I_poly,
                "meta": meta,
            }
        )
    return cases

def load_synth_cases(prefix="data/lnp/synth_case_", k=5):
    cases = []
    for i in range(1, k+1):
        if i == 2:
            continue
        npz = f"{prefix}{i:03d}.npz"
        jsn = f"{prefix}{i:03d}.json"
        d = np.load(npz, allow_pickle=False)
        q = d["q"]
        I_poly = d["I_poly"]
        meta = None
        if os.path.exists(jsn):
            with open(jsn, "r") as f:
                meta = json.load(f)
        cases.append({"npz": npz, "q": q, "I_poly": I_poly, "meta": meta})
    return cases


def prepare_synth_curve(case: dict, q_target: torch.Tensor) -> torch.Tensor:
    q_s = torch.from_numpy(case["q"])
    I_s = torch.from_numpy(case["I_poly"])

    I_synth = cubic_interp_monotone(q_s, I_s.unsqueeze(0), q_target).flatten()
    I_synth, _ = normalize_q2(q_target, I_synth)
    return I_synth


def main() -> None:
    experimental_data = loadmat("data/lnp/experimental_curve.mat")

    output_data = torch.load("data/lnp/setup_shell_shift/output_data.pt")
    metadata = torch.load("data/lnp/setup_shell_shift/metadata.pt")

    q = output_data["q"]

    y_mean = metadata["y_mean"].item()
    y_std = metadata["y_std"].item()

    q_exp = torch.from_numpy(experimental_data["q_data"].flatten())
    I_exp = torch.from_numpy(experimental_data["I_data"].flatten())
    I_experiemental = cubic_interp_monotone(q_exp, I_exp.unsqueeze(0), q).flatten()
    I_exp_norm, Qw = normalize_q2(q, I_experiemental)

    Phi_dct = torch.from_numpy(make_dct2_ortho_uniform(300, 300)).float()
    model = DCTModel(Phi_dct, 9, hidden=1025).to("cuda")
    model.load_state_dict(torch.load("saved_models/model_tot.pt"))

    curve_predictor = CurvePredictor(y_mean, y_std, model)
    curve_predictor = curve_predictor.to("cuda")

    cfg_path = "data/lnp/config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    ranges = cfg["Parameter_ranges"]

    device = "cuda"
    dtype = torch.float32

    overrides = {
        "GRF_shift": {"lo": -0.98, "hi": 0.98, "init": -0.1, "learn": True},
        "w_q1": {"lo": 0.01, "hi": 0.99, "init": 0.7, "learn": True},
        "m_q1": {"lo": 0.05, "hi": 0.15, "init": 0.10, "learn": True},
        "m_q2": {"lo": 0.05, "hi": 0.15, "init": 0.11, "learn": True},
        "s_q1": {"lo": 0.01, "hi": 0.15, "init": 0.015, "learn": True},
        "s_q2": {"lo": 0.01, "hi": 0.15, "init": 0.10, "learn": True},
        "shell_thickness": {"lo": 40.0, "hi": 70.0, "init": 60.0, "learn": True},
        "relative_rho_shell": {"lo": 0.7, "hi": 1.3, "init": 0.90, "learn": True},
    }

    specs = make_specs(
        ranges,
        default_learn=True,
        overrides=overrides,
    )

    n_b = 16
    r_col = 1
    d = max(int(s["col"]) for s in specs.values()) + 1
    Rb = 64
    R = n_b * Rb

    case_configs = [
        (1, "data/lnp/syntheteic_cases/synth_case_", "fitted_curves_synth1"),
        (2, "data/lnp/synth_case_", "fitted_curves_synth2"),
        (3, "data/lnp/synth_case_", "fitted_curves_synth3"),
        (4, "data/lnp/synth_case_", "fitted_curves_synth4"),
        (5, "data/lnp/synth_case_", "fitted_curves_synth5"),
    ]

    for case_idx, prefix, save_dir in case_configs:
        cases = load_synth_cases(prefix=prefix, k=case_idx)
        case = cases[case_idx - 1]

        I_synth = prepare_synth_curve(case, q)
        y_syn = I_synth.to(device, dtype=dtype)

        fit_with_random_inits_batched(
            num_restarts=R,
            batch_size=Rb,
            device=device,
            dtype=dtype,
            q=q,
            y_exp=y_syn,
            curve_predictor=curve_predictor,
            base_specs=specs,
            make_boxed=lambda R: UnifiedParamsBatched(
                specs,
                R=Rb,
                r_col=r_col,
                device="cuda",
                dtype=dtype,
            ),
            sampler_factory=lambda R: LogNormalSamplerBatched(
                R=Rb,
                trunc_A=101.0,
                trunc_B=499.0,
                ml_A=100.0,
                ml_B=500.0,
                mean_lo=150.0,
                mean_hi=320.0,
                cv_lo=0.10,
                cv_hi=0.40,
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
                device="cuda",
                dtype=dtype,
            ),
            total_steps=1000,
            warmup_steps=200,
            lr_sampler=1e-2,
            lr_params=1e-2,
            grad_clip=5.0,
            log_every=100,
            weight_decay=0.0,
            save_dir=save_dir,
            save_batch_every=1,
            save_curves=True,
        )


if __name__ == "__main__":
    main()