import os, copy, math, json
from typing import Callable, Optional, Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.quasirandom import SobolEngine
from .loss_functions import corr_loss, cdf_loss,corr_loss_batched, cdf_loss_batched
from .utils import Q2Normalizer
from .parameters import UnifiedParams, UnifiedParamsBatched
from.parameterized_distributions import BaseRadiusSampler
from .integrator import make_X_from_specs_physical
# --- Simple warmup+cosine scheduler factory (works with LambdaLR) ---
def _warmup_cosine_with_floor_factory(
    *,
    total_steps: int,
    warmup_steps: int,
    min_factor: float,
) -> Callable[[int], float]:
    """
    Returns a lr_lambda(step) for LambdaLR:
    - Linear warmup to 1.0 over warmup_steps
    - Cosine decay to min_factor over the remainder
    """
    warmup_steps = max(1, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        t = max(0.0, min(1.0, t))
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_factor + (1.0 - min_factor) * cos_decay

    return lr_lambda


# --- Helper: shrink interval to avoid exact boundaries ---
def _shrink_interval(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
    """
    Shrinks [lo, hi] to [lo_eff, hi_eff] to avoid sampling right on the edges.
    frac is the relative margin on each side (e.g., 0.05 -> 5% per side).
    """
    span = hi - lo
    if span <= 0.0:
        return lo, hi
    margin = frac * span
    if 2.0 * margin >= span:  # avoid killing the interval
        margin = 0.49 * span
    return lo + margin, hi - margin

def _sobol_specs_inits(
    base_specs: Dict[str, Dict[str, Any]],
    num_restarts: int,
    *,
    only_learnable: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    boundary_frac: float = 0.05,
    mq_sep_frac: float = 0.10,
) -> List[Dict[str, Dict[str, Any]]]:

    names: List[str] = []
    lo_list: List[float] = []
    hi_list: List[float] = []

    for name, s in base_specs.items():
        # Always include m_q1/m_q2 in init sampling if present (so ordering can be enforced)
        if only_learnable and (not s.get("learn", True)) and (name not in ("m_q1", "m_q2")):
            continue
        lo = float(s["lo"])
        hi = float(s["hi"])
        names.append(name)
        lo_list.append(lo)
        hi_list.append(hi)

    D = len(names)
    if D == 0:
        return [copy.deepcopy(base_specs) for _ in range(num_restarts)]

    name_to_idx = {name: i for i, name in enumerate(names)}

    lo_eff, hi_eff = [], []
    for lo, hi in zip(lo_list, hi_list):
        lo_e, hi_e = _shrink_interval(lo, hi, frac=boundary_frac)
        lo_eff.append(lo_e)
        hi_eff.append(hi_e)

    lo_t = torch.tensor(lo_eff, device=device, dtype=dtype)
    hi_t = torch.tensor(hi_eff, device=device, dtype=dtype)

    engine = SobolEngine(dimension=D, scramble=True)
    U = engine.draw(num_restarts).to(device=device, dtype=dtype)

    specs_list: List[Dict[str, Dict[str, Any]]] = []

    has_mq_pair = ("m_q1" in base_specs) and ("m_q2" in base_specs)
    idx_mq1 = name_to_idx.get("m_q1", None) if has_mq_pair else None
    idx_mq2 = name_to_idx.get("m_q2", None) if has_mq_pair else None

    mq_lo_eff = mq_hi_eff = None
    if has_mq_pair and (idx_mq1 is not None):
        mq_lo_eff = float(lo_t[idx_mq1].item())
        mq_hi_eff = float(hi_t[idx_mq1].item())

    for r in range(num_restarts):
        specs_r = copy.deepcopy(base_specs)
        u_r = U[r]
        vals_r = lo_t + (hi_t - lo_t) * u_r

        if has_mq_pair and (idx_mq1 is not None) and (idx_mq2 is not None):
            u1 = float(u_r[idx_mq1].item())
            u2 = float(u_r[idx_mq2].item())

            if (mq_lo_eff is None) or (mq_hi_eff is None):
                m1_init = float(vals_r[idx_mq1].item())
                m2_init = float(vals_r[idx_mq2].item())
                m1_init, m2_init = min(m1_init, m2_init), max(m1_init, m2_init)
            else:
                range_m = mq_hi_eff - mq_lo_eff
                if range_m <= 0.0:
                    m1_init = mq_lo_eff
                    m2_init = mq_lo_eff
                else:
                    delta_min = mq_sep_frac * range_m
                    hi_for_m1 = mq_hi_eff - delta_min
                    if hi_for_m1 <= mq_lo_eff:
                        hi_for_m1 = mq_lo_eff
                        delta_min = 0.0

                    m1_init = mq_lo_eff + (hi_for_m1 - mq_lo_eff) * u1

                    available = mq_hi_eff - m1_init
                    if available <= 0.0:
                        m2_init = m1_init
                    else:
                        delta_min_eff = min(delta_min, available)
                        span = available - delta_min_eff
                        m2_init = m1_init + delta_min_eff if span <= 0.0 else (m1_init + delta_min_eff + u2 * span)

            vals_r[idx_mq1] = torch.tensor(m1_init, device=device, dtype=dtype)
            vals_r[idx_mq2] = torch.tensor(m2_init, device=device, dtype=dtype)

        for j, name in enumerate(names):
            # Only overwrite init for names we sampled (others keep base_specs init)
            specs_r[name]["init"] = float(vals_r[j].item())

        specs_list.append(specs_r)

    return specs_list




# --- Helper: safely snapshot a module's named parameters to plain tensors ---
def _dump_named_parameters(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone().cpu() for name, p in module.named_parameters()}


# --- Eval helper ---
@torch.no_grad()
def _eval_curve(curve_predictor, integrator, norm_I, boxed):
    curve_predictor.eval()
    ln_mean, _,_ = integrator(curve_predictor, boxed)
    pred_l = norm_I(ln_mean)
    pred_r = pred_l.exp()
    curve_predictor.train()
    return pred_l, pred_r, ln_mean


def export_physical_state(boxed: UnifiedParams, sampler: BaseRadiusSampler) -> dict:
    struct_params = {k: v.detach().cpu() for k, v in boxed.values().items()}
    radius_params = sampler.physical_radius_params()  # already CPU
    out = {}
    out.update(struct_params)
    out.update(radius_params)
    return out

def fit_from_init(
    *,
    device: str,
    dtype: torch.dtype,
    q: torch.Tensor,
    y_exp: torch.Tensor,
    curve_predictor: torch.nn.Module,
    base_specs: Dict[str, Dict[str, Any]],
    sampler_factory: Callable[[], torch.nn.Module],
    integrator_factory: Callable[[torch.nn.Module], Any],
    total_steps: int = 3000,
    warmup_steps: int = 300,
    lr_sampler: float = 1e-2,
    lr_params: float = 1e-3,
    grad_clip: float = 5.0,
    min_factor_sampler: float = 0.6,
    min_factor_params: float = 0.6,
    log_every: int = 100,
    save_dir: Optional[str] = None,
    status_cb: Optional[Callable[[int, int, float], None]] = None,  # (step, loss)
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    r_col: int = 1,
) -> Dict[str, Any]:
    """
    Single-start fit that uses the initial values in base_specs["..."]["init"]
    directly (no Sobol / randomised inits). Optimises once and returns a
    single result dict in the same format as a single element of the
    multi-start version.
    """

    # ---- canonical device object ----
    dev = torch.device(device)

    # ---- setup model + data ----
    curve_predictor = curve_predictor.to(device=dev, dtype=dtype).eval()
    q = q.to(device=dev, dtype=dtype)
    y_exp = y_exp.to(device=dev, dtype=dtype)

    norm_I = Q2Normalizer(q)
    mse = nn.MSELoss()

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Deep copy so caller's specs are not modified
    specs_0 = copy.deepcopy(base_specs)
    boxed = UnifiedParams(specs_0, r_col=r_col, device=dev, dtype=dtype)

    # Fresh sampler + integrator
    sampler = sampler_factory().to(device=dev, dtype=dtype)
    integrator = integrator_factory(sampler)

    # ---- Optimiser + schedulers ----
    opt = AdamW(
        [
            {"params": sampler.parameters(), "lr": lr_sampler},
            {"params": boxed.parameters(),   "lr": lr_params},
        ],
        weight_decay=weight_decay,
        betas=betas,
    )

    lambdas = [
        _warmup_cosine_with_floor_factory(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_factor=min_factor_params,
        ),
        _warmup_cosine_with_floor_factory(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_factor=min_factor_params,
        ),
    ]
    sched = LambdaLR(opt, lr_lambda=lambdas)

    # ---- helper: handle integrator returning (ln_mean, ...) with variable arity ----
    def _get_ln_mean():
        out,_ ,_ = integrator(curve_predictor, boxed)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    # ---- training loop ----
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)

        ln_mean = _get_ln_mean()
        pred_l = norm_I(ln_mean)      # log-domain normalised curve
        pred_r = pred_l.exp()         # linear-domain normalised curve

        y_log = y_exp.log()

        loss = (
            corr_loss(pred_l, y_log) * 10.0 +
            cdf_loss(pred_l,  y_log) * 100.0 +
            mse(pred_l,       y_log) * 0.1 +
            mse(pred_r / 1e5, y_exp / 1e5)
        )

        loss.backward()

        # (Optional micro-optimisation: build list once. But this is fine.)
        torch.nn.utils.clip_grad_norm_(
            [*sampler.parameters(), *boxed.parameters()],
            max_norm=grad_clip,
        )

        opt.step()
        sched.step()

        if (step % log_every == 0) and (status_cb is not None):
            status_cb(step, float(loss.detach().cpu()))

    # ---- final eval ----
    with torch.no_grad():
        pred_l, pred_r, raw_ln = _eval_curve(curve_predictor, integrator, norm_I, boxed)
        phys_params = export_physical_state(boxed, sampler)

    out = {
        "restart": 0,
        "final_loss": float(loss.detach().cpu()),
        "boxed_state": _dump_named_parameters(boxed),
        "boxed_physical": phys_params,
        "sampler_state": _dump_named_parameters(sampler),
        "curve_norm": pred_r.detach().clone().cpu(),
        "curve_log_norm": pred_l.detach().clone().cpu(),
        "raw_log_curve": raw_ln.detach().clone().cpu(),
        "specs_used": specs_0,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(out, os.path.join(save_dir, "fit_single.pt"))
        # (meta is currently unused; fine to keep or remove)

    return out



def fit_with_random_inits(
    *,
    num_restarts: int,
    device: str,
    dtype: torch.dtype,
    q: torch.Tensor,
    y_exp: torch.Tensor,
    curve_predictor: torch.nn.Module,
    base_specs: Dict[str, Dict[str, Any]],
    sampler_factory: Callable[[], torch.nn.Module],
    integrator_factory: Callable[[torch.nn.Module], Any],
    total_steps: int = 3000,
    warmup_steps: int = 300,
    lr_sampler: float = 1e-2,
    r_col: int = 1,
    lr_params: float = 1e-3,
    grad_clip: float = 5.0,
    min_factor_params: float = 0.6,
    log_every: int = 100,
    save_dir: Optional[str] = None,
    status_cb: Optional[Callable[[int, int, float], None]] = None,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
) -> List[Dict[str, Any]]:

    assert num_restarts >= 1

    device_t = torch.device(device)
    q = q.to(device=device_t, dtype=dtype)
    y_exp = y_exp.to(device=device_t, dtype=dtype)
    y_log = y_exp.log()

    curve_predictor = curve_predictor.to(device=device_t, dtype=dtype).eval()
    norm_I = Q2Normalizer(q)

    mse = nn.MSELoss()
    results: List[Dict[str, Any]] = []

    sobol_specs_list = _sobol_specs_inits(
        base_specs,
        num_restarts,
        only_learnable=True,
        device=device,
        dtype=dtype,
        boundary_frac=0.05,
        mq_sep_frac=0.10,
    )

    for r in range(num_restarts):
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        specs_r = sobol_specs_list[r]
        boxed = UnifiedParams(specs_r, r_col=r_col, device=device, dtype=dtype)

        sampler = sampler_factory().to(device=device_t, dtype=dtype)
        integrator = integrator_factory(sampler)

        opt = AdamW(
            [
                {"params": sampler.parameters(), "lr": lr_sampler},
                {"params": list(boxed.parameters()), "lr": lr_params},
            ],
            weight_decay=weight_decay,
            betas=betas,
        )

        sched = LambdaLR(
            opt,
            lr_lambda=[
                _warmup_cosine_with_floor_factory(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_factor=min_factor_params,
                ),
                _warmup_cosine_with_floor_factory(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_factor=min_factor_params,
                ),
            ],
        )

        for step in range(total_steps):
            opt.zero_grad(set_to_none=True)

            ln_mean, _,_ = integrator(curve_predictor, boxed)   # <-- IMPORTANT
            pred_l = norm_I(ln_mean)
            pred_r = pred_l.exp()

            loss = (
                corr_loss(pred_l, y_log) * 10.0 +
                cdf_loss(pred_l,  y_log) * 100.0 +
                mse(pred_l,       y_log) * 0.1 +
                mse(pred_r / 1e5, y_exp / 1e5)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*sampler.parameters(), *boxed.parameters()],
                max_norm=grad_clip,
            )
            opt.step()
            sched.step()

            if (step % log_every == 0) and (status_cb is not None):
                status_cb(r, step, float(loss.detach().cpu()))

        with torch.no_grad():
            pred_l, pred_r, raw_ln = _eval_curve(curve_predictor, integrator, norm_I, boxed)
            phys_params = export_physical_state(boxed, sampler)

        out = {
            "restart": r,
            "final_loss": float(loss.detach().cpu()),
            "boxed_state": _dump_named_parameters(boxed),
            "boxed_physical": phys_params,
            "sampler_state": _dump_named_parameters(sampler),
            "curve_norm": pred_r.detach().clone().cpu(),
            "curve_log_norm": pred_l.detach().clone().cpu(),
            "raw_log_curve": raw_ln.detach().clone().cpu(),
            "specs_used": specs_r,
        }

        results.append(out)

    return results


@torch.no_grad()
def derive_curve_from_specs(
    *,
    specs: Dict[str, Dict[str, Any]],
    sampler_factory: Callable[[], BaseRadiusSampler],
    integrator_factory: Callable[[BaseRadiusSampler], Any],
    curve_predictor: nn.Module,
    q: torch.Tensor,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    r_col: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Perform a SINGLE forward evaluation:
        specs -> UnifiedParams -> sampler -> integrator -> curve
    
    Returns:
        {
            "pred_log_norm":   log(I_norm),
            "pred_norm":       I_norm,
            "raw_log_curve":   ln(I),
            "physical_params": {...}
        }
    """

    device = torch.device(device)
    curve_predictor = curve_predictor.to(device=device, dtype=dtype).eval()
    q = q.to(device=device, dtype=dtype)

    # --- Setup normalizer for Q^2 normalization ---
    norm_I = Q2Normalizer(q)

    # --- Build UnifiedParams with given initial specs ---
    specs_local = {k: dict(v) for k, v in specs.items()}  # deep copy
    boxed = UnifiedParams(specs_local, r_col=r_col,
                          device=device, dtype=dtype)

    sampler = sampler_factory().to(device=device, dtype=dtype)
    integrator = integrator_factory(sampler)

    import time
    t_ = time.time()
    ln_mean, y, theta = integrator(curve_predictor, boxed)
    t_der = (time.time()-t_)
    pred_l = norm_I(ln_mean)                  # log(I_norm)
    pred_r = pred_l.exp()                     # I_norm

    # --- Export physical parameter values ---
    phys_params = export_physical_state(boxed, sampler)

    return {
        "pred_log_norm": pred_l.cpu(),
        "pred_norm": pred_r.cpu(),
        "raw_log_curve": ln_mean.cpu(),
        "physical_params": phys_params,
        "y":y,
        "theta":theta,
        "time": t_der
    }




import copy
from typing import Dict, Any, Callable, Optional, List

@torch.no_grad()
def export_physical_state_batched(boxed, sampler) -> dict:
    # boxed.values(): dict[str, Tensor] where each Tensor is [R] (or [R,...])
    struct_params = boxed.values()  # already detached ideally
    radius_params = sampler.physical_radius_params()  # should be dict of [R]
    out = {}
    out.update(struct_params)
    out.update(radius_params)
    return out




def sobol_inits_from_specs(
    specs: dict,
    *,
    R: int,
    device: str,
    dtype: torch.dtype,
    boundary_frac: float = 0.05,
    mq_sep_frac: float = 0.10,
    r_col: int = 1,
):
    dev = torch.device(device)

    # --- collect learnable names (excluding dependent m_q2) ---
    learn_names = []
    lo = []
    hi = []
    for name, s in specs.items():
        j = int(s["col"])
        if j == r_col:
            raise ValueError(f"Spec '{name}' uses r_col={r_col}. r_col must NOT appear in specs.")
        if name == "m_q2":
            continue
        if bool(s.get("learn", False)):
            learn_names.append(name)
            lo.append(float(s["lo"]))
            hi.append(float(s["hi"]))

    L = len(learn_names)
    if L == 0:
        # no learnables: still need eta init maybe
        learn_cols = []
        init_phys = torch.empty((R, 0), device=dev, dtype=dtype)
    else:
        learn_cols = [int(specs[n]["col"]) for n in learn_names]

        lo_t = torch.tensor(lo, device=dev, dtype=dtype)
        hi_t = torch.tensor(hi, device=dev, dtype=dtype)

        # shrink interval
        span = hi_t - lo_t
        margin = boundary_frac * span
        margin = torch.minimum(margin, 0.49 * span)
        lo_eff = lo_t + margin
        hi_eff = hi_t - margin

        # Sobol U: [R,L]
        U = SobolEngine(dimension=L, scramble=True).draw(R).to(device=dev, dtype=dtype)
        init_phys = lo_eff + (hi_eff - lo_eff) * U  # [R,L]

    # --- handle m_q1/m_q2 ordering at init by constructing eta per restart ---
    has_mq = ("m_q1" in specs) and ("m_q2" in specs)
    init_eta = None
    if has_mq:
        mq_lo = float(specs["m_q1"]["lo"])
        mq_hi = float(specs["m_q1"]["hi"])

        # We will get m_q1 init from init_phys if it's learnable; else from specs["m_q1"]["init"]
        if "m_q1" in learn_names:
            k1 = learn_names.index("m_q1")
            m1 = init_phys[:, k1]  # [R]
        else:
            m1 = torch.full((R,), float(specs["m_q1"]["init"]), device=dev, dtype=dtype)

        # For m_q2, we DO NOT make it an independent parameter; we set eta so that m_q2 init matches a Sobol draw.
        # We'll draw u2 separately to define m2 init in [m1+sep, hi].
        u2 = SobolEngine(dimension=1, scramble=True).draw(R).to(device=dev, dtype=dtype).squeeze(1)  # [R]

        # enforce separation (in physical space) inside [mq_lo, mq_hi]
        range_m = max(mq_hi - mq_lo, 1e-12)
        delta_min = mq_sep_frac * range_m

        hi_for_m1 = mq_hi - delta_min
        # if m1 could be > hi_for_m1 because it came from other rules, clamp it
        m1 = torch.minimum(m1, torch.tensor(hi_for_m1, device=dev, dtype=dtype))

        # choose m2 in [m1+delta_min, mq_hi]
        available = (mq_hi - m1).clamp_min(0.0)
        delta_eff = torch.minimum(torch.full_like(available, delta_min), available)
        span2 = (available - delta_eff).clamp_min(0.0)
        m2 = m1 + delta_eff + u2 * span2  # [R]
        m2 = torch.maximum(m2, m1)        # safety

        # eta = (m2 - m1)/(mq_hi - m1)
        denom = (mq_hi - m1).clamp_min(1e-12)
        eta = (m2 - m1) / denom

        eps = 1e-4
        eta = eta.clamp(eps, 1.0 - eps)
        init_eta = eta  # [R]

    return {
        "learn_names": learn_names,
        "learn_cols": learn_cols,
        "init_phys_learn": init_phys,  # [R,L]
        "init_eta": init_eta,          # [R] or None
    }





def fit_with_random_inits_batched(
    *,
    num_restarts: int,
    batch_size: int = 256,

    device: str,
    dtype: torch.dtype,
    q: torch.Tensor,
    y_exp: torch.Tensor,
    curve_predictor: torch.nn.Module,
    base_specs: Dict[str, Dict[str, Any]],

    sampler_factory: Callable[[int], torch.nn.Module],      # takes R_batch
    integrator_factory: Callable[[torch.nn.Module], Any],   # takes sampler -> batched integrator
    make_boxed: Callable[[int], Any],                       # takes R_batch -> UnifiedParamsBatched

    total_steps: int = 3000,
    warmup_steps: int = 300,
    lr_sampler: float = 1e-2,
    lr_params: float = 1e-3,
    r_col: int = 1,
    grad_clip: float = 5.0,
    min_factor_params: float = 0.6,
    log_every: int = 100,
    status_cb: Optional[Callable[[int, int, float], None]] = None,  # (batch_id, step, loss)
    weight_decay: float = 1e-5,
    betas: tuple = (0.9, 0.999),

    # saving
    save_dir: Optional[str] = None,
    save_batch_every: int = 1,          # save every N batches (1 = always)
    save_curves: bool = True,           # if False: save only params + losses

    # q range restriction
    q_mask: Optional[torch.Tensor] = None,  # bool [n_q]; if None use all q points

    # init control
    boundary_frac: float = 0.05,
    mq_sep_frac: float = 0.10,
):
    """
    Chunked batched fitting:
      - builds Sobol initializations ONCE for full R
      - runs optimization in chunks of size batch_size
      - returns a flat list of length R with global restart indices
    """
    R = int(num_restarts)
    assert R >= 1
    B = int(batch_size)
    assert B >= 1

    dev = torch.device(device)

    # constants on device
    q = q.to(dev, dtype=dtype)
    y_exp = y_exp.to(dev, dtype=dtype)
    y_log = y_exp.log()

    curve_predictor = curve_predictor.to(dev, dtype=dtype).eval()
    norm_I = Q2Normalizer(q)

    # base_row in PHYSICAL scale (single row [d])
    X0 = make_X_from_specs_physical(base_specs, r_col=r_col, device=device, dtype=dtype)  # [1,d]
    base_row_phys = X0[0]  # [d]
    d = base_row_phys.numel()

    # ---- Sobol init for FULL R, ONCE ----
    sob_full = sobol_inits_from_specs(
        base_specs,
        R=R,
        device=device,
        dtype=dtype,
        boundary_frac=boundary_frac,
        mq_sep_frac=mq_sep_frac,
        r_col=r_col,
    )
    init_phys_full = sob_full["init_phys_learn"]   # [R, L]
    init_eta_full  = sob_full["init_eta"]          # [R] or None

    # prep saving
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if q_mask is not None:
            torch.save({"q_mask": q_mask.cpu(), "q": q.cpu()},
                       os.path.join(save_dir, "q_mask.pt"))

    mse = nn.MSELoss(reduction="none")

    results: List[Dict[str, Any]] = []

    num_batches = (R + B - 1) // B

    for b in range(num_batches):
        start = b * B
        end = min(R, (b + 1) * B)
        Rb = end - start

        # slice initializations for this batch
        init_phys_b = init_phys_full[start:end]  # [Rb, L]
        init_eta_b = None if init_eta_full is None else init_eta_full[start:end]  # [Rb]

        # create modules for this batch
        boxed = make_boxed(Rb).to(dev, dtype=dtype)
        sampler = sampler_factory(Rb).to(dev, dtype=dtype)
        integrator = integrator_factory(sampler)  # must support (curve_predictor, base_row_norm)

        # load Sobol init into boxed params
        # NOTE: your boxed.set_init_from_phys expects [R,L] + [R] (or None)
        boxed.set_init_from_phys(init_phys_b, init_eta_b)

        opt = AdamW(
            [{"params": sampler.parameters(), "lr": lr_sampler},
             {"params": boxed.parameters(),    "lr": lr_params}],
            weight_decay=weight_decay,
            betas=betas,
        )

        sched = LambdaLR(
            opt,
            lr_lambda=[
                _warmup_cosine_with_floor_factory(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_factor=min_factor_params,
                ),
                _warmup_cosine_with_floor_factory(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_factor=min_factor_params,
                ),
            ],
        )

        # ---- train batch ----
        for step in range(total_steps):
            opt.zero_grad(set_to_none=True)

            # [Rb,d] model-ready (normalized to the model’s expected scaling)
            base_row_norm = boxed.apply_to_base(base_row_phys, normalize=True)

            # integrator returns ln_mean [Rb,n_q]
            ln_mean, _, _ = integrator(curve_predictor, base_row_norm)

            pred_l = norm_I(ln_mean)       # [Rb,n_q]
            pred_r = pred_l.exp()

            if q_mask is not None:
                pl = pred_l[:, q_mask];  pr = pred_r[:, q_mask]
                yl = y_log[q_mask];      ye = y_exp[q_mask]
            else:
                pl, pr, yl, ye = pred_l, pred_r, y_log, y_exp
            if yl.ndim == 1:
                yl = yl.unsqueeze(0)   # [Q] -> [1, Q]
            elif yl.ndim != 2 or yl.shape[0] != 1:
                raise ValueError(f"Expected yl to have shape [Q] or [1,Q], got {tuple(yl.shape)}")

            if ye.ndim == 1:
                ye = ye.unsqueeze(0)   # [Q] -> [1, Q]
            elif ye.ndim != 2 or ye.shape[0] != 1:
                raise ValueError(f"Expected ye to have shape [Q] or [1,Q], got {tuple(ye.shape)}")

            yl_expanded = yl.expand(pl.shape[0], -1)   # [Rb, Q]
            ye_expanded = ye.expand(pr.shape[0], -1)   # [Rb, Q]

            Lmse_l = ((pl - yl_expanded) ** 2).mean(dim=1)                  # [Rb]
            Lmse_r = (((pr / 1e5) - (ye_expanded / 1e5)) ** 2).mean(dim=1)  # [Rb]

            loss_r = 0.1 * Lmse_l + Lmse_r
            loss = loss_r.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*sampler.parameters(), *boxed.parameters()],
                max_norm=grad_clip
            )
            opt.step()
            sched.step()

            if (step % log_every == 0) and (status_cb is not None):
                status_cb(b, step, float(loss.detach().cpu()))
            if (step % 100 == 0) :
                print(loss)
        # ---- final eval for this batch ----
        with torch.no_grad():
            base_row_norm = boxed.apply_to_base(base_row_phys, normalize=True)
            ln_mean, y_nodes, theta = integrator(curve_predictor, base_row_norm)

            pred_l = norm_I(ln_mean)
            pred_r = pred_l.exp()

            if q_mask is not None:
                pl = pred_l[:, q_mask];  pr = pred_r[:, q_mask]
                yl = y_log[q_mask];      ye = y_exp[q_mask]
            else:
                pl, pr, yl, ye = pred_l, pred_r, y_log, y_exp

            if yl.ndim == 1:
                yl = yl.unsqueeze(0)   # [Q] -> [1, Q]
            elif yl.ndim != 2 or yl.shape[0] != 1:
                raise ValueError(f"Expected yl to have shape [Q] or [1,Q], got {tuple(yl.shape)}")

            if ye.ndim == 1:
                ye = ye.unsqueeze(0)   # [Q] -> [1, Q]
            elif ye.ndim != 2 or ye.shape[0] != 1:
                raise ValueError(f"Expected ye to have shape [Q] or [1,Q], got {tuple(ye.shape)}")

            yl_expanded = yl.expand(pl.shape[0], -1)   # [Rb, Q]
            ye_expanded = ye.expand(pr.shape[0], -1)   # [Rb, Q]

            Lmse_l = ((pl - yl_expanded) ** 2).mean(dim=1)                  # [Rb]
            Lmse_r = (((pr / 1e5) - (ye_expanded / 1e5)) ** 2).mean(dim=1)  # [Rb]

            loss_r = 0.1 * Lmse_l + Lmse_r

            phys = export_physical_state_batched(boxed, sampler)  # dict[k]->[Rb]

        # ---- pack results (global indices) ----
        batch_results = []
        for i in range(Rb):
            ridx = start + i
            out_i = {
                "restart": ridx,
                "batch": b,
                "final_loss": float(loss_r[i].cpu()),
                "boxed_physical": {k: v[i].detach().cpu() for k, v in phys.items()},
            }
            if save_curves:
                out_i.update({
                    "curve_norm": pred_r[i].detach().cpu(),
                    "curve_log_norm": pred_l[i].detach().cpu(),
                    "raw_log_curve": ln_mean[i].detach().cpu(),
                })
            batch_results.append(out_i)

        results.extend(batch_results)

        # ---- optional batch save ----
        if save_dir is not None and (b % save_batch_every == 0):
            pt_path = os.path.join(save_dir, f"batch_{b:05d}_results.pt")
            torch.save(batch_results, pt_path)
            print(
                f"[SAVE] batch {b+1:4d}/{num_batches} |"
            )
            meta = {
                "batch": b,
                "start": start,
                "end": end,
                "R_batch": Rb,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "lr_sampler": lr_sampler,
                "lr_params": lr_params,
                "r_col": r_col,
                "d": d,
                "K": getattr(integrator, "K", None),
            }
            with open(os.path.join(save_dir, f"batch_{b:05d}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        # avoid VRAM fragmentation between big batches
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results
