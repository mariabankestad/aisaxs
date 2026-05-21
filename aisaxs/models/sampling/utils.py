from __future__ import annotations

import torch
from typing import Any, Dict, Optional, Iterable

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

class Q2Normalizer(torch.nn.Module):
    """Normalize log-intensity by q^2 area: logI_norm = logI - ln ∫ q^2 I dq."""
    def __init__(self, q: torch.Tensor):
        super().__init__()
        q64 = q.flatten().to(torch.float64)
        assert torch.all(q64[1:] > q64[:-1]), "q must be strictly increasing"
        self.register_buffer("_logq", torch.log(q64))
        self.register_buffer("_logw", torch.log(0.5 * (q64[1:] - q64[:-1])))
    def forward(self, logI: torch.Tensor) -> torch.Tensor:
        logq = self._logq.to(device=logI.device)
        logw = self._logw.to(device=logI.device)
        logI64 = logI.to(torch.float64)
        z0 = 2 * logq[:-1] + logI64[..., :-1]
        z1 = 2 * logq[1:]  + logI64[...,  1:]
        seg = logw + torch.logsumexp(torch.stack([z0, z1], dim=-1), dim=-1)
        logQw = torch.logsumexp(seg, dim=-1)
        return (logI64 - logQw.unsqueeze(-1)).to(logI.dtype)
    



def _get_minmax(ranges: Dict[str, Any], name: str) -> tuple[float, float]:
    r = ranges[name]
    if isinstance(r, dict):
        return float(r["min"]), float(r["max"])
    return float(getattr(r, "min")), float(getattr(r, "max"))

def make_param_spec(
    *,
    name: str,
    ranges: Dict[str, Any],
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    init: Optional[float] = None,
    learn: Optional[bool] = None,
    lo_norm: Optional[float] = None,
    hi_norm: Optional[float] = None,
) -> Dict[str, Any]:
    if name not in ranges:
        raise KeyError(f"'{name}' not in ranges. Available: {list(ranges.keys())}")

    rmin, rmax = _get_minmax(ranges, name)

    # Defaults from ranges
    lo = rmin if lo is None else float(lo)
    hi = rmax if hi is None else float(hi)

    # Validate lo/hi within ranges
    if not (rmin <= lo <= rmax) or not (rmin <= hi <= rmax):
        raise ValueError(
            f"{name}: lo/hi must be within ranges[{name}]=[{rmin}, {rmax}] "
            f"but got lo={lo}, hi={hi}"
        )
    if not (hi > lo):
        raise ValueError(f"{name}: expected hi > lo, got lo={lo}, hi={hi}")

    # Default init = midpoint of [lo, hi]
    if init is None:
        init = 0.5 * (lo + hi)
    init = float(init)
    if not (lo <= init <= hi):
        raise ValueError(f"{name}: init must be within [lo,hi]=[{lo},{hi}], got {init}")

    # Normalization defaults to the full training range
    lo_norm = rmin if lo_norm is None else float(lo_norm)
    hi_norm = rmax if hi_norm is None else float(hi_norm)
    if not (hi_norm > lo_norm):
        raise ValueError(f"{name}: expected hi_norm > lo_norm, got {lo_norm}, {hi_norm}")

    col = list(ranges.keys()).index(name)

    spec = {
        "lo": float(lo),
        "hi": float(hi),
        "lo_norm": float(lo_norm),
        "hi_norm": float(hi_norm),
        "col": int(col),
        "init": float(init),
    }
    if learn is not None:
        spec["learn"] = bool(learn)
    return spec

def make_specs(
    ranges: Dict[str, Any],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    exclude: Optional[Iterable[str]]=["m_R"],
    include: Optional[Iterable[str]] = None,
    default_learn: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    - If a param is excluded (e.g. "m_R"), it will not appear in specs.
    - If overrides[param] omits fields (lo/hi/init/...), they default from ranges.
    - If overrides is None, everything (except excluded) uses ranges defaults.
    """
    overrides = overrides or {}
    exclude_set = set(exclude or [])
    if include is None:
        names = [k for k in ranges.keys() if k not in exclude_set]
    else:
        include_set = set(include)
        names = [k for k in ranges.keys() if (k in include_set and k not in exclude_set)]

    # Catch overrides for unknown params early
    unknown = set(overrides.keys()) - set(ranges.keys())
    if unknown:
        raise KeyError(f"Overrides contain keys not in ranges: {sorted(unknown)}")

    specs: Dict[str, Dict[str, Any]] = {}
    for name in names:
        kw = dict(overrides.get(name, {}))  # may be empty -> all defaults

        # set default learn if not specified
        if "learn" not in kw:
            kw["learn"] = default_learn

        specs[name] = make_param_spec(name=name, ranges=ranges, **kw)

        # ensure learn exists (make_param_spec sets it only if provided)
        specs[name].setdefault("learn", default_learn)

    return specs