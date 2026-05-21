import torch
import torch.nn as nn
from typing import Dict, Any, List
import math

def _atanh_clamped(z: torch.Tensor) -> torch.Tensor:
    # z must be in (-1,1); clamp for numeric safety
    z = z.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(z) - torch.log1p(-z))

class TanhBoxParam(nn.Module):
    """Learnable scalar constrained to [lo, hi] via tanh mapping."""
    def __init__(self, lo, hi, init=None, device="cuda", dtype=torch.float32):
        super().__init__()
        lo = float(lo)
        hi = float(hi)
        assert hi > lo

        dev = torch.device(device)

        lo_t = torch.tensor(lo, device=dev, dtype=dtype)
        hi_t = torch.tensor(hi, device=dev, dtype=dtype)
        self.register_buffer("lo_buf", lo_t)
        self.register_buffer("hi_buf", hi_t)

        mid = 0.5 * (lo_t + hi_t)
        span_half = 0.5 * (hi_t - lo_t)
        self.register_buffer("mid_buf", mid)
        self.register_buffer("span_half_buf", span_half)

        if init is None:
            init = float(mid)
        init = float(init)
        init = max(lo + 1e-8, min(hi - 1e-8, init))

        z = (init - float(mid)) / float(span_half)
        z = max(-0.999999, min(0.999999, z))

        # raw0 = atanh(z) = 0.5*log((1+z)/(1-z))
        raw0 = 0.5 * torch.log(torch.tensor((1.0 + z) / (1.0 - z), device=dev, dtype=dtype))
        self.raw = nn.Parameter(raw0)

    def forward(self):
        return self.mid_buf + self.span_half_buf * torch.tanh(self.raw)


class UnifiedParams(nn.Module):
    """
    - r_col must NOT appear in specs.
    - m_q2 is dependent: m_q2 = m_q1 + eta*(hi - m_q1)
    - eta in [eps, 1-eps]: eta = eps + (1-2eps)*sigmoid(raw_eta)
    - values() returns detached tensors on module device.
    """
    def __init__(self, specs: dict, r_col: int, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.r_col = int(r_col)

        self.params = nn.ModuleDict()
        self._specs = specs
        self.eps = 1e-4

        learn_names, learn_cols = [], []
        fixed_cols, fixed_vals = [], []
        norm_cols, norm_lo, norm_span = [], [], []

        # ---- mq-pair detection ----
        self._has_mq_pair = ("m_q1" in specs) and ("m_q2" in specs)
        if self._has_mq_pair:
            self._idx_mq1  = int(specs["m_q1"]["col"])
            self._idx_m_q2 = int(specs["m_q2"]["col"])
            self._mq_hi = float(specs["m_q1"]["hi"])
            self.register_buffer("_mq_hi_t", torch.tensor(self._mq_hi, device=self.device, dtype=dtype))
        else:
            self._idx_mq1 = self._idx_m_q2 = None
            self._mq_hi = None
            self.register_buffer("_mq_hi_t", torch.tensor(0.0, device=self.device, dtype=dtype))  # unused

        # -----------------------------
        # Build standard TanhBox params
        # -----------------------------
        for name, s in specs.items():
            j = int(s["col"])
            if j == self.r_col:
                raise ValueError(
                    f"Spec '{name}' uses col={j} which equals r_col. "
                    "r_col must NOT appear in specs."
                )

            # m_q2 is dependent -> skip learnable/fixed creation (but keep normalization)
            if name == "m_q2":
                norm_cols.append(j)
                lo_n = float(s.get("lo_norm", s["lo"]))
                hi_n = float(s.get("hi_norm", s["hi"]))
                norm_lo.append(lo_n)
                norm_span.append(max(hi_n - lo_n, 1e-12))
                continue

            lo = float(s["lo"])
            hi = float(s["hi"])
            init = float(s["init"])  # make_specs always provides init

            lo_n = float(s.get("lo_norm", lo))
            hi_n = float(s.get("hi_norm", hi))
            norm_cols.append(j)
            norm_lo.append(lo_n)
            norm_span.append(max(hi_n - lo_n, 1e-12))

            if bool(s.get("learn", False)):
                self.params[name] = TanhBoxParam(lo, hi, init, device=self.device, dtype=dtype)
                learn_names.append(name)
                learn_cols.append(j)
            else:
                fixed_cols.append(j)
                fixed_vals.append(init)

        self.learn_names = learn_names

        # buffers
        self.register_buffer("learn_idx", torch.tensor(learn_cols, device=self.device, dtype=torch.long))
        self.register_buffer("fixed_idx", torch.tensor(fixed_cols, device=self.device, dtype=torch.long))
        self.register_buffer("fixed_vals", torch.tensor(fixed_vals, device=self.device, dtype=dtype))

        self.register_buffer("norm_cols", torch.tensor(norm_cols, device=self.device, dtype=torch.long))
        self.register_buffer("norm_lo", torch.tensor(norm_lo, device=self.device, dtype=dtype))
        self.register_buffer("norm_span", torch.tensor(norm_span, device=self.device, dtype=dtype))

        # -----------------------------
        # m_q2 eta parameter
        # -----------------------------
        if self._has_mq_pair:
            learn_eta = bool(specs["m_q2"].get("learn", False))

            init_m2 = float(specs["m_q2"]["init"])
            m1_0    = float(specs["m_q1"]["init"])
            hi      = float(self._mq_hi)

            # since you REQUIRE m_q2 >= m_q1:
            if init_m2 < m1_0:
                raise ValueError(f"Invalid init: m_q2.init ({init_m2}) < m_q1.init ({m1_0}).")

            # target eta so that m2(init) matches init_m2
            eta_target = (init_m2 - m1_0) / max(hi - m1_0, 1e-12)

            eps = float(self.eps)
            eta_target = float(max(eps, min(1.0 - eps, eta_target)))

            # invert eps-squash:
            denom = max(1e-12, 1.0 - 2.0 * eps)
            p = (eta_target - eps) / denom
            p = float(max(1e-6, min(1.0 - 1e-6, p)))
            raw0 = torch.log(torch.tensor(p / (1.0 - p), device=self.device, dtype=dtype))

            if learn_eta:
                self._mq2_eta_raw = nn.Parameter(raw0)
            else:
                self.register_buffer("_mq2_eta_raw", raw0)
        else:
            self._mq2_eta_raw = None  # explicit

    def apply_to_base(self, base_row: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        out = base_row.to(device=self.device, dtype=self.dtype).clone()

        # learnables
        if self.learn_idx.numel() > 0:
            vals = torch.stack([self.params[n]() for n in self.learn_names]).to(dtype=out.dtype)
            out.scatter_(0, self.learn_idx, vals)

        # fixed
        if self.fixed_idx.numel() > 0:
            out.scatter_(0, self.fixed_idx, self.fixed_vals.to(dtype=out.dtype))

        # dependent m_q2
        if self._has_mq_pair:
            m1 = out[self._idx_mq1]
            hi_m = self._mq_hi_t.to(dtype=out.dtype)
            eta = self.eps + (1.0 - 2.0 * self.eps) * torch.sigmoid(self._mq2_eta_raw).to(dtype=out.dtype)
            out[self._idx_m_q2] = m1 + eta * (hi_m - m1)

        if not normalize:
            return out

        # normalize
        outN = out.clone()
        cur = outN.index_select(0, self.norm_cols)
        curN = (cur - self.norm_lo.to(dtype=out.dtype)) / (self.norm_span.to(dtype=out.dtype))
        outN.scatter_(0, self.norm_cols, curN)
        return outN

    @torch.no_grad()
    def values(self) -> dict[str, torch.Tensor]:
        vals = {name: self.params[name]().detach() for name in self.learn_names}

        if self._has_mq_pair:
            # m_q1: learnable > fixed > init
            if "m_q1" in vals:
                m1 = vals["m_q1"]
            else:
                m1 = None
                if self.fixed_idx.numel() > 0:
                    mask = (self.fixed_idx == self._idx_mq1)
                    if bool(mask.any()):
                        i = mask.nonzero(as_tuple=True)[0][0]
                        m1 = self.fixed_vals[i].detach()
                if m1 is None:
                    m1 = torch.tensor(float(self._specs["m_q1"]["init"]), device=self.device, dtype=self.dtype)

            hi_m = self._mq_hi_t.to(dtype=m1.dtype)
            eta = self.eps + (1.0 - 2.0 * self.eps) * torch.sigmoid(self._mq2_eta_raw).to(dtype=m1.dtype)
            vals["m_q2"] = m1 + eta * (hi_m - m1)

        return vals




def _atanh_clamped(z: torch.Tensor) -> torch.Tensor:
    # z must be in (-1,1); clamp for numeric safety
    z = z.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(z) - torch.log1p(-z))


class UnifiedParamsBatched(nn.Module):
    """
    Batched version of UnifiedParams.

    External behavior:
      - r_col must NOT appear in specs.
      - m_q2 is dependent and always > m_q1:
            m_q2 = m_q1 + eta*(hi - m_q1),
            eta = eps + (1-2eps)*sigmoid(raw_eta)
      - apply_to_base(base_row, normalize) returns [R, P]
      - values() returns dict of tensors on module device, detached (each is [R] or scalar)
    """

    def __init__(
        self,
        specs: Dict[str, Dict[str, Any]],
        *,
        R: int,
        r_col: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.R = int(R)
        self.r_col = int(r_col)
        self.eps = float(eps)

        self._specs = specs

        # ---- mq pair ----
        self._has_mq_pair = ("m_q1" in specs) and ("m_q2" in specs)
        if self._has_mq_pair:
            self._idx_mq1 = int(specs["m_q1"]["col"])
            self._idx_mq2 = int(specs["m_q2"]["col"])
            self._mq_hi = float(specs["m_q1"]["hi"])
            self.register_buffer("_mq_hi_t", torch.tensor(self._mq_hi, device=self.device, dtype=dtype))
        else:
            self._idx_mq1 = self._idx_mq2 = -1
            self._mq_hi = None
            self.register_buffer("_mq_hi_t", torch.tensor(0.0, device=self.device, dtype=dtype))

        # ---- build column + normalization metadata ----
        learn_cols: List[int] = []
        learn_names: List[str] = []
        learn_mid: List[float] = []
        learn_span_half: List[float] = []
        learn_init: List[float] = []

        fixed_cols: List[int] = []
        fixed_vals: List[float] = []

        norm_cols: List[int] = []
        norm_lo: List[float] = []
        norm_span: List[float] = []

        for name, s in specs.items():
            j = int(s["col"])
            if j == self.r_col:
                raise ValueError(
                    f"Spec '{name}' uses col={j} which equals r_col. "
                    "r_col must NOT appear in specs."
                )

            # normalization always includes every param column in specs (including m_q2)
            lo = float(s["lo"])
            hi = float(s["hi"])
            lo_n = float(s.get("lo_norm", lo))
            hi_n = float(s.get("hi_norm", hi))
            norm_cols.append(j)
            norm_lo.append(lo_n)
            norm_span.append(max(hi_n - lo_n, 1e-12))

            # m_q2 is dependent -> do not create learn/fixed direct param
            if name == "m_q2":
                continue

            init = float(s["init"])  # you said make_specs guarantees init exists

            if bool(s.get("learn", False)):
                learn_names.append(name)
                learn_cols.append(j)
                mid = 0.5 * (lo + hi)
                span_half = 0.5 * (hi - lo)
                learn_mid.append(mid)
                learn_span_half.append(span_half)
                learn_init.append(init)
            else:
                fixed_cols.append(j)
                fixed_vals.append(init)

        self.learn_names = learn_names

        # buffers
        self.register_buffer("learn_idx", torch.tensor(learn_cols, device=self.device, dtype=torch.long))
        self.register_buffer("fixed_idx", torch.tensor(fixed_cols, device=self.device, dtype=torch.long))
        self.register_buffer("fixed_vals", torch.tensor(fixed_vals, device=self.device, dtype=dtype))

        self.register_buffer("norm_cols", torch.tensor(norm_cols, device=self.device, dtype=torch.long))
        self.register_buffer("norm_lo", torch.tensor(norm_lo, device=self.device, dtype=dtype))
        self.register_buffer("norm_span", torch.tensor(norm_span, device=self.device, dtype=dtype))

        # packed learnable transforms: [L]
        if len(learn_cols) > 0:
            mid_t = torch.tensor(learn_mid, device=self.device, dtype=dtype)
            span_t = torch.tensor(learn_span_half, device=self.device, dtype=dtype)
            self.register_buffer("learn_mid", mid_t)
            self.register_buffer("learn_span_half", span_t)

            # init raw via atanh((init-mid)/span)
            init_t = torch.tensor(learn_init, device=self.device, dtype=dtype)
            z = (init_t - mid_t) / span_t.clamp_min(1e-12)
            raw0 = _atanh_clamped(z)  # [L]
            # replicate across R
            raw0 = raw0.unsqueeze(0).expand(self.R, -1).contiguous()
            self.raw_learn = nn.Parameter(raw0)  # [R, L]
        else:
            self.register_buffer("learn_mid", torch.empty(0, device=self.device, dtype=dtype))
            self.register_buffer("learn_span_half", torch.empty(0, device=self.device, dtype=dtype))
            self.raw_learn = None

        # ---- eta for m_q2 (trainable or fixed), per restart: [R] ----
        if self._has_mq_pair:
            learn_eta = bool(specs["m_q2"].get("learn", False))

            init_m2 = float(specs["m_q2"]["init"])
            init_m1 = float(specs["m_q1"]["init"])
            hi = float(self._mq_hi)
            if init_m2 < init_m1:
                raise ValueError(f"Invalid init: m_q2.init ({init_m2}) < m_q1.init ({init_m1}).")

            # eta_target for init match
            denom = max(hi - init_m1, 1e-12)
            eta_target = (init_m2 - init_m1) / denom
            eta_target = float(max(self.eps, min(1.0 - self.eps, eta_target)))

            # invert eta = eps + (1-2eps)*sigmoid(raw)
            a = 1.0 - 2.0 * self.eps
            p = (eta_target - self.eps) / max(a, 1e-12)
            p = float(max(1e-6, min(1.0 - 1e-6, p)))
            raw_eta0 = math.log(p / (1.0 - p))

            raw_eta0 = torch.tensor(raw_eta0, device=self.device, dtype=dtype).expand(self.R).contiguous()
            if learn_eta:
                self.raw_eta = nn.Parameter(raw_eta0)  # [R]
            else:
                self.register_buffer("raw_eta", raw_eta0)
        else:
            self.raw_eta = None

    def apply_to_base(self, base_row: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # base_row: [P] -> out: [R, P]
        base = base_row.to(device=self.device, dtype=self.dtype).view(1, -1).expand(self.R, -1).clone()

        # fill learnables
        if self.raw_learn is not None and self.learn_idx.numel() > 0:
            vals = self.learn_mid + self.learn_span_half * torch.tanh(self.raw_learn)  # [R, L]
            idx = self.learn_idx.view(1, -1).expand(self.R, -1)
            base.scatter_(1, idx, vals.to(dtype=base.dtype))

        # fill fixed
        if self.fixed_idx.numel() > 0:
            idx = self.fixed_idx.view(1, -1).expand(self.R, -1)
            base.scatter_(1, idx, self.fixed_vals.view(1, -1).expand(self.R, -1).to(dtype=base.dtype))

        # dependent m_q2
        if self._has_mq_pair:
            m1 = base[:, self._idx_mq1]                                # [R]
            hi_m = self._mq_hi_t.to(dtype=base.dtype)                  # scalar
            eta = self.eps + (1.0 - 2.0 * self.eps) * torch.sigmoid(self.raw_eta).to(dtype=base.dtype)  # [R]
            base[:, self._idx_mq2] = m1 + eta * (hi_m - m1)

        if not normalize:
            return base

        cur = base.index_select(1, self.norm_cols)  # [R, C]
        curN = (cur - self.norm_lo.to(dtype=base.dtype)) / self.norm_span.to(dtype=base.dtype)
        outN = base.clone()
        outN.scatter_(1, self.norm_cols.view(1, -1).expand(self.R, -1), curN)
        return outN

    @torch.no_grad()
    def values(self) -> Dict[str, torch.Tensor]:
        # return physical values for learnable names + m_q2
        out: Dict[str, torch.Tensor] = {}

        if self.raw_learn is not None:
            vals = self.learn_mid + self.learn_span_half * torch.tanh(self.raw_learn)  # [R, L]
            for k, name in enumerate(self.learn_names):
                out[name] = vals[:, k].detach()

        if self._has_mq_pair:
            # need m_q1 value (learnable OR fixed OR init)
            if "m_q1" in out:
                m1 = out["m_q1"]
            else:
                # fixed or init
                m1 = torch.full((self.R,), float(self._specs["m_q1"]["init"]), device=self.device, dtype=self.dtype)
                # if fixed_idx contains idx_mq1, override
                if self.fixed_idx.numel() > 0:
                    mask = (self.fixed_idx == self._idx_mq1)
                    if bool(mask.any()):
                        i = int(mask.nonzero(as_tuple=True)[0][0].item())
                        m1 = torch.full((self.R,), float(self.fixed_vals[i].item()), device=self.device, dtype=self.dtype)

            hi_m = self._mq_hi_t.to(dtype=m1.dtype)
            eta = self.eps + (1.0 - 2.0 * self.eps) * torch.sigmoid(self.raw_eta).to(dtype=m1.dtype)
            out["m_q2"] = (m1 + eta * (hi_m - m1)).detach()

        return out
    
    @torch.no_grad()
    def set_init_from_phys(self, init_phys_learn: torch.Tensor, init_eta: torch.Tensor | None):
        """
        init_phys_learn: [R,L] physical values matching self.learn_names order
        init_eta: [R] or None
        """
        if self.raw_learn is not None:
            # z = (x-mid)/span
            z = (init_phys_learn.to(self.device, self.dtype) - self.learn_mid) / self.learn_span_half.clamp_min(1e-12)
            self.raw_learn.copy_(_atanh_clamped(z))

        if self._has_mq_pair and (init_eta is not None):
            # invert eta = eps + (1-2eps)*sigmoid(raw)
            eps = self.eps
            a = 1.0 - 2.0 * eps
            p = ((init_eta.to(self.device, self.dtype) - eps) / max(a, 1e-12)).clamp(1e-6, 1.0 - 1e-6)
            self.raw_eta.copy_(torch.log(p / (1.0 - p)))