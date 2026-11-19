// Test your own curve: the cellulose surrogate runs in the browser.
// A small residual network (site_precompute/surrogate_export.py) gives the three
// curve parts of a structure; the densities are combined exactly:
//   I = d1^2 S11 + d2^2 S22 + 2 d1 d2 S12,  d_i = rho_i - rho_fibril.
// A loaded curve is read locally, moved to the simulated curve's height (one
// shift in log intensity) and compared on 0.01-0.30 1/A.
(function () {
  const D = window.SURROGATE;
  const root = document.getElementById("try");
  const NQ = D.n_q, ND = D.n_dct, Q = D.q, LQ = Q.map(Math.log);
  const Q_LO = 0.01, Q_HI = 0.30;

  // ---- decode ---------------------------------------------------------------------
  function bytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }
  function f16(b64) {
    const u = bytes(b64), n = u.length / 2, out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = u[2 * i] | (u[2 * i + 1] << 8), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
      out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
    }
    return out;
  }
  const f32 = (b64) => new Float32Array(bytes(b64).buffer);
  const L = D.layers.map((l) => ({ rows: l.rows, cols: l.cols, w: f16(l.w), b: f16(l.b) }));
  const yMean = f32(D.y_mean), yStd = f32(D.y_std);
  const phi = new Float32Array(ND * NQ);                       // DCT-II rows 1..ND
  for (let k = 0; k < ND; k++) for (let j = 0; j < NQ; j++) phi[k * NQ + j] = Math.cos(Math.PI / NQ * (j + 0.5) * (k + 1)) * Math.sqrt(2 / NQ);

  // ---- network ----------------------------------------------------------------------
  const silu = (v) => v / (1 + Math.exp(-v));
  function dense(l, x) {
    const y = new Float32Array(l.rows);
    for (let r = 0; r < l.rows; r++) { let s = l.b[r]; const o = r * l.cols; for (let c = 0; c < l.cols; c++) s += l.w[o + c] * x[c]; y[r] = s; }
    return y;
  }
  // unit parameters (6, in [0, 1]) -> [a, b, c] on the q grid: log10 S11, log10 S22, correlation
  function parts(unit) {
    let h = dense(L[0], unit).map(silu);
    for (let i = 0; i < D.blocks; i++) {
      const t = dense(L[2 + 2 * i], dense(L[1 + 2 * i], h).map(silu));
      for (let j = 0; j < h.length; j++) h[j] = silu(t[j] + h[j]);
    }
    const z = dense(L[L.length - 1], h), out = [];
    for (let ch = 0; ch < 3; ch++) {
      const o = ch * (ND + 2), mu = z[o], s = Math.exp(z[o + 1]), y = new Float64Array(NQ);
      for (let j = 0; j < NQ; j++) {
        let v = 0; for (let k = 0; k < ND; k++) v += z[o + 2 + k] * phi[k * NQ + j];
        const val = (mu + s * v) * yStd[ch * NQ + j] + yMean[ch * NQ + j];
        y[j] = ch === 2 ? Math.tanh(val) : val;
      }
      out.push(y);
    }
    return out;
  }
  function logI(P, rho) {
    const d1 = rho[1] - rho[0], d2 = rho[2] - rho[0], y = new Float64Array(NQ);
    for (let j = 0; j < NQ; j++) {
      const s1 = 10 ** (0.5 * P[0][j]), s2 = 10 ** (0.5 * P[1][j]), c = P[2][j];
      const I = (d1 * s1 + c * d2 * s2) ** 2 + (1 - c * c) * (d2 * s2) ** 2;
      y[j] = Math.log10(Math.max(I, 1e-30));
    }
    return y;
  }

  // ---- parameters ---------------------------------------------------------------------
  const toUnit = (k, raw) => (D.log[k] ? (Math.log(raw) - Math.log(D.lo[k])) / (Math.log(D.hi[k]) - Math.log(D.lo[k])) : (raw - D.lo[k]) / (D.hi[k] - D.lo[k]));
  const K = Object.fromEntries(D.names.map((n, i) => [n, i]));
  // display sliders: [key, label, min, max, log, toRaw(display) -> [bank name, value], format, default]
  const P = [
    ["radius", "Fibril radius", 0.5, 2.5, true, (v) => ["tube_scaling", D.r_per_af_nm / v], (v) => `${v.toFixed(2)} nm`, 1.5, "fib"],
    ["spread", "Fibril size spread", 0, 0.6, false, (v) => ["size_cv", v], (v) => v.toFixed(2), 0.15, "fib"],
    ["sep", "Minimum separation", 0, 3, false, (v) => ["min_space", v / D.ref_px_nm], (v) => `${v.toFixed(1)} nm`, 0.4, "fib"],
    ["count", "Fibrils per area", 1000, 15000, true, (v) => ["no_rods_per_unit", v], null, 8000, "fib"],
    ["vf", "Matrix volume fraction", 0.05, 0.9, false, (v) => ["volfrac_grf", v], (v) => `${Math.round(v * 100)} %`, 0.4, "mat"],
    ["domain", "Matrix domain size", 16, 192, true, (v) => ["kappa_sc", 48 / v], (v) => `${Math.round(v)} nm`, 48, "mat"],
  ];
  const state = Object.fromEntries(P.map((p) => [p[0], p[7]]));
  state.rho = [0.8, 0.72, 0.64];
  function unitVector() {
    const u = new Float32Array(6);
    for (const p of P) { const [name, raw] = p[5](state[p[0]]); u[K[name]] = Math.min(Math.max(toUnit(K[name], raw), 0), 1); }
    return u;
  }
  // Fibrils actually placed per um^2 (the cellulose page's count map); the slider steps
  // the simulator's packing parameter.
  function perArea() {
    const CM = D.count_map, r = state.radius;
    const f = [Math.log(state.count * CM.r_per_af_nm / r), Math.log(1 - state.vf), Math.log(r), state.sep, state.spread, Math.log(48 / state.domain)];
    let s = 0;
    for (let t = 0; t < CM.coef.length; t++) { let m = CM.coef[t]; const p = CM.powers[t]; for (let j = 0; j < 6; j++) for (let e = 0; e < p[j]; e++) m *= f[j]; s += m; }
    return Math.exp(s) * CM.per_um2;
  }
  const fmtArea = (v) => `about ${(Math.round(v / 100) * 100).toLocaleString("en-US")} per µm²`;

  // ---- the loaded curve -----------------------------------------------------------------
  let user = null;            // { q: raw q, I, name, example }
  const unitSel = root.querySelector(".q-unit"), status = root.querySelector(".status");
  function parse(text) {
    const q = [], I = [];
    for (let line of text.split(/\r?\n/)) {
      line = line.trim();
      if (!/^[+\-.\d]/.test(line)) continue;                  // headers, comments
      if (line.includes(";")) line = line.replace(/,/g, ".");  // 0,12;3,4 style
      const v = line.split(/[\s,;]+/).map(Number);
      if (v.length < 2 || !isFinite(v[0]) || !isFinite(v[1]) || v[0] <= 0 || v[1] <= 0) continue;
      q.push(v[0]); I.push(v[1]);
    }
    return { q, I };
  }
  function setUser(c, name, example) {
    if (c.q.length < 5) { status.textContent = `Could not read a curve from ${name}: need at least 5 lines with q and a positive intensity.`; return; }
    const qmax = Math.max(...c.q);
    unitSel.value = qmax > 1.5 ? "nm" : "A";                  // a guess; the reader can change it
    user = { q: c.q, I: c.I, name, example };
    root.querySelector(".truth").hidden = !example;
    draw();
  }
  root.querySelector(".load").addEventListener("click", () => root.querySelector(".file").click());
  root.querySelector(".file").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    f.text().then((t) => setUser(parse(t), f.name, false));
    e.target.value = "";
  });
  root.querySelector(".example").addEventListener("click", () => setUser({ q: D.example.q, I: D.example.I }, "the example", true));
  root.querySelector(".clear").addEventListener("click", () => { user = null; root.querySelector(".truth").hidden = true; draw(); });
  unitSel.addEventListener("change", () => draw());
  root.querySelector(".truth").addEventListener("click", () => {
    const t = D.example.truth;
    const set = { radius: t.radius_nm, spread: t.size_cv, sep: t.min_space_nm, count: t.no_rods_per_unit, vf: t.volfrac_grf, domain: 48 / t.kappa_sc };
    Object.assign(state, set); state.rho = [...t.rho];
    syncSliders(); draw();
  });

  // ---- comparison options ---------------------------------------------------------------
  // View: log I against log q (default) or Kratky, I q^2 against q.
  // Normalisation: one scale factor fitted over the window (least squares in log I, the
  // usual choice for curves in arbitrary units), the same invariant (integral of q^2 I
  // over the window), or both curves equal at one q.
  // Window: the q range used for the height and for the difference.
  const opt = { view: "log", norm: "fit", qref: 0.05, qlo: Q_LO, qhi: Q_HI };
  const el = (s) => root.querySelector(s);
  function readOptions() {
    opt.view = el(".view").value; opt.norm = el(".norm").value;
    const num = (s, d) => { const v = parseFloat(el(s).value.replace(",", ".")); return isFinite(v) ? v : d; };   // 0,1 or 0.1
    opt.qlo = Math.min(Math.max(num(".qlo", Q_LO), Q_LO), Q_HI);
    opt.qhi = Math.min(Math.max(num(".qhi", Q_HI), Q_LO), Q_HI);
    if (opt.qhi - opt.qlo < 0.005) opt.qhi = Math.min(Q_HI, opt.qlo + 0.005), opt.qlo = Math.min(opt.qlo, opt.qhi - 0.005);
    opt.qref = Math.min(Math.max(num(".qref", 0.05), opt.qlo), opt.qhi);
    el(".qref").disabled = opt.norm !== "point";
  }
  for (const s of [".view", ".norm", ".qlo", ".qhi", ".qref"]) el(s).addEventListener("input", () => { readOptions(); draw(); });
  for (const s of [".qlo", ".qhi", ".qref"]) el(s).addEventListener("change", () => {          // show the clamped value
    el(".qlo").value = opt.qlo; el(".qhi").value = opt.qhi; el(".qref").value = +opt.qref.toFixed(3);
  });
  const simAt = (sim, q) => {                                  // simulated log I at q, linear in log q
    const lq = Math.log(q); let j = 1; while (j < NQ - 1 && LQ[j] < lq) j++;
    const t = Math.min(Math.max((lq - LQ[j - 1]) / (LQ[j] - LQ[j - 1]), 0), 1);
    return sim[j - 1] * (1 - t) + sim[j] * t;
  };

  // All loaded points inside the simulated range; the height and the difference
  // use only those inside the window.
  function compare(sim) {
    if (!user) return null;
    const f = unitSel.value === "nm" ? 0.1 : 1, pts = [];
    for (let i = 0; i < user.q.length; i++) {
      const qa = user.q[i] * f;
      if (qa < Q_LO || qa > Q_HI) continue;
      pts.push({ q: qa, y: Math.log10(user.I[i]), s: simAt(sim, qa), win: qa >= opt.qlo && qa <= opt.qhi });
    }
    pts.sort((a, b) => a.q - b.q);
    const win = pts.filter((p) => p.win);
    if (win.length < 5) return { pts: [], n: win.length, nAll: pts.length };
    let shift;
    if (opt.norm === "point") {                                // interpolate the loaded curve at qref
      let k = win.findIndex((p) => p.q >= opt.qref);
      if (k <= 0) k = k === 0 ? 1 : win.length - 1;
      const a = win[k - 1], b = win[k], t = Math.min(Math.max(Math.log(opt.qref / a.q) / Math.log(b.q / a.q), 0), 1);
      shift = simAt(sim, opt.qref) - (a.y + (b.y - a.y) * t);
    } else if (opt.norm === "invariant") {
      // same integral of q^2 I over the window; both integrated over the loaded curve's
      // points (trapezoid in q, the simulated curve interpolated there), so the sampling is the same
      let Qs = 0, Qu = 0;
      for (let i = 1; i < win.length; i++) {
        const a = win[i - 1], b = win[i], dq = b.q - a.q;
        Qs += 0.5 * dq * (a.q * a.q * 10 ** a.s + b.q * b.q * 10 ** b.s);
        Qu += 0.5 * dq * (a.q * a.q * 10 ** a.y + b.q * b.q * 10 ** b.y);
      }
      shift = Math.log10(Qs / Qu);
    } else {
      shift = win.reduce((s, p) => s + p.s - p.y, 0) / win.length;
    }
    let ss = 0;
    for (const p of pts) { p.y += shift; p.d = p.y - p.s; if (p.win) ss += p.d * p.d; }
    return { pts, n: win.length, nAll: pts.length, rms: Math.sqrt(ss / win.length) };
  }

  // ---- drawing --------------------------------------------------------------------------
  const curveCanvas = root.querySelector("canvas.curve"), diffCanvas = root.querySelector("canvas.diff");
  const M = { l: 58, r: 14 };
  const X = (q, w) => opt.view === "kratky"
    ? M.l + q / Q_HI * (w - M.l - M.r)
    : M.l + (Math.log10(q) - Math.log10(Q_LO)) / (Math.log10(Q_HI) - Math.log10(Q_LO)) * (w - M.l - M.r);
  function xTicks(ctx, w, top, bottom) {
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.lineWidth = 1; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const ticks = opt.view === "kratky" ? [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] : [0.01, 0.02, 0.05, 0.1, 0.2, 0.3];
    for (const t of ticks) { ctx.beginPath(); ctx.moveTo(X(t, w), top); ctx.lineTo(X(t, w), bottom); ctx.stroke(); ctx.fillText(String(t), X(t, w), bottom + 6); }
  }
  function shadeOutside(ctx, w, top, bottom) {                 // the q outside the window
    ctx.fillStyle = "rgba(95,103,112,0.07)";
    const x0 = opt.view === "kratky" ? X(0, w) : X(Q_LO, w), xa = X(opt.qlo, w), xb = X(opt.qhi, w), x1 = X(Q_HI, w);
    if (xa > x0) ctx.fillRect(x0, top, xa - x0, bottom - top);
    if (x1 > xb) ctx.fillRect(xb, top, x1 - xb, bottom - top);
  }
  function yLabel(ctx, text, top, bottom) {
    ctx.save(); ctx.translate(12, (top + bottom) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = C.muted;
    ctx.fillText(text, 0, 0); ctx.restore();
  }
  function niceStep(span) { const r = span / 5, p = 10 ** Math.floor(Math.log10(r)), m = r / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : 5) * p; }
  function drawCurve(sim, cmp) {
    const { ctx, w, h } = fitCanvas(curveCanvas), top = 12, bottom = h - 22;
    const kr = opt.view === "kratky";
    // Kratky: I q^2 relative to the simulated curve's maximum inside the window
    let norm = 1;
    if (kr) { norm = 0; Q.forEach((q, j) => { if (q >= opt.qlo && q <= opt.qhi) norm = Math.max(norm, 10 ** sim[j] * q * q); }); }
    const val = (q, ly) => (kr ? 10 ** ly * q * q / norm : ly);
    const sv = Array.from(sim, (v, j) => val(Q[j], v));
    const pv = cmp && cmp.pts.length ? cmp.pts.map((p) => val(p.q, p.y)) : [];
    let lo, hi;
    if (kr) { lo = 0; hi = 1.15 * Math.max(1, ...pv.filter((v, i) => cmp.pts[i].win)); }
    else { lo = Math.min(...sv, ...pv) - 0.15; hi = Math.max(...sv, ...pv) + 0.15; }
    const Y = (v) => bottom - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) * (bottom - top);
    shadeOutside(ctx, w, top, bottom);
    xTicks(ctx, w, top, bottom);
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const step = kr ? niceStep(hi - lo) : hi - lo > 6 ? 2 : hi - lo > 2.5 ? 1 : 0.5;
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
      ctx.beginPath(); ctx.moveTo(M.l, Y(t)); ctx.lineTo(w - M.r, Y(t)); ctx.stroke(); ctx.fillText(String(+t.toFixed(2)), M.l - 6, Y(t));
    }
    yLabel(ctx, kr ? "I·q² (relative)" : "log₁₀ intensity", top, bottom);
    if (pv.length) {
      cmp.pts.forEach((p, i) => {
        ctx.fillStyle = p.win ? C.obs : "#cfd5db";
        ctx.beginPath(); ctx.arc(X(p.q, w), Y(pv[i]), 2.1, 0, 2 * Math.PI); ctx.fill();
      });
    }
    ctx.strokeStyle = C.navy; ctx.lineWidth = 2.2; ctx.beginPath();
    sv.forEach((v, j) => (j ? ctx.lineTo(X(Q[j], w), Y(v)) : ctx.moveTo(X(Q[j], w), Y(v))));
    ctx.stroke();
    if (opt.norm === "point" && cmp && cmp.pts.length) {      // mark the matching point
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(X(opt.qref, w), top); ctx.lineTo(X(opt.qref, w), bottom); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  function drawDiff(cmp) {
    const { ctx, w, h } = fitCanvas(diffCanvas), top = 8, bottom = h - 34;
    // range adapts to the largest difference inside the window: 0.1, 0.2, 0.5 or 1 decade
    const inWin = cmp && cmp.pts.length ? cmp.pts.filter((p) => p.win) : [];
    const big = inWin.length ? Math.max(...inWin.map((p) => Math.abs(p.d))) : 0;
    const R = [0.1, 0.2, 0.5, 1].find((r) => big <= 0.95 * r) || 1;
    const Y = (v) => top + (R - Math.max(-R, Math.min(R, v))) / (2 * R) * (bottom - top);
    shadeOutside(ctx, w, top, bottom);
    xTicks(ctx, w, top, bottom);
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillStyle = C.muted;
    for (const t of [-R / 2, 0, R / 2]) {
      ctx.strokeStyle = t === 0 ? C.muted : C.grid; ctx.beginPath(); ctx.moveTo(M.l, Y(t)); ctx.lineTo(w - M.r, Y(t)); ctx.stroke();
      ctx.fillText(String(+t.toFixed(2)), M.l - 6, Y(t));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"; ctx.fillText("q (Å⁻¹)", (M.l + w - M.r) / 2, h - 4);
    yLabel(ctx, "difference", top, bottom);
    if (!cmp || !cmp.pts.length) return;
    for (const p of cmp.pts) {
      ctx.fillStyle = p.win ? C.accent : "#f2c2a6";
      ctx.beginPath(); ctx.arc(X(p.q, w), Y(p.d), 1.8, 0, 2 * Math.PI); ctx.fill();
    }
  }
  function showStatus(cmp) {
    root.querySelector(".legend-user").hidden = !user;
    if (!user) { status.textContent = "No curve loaded. The plot shows the simulated curve only."; return; }
    if (!cmp.pts.length) {
      status.textContent = cmp.nAll >= 5 ? `${user.name}: only ${cmp.n} points inside the window. Widen the window.`
        : `${user.name}: only ${cmp.nAll} points between 0.01 and 0.30 Å⁻¹. Check the unit of q.`;
      return;
    }
    const pct = Math.round((10 ** cmp.rms - 1) * 100);
    status.innerHTML = `Difference: <b>${cmp.rms.toFixed(3)} decades</b> (rms, about ${pct} % in intensity) over ${cmp.n} points of ${user.name}, ${+opt.qlo.toFixed(3)}&ndash;${+opt.qhi.toFixed(3)}&nbsp;Å⁻¹.`;
  }
  const draw = throttled(() => {
    const sim = logI(parts(unitVector()), state.rho);
    const cmp = compare(sim);
    drawCurve(sim, cmp); drawDiff(cmp); showStatus(cmp);
    if (showArea) showArea();
  });

  // ---- controls -------------------------------------------------------------------------
  const STEPS = 1000, inputs = {};
  let showArea = null;
  function addSlider(parent, label, toPos, fromPos, format, key) {
    const row = document.createElement("label");
    row.className = "ctrl";
    row.innerHTML = `<span class="ctrl-name">${label}</span><output></output><input type="range" min="0" max="${STEPS}" step="1">`;
    const input = row.querySelector("input"), out = row.querySelector("output");
    const show = () => { out.textContent = format(); };
    input.value = toPos();
    input.addEventListener("input", () => { fromPos(Number(input.value)); show(); draw(); });
    show(); parent.append(row);
    inputs[key] = { input, toPos, show };
    return show;
  }
  const groups = { fib: root.querySelector(".fibrils"), mat: root.querySelector(".matrix") };
  for (const [key, label, lo, hi, log, , fmt, , g] of P) {
    const toPos = () => Math.round(STEPS * (log ? Math.log(state[key] / lo) / Math.log(hi / lo) : (state[key] - lo) / (hi - lo)));
    const fromPos = (v) => { const u = v / STEPS; state[key] = log ? lo * (hi / lo) ** u : lo + u * (hi - lo); };
    const show = addSlider(groups[g], label, toPos, fromPos, fmt ? () => fmt(state[key]) : () => fmtArea(perArea()), key);
    if (!fmt) showArea = show;
  }
  ["Fibril density", "Matrix density", "Interstitial density"].forEach((label, k) => {
    addSlider(root.querySelector(".densities"), label, () => Math.round((state.rho[k] - 0.5) / 0.5 * STEPS),
      (v) => { state.rho[k] = 0.5 + 0.5 * v / STEPS; }, () => state.rho[k].toFixed(2), `rho${k}`);
  });
  function syncSliders() { for (const s of Object.values(inputs)) { s.input.value = s.toPos(); s.show(); } }

  // ---- self-check against the Python export (console only) --------------------------------
  {
    let worst = 0;
    D.test.unit.forEach((u, i) => {
      const y = logI(parts(Float32Array.from(u)), D.test.rho[i]);
      for (let j = 0; j < NQ; j++) worst = Math.max(worst, Math.abs(y[j] - D.test.logI[i][j]));
    });
    root.dataset.check = worst.toFixed(5);
    console.log(`surrogate JS vs Python: max difference ${worst.toFixed(5)} decades`);
  }

  readOptions();
  window.addEventListener("resize", draw);
  setUser({ q: D.example.q, I: D.example.I }, "the example", true);    // the page starts with the example
  draw();
})();
