// LNP: from curve to material. A thousand multi-start fits of one simulated curve were
// run with the released surrogate and fitter (site_precompute/lnp_fit.py); the
// near-optimal ones are kept. Click a fit to see its curve and its particle.
(function () {
  const D = window.LNP_DATA;
  const root = document.getElementById("infer");
  const NQ = D.q.length, NR = D.loss.length, SHOW = D.show, NP = SHOW.length;
  const curves = decodeU16(D.curves_b64);
  const col = Object.fromEntries(D.names.map((k, i) => [k, i]));
  const state = { sel: D.best, truth: false, pair: [4, 3] };     // interior density vs shell density
  const kept = [], rest = [];
  D.keep.forEach((k, i) => (k ? kept : rest).push(i));

  const curve = (i) => { const o = i * NQ, y = new Float64Array(NQ); for (let j = 0; j < NQ; j++) y[j] = D.log_lo + curves[o + j] / 65535 * (D.log_hi - D.log_lo); return y; };
  const pos = (k, v) => (v - SHOW[k].lo) / (SHOW[k].hi - SHOW[k].lo);          // display value -> unit position
  const val = (k, i) => D.values[i][k];
  const fmtv = (k, v) => `${SHOW[k].key === "cv" ? Math.round(v) : Number(v.toPrecision(3))} ${SHOW[k].unit}`;
  const lossRatio = (i) => D.loss[i] / D.loss[D.best];

  // ---- particle pictures from the fitted parameters ----------------------------------
  const WIDTH = 1000, DX = WIDTH / LNP.N;                       // 100 nm wide cross-sections
  function particle(canvas, p) {
    const spec = { w1: p.w_q1, m1: p.m_q1, m2: p.m_q2, s1: p.s_q1, s2: p.s_q2 };
    const fld = LNP.field(spec, p.GRF_shift, DX);
    LNP.draw(canvas, WIDTH, [{ x: WIDTH / 2, y: WIDTH / 2, R: p.r_mean, d: p.shell_thickness, rhoShell: p.relative_rho_shell * (1 + p.GRF_shift) }], fld, DX);
    LNP.scaleBar(canvas, WIDTH, 200, "20 nm");
  }
  const fitParams = (i) => {
    const p = Object.fromEntries(D.names.map((k) => [k, D.params[i][col[k]]]));
    p.r_mean = p.radius_mean_A; p.cv = p.radius_cv;
    return p;
  };

  // ---- the pair: truth and the most different kept fit ------------------------------
  function drawPair() {
    particle(root.querySelector(".pic-a"), D.truth);
    particle(root.querySelector(".pic-b"), fitParams(D.alt));
    const { ctx, w, h } = fitCanvas(root.querySelector(".alt-curve"));
    const { X, Y } = curveAxes(ctx, w, h, D.y_true);
    line(ctx, X, Y, D.q, D.y_true, C.navy, 2.5);
    line(ctx, X, Y, D.q, curve(D.alt), C.accent, 1.5);
    const t = D.truth_values, a = D.values[D.alt];
    root.querySelector(".pair-note").textContent =
      `Simulated cross-sections, 100 nm wide, both drawn from the same random interior. Left, the particle behind the curve: ` +
      `mean radius ${t[0].toFixed(1)} nm, shell ${t[2].toFixed(1)} nm at ${t[3].toFixed(2)} times the interior density, interior at ${t[4].toFixed(2)} times the solvent. ` +
      `Right, one of the kept fits: mean radius ${a[0].toFixed(1)} nm, shell ${a[2].toFixed(1)} nm at ${a[3].toFixed(2)} times the interior, interior at ${a[4].toFixed(2)} times the solvent. ` +
      `Their curves are nearly the same.`;
  }

  // ---- curve panels -----------------------------------------------------------------
  function curveAxes(ctx, w, h, ys) {
    const m = { l: 50, r: 12, t: 10, b: 40 };
    const q0 = Math.log10(D.q[0]), q1 = Math.log10(D.q[NQ - 1]);
    let y0 = Infinity, y1 = -Infinity;
    for (const v of ys) { y0 = Math.min(y0, v); y1 = Math.max(y1, v); }
    y0 -= 0.15; y1 += 0.15;
    const X = (q) => m.l + (Math.log10(q) - q0) / (q1 - q0) * (w - m.l - m.r);
    const Y = (y) => h - m.b - (y - y0) / (y1 - y0) * (h - m.t - m.b);
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.lineWidth = 1; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of [0.003, 0.01, 0.03, 0.1]) {
      ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), h - m.b); ctx.stroke(); ctx.fillText(String(t), X(t), h - m.b + 6);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const step = y1 - y0 > 6 ? 2 : 1;
    for (let t = Math.ceil(y0); t <= y1; t++) {
      if (t % step) continue;
      ctx.beginPath(); ctx.moveTo(m.l, Y(t)); ctx.lineTo(w - m.r, Y(t)); ctx.stroke(); ctx.fillText(String(t), m.l - 6, Y(t));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillText("q (Å⁻¹)", (m.l + w - m.r) / 2, h - 6);
    ctx.save(); ctx.translate(12, (m.t + h - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle";
    ctx.fillText("log₁₀ intensity, scaled", 0, 0); ctx.restore();
    return { X, Y };
  }
  function line(ctx, X, Y, q, ys, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    ys.forEach((y, i) => (i ? ctx.lineTo(X(q[i]), Y(y)) : ctx.moveTo(X(q[i]), Y(y))));
    ctx.stroke();
  }
  function drawCurve() {
    const { ctx, w, h } = fitCanvas(root.querySelector(".curve"));
    const { X, Y } = curveAxes(ctx, w, h, D.y_obs);
    for (const i of kept) line(ctx, X, Y, D.q, curve(i), "rgba(31, 78, 121, 0.22)", 1.1);
    line(ctx, X, Y, D.q, curve(state.sel), C.accent, 2);
    ctx.fillStyle = C.obs;
    D.y_obs.forEach((y, i) => { ctx.beginPath(); ctx.arc(X(D.q[i]), Y(y), 1.5, 0, 7); ctx.fill(); });
  }

  // ---- joint panel ------------------------------------------------------------------
  const jointCanvas = root.querySelector(".joint");
  const JM = { l: 76, r: 12, t: 10, b: 52 };
  let jointGeom = null;
  function drawJoint() {
    const { ctx, w, h } = fitCanvas(jointCanvas);
    const [a, b] = state.pair, m = JM, W = w - m.l - m.r, H = h - m.t - m.b;
    const X = (d) => m.l + d * W, Y = (d) => h - m.b - d * H;
    jointGeom = { X, Y, w, h };
    ctx.fillStyle = "#f4f6f8"; ctx.fillRect(m.l, m.t, W, H);
    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const [d, label] of SHOW[a].ticks) ctx.fillText(label, X(d), h - m.b + 6);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const [d, label] of SHOW[b].ticks) ctx.fillText(label, m.l - 6, Y(d));
    ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, W, H); ctx.clip();
    const dot = (i, color, r) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(pos(a, val(a, i))), Y(pos(b, val(b, i))), r, 0, 7); ctx.fill(); };
    for (const i of rest) dot(i, C.free, 1.6);
    for (const i of kept) dot(i, C.navy, 2.2);
    const sx = X(pos(a, val(a, state.sel))), sy = Y(pos(b, val(b, state.sel)));
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx, sy, 5.5, 0, 7); ctx.stroke();
    if (state.truth) {
      const x = X(pos(a, D.truth_values[a])), y = Y(pos(b, D.truth_values[b]));
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x - 7, y); ctx.lineTo(x + 7, y); ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7); ctx.stroke();
    }
    ctx.restore();
  }
  jointCanvas.addEventListener("pointerdown", (e) => {
    if (!jointGeom) return;
    const r = jointCanvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, [a, b] = state.pair;
    let best = -1, bd = 14 * 14;
    for (let i = 0; i < NR; i++) {
      const d = (jointGeom.X(pos(a, val(a, i))) - x) ** 2 + (jointGeom.Y(pos(b, val(b, i))) - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0) select(best);
  });

  // ---- parameter rows ------------------------------------------------------------------
  const rows = [];
  const table = root.querySelector(".params"), RM = 10;
  const hover = { k: null, d: 0 };
  SHOW.forEach((s, k) => {
    const row = document.createElement("div");
    row.className = "param";
    row.innerHTML = `<div class="name">${s.label}<small>${s.unit}</small></div><canvas></canvas><span></span>`;
    const canvas = row.querySelector("canvas");
    const dpos = (e) => { const r = canvas.getBoundingClientRect(); return (e.clientX - r.left - RM) / (r.width - 2 * RM); };
    canvas.addEventListener("pointerdown", (e) => {                    // the kept fit nearest this value
      const v = SHOW[k].lo + dpos(e) * (SHOW[k].hi - SHOW[k].lo);
      let best = kept[0], bd = Infinity;
      for (const i of kept) { const d = Math.abs(val(k, i) - v); if (d < bd) { bd = d; best = i; } }
      select(best);
    });
    canvas.addEventListener("pointermove", (e) => { hover.k = k; hover.d = dpos(e); drawRow(k); });
    canvas.addEventListener("pointerleave", () => { if (hover.k === k) { hover.k = null; drawRow(k); } });
    table.append(row);
    rows.push({ row, canvas });
  });
  function strip(ctx, X, top, base, values, color) {
    const dens = density(values), n = dens.length;
    ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(X(0), base);
    for (let i = 0; i < n; i++) ctx.lineTo(X((i + 0.5) / n), base - dens[i] * (base - top));
    ctx.lineTo(X(1), base); ctx.closePath(); ctx.fill();
  }
  function drawRow(k) {
    const { canvas } = rows[k];
    const { ctx, w, h } = fitCanvas(canvas);
    const top = 6, base = h - 18, width = w - 2 * RM, X = (d) => RM + d * width;
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.textBaseline = "top";
    for (const [d, label] of SHOW[k].ticks) {
      ctx.beginPath(); ctx.moveTo(X(d), base); ctx.lineTo(X(d), base + 4); ctx.stroke();
      ctx.textAlign = d < 0.02 ? "left" : d > 0.98 ? "right" : "center"; ctx.fillText(label, X(d), base + 6);
    }
    ctx.beginPath(); ctx.moveTo(RM, base + 0.5); ctx.lineTo(w - RM, base + 0.5); ctx.stroke();
    // other restarts as faint marks, kept fits as the strip
    ctx.fillStyle = "rgba(184, 192, 200, 0.5)";
    for (const i of rest) { const x = X(Math.min(1, Math.max(0, pos(k, val(k, i))))); ctx.fillRect(x - 0.5, base - 8, 1, 8); }
    strip(ctx, X, top, base, kept.map((i) => Math.min(1, Math.max(0, pos(k, val(k, i))))), C.navy);
    const xs = X(pos(k, val(k, state.sel)));
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xs, top); ctx.lineTo(xs, base); ctx.stroke();
    ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(xs, base, 5, 0, 7); ctx.fill(); ctx.lineWidth = 1;
    if (hover.k === k && hover.d >= 0 && hover.d <= 1) {
      const x = X(hover.d);
      ctx.strokeStyle = C.muted; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, base); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.muted; ctx.textBaseline = "top"; ctx.textAlign = hover.d > 0.5 ? "right" : "left";
      ctx.fillText(fmtv(k, SHOW[k].lo + hover.d * (SHOW[k].hi - SHOW[k].lo)), x + (hover.d > 0.5 ? -6 : 6), top);
    }
    if (state.truth) {
      const x = X(pos(k, D.truth_values[k]));
      ctx.fillStyle = C.accent; ctx.beginPath(); ctx.moveTo(x, base - 7); ctx.lineTo(x - 5, base + 1); ctx.lineTo(x + 5, base + 1); ctx.closePath(); ctx.fill();
    }
  }

  // ---- selected fit ---------------------------------------------------------------------
  function drawFit() {
    const i = state.sel, r = lossRatio(i);
    root.querySelector(".fit-label").textContent =
      (i === D.best ? "Best fit" : `Fit ${i + 1} of ${NR}`) + (D.keep[i] ? "" : ", not kept") + (i === D.best ? "" : ` · loss ${r.toFixed(2)} × best`);
    particle(root.querySelector(".pic-fit"), fitParams(i));
  }
  function select(i) { state.sel = i; draw(); }

  // ---- controls -------------------------------------------------------------------------
  const truthBox = root.querySelector(".show-truth");
  truthBox.addEventListener("change", () => { state.truth = truthBox.checked; draw(); });
  const selects = [root.querySelector(".pair-x"), root.querySelector(".pair-y")];
  selects.forEach((sel, i) => {
    SHOW.forEach((s, k) => sel.add(new Option(s.label, k)));
    sel.value = state.pair[i];
    sel.addEventListener("change", () => { state.pair[i] = Number(sel.value); drawJoint(); });
  });
  const ki = D.keep_info;
  root.querySelector(".fits-note").textContent =
    `${NR} starting points, ${D.settings.steps} gradient steps each, ${D.n_kept} fits kept: those whose loss is within ` +
    `${Math.round((ki.alpha - 1) * 100)} % of the mean loss of the best tenth. Tracks show the kept fits; the short marks are the other restarts. ` +
    `The kept fits agree on the size distribution and the interior length scale. On the densities they split in two families: shell and interior both lighter than the solvent, ` +
    `as in the material behind the curve, or both denser. Scattering sees density differences squared, so the sign is lost. The shell thickness trades against the densities.`;

  const draw = throttled(() => { drawCurve(); drawJoint(); SHOW.forEach((s, k) => drawRow(k)); drawFit(); });
  window.addEventListener("resize", () => { drawPair(); draw(); });
  drawPair();
  draw();
})();
