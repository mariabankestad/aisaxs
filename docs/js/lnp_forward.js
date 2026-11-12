// LNP: from material to curve. Curves were precomputed on a parameter grid with the
// released surrogate and the fitter's polydispersity quadrature (site_precompute/
// lnp_forward.py); the particle pictures are drawn here from the structure model.
(function () {
  const D = window.LNP_FORWARD;
  const root = document.getElementById("forward");
  const NQ = D.q.length;
  const n = Object.fromEntries(D.axes.map((a) => [a, D.grid[a].length]));
  const curves = decodeU16(D.curves_b64);
  const SIZE = [["r_mean", "Mean radius", (v) => `${(v / 10).toFixed(1)} nm`],
                ["cv", "Size spread", (v) => `${Math.round(v * 100)} %`]];
  const SHELL = [["shell", "Shell thickness", (v) => `${(v / 10).toFixed(1)} nm`],
                 ["rho_shell", "Shell density", (v) => `${v.toFixed(2)} × interior`]];
  const INTERIOR = [["shift", "Interior density", (v) => `${(1 + v).toFixed(2)} × solvent`],
                    ["m_q1", "Interior length scale", (v) => `${(2 * Math.PI / v / 10).toFixed(1)} nm`],
                    ["s_q1", "Length-scale spread", (v, i) => ["narrow", "medium", "wide"][i]]];

  const index = (s) => { let o = 0; for (const a of D.axes) o = o * n[a] + s[a]; return o; };
  function curve(s) {
    const o = index(s) * NQ, y = new Float64Array(NQ);
    for (let i = 0; i < NQ; i++) y[i] = D.log_lo + curves[o + i] / 65535 * (D.log_hi - D.log_lo);
    return y;
  }
  const phys = (s) => Object.fromEntries(D.axes.map((a) => [a, D.grid[a][s[a]]]));

  // ---- state and tweening -----------------------------------------------------------
  const state = Object.fromEntries(D.axes.map((a) => [a, D.grid[a].indexOf(D.default[a])]));
  let kept = null;
  const shown = { from: null, to: null, t: 1, start: 0 };
  const DUR = 350;
  const snapshot = (s) => ({ y: curve(s), p: phys(s) });
  function target(s) {
    const next = snapshot(s);
    if (!shown.to) { shown.from = shown.to = next; shown.t = 1; return; }
    shown.from = current(); shown.to = next; shown.t = 0; shown.start = performance.now();
    requestAnimationFrame(tick);
  }
  function current() {
    if (shown.t >= 1) return shown.to;
    const t = shown.t, a = shown.from, b = shown.to;
    const y = a.y.map((v, i) => v + (b.y[i] - v) * t);
    const p = Object.fromEntries(D.axes.map((k) => [k, a.p[k] + (b.p[k] - a.p[k]) * t]));
    return { y, p, blend: { a: a.p, b: b.p, t } };
  }
  function tick(now) {
    const u = Math.min(1, (now - shown.start) / DUR);
    shown.t = u * u * (3 - 2 * u);
    draw();
    if (u < 1) requestAnimationFrame(tick); else { shown.t = 1; shown.from = shown.to; }
  }

  // ---- drawing ---------------------------------------------------------------------
  const picCanvas = root.querySelector(".material"), curveCanvas = root.querySelector(".curve");
  const VIEWS = { close: { width: 1000, bar: 200, label: "Cross-section, 100 nm wide" },
                  wide: { width: 3000, bar: 500, label: "Suspension, 300 nm wide" } };
  let view = "close";
  const spectrum = (p) => ({ w1: D.w_q1, m1: p.m_q1, m2: Math.min(p.m_q1 + D.dm_q2, 0.15), s1: p.s_q1, s2: D.s_q2 });
  const blendBuf = new Float32Array(LNP.N * LNP.N);
  // While tweening, the two end fields are blended pixel by pixel; the ends are exact.
  function fieldFor(cur, dx) {
    if (!cur.blend) return LNP.field(spectrum(cur.p), cur.p.shift, dx);
    const { a, b, t } = cur.blend, fa = LNP.field(spectrum(a), 0, dx).f, fb = LNP.field(spectrum(b), 0, dx).f;
    if (fa === fb) return { f: fa, shift: cur.p.shift };
    for (let i = 0; i < blendBuf.length; i++) blendBuf[i] = fa[i] + (fb[i] - fa[i]) * t;
    return { f: blendBuf, shift: cur.p.shift };
  }
  function drawMaterial(cur) {
    const V = VIEWS[view], dx = V.width / LNP.N, p = cur.p;
    const rhoShell = p.rho_shell * (1 + p.shift);
    const parts = view === "close"
      ? [{ x: V.width / 2, y: V.width / 2, R: p.r_mean, d: p.shell, rhoShell }]
      : LNP.population(V.width, p.r_mean, p.cv, p.shell, rhoShell, D.trunc[0], D.trunc[1]);
    LNP.draw(picCanvas, V.width, parts, fieldFor(cur, dx), dx);
    LNP.scaleBar(picCanvas, V.width, V.bar, `${V.bar / 10} nm`);
    if (view === "close") sizeInset(p);
  }
  // Inset: the size distribution, since the close view shows one particle at the mean.
  function sizeInset(p) {
    const ctx = picCanvas.getContext("2d"), ratio = window.devicePixelRatio || 1;
    ctx.save(); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const w = picCanvas.clientWidth, bw = 120, bh = 54, x0 = w - bw - 10, y0 = 10;
    ctx.fillStyle = "rgba(255,255,255,0.88)"; ctx.fillRect(x0, y0, bw, bh);
    ctx.font = FONT; ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText("radius distribution, nm", x0 + 6, y0 + 4);
    const rs = Array.from({ length: 60 }, (_, i) => 80 + i * (480 - 80) / 59), dens = LNP.radiusDensity(p.r_mean, p.cv, rs);
    const top = Math.max(...dens), X = (r) => x0 + 6 + (r - 80) / 400 * (bw - 12), Y = (d) => y0 + bh - 14 - d / top * (bh - 32);
    ctx.strokeStyle = C.navy; ctx.lineWidth = 1.5; ctx.beginPath();
    rs.forEach((r, i) => (i ? ctx.lineTo(X(r), Y(dens[i])) : ctx.moveTo(X(r), Y(dens[i]))));
    ctx.stroke();
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(x0 + 6, y0 + bh - 13.5); ctx.lineTo(x0 + bw - 6, y0 + bh - 13.5); ctx.stroke();
    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    for (const r of [100, 200, 300, 400]) ctx.fillText(String(r / 10), X(r), y0 + bh - 3);
    ctx.restore();
  }

  let yRange = null;
  function drawCurve(cur) {
    const { ctx, w, h } = fitCanvas(curveCanvas);
    const m = { l: 50, r: 12, t: 12, b: 40 }, y = cur.y;
    let lo = Math.min(...y), hi = Math.max(...y);
    if (kept) { lo = Math.min(lo, ...kept); hi = Math.max(hi, ...kept); }
    if (!yRange) yRange = [lo - 0.2, hi + 0.2];
    yRange = [Math.min(yRange[0], lo - 0.2), Math.max(yRange[1], hi + 0.2)];
    const q0 = Math.log10(D.q[0]), q1 = Math.log10(D.q[NQ - 1]);
    const X = (q) => m.l + (Math.log10(q) - q0) / (q1 - q0) * (w - m.l - m.r);
    const Y = (v) => h - m.b - (v - yRange[0]) / (yRange[1] - yRange[0]) * (h - m.t - m.b);
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.lineWidth = 1; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of [0.003, 0.01, 0.03, 0.1]) {
      ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), h - m.b); ctx.stroke(); ctx.fillText(String(t), X(t), h - m.b + 6);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const step = yRange[1] - yRange[0] > 6 ? 2 : 1;
    for (let t = Math.ceil(yRange[0]); t <= yRange[1]; t++) {
      if (t % step) continue;
      ctx.beginPath(); ctx.moveTo(m.l, Y(t)); ctx.lineTo(w - m.r, Y(t)); ctx.stroke(); ctx.fillText(String(t), m.l - 6, Y(t));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillText("q (Å⁻¹)", (m.l + w - m.r) / 2, h - 6);
    ctx.save(); ctx.translate(12, (m.t + h - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle";
    ctx.fillText("log₁₀ intensity", 0, 0); ctx.restore();
    const line = (ys, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      ys.forEach((v, i) => (i ? ctx.lineTo(X(D.q[i]), Y(v)) : ctx.moveTo(X(D.q[i]), Y(v))));
      ctx.stroke();
    };
    if (kept) line(kept, C.free, 1.8);
    line(y, C.navy, 2.2);
  }
  const draw = throttled(() => { const cur = current(); drawMaterial(cur); drawCurve(cur); });

  // ---- controls --------------------------------------------------------------------
  function slider(parent, label, max, value, format, onInput) {
    const row = document.createElement("label");
    row.className = "ctrl";
    row.innerHTML = `<span class="ctrl-name">${label}</span><output></output><input type="range" min="0" max="${max}" step="1" value="${value}">`;
    const input = row.querySelector("input"), out = row.querySelector("output");
    const show = () => { out.textContent = format(Number(input.value)); };
    input.addEventListener("input", () => { onInput(Number(input.value)); show(); });
    show(); parent.append(row);
  }
  const groups = { size: root.querySelector(".size"), shell: root.querySelector(".shell"), interior: root.querySelector(".interior") };
  for (const [g, list] of [["size", SIZE], ["shell", SHELL], ["interior", INTERIOR]]) {
    for (const [key, label, fmt] of list) {
      slider(groups[g], label, n[key] - 1, state[key], (i) => fmt(D.grid[key][i], i), (i) => { state[key] = i; target(state); });
    }
  }
  const keep = root.querySelector(".keep");
  keep.addEventListener("click", () => {
    kept = kept ? null : current().y;
    keep.textContent = kept ? "Clear kept curve" : "Keep this curve";
    draw();
  });
  const zoom = root.querySelector(".zoom"), viewLabel = root.querySelector(".view-label");
  zoom.addEventListener("click", () => {
    view = view === "close" ? "wide" : "close";
    zoom.textContent = view === "close" ? "Zoom out" : "Zoom in";
    viewLabel.textContent = VIEWS[view].label;
    draw();
  });

  window.addEventListener("resize", draw);
  target(state);
  draw();
})();
