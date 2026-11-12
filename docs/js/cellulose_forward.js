// Cellulose: from material to curve. Structures and partial spectra were precomputed
// on a parameter grid with one fixed random realization, so the material changes
// gradually; see site_precompute/cellulose_forward.py. Phase densities act live.
(function () {
  const D = window.CELLULOSE_FORWARD;
  const root = document.getElementById("forward");
  const NQ = D.q.length, NM = D.n_matrix;
  const AX = [["radius", "Fibril radius", (v) => `${v.toFixed(1)} nm`],
              ["spread", "Fibril size spread", (v) => v.toFixed(2)],
              ["sep", "Minimum separation", (v) => `${v.toFixed(1)} nm`],
              ["count", "Fibrils per area", null]];                // shows the realized density, see fibrilDensity
  const MX = [["vf", "Matrix volume fraction", (v) => `${Math.round(v * 100)} %`],
              ["kappa", "Matrix domain size", (v) => `${Math.round(48 / v)} nm`]];
  const n = Object.fromEntries(["radius", "spread", "sep", "count", "kappa", "vf"].map((k) => [k, D[k].length]));

  // ---- decode ---------------------------------------------------------------------
  const spec = decodeU16(D.spectra_b64);                       // (size, matrix, 3, q)
  // Fibrils: a candidate position table plus, per (radius, spread, separation) group,
  // the accepted rods in acceptance order; each count is a prefix of its group.
  const cand = (() => {
    const u = decodeU16(D.cand_b64), x = new Float32Array(D.n_cand), y = new Float32Array(D.n_cand);
    for (let i = 0; i < D.n_cand; i++) { x[i] = u[2 * i] / 65535 * D.box_nm - D.box_nm / 2; y[i] = u[2 * i + 1] / 65535 * D.box_nm - D.box_nm / 2; }
    return { x, y };
  })();
  const groups = (() => {
    const bin = atob(D.pics_b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const out = []; let o = 0;
    for (const len of D.group_len) {
      const idx = new Uint32Array(len), r = new Float32Array(len);
      for (let i = 0; i < len; i++, o += 4) { idx[i] = bytes[o] | (bytes[o + 1] << 8) | (bytes[o + 2] << 16); r[i] = bytes[o + 3] / 255 * D.r_max_nm; }
      out.push({ idx, r });
    }
    return out;
  })();
  // Two views of the same material: the close one shows the central CROP, the wide one the whole box.
  const VIEWS = { close: { width: D.crop_nm, bar: 10 }, wide: { width: D.box_nm, bar: 100 } };
  let view = "wide";            // start with the whole simulated box; the button zooms in
  const picCache = new Map();
  function pic(s, v) {
    const g = (s.radius * n.spread + s.spread) * n.sep + s.sep, cut = D.cuts[g][s.count], key = `${g}|${cut}|${v}`;
    if (picCache.has(key)) return picCache.get(key);
    const G = groups[g], half = VIEWS[v].width / 2 + D.r_max_nm, idx = [], x = [], y = [], r = [];
    for (let i = 0; i < cut; i++) {
      const k = G.idx[i], px = cand.x[k], py = cand.y[k];
      if (Math.abs(px) < half && Math.abs(py) < half) { idx.push(k); x.push(px); y.push(py); r.push(G.r[i]); }
    }
    const out = { idx, x, y, r };
    picCache.set(key, out);
    return out;
  }
  // The matrix is the region where a smooth random field is below a threshold; one
  // field per domain size, thresholds per volume fraction. Stored at the simulator's
  // pixel size (close view) or half of it (wide view) and interpolated, so the boundary
  // is smooth at any zoom.
  function fieldSet(b64, W, lo, hi, bits) {
    const u = bits === 16 ? decodeU16(b64) : (() => { const bin = atob(b64), o = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) o[i] = bin.charCodeAt(i); return o; })();
    const full = bits === 16 ? 65535 : 255, fields = [], slopes = [];
    for (let k = 0; k < D.kappa.length; k++) {
      const f = new Float32Array(W * W);
      for (let i = 0; i < W * W; i++) f[i] = lo[k] + u[k * W * W + i] / full * (hi[k] - lo[k]);
      const g = new Float32Array(W * W);                       // |gradient| per field pixel, for anti-aliasing
      for (let j = 0; j < W; j++) for (let i = 0; i < W; i++) {
        const gx = (f[j * W + Math.min(i + 1, W - 1)] - f[j * W + Math.max(i - 1, 0)]) / 2;
        const gy = (f[Math.min(j + 1, W - 1) * W + i] - f[Math.max(j - 1, 0) * W + i]) / 2;
        g[j * W + i] = Math.max(Math.hypot(gx, gy), 1e-9);
      }
      fields.push(f); slopes.push(g);
    }
    return { W, fields, slopes };
  }
  const fieldSets = { close: fieldSet(D.field_close_b64, D.close_px, D.close_lo, D.close_hi, 16),
                      wide: fieldSet(D.field_wide_b64, D.wide_px, D.wide_lo, D.wide_hi, 8) };
  function bilinear(f, W, fx, fy) {                           // fx, fy in field pixels
    fx = Math.min(Math.max(fx, 0), W - 1.001); fy = Math.min(Math.max(fy, 0), W - 1.001);
    const i = Math.floor(fx), j = Math.floor(fy), a = fx - i, b = fy - j, o = j * W + i;
    return (f[o] * (1 - a) + f[o + 1] * a) * (1 - b) + (f[o + W] * (1 - a) + f[o + W + 1] * a) * b;
  }
  // m = { ka, kb, t, thr }: the field is blended between two domain sizes while tweening.
  function fieldAt(m, F, fx, fy) {
    const W = F.W;
    return m.t <= 0 ? bilinear(F.fields[m.ka], W, fx, fy) : m.t >= 1 ? bilinear(F.fields[m.kb], W, fx, fy)
      : bilinear(F.fields[m.ka], W, fx, fy) * (1 - m.t) + bilinear(F.fields[m.kb], W, fx, fy) * m.t;
  }
  const matrixLayer_ = document.createElement("canvas");
  let layerImg = null, layerKey = "";
  function matrixLayer(m, rgb, L, F) {
    const key = [m.ka, m.kb, m.t.toFixed(3), m.thr.toFixed(5), rgb.join(), L, F.W].join("|");
    if (key === layerKey) return matrixLayer_;
    if (matrixLayer_.width !== L) { matrixLayer_.width = matrixLayer_.height = L; layerImg = matrixLayer_.getContext("2d").createImageData(L, L); }
    const W = F.W, px = layerImg.data, s = W / L, t = m.t;     // s: field pixels per layer pixel
    let f = F.fields[m.ka], sl = F.slopes[m.ka];
    if (t > 0 && m.ka !== m.kb) {                              // blend on the field grid first (cheap)
      const fa = f, sa = sl, fb = F.fields[m.kb], sb = F.slopes[m.kb];
      f = new Float32Array(W * W); sl = new Float32Array(W * W);
      for (let i = 0; i < W * W; i++) { f[i] = fa[i] * (1 - t) + fb[i] * t; sl[i] = sa[i] * (1 - t) + sb[i] * t; }
    }
    const i0 = new Int32Array(L), wa = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      const fx = Math.min(Math.max((i + 0.5) * s - 0.5, 0), W - 1.001);
      i0[i] = Math.floor(fx); wa[i] = fx - i0[i];
    }
    const thr = m.thr, r0 = rgb[0], r1 = rgb[1], r2 = rgb[2];
    for (let j = 0; j < L; j++) {
      const jj = i0[j], b = wa[j], row0 = jj * W, row1 = row0 + W;
      for (let i = 0; i < L; i++) {
        const ii = i0[i], a = wa[i], o00 = row0 + ii, o10 = row1 + ii;
        const g = (f[o00] * (1 - a) + f[o00 + 1] * a) * (1 - b) + (f[o10] * (1 - a) + f[o10 + 1] * a) * b;
        const sg = (sl[o00] * (1 - a) + sl[o00 + 1] * a) * (1 - b) + (sl[o10] * (1 - a) + sl[o10 + 1] * a) * b;
        const alpha = Math.min(Math.max(0.5 + (thr - g) / (sg * s), 0), 1);   // one layer pixel wide edge
        const o = 4 * (j * L + i);
        px[o] = r0; px[o + 1] = r1; px[o + 2] = r2; px[o + 3] = (alpha * 255) | 0;
      }
    }
    matrixLayer_.getContext("2d").putImageData(layerImg, 0, 0);
    layerKey = key;
    return matrixLayer_;
  }
  // The simulator removes a fibril whose centre is within its radius of the matrix.
  function nearMatrix(m, x, y, r, F, width) {
    const s = F.W / width;
    for (let a = 0; a < 12; a++) {
      const fx = (x + width / 2 + r * Math.cos(a * Math.PI / 6)) * s - 0.5;
      const fy = (y + width / 2 + r * Math.sin(a * Math.PI / 6)) * s - 0.5;
      if (fx < -0.5 || fy < -0.5 || fx > F.W - 0.5 || fy > F.W - 0.5) continue;
      if (fieldAt(m, F, fx, fy) <= m.thr) return true;
    }
    return false;
  }

  // The number of fibrils actually in the material: the rods of the whole box that
  // survive the matrix rule, per um^2. The slider steps the simulator's packing
  // parameter (the nominal count); this is what it produces.
  function fibrilDensity(s) {
    const p = pic(s, "wide"), F = fieldSets.wide, m = { ka: s.kappa, kb: s.kappa, t: 0, thr: D.thresholds[s.kappa][s.vf] };
    let count = 0;
    for (let i = 0; i < p.idx.length; i++) {
      if (Math.abs(p.x[i]) >= D.box_nm / 2 || Math.abs(p.y[i]) >= D.box_nm / 2) continue;
      if (!nearMatrix(m, p.x[i], p.y[i], p.r[i], F, D.box_nm)) count++;
    }
    return count / (D.box_nm / 1000) ** 2;
  }
  const fmtDensity = (v) => `${(Math.round(v / 100) * 100).toLocaleString("en-US")} per µm²`;

  const sizeIndex = (s) => ((s.radius * n.spread + s.spread) * n.sep + s.sep) * n.count + s.count;
  const matrixIndex = (s) => s.kappa * n.vf + s.vf;
  function partials(s) {
    const o = (sizeIndex(s) * NM + matrixIndex(s)) * 3 * NQ, out = [new Float64Array(NQ), new Float64Array(NQ), new Float64Array(NQ)];
    for (let i = 0; i < NQ; i++) {
      const l11 = D.log_lo + spec[o + i] / 65535 * (D.log_hi - D.log_lo);
      const l22 = D.log_lo + spec[o + NQ + i] / 65535 * (D.log_hi - D.log_lo);
      const corr = spec[o + 2 * NQ + i] / 65535 * 2 - 1;
      out[0][i] = 10 ** l11; out[1][i] = 10 ** l22; out[2][i] = corr * Math.sqrt(out[0][i] * out[1][i]);
    }
    return out;
  }
  function intensity(S, rho) {
    const d1 = rho[1] - rho[0], d2 = rho[2] - rho[0], y = new Float64Array(NQ);
    for (let i = 0; i < NQ; i++) y[i] = Math.log10(Math.max(d1 * d1 * S[0][i] + d2 * d2 * S[1][i] + 2 * d1 * d2 * S[2][i], 1e-12));
    return y;
  }

  // ---- state and tweening ---------------------------------------------------------
  const state = Object.fromEntries(Object.entries(D.default).map(([k, v]) => [k, D[k].indexOf(v)]));
  state.rho = [...D.rho_example];
  let kept = null;
  const shown = { from: null, to: null, t: 1, start: 0 };     // tween between two grid points
  const DUR = 350;
  // Blue tones: hue fixed, lightness from density (0.5 pale to 1.0 navy).
  function tone(rho) {
    const l = (94 - (rho - 0.5) * 128) / 100, s = 0.5, h = 210 / 360;
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
    const f = (t) => { t = (t + 1) % 1; return t < 1 / 6 ? p + (q - p) * 6 * t : t < 0.5 ? q : t < 2 / 3 ? p + (q - p) * (2 / 3 - t) * 6 : p; };
    return [f(h + 1 / 3), f(h), f(h - 1 / 3)].map((v) => Math.round(v * 255));
  }

  function snapshot(s) {
    return { pic: pic(s, view), S: partials(s), m: { ka: s.kappa, kb: s.kappa, t: 0, thr: D.thresholds[s.kappa][s.vf] } };
  }
  function target(s) {
    const next = snapshot(s);
    if (!shown.to) { shown.from = shown.to = next; shown.t = 1; return; }
    shown.from = current();                                   // continue from wherever we are
    shown.to = next; shown.t = 0; shown.start = performance.now();
    requestAnimationFrame(tick);
  }
  // The structure as it is on screen now: fibrils matched by index, curves blended.
  function current() {
    if (shown.t >= 1) return shown.to;
    const t = shown.t, a = shown.from.pic, b = shown.to.pic;
    const mapA = new Map(); for (let i = 0; i < a.idx.length; i++) mapA.set(a.idx[i], i);
    const idx = [], x = [], y = [], r = [];
    const seen = new Set();
    for (let j = 0; j < b.idx.length; j++) {
      const i = mapA.get(b.idx[j]); seen.add(b.idx[j]);
      if (i === undefined) { idx.push(b.idx[j]); x.push(b.x[j]); y.push(b.y[j]); r.push(b.r[j] * t); }
      else { idx.push(b.idx[j]); x.push(a.x[i] + (b.x[j] - a.x[i]) * t); y.push(a.y[i] + (b.y[j] - a.y[i]) * t); r.push(a.r[i] + (b.r[j] - a.r[i]) * t); }
    }
    for (let i = 0; i < a.idx.length; i++) {
      if (!seen.has(a.idx[i])) { idx.push(a.idx[i]); x.push(a.x[i]); y.push(a.y[i]); r.push(a.r[i] * (1 - t)); }
    }
    const S = shown.from.S.map((sa, k) => sa.map((v, i) => v + (shown.to.S[k][i] - v) * t));
    const ma = shown.from.m, mb = shown.to.m;                   // matrix: field and threshold blended
    const ka = ma.t < 0.5 ? ma.ka : ma.kb, m = { ka, kb: mb.kb, t: ka === mb.kb ? 0 : t, thr: ma.thr + (mb.thr - ma.thr) * t };
    return { pic: { idx, x, y, r }, S, m };
  }
  function tick(now) {
    const u = Math.min(1, (now - shown.start) / DUR);
    shown.t = u * u * (3 - 2 * u);
    draw();
    if (u < 1) requestAnimationFrame(tick); else { shown.t = 1; shown.from = shown.to; }
  }

  // ---- drawing --------------------------------------------------------------------
  const picCanvas = root.querySelector(".material"), curveCanvas = root.querySelector(".curve");
  function drawMaterial(cur) {
    const { ctx, w, h } = fitCanvas(picCanvas);
    const V = VIEWS[view], F = fieldSets[view];
    const scale = w / V.width, X = (v) => (v + V.width / 2) * scale, Y = (v) => (v + V.width / 2) * scale;
    const g = state.rho.map(tone);
    ctx.fillStyle = `rgb(${g[2]})`; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = `rgb(${g[0]})`;
    const p = cur.pic, s3 = Math.sqrt(3) / 2;
    for (let i = 0; i < p.idx.length; i++) {
      const L = p.r[i] * scale; if (L <= 0) continue;
      if (nearMatrix(cur.m, p.x[i], p.y[i], p.r[i], F, V.width)) continue;
      const cx = X(p.x[i]), cy = Y(p.y[i]);
      ctx.beginPath();
      ctx.moveTo(cx + L, cy); ctx.lineTo(cx + L / 2, cy + s3 * L); ctx.lineTo(cx - L / 2, cy + s3 * L);
      ctx.lineTo(cx - L, cy); ctx.lineTo(cx - L / 2, cy - s3 * L); ctx.lineTo(cx + L / 2, cy - s3 * L);
      ctx.closePath(); ctx.fill();
    }
    const L = Math.min(1024, Math.round(w * (window.devicePixelRatio || 1)));
    ctx.drawImage(matrixLayer(cur.m, g[1], L, F), 0, 0, w, h);
    const bar = V.bar * scale, label = `${V.bar} nm`;
    ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(10, h - 30, bar + 20 + ctx.measureText(label).width + 10, 22);
    ctx.fillStyle = C.ink; ctx.fillRect(14, h - 20, bar, 3);
    ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.fillText(label, bar + 20, h - 19);
  }

  let yRange = null;
  function drawCurve(cur) {
    const { ctx, w, h } = fitCanvas(curveCanvas);
    const m = { l: 50, r: 12, t: 12, b: 40 };
    const y = intensity(cur.S, state.rho);
    let lo = Math.min(...y), hi = Math.max(...y);
    if (kept) { lo = Math.min(lo, ...kept); hi = Math.max(hi, ...kept); }
    if (!yRange) yRange = [lo - 0.2, hi + 0.2];
    yRange = [Math.min(yRange[0], lo - 0.2), Math.max(yRange[1], hi + 0.2)];
    const q0 = Math.log10(D.q[0]), q1 = Math.log10(D.q[NQ - 1]);
    const X = (q) => m.l + (Math.log10(q) - q0) / (q1 - q0) * (w - m.l - m.r);
    const Y = (v) => h - m.b - (v - yRange[0]) / (yRange[1] - yRange[0]) * (h - m.t - m.b);
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.lineWidth = 1; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of [0.01, 0.03, 0.1, 0.3]) {
      ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), h - m.b); ctx.stroke(); ctx.fillText(String(t), X(t), h - m.b + 6);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let t = Math.ceil(yRange[0]); t <= yRange[1]; t++) {
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

  // ---- controls -------------------------------------------------------------------
  function slider(parent, label, min, max, step, value, format, onInput) {
    const row = document.createElement("label");
    row.className = "ctrl";
    row.innerHTML = `<span class="ctrl-name">${label}</span><output></output><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
    const input = row.querySelector("input"), out = row.querySelector("output");
    const show = () => { out.textContent = format(Number(input.value)); };
    input.addEventListener("input", () => { onInput(Number(input.value)); show(); });
    show(); parent.append(row);
    return show;
  }
  const fib = root.querySelector(".fibrils"), mat = root.querySelector(".matrix"), dens = root.querySelector(".densities");
  let showDensity = () => {};
  const setGrid = (key, i) => { state[key] = i; target(state); showDensity(); };
  for (const [key, label, fmt] of AX) {
    const show = slider(fib, label, 0, n[key] - 1, 1, state[key], fmt ? (i) => fmt(D[key][i]) : () => fmtDensity(fibrilDensity(state)), (i) => setGrid(key, i));
    if (!fmt) showDensity = show;
  }
  for (const [key, label, fmt] of MX) {
    slider(mat, label, 0, n[key] - 1, 1, state[key], (i) => fmt(D[key][i]), (i) => setGrid(key, i));
  }
  ["Fibril density", "Matrix density", "Interstitial density"].forEach((label, k) => {
    slider(dens, label, 0.5, 1.0, 0.01, state.rho[k], (v) => v.toFixed(2), (v) => { state.rho[k] = v; draw(); });
  });
  const keep = root.querySelector(".keep");
  keep.addEventListener("click", () => {
    kept = kept ? null : intensity(current().S, state.rho);
    keep.textContent = kept ? "Clear kept curve" : "Keep this curve";
    draw();
  });

  const zoom = root.querySelector(".zoom"), viewLabel = root.querySelector(".view-label");
  zoom.addEventListener("click", () => {
    view = view === "close" ? "wide" : "close";
    zoom.textContent = view === "close" ? "Zoom out" : "Zoom in";
    viewLabel.textContent = view === "close" ? `Cross-section, ${D.crop_nm} nm wide` : `Whole simulated box, ${Math.round(D.box_nm)} nm`;
    shown.from = shown.to = snapshot(state); shown.t = 1;      // same material, other extent
    draw();
  });

  window.addEventListener("resize", draw);
  target(state);
  draw();
})();
