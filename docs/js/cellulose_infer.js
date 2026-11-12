// Cellulose: from curve to material. Posterior samples were precomputed with the
// trained model for one simulated curve; see site_precompute/cellulose_posterior.py.
(function () {
  const D = window.CELLULOSE_DATA;
  const root = document.getElementById("infer");
  const samples = decodeU16(D.samples_b64);
  const P = 9, G = D.grid.length, S = D.sds.length;
  // Parameters shown: the ones that can be locked over most of their range and
  // visibly move the others. The rest are tightly fixed by the curve already.
  const SHOW = ["radius_nm", "SLD_tubes", "SLD_grf_1", "SLD_grf_2", "volfrac_grf"].map((n) => D.names.indexOf(n));
  const state = { lock: null, g: 10, s: 0, truth: false, pair: [D.names.indexOf("SLD_grf_1"), D.names.indexOf("SLD_grf_2")] };
  // A link can open a locked state: cellulose.html#lock=SLD_tubes&g=10&s=1
  const hash = new URLSearchParams(location.hash.slice(1));
  if (SHOW.includes(D.names.indexOf(hash.get("lock")))) {
    state.lock = D.names.indexOf(hash.get("lock"));
    state.g = Math.min(G - 1, Math.max(0, Number(hash.get("g") ?? state.g)));
    state.s = Math.min(S - 1, Math.max(0, Number(hash.get("s") ?? 0)));
    state.truth = hash.get("truth") === "1";
  }

  // ---- data access -------------------------------------------------------------
  const block = (offset, n) => {
    const cols = Array.from({ length: P }, () => new Float32Array(n));
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < P; k++) cols[k][i] = samples[(offset + i) * P + k] / 65535;
    }
    return cols;
  };
  const free = block(0, D.n_free);
  const cache = new Map();
  function conditional() {
    if (state.lock === null) return null;
    const key = `${state.lock},${state.g},${state.s}`;
    if (!cache.has(key)) {
      const idx = (state.lock * G + state.g) * S + state.s;
      cache.set(key, block(D.n_free + idx * D.n_case, D.n_case));
    }
    return cache.get(key);
  }
  // Fibrils per area: the number the simulator actually places per um^2 of material,
  // as a function of the parameters fitted on its recorded counts
  // (site_precompute/cellulose_count.py). Derived from every sample, not inferred.
  const CM = D.count_map;
  const phys = (k, u) => (D.log[k] ? Math.exp(Math.log(D.lo[k]) + u * (Math.log(D.hi[k]) - Math.log(D.lo[k]))) : D.lo[k] + u * (D.hi[k] - D.lo[k]));
  const GEO = ["radius_nm", "kappa_sc", "volfrac_grf", "no_rods_per_unit", "min_space_nm", "size_cv"].map((n) => D.names.indexOf(n));
  const AX0 = Math.log10(CM.axis_per_um2[0]), AX1 = Math.log10(CM.axis_per_um2[1]);
  const areaPos = (v) => (Math.log10(v) - AX0) / (AX1 - AX0);
  function perArea(cols) {
    const n = cols[0].length, pos = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const [r, kap, vf, nr, gap, cv] = GEO.map((k) => phys(k, cols[k][i]));
      const f = [Math.log(nr * CM.r_per_af_nm / r), Math.log(1 - vf), Math.log(r), gap, cv, Math.log(kap)];
      let s = 0;
      for (let t = 0; t < CM.coef.length; t++) {
        let m = CM.coef[t]; const p = CM.powers[t];
        for (let j = 0; j < 6; j++) for (let e = 0; e < p[j]; e++) m *= f[j];
        s += m;
      }
      pos[i] = areaPos(Math.exp(s) * CM.per_um2);
    }
    return pos;
  }
  const freeArea = perArea(free);
  const areaCache = new Map();
  function conditionalArea() {
    const cond = conditional();
    if (!cond) return null;
    const key = `${state.lock},${state.g},${state.s}`;
    if (!areaCache.has(key)) areaCache.set(key, perArea(cond));
    return areaCache.get(key);
  }
  const fmtArea = (v) => (v >= 1000 ? `${Number((v / 1000).toPrecision(2))}k` : String(Math.round(v)));

  const disp = (k, u) => (D.flip[k] ? 1 - u : u);        // unit -> display position
  const toGrid = (k, d) => Math.round(disp(k, Math.min(1, Math.max(0, d))) * (G - 1));
  const fmt = (k, v) => {
    const t = v >= 1000 ? `${(v / 1000).toPrecision(3)}k` : v.toPrecision(3);
    return D.units[k] ? `${t} ${D.units[k]}` : t;
  };
  const compat = () => D.compat[state.lock][state.g][state.s];
  const lockAt = (k, d) => { state.lock = k; state.g = toGrid(k, d); draw(); };

  // ---- curve panels ------------------------------------------------------------
  function curveAxes(ctx, w, h, ys) {
    const m = { l: 50, r: 12, t: 10, b: 40 };
    const q0 = Math.log10(D.window[0]), q1 = Math.log10(D.window[1]);
    let y0 = Infinity, y1 = -Infinity;
    for (const v of ys) { y0 = Math.min(y0, v); y1 = Math.max(y1, v); }
    y0 -= 0.1; y1 += 0.1;
    const X = (q) => m.l + (Math.log10(q) - q0) / (q1 - q0) * (w - m.l - m.r);
    const Y = (y) => h - m.b - (y - y0) / (y1 - y0) * (h - m.t - m.b);

    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.lineWidth = 1;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of [0.01, 0.03, 0.1, 0.3]) {
      ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), h - m.b); ctx.stroke();
      ctx.fillText(String(t), X(t), h - m.b + 6);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let t = Math.ceil(y0); t <= y1; t++) {
      ctx.beginPath(); ctx.moveTo(m.l, Y(t)); ctx.lineTo(w - m.r, Y(t)); ctx.stroke();
      ctx.fillText(String(t), m.l - 6, Y(t));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillText("q (Å⁻¹)", (m.l + w - m.r) / 2, h - 6);
    ctx.save(); ctx.translate(12, (m.t + h - m.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle"; ctx.fillText("log₁₀ intensity", 0, 0); ctx.restore();
    return { X, Y };
  }
  function line(ctx, X, Y, q, ys, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath();
    ys.forEach((y, i) => (i ? ctx.lineTo(X(q[i]), Y(y)) : ctx.moveTo(X(q[i]), Y(y))));
    ctx.stroke();
  }

  function drawCurve() {
    const { ctx, w, h } = fitCanvas(root.querySelector(".curve"));
    const { X, Y } = curveAxes(ctx, w, h, D.y_obs);
    for (const curve of D.ppc) line(ctx, X, Y, D.q, curve, "rgba(31, 78, 121, 0.45)", 1.2);
    ctx.fillStyle = C.obs;
    D.y_obs.forEach((y, i) => { ctx.beginPath(); ctx.arc(X(D.q[i]), Y(y), 1.4, 0, 7); ctx.fill(); });
  }

  function drawAlt() {
    if (!D.alt) return;
    const { ctx, w, h } = fitCanvas(root.querySelector(".alt-curve"));
    const { X, Y } = curveAxes(ctx, w, h, D.alt.curve_a);
    line(ctx, X, Y, D.alt.q, D.alt.curve_a, C.navy, 2.5);
    line(ctx, X, Y, D.alt.q, D.alt.curve_b, C.accent, 1.5);
  }

  // ---- joint panel -------------------------------------------------------------
  const jointCanvas = root.querySelector(".joint");
  const JM = { l: 76, r: 12, t: 10, b: 52 };               // margins hold the tick labels and the axis selectors
  // An axis of the joint plot is a parameter index or "area" (fibrils per area, derived).
  const AREA = "area";
  const AREA_TICKS = [1e3, 3e3, 1e4, 3e4, 1e5].map((v) => [areaPos(v), fmtArea(v)]);
  const axisTicks = (k) => (k === AREA ? AREA_TICKS : D.ticks[k]);
  const axisPos = (k, cols) => (k === AREA ? (cols === free ? freeArea : conditionalArea()) : Array.from(cols[k], (u) => disp(k, u)));
  const truthPos = (k) => (k === AREA ? areaPos(CM.truth_per_um2) : disp(k, D.truth_unit[k]));
  function drawJoint() {
    const { ctx, w, h } = fitCanvas(jointCanvas);
    const [a, b] = state.pair, m = JM, W = w - m.l - m.r, H = h - m.t - m.b;
    const X = (d) => m.l + d * W;
    const Y = (d) => h - m.b - d * H;
    ctx.fillStyle = "#f4f6f8"; ctx.fillRect(m.l, m.t, W, H);
    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const [d, label] of axisTicks(a)) ctx.fillText(label, X(d), h - m.b + 6);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const [d, label] of axisTicks(b)) ctx.fillText(label, m.l - 6, Y(d));
    // axis titles are the two <select> elements placed over the margins (css .axis-pick)
    ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, W, H); ctx.clip();
    const cond = conditional();
    if (cond) {                                            // the locked value, if on an axis
      const u = D.grid[state.g], sd = D.sds[state.s];
      ctx.fillStyle = "rgba(232, 89, 12, 0.14)"; ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
      if (state.lock === a) {
        const x0 = X(disp(a, u - sd)), x1 = X(disp(a, u + sd)), x = X(disp(a, u));
        if (sd) ctx.fillRect(Math.min(x0, x1), m.t, Math.abs(x1 - x0), H);
        ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, h - m.b); ctx.stroke();
      } else if (state.lock === b) {
        const y0 = Y(disp(b, u - sd)), y1 = Y(disp(b, u + sd)), y = Y(disp(b, u));
        if (sd) ctx.fillRect(m.l, Math.min(y0, y1), W, Math.abs(y1 - y0));
        ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(w - m.r, y); ctx.stroke();
      }
    }
    const dots = (cols, color, r) => {
      const xs = axisPos(a, cols), ys = axisPos(b, cols);
      ctx.fillStyle = color;
      for (let i = 0; i < xs.length; i++) {
        ctx.beginPath(); ctx.arc(X(xs[i]), Y(ys[i]), r, 0, 7); ctx.fill();
      }
    };
    dots(free, cond ? C.free : C.navy, 2);
    if (cond) dots(cond, C.navy, 2.2);
    if (state.truth) {
      const x = X(truthPos(a)), y = Y(truthPos(b));
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x - 7, y); ctx.lineTo(x + 7, y); ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7); ctx.stroke();
    }
    ctx.restore();
  }

  // Drag in the joint plot: horizontal locks the x parameter, vertical the y one.
  // Fibrils per area is derived, not a parameter, so a drag along it does nothing.
  const lockAxis = (k, d) => { if (k !== AREA) lockAt(k, d); };
  let jointDrag = null;
  jointCanvas.addEventListener("pointerdown", (e) => {
    const r = jointCanvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    if (x < JM.l || x > r.width - JM.r || y < JM.t || y > r.height - JM.b) return;
    jointDrag = { x, y, axis: null };
    jointCanvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  jointCanvas.addEventListener("pointermove", (e) => {
    if (!jointDrag) return;
    const r = jointCanvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    if (!jointDrag.axis) {
      const dx = Math.abs(x - jointDrag.x), dy = Math.abs(y - jointDrag.y);
      if (dx < 4 && dy < 4) return;
      jointDrag.axis = dx >= dy ? 0 : 1;
    }
    if (jointDrag.axis === 0) lockAxis(state.pair[0], (x - JM.l) / (r.width - JM.l - JM.r));
    else lockAxis(state.pair[1], (r.height - JM.b - y) / (r.height - JM.t - JM.b));
  });
  const endJointDrag = (e) => {
    if (!jointDrag) return;
    if (!jointDrag.axis) {                                 // a click: lock the x parameter here
      const r = jointCanvas.getBoundingClientRect();
      lockAxis(state.pair[0], (jointDrag.x - JM.l) / (r.width - JM.l - JM.r));
    }
    jointDrag = null;
  };
  jointCanvas.addEventListener("pointerup", endJointDrag);
  jointCanvas.addEventListener("pointercancel", endJointDrag);

  // ---- parameter rows ------------------------------------------------------------
  const rows = new Map();
  const table = root.querySelector(".params");
  const RM = 10;                                           // track margin
  const hover = { k: null, d: 0 };
  SHOW.forEach((k) => {
    const row = document.createElement("div");
    row.className = "param";
    row.innerHTML = `
      <div class="name">${D.labels[k]}${D.units[k] ? `<small>${D.units[k]}</small>` : ""}</div>
      <canvas></canvas>
      <button type="button" class="unlock">Unlock</button>
      <div class="statement">
        <span class="value"></span>
        <span class="choices"></span>
        <span class="verdict"></span>
      </div>`;
    const canvas = row.querySelector("canvas"), choices = row.querySelector(".choices");
    D.sd_labels[k].forEach((label, s) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label;
      b.addEventListener("click", () => { state.s = s; draw(); });
      choices.append(b);
    });
    row.querySelector(".unlock").addEventListener("click", () => { state.lock = null; draw(); });

    const pos = (e) => {
      const r = canvas.getBoundingClientRect();
      return (e.clientX - r.left - RM) / (r.width - 2 * RM);
    };
    let dragging = false;
    canvas.addEventListener("pointerdown", (e) => {
      dragging = true; canvas.setPointerCapture(e.pointerId); e.preventDefault();
      lockAt(k, pos(e));
    });
    canvas.addEventListener("pointermove", (e) => {
      if (dragging) { lockAt(k, pos(e)); return; }
      hover.k = k; hover.d = pos(e); drawRow(k);
    });
    const stop = () => { dragging = false; };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("pointerleave", () => { if (hover.k === k) { hover.k = null; drawRow(k); } });
    table.append(row);
    rows.set(k, { row, canvas, choices });
  });
  // The derived row: read only, no lock.
  const areaRow = document.createElement("div");
  areaRow.className = "param derived";
  areaRow.innerHTML = `<div class="name">Fibrils per area<small>per µm² of material, derived from the samples</small></div><canvas></canvas><span></span>`;
  const areaCanvas = areaRow.querySelector("canvas");
  areaCanvas.addEventListener("pointermove", (e) => {
    const r = areaCanvas.getBoundingClientRect();
    hover.k = "area"; hover.d = (e.clientX - r.left - RM) / (r.width - 2 * RM); drawArea();
  });
  areaCanvas.addEventListener("pointerleave", () => { if (hover.k === "area") { hover.k = null; drawArea(); } });
  table.append(areaRow);

  // A posterior as a filled strip over display positions in [0, 1].
  function strip(ctx, X, top, base, pos, color) {
    const dens = density(pos), n = dens.length;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.moveTo(X(0), base);
    for (let i = 0; i < n; i++) ctx.lineTo(X((i + 0.5) / n), base - dens[i] * (base - top));
    ctx.lineTo(X(1), base); ctx.closePath(); ctx.fill();
  }

  function drawArea() {
    const { ctx, w, h } = fitCanvas(areaCanvas);
    const top = 6, base = h - 18, width = w - 2 * RM;
    const X = (d) => RM + d * width;
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.textBaseline = "top"; ctx.textAlign = "center";
    for (let e = Math.ceil(AX0); e <= AX1; e++) {
      for (const m of [1, 3]) {
        const d = areaPos(m * 10 ** e);
        if (d < -1e-6 || d > 1 + 1e-6) continue;
        ctx.beginPath(); ctx.moveTo(X(d), base); ctx.lineTo(X(d), base + 4); ctx.stroke();
        ctx.textAlign = d < 0.02 ? "left" : d > 0.98 ? "right" : "center";
        ctx.fillText(fmtArea(m * 10 ** e), X(d), base + 6);
      }
    }
    ctx.beginPath(); ctx.moveTo(RM, base + 0.5); ctx.lineTo(w - RM, base + 0.5); ctx.stroke();
    const cond = conditionalArea();
    strip(ctx, X, top, base, freeArea, cond ? C.free : C.navy);
    if (cond) strip(ctx, X, top, base, cond, C.navy);
    if (hover.k === "area" && hover.d >= 0 && hover.d <= 1) {
      const x = X(hover.d);
      ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, base); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.muted; ctx.textBaseline = "top"; ctx.textAlign = hover.d > 0.5 ? "right" : "left";
      ctx.fillText(`${fmtArea(10 ** (AX0 + hover.d * (AX1 - AX0)))} per µm²`, x + (hover.d > 0.5 ? -6 : 6), top);
    }
    if (state.truth) {
      const x = X(areaPos(CM.truth_per_um2));
      ctx.fillStyle = C.accent;
      ctx.beginPath(); ctx.moveTo(x, base - 7); ctx.lineTo(x - 5, base + 1); ctx.lineTo(x + 5, base + 1); ctx.closePath(); ctx.fill();
    }
  }

  function drawRow(k) {
    const { row, canvas, choices } = rows.get(k);
    const locked = state.lock === k;
    row.classList.toggle("locked", locked);
    const { ctx, w, h } = fitCanvas(canvas);
    const top = 6, base = h - 18, width = w - 2 * RM;
    const X = (d) => RM + d * width;

    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.textBaseline = "top";
    for (const [d, label] of D.ticks[k]) {
      ctx.beginPath(); ctx.moveTo(X(d), base); ctx.lineTo(X(d), base + 4); ctx.stroke();
      ctx.textAlign = d === 0 ? "left" : d === 1 ? "right" : "center";
      ctx.fillText(label, X(d), base + 6);
    }
    ctx.beginPath(); ctx.moveTo(RM, base + 0.5); ctx.lineTo(w - RM, base + 0.5); ctx.stroke();

    const shown = (cols) => (D.flip[k] ? cols[k].map((u) => 1 - u) : cols[k]);
    const cond = conditional();
    if (!cond) strip(ctx, X, top, base, shown(free), C.navy);
    else if (locked) {
      strip(ctx, X, top, base, shown(free), C.free);
      const u = D.grid[state.g], sd = D.sds[state.s], x = X(disp(k, u));
      if (sd > 0) {
        const x1 = X(disp(k, u - sd)), x2 = X(disp(k, u + sd));
        ctx.fillStyle = "rgba(232, 89, 12, 0.18)";
        ctx.fillRect(Math.min(x1, x2), top, Math.abs(x2 - x1), base - top);
      }
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, base); ctx.stroke();
      ctx.fillStyle = C.accent;
      ctx.beginPath(); ctx.arc(x, base, 7, 0, 7); ctx.fill();
      ctx.lineWidth = 1;
    } else { strip(ctx, X, top, base, shown(free), C.free); strip(ctx, X, top, base, shown(cond), C.navy); }

    if (!locked && hover.k === k && hover.d >= 0 && hover.d <= 1) {   // invitation to drag
      const x = X(hover.d);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, base); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.accent; ctx.textBaseline = "top"; ctx.textAlign = hover.d > 0.5 ? "right" : "left";
      ctx.fillText(fmt(k, D.grid_phys[k][toGrid(k, hover.d)]), x + (hover.d > 0.5 ? -6 : 6), top);
    }
    if (state.truth) {
      const x = X(disp(k, D.truth_unit[k]));
      ctx.fillStyle = C.accent;
      ctx.beginPath(); ctx.moveTo(x, base - 7); ctx.lineTo(x - 5, base + 1); ctx.lineTo(x + 5, base + 1); ctx.closePath(); ctx.fill();
    }

    if (locked) {
      row.querySelector(".value").textContent = `${D.labels[k]} = ${fmt(k, D.grid_phys[k][state.g])}`;
      [...choices.children].forEach((b, s) => b.classList.toggle("on", s === state.s));
      const verdict = row.querySelector(".verdict"), c = compat();
      const conflict = c < 0.01;
      verdict.textContent = conflict
        ? `Conflicts with the curve (compatibility ${c < 0.001 ? "< 0.001" : c.toFixed(3)})`
        : `Compatible with the curve (${c.toFixed(2)})`;
      verdict.className = conflict ? "verdict conflict" : "verdict";
    }
  }

  // ---- controls --------------------------------------------------------------------
  const truthBox = root.querySelector(".show-truth");
  truthBox.checked = state.truth;
  truthBox.addEventListener("change", () => { state.truth = truthBox.checked; draw(); });
  const selects = [root.querySelector(".pair-x"), root.querySelector(".pair-y")];
  selects.forEach((sel, i) => {
    SHOW.forEach((k) => sel.add(new Option(D.labels[k], k)));
    sel.add(new Option("Fibrils per area", AREA));
    sel.value = state.pair[i];
    sel.addEventListener("change", () => { state.pair[i] = sel.value === AREA ? AREA : Number(sel.value); drawJoint(); });
  });

  const draw = throttled(() => {
    root.querySelector(".legend-cond").hidden = state.lock === null;
    drawJoint();
    for (const k of SHOW) drawRow(k);
    drawArea();
  });
  window.addEventListener("resize", () => { drawCurve(); drawAlt(); draw(); });
  drawCurve();
  drawAlt();
  draw();
})();
