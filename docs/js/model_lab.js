// How the model works: one interactive figure of the whole model, and a second one
// for flow matching. Both run on a toy problem whose exact answer is known (a mixture
// of Gaussians), so every velocity and every posterior here is computed exactly in the
// browser. Nothing is output from the trained network, and nothing is about SAXS.
(function () {
  const CAT = { meas: "#1f8a8a", cond: "#c17f59", gen: "#1f4e79" };
  const MUTED = "#5f6770", LINE = "#c9d1d9", INK = "#1a1a1a";
  const X0 = -2.6, X1 = 2.6, Y0 = -2.0, Y1 = 2.2, A_LO = -2.6, A_HI = 2.6, SOFT = 0.52;   // SOFT is 0.10 on the 0..1 scale
  const NAME = ["θ₁", "θ₂", "θ₃", "θ₄"];

  // ---- the toy: for each measurement, a posterior over four parameters ---------------
  // (θ₁, θ₂) lie on a curved band; θ₃ changes along the band, θ₄ across it.
  const MEAS = [{ c: 0.85, sh: -1.05 }, { c: -0.7, sh: 0.95 }, { c: 0.22, sh: -0.2 }];
  function buildComps(m) {
    const out = [], KU = 11, OFF = [-0.17, 0, 0.17];
    for (let k = 0; k < KU; k++) {
      const u = -1.55 + (3.1 * k) / (KU - 1), ty = 2 * m.c * u, n = Math.hypot(1, ty), dx = 1 / n, dy = ty / n;
      for (let j = 0; j < 3; j++) {
        const sT = 0.3 ** 2, sN = 0.12 ** 2;
        out.push({ m: [u - OFF[j] * dy, m.c * u * u + m.sh + OFF[j] * dx], k3: u, k4: (j - 1) * 1.05,
          S: [sT * dx * dx + sN * dy * dy, sT * dx * dy - sN * dx * dy, sT * dy * dy + sN * dx * dx] });
      }
    }
    return out;
  }
  const COMPS = MEAS.map(buildComps), V3 = 0.3 ** 2, V4 = 0.38 ** 2;
  const toy = (d) => A_LO + d * (A_HI - A_LO);
  function weights(comps, roles, vals) {                // what is known about θ₃ and θ₄
    const w = comps.map((c) => {
      let l = 0;
      for (const [i, key, v] of [[2, "k3", V3], [3, "k4", V4]]) {
        if (roles[i] === "unknown") continue;
        const s2 = v + (roles[i] === "exact" ? 0 : SOFT * SOFT), d = toy(vals[i]) - c[key];
        l += -0.5 * d * d / s2 - 0.5 * Math.log(s2);
      }
      return Math.exp(l);
    });
    const tot = w.reduce((a, b) => a + b, 0) || 1;
    return w.map((v) => v / tot);
  }
  // density p_t and velocity u_t for the straight path from N(0, I) to the mixture
  function field(x, y, t, w, comps) {
    let dens = 0, ux = 0, uy = 0;
    const a = (1 - t) * (1 - t), b = t * t;
    for (let k = 0; k < comps.length; k++) {
      if (w[k] < 1e-7) continue;
      const c = comps[k], cxx = a + b * c.S[0], cxy = b * c.S[1], cyy = a + b * c.S[2], det = cxx * cyy - cxy * cxy;
      const rx = x - t * c.m[0], ry = y - t * c.m[1], ix = (cyy * rx - cxy * ry) / det, iy = (-cxy * rx + cxx * ry) / det;
      const g = (w[k] * Math.exp(-0.5 * (rx * ix + ry * iy))) / (2 * Math.PI * Math.sqrt(det));
      const axx = t * c.S[0] - (1 - t), axy = t * c.S[1], ayy = t * c.S[2] - (1 - t);
      dens += g; ux += g * (c.m[0] + axx * ix + axy * iy); uy += g * (c.m[1] + axy * ix + ayy * iy);
    }
    return dens > 1e-300 ? [dens, ux / dens, uy / dens] : [0, 0, 0];
  }
  let seed = 12345;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const N = 300, STEPS = 50;
  const noise = Array.from({ length: N }, () => [gauss(), gauss()]);
  function paths(w, comps) {
    const P = new Float32Array(N * (STEPS + 1) * 2), dt = 1 / STEPS;
    for (let i = 0; i < N; i++) {
      let x = noise[i][0], y = noise[i][1];
      P[i * (STEPS + 1) * 2] = x; P[i * (STEPS + 1) * 2 + 1] = y;
      for (let s = 0; s < STEPS; s++) {
        const t = s * dt, [, u1, v1] = field(x, y, t, w, comps), [, u2, v2] = field(x + 0.5 * dt * u1, y + 0.5 * dt * v1, t + 0.5 * dt, w, comps);
        x += dt * u2; y += dt * v2;
        const o = (i * (STEPS + 1) + s + 1) * 2; P[o] = x; P[o + 1] = y;
      }
    }
    return P;
  }
  const pos = (P, i, t) => { const x = t * STEPS, s0 = Math.min(STEPS - 1, Math.floor(x)), f = x - s0, o = (i * (STEPS + 1) + s0) * 2; return [P[o] * (1 - f) + P[o + 2] * f, P[o + 1] * (1 - f) + P[o + 3] * f]; };

  // ---- shared drawing helpers ------------------------------------------------------------
  function frame(canvas) {
    const { ctx, w, h } = fitCanvas(canvas), m = { l: 26, r: 8, t: 8, b: 24 };
    const X = (x) => m.l + ((x - X0) / (X1 - X0)) * (w - m.l - m.r), Y = (y) => h - m.b - ((y - Y0) / (Y1 - Y0)) * (h - m.t - m.b);
    ctx.strokeStyle = LINE; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, h - m.b); ctx.lineTo(w - m.r, h - m.b); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.textAlign = "center"; ctx.fillText("θ₁", (m.l + w - m.r) / 2, h - 6);
    ctx.save(); ctx.translate(12, (m.t + h - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("θ₂", 0, 0); ctx.restore();
    return { ctx, w, h, m, X, Y };
  }
  const off = document.createElement("canvas"); off.width = 70; off.height = 64;
  function shade(f, t, w, comps, o = {}) {
    const rgb = o.rgb || [31, 78, 121], lv = o.levels || [[0.6, 0.5], [0.3, 0.32], [0.08, 0.17]];
    const g = off.getContext("2d"), img = g.createImageData(off.width, off.height), d = new Float32Array(off.width * off.height);
    let top = 0;
    for (let j = 0; j < off.height; j++) for (let i = 0; i < off.width; i++) {
      const v = field(X0 + ((i + 0.5) / off.width) * (X1 - X0), Y1 - ((j + 0.5) / off.height) * (Y1 - Y0), t, w, comps)[0];
      d[j * off.width + i] = v; if (v > top) top = v;
    }
    for (let p = 0; p < d.length; p++) {
      const r = d[p] / top; let al = 0;
      for (const [th, a] of lv) if (r > th) { al = a; break; }
      img.data[4 * p] = rgb[0]; img.data[4 * p + 1] = rgb[1]; img.data[4 * p + 2] = rgb[2]; img.data[4 * p + 3] = 255 * al;
    }
    g.putImageData(img, 0, 0);
    f.ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha;
    f.ctx.drawImage(off, f.m.l, f.m.t, f.w - f.m.l - f.m.r, f.h - f.m.t - f.m.b);
    f.ctx.globalAlpha = 1;
  }
  // the velocity field as slim tapered streaks (wide at the tail, pointed where it leads)
  function quiver(f, t, w, comps, rgb) {
    const NX = 15, NY = 12;
    for (let gy = 0; gy < NY; gy++) for (let gx = 0; gx < NX; gx++) {
      const x = X0 + ((gx + (gy % 2 ? 0.75 : 0.25)) / NX) * (X1 - X0), y = Y0 + ((gy + 0.5) / NY) * (Y1 - Y0), [dn, u, v] = field(x, y, t, w, comps);
      if (dn < 4e-4) continue;
      const sp = Math.hypot(u, v); if (sp < 0.05) continue;
      const len = Math.min(17, 4 + 5 * sp), ex = u / sp, ey = -v / sp, ax = f.X(x) - ex * len / 2, ay = f.Y(y) - ey * len / 2, hw = 1.5;
      f.ctx.fillStyle = `rgba(${rgb},${Math.min(0.5, 0.16 + 0.12 * sp)})`;
      f.ctx.beginPath(); f.ctx.moveTo(ax + ey * hw, ay - ex * hw); f.ctx.lineTo(ax + ex * len, ay + ey * len); f.ctx.lineTo(ax - ey * hw, ay + ex * hw); f.ctx.closePath(); f.ctx.fill();
    }
  }
  // a clean arrow: round line, small solid head in proportion
  function arrow(ctx, ax, ay, bx, by, col, lw = 1.5) {
    const L = Math.hypot(bx - ax, by - ay); if (L < 2) return;
    const ux = (bx - ax) / L, uy = (by - ay) / L, h = Math.min(8, 3 + L * 0.28), hw = h * 0.42;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx - ux * h * 0.9, by - uy * h * 0.9); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - ux * h + uy * hw, by - uy * h - ux * hw); ctx.lineTo(bx - ux * h - uy * hw, by - uy * h + ux * hw); ctx.closePath(); ctx.fill();
    ctx.lineCap = "butt";
  }
  // the points, each with a short fading trail that shows where it came from
  function swarm(f, P, t) {
    const back = Math.max(0, t - 0.07);
    if (t > 0.01 && t < 0.995) {
      f.ctx.lineCap = "round"; f.ctx.lineWidth = 1.4; f.ctx.strokeStyle = "rgba(31,78,121,0.22)";
      f.ctx.beginPath();
      for (let i = 0; i < N; i++) { const [x0, y0] = pos(P, i, back), [x1, y1] = pos(P, i, t); f.ctx.moveTo(f.X(x0), f.Y(y0)); f.ctx.lineTo(f.X(x1), f.Y(y1)); }
      f.ctx.stroke(); f.ctx.lineCap = "butt";
    }
    const col = t < 0.02 ? "rgba(120,130,140,0.8)" : "rgba(31,78,121,0.75)";
    for (let i = 0; i < N; i++) { const [x, y] = pos(P, i, t); dot(f, x, y, 2.1, col); }
  }
  const dot = (f, x, y, r, col) => { f.ctx.fillStyle = col; f.ctx.beginPath(); f.ctx.arc(f.X(x), f.Y(y), r, 0, 2 * Math.PI); f.ctx.fill(); };

  // =========================================================================================
  // The explorer: one question (what do we know about θ₃?), one model that stays in place,
  // explored in four steps, and what comes out.
  // =========================================================================================
  const box = document.getElementById("explore");
  if (!box) return;
  const P3 = ["θ₁", "θ₂", "θ₃"];
  const st = { role: "unknown", val: 0.7, step: -1, t: 1, anim: null };
  const comps = COMPS[0], FREE = ["unknown", "unknown", "unknown", "unknown"];
  const roles = () => ["unknown", "unknown", st.role, "unknown"], vals = () => [0, 0, st.val, 0.5];
  const STEPS_TXT = [
    ["1", "The measurement", () => "<b>What did we measure?</b> The measurement goes through an encoder, which turns it into a short summary. This is done once. The summary is reused for everything that follows."],
    ["2", "What we know", () => `<b>What do we already know?</b> Every parameter gets a token, a small note that says what we know about it. ${st.role === "unknown" ? "The note for θ₃ says: not known. It will be generated." : st.role === "exact" ? `The note for θ₃ says: its value is ${st.val.toFixed(2)}. It will not be generated.` : `The note for θ₃ says: about ${st.val.toFixed(2)}, give or take 0.10. A hint, not a fixed value.`} A new question changes only this note. The network stays the same, so nothing has to be retrained.`],
    ["3", "Inside the network", () => "<b>What happens inside?</b> The network answers one thing: in which direction should each unknown parameter move? To do that, the tokens first read the summary of the measurement (cross-attention), then read each other (self-attention), and are then updated (MLP). The same three steps are repeated several times."],
    ["4", "What comes out", () => "<b>What comes out?</b> The unknown parameters start as random numbers. The network says how to move them, they take a small step, and this repeats until they have settled. That gives one possible set of parameters. Many random starts give many: the cloud on the right, the posterior. Press Play to watch."],
  ];
  const SAY = {
    unknown: "<b>θ₃ is unknown.</b> The network generates θ₃ together with the other unknown parameters. The measurement allows a whole band of combinations.",
    exact: () => `<b>θ₃ is ${st.val.toFixed(2)}.</b> The network uses this value and generates only the other parameters. Fewer combinations remain.`,
    rough: () => `<b>θ₃ is about ${st.val.toFixed(2)}, give or take 0.10.</b> The network takes this as a hint and weighs every value it allows. More combinations remain than for an exact value.`,
  };
  box.innerHTML = `
    <div class="e-q">
      <span class="e-ask">What do we know about θ₃?</span>
      <div class="e-opts" role="group" aria-label="what we know about theta 3">
        <button type="button" data-role="unknown">Nothing</button><button type="button" data-role="exact">Its exact value</button><button type="button" data-role="rough">An approximate value</button>
      </div>
      <label class="e-val"><span>value</span><input type="range" min="12" max="88" value="70" aria-label="value of theta 3"><output></output></label>
    </div>
    <p class="e-say" aria-live="polite"></p>
    <div class="e-fig">
      <div class="e-model"></div>
      <div class="e-out">
        <p class="e-cap">What comes out: the posterior of θ₁ and θ₂</p>
        <canvas></canvas>
        <div class="e-legend"><span class="legend"><i class="cond"></i>with what we know</span><span class="legend"><i class="ghost"></i>the measurement alone</span><span class="spacer" style="flex:1"></span><button type="button" class="e-play">Play</button></div>
      </div>
    </div>
    <div class="e-steps">
      <div class="e-tabs" role="tablist">${STEPS_TXT.map((s, k) => `<button type="button" role="tab" data-step="${k}"><i>${s[0]}</i>${s[1]}</button>`).join("")}</div>
      <p class="e-text" aria-live="polite"></p>
      <div class="e-nav"><button type="button" data-go="-1">Previous</button><span class="e-pos"></span><button type="button" data-go="1">Next</button></div>
    </div>
    <p class="e-toy">The numbers come from a small made-up example whose answer can be computed exactly. They are not output from the trained network.</p>`;
  const $ = (q) => box.querySelector(q);
  const modelBox = $(".e-model"), canvas = $("canvas"), play = $(".e-play"), slider = $(".e-val input");
  let W, P, W0, cur = null, from = null, morph = 1, morphId = 0;

  function drawModel() {
    const k = st.step, lit = (...g) => (k < 0 || g.includes(k) ? 1 : 0.22), open = k === 2;
    const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" text-anchor="${o.a || "middle"}" fill="${o.f || INK}" font-size="${o.s || 14}" font-weight="${o.w || 400}">${s}</text>`;
    const mk = (id, c) => `<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0 1.2L10 5L0 8.8z" fill="${c}"/></marker>`;
    const G = (g, inner, extra = []) => `<g class="e-g" data-step="${g}" opacity="${lit(g, ...extra)}">${inner}</g>`;
    let curve = ""; for (let i = 0; i <= 30; i++) { const u = i / 30, y = 0.9 * Math.exp(-3.2 * u) + 0.13 * Math.exp(-(((u - 0.55) / 0.1) ** 2)) + 0.05; curve += `${i ? "L" : "M"}${(24 + 96 * u).toFixed(1)} ${(74 - 50 * y).toFixed(1)}`; }
    let s = `<defs>${mk("em", MUTED)}${mk("et", CAT.meas)}${mk("eg", CAT.gen)}</defs>`;
    // 1. the measurement
    s += G(0, `<rect x="16" y="18" width="112" height="62" rx="6" fill="#fff" stroke="${LINE}" stroke-width="1.4"/><path d="${curve}" fill="none" stroke="${CAT.meas}" stroke-width="2.4"/>${T(72, 100, "measurement", { s: 13, f: MUTED })}
      <path d="M134 49 H156" stroke="${CAT.meas}" stroke-width="1.6" marker-end="url(#et)"/>
      <rect x="160" y="30" width="88" height="38" rx="6" fill="#fff" stroke="${CAT.meas}" stroke-width="1.6"/>${T(204, 54, "encoder")}${T(204, 100, "once", { s: 13, f: MUTED })}
      <path d="M252 49 H274" stroke="${CAT.meas}" stroke-width="1.6" marker-end="url(#et)"/>
      ${[1, 0.55, 0.8, 0.4, 0.7].map((a, j) => `<rect x="${278 + j * 25}" y="39" width="20" height="20" rx="3" fill="${CAT.meas}" fill-opacity="${a}" stroke="${CAT.meas}"/>`).join("")}${T(338, 100, "summary", { s: 13, f: MUTED })}`);
    s += `<path d="M404 49 H512 V${open ? 104 : 124}" fill="none" stroke="${CAT.meas}" stroke-width="1.6" marker-end="url(#et)" opacity="${lit(0, 2)}"/>`;
    // 2. what we know: one token per parameter
    const tok = P3.map((n, i) => {
      const r = i === 2 ? st.role : "unknown", x = 16 + i * 96, cx = x + 42;
      const stl = r === "unknown" ? `fill="#fff" stroke="${CAT.gen}"` : r === "rough" ? `fill="#f6e7db" stroke="${CAT.cond}"` : `fill="${CAT.cond}" stroke="${CAT.cond}"`, tc = r === "exact" ? "#fff" : INK;
      const l2 = r === "unknown" ? "not known" : r === "exact" ? `= ${st.val.toFixed(2)}` : `≈ ${st.val.toFixed(2)}`;
      return `<rect class="${i === 2 ? "e-t3" : ""}" x="${x}" y="226" width="84" height="64" rx="8" ${stl} stroke-width="2"/>${T(cx, 252, n, { w: 600, s: 16, f: tc })}${T(cx, 274, l2, { s: 13, f: r === "exact" ? "#fff" : MUTED })}`;
    }).join("");
    s += G(1, tok + T(154, 214, "one token per parameter: what we know", { s: 13, f: MUTED }));
    s += `<path d="M306 258 H394" stroke="${MUTED}" stroke-width="1.6" marker-end="url(#em)" opacity="${lit(1, 2)}"/>`;
    // 3. the network (opens when selected)
    let net = open
      ? `<rect x="398" y="108" width="228" height="194" rx="10" fill="#fff" stroke="#4b5560" stroke-width="1.8"/>${T(512, 130, "the network", { w: 600 })}` +
        [["cross-attention", "read the measurement"], ["self-attention", "read each other"], ["MLP", "update each token"]].map(([a, b], j) => `<rect x="410" y="${140 + j * 50}" width="204" height="42" rx="7" fill="#f1f3f5"/>${T(512, 157 + j * 50, a, { w: 600, s: 13 })}${T(512, 174 + j * 50, b, { s: 12, f: MUTED })}`).join("") + T(512, 296, "repeated several times", { s: 11.5, f: MUTED })
      : `<rect x="398" y="128" width="228" height="150" rx="10" fill="#fff" stroke="#4b5560" stroke-width="1.8"/>${T(512, 196, "the network", { w: 600, s: 17 })}${T(512, 220, "the same for every question", { s: 13, f: MUTED })}`;
    s += G(2, net);
    // 4. directions, a small step, and again
    const y4 = open ? 304 : 280;
    s += G(3, `<path d="M512 ${y4} V326" stroke="${CAT.gen}" stroke-width="1.6" marker-end="url(#eg)"/>` +
      P3.map((n, i) => { const u = i < 2 || st.role === "unknown"; return `<rect x="${440 + i * 50}" y="330" width="44" height="26" rx="13" fill="${u ? "#e9eff5" : "#fff"}" stroke="${u ? CAT.gen : LINE}" ${u ? "" : 'stroke-dasharray="3 3"'} stroke-width="1.4"/>${T(462 + i * 50, 348, u ? "move" : "–", { s: 12, f: u ? CAT.gen : MUTED })}`; }).join("") +
      T(512, 376, "a direction for each unknown parameter", { s: 12.5, f: MUTED }) +
      `<path d="M434 343 H58 V296" fill="none" stroke="${CAT.gen}" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#eg)"/>${T(250, 334, "small step, then ask again", { s: 12.5, f: CAT.gen })}
       <path d="M592 343 H634" stroke="${CAT.gen}" stroke-width="1.8" marker-end="url(#eg)"/>`);
    modelBox.innerHTML = `<svg viewBox="0 0 640 388" role="img" aria-label="The method. The measurement goes through an encoder, once, and becomes a summary. Every parameter has a token that says what is known about it. The network, the same for every question, reads the tokens and the summary and gives a direction for each unknown parameter. The unknown parameters take a small step and the network is asked again; at the end they are one possible set of parameters."><g font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">${s}</g></svg>`;
    modelBox.classList.toggle("run", !!st.anim);
  }
  function drawText() {
    box.querySelectorAll("[data-role]").forEach((b) => b.classList.toggle("on", b.dataset.role === st.role));
    $(".e-val").style.visibility = st.role === "unknown" ? "hidden" : "visible";
    $(".e-val output").textContent = st.val.toFixed(2);
    $(".e-say").innerHTML = st.role === "unknown" ? SAY.unknown : SAY[st.role]();
    box.querySelectorAll(".e-tabs button").forEach((b) => b.classList.toggle("on", Number(b.dataset.step) === st.step));
    $(".e-text").innerHTML = st.step < 0 ? "Three questions to follow through the figure: <b>what goes in</b> (the measurement, and what we know), <b>what happens inside</b> (the network), and <b>what comes out</b> (many possible parameter values). Step through them, or change what we know about θ₃ above." : STEPS_TXT[st.step][2]();
    $(".e-pos").textContent = st.step < 0 ? "" : `${st.step + 1} of 4`;
    $('[data-go="-1"]').disabled = st.step < 0; $('[data-go="1"]').disabled = st.step === 3;
  }
  function drawOut() {
    const f = frame(canvas), t = st.t;
    shade(f, 0.999, W0, comps, { rgb: [120, 130, 140], levels: [[0.14, 0.1], [0.08, 0.3]] });           // the measurement alone
    if (st.anim) { shade(f, Math.min(t, 0.999), W, comps); if (t < 0.995) quiver(f, t, W, comps, "31,78,121"); swarm(f, P, t); return; }
    shade(f, 0.999, W, comps, { alpha: morph });
    const e = morph * morph * (3 - 2 * morph);
    for (let i = 0; i < N; i++) dot(f, from[2 * i] * (1 - e) + cur[2 * i] * e, from[2 * i + 1] * (1 - e) + cur[2 * i + 1] * e, 2.3, "rgba(31,78,121,0.8)");
  }
  function stopFlow() { if (st.anim) cancelAnimationFrame(st.anim); st.anim = null; st.t = 1; play.textContent = "Play"; modelBox.classList.remove("run"); }
  function requery(smooth) {
    stopFlow();
    const old = cur;
    W = weights(comps, roles(), vals()); P = paths(W, comps); W0 = weights(comps, FREE, vals());
    cur = new Float32Array(2 * N); for (let i = 0; i < N; i++) { const [x, y] = pos(P, i, 1); cur[2 * i] = x; cur[2 * i + 1] = y; }
    drawText(); drawModel(); cancelAnimationFrame(morphId);
    if (!old || !smooth) { from = cur; morph = 1; return drawOut(); }
    from = old; morph = 0;
    const t0 = performance.now(), step = (now) => { morph = Math.min(1, (now - t0) / 600); drawOut(); if (morph < 1) morphId = requestAnimationFrame(step); };
    morphId = requestAnimationFrame(step);
  }
  function runFlow() {
    stopFlow(); cancelAnimationFrame(morphId); morph = 1; from = cur; play.textContent = "Stop"; st.step = 3; drawText(); drawModel(); modelBox.classList.add("run");
    let t0 = null;
    const step = (now) => { if (t0 === null) t0 = now; const t = (now - t0) / 4500; if (t < 1) { st.t = t; st.anim = requestAnimationFrame(step); drawOut(); } else { stopFlow(); drawOut(); } };
    st.anim = requestAnimationFrame(step);
  }
  box.addEventListener("click", (e) => {
    const b = e.target.closest("button"), g = e.target.closest(".e-g");
    if (b && b.dataset.role) { st.role = b.dataset.role; st.step = 1; return requery(true); }
    if (b && b.dataset.step !== undefined) { st.step = Number(b.dataset.step); drawText(); return drawModel(); }
    if (b && b.dataset.go) { st.step = Math.max(-1, Math.min(3, st.step + Number(b.dataset.go))); drawText(); return drawModel(); }
    if (b && b.classList.contains("e-play")) { if (st.anim) { stopFlow(); drawOut(); drawModel(); } else runFlow(); return; }
    if (g) { st.step = Number(g.dataset.step); drawText(); drawModel(); }
  });
  slider.addEventListener("input", () => { st.val = slider.value / 100; requery(false); });
  window.addEventListener("resize", drawOut);
  // a link can open a state: model.html#know=exact&step=3
  const hp = new URLSearchParams(location.hash.slice(1));
  if (["exact", "rough"].includes(hp.get("know"))) st.role = hp.get("know");
  if (hp.get("step")) st.step = Math.max(-1, Math.min(3, Number(hp.get("step")) - 1));
  requery(false);
})();
