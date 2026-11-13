// Homepage: a SAXS measurement in one picture. Spheres in a liquid, an X-ray beam,
// the scattered intensity on the detector (side view and face on) and the curve.
// Physics: polydisperse homogeneous spheres, I(q) = sum_R n(R) R^6 F(qR)^2, with a
// Gaussian number distribution of radii. The scene is a sketch; angles are exaggerated.
(function () {
  const root = document.getElementById("saxs-demo");
  if (!root) return;

  // ---- physics ---------------------------------------------------------------------
  const QMAX = 1.0;                // nm^-1 at the detector edge
  const NQ = 400;
  const qs = Float64Array.from({ length: NQ }, (_, i) => (i + 0.5) * QMAX / NQ);
  const state = { R: 20, s: 0.10 };

  function intensity(R, s) {
    const out = new Float64Array(NQ);
    const nodes = s > 0.001 ? 61 : 1;
    for (let k = 0; k < nodes; k++) {
      const z = nodes === 1 ? 0 : -3 + 6 * k / (nodes - 1);
      const r = R * (1 + s * z);
      if (r < 0.5) continue;
      const w = Math.exp(-0.5 * z * z) * r ** 6;
      for (let i = 0; i < NQ; i++) {
        const x = qs[i] * r;
        const F = 3 * (Math.sin(x) - x * Math.cos(x)) / (x * x * x);
        out[i] += w * F * F;
      }
    }
    const top = out[0];
    for (let i = 0; i < NQ; i++) out[i] /= top;
    return out;
  }

  // ---- fixed randomness, so the sliders morph the picture instead of re-drawing it --
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const zoomParts = [];                            // positions in nm from the zoom centre
  for (let t = 0; t < 4000 && zoomParts.length < 24; t++) {
    const a = 2 * Math.PI * rand(), r = Math.sqrt(rand()) * 150, z = Math.max(-2, Math.min(2, gauss()));
    const p = { x: r * Math.cos(a), y: r * Math.sin(a), z };
    if (zoomParts.every((o) => Math.hypot(o.x - p.x, o.y - p.y) > 50)) zoomParts.push(p);
  }
  const N = 256;
  const grain = Float32Array.from({ length: N * N }, () => Math.max(0.5, 1 + 0.15 * gauss()));

  // ---- detector colours (as in the video): dark where nothing arrives, white where most does
  const STOPS = ["#0e1f33", "#1f4e79", "#3a78b8", "#7ea6cf", "#dbe6f2", "#ffffff"].map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
  function cmap(v) {
    const t = (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i;
    return STOPS[i].map((c, k) => c + (STOPS[i + 1][k] - c) * f);
  }

  // the face-on detector image, drawn once per change into an offscreen canvas
  const det = document.createElement("canvas");
  det.width = det.height = N;
  function renderDetector(I) {
    const ctx = det.getContext("2d"), img = ctx.createImageData(N, N);
    // log scale, stretched between the brightest ring region and the edge
    // (the centre saturates to white, as on a real detector image)
    const i1 = Math.min(NQ - 1, Math.floor((4.49 / state.R) / QMAX * NQ));   // first zero of a sphere
    let peak = 0;
    for (let i = i1; i < NQ; i++) peak = Math.max(peak, I[i]);
    const hi = Math.log10(peak) + 0.35, lo = Math.log10(Math.min(...I)) - 0.4;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const r = Math.hypot(x - N / 2 + 0.5, y - N / 2 + 0.5) / (N / 2);
        const j = (y * N + x) * 4;
        let v = 0;
        if (r < 1) {
          const qi = Math.min(NQ - 1, Math.floor(r * NQ));
          v = (Math.log10(I[qi] * grain[y * N + x] + 1e-12) - lo) / (hi - lo);
          v = Math.max(0, v) ** 1.3;
        }
        const c = r < 0.045 ? [40, 46, 54] : cmap(v);
        img.data[j] = c[0]; img.data[j + 1] = c[1]; img.data[j + 2] = c[2]; img.data[j + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // ---- the scene (side view) -----------------------------------------------------------
  const scene = root.querySelector("canvas.scene");
  const VW = 1000, VH = 380;                       // virtual units
  const BY = 220;                                  // beam height
  const CAP = { x: 360, r: 17, top: 112, bot: 330 };    // a thin glass capillary
  const DET = { x: 770, w: 96, hl: 210, hr: 300 }; // left edge shorter: perspective
  const EXIT = { x: CAP.x + CAP.r, y: BY };
  const DC = { x: DET.x + DET.w / 2, y: BY };
  const ZOOM = { x: 196, y: 112, r: 84, nmPerPx: 1.8 };  // about 300 nm across
  const SQ = { x: CAP.x - 5, y: BY - 5, s: 10 };

  function sphere(ctx, x, y, r) {
    const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r);
    g.addColorStop(0, "#9fbfdf"); g.addColorStop(0.55, C.blue); g.addColorStop(1, C.navy);
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
  }
  const radius = (p) => Math.max(0.3, state.R * (1 + state.s * p.z));

  function drawScene() {
    const { ctx, w, h } = fitCanvas(scene);
    const k = Math.min(w / VW, h / VH);
    ctx.save();
    ctx.translate((w - VW * k) / 2, (h - VH * k) / 2);
    ctx.scale(k, k);
    ctx.font = "15px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";

    // incoming beam
    ctx.strokeStyle = C.accent; ctx.lineWidth = 6; ctx.lineCap = "butt";
    ctx.beginPath(); ctx.moveTo(20, BY); ctx.lineTo(CAP.x - CAP.r, BY); ctx.stroke();
    ctx.fillStyle = C.accent; ctx.font = "600 17px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
    ctx.fillText("X-ray beam", 24, BY - 14);

    // the sample: a thin glass capillary filled with the liquid
    const { x: cx, r: cr, top, bot } = CAP;
    const glass = ctx.createLinearGradient(cx - cr, 0, cx + cr, 0);
    glass.addColorStop(0, "#c9d8e8"); glass.addColorStop(0.3, "#eef4fa"); glass.addColorStop(0.55, "#dbe7f3"); glass.addColorStop(1, "#b9cde2");
    ctx.fillStyle = glass;
    ctx.beginPath(); ctx.moveTo(cx - cr, top); ctx.lineTo(cx - cr, bot); ctx.ellipse(cx, bot, cr, cr * 0.32, 0, Math.PI, 0, true); ctx.lineTo(cx + cr, top); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "#8a939c"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx - cr, top); ctx.lineTo(cx - cr, bot); ctx.ellipse(cx, bot, cr, cr * 0.32, 0, Math.PI, 0, true); ctx.lineTo(cx + cr, top); ctx.stroke();
    ctx.fillStyle = "#f5f8fb"; ctx.beginPath(); ctx.ellipse(cx, top, cr, cr * 0.32, 0, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 2;               // glass highlight
    ctx.beginPath(); ctx.moveTo(cx - cr * 0.45, top + 12); ctx.lineTo(cx - cr * 0.45, bot - 10); ctx.stroke();
    // where the beam is inside the liquid
    const glow = ctx.createLinearGradient(0, BY - 9, 0, BY + 9);
    glow.addColorStop(0, "rgba(232,89,12,0)"); glow.addColorStop(0.5, "rgba(232,89,12,0.75)"); glow.addColorStop(1, "rgba(232,89,12,0)");
    ctx.fillStyle = glow; ctx.fillRect(cx - cr, BY - 9, 2 * cr, 18);
    ctx.fillStyle = C.muted; ctx.font = "15px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
    ctx.textAlign = "center"; ctx.fillText("sample: particles in a liquid,", cx, bot + 30); ctx.fillText("about 1 mm wide", cx, bot + 48);

    // zoom: a tiny spot where the beam passes, magnified
    ctx.strokeStyle = C.navy; ctx.lineWidth = 1.2;
    ctx.strokeRect(SQ.x, SQ.y, SQ.s, SQ.s);
    ctx.strokeStyle = "rgba(31, 78, 121, 0.45)";
    ctx.beginPath(); ctx.moveTo(SQ.x, SQ.y + SQ.s); ctx.lineTo(ZOOM.x + ZOOM.r * Math.cos(0.9), ZOOM.y + ZOOM.r * Math.sin(0.9)); ctx.moveTo(SQ.x + SQ.s, SQ.y); ctx.lineTo(ZOOM.x + ZOOM.r * Math.cos(-0.35), ZOOM.y + ZOOM.r * Math.sin(-0.35)); ctx.stroke();
    ctx.save();
    ctx.beginPath(); ctx.arc(ZOOM.x, ZOOM.y, ZOOM.r, 0, 2 * Math.PI); ctx.fillStyle = "#f4f8fc"; ctx.fill(); ctx.clip();
    for (const p of zoomParts) sphere(ctx, ZOOM.x + p.x / ZOOM.nmPerPx, ZOOM.y + p.y / ZOOM.nmPerPx, radius(p) / ZOOM.nmPerPx);
    ctx.restore();
    ctx.strokeStyle = C.navy; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(ZOOM.x, ZOOM.y, ZOOM.r, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "13px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
    ctx.fillText("inside the beam, about 300 nm across", ZOOM.x, ZOOM.y - ZOOM.r - 10);
    const bar = 50 / ZOOM.nmPerPx;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    const bx = ZOOM.x - ZOOM.r - 18 - bar;
    ctx.beginPath(); ctx.moveTo(bx, ZOOM.y + 40); ctx.lineTo(bx + bar, ZOOM.y + 40); ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.fillText("50 nm", bx + bar / 2, ZOOM.y + 58);

    // scattered X-rays: a cone to the first bright ring after the centre
    const ring = Math.min(1, 5.76 / state.R / QMAX);    // first ring of a sphere, qR = 5.76
    const hc = (DET.hl + DET.hr) / 2;
    if (ring) {
      const dy2 = ring * hc / 2;
      ctx.fillStyle = "rgba(232, 89, 12, 0.10)";
      ctx.beginPath(); ctx.moveTo(EXIT.x, EXIT.y); ctx.lineTo(DC.x, DC.y - dy2); ctx.lineTo(DC.x, DC.y + dy2); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "rgba(232, 89, 12, 0.55)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(EXIT.x, EXIT.y); ctx.lineTo(DC.x, DC.y - dy2); ctx.moveTo(EXIT.x, EXIT.y); ctx.lineTo(DC.x, DC.y + dy2); ctx.stroke();
    }

    // detector in perspective: the face-on image drawn column by column
    const cols = 96;
    for (let c = 0; c < cols; c++) {
      const u0 = c / cols, hh = DET.hl + (DET.hr - DET.hl) * (u0 + 0.5 / cols);
      ctx.drawImage(det, Math.floor(u0 * N), 0, Math.max(1, N / cols), N, DET.x + u0 * DET.w, BY - hh / 2, DET.w / cols + 0.6, hh);
    }
    ctx.strokeStyle = "#0e1f33"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(DET.x, BY - DET.hl / 2); ctx.lineTo(DET.x + DET.w, BY - DET.hr / 2); ctx.lineTo(DET.x + DET.w, BY + DET.hr / 2); ctx.lineTo(DET.x, BY + DET.hl / 2); ctx.closePath(); ctx.stroke();

    // transmitted beam, on top, ending at the centre of the detector
    ctx.strokeStyle = C.accent; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(EXIT.x, EXIT.y); ctx.lineTo(DC.x - 4, DC.y); ctx.stroke();
    ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(DC.x, DC.y, 4, 0, 2 * Math.PI); ctx.fill();

    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "15px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
    ctx.fillText("detector", DC.x, BY + DET.hr / 2 + 26);
    ctx.font = "italic 13px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"; ctx.textAlign = "left";
    ctx.fillText("sketch, angles exaggerated", 20, VH - 6);
    ctx.restore();
  }

  // ---- the detector face on, and the curve -----------------------------------------------
  function drawFace() {
    const cv = root.querySelector("canvas.face");
    const { ctx, w, h } = fitCanvas(cv);
    const s = Math.min(w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(det, (w - s) / 2, (h - s) / 2, s, s);
  }

  function drawCurve() {
    const cv = root.querySelector("canvas.iq");
    const { ctx, w, h } = fitCanvas(cv);
    const L = 44, R = 20, T = 10, B = 34;
    const X = (q) => L + (q / QMAX) * (w - L - R);
    const Y = (v) => T + (-v / 6) * (h - T - B);     // v = log10 I in [-6, 0]
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.fillStyle = C.muted; ctx.textAlign = "right";
    for (let v = 0; v >= -6; v -= 2) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w - R, Y(v)); ctx.stroke(); ctx.fillText(String(v), L - 6, Y(v) + 4); }
    ctx.textAlign = "center";
    for (let q = 0; q <= QMAX + 1e-9; q += 0.25) ctx.fillText(q.toFixed(2), X(q), h - B + 16);
    ctx.fillText("q (nm⁻¹)", (L + w - R) / 2, h - 4);
    ctx.save(); ctx.translate(12, (T + h - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("log₁₀ intensity", 0, 0); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(L, T, w - L - R, h - T - B); ctx.clip();
    ctx.strokeStyle = C.navy; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < NQ; i++) { const v = Math.max(-6.5, Math.log10(state.I[i] + 1e-12)); i ? ctx.lineTo(X(qs[i]), Y(v)) : ctx.moveTo(X(qs[i]), Y(v)); }
    ctx.stroke(); ctx.restore();
  }

  // ---- controls ------------------------------------------------------------------------
  const update = throttled(() => {
    state.I = intensity(state.R, state.s);
    renderDetector(state.I);
    drawScene(); drawFace(); drawCurve();
  });
  function slider(label, min, max, step, value, fmt, set) {
    const row = document.createElement("label");
    row.className = "ctrl";
    row.innerHTML = `<span class="ctrl-name">${label}</span><output></output><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
    const input = row.querySelector("input"), out = row.querySelector("output");
    const show = () => { out.textContent = fmt(Number(input.value)); };
    input.addEventListener("input", () => { set(Number(input.value)); show(); update(); });
    show();
    root.querySelector(".demo-controls").append(row);
  }
  slider("Mean radius", 8, 40, 0.5, state.R, (v) => `${v.toFixed(1)} nm`, (v) => { state.R = v; });
  slider("Size spread", 0, 40, 1, state.s * 100, (v) => `${v} %`, (v) => { state.s = v / 100; });
  window.addEventListener("resize", update);
  update();
})();
