// LNP particle pictures, drawn in the browser from the simulator's structure model
// (lnp_release/aisaxs/physical_models/lnp_model.py): a sphere of radius R with a
// uniform shell of thickness d, and an interior that is a Gaussian random field with
// the two-component lognormal spectrum G(q), passed through a sigmoid and shifted.
// What is shown is a planar slice of that field: a slice of a 3D Gaussian field is a
// 2D Gaussian field whose spectrum is G integrated along the third axis. One fixed
// noise field is reused for every picture, so a change in the picture comes from the
// parameters only. Densities are relative to the solvent (1).
const LNP = (() => {
  const N = 512;                                              // field pixels per side

  // ---- FFT (radix 2, in place, separate real and imaginary arrays) ----------------
  const rev = new Uint32Array(N), cosT = new Float32Array(N / 2), sinT = new Float32Array(N / 2);
  for (let i = 0, bits = Math.log2(N); i < N; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  for (let i = 0; i < N / 2; i++) { cosT[i] = Math.cos(-2 * Math.PI * i / N); sinT[i] = Math.sin(-2 * Math.PI * i / N); }
  function fft1(re, im, off, stride, inverse) {
    for (let i = 0; i < N; i++) {
      const j = rev[i];
      if (j > i) {
        const a = off + i * stride, b = off + j * stride;
        let t = re[a]; re[a] = re[b]; re[b] = t; t = im[a]; im[a] = im[b]; im[b] = t;
      }
    }
    const sgn = inverse ? -1 : 1;
    for (let len = 2; len <= N; len <<= 1) {
      const half = len >> 1, step = N / len;
      for (let i = 0; i < N; i += len) {
        for (let k = 0; k < half; k++) {
          const wr = cosT[k * step], wi = sgn * sinT[k * step];
          const a = off + (i + k) * stride, b = off + (i + k + half) * stride;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
  }
  function fft2(re, im, inverse) {
    for (let r = 0; r < N; r++) fft1(re, im, r * N, 1, inverse);
    for (let c = 0; c < N; c++) fft1(re, im, c, N, inverse);
  }

  // ---- fixed white noise, transformed once ------------------------------------------
  const noiseF = (() => {
    let s = 0x9e3779b9;
    const rnd = () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const re = new Float32Array(N * N), im = new Float32Array(N * N);
    for (let i = 0; i < N * N; i += 2) {
      const u = Math.max(rnd(), 1e-12), v = rnd(), m = Math.sqrt(-2 * Math.log(u));
      re[i] = m * Math.cos(2 * Math.PI * v); if (i + 1 < N * N) re[i + 1] = m * Math.sin(2 * Math.PI * v);
    }
    fft2(re, im, false);
    return { re, im };
  })();

  // Lognormal density in q with mean m and standard deviation s, as build_gamma.
  function lognorm(q, m, s) {
    const mu = Math.log(m * m / Math.sqrt(m * m + s * s)), sig = Math.sqrt(Math.log(1 + s * s / (m * m)));
    if (q <= 0) return 0;
    const z = (Math.log(q) - mu) / sig;
    return Math.exp(-0.5 * z * z) / (q * sig * Math.sqrt(2 * Math.PI));
  }
  // Spectrum of a planar slice: G integrated along the perpendicular wave number.
  function sliceSpectrum(spec, kmax, bins) {
    const G = (q) => spec.w1 * lognorm(q, spec.m1, spec.s1) + (1 - spec.w1) * lognorm(q, spec.m2, spec.s2);
    const out = new Float32Array(bins), KZ = 1.2, nz = 240, dz = KZ / nz;
    for (let i = 0; i < bins; i++) {
      const k = (i + 0.5) / bins * kmax;
      let s = 0;
      for (let j = 0; j < nz; j++) { const kz = (j + 0.5) * dz; s += G(Math.hypot(k, kz)); }
      out[i] = 2 * s * dz;
    }
    return out;
  }

  // The interior field for a spectrum and pixel size (A): N x N values, relative density.
  const fieldCache = new Map();
  function field(spec, shift, dx) {
    const key = [spec.w1, spec.m1, spec.m2, spec.s1, spec.s2, dx].map((v) => v.toFixed(5)).join();
    let f = fieldCache.get(key);
    if (!f) {
      const kmax = Math.PI / dx * Math.SQRT2, bins = 1024, S2 = sliceSpectrum(spec, kmax, bins);
      const re = new Float32Array(N * N), im = new Float32Array(N * N), dk = 2 * Math.PI / (N * dx);
      for (let r = 0; r < N; r++) {
        const ky = (r < N / 2 ? r : r - N) * dk;
        for (let c = 0; c < N; c++) {
          const kx = (c < N / 2 ? c : c - N) * dk, k = Math.hypot(kx, ky);
          const b = Math.min(bins - 1, Math.floor(k / kmax * bins)), amp = k === 0 ? 0 : Math.sqrt(Math.max(S2[b], 0));
          const o = r * N + c;
          re[o] = noiseF.re[o] * amp; im[o] = noiseF.im[o] * amp;
        }
      }
      fft2(re, im, true);
      let mean = 0, sq = 0;
      for (let i = 0; i < N * N; i++) mean += re[i];
      mean /= N * N;
      for (let i = 0; i < N * N; i++) { re[i] -= mean; sq += re[i] * re[i]; }
      const sd = Math.sqrt(sq / (N * N)) || 1;
      let m2 = 0;
      for (let i = 0; i < N * N; i++) { re[i] = 1 / (1 + Math.exp(-re[i] / sd)); m2 += re[i]; }
      m2 /= N * N;
      for (let i = 0; i < N * N; i++) re[i] -= m2;                 // zero mean; the shift is added when drawing
      f = re;
      if (fieldCache.size > 24) fieldCache.delete(fieldCache.keys().next().value);
      fieldCache.set(key, f);
    }
    return { f, shift };
  }

  // ---- colours: white at the solvent density, navy denser, orange less dense -------
  const NAVY = [31, 78, 121], ACC = [232, 89, 12], SCALE = 0.75;
  function rgb(rho, out, o) {
    let c = (rho - 1) / SCALE;
    const t = Math.min(1, Math.abs(c)), col = c > 0 ? NAVY : ACC;
    out[o] = 255 + (col[0] - 255) * t; out[o + 1] = 255 + (col[1] - 255) * t; out[o + 2] = 255 + (col[2] - 255) * t; out[o + 3] = 255;
  }

  // Draw particles [{x, y, R, d, rhoShell}] (A, centre coordinates) with a given
  // interior field into a canvas showing `width` A. The field is sampled bilinearly,
  // the particle edges are anti-aliased over one pixel.
  function draw(canvas, width, particles, fld, dx) {
    const ratio = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * ratio), h = Math.round(canvas.clientHeight * ratio);
    if (!w || !h) return;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h), px = img.data;
    const s = width / w;                                      // A per canvas pixel
    const f = fld.f, shift = fld.shift, fs = 1 / dx;          // field pixels per A
    px.fill(255);
    for (const p of particles) {
      const Rd = p.R + s, x0 = Math.max(0, Math.floor((p.x - Rd) / s)), x1 = Math.min(w - 1, Math.ceil((p.x + Rd) / s));
      const y0 = Math.max(0, Math.floor((p.y - Rd) / s)), y1 = Math.min(h - 1, Math.ceil((p.y + Rd) / s));
      const Rc = Math.max(p.R - p.d, 0), shellRho = p.rhoShell;
      for (let j = y0; j <= y1; j++) {
        const yA = (j + 0.5) * s - p.y;
        for (let i = x0; i <= x1; i++) {
          const xA = (i + 0.5) * s - p.x, r = Math.hypot(xA, yA);
          if (r > p.R + s) continue;
          const o = 4 * (j * w + i);
          // interior value, bilinear in the field (wrapping)
          let fx = ((i + 0.5) * s) * fs, fy = ((j + 0.5) * s) * fs;
          fx -= Math.floor(fx / N) * N; fy -= Math.floor(fy / N) * N;
          const ix = Math.floor(fx), iy = Math.floor(fy), ax = fx - ix, ay = fy - iy;
          const ix1 = (ix + 1) % N, iy1 = (iy + 1) % N;
          const v = (f[iy * N + ix] * (1 - ax) + f[iy * N + ix1] * ax) * (1 - ay) + (f[iy1 * N + ix] * (1 - ax) + f[iy1 * N + ix1] * ax) * ay;
          const rhoCore = v + 1 + shift;
          const aCore = Math.min(1, Math.max(0, (Rc - r) / s + 0.5));       // coverage of the core
          const aPart = Math.min(1, Math.max(0, (p.R - r) / s + 0.5));      // coverage of the particle
          const rho = aPart <= 0 ? 1 : (rhoCore * aCore + shellRho * (1 - aCore)) * aPart + 1 * (1 - aPart);
          rgb(rho, px, o);
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    return { ctx, w, h, ratio };
  }

  // Scale bar in CSS pixels on top of a drawn canvas.
  function scaleBar(canvas, width, barA, label) {
    const ctx = canvas.getContext("2d"), ratio = window.devicePixelRatio || 1;
    ctx.save(); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const w = canvas.clientWidth, h = canvas.clientHeight, bar = barA / width * w;
    ctx.font = FONT;
    ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(10, h - 30, bar + 20 + ctx.measureText(label).width + 10, 22);
    ctx.fillStyle = C.ink; ctx.fillRect(14, h - 20, bar, 3);
    ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.fillText(label, bar + 20, h - 19);
    ctx.restore();
  }

  // ---- size distribution: lognormal in the radius with mean and cv, truncated ------
  function normCdf(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }
  function normInv(p) {                                     // Acklam
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    p = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
    if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  function radiusQuantile(u, mean, cv, lo, hi) {
    const s2 = Math.log(1 + cv * cv), mu = Math.log(mean) - s2 / 2, sig = Math.sqrt(s2);
    const fa = normCdf((Math.log(lo) - mu) / sig), fb = normCdf((Math.log(hi) - mu) / sig);
    return Math.min(hi, Math.max(lo, Math.exp(mu + sig * normInv(fa + u * (fb - fa)))));
  }
  // Density of the radius distribution on a grid (unnormalized shape).
  function radiusDensity(mean, cv, rs) {
    const s2 = Math.log(1 + cv * cv), mu = Math.log(mean) - s2 / 2, sig = Math.sqrt(s2);
    return rs.map((r) => Math.exp(-0.5 * ((Math.log(r) - mu) / sig) ** 2) / (r * sig));
  }

  // A population: nine particles on a jittered lattice, radii at fixed quantiles of
  // the size distribution, so the picture changes smoothly with the parameters.
  const LATTICE = [[0.39, -0.03, 0.04], [0.94, 0.05, -0.02], [0.17, -0.04, 0.03], [0.06, 0.02, -0.05], [0.61, -0.01, 0.01],
                   [0.28, 0.04, 0.04], [0.83, -0.03, -0.03], [0.50, 0.03, 0.02], [0.72, -0.02, -0.04]];
  function population(width, mean, cv, d, rhoShell, lo, hi) {
    const out = [];
    LATTICE.forEach(([u, jx, jy], i) => {
      const cx = ((i % 3) + 0.5) / 3 + jx, cy = (Math.floor(i / 3) + 0.5) / 3 + jy;
      out.push({ x: cx * width, y: cy * width, R: radiusQuantile(u, mean, cv, lo, hi), d, rhoShell });
    });
    out.sort((a, b) => b.R - a.R);
    const kept = [];
    for (const p of out) { if (kept.every((k) => Math.hypot(k.x - p.x, k.y - p.y) > k.R + p.R + 20)) kept.push(p); }
    return kept;
  }

  return { N, field, draw, scaleBar, radiusQuantile, radiusDensity, population, lognorm };
})();
