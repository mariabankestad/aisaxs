// The cellulose simulator in JavaScript: a port of simulator_revision/saxs_jax/v2.py
// (structure_spectra with placement "outside" and fibril sizes truncated to
// 0.5-5 nm, as in the paper's banks), for a box smaller than the paper's.
//
// One realization:
//   1. matrix: a Gaussian random field with the Matern spectrum, thresholded so
//      that a fraction volfrac of the box is matrix;
//   2. fibril sizes: lognormal circumradii, truncated to [R_MIN, R_MAX];
//   3. placement: random sequential adsorption of hexagons on the torus, only
//      where a fibril's centre is farther than its radius from the matrix, with
//      an edge gap of at least min_space between fibrils;
//   4. rasterization: the fraction of each pixel a hexagon covers;
//   5. phase 1 = matrix, phase 2 = interstitial (neither matrix nor fibril);
//      their power spectra, radially averaged on the reference q grid.
// Parameters keep their meaning in the paper's 743.5 nm box at 1024 reference
// pixels; fov is the box side relative to that box, N the pixels per side
// (N / fov = 2048 keeps the paper's 0.36 nm pixel).
// The random numbers differ from numpy's, so realizations agree with Python
// in distribution, not one by one. Placement handles candidates one at a time
// in rounds of 4096, where Python checks a round's candidates as a batch.
(function (root) {
  "use strict";
  const NX_REF = 1024, K_SUB = 16, MAX_TRIES = 200, BATCH = 4096;
  const SIGMA_SMOOTH = 2.0, BIN_W = 0.05, SQ3 = Math.sqrt(3);
  const BOX_NM = 743.5, R_MIN_NM = 0.5, R_MAX_NM = 5.0;

  // ---- random numbers (xoshiro128**, seeded by splitmix32) -------------------------
  function rng(seed) {
    let s = seed >>> 0;
    const sm = () => { s = (s + 0x9e3779b9) >>> 0; let z = s; z = Math.imul(z ^ (z >>> 16), 0x85ebca6b); z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35); return (z ^ (z >>> 16)) >>> 0; };
    let a = sm(), b = sm(), c = sm(), d = sm();
    const next = () => {
      const r = Math.imul(((Math.imul(b, 5) << 7) | (Math.imul(b, 5) >>> 25)), 9) >>> 0;
      const t = b << 9;
      c ^= a; d ^= b; b ^= c; a ^= d; c ^= t; d = (d << 11) | (d >>> 21);
      return r;
    };
    const random = () => (next() * 4294967296 + next()) / 18446744073709551616;   // [0, 1)
    let spare = null;
    const normal = () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u; do { u = random(); } while (u === 0);
      const r = Math.sqrt(-2 * Math.log(u)), t = 2 * Math.PI * random();
      spare = r * Math.sin(t);
      return r * Math.cos(t);
    };
    return { random, normal };
  }

  // ---- FFT (radix 2, in place, separate real and imaginary arrays) -----------------
  const fftPlans = {};
  function plan(n) {
    if (fftPlans[n]) return fftPlans[n];
    const rev = new Uint32Array(n), bits = Math.log2(n);
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    return (fftPlans[n] = { rev, cos, sin });
  }
  function fft1(re, im, off, stride, n, inv, P, tr, ti) {
    for (let i = 0; i < n; i++) { const j = P.rev[i], o = off + j * stride; tr[i] = re[o]; ti[i] = im[o]; }
    const sg = inv ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let s = 0; s < n; s += size) {
        for (let k = 0; k < half; k++) {
          const wr = P.cos[k * step], wi = sg * P.sin[k * step];
          const a = s + k, b = a + half;
          const xr = tr[b] * wr - ti[b] * wi, xi = tr[b] * wi + ti[b] * wr;
          tr[b] = tr[a] - xr; ti[b] = ti[a] - xi; tr[a] += xr; ti[a] += xi;
        }
      }
    }
    const f = inv ? 1 / n : 1;
    for (let i = 0; i < n; i++) { const o = off + i * stride; re[o] = tr[i] * f; im[o] = ti[i] * f; }
  }
  function fft2(re, im, N, inv) {
    const P = plan(N), tr = new Float64Array(N), ti = new Float64Array(N);
    for (let i = 0; i < N; i++) fft1(re, im, i * N, 1, N, inv, P, tr, ti);   // rows
    for (let j = 0; j < N; j++) fft1(re, im, j, N, N, inv, P, tr, ti);       // columns
  }

  // ---- matrix ----------------------------------------------------------------------
  function maternSqrt(N, kappaSc, fov) {
    const kappa = 1.5e-2 * kappaSc * NX_REF * fov / N, k2 = kappa * kappa;
    const f = new Float64Array(N);
    for (let i = 0; i < N; i++) f[i] = (i < (N + 1) / 2 ? i : i - N) / N;   // np.fft.fftfreq
    const G = new Float64Array(N * N), gmax = 1 / (k2 * k2 * k2);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const g = 1 / (k2 + f[i] * f[i] + f[j] * f[j]) ** 3 / gmax;
      G[i * N + j] = g < 1e-6 ? 0 : Math.sqrt(g);
    }
    return G;
  }
  // Two independent fields from one complex FFT pair (the spectrum is real and
  // symmetric, so the real and imaginary parts stay separate).
  function fieldPair(N, sqrtG, R) {
    const re = new Float64Array(N * N), im = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) { re[i] = R.normal(); im[i] = R.normal(); }
    fft2(re, im, N, false);
    for (let i = 0; i < N * N; i++) { re[i] *= sqrtG[i]; im[i] *= sqrtG[i]; }
    fft2(re, im, N, true);
    return [re, im];
  }
  function quantile(g, p) {                                    // numpy's default (linear)
    const s = Float64Array.from(g).sort(), h = p * (s.length - 1), lo = Math.floor(h);
    return lo + 1 < s.length ? s[lo] + (h - lo) * (s[lo + 1] - s[lo]) : s[lo];
  }

  // Exact Euclidean distance to the matrix (Felzenszwalb and Huttenlocher), on
  // the box padded by wrapping, as distance_to_matrix in v2.py.
  function edt1(f, n, d, v, z) {
    let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < n; q++) {
      let p = v[k], s = ((f[q] + q * q) - (f[p] + p * p)) / (2 * q - 2 * p);
      while (s <= z[k]) { k--; p = v[k]; s = ((f[q] + q * q) - (f[p] + p * p)) / (2 * q - 2 * p); }
      k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; const p = v[k]; d[q] = (q - p) * (q - p) + f[p]; }
  }
  function distanceToMatrix(m, N, pad) {
    const M = N + 2 * pad, BIG = 1e12, g = new Float64Array(M * M);
    for (let a = 0; a < M; a++) {
      const i = ((a - pad) % N + N) % N;
      for (let b = 0; b < M; b++) { const j = ((b - pad) % N + N) % N; g[a * M + b] = m[i * N + j] ? 0 : BIG; }
    }
    const f = new Float64Array(M), d = new Float64Array(M), v = new Int32Array(M), z = new Float64Array(M + 1);
    for (let b = 0; b < M; b++) {                              // columns
      for (let a = 0; a < M; a++) f[a] = g[a * M + b];
      edt1(f, M, d, v, z);
      for (let a = 0; a < M; a++) g[a * M + b] = d[a];
    }
    const out = new Float32Array(N * N);
    for (let a = pad; a < pad + N; a++) {                      // rows, keep the centre
      for (let b = 0; b < M; b++) f[b] = g[a * M + b];
      edt1(f, M, d, v, z);
      for (let b = pad; b < pad + N; b++) out[(a - pad) * N + (b - pad)] = Math.sqrt(d[b]);
    }
    return out;
  }

  // ---- fibril sizes --------------------------------------------------------------
  function drawSizes(n, alpha0, cv, R, floor, cap) {
    const out = new Float64Array(n);
    if (cv <= 0) { const a = Math.min(Math.max(alpha0, floor), cap); out.fill(a); return { sizes: out, amax: a }; }
    const s = Math.sqrt(Math.log1p(cv * cv)), mu = Math.log(alpha0) - 0.5 * s * s;
    const zlo = floor > 0 ? (Math.log(floor) - mu) / s : -Infinity, zhi = (Math.log(cap) - mu) / s;
    if (zlo >= zhi) { const a = zlo > 0 ? floor : cap; out.fill(a); return { sizes: out, amax: a }; }
    for (let i = 0; i < n; i++) { let z; do { z = R.normal(); } while (z > zhi || z < zlo); out[i] = Math.exp(mu + s * z); }
    return { sizes: out, amax: Math.exp(mu + zhi * s) };
  }

  // ---- placement -------------------------------------------------------------------
  function placeRods(sizes, gap, amax, R, allowed) {
    const P = sizes.length, reach = 2 * amax + gap;
    const nc = Math.max(1, Math.min(1024, Math.floor(1 / reach)));
    const head = new Int32Array(nc * nc).fill(-1), next = new Int32Array(P);
    const X = new Float64Array(P), Y = new Float64Array(P), Rad = new Float64Array(P);
    let n = 0;
    const cell = (v) => Math.min(nc - 1, Math.floor(v * nc));
    function free(x, y, r) {
      const ci = cell(x), cj = cell(y), span = nc < 3 ? 0 : 1;
      const cells = nc < 3 ? [...Array(nc * nc).keys()] : null;
      const check = (c) => {
        for (let k = head[c]; k >= 0; k = next[k]) {
          let dx = x - X[k], dy = y - Y[k];
          dx -= Math.round(dx); dy -= Math.round(dy);
          const lim = r + Rad[k] + gap;
          if (dx * dx + dy * dy < lim * lim) return false;
        }
        return true;
      };
      if (cells) { for (const c of cells) if (!check(c)) return false; return true; }
      for (let a = -span; a <= span; a++) for (let b = -span; b <= span; b++) {
        if (!check(((ci + a + nc) % nc) * nc + ((cj + b + nc) % nc))) return false;
      }
      return true;
    }
    let pending = Array.from({ length: P }, (_, i) => i);
    const tries = new Int32Array(P);
    while (pending.length) {
      const cand = pending.slice(0, BATCH), placed = new Set();
      for (const idx of cand) {
        const r = sizes[idx];
        let x = R.random(), y = R.random(), ok = allowed(x, y, r);
        for (let t = 0; t < 50 && !ok; t++) { x = R.random(); y = R.random(); ok = allowed(x, y, r); }
        if (ok && free(x, y, r)) {
          X[n] = x; Y[n] = y; Rad[n] = r;
          const c = cell(x) * nc + cell(y); next[n] = head[c]; head[c] = n; n++;
          placed.add(idx);
        } else tries[idx]++;
      }
      pending = pending.filter((i) => !placed.has(i) && tries[i] < MAX_TRIES);
    }
    return { x: X.subarray(0, n), y: Y.subarray(0, n), r: Rad.subarray(0, n), n };
  }

  // ---- rasterizer: fraction of each pixel covered by hexagons ----------------------
  // Circumradius L, vertices at (+-L, 0) along the first array axis; pixel (i, j)
  // covers [i, i+1) x [j, j+1); K_SUB sub-columns in x, exact in y.
  function rasterize(rods, N) {
    const cov = new Float32Array(N * N);
    for (let k = 0; k < rods.n; k++) {
      const cx = rods.x[k] * N, cy = rods.y[k] * N, L = rods.r[k] * N;
      const ncol = Math.ceil(2 * L) + 2, i0 = Math.floor(cx - L);
      for (let c = 0; c < ncol; c++) {
        const ci = i0 + c, row = (((ci % N) + N) % N) * N;
        for (let s = 0; s < K_SUB; s++) {
          const dx = Math.abs(ci + (s + 0.5) / K_SUB - cx);
          const h = dx <= L / 2 ? SQ3 / 2 * L : Math.max(SQ3 * (L - dx), 0);
          if (h <= 0) continue;
          const yb = cy - h, yt = cy + h;
          for (let rj = Math.floor(yb); rj < yt; rj++) {
            const f = Math.min(rj + 1, yt) - Math.max(rj, yb);
            if (f > 0) cov[row + (((rj % N) + N) % N)] += f / K_SUB;
          }
        }
      }
    }
    for (let i = 0; i < N * N; i++) if (cov[i] > 1) cov[i] = 1;
    return cov;
  }

  // ---- radial average on given reference q (Fourier pixels of the reference box) ----
  function radialTables(qRef, fov, N) {
    const sig = SIGMA_SMOOTH * fov, q = qRef.map((v) => v * fov);
    const R = Math.ceil(Math.max(...q) + 6 * sig) + 1;
    if (2 * R + 1 > N) throw new Error(`N=${N} too small for this box`);
    const W = 2 * R + 1, bin = new Int32Array(W * W).fill(-1);
    const nb = Math.floor(R / BIN_W) + 1, cnt = new Float64Array(nb), qsum = new Float64Array(nb);
    const idx = new Int32Array(W * W);
    for (let a = -R; a <= R; a++) for (let b = -R; b <= R; b++) {
      const Q = Math.hypot(a, b), o = (a + R) * W + (b + R);
      idx[o] = (((a % N) + N) % N) * N + (((b % N) + N) % N);
      if (Q > 0 && Q <= R) { const k = Math.floor(Q / BIN_W); bin[o] = k; cnt[k]++; qsum[k] += Q; }
    }
    const A = q.map((qq) => {
      const row = new Float64Array(nb); let norm = 0;
      for (let k = 0; k < nb; k++) {
        if (!cnt[k]) continue;
        const Qb = qsum[k] / cnt[k]; row[k] = Math.exp(-0.5 * ((Qb - qq) / sig) ** 2) / Qb; norm += row[k] * cnt[k];
      }
      for (let k = 0; k < nb; k++) row[k] /= norm;
      return row;
    });
    return { bin, idx, nb, A };
  }
  function radialAverage(P2, T) {
    const sums = new Float64Array(T.nb);
    for (let o = 0; o < T.bin.length; o++) if (T.bin[o] >= 0) sums[T.bin[o]] += P2[T.idx[o]];
    return T.A.map((row) => { let s = 0; for (let k = 0; k < T.nb; k++) s += row[k] * sums[k]; return s; });
  }

  // ---- a structure: realizations one at a time ---------------------------------------
  // geo = [tube_scaling, kappa_sc, volfrac_grf, no_rods_per_unit, min_space, size_cv]
  // qRef null: structures only, no scattering curve (skips the FFT of the phases).
  function Structure(geo, N, fov, seed, qRef) {
    const [aF, kappaSc, vf, nR, minSpace, cv] = geo;
    const R = rng(seed), sqrtG = maternSqrt(N, kappaSc, fov), T = qRef ? radialTables(qRef, fov, N) : null;
    const alpha0 = 0.005 / aF / fov, Pbox = Math.round(nR * aF * fov * fov), gap = minSpace / NX_REF / fov;
    const floor = R_MIN_NM / BOX_NM / fov, cap = R_MAX_NM / BOX_NM / fov;
    const p = N / (NX_REF * fov), scale = p ** -4 / (fov * fov);
    const acc = T ? [new Float32Array(N * N), new Float32Array(N * N), new Float32Array(N * N)] : null;
    let spare = null, count = 0;
    function realize() {
      let g;
      if (spare) { g = spare; spare = null; } else { const pr = fieldPair(N, sqrtG, R); g = pr[0]; spare = pr[1]; }
      const thr = quantile(g, vf), m = new Uint8Array(N * N);
      for (let i = 0; i < N * N; i++) m[i] = g[i] <= thr ? 1 : 0;
      const { sizes, amax } = drawSizes(Pbox, alpha0, cv, R, floor, cap);
      const pad = Math.ceil(amax * N) + 3, dist = distanceToMatrix(m, N, pad);
      let avail = 0; for (let i = 0; i < N * N; i++) if (dist[i] > alpha0 * N) avail++;
      const P = Math.round(Pbox * avail / (N * N));
      const allowed = (x, y, r) => dist[Math.min(Math.floor(x * N), N - 1) * N + Math.min(Math.floor(y * N), N - 1)] > r * N;
      const rods = placeRods(sizes.subarray(0, P), gap, amax, R, allowed);
      const cov = rasterize(rods, N);
      let f1 = 0, f3 = 0;
      for (let i = 0; i < N * N; i++) { f1 += m[i]; f3 += (1 - m[i]) * cov[i]; }
      count++;
      const out = { m, cov, n: rods.n, requested: P, fracMatrix: f1 / (N * N), fracFibril: f3 / (N * N) };
      if (!T) return out;
      // z = phase1 + i phase2; F1, F2 from Z(k) and Z(-k)
      const re = new Float64Array(N * N), im = new Float64Array(N * N);
      for (let i = 0; i < N * N; i++) { re[i] = m[i]; im[i] = (1 - m[i]) * (1 - cov[i]); }
      fft2(re, im, N, false);
      for (let i = 0; i < N; i++) {
        const ni = (N - i) % N;
        for (let j = 0; j < N; j++) {
          const o = i * N + j, o2 = ni * N + (N - j) % N;
          const ar = re[o], ai = im[o], br = re[o2], bi = -im[o2];   // Z(k), conj Z(-k)
          const F1r = 0.5 * (ar + br), F1i = 0.5 * (ai + bi);
          const F2r = 0.5 * (ai - bi), F2i = -0.5 * (ar - br);
          acc[0][o] += F1r * F1r + F1i * F1i;
          acc[1][o] += F2r * F2r + F2i * F2i;
          acc[2][o] += F1r * F2r + F1i * F2i;
        }
      }
      return out;
    }
    // mean cross spectra [S11, S22, S12] so far, in the paper's units
    function spectra() { return acc.map((a) => radialAverage(a, T).map((v) => v / count * scale)); }
    return { realize, spectra, get count() { return count; } };
  }

  root.CelluloseSim = { Structure, rng, fft2, distanceToMatrix, quantile, BOX_NM };
})(typeof self !== "undefined" ? self : this);
