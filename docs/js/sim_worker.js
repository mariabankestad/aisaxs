// Runs the cellulose simulator (cellulose_sim.js) off the page's main thread.
// In: { geo, N, fov, seed }. Out: one structure (matrix mask, fibril coverage)
// and the number of fibrils. No scattering curve is computed. The page stops a
// run by terminating the worker.
importScripts("cellulose_sim.js");

onmessage = (e) => {
  const { geo, N, fov, seed } = e.data;
  const t0 = performance.now();
  const r = CelluloseSim.Structure(geo, N, fov, seed, null).realize();
  postMessage({ N, m: r.m, cov: r.cov, fibrils: r.n, ms: performance.now() - t0 }, [r.m.buffer, r.cov.buffer]);
};
