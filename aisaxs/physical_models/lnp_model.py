import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict
from functools import partial
import jax
import jax.scipy as jsp
from jax import lax


# =========================
#  Globala grid-parametrar
# =========================

NUMBER_OF_VOXELS      = int(512)   # 3D GRF-grid (N)
NUMBER_OF_VOXELS_FULL = int(512*2)  # 2D SAXS-plane (Nf), >= N och jämnt
DELTA_X               = 5.0   # Å
SIGMA_SMOOTH_Q         = 0.5
ALPHA                   = 1.0
K_NONZERO = 1024
# =========================
#  Hjälpfunktioner
# =========================

def fft_2d_fold(A: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-version av MATLAB fft_2D_fold.
    Input:  A[N, N]
    Output: A_fold[N/2+1, N/2+1]
    """
    N = A.shape[0]
    assert A.shape[0] == A.shape[1], "A måste vara kvadratisk"
    assert N % 2 == 0, "N måste vara jämnt"

    # Första vikningen över rader
    rows_left  = jnp.arange(1, N // 2)          # 1..N/2-1
    rows_right = jnp.arange(N - 1, N // 2, -1)  # N-1..N/2+1
    A_fold1 = A.at[rows_left, :].add(A[rows_right, :])
    A_fold1 = A_fold1[: N // 2 + 1, :]

    # Andra vikningen över kolumner
    cols_left  = jnp.arange(1, N // 2)
    cols_right = jnp.arange(N - 1, N // 2, -1)
    A_fold2 = A_fold1.at[:, cols_left].add(A_fold1[:, cols_right])
    A_fold2 = A_fold2[:, : N // 2 + 1]

    return A_fold2


def build_realspace_grid() -> jnp.ndarray:
    """
    3D real-space squared radius D2 [N, N, N].
    Motsvarar X^2 + Y^2 + Z^2 i MATLAB.
    """
    x_1d = jnp.arange(-NUMBER_OF_VOXELS // 2, NUMBER_OF_VOXELS // 2, dtype=jnp.float32) * float(DELTA_X)
    X = x_1d.reshape(NUMBER_OF_VOXELS, 1, 1)
    Y = x_1d.reshape(1, NUMBER_OF_VOXELS, 1)
    Z = x_1d.reshape(1, 1, NUMBER_OF_VOXELS)
    D2 = X**2 + Y**2 + Z**2
    return D2.astype(jnp.float32)


def build_Q_folded_and_indicator(
    Nf: int,
    delta_q: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Bygger:
      - Q_folded: flatten av Q på foldade domänen [ (Nf/2+1)^2 ]
      - indicator_folded: flatten av fft_2D_fold(ones(Nf,Nf))
    som i compute_saxs_curve_2d.m.
    """
    q1d = jnp.arange(-Nf // 2, Nf // 2, dtype=jnp.float32) * delta_q
    QX = q1d.reshape(Nf, 1)
    QY = q1d.reshape(1, Nf)
    Q  = jnp.sqrt(QX**2 + QY**2)  # [Nf, Nf]

    Q_folded = Q[: Nf // 2 + 1, : Nf // 2 + 1].reshape(-1)  # [M_fold]

    ones_full = jnp.ones((Nf, Nf), dtype=jnp.float32)
    indicator_folded = fft_2d_fold(ones_full).reshape(-1)   # [M_fold]

    return Q_folded.astype(jnp.float32), indicator_folded.astype(jnp.float32)


def build_gaussian_fft_kernel(N: int, sigma: float = 1.0) -> jnp.ndarray:
    """
    Förskapar G_fft för 2D Gauss-filtrering via FFT på en N x N-bild.
    Används tillsammans med fftshift(fftn(image)).
    """
    k  = jnp.arange(-N // 2, N // 2, dtype=jnp.float32) * (2.0 * jnp.pi / N)
    KX = k.reshape(N, 1)
    KY = k.reshape(1, N)
    K2 = KX**2 + KY**2
    G_fft = jnp.exp(-0.5 * (sigma**2) * K2)
    return G_fft.astype(jnp.complex64)


def gaussian_filter_fft_with_kernel(
    image: jnp.ndarray,
    G_fft: jnp.ndarray,
) -> jnp.ndarray:
    """
    2D Gauss-smoothing via FFT med förskapad G_fft.
    Motsvarar MATLABs imgaussfilt på en 2D-bild.
    """
    F = jnp.fft.fftshift(jnp.fft.fftn(image))
    F_smooth = F * G_fft
    smoothed = jnp.fft.ifftn(jnp.fft.ifftshift(F_smooth)).real
    return smoothed.astype(jnp.float32)


def build_Q3(N = None)-> jnp.ndarray:
    if N is None:
        N = int(NUMBER_OF_VOXELS)
    delta_q = 2.0 * jnp.pi / (DELTA_X * N)

    q1d = jnp.arange(-N // 2, N // 2, dtype=jnp.float32) * delta_q
    QX = q1d.reshape(N, 1, 1)
    QY = q1d.reshape(1, N, 1)
    QZ = q1d.reshape(1, 1, N)
    Q3 = jnp.sqrt(QX**2 + QY**2 + QZ**2)
    return Q3




# =========================
#  GRF + core-shell i 3D
# =========================

def simulate_one_core_shell(
    key,
    R: float,
    D2: jnp.ndarray,
    GAMMA: jnp.ndarray,
    number_of_voxels_full: int,
    G_fft_2d: jnp.ndarray,
    *,
    rho_solvent: float = 1.0,
    shell_thickness: float = 20.0,
    GRF_shift: float = 0.0,
    relative_rho_shell: float = 1.0,
) -> jnp.ndarray:
    """
    En GRF-realisation:
      - genererar 3D GRF
      - bygger core+shell-partikel
      - projektion till 2D, smoothing, padding
      - returnerar |FFT(RHO_2D_padded)|^2 på [Nf, Nf]
    """
    N  = D2.shape[0]
    Nf = int(number_of_voxels_full)
    assert Nf >= N and (Nf - N) % 2 == 0, "Nf måste vara >= N och ge heltals-padding."

    # 3D GRF
    noise = jax.random.normal(key, D2.shape, dtype=jnp.float32)
    FW    = jnp.fft.fftn(noise)       # complex64
    GRF   = jnp.fft.ifftn(FW * GAMMA).real

    # Logit-GRF (use_logit_GRF = true)
    GRF = GRF - GRF.mean()
    GRF = GRF / GRF.std()
    GRF = jax.nn.sigmoid(ALPHA * GRF)
    GRF = (GRF - GRF.mean()) + 1 + GRF_shift

    # core + shell
    RHO = jnp.full_like(GRF, rho_solvent, dtype=jnp.float32)

    idx_shell = (D2 <= R**2)
    shell_val = relative_rho_shell * GRF.mean()
    RHO = jnp.where(idx_shell, shell_val, RHO)

    R_core   = R - shell_thickness
    idx_core = (D2 <= R_core**2)
    RHO = jnp.where(idx_core, GRF, RHO)

    # 2D-projektion (sum över z), smoothing i 2D, padding
    RHO_2d = RHO.sum(axis=-1)                     # [N, N]
    RHO_2d = gaussian_filter_fft_with_kernel(RHO_2d, G_fft_2d)

    pad     = (Nf - N) // 2
    pad_val = N * rho_solvent                    # exakt MATLAB: number_of_voxels * rho_solvent
    RHO_2d  = jnp.pad(
        RHO_2d,
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=pad_val,
    )

    # 2D FFT och kvadrerad magnitud
    F  = jnp.fft.fftn(RHO_2d)
    F2 = jnp.abs(jnp.fft.fftshift(F)) ** 2
    return F2.astype(jnp.float32)



@jax.jit
def build_gamma(
    w_q: jnp.ndarray,
    m_q: jnp.ndarray,
    s_q: jnp.ndarray,
    Q3: jnp.ndarray,
) -> jnp.ndarray:
    w_q = w_q.astype(jnp.float32)
    m_q = m_q.astype(jnp.float32)
    s_q = s_q.astype(jnp.float32)

    mu_q  = jnp.log(m_q**2 / jnp.sqrt(m_q**2 + s_q**2))
    sig_q = jnp.sqrt(jnp.log(1.0 + (s_q**2) / (m_q**2)))

    def lognorm_density(q, mu, sigma):
        q_safe = jnp.where(q > 0, q, 1e-12)
        return jnp.exp(-0.5 * ((jnp.log(q_safe) - mu) / sigma) ** 2) / (
            q_safe * jnp.sqrt(2.0 * jnp.pi) * sigma
        )

    G = (
        w_q[0] * lognorm_density(Q3, mu_q[0], sig_q[0])
        + w_q[1] * lognorm_density(Q3, mu_q[1], sig_q[1])
    )

    center = (NUMBER_OF_VOXELS // 2,
              NUMBER_OF_VOXELS // 2,
              NUMBER_OF_VOXELS // 2)
    G = G.at[center].set(0.0)
    G = jnp.sqrt(G)
    G_shifted = jnp.fft.fftshift(G)
    return G_shifted.astype(jnp.float32)


def build_sparse_weight_matrix(
    q: jnp.ndarray,                # [K_q]
    Q_folded: jnp.ndarray,         # [M_fold]
    indicator_folded: jnp.ndarray, # [M_fold]
    delta_q: float,
    constant_smoothing: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    Qf  = Q_folded.astype(jnp.float32)
    ind = indicator_folded.astype(jnp.float32)
    eps = 1e-12

    if constant_smoothing:
        s_vec = jnp.full_like(q, SIGMA_SMOOTH_Q)
    else:
        q_voxel_scale = q / delta_q
        s_vec = SIGMA_SMOOTH_Q * (1.0 + jnp.log(q_voxel_scale))

    s_scaled = s_vec * delta_q  # [K_q]

    def one_q(m, s):
        # 1) Bygg fulla vikten som i MATLAB
        W_raw = jnp.exp(-0.5 * ((Qf - m) / s) ** 2) / (Qf + eps)
        W_raw = W_raw.at[-1].set(0.0)
        denom_full = jnp.sum(W_raw * ind) + eps  # W_raw' * indicator_folded
        W_full = W_raw / denom_full              # ⇒ sum(W_full * ind) = 1

        # 2) Ta K största komponenterna
        M = W_full.shape[0]
        K = jnp.minimum(K_NONZERO, M)
        top_vals, top_idx = lax.top_k(W_full, K)

        # 3) Renormalisera topparna så att sum(top_vals * indicator_subset) = 1
        ind_sub = ind[top_idx]                               # [K]
        denom_sparse = jnp.sum(top_vals * ind_sub) + eps
        top_vals = top_vals / denom_sparse                   # nu: sum(top_vals * ind_sub) ≈ 1

        # 4) Pad om K < K_nonzero
        pad_len = K_NONZERO - K

        def pad(v, pad_val):
            return jnp.pad(v, (0, pad_len), constant_values=pad_val)

        top_idx  = pad(top_idx, 0)
        top_vals = pad(top_vals, 0.0)

        return top_idx.astype(jnp.int32), top_vals.astype(jnp.float32)

    idx_sparse, w_sparse = jax.vmap(one_q)(q, s_scaled)
    return idx_sparse, w_sparse


def radial_average_sparse(
    F2_full: jnp.ndarray,   # [Nf, Nf]
    idx_sparse: jnp.ndarray, # [K_q, K_nonzero]
    w_sparse: jnp.ndarray,   # [K_q, K_nonzero]
) -> jnp.ndarray:
    """
    Radialmedelvärde med glesa vikter:
        I(q_k) ≈ sum_j w_sparse[k,j] * F2_folded[ idx_sparse[k,j] ].
    """
    F2_folded = fft_2d_fold(F2_full).reshape(-1)  # [M_fold]

    def one_q(idx_k, w_k):
        return jnp.sum(F2_folded[idx_k] * w_k)

    I_q = jax.vmap(one_q)(idx_sparse, w_sparse)   # [K_q]
    return I_q.astype(jnp.float32)

def radial_average_precomputed(
    F2_full: jnp.ndarray,   # [Nf, Nf]
    W_eff: jnp.ndarray,     # [K, M_fold]
) -> jnp.ndarray:
    """
    Radialmedelvärde med förskapade vikter:
        I(q) = W_eff @ vec(fft_2D_fold(F2_full)).
    """
    F2_folded = fft_2d_fold(F2_full).reshape(-1)  # [M_fold]
    I_q = W_eff @ F2_folded
    return I_q.astype(jnp.float32)

@partial(jax.jit, static_argnames=('n_cores',))
def mean_over_cores(
    R,
    D2,
    GAMMA,
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    G_fft_2d,
    n_cores: int,
):
    Nf = int(NUMBER_OF_VOXELS_FULL)
    init_acc = jnp.zeros((Nf, Nf), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    def body(i, state):
        key_i, acc = state
        key_i, sub = jax.random.split(key_i)

        F2_i = simulate_one_core_shell(
            sub,
            R,
            D2,
            GAMMA,
            NUMBER_OF_VOXELS_FULL,
            G_fft_2d,
            rho_solvent=rho_solvent,
            shell_thickness=shell_thickness,
            GRF_shift=GRF_shift,
            relative_rho_shell=relative_rho_shell,
        )
        return (key_i, acc + F2_i)

    _, acc_out = lax.fori_loop(0, n_cores, body, (key, init_acc))
    return acc_out



def saxs_kernel(
    w_q, m_q, s_q,
    R,
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    D2,
    Q3,
    G_fft_2d,
    idx_sparse,
    w_sparse,
    n_cores: int = 30,
):
    GAMMA = build_gamma(w_q, m_q, s_q, Q3)
    F2_sum = mean_over_cores(
        R,
        D2,
        GAMMA,
        rho_solvent,
        shell_thickness,
        GRF_shift,
        relative_rho_shell,
        G_fft_2d,
        n_cores,
    )
    I_q = radial_average_sparse(F2_sum, idx_sparse, w_sparse)
    return I_q



#key0 = jax.random.PRNGKey(0)
#keys = jax.random.split(key0, num_samples)


def build_saxs_simulation_context(
    *,
    q_min: float = 0.001,
    q_max: float = 0.2,
    n_q: int = 500,
) -> Dict[str, jnp.ndarray]:
    """
    Builds and returns all geometry- and grid-dependent objects
    needed for SAXS + real-space LNP simulations.

    This function should be called ONCE and reused everywhere.
    """

    # --- real-space geometry ---
    D2 = build_realspace_grid()          # [N, N, N]
    Q3 = build_Q3()                      # [N, N, N]

    # --- FFT kernels ---
    G_fft_2d = build_gaussian_fft_kernel(NUMBER_OF_VOXELS)  # [N, N]

    # --- q grid ---
    delta_q = 2.0 * jnp.pi / (DELTA_X * NUMBER_OF_VOXELS_FULL)
    q_grid = jnp.linspace(q_min, q_max, n_q, dtype=jnp.float32)

    # --- folded q-space + indicator ---
    Q_folded, indicator_folded = build_Q_folded_and_indicator(
        NUMBER_OF_VOXELS_FULL, delta_q
    )

    # --- sparse radial weights ---
    idx_sparse, w_sparse = build_sparse_weight_matrix(
        q_grid,
        Q_folded,
        indicator_folded,
        delta_q,
        constant_smoothing=True,
    )

    return {
        # geometry
        "D2": D2,
        "Q3": Q3,

        # FFT helpers
        "G_fft_2d": G_fft_2d,

        # q-space
        "q_grid": q_grid,
        "delta_q": delta_q,
        "Q_folded": Q_folded,
        "indicator_folded": indicator_folded,

        # radial averaging
        "idx_sparse": idx_sparse,
        "w_sparse": w_sparse,
    }


def saxs_kernel_one(
    w_q,
    m_q,
    s_q,
    R,
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    context,
    n_cores = 50
):
    # uses global / outer-scope D2, Q3, G_fft_2d, idx_sparse, w_sparse
    
    return saxs_kernel(
        w_q, m_q, s_q,
        R,
        rho_solvent,
        shell_thickness,
        GRF_shift,
        relative_rho_shell,
        context["D2"],
        context["Q3"],
        context["G_fft_2d"],
        context["idx_sparse"],
        context["w_sparse"],
        n_cores=n_cores,
    )

def saxs_kernel_polydisperse_one(
    key,
    w_q,
    m_q,
    s_q,
    R_batch,                # [n_R]
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    context,
):
    """
    Returns I_batch: [n_R, K_q], one GRF realisation per radius.
    Uses global D2, Q3, G_fft_2d, idx_sparse, w_sparse.
    """
    R_batch = jnp.asarray(R_batch, dtype=jnp.float32).reshape(-1)

    GAMMA = build_gamma(w_q, m_q, s_q, context["Q3"])  # fftshift'ed, as in your code
    keys = jax.random.split(key, R_batch.shape[0])  # resample GRF per radius

    def one_radius(k, R):
        F2 = simulate_one_core_shell(
            k, R, context["D2"], GAMMA, 
            NUMBER_OF_VOXELS_FULL, context["G_fft_2d"],
            rho_solvent=rho_solvent,
            shell_thickness=shell_thickness,
            GRF_shift=GRF_shift,
            relative_rho_shell=relative_rho_shell,
        )
        return radial_average_sparse(F2, context["idx_sparse"], context["w_sparse"])  # [K_q]

    return jax.vmap(one_radius)(keys, R_batch)


def lnp_volume_and_core_slice_one(
    key,
    w_q,
    m_q,
    s_q,
    R: float,
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    context
):
    """
    Returns:
      RHO_3d: [N,N,N] float32
      core_slice: [N,N] float32 (central z-slice masked to the core)
    Uses global D2, Q3.
    """
    GAMMA = build_gamma(w_q, m_q, s_q, context["Q3"])

    N = context["D2"].shape[0]
    noise = jax.random.normal(key, (N, N, N), dtype=jnp.float32)
    FW = jnp.fft.fftn(noise)
    GRF = jnp.fft.ifftn(FW * GAMMA).real

    GRF = GRF - GRF.mean()
    GRF = GRF / (GRF.std() + 1e-12)
    GRF = jax.nn.sigmoid(ALPHA * GRF)
    GRF = (GRF - GRF.mean()) + 1.0 + GRF_shift

    # core + shell density
    RHO = jnp.full_like(GRF, rho_solvent, dtype=jnp.float32)

    idx_shell = (context["D2"] <= R**2)
    shell_val = relative_rho_shell * GRF.mean()
    RHO = jnp.where(idx_shell, shell_val, RHO)

    R_core = jnp.maximum(R - shell_thickness, 0.0)
    idx_core = (context["D2"] <= R_core**2)
    RHO = jnp.where(idx_core, GRF, RHO)

    z0 = N // 2
    core_slice = jnp.where(idx_core[:, :, z0], RHO[:, :, z0], jnp.nan)

    return RHO.astype(jnp.float32), core_slice.astype(jnp.float32)


@partial(jax.jit, static_argnames=("N_max", "return_F2_stack"))
def saxs_curves_vs_num_cores(
    key,
    *,
    # physical / model params (same for all particles)
    w_q,
    m_q,
    s_q,
    R: float,
    rho_solvent: float,
    shell_thickness: float,
    GRF_shift: float,
    relative_rho_shell: float,
    # precomputed context from build_saxs_simulation_context()
    context: dict,
    # how many independent particles
    N_max: int = 100,
    # memory heavy; usually keep False
    return_F2_stack: bool = False,
):
    """
    Returns SAXS curves corresponding to averaging n=1..N_max independent
    GRF particle realisations (same radius and same parameters).
    
    Output:
      I_stack: [N_max, K_q] float32  (curve after 1..N_max particles)
      (optional) F2_mean_stack: [N_max, Nf, Nf] float32
    """

    # Build GRF frequency-domain kernel once
    GAMMA = build_gamma(w_q, m_q, s_q, context["Q3"])  # [N,N,N], fftshifted

    Nf = int(NUMBER_OF_VOXELS_FULL)
    K_q = context["idx_sparse"].shape[0]

    # scan state: (rng_key, running_sum_F2)
    init_sum = jnp.zeros((Nf, Nf), dtype=jnp.float32)

    def step(state, _):
        key_i, sum_F2 = state
        key_i, sub = jax.random.split(key_i)

        # one particle (independent GRF)
        F2_i = simulate_one_core_shell(
            sub,
            R,
            context["D2"],
            GAMMA,
            NUMBER_OF_VOXELS_FULL,
            context["G_fft_2d"],
            rho_solvent=rho_solvent,
            shell_thickness=shell_thickness,
            GRF_shift=GRF_shift,
            relative_rho_shell=relative_rho_shell,
        )

        sum_F2 = sum_F2 + F2_i

        # running mean (after k+1 particles)
        # scan index is not directly provided, so we track count via carry or use lax.scan with xs=arange
        return (key_i, sum_F2), F2_i  # temporarily return F2_i; we'll convert to mean below

    # We need the running mean, so scan over indices 1..N_max
    idxs = jnp.arange(1, N_max + 1, dtype=jnp.float32)

    def step_with_count(state, count):
        key_i, sum_F2 = state
        key_i, sub = jax.random.split(key_i)

        F2_i = simulate_one_core_shell(
            sub,
            R,
            context["D2"],
            GAMMA,
            NUMBER_OF_VOXELS_FULL,
            context["G_fft_2d"],
            rho_solvent=rho_solvent,
            shell_thickness=shell_thickness,
            GRF_shift=GRF_shift,
            relative_rho_shell=relative_rho_shell,
        )
        sum_F2 = sum_F2 + F2_i
        mean_F2 = sum_F2 / count  # count = 1..N_max

        # radial average each running mean => I(q) for this count
        I_q = radial_average_sparse(mean_F2, context["idx_sparse"], context["w_sparse"])  # [K_q]

        if return_F2_stack:
            return (key_i, sum_F2), (I_q, mean_F2)
        else:
            return (key_i, sum_F2), I_q

    (key_out, _), out = lax.scan(step_with_count, (key, init_sum), idxs)

    if return_F2_stack:
        I_stack, F2_mean_stack = out  # [N_max,K_q], [N_max,Nf,Nf]
        return I_stack.astype(jnp.float32), F2_mean_stack.astype(jnp.float32)
    else:
        I_stack = out  # [N_max,K_q]
        return I_stack.astype(jnp.float32)



def sample_truncnorm(key, *, mu, sigma, lo, hi, n):
    """
    Sample n values from a truncated Normal(mu, sigma^2) on [lo, hi].
    Rejection-free via inverse-CDF.
    """
    mu = jnp.asarray(mu, dtype=jnp.float32)
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    lo = jnp.asarray(lo, dtype=jnp.float32)
    hi = jnp.asarray(hi, dtype=jnp.float32)

    a = (lo - mu) / (sigma + 1e-12)
    b = (hi - mu) / (sigma + 1e-12)

    Phi_a = jsp.special.ndtr(a)
    Phi_b = jsp.special.ndtr(b)

    u = jax.random.uniform(key, (n,), dtype=jnp.float32)
    z = jsp.special.ndtri(Phi_a + u * (Phi_b - Phi_a))
    return (mu + sigma * z).astype(jnp.float32)



@partial(jax.jit, static_argnames=("n_particles",))
def synthetic_polydisperse_curve_mc(
    key,
    *,
    # fixed structure params (same for all particles)
    w_q, m_q, s_q,
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    context,

    # truncated normal over radii
    mu_R,
    sigma_R,
    R_lo,
    R_hi,

    # number of particles in the Monte-Carlo mixture
    n_particles: int = 512,
):
    """
    Polydisperse SAXS curve via Monte-Carlo mixture over particles:
      - sample R_j ~ TruncNormal(mu_R, sigma_R, [R_lo, R_hi])
      - for each j: draw independent GRF, simulate one particle, compute I_j(q)
      - average curves: I_poly(q) = mean_j I_j(q)

    Returns:
      R_batch:  [n_particles]
      I_batch:  [n_particles, K_q]
      I_poly:   [K_q]
    """
    key_R, key_sim = jax.random.split(key, 2)

    R_batch = sample_truncnorm(
        key_R, mu=mu_R, sigma=sigma_R, lo=R_lo, hi=R_hi, n=n_particles
    )

    # one independent GRF per radius sample is already done inside saxs_kernel_polydisperse_one
    I_batch = saxs_kernel_polydisperse_one(
        key_sim,
        w_q=w_q, m_q=m_q, s_q=s_q,
        R_batch=R_batch,
        rho_solvent=rho_solvent,
        shell_thickness=shell_thickness,
        GRF_shift=GRF_shift,
        relative_rho_shell=relative_rho_shell,
        context=context,
    )  # [n_particles, K_q]

    I_poly = jnp.mean(I_batch, axis=0)
    return R_batch, I_batch, I_poly




def _truncnorm_one(u, mu, sigma, lo, hi):
    a = (lo - mu) / (sigma + 1e-12)
    b = (hi - mu) / (sigma + 1e-12)
    Phi_a = jsp.special.ndtr(a)
    Phi_b = jsp.special.ndtr(b)
    z = jsp.special.ndtri(Phi_a + u * (Phi_b - Phi_a))
    return mu + sigma * z

@partial(jax.jit, static_argnames=("n_particles",))
def synthetic_polydisperse_curve_mc_sequential(
    key,
    *,
    # fixed structure params
    w_q, m_q, s_q,
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    context,
    # radius distribution
    mu_R,
    sigma_R,
    R_lo,
    R_hi,
    # number of particles (each particle = new R + new GRF)
    n_particles: int = 200,
):
    """
    Returns:
      I_poly: [K_q]  (mean intensity over n_particles)
    """
    GAMMA = build_gamma(w_q, m_q, s_q, context["Q3"])  # built once

    K_q = context["idx_sparse"].shape[0]
    I_sum0 = jnp.zeros((K_q,), dtype=jnp.float32)

    def body(i, carry):
        key_i, I_sum = carry
        key_i, kR, kGRF = jax.random.split(key_i, 3)

        # sample radius for this particle
        u = jax.random.uniform(kR, (), dtype=jnp.float32)
        R = _truncnorm_one(u, mu_R, sigma_R, R_lo, R_hi).astype(jnp.float32)

        # fresh GRF for this particle
        F2 = simulate_one_core_shell(
            kGRF,
            R,
            context["D2"],
            GAMMA,
            NUMBER_OF_VOXELS_FULL,
            context["G_fft_2d"],
            rho_solvent=rho_solvent,
            shell_thickness=shell_thickness,
            GRF_shift=GRF_shift,
            relative_rho_shell=relative_rho_shell,
        )

        I_q = radial_average_sparse(F2, context["idx_sparse"], context["w_sparse"])
        return (key_i, I_sum + I_q)

    _, I_sum = lax.fori_loop(0, n_particles, body, (key, I_sum0))
    return (I_sum / float(n_particles)).astype(jnp.float32)


@partial(jax.jit, static_argnames=("n_particles", "n_store"))
def synthetic_polydisperse_mc_mean_and_examples(
    key,
    *,
    w_q, m_q, s_q,
    rho_solvent,
    shell_thickness,
    GRF_shift,
    relative_rho_shell,
    context,
    mu_R,
    sigma_R,
    R_lo,
    R_hi,
    n_particles: int = 200,
    n_store: int = 10,   # how many individual curves to keep
):
    """
    Returns:
      I_mean:   [K_q]
      I_store:  [n_store, K_q]  (first n_store particles)
      R_store:  [n_store]
    """
    GAMMA = build_gamma(w_q, m_q, s_q, context["Q3"])
    K_q = context["idx_sparse"].shape[0]

    I_sum0   = jnp.zeros((K_q,), dtype=jnp.float32)
    I_store0 = jnp.zeros((n_store, K_q), dtype=jnp.float32)
    R_store0 = jnp.zeros((n_store,), dtype=jnp.float32)

    def body(i, carry):
        key_i, I_sum, I_store, R_store = carry
        key_i, kR, kGRF = jax.random.split(key_i, 3)

        u = jax.random.uniform(kR, (), dtype=jnp.float32)
        R = _truncnorm_one(u, mu_R, sigma_R, R_lo, R_hi).astype(jnp.float32)

        F2 = simulate_one_core_shell(
            kGRF, R, context["D2"], GAMMA,
            NUMBER_OF_VOXELS_FULL, context["G_fft_2d"],
            rho_solvent=rho_solvent,
            shell_thickness=shell_thickness,
            GRF_shift=GRF_shift,
            relative_rho_shell=relative_rho_shell,
        )
        I_q = radial_average_sparse(F2, context["idx_sparse"], context["w_sparse"])  # [K_q]
        I_sum = I_sum + I_q

        def do_store(args):
            I_store, R_store = args
            I_store = I_store.at[i].set(I_q)
            R_store = R_store.at[i].set(R)
            return (I_store, R_store)

        I_store, R_store = lax.cond(
            i < n_store,
            do_store,
            lambda args: args,
            (I_store, R_store),
        )

        return (key_i, I_sum, I_store, R_store)

    _, I_sum, I_store, R_store = lax.fori_loop(
        0, n_particles, body, (key, I_sum0, I_store0, R_store0)
    )

    I_mean = (I_sum / float(n_particles)).astype(jnp.float32)
    return I_mean, I_store, R_store





def build_lengthscale_context(*, N=NUMBER_OF_VOXELS, delta_x=DELTA_X, n_bins=256,
                             q_min=None, q_max=None):
    # unshifted FFT frequencies (matches jnp.fft.fftn output ordering)
    k = jnp.fft.fftfreq(N, d=float(delta_x)).astype(jnp.float32)  # cycles/Å
    k = jnp.arange(-N//2, N//2, dtype=jnp.float32)
    delta_q = 2*jnp.pi/(N*delta_x)
    q1d = k * delta_q

    QX = q1d.reshape(N,1,1)
    QY = q1d.reshape(1,N,1)
    QZ = q1d.reshape(1,1,N)
    Q  = jnp.sqrt(QX**2 + QY**2 + QZ**2)

    q_nyq = jnp.pi / float(delta_x)
    if q_min is None:
        q_min = float(2.0 * (2.0*jnp.pi/(float(N)*float(delta_x))))  # roughly 2*fundamental
    if q_max is None:
        q_max = float(0.7 * q_nyq)

    q_min = jnp.asarray(q_min, jnp.float32)
    q_max = jnp.asarray(q_max, jnp.float32)

    edges   = jnp.linspace(q_min, q_max, n_bins + 1, dtype=jnp.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    Qf = Q.reshape(-1)
    bin_idx = jnp.searchsorted(edges, Qf, side="right") - 1

    valid = (bin_idx >= 0) & (bin_idx < n_bins) & (Qf > 0.0)
    bin_idx_valid = jnp.where(valid, bin_idx, 0).astype(jnp.int32)
    valid_w = valid.astype(jnp.float32)

    counts = jnp.bincount(bin_idx_valid, weights=valid_w, length=n_bins).astype(jnp.float32)

    return dict(
        N=N, delta_x=float(delta_x),
        q_edges=edges, q_centers=centers,
        bin_idx=bin_idx_valid, valid_w=valid_w, counts=counts
    )

def build_Q3_lengthscale(*, N=NUMBER_OF_VOXELS, delta_x=DELTA_X):
    k = jnp.fft.fftfreq(N, d=float(delta_x)).astype(jnp.float32)  # cycles/Å
    q1d = (2.0 * jnp.pi) * k                                      # rad/Å
    QX = q1d.reshape(N,1,1)
    QY = q1d.reshape(1,N,1)
    QZ = q1d.reshape(1,1,N)
    return jnp.sqrt(QX**2 + QY**2 + QZ**2).astype(jnp.float32)


@jax.jit
def build_gamma_(
    w_q: jnp.ndarray,
    m_q: jnp.ndarray,
    s_q: jnp.ndarray,
    Q3: jnp.ndarray,
) -> jnp.ndarray:
    w_q = w_q.astype(jnp.float32)
    m_q = m_q.astype(jnp.float32)
    s_q = s_q.astype(jnp.float32)

    mu_q  = jnp.log(m_q**2 / jnp.sqrt(m_q**2 + s_q**2))
    sig_q = jnp.sqrt(jnp.log(1.0 + (s_q**2) / (m_q**2)))

    def lognorm_density(q, mu, sigma):
        q_safe = jnp.where(q > 0, q, 1e-12)
        return jnp.exp(-0.5 * ((jnp.log(q_safe) - mu) / sigma) ** 2) / (
            q_safe * jnp.sqrt(2.0 * jnp.pi) * sigma
        )

    G = (
        w_q[0] * lognorm_density(Q3, mu_q[0], sig_q[0])
        + w_q[1] * lognorm_density(Q3, mu_q[1], sig_q[1])
    )

    center = (NUMBER_OF_VOXELS // 2,
              NUMBER_OF_VOXELS // 2,
              NUMBER_OF_VOXELS // 2)
    G = G.at[center].set(0.0)
    G = jnp.sqrt(G)
    G_shifted = jnp.fft.fftshift(G)
    return G_shifted.astype(jnp.float32)



def make_lengthscale_profile_fn(Q3, context_ls: dict):
    N       = int(context_ls["N"])
    bin_idx = context_ls["bin_idx"]
    valid_w = context_ls["valid_w"]
    counts  = context_ls["counts"]
    q       = context_ls["q_centers"]
    ell     = (2.0 * jnp.pi) / (q + 1e-12)

    # precompute ordering for quantiles (ell constant!)
    order = jnp.argsort(ell)
    ell_s = ell[order]

    @jax.jit
    def one(key, w_q, m_q, s_q, GRF_shift):
        GAMMA = build_gamma_(w_q, m_q, s_q, Q3)   # oshiftad nu

        noise = jax.random.normal(key, (N,N,N), dtype=jnp.float32)
        FW = jnp.fft.fftshift(jnp.fft.fftn(noise))
        GRF = jnp.fft.ifftn(jnp.fft.ifftshift(FW * GAMMA)).real

        GRF = jnp.fft.ifftn(jnp.fft.ifftshift(FW * GAMMA)).real
        rho_int = jax.nn.sigmoid(ALPHA * GRF)
        rho_int = (rho_int - rho_int.mean()) + 1.0 + GRF_shift

        drho = rho_int - rho_int.mean()

        F = jnp.fft.fftshift(jnp.fft.fftn(drho))
        P = (jnp.abs(F)**2).reshape(-1)

        sumP = jnp.bincount(bin_idx, weights=P * valid_w, length=counts.shape[0]).astype(jnp.float32)
        S_q  = sumP / (counts + 1e-12)

        w = (q**2) * S_q
        w = jnp.where(jnp.isfinite(w), w, 0.0)

        # This is probability MASS per bin (up to a constant Δq which cancels in normalization)
        p_ell = w / (jnp.sum(w) + 1e-12)

        # summaries (re-use precomputed order)
        p_s  = p_ell[order]
        cdf  = jnp.cumsum(p_s)

        def qtile(qt):
            idx = jnp.searchsorted(cdf, qt, side="left")
            idx = jnp.clip(idx, 0, ell_s.shape[0] - 1)
            return ell_s[idx]

        i_mode = jnp.argmax(p_ell)
        ell_mode = ell[i_mode]
        ell_med  = qtile(0.50)
        ell_iqr  = qtile(0.75) - qtile(0.25)

        return ell, p_ell, ell_mode, ell_med, ell_iqr

    batch = jax.jit(jax.vmap(one, in_axes=(0,0,0,0,0)))
    return one, batch



def make_lengthscale_profiles_batch(Q3, context_ls: dict):
    one, _batch_unused = make_lengthscale_profile_fn(Q3, context_ls)  # one är jittad
    ell = (2.0 * jnp.pi) / (context_ls["q_centers"] + 1e-12)
    n_bins = int(context_ls["counts"].shape[0])

    @jax.jit
    def run(keys, *, w_q_batch, m_q_batch, s_q_batch, GRF_shift_batch):
        B = keys.shape[0]

        p0    = jnp.zeros((B, n_bins), dtype=jnp.float32)
        mode0 = jnp.zeros((B,), dtype=jnp.float32)
        med0  = jnp.zeros((B,), dtype=jnp.float32)
        iqr0  = jnp.zeros((B,), dtype=jnp.float32)

        def body(i, carry):
            p, mode, med, iqr = carry
            _ell, p_i, mode_i, med_i, iqr_i = one(
                keys[i],
                w_q_batch[i], m_q_batch[i], s_q_batch[i],
                GRF_shift_batch[i],
            )
            p    = p.at[i].set(p_i)
            mode = mode.at[i].set(mode_i)
            med  = med.at[i].set(med_i)
            iqr  = iqr.at[i].set(iqr_i)
            return (p, mode, med, iqr)

        p, mode, med, iqr = lax.fori_loop(0, B, body, (p0, mode0, med0, iqr0))
        return ell, p, mode, med, iqr

    return run