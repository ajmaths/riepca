"""riepca_core -- one shared RiePCA implementation for all the notebooks.

Written in plain NumPy so a notebook runs without the (unpublished) `riepca`
package. `check_against_package` proves the two agree to machine precision, so
using the package elsewhere for convenience stays licensed.

    from riepca_core import (
        EuclideanGeometry, SphereGeometry, FlatTorusGeometry,
        riepca, check_riepca, field_capacity,
        degenerate_blocks, block_magnitude, block_angle, block_summary,
        plot_eigenvalues, plot_variance_explained, plot_cumulative_variance,
        plot_spectrum_across_t, plot_block_magnitude,
    )

THE MATHEMATICS (one place, so every notebook cites the same thing)

    kappa_t(x,y) = (4 pi t)^{-q/2} exp(-d(x,y)^2 / 4t)
    g_{t,y}(x)   = grad_x kappa_t(x,y) = kappa_t(x,y) Log_x(y) / (2t)

Each observation becomes a tangent FIELD sampled at the references x_1..x_r,
living in the direct sum H_R = T_{x_1}M (+) ... (+) T_{x_r}M with

    <U,V>_R = sum_a omega_a g_{x_a}(U_a, V_a),        omega_a = 1 by default.

After centering with the observation weights mu,

    Gamma_ij = sqrt(mu_i mu_j) <f_i - fbar, f_j - fbar>_R
    alpha_k(y_i) = sum_a omega_a g_{x_a}(f_i(x_a) - fbar(x_a), E_k(x_a))

No Fréchet mean, and no chart is used anywhere: only Log and d at the references.

DEGENERATE BLOCKS -- why `block_magnitude` exists

When two eigenvalues are nearly equal (a circle produces a degenerate PAIR),
the individual axes inside that block are arbitrary: any rotation of the block
diagonalizes the covariance just as well. Individual PC scores are therefore
NOT reproducible, but the norm of the projection onto the block is:

    m_B(y) = sqrt( sum_{k in B} alpha_k(y)^2 )

is invariant under any orthogonal change of basis inside B. For a 2-D block
that is exactly sqrt(PC1^2 + PC2^2). `check_block_invariance` verifies it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "EuclideanGeometry", "SphereGeometry", "FlatTorusGeometry",
    "SPDGeometry",
    "RiePCAResult", "riepca", "heat_kernel_fields", "mean_free_scale",
    "field_capacity", "check_riepca", "transform",
    "degenerate_blocks", "block_magnitude", "block_angle", "block_summary",
    "check_block_invariance",
    "plot_eigenvalues", "plot_variance_explained", "plot_cumulative_variance",
    "plot_spectrum_across_t", "plot_block_magnitude",
    "check_against_package",
]

# Categorical palette for PC index: all six colour checks pass on a light
# surface (worst adjacent CVD dE 10.6). Marker shapes give a second encoding
# so the figures survive greyscale printing.
PC_COLORS = ["#3B6FD4", "#E4572E", "#1B9E77", "#9467BD"]
PC_MARKERS = ["o", "s", "^", "D"]
INK, MUTED = "#1a1a1a", "#6b6b6b"
# Heat time is an ORDERED variable, so it gets a sequential ramp, never
# categorical hues.
SEQUENTIAL_CMAP = "viridis"


# =====================================================================
# geometry backends
# =====================================================================
class EuclideanGeometry:
    """R^d with the flat metric. Log_x(y) = y - x."""

    name = "euclidean"

    def __init__(self, dimension: int | None = None):
        self.dimension = dimension

    def dim(self, points):
        return self.dimension if self.dimension is not None else int(np.shape(points)[1])

    def log(self, base, points):
        return np.asarray(points, float) - np.asarray(base, float)[None, :]

    def dist_sq(self, base, points):
        d = self.log(base, points)
        return np.einsum("ij,ij->i", d, d)


class SphereGeometry:
    """Unit S^{d}. Tangent vectors are ambient vectors orthogonal to the base."""

    name = "sphere"

    def __init__(self, dimension: int = 2):
        self.dimension = dimension

    def dim(self, points):
        return self.dimension

    def log(self, base, points):
        base = np.asarray(base, float).ravel()
        pts = np.asarray(points, float)
        inner = np.clip(pts @ base, -1.0, 1.0)[:, None]
        tangential = pts - inner * base[None, :]
        norm = np.linalg.norm(tangential, axis=1, keepdims=True)
        angle = np.arccos(inner)
        return np.where(norm > 1e-14, tangential / np.maximum(norm, 1e-14) * angle, 0.0)

    def dist_sq(self, base, points):
        base = np.asarray(base, float).ravel()
        inner = np.clip(np.asarray(points, float) @ base, -1.0, 1.0)
        return np.arccos(inner) ** 2


class FlatTorusGeometry:
    """Flat torus with circle circumferences `lengths`, in arc-length coordinates.

    Geodesic distance is the wrapped coordinate difference -- the whole trick.
    Note the lengths MATTER: equal circumferences make the two circles
    interchangeable, so cos/sin of each share an eigenvalue and the leading PCs
    mix them. Use the physical lengths (e.g. 2 pi R and 2 pi r).
    """

    name = "flat_torus"

    def __init__(self, lengths: Sequence[float]):
        self.lengths = np.asarray(lengths, float)
        self.dimension = len(self.lengths)

    def dim(self, points):
        return self.dimension

    def wrap(self, s):
        L = self.lengths
        return (np.asarray(s, float) + L / 2) % L - L / 2

    def log(self, base, points):
        return self.wrap(np.asarray(points, float) - np.asarray(base, float)[None, :])

    def dist_sq(self, base, points):
        d = self.log(base, points)
        return np.einsum("ij,ij->i", d, d)


class SPDGeometry:
    """SPD(d) with the affine-invariant metric.

        Log_X(Y) = X^{1/2} logm(X^{-1/2} Y X^{-1/2}) X^{1/2}
        d(X,Y)^2 = || logm(X^{-1/2} Y X^{-1/2}) ||_F^2
        g_X(U,V) = tr(X^{-1} U X^{-1} V)

    Points are (n, d, d) matrix stacks. `log` returns tangent vectors in the
    WHITENED frame -- the symmetric matrix logm(X^{-1/2} Y X^{-1/2}) written
    as a q = d(d+1)/2 vector with off-diagonals scaled by sqrt(2). In that
    frame g_X is the plain Euclidean dot product, which is exactly what the
    direct sum <U,V>_R needs, so the shared engine applies unchanged.
    """

    name = "spd_affine_invariant"

    def __init__(self, matrix_size: int):
        self.matrix_size = int(matrix_size)
        self.dimension = self.matrix_size * (self.matrix_size + 1) // 2
        iu = np.triu_indices(self.matrix_size)
        self._rows, self._cols = iu
        self._scale = np.where(self._rows == self._cols, 1.0, np.sqrt(2.0))

    def dim(self, points=None):
        return self.dimension

    @staticmethod
    def _sym_power(A, power):
        w, U = np.linalg.eigh(A)
        return (U * (w ** power)[..., None, :]) @ np.swapaxes(U, -1, -2)

    @staticmethod
    def _sym_logm(A):
        w, U = np.linalg.eigh(A)
        return (U * np.log(np.maximum(w, np.finfo(float).tiny))[..., None, :]) \
            @ np.swapaxes(U, -1, -2)

    def _whitened_log(self, base, points):
        base = np.asarray(base, float)
        pts = np.asarray(points, float)
        inv_sqrt = self._sym_power(base, -0.5)
        return self._sym_logm(inv_sqrt @ pts @ inv_sqrt)

    def to_vector(self, sym):
        """Symmetric matrices -> orthonormal coordinates (Frobenius preserved)."""
        sym = np.asarray(sym, float)
        return sym[..., self._rows, self._cols] * self._scale

    def log(self, base, points):
        return self.to_vector(self._whitened_log(base, points))

    def dist_sq(self, base, points):
        L = self._whitened_log(base, points)
        return np.einsum("...ij,...ij->...", L, L)


# =====================================================================
# core
# =====================================================================
@dataclass
class RiePCAResult:
    t: float
    eigenvalues: np.ndarray            # (k,)
    scores: np.ndarray                 # (n, k)   alpha_k(y_i)
    pc_fields: np.ndarray              # (r, q, k)
    mean_field: np.ndarray             # (r, q)
    centered_fields: np.ndarray        # (n, r, q)
    total_variance: float
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    capacity: int
    reference_points: np.ndarray
    omega: np.ndarray
    # The complete spectrum of the (rq x rq) field covariance, before the
    # top_k truncation. Kept because a caller may want the whole decay curve
    # (or the full eigenbasis) even when only k components are retained.
    all_eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0))
    all_components: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_components(self):
        return self.scores.shape[1]

    @property
    def k_over_capacity(self):
        """Fraction of the field space used. At 1.0 the PCA rotation is
        invertible, so no dimension reduction is happening."""
        return self.n_components / self.capacity


def field_capacity(n_references: int, manifold_dim: int) -> int:
    """dim H_R = r * dim T_x M -- the hard ceiling on the number of PCs."""
    return int(n_references) * int(manifold_dim)


def heat_kernel_fields(points, references, t, geometry, manifold_dim=None):
    """g_{t,y}(x_a) = kappa_t(x_a,y) Log_{x_a}(y) / (2t).  Returns (n, r, q)."""
    points = np.asarray(points, float)
    references = np.asarray(references, float)
    q = geometry.dim(points) if manifold_dim is None else int(manifold_dim)
    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("t must be positive and finite.")

    blocks = []
    for base in references:
        logs = np.asarray(geometry.log(base, points), float)
        dist_sq = np.asarray(geometry.dist_sq(base, points), float)
        log_factor = (-(q / 2.0) * np.log(4.0 * np.pi * t)
                      - np.log(2.0 * t) - dist_sq / (4.0 * t))
        blocks.append(np.exp(log_factor)[:, None] * logs)
    fields = np.stack(blocks, axis=1)
    if not np.all(np.isfinite(fields)):
        raise FloatingPointError(
            f"non-finite fields at t={t:g}. The normaliser (4 pi t)^(-q/2)/(2t) "
            f"has overflowed for q={q}; use a larger t or a smaller q.")
    return fields


def mean_free_scale(points, mu, geometry):
    """s^2 = (1/2) sum_ij mu_i mu_j d(y_i,y_j)^2.

    Equals the Frechet variance for Euclidean data (Konig-Huygens) but never
    references a centre, so tau can be reported without a Frechet mean.
    """
    points = np.asarray(points, float)
    mu = np.asarray(mu, float)
    mu = mu / mu.sum()
    total = 0.0
    for i, base in enumerate(points):
        total += mu[i] * float(mu @ np.asarray(geometry.dist_sq(base, points), float))
    return 0.5 * total


def riepca(points, references, t, mu=None, geometry=None, manifold_dim=None,
           top_k=10, omega=None, eig_rtol=1e-12, fields=None):
    """Weighted PCA of the centred heat-gradient fields in the direct sum.

    `fields` -- optional (n, r, c) array of PRECOMPUTED tangent fields. Pass it
    when the manifold has an exact heat kernel and you do not want the
    geodesic-Gaussian surrogate that `heat_kernel_fields` builds: the flat
    torus (a product of exact circle kernels), S^2 (the Legendre series), or a
    space with no `log` in closed form. Everything downstream -- centring,
    the eigenproblem, the sign convention, degenerate blocks, the plots, the
    identity checks -- is then shared with every other notebook, and only the
    field definition differs. That is the seam to unify along, because the
    field definition is the part that is genuinely geometry-specific.

    `geometry` may be None when `fields` is supplied, but then pass
    `manifold_dim` so the capacity r*q is still reported correctly.
    """
    points = np.asarray(points, float)
    references = np.asarray(references, float)
    geometry = EuclideanGeometry() if geometry is None else geometry
    n = len(points)
    mu = np.full(n, 1.0 / n) if mu is None else np.asarray(mu, float)
    mu = mu / mu.sum()
    r = len(references)
    omega = np.ones(r) if omega is None else np.asarray(omega, float)
    q = geometry.dim(points) if manifold_dim is None else int(manifold_dim)

    if fields is None:
        raw = heat_kernel_fields(points, references, t, geometry, q)
    else:
        raw = np.asarray(fields, float)
        if raw.ndim != 3 or raw.shape[0] != n or raw.shape[1] != r:
            raise ValueError(
                f"fields must have shape (n, r, c) = ({n}, {r}, c), got {raw.shape}")
    mean_field = np.tensordot(mu, raw, axes=(0, 0))
    centered = raw - mean_field[None]

    # `c` is how many NUMBERS represent a tangent vector, which need not be q:
    # on S^2 the log map returns an ambient 3-vector for a 2-dimensional
    # tangent space. q drives the kernel normaliser and the capacity; c only
    # drives the array shapes. The normal direction contributes exactly zero,
    # so the covariance has rank <= r*q regardless.
    c = centered.shape[-1]

    # Work in coordinates that absorb omega, so <U,V>_R becomes a plain dot
    # product: D_i = (sqrt(omega_a) * centered_i(x_a))_a, flattened to R^{rc}.
    root_omega = np.sqrt(omega)
    design = (centered * root_omega[None, :, None]).reshape(n, r * c)

    # FEATURE-space covariance, (rq x rq) -- not the (n x n) observation Gram.
    # They have the same non-zero spectrum, but here n >> rq (n=2000, rq=60 is
    # typical), so this is the cheap side of the SVD by a wide margin: one
    # eigh of a 60x60 instead of a 2000x2000.
    cov = design.T @ (mu[:, None] * design)
    cov = 0.5 * (cov + cov.T)                # kill asymmetric round-off

    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals[::-1], 0.0, None)
    evecs = evecs[:, ::-1]
    total_variance = float(np.trace(cov))

    capacity = field_capacity(r, q)
    keep = (min(int(top_k), capacity, int(np.sum(evals > eig_rtol * evals[0])))
            if len(evals) and evals[0] > 0 else 0)
    lam, vec = evals[:keep], evecs[:, :keep]

    if keep:
        # deterministic sign convention: largest-magnitude entry is positive.
        # Without it every score can flip between adjacent t and the panels
        # mirror for no reason.
        signs = np.sign(vec[np.argmax(np.abs(vec), axis=0), np.arange(keep)])
        signs[signs == 0] = 1.0
        vec = vec * signs[None, :]

        scores = design @ vec
        # back out of the omega-scaled coordinates: E_a = V_a / sqrt(omega_a)
        pc_fields = vec.reshape(r, c, keep) / root_omega[:, None, None]
    else:
        pc_fields = np.zeros((r, c, 0))
        scores = np.zeros((n, 0))

    # the whole eigenbasis, same sign convention, for callers that want it
    all_signs = np.sign(evecs[np.argmax(np.abs(evecs), axis=0),
                              np.arange(evecs.shape[1])])
    all_signs[all_signs == 0] = 1.0
    all_components = (evecs * all_signs[None, :]) / np.repeat(
        root_omega, c)[:, None]

    ratio = lam / total_variance if total_variance > 0 else lam * np.nan

    return RiePCAResult(
        t=float(t), eigenvalues=lam, scores=scores, pc_fields=pc_fields,
        mean_field=mean_field, centered_fields=centered,
        total_variance=total_variance, explained_variance_ratio=ratio,
        cumulative_explained_variance_ratio=np.cumsum(ratio),
        capacity=capacity, reference_points=references, omega=omega,
        all_eigenvalues=evals, all_components=all_components,
        diagnostics={"geometry": geometry.name, "manifold_dim": q,
                     "n_references": r, "top_k_requested": int(top_k)})


def transform(result: RiePCAResult, new_points, geometry=None, manifold_dim=None,
              fields=None):
    """Project UNSEEN points onto an already-fitted RiePCA basis.

    This is what makes an inductive (train/test) evaluation possible: fit on the
    training fold, then

        alpha_k(y) = sum_a omega_a < f_y(x_a) - fbar(x_a),  E_k(x_a) >

    using the TRAINING references, training mean field and training components.
    Nothing about the test points enters the fit, so a probe trained on these
    scores is a genuine out-of-sample estimate.

    Reusing `riepca(...)` on the pooled data instead is transductive: the
    unsupervised fit has seen the test points. Both are legitimate, but only
    this one answers "would it generalise?".
    """
    q = result.diagnostics.get("manifold_dim") if manifold_dim is None \
        else int(manifold_dim)
    if fields is None:
        raw = heat_kernel_fields(new_points, result.reference_points, result.t,
                                 geometry, q)
    else:
        raw = np.asarray(fields, float)      # same exact-kernel escape as riepca()
    centered = raw - result.mean_field[None]
    k = result.n_components
    if k == 0:
        return np.zeros((len(np.atleast_2d(new_points)), 0))
    return (centered * result.omega[None, :, None]).reshape(len(centered), -1) \
        @ result.pc_fields.reshape(-1, k)


def check_riepca(result: RiePCAResult, mu=None, rtol=1e-9, atol=1e-12,
                 verbose=False):
    """Four identities any correct weighted PCA in this metric must satisfy.

        1. the PC fields are orthonormal in <.,.>_R
        2. the scores are mu-centred
        3. the score covariance is diag(lambda)
        4. the centred fields have zero mu-mean

    Tolerances are relative to the natural scale of each quantity (the
    eigenvalues run over many orders of magnitude as t sweeps, so a fixed
    absolute tolerance would be meaningless at one end or the other).
    """
    n = result.scores.shape[0]
    mu = np.full(n, 1.0 / n) if mu is None else np.asarray(mu, float)
    mu = mu / mu.sum()
    S, lam, F, omega = (result.scores, result.eigenvalues,
                        result.pc_fields, result.omega)
    k = S.shape[1]
    lam_scale = float(lam[0]) if k else 1.0
    score_scale = float(np.max(np.abs(S))) if k else 1.0
    field_scale = float(np.max(np.abs(result.centered_fields))) or 1.0

    report = {
        "orthonormality_error": (float(np.max(np.abs(
            np.einsum("a,ack,acl->kl", omega, F, F) - np.eye(k)))), 1.0) if k
            else (0.0, 1.0),
        "score_mean_error": (float(np.max(np.abs(mu @ S))), score_scale) if k
            else (0.0, 1.0),
        "score_covariance_error": (float(np.max(np.abs(
            np.einsum("i,ik,il->kl", mu, S, S) - np.diag(lam)))), lam_scale)
            if k else (0.0, 1.0),
        "field_centring_error": (float(np.max(np.abs(
            np.tensordot(mu, result.centered_fields, axes=(0, 0))))), field_scale),
    }
    out = {}
    for name, (value, scale) in report.items():
        out[name] = value
        out[name + "_relative"] = value / scale
        if value >= atol + rtol * scale:
            raise AssertionError(
                f"{name} = {value:.3e} exceeds {atol + rtol * scale:.3e} "
                f"(relative {value / scale:.3e})")
    if verbose:
        for name, (value, scale) in report.items():
            print(f"  {name:<26} {value:.3e}   relative {value / scale:.3e}")
        print(f"  all four identities hold at t={result.t:g}")
    return out


# =====================================================================
# degenerate blocks -- the averaging
# =====================================================================
def degenerate_blocks(eigenvalues, ratio_threshold=1.2, max_size=None):
    """Group consecutive eigenvalues that are too close to separate.

    lambda_{k+1} joins the current block only if the WHOLE block then still
    spans less than `ratio_threshold`:

        lambda_first / lambda_last < ratio_threshold.

    That is complete linkage, and the stricter rule is the point. Chaining on
    consecutive ratios alone (single linkage) merges a slowly decaying tail
    into one huge "block": a circle with

        lambda = [0.790, 0.713, 0.598, 0.502, ...]

    has consecutive ratios 1.11, 1.19, 1.19 -- all under 1.2 -- so single
    linkage reports one block PC1-PC4 even though lambda_1/lambda_4 = 1.58 and
    those axes are perfectly well separated. Complete linkage returns the
    correct [[0,1],[2,3]]: the Fourier pairs.

    A block means "the eigenbasis inside is arbitrary to within tolerance", so
    every pair in it must satisfy the tolerance, not just neighbours.

    `max_size` caps a block (2 forbids anything but pairs).

    Returns a list of index lists, e.g. [[0, 1], [2, 3], [4]].
    """
    lam = np.asarray(eigenvalues, float)
    if lam.size == 0:
        return []
    tiny = np.finfo(float).tiny
    blocks, current = [], [0]
    for k in range(1, len(lam)):
        span = lam[current[0]] / max(lam[k], tiny)
        room = max_size is None or len(current) < max_size
        if span < ratio_threshold and room:
            current.append(k)
        else:
            blocks.append(current)
            current = [k]
    blocks.append(current)
    return blocks


def block_magnitude(scores, block, rms=False):
    """m_B(y) = sqrt( sum_{k in B} alpha_k(y)^2 ).

    THE rotation-invariant summary of a degenerate block: for a 2-D block this
    is sqrt(PC1^2 + PC2^2). Unlike the individual scores it does not change
    when the eigen-solver picks a different basis inside the block.

    `rms=True` divides by sqrt(|B|), which puts blocks of different sizes on a
    comparable scale; it is the same quantity up to that constant.
    """
    idx = list(block)
    m = np.sqrt(np.sum(np.asarray(scores, float)[:, idx] ** 2, axis=1))
    return m / np.sqrt(len(idx)) if rms else m


def block_angle(scores, block):
    """atan2 within a 2-D block.

    Defined only up to a global rotation/reflection of the block, so compare it
    with a shift- and reflection-invariant statistic (see `circ_assoc` in the
    torus notebook), never with a plain correlation.
    """
    idx = list(block)
    if len(idx) != 2:
        raise ValueError(f"block_angle needs a 2-D block, got {len(idx)} axes")
    S = np.asarray(scores, float)
    return np.arctan2(S[:, idx[1]], S[:, idx[0]])


def block_summary(result: RiePCAResult, ratio_threshold=1.2):
    """Every degenerate block, with its invariant magnitude and share."""
    lam = result.eigenvalues
    blocks = degenerate_blocks(lam, ratio_threshold)
    out = []
    for b in blocks:
        out.append({
            "indices": b,
            "pcs": [i + 1 for i in b],
            "size": len(b),
            "eigenvalues": lam[b],
            "variance_share": float(np.sum(result.explained_variance_ratio[b])),
            "magnitude": block_magnitude(result.scores, b),
            "angle": block_angle(result.scores, b) if len(b) == 2 else None,
            "label": ("PC" + "-PC".join(str(i + 1) for i in (b[0], b[-1]))
                      if len(b) > 1 else f"PC{b[0]+1}"),
        })
    return out


def check_block_invariance(result: RiePCAResult, block, n_trials=5, seed=0,
                           atol=1e-10):
    """Prove the block magnitude is basis-independent.

    Rotate the scores inside the block by a random orthogonal matrix -- exactly
    the freedom a degenerate eigenvalue leaves the solver -- and confirm the
    magnitude is unchanged while the individual axes move.
    """
    rng = np.random.default_rng(seed)
    idx = list(block)
    base_mag = block_magnitude(result.scores, idx)
    worst_mag, worst_axis = 0.0, 0.0
    for _ in range(n_trials):
        Q, _ = np.linalg.qr(rng.normal(size=(len(idx), len(idx))))
        rotated = result.scores.copy()
        rotated[:, idx] = result.scores[:, idx] @ Q
        worst_mag = max(worst_mag, float(np.max(np.abs(
            block_magnitude(rotated, idx) - base_mag))))
        worst_axis = max(worst_axis, float(np.max(np.abs(
            rotated[:, idx] - result.scores[:, idx]))))
    if worst_mag >= atol:
        raise AssertionError(f"block magnitude moved by {worst_mag:.3e}")
    return {"magnitude_change": worst_mag, "axis_change": worst_axis}


# =====================================================================
# plotting
# =====================================================================
def _finish(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_eigenvalues(result, ax=None, top_k=None, log=True, mark_blocks=True,
                     ratio_threshold=1.2, title=None):
    """Spectrum, with degenerate blocks shaded so the pairing is visible."""
    import matplotlib.pyplot as plt

    lam = np.asarray(result.eigenvalues, float)
    if log:
        lam = lam[lam > 0]
    if top_k:
        lam = lam[:top_k]
    idx = np.arange(1, len(lam) + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]

    if mark_blocks:
        for b in degenerate_blocks(result.eigenvalues, ratio_threshold):
            b = [i for i in b if i < len(lam)]
            if len(b) > 1:
                ax.axvspan(b[0] + 0.6, b[-1] + 1.4, color=MUTED, alpha=0.10,
                           lw=0, zorder=0)
    ax.plot(idx, lam, "-", color=MUTED, lw=1.2, zorder=1)
    ax.scatter(idx, lam, s=42, color=PC_COLORS[0], marker=PC_MARKERS[0],
               zorder=3, linewidths=0)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(idx)
    _finish(ax, "component $k$", r"$\lambda_k$",
            title or rf"eigenvalues, $t={result.t:g}$"
            + ("   (shaded = degenerate block)" if mark_blocks else ""))
    return ax


def plot_variance_explained(result, ax=None, top_k=10, annotate=True, title=None):
    import matplotlib.pyplot as plt

    ratio = np.asarray(result.explained_variance_ratio, float)[:top_k]
    idx = np.arange(1, len(ratio) + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    ax.bar(idx, ratio, color=PC_COLORS[0], width=0.68, linewidth=0)
    if annotate:
        for k, v in zip(idx, ratio):
            if v >= 0.01:
                ax.annotate(f"{v:.1%}", (k, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(idx)
    ax.set_ylim(0, min(1.0, float(ratio.max()) * 1.2) if ratio.size else 1.0)
    _finish(ax, "component $k$", "explained variance ratio",
            title or rf"variance explained, $t={result.t:g}$")
    return ax


def plot_cumulative_variance(result, ax=None, top_k=10, threshold=0.9,
                             title=None):
    import matplotlib.pyplot as plt

    cum = np.asarray(result.cumulative_explained_variance_ratio, float)[:top_k]
    idx = np.arange(1, len(cum) + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    ax.axhline(1.0, color=MUTED, ls="--", lw=0.8)
    if threshold:
        ax.axhline(threshold, color=PC_COLORS[1], ls=":", lw=1.2,
                   label=f"{threshold:.0%}")
        ax.legend(fontsize=9, frameon=False, loc="lower right", labelcolor=INK)
    ax.plot(idx, cum, "-", color=MUTED, lw=1.2, zorder=1)
    ax.scatter(idx, cum, s=42, color=PC_COLORS[2], marker=PC_MARKERS[2],
               zorder=3, linewidths=0)
    ax.set_xticks(idx)
    ax.set_ylim(0, 1.08)
    if title is None:
        title = rf"cumulative variance, $t={result.t:g}$"
        if result.capacity:
            r = result.diagnostics.get("n_references", "r")
            q = result.diagnostics.get("manifold_dim", "q")
            title += f"   (field capacity $rq={r}\\times{q}={result.capacity}$)"
    _finish(ax, "number of components", "cumulative variance", title)
    return ax


def plot_spectrum_across_t(results, ax=None, top_k=10, title=None):
    """One curve per heat time. t is ORDERED, so it gets a sequential ramp,
    not categorical hues."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors

    items = list(results.items()) if isinstance(results, dict) else \
        [(r.t, r) for r in results]
    items.sort(key=lambda kv: kv[0])       # sort on t only: results do not compare
    ax = ax or plt.subplots(figsize=(7.0, 4.6))[1]
    ts = [float(t) for t, _ in items]
    lo, hi = min(ts), max(ts)
    if hi <= lo:                            # a single t, or all equal
        hi = lo * 1.001 + 1e-12
    norm = mcolors.LogNorm(lo, hi) if lo > 0 and hi / lo > 5 \
        else mcolors.Normalize(lo, hi)
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)    # cm.get_cmap was removed in mpl 3.9
    for t, res in items:
        lam = np.asarray(res.eigenvalues, float)
        lam = lam[lam > 0][:top_k]
        if lam.size:
            ax.plot(np.arange(1, lam.size + 1), lam, "o-", ms=4, lw=1.6,
                    color=cmap(0.12 + 0.78 * float(norm(t))))
    ax.set_yscale("log")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("heat time $t$", color=INK)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    _finish(ax, "component $k$", r"$\lambda_k$",
            title or "spectrum across heat times")
    return ax


def plot_block_magnitude(result, target=None, ratio_threshold=1.2, ax=None,
                         target_label="", title=None):
    """The averaged block coordinate sqrt(sum PC_k^2), per degenerate block.

    If `target` is given (a known latent, e.g. a radius or an angle) each block
    magnitude is plotted against it, which is the direct way to see which block
    carries which structure.
    """
    import matplotlib.pyplot as plt

    blocks = [b for b in block_summary(result, ratio_threshold) if b["size"] > 1]
    if not blocks:
        blocks = block_summary(result, ratio_threshold)[:2]
    n = len(blocks)
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(4.4 * n, 3.9), squeeze=False)
        ax = ax.ravel()
    ax = np.atleast_1d(ax)

    for j, b in enumerate(blocks[:len(ax)]):
        a = ax[j]
        colour = PC_COLORS[j % len(PC_COLORS)]
        marker = PC_MARKERS[j % len(PC_MARKERS)]
        if target is None:
            a.hist(b["magnitude"], bins=40, color=colour, alpha=0.85, lw=0)
            _finish(a, rf"$\|\alpha_{{{b['label']}}}\|$", "count",
                    f"{b['label']}   {b['variance_share']:.1%} of variance")
        else:
            a.scatter(np.asarray(target, float), b["magnitude"], s=9,
                      color=colour, marker=marker, alpha=0.6, linewidths=0)
            _finish(a, target_label or "target", rf"$\sqrt{{\sum_k \alpha_k^2}}$",
                    f"{b['label']}   {b['variance_share']:.1%} of variance")
    if title:
        ax[0].figure.suptitle(title, color=INK)
    return ax


# =====================================================================
# optional: agreement with the package
# =====================================================================
def check_against_package(points, references, t_values, mu=None, geometry=None,
                          package_geometry=None, manifold_dim=None, top_k=10,
                          atol=1e-8, verbose=True):
    """Compare this module against `riepca.fit_riepca` at machine precision.

    `package_geometry` is the package's own backend (e.g.
    GeomstatsGeometry(Hypersphere(dim=2))); `geometry` is this module's.
    Returns a list of per-t dictionaries; raises if anything disagrees.
    """
    from riepca import fit_riepca

    rows = []
    kw = {}
    try:                       # the patched package supports mean_free
        from riepca import pairwise_dispersion  # noqa: F401
        kw["mean_free"] = True
    except ImportError:
        pass

    pkg = fit_riepca(points, references, mu=mu, t_values=tuple(t_values),
                     geometry=package_geometry, top_k=top_k, **kw)
    for tau, t in zip(pkg.tau_values, t_values):
        own = riepca(points, references, t, mu=mu, geometry=geometry,
                     manifold_dim=manifold_dim, top_k=top_k)
        k = min(len(pkg.evals_dict[tau]), own.n_components)
        row = {
            "t": float(t), "k": k,
            "d_eigenvalues": float(np.max(np.abs(
                pkg.evals_dict[tau][:k] - own.eigenvalues[:k]))),
            "d_scores": float(np.max(np.abs(
                pkg.projections_dict[tau][:, :k] - own.scores[:, :k]))),
            "d_explained": float(np.max(np.abs(
                pkg.explained_variance_ratio_dict[tau][:k]
                - own.explained_variance_ratio[:k]))),
        }
        rows.append(row)
        if verbose:
            print(f"  t={t:<7g} d_eig {row['d_eigenvalues']:.2e}   "
                  f"d_scores {row['d_scores']:.2e}   "
                  f"d_explained {row['d_explained']:.2e}")
        for key in ("d_eigenvalues", "d_scores", "d_explained"):
            if row[key] >= atol:
                raise AssertionError(f"{key} = {row[key]:.3e} at t={t}")
    if verbose:
        print("PASS: riepca_core reproduces the package, sign convention included.")
    return rows
