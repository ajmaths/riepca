# %% [markdown]
# # RiePCA for Euclidean data
#
# This file is both an importable Python module and a cell-by-cell account of
# the Euclidean construction.  It deliberately contains no manifold backend:
# data and reference points are vectors in the same Euclidean space R^d.
#
# Each observation y_i produces the sampled vector field
#
#     F_i(x_a) = grad_x k_t(x_a, y_i),    a = 1, ..., r,
#
# at selected reference points x_1, ..., x_r.  After centering over i, ordinary
# weighted PCA is applied in the finite-dimensional Hilbert space
#
#     H_R = (R^d)^r,
#     <U,V>_R = sum_a omega_a U(x_a)^T V(x_a).
#
# No Frechet mean is used in the RiePCA fit.  A diffusion-Frechet function is
# used only by the optional reference-point selectors in Sections 9--12.

# %%
"""A focused, self-contained implementation of RiePCA for Euclidean data.

The numerical core requires NumPy and SciPy.  The optional epsilon-neighborhood
reference selectors additionally use scikit-learn for KMeans.

Conventions
-----------
* ``data_points[i]`` is an observation y_i in R^d with probability ``mu[i]``.
* ``reference_points[a]`` is a point x_a in R^d.
* The Euclidean heat kernel is

      k_t(x,y) = (4*pi*t)^(-d/2) exp(-||x-y||^2/(4t)).

* The sampled field is

      F_y(x) = grad_x k_t(x,y) = k_t(x,y) (y-x)/(2t).

* Fields and projection scores are centered with the same probability ``mu``.
* ``reference_volumes[a]`` supplies the weight omega_a in the field inner
  product.  Its default is one, the counting measure on the reference set.
* ``solver='direct'`` diagonalizes the feature covariance, whereas
  ``solver='dual'`` diagonalizes the observation Gram matrix.  Their nonzero
  eigenvalues, principal fields, and scores agree up to roundoff and signs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.ndimage import generate_binary_structure, label, maximum_filter
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


__all__ = [
    "RiePCAResult",
    "ReferenceSelectionResult",
    "validate_weighted_point_cloud",
    "validate_reference_points",
    "field_capacity",
    "heat_kernel_values",
    "heat_gradient_fields",
    "diffusion_frechet_values",
    "riepca_euclidean",
    "fit_riepca",
    "check_riepca",
    "select_reference_points",
    "select_reference_points_over_time",
    "degenerate_blocks",
    "block_magnitude",
    "block_angle",
    "block_summary",
    "check_block_invariance",
    "plot_eigenvalues",
    "plot_variance_explained",
    "plot_cumulative_variance",
    "plot_spectrum_across_t",
    "plot_block_magnitude",
]


# %% [markdown]
# ## 1. Output of the Euclidean construction
#
# A principal component is a sampled vector field
#
#     E_k = (E_k(x_1), ..., E_k(x_r)),    E_k(x_a) in R^d.
#
# Thus ``pc_fields`` has shape ``(r, d, k)``.  ``scores[i, k]`` is the
# projection of centered field i onto principal field k.

# %%
@dataclass
class RiePCAResult:
    """RiePCA output at one heat time; the last field axis indexes PCs."""

    t: float
    kernel_normalization: str
    solver: str
    eigenvalues: np.ndarray
    pc_fields: np.ndarray
    scores: np.ndarray
    mean_field: np.ndarray
    total_variance: float
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    field_capacity: int
    rank_capacity: int
    mu: np.ndarray
    data_points: np.ndarray
    reference_points: np.ndarray
    reference_volumes: np.ndarray
    all_eigenvalues: np.ndarray
    all_components: np.ndarray
    raw_fields: np.ndarray | None = None
    centered_fields: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def omega(self) -> np.ndarray:
        """Short alias for ``reference_volumes``."""
        return self.reference_volumes

    @property
    def components(self) -> np.ndarray:
        """Retained PCs flattened as columns, shape ``(r*d, k)``."""
        return self.pc_fields.reshape(self.field_capacity, self.n_components)


@dataclass
class ReferenceSelectionResult:
    """Output from a dimension-aware reference-point selection."""

    reference_points: np.ndarray
    t: float | None
    backend: str
    method: str
    modes: np.ndarray
    mode_values: np.ndarray
    candidate_points: np.ndarray
    candidate_values: np.ndarray
    selected_mask: np.ndarray
    component_labels: np.ndarray
    epsilon: float | None
    relative_epsilon: float | None
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_references(self) -> int:
        return int(self.reference_points.shape[0])


# %% [markdown]
# ## 2. Input validation and the two sets of weights
#
# ``mu[i]`` weights observations and is normalized to sum to one.
# ``reference_volumes[a]`` weights locations in the sampled-field inner
# product and is not normalized: rescaling every reference volume changes all
# eigenvalues by that common factor, as an integration weight should.

# %%
def validate_weighted_point_cloud(data_points, mu=None):
    """Return a finite ``(n,d)`` array and normalized nonnegative weights."""
    points = np.asarray(data_points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError("data_points must have shape (n, d), with n >= 2 and d >= 1")
    if not np.all(np.isfinite(points)):
        raise ValueError("data_points must be finite")

    n = points.shape[0]
    if mu is None:
        weights = np.full(n, 1.0 / n)
    else:
        weights = np.asarray(mu, dtype=float).reshape(-1)
        if (
            weights.shape != (n,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(f"mu must be {n} nonnegative finite weights with positive sum")
        weights = weights / weights.sum()
    return points, weights


def validate_reference_points(reference_points, ambient_dimension):
    """Return a finite reference array with shape ``(r, ambient_dimension)``."""
    references = np.asarray(reference_points, dtype=float)
    expected = int(ambient_dimension)
    if references.ndim != 2 or references.shape[0] < 1 or references.shape[1] != expected:
        raise ValueError(f"reference_points must have shape (r, {expected}), with r >= 1")
    if not np.all(np.isfinite(references)):
        raise ValueError("reference_points must be finite")
    return references


def _reference_volumes(reference_volumes, n_references):
    if reference_volumes is None:
        return np.ones(n_references)
    volumes = np.asarray(reference_volumes, dtype=float).reshape(-1)
    if (
        volumes.shape != (n_references,)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0)
    ):
        raise ValueError(
            "reference_volumes must contain one finite, strictly positive weight "
            "per reference point"
        )
    return volumes.copy()


def _positive_float(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def field_capacity(n_references: int, ambient_dimension: int) -> int:
    """Return ``dim((R^d)^r) = r*d``."""
    return int(n_references) * int(ambient_dimension)


# %% [markdown]
# ## 3. Euclidean heat kernel and heat-gradient fields
#
# For y in R^d and a reference x,
#
#     k_t(x,y) = (4*pi*t)^(-d/2) exp(-||x-y||^2/(4t)),
#     grad_x k_t(x,y) = k_t(x,y) (y-x)/(2t).
#
# In very high dimension, the common heat-kernel normalizer can overflow or
# underflow even though it does not affect the PC directions at a fixed t.
# ``kernel_normalization='none'`` drops only this common scalar.  It preserves
# PC directions and explained-variance ratios, but rescales scores and
# eigenvalues and therefore must be reported when it is used.
#
# Removing the normalizer does not prevent underflow in
# ``exp(-||x-y||^2/(4t))`` itself.  The helpers therefore inspect the
# log-kernel before exponentiating.  Because the heat kernel is local, most
# (observation, reference) pairs are legitimately zero at small t; a large
# fraction of underflowed entries is normal, not a fault.  What matters is
# whether every observation is seen by at least one reference and every
# reference sees at least one observation.  An observation whose kernel
# underflows at every reference has an identically zero field and cannot
# influence the fit; a reference that sees no observation carries no signal.
# Each condition raises a warning naming the count, and a completely
# underflowed matrix raises an error with guidance to increase t or rescale
# the data.  No data-dependent rescaling is applied silently.

# %%
_LOG_SMALLEST_SUBNORMAL = float(np.log(np.nextafter(0.0, 1.0)))
_LOG_SMALLEST_NORMAL = float(np.log(np.finfo(float).tiny))

#: Default grid padding, as a fraction of each axis's data range.  Modes of
#: the diffusion-Frechet function always lie inside the convex hull of the
#: data (every mean-shift fixed point is a convex combination of the data),
#: so padding only matters for the epsilon sublevel regions around them.
_RELATIVE_PADDING = 0.10


def heat_kernel_values(
    data_points,
    reference_points,
    t,
    *,
    kernel_normalization="heat",
):
    """Evaluate ``k_t(x_a,y_i)`` and return an array with shape ``(n,r)``."""
    points = np.asarray(data_points, dtype=float)
    if points.ndim != 2:
        raise ValueError("data_points must have shape (n, d)")
    references = validate_reference_points(reference_points, points.shape[1])
    t = _positive_float(t, "t")
    if kernel_normalization not in {"heat", "none"}:
        raise ValueError("kernel_normalization must be 'heat' or 'none'")

    diff = points[:, None, :] - references[None, :, :]
    with np.errstate(over="ignore", invalid="ignore"):
        squared_distance = np.einsum("nrd,nrd->nr", diff, diff)
    if not np.all(np.isfinite(squared_distance)):
        raise FloatingPointError(
            "squared Euclidean distances overflowed; rescale the coordinates"
        )
    log_kernel = -squared_distance / (4.0 * t)
    if kernel_normalization == "heat":
        log_kernel -= (points.shape[1] / 2.0) * np.log(4.0 * np.pi * t)

    if float(np.max(log_kernel, initial=-np.inf)) > np.log(np.finfo(float).max):
        raise FloatingPointError(
            "the Euclidean heat-kernel normalizer overflows at this dimension and t; "
            "use kernel_normalization='none' and report that common rescaling"
        )

    # The kernel is local, so a large fraction of underflowed entries is
    # expected at small t and is not a fault.  Diagnose per observation and
    # per reference instead: what breaks the fit is an observation that no
    # reference can see, or a reference that sees nothing.
    underflowed = log_kernel < _LOG_SMALLEST_SUBNORMAL
    if np.all(underflowed):
        raise FloatingPointError(
            "all Euclidean heat-kernel values underflow at this t; increase t "
            "or rescale the coordinates before fitting RiePCA"
        )
    n_invisible_observations = int(np.count_nonzero(np.all(underflowed, axis=1)))
    n_blind_references = int(np.count_nonzero(np.all(underflowed, axis=0)))
    if n_invisible_observations:
        warnings.warn(
            f"{n_invisible_observations} of {points.shape[0]} observations have "
            "a heat-kernel value of exactly zero at every reference point. Their "
            "fields are identically zero, so they receive zero scores and have "
            "no influence on the principal fields. Increase t, add reference "
            "points near those observations, or rescale the coordinates.",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_blind_references:
        warnings.warn(
            f"{n_blind_references} of {references.shape[0]} reference points see "
            "no observation (the kernel underflows to zero for every one). They "
            "contribute nothing to the field inner product and can be dropped. "
            "Increase t or move those references closer to the data.",
            RuntimeWarning,
            stacklevel=2,
        )

    with np.errstate(under="ignore"):
        kernel = np.exp(log_kernel)
    if not np.all(np.isfinite(kernel)):
        raise FloatingPointError("non-finite heat-kernel values")
    return kernel


def heat_gradient_fields(
    data_points,
    reference_points,
    t,
    *,
    kernel_normalization="heat",
):
    """Return ``grad_x k_t(x_a,y_i)`` with shape ``(n,r,d)``."""
    points = np.asarray(data_points, dtype=float)
    references = validate_reference_points(reference_points, points.shape[1])
    t = _positive_float(t, "t")
    kernel = heat_kernel_values(
        points,
        references,
        t,
        kernel_normalization=kernel_normalization,
    )
    fields = kernel[:, :, None] * (
        points[:, None, :] - references[None, :, :]
    ) / (2.0 * t)
    if not np.all(np.isfinite(fields)):
        raise FloatingPointError("non-finite heat-gradient fields")
    if not np.any(fields):
        raise FloatingPointError(
            "all heat-gradient fields are numerically zero; increase t, rescale "
            "the coordinates, or choose references closer to the observations"
        )
    return fields


# %% [markdown]
# ## 4. Centered fields and their finite-dimensional design matrix
#
# The mean field and centered fields are
#
#     Fbar = sum_i mu_i F_i,            Ftilde_i = F_i - Fbar.
#
# Multiplying the block at reference a by sqrt(omega_a) turns the weighted
# direct-sum inner product into the ordinary dot product.  Flattening gives a
# design matrix D of shape ``(n, r*d)``.

# %%
def _centered_design(raw_fields, mu, reference_volumes):
    mean_field = np.tensordot(mu, raw_fields, axes=(0, 0))
    centered = raw_fields - mean_field[None, :, :]
    root_volumes = np.sqrt(reference_volumes)
    design = (
        centered * root_volumes[None, :, None]
    ).reshape(centered.shape[0], -1)
    return mean_field, centered, design


def _pin_component_signs(components):
    components = np.asarray(components, dtype=float).copy()
    if components.shape[1] == 0:
        return components
    anchors = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[anchors, np.arange(components.shape[1])])
    signs[signs == 0] = 1.0
    return components * signs[None, :]


def _descending_psd_eigh(matrix):
    values, vectors = eigh(matrix, check_finite=True)
    values = values[::-1]
    vectors = vectors[:, ::-1]
    spectral_scale = max(float(np.max(np.abs(values), initial=0.0)), np.finfo(float).tiny)
    if np.any(values < -1e-10 * spectral_scale):
        raise ValueError("the covariance matrix is not positive semidefinite")
    return np.maximum(values, 0.0), vectors


# %% [markdown]
# ## 5. Two equivalent eigenproblems
#
# Put A = diag(sqrt(mu)) D.
#
# * The direct solver diagonalizes ``A.T @ A``, an ``(r*d) by (r*d)`` field
#   covariance matrix.  This is usually best when ``r*d <= n``.
# * The dual solver diagonalizes ``A @ A.T``, an ``n by n`` observation Gram
#   matrix.  If ``q_k`` is a dual eigenvector, the corresponding field-space
#   eigenvector is ``A.T @ q_k / sqrt(lambda_k)``.  This is usually best when
#   ``n < r*d``.
#
# These are the right- and left-singular-vector routes for the same weighted
# matrix A.  ``solver='auto'`` selects the smaller symmetric eigenproblem.

# %%
def _solve_pca(design, mu, solver):
    weighted_design = np.sqrt(mu)[:, None] * design
    n, p = weighted_design.shape
    if solver == "auto":
        solver_used = "direct" if p <= n else "dual"
    elif solver in {"direct", "dual"}:
        solver_used = solver
    else:
        raise ValueError("solver must be 'auto', 'direct', or 'dual'")

    if solver_used == "direct":
        covariance = weighted_design.T @ weighted_design
        covariance = 0.5 * (covariance + covariance.T)
        values, feature_vectors = _descending_psd_eigh(covariance)
    else:
        dual_gram = weighted_design @ weighted_design.T
        dual_gram = 0.5 * (dual_gram + dual_gram.T)
        values, dual_vectors = _descending_psd_eigh(dual_gram)
        feature_vectors = weighted_design.T @ dual_vectors

    # A dual zero-eigenvalue vector cannot be mapped uniquely into field
    # space.  Apply the same numerical-rank cutoff to both solvers so that
    # ``all_eigenvalues`` and ``all_components`` have one consistent meaning:
    # the reconstructible, numerically nonzero field-space spectrum.
    if values.size and values[0] > 0:
        reconstruction_cutoff = max(
            values[0] * 100.0 * np.finfo(float).eps,
            np.finfo(float).tiny,
        )
        reconstruct = values > reconstruction_cutoff
    else:
        reconstruct = np.zeros(values.shape, dtype=bool)
    values = values[reconstruct]
    feature_vectors = feature_vectors[:, reconstruct]

    if solver_used == "dual":
        feature_vectors = feature_vectors / np.sqrt(values)[None, :]
        # The reconstruction is orthonormal in exact arithmetic.  Normalize
        # columns without rotating them: a QR factorization could mix distinct
        # eigenvectors and would then break the eigenvalue/component pairing.
        if feature_vectors.shape[1]:
            norms = np.linalg.norm(feature_vectors, axis=0)
            feature_vectors /= norms[None, :]

    feature_vectors = _pin_component_signs(feature_vectors)
    return values, feature_vectors, solver_used


# %% [markdown]
# ## 6. Fit RiePCA at one heat time

# %%
def riepca_euclidean(
    data_points,
    reference_points,
    t,
    *,
    mu=None,
    reference_volumes=None,
    k_keep=10,
    solver="auto",
    kernel_normalization="heat",
    eig_rtol=1e-10,
    eig_atol=0.0,
    store_fields=True,
):
    """Run centered RiePCA for observations and references in Euclidean space."""
    points, mu_used = validate_weighted_point_cloud(data_points, mu)
    references = validate_reference_points(reference_points, points.shape[1])
    volumes = _reference_volumes(reference_volumes, references.shape[0])
    t = _positive_float(t, "t")

    if not isinstance(k_keep, (int, np.integer)) or int(k_keep) < 1:
        raise ValueError("k_keep must be a positive integer")
    eig_rtol = float(eig_rtol)
    eig_atol = float(eig_atol)
    if eig_rtol < 0 or eig_atol < 0:
        raise ValueError("eig_rtol and eig_atol must be nonnegative")

    raw_fields = heat_gradient_fields(
        points,
        references,
        t,
        kernel_normalization=kernel_normalization,
    )
    mean_field, centered_fields, design = _centered_design(
        raw_fields,
        mu_used,
        volumes,
    )

    all_values, all_weighted_components, solver_used = _solve_pca(
        design,
        mu_used,
        solver,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        total_variance = float(mu_used @ np.sum(design * design, axis=1))
    if not np.isfinite(total_variance):
        raise FloatingPointError(
            "the centered-field variance overflowed; use "
            "kernel_normalization='none', increase t, or rescale the coordinates"
        )
    if total_variance <= 0 or all_values.size == 0:
        raise ValueError(
            "the centered heat-gradient fields have zero numerical variance; "
            "change t or the reference points, or rescale the coordinates"
        )

    field_cap = field_capacity(references.shape[0], points.shape[1])
    rank_cap = min(field_cap, max(int(np.count_nonzero(mu_used)) - 1, 0))
    if all_values.size and all_values[0] > 0:
        cutoff = max(eig_rtol * all_values[0], eig_atol)
        n_positive = int(np.count_nonzero(all_values > cutoff))
    else:
        n_positive = 0
    k = min(int(k_keep), rank_cap, n_positive, all_weighted_components.shape[1])

    root_volumes = np.sqrt(volumes)
    retained_weighted = all_weighted_components[:, :k]
    eigenvalues = all_values[:k]
    if k:
        scores = design @ retained_weighted
        pc_fields = retained_weighted.reshape(
            references.shape[0], points.shape[1], k
        ) / root_volumes[:, None, None]
    else:
        scores = np.empty((points.shape[0], 0))
        pc_fields = np.empty((references.shape[0], points.shape[1], 0))

    # Store the numerically nonzero field-space spectrum for either solver.
    all_components = all_weighted_components / np.repeat(
        root_volumes,
        points.shape[1],
    )[:, None]

    ratio = (
        eigenvalues / total_variance
        if total_variance > 0
        else np.full(eigenvalues.shape, np.nan)
    )
    return RiePCAResult(
        t=t,
        kernel_normalization=kernel_normalization,
        solver=solver_used,
        eigenvalues=eigenvalues,
        pc_fields=pc_fields,
        scores=scores,
        mean_field=mean_field,
        total_variance=total_variance,
        explained_variance_ratio=ratio,
        cumulative_explained_variance_ratio=np.cumsum(ratio),
        field_capacity=field_cap,
        rank_capacity=rank_cap,
        mu=mu_used,
        data_points=points.copy(),
        reference_points=references.copy(),
        reference_volumes=volumes,
        all_eigenvalues=all_values,
        all_components=all_components,
        raw_fields=raw_fields if store_fields else None,
        centered_fields=centered_fields if store_fields else None,
        diagnostics={
            "ambient_dimension": points.shape[1],
            "n_observations": points.shape[0],
            "n_references": references.shape[0],
            "top_k_requested": int(k_keep),
            "solver_requested": solver,
            "field_capacity": field_cap,
            "numerical_rank": int(all_values.size),
        },
    )


# %% [markdown]
# ## 7. Multiple heat times
#
# Reference points may be fixed across t or supplied as a mapping from heat
# time to an array.  Each heat time is analyzed independently; there is no
# train/test stage in this pipeline.

# %%
def _references_at_time(reference_points, t, dimension):
    if isinstance(reference_points, ReferenceSelectionResult):
        return validate_reference_points(reference_points.reference_points, dimension)
    if not isinstance(reference_points, Mapping):
        return validate_reference_points(reference_points, dimension)
    keys = list(reference_points)
    numeric_keys = np.asarray([float(key) for key in keys])
    matches = np.flatnonzero(np.isclose(numeric_keys, t, rtol=1e-10, atol=1e-12))
    if matches.size == 0:
        raise KeyError(f"no reference points were supplied for t={t:g}")
    selected = reference_points[keys[int(matches[0])]]
    if isinstance(selected, ReferenceSelectionResult):
        selected = selected.reference_points
    return validate_reference_points(selected, dimension)


def fit_riepca(
    data_points,
    reference_points,
    *,
    t_values,
    mu=None,
    **kwargs,
):
    """Fit Euclidean RiePCA over explicit heat times; return ``{t: result}``."""
    points, mu_used = validate_weighted_point_cloud(data_points, mu)
    times = np.asarray(list(t_values), dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("t_values must be a nonempty sequence of positive finite values")
    if np.unique(times).size != times.size:
        raise ValueError("t_values must not contain duplicates")
    results = {}
    for t_raw in times:
        t = float(t_raw)
        refs = _references_at_time(reference_points, t, points.shape[1])
        results[t] = riepca_euclidean(points, refs, t, mu=mu_used, **kwargs)
    return results


# %% [markdown]
# ## 8. Mathematical checks
#
# The checks verify centering, total variance, PC orthonormality, centered and
# diagonal score covariance, the covariance eigenfield equation, and the
# direct-versus-stored projections.  Scale-free forms are used because
# the heat-kernel magnitude varies sharply with d and t.

# %%
def check_riepca(result: RiePCAResult, mu=None, rtol=1e-8, atol=1e-8, verbose=False):
    """Assert the defining centered-PCA identities and return error diagnostics."""
    if mu is not None:
        _, supplied = validate_weighted_point_cloud(result.data_points, mu)
        np.testing.assert_allclose(supplied, result.mu, rtol=0.0, atol=1e-14)
    raw = result.raw_fields
    if raw is None:
        raw = heat_gradient_fields(
            result.data_points,
            result.reference_points,
            result.t,
            kernel_normalization=result.kernel_normalization,
        )
    centered = result.centered_fields
    if centered is None:
        centered = raw - result.mean_field[None, :, :]

    field_scale = max(float(np.max(np.abs(centered), initial=0.0)), 1e-300)
    centered_mean = np.tensordot(result.mu, centered, axes=(0, 0))
    centering_error = float(np.max(np.abs(centered_mean), initial=0.0))
    np.testing.assert_allclose(centered_mean / field_scale, 0.0, atol=atol)

    squared_norms = np.einsum(
        "a,nad,nad->n",
        result.reference_volumes,
        centered,
        centered,
    )
    direct_total = float(result.mu @ squared_norms)
    total_variance_error = abs(result.total_variance - direct_total)
    np.testing.assert_allclose(result.total_variance, direct_total, rtol=rtol)

    k = result.n_components
    if k == 0:
        report = {
            "field_centring_error": centering_error,
            "field_centring_error_relative": centering_error / field_scale,
            "total_variance_error": total_variance_error,
            "total_variance_error_relative": total_variance_error / max(direct_total, 1e-300),
            "orthonormality_error": 0.0,
            "score_mean_error": 0.0,
            "score_covariance_error": 0.0,
            "score_covariance_error_relative": 0.0,
            "projection_error": 0.0,
            "eigenfield_error_relative": 0.0,
        }
        if verbose:
            print(f"all defining identities hold at t={result.t:g} (no retained PCs)")
        return report

    component_gram = np.einsum(
        "a,adk,adl->kl",
        result.reference_volumes,
        result.pc_fields,
        result.pc_fields,
    )
    orthonormality_error = float(np.max(np.abs(component_gram - np.eye(k))))
    np.testing.assert_allclose(component_gram, np.eye(k), atol=atol)

    whitened = result.scores / np.sqrt(result.eigenvalues)[None, :]
    whitened_mean = result.mu @ whitened
    score_mean_error = float(np.max(np.abs(whitened_mean), initial=0.0))
    np.testing.assert_allclose(result.mu @ whitened, np.zeros(k), atol=atol)
    whitened_covariance = (whitened * result.mu[:, None]).T @ whitened
    score_covariance_error_relative = float(
        np.max(np.abs(whitened_covariance - np.eye(k)))
    )
    np.testing.assert_allclose(
        whitened_covariance,
        np.eye(k),
        atol=atol,
    )

    direct_scores = np.einsum(
        "a,nad,adk->nk",
        result.reference_volumes,
        centered,
        result.pc_fields,
    )
    score_scale = np.maximum(np.sqrt(result.eigenvalues), 1e-300)
    projection_error = float(np.max(np.abs(direct_scores - result.scores)))
    np.testing.assert_allclose(
        direct_scores / score_scale[None, :],
        result.scores / score_scale[None, :],
        atol=atol,
    )

    covariance_fields = np.stack(
        [
            np.tensordot(
                result.mu * result.scores[:, j],
                centered,
                axes=(0, 0),
            )
            for j in range(k)
        ],
        axis=-1,
    )
    residual = covariance_fields - (
        result.pc_fields * result.eigenvalues[None, None, :]
    )
    residual_norms = np.sqrt(np.maximum(np.einsum(
        "a,adk,adk->k",
        result.reference_volumes,
        residual,
        residual,
    ), 0.0))
    eigenfield_error_relative = float(
        np.max(residual_norms / np.maximum(result.eigenvalues, 1e-300))
    )
    np.testing.assert_allclose(
        residual_norms / np.maximum(result.eigenvalues, 1e-300),
        np.zeros(k),
        atol=atol,
    )
    retained_variance = float(np.sum(result.eigenvalues))
    if retained_variance > result.total_variance * (1.0 + rtol) + atol:
        raise AssertionError("retained eigenvalues exceed the total variance")
    raw_score_covariance_error = float(np.max(np.abs(
        (result.scores * result.mu[:, None]).T @ result.scores
        - np.diag(result.eigenvalues)
    )))
    report = {
        "field_centring_error": centering_error,
        "field_centring_error_relative": centering_error / field_scale,
        "total_variance_error": total_variance_error,
        "total_variance_error_relative": total_variance_error / max(direct_total, 1e-300),
        "orthonormality_error": orthonormality_error,
        "score_mean_error": score_mean_error,
        "score_covariance_error": raw_score_covariance_error,
        "score_covariance_error_relative": score_covariance_error_relative,
        "projection_error": projection_error,
        "eigenfield_error_relative": eigenfield_error_relative,
    }
    if verbose:
        for name, value in report.items():
            print(f"  {name:<38} {value:.3e}")
        print(f"  all defining identities hold at t={result.t:g}")
    return report


# %% [markdown]
# ## 9. Diffusion-Frechet function used for reference selection
#
# In Euclidean space the heat-kernel semigroup gives
#
#     V_t(x) = sum_i mu_i d_t(x,y_i)^2
#            = 2 k_{2t}(x,x) - 2 sum_i mu_i k_{2t}(x,y_i).
#
# Since ``k_{2t}(x,x)`` is constant in x, minimizing V_t is the same as
# maximizing the Gaussian mixture ``sum_i mu_i k_{2t}(x,y_i)``.  This function
# chooses reference points only; it is not the centering operation in RiePCA.

# %%
def _scaled_diffusion_energy(candidates, data_points, mu, t, chunk_size=4096):
    """Return ``1 - sum_i mu_i exp(-||x-y_i||^2/(8t))`` in chunks."""
    candidates = np.asarray(candidates, dtype=float)
    data = np.asarray(data_points, dtype=float)
    t = _positive_float(t, "t")
    if candidates.ndim != 2 or candidates.shape[1] != data.shape[1]:
        raise ValueError("candidates and data_points must have the same ambient dimension")
    data_norm = np.einsum("ij,ij->i", data, data)
    values = np.empty(candidates.shape[0])
    for start in range(0, candidates.shape[0], int(chunk_size)):
        block = candidates[start : start + int(chunk_size)]
        squared_distance = np.maximum(
            np.einsum("ij,ij->i", block, block)[:, None]
            + data_norm[None, :]
            - 2.0 * block @ data.T,
            0.0,
        )
        with np.errstate(under="ignore"):
            density = np.exp(-squared_distance / (8.0 * t)) @ mu
        values[start : start + len(block)] = 1.0 - density
    if np.all(values == 1.0):
        raise FloatingPointError(
            "all Gaussian-mixture terms underflowed while evaluating the "
            "diffusion-Frechet function; increase t, rescale the coordinates, "
            "or use candidates closer to the data"
        )
    return values


def diffusion_frechet_values(candidates, data_points, t, *, mu=None, chunk_size=4096):
    """Evaluate the Euclidean diffusion-Frechet function ``V_t``.

    The returned value uses the fully normalized heat kernel.  For selecting
    references in very high dimension, the selectors instead use the scaled
    function ``V_t / (2*k_(2t)(x,x))`` to avoid a harmless common numerical
    overflow or underflow.
    """
    data, weights = validate_weighted_point_cloud(data_points, mu)
    candidates = validate_reference_points(candidates, data.shape[1])
    t = _positive_float(t, "t")
    scaled = _scaled_diffusion_energy(candidates, data, weights, t, chunk_size)
    log_factor = np.log(2.0) - (data.shape[1] / 2.0) * np.log(8.0 * np.pi * t)
    if log_factor > np.log(np.finfo(float).max):
        raise FloatingPointError(
            "the normalized diffusion-Frechet values overflow; use the scaled "
            "candidate_values returned by select_reference_points"
        )
    if log_factor < _LOG_SMALLEST_SUBNORMAL:
        raise FloatingPointError(
            "the normalized diffusion-Frechet scale underflows; use the scaled "
            "candidate_values returned by select_reference_points"
        )
    return np.exp(log_factor) * scaled


def _energy_band(
    candidate_values, epsilon, relative_epsilon, dimension, t, span_mask=None
):
    """Return the scaled sublevel band and the span it was measured on.

    ``span_mask`` restricts the range used for ``relative_epsilon`` to a
    subset of the candidates.  The grid backend passes the candidates inside
    the data's bounding box, so that the band does not depend on how far the
    grid was padded beyond the data: the energy is largest in the padded
    corners, and measuring the range there would tie every selection to the
    padding constant.  ``None`` uses every candidate.
    """
    if epsilon is not None and relative_epsilon is not None:
        raise ValueError("pass either epsilon or relative_epsilon, not both")
    values = np.asarray(candidate_values, dtype=float)
    if span_mask is not None and np.any(span_mask):
        values = values[np.asarray(span_mask, dtype=bool)]
    span = float(np.ptp(values))
    if epsilon is None and relative_epsilon is None:
        relative_epsilon = 0.30
    if epsilon is not None:
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        # candidate_values are V/(2*k_(2t)(x,x)); convert an absolute V band.
        log_diagonal_factor = np.log(2.0) - (dimension / 2.0) * np.log(8.0 * np.pi * t)
        if abs(log_diagonal_factor) > 700:
            raise FloatingPointError(
                "absolute epsilon is numerically ill-scaled in this dimension; "
                "use relative_epsilon instead"
            )
        band = epsilon / np.exp(log_diagonal_factor)
    else:
        relative_epsilon = float(relative_epsilon)
        if not np.isfinite(relative_epsilon) or relative_epsilon <= 0:
            raise ValueError("relative_epsilon must be finite and positive")
        band = relative_epsilon * span
    if not np.isfinite(band) or band <= 0:
        raise ValueError(
            "the epsilon band is zero or non-finite; the candidate energy may "
            "be numerically flat at this t. Increase t or rescale the coordinates"
        )
    return float(band), epsilon, relative_epsilon, span


def _grid_candidates(points, grid_shape, padding):
    d = points.shape[1]
    if grid_shape is None:
        grid_shape = (80, 80) if d == 2 else (30, 30, 30)
    if isinstance(grid_shape, (int, np.integer)):
        grid_shape = (int(grid_shape),) * d
    grid_shape = tuple(int(value) for value in grid_shape)
    if len(grid_shape) != d or any(value < 3 for value in grid_shape):
        raise ValueError(f"grid_shape must contain {d} integers, each at least 3")
    # ``padding=None`` pads each axis by a fixed fraction of that axis's data
    # range, so the grid looks the same for data of extent 1 or 100.  An
    # explicit float is an absolute length in data units, as before.
    extent = np.ptp(points, axis=0)
    if padding is None:
        fallback = float(np.max(extent)) if np.max(extent) > 0 else 1.0
        padding = _RELATIVE_PADDING * np.where(extent > 0, extent, fallback)
    else:
        padding = float(padding)
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("padding must be finite and nonnegative")
        padding = np.full(d, padding)
    lower = points.min(axis=0) - padding
    upper = points.max(axis=0) + padding
    axes = [np.linspace(lower[j], upper[j], grid_shape[j]) for j in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    candidates = np.column_stack([coordinate.ravel() for coordinate in mesh])
    return axes, candidates, grid_shape, padding


def _inside_data_box(candidates, points):
    """Mask of candidates inside the data's axis-aligned bounding box."""
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return np.all((candidates >= lower) & (candidates <= upper), axis=1)


def _density_gradient(x, data, mu, t):
    difference = x[None, :] - data
    squared_distance = np.einsum("ij,ij->i", difference, difference)
    weights = mu * np.exp(-squared_distance / (8.0 * t))
    return -np.sum(weights[:, None] * difference, axis=0) / (4.0 * t)


def _refine_grid_modes(seeds, data, mu, t, bounds, merge_tolerance):
    modes = []
    for seed in seeds:
        objective = lambda x: _scaled_diffusion_energy(
            np.asarray(x)[None, :], data, mu, t, 1
        )[0]
        gradient = lambda x: -_density_gradient(np.asarray(x), data, mu, t)
        optimized = minimize(
            objective,
            seed,
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-12},
        )
        mode = optimized.x
        if not any(np.linalg.norm(mode - previous) <= merge_tolerance for previous in modes):
            modes.append(mode)
    return np.asarray(modes, dtype=float)


def _grid_modes(points, mu, t, axes, values, grid_shape, max_modes, merge_tolerance):
    shaped = values.reshape(grid_shape)
    local_maxima = shaped == -maximum_filter(-shaped, size=3, mode="nearest")
    # The expression above is equivalent to local minima of the energy but
    # avoids a second sign-sensitive tolerance.  Label plateaus and use one
    # deterministic seed from each plateau.
    plateau_labels, count = label(local_maxima, structure=generate_binary_structure(len(grid_shape), 1))
    seeds = []
    for component in range(1, count + 1):
        indices = np.flatnonzero(plateau_labels.ravel() == component)
        if indices.size:
            chosen = indices[np.argmin(values[indices])]
            multi = np.unravel_index(chosen, grid_shape)
            seeds.append(np.array([axes[j][multi[j]] for j in range(len(grid_shape))]))
    if not seeds:
        chosen = int(np.argmin(values))
        multi = np.unravel_index(chosen, grid_shape)
        seeds = [np.array([axes[j][multi[j]] for j in range(len(grid_shape))])]
    seed_values = _scaled_diffusion_energy(np.asarray(seeds), points, mu, t)
    order = np.argsort(seed_values, kind="stable")
    if max_modes is not None:
        order = order[: int(max_modes)]
    bounds = [(axis[0], axis[-1]) for axis in axes]
    modes = _refine_grid_modes(
        np.asarray(seeds)[order],
        points,
        mu,
        t,
        bounds,
        merge_tolerance,
    )
    mode_values = _scaled_diffusion_energy(modes, points, mu, t)
    order = np.argsort(mode_values, kind="stable")
    return modes[order], mode_values[order]


def _knn_graph(points, n_neighbors):
    n = points.shape[0]
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    k_query = min(n_neighbors + 1, n)
    _, neighbors = cKDTree(points).query(points, k=k_query)
    neighbors = np.asarray(neighbors)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    neighbors = neighbors[:, 1:]
    rows = np.repeat(np.arange(n), neighbors.shape[1])
    cols = neighbors.reshape(-1)
    graph = sparse.csr_matrix(
        (np.ones(rows.size, dtype=bool), (rows, cols)),
        shape=(n, n),
    )
    return graph.maximum(graph.T), neighbors


def _mean_shift_modes(seed_indices, points, mu, t, tolerance, max_iter, merge_tolerance):
    modes = []
    seed_for_mode = []
    for seed_index in seed_indices:
        current = points[int(seed_index)].copy()
        for _ in range(int(max_iter)):
            squared_distance = np.sum((points - current) ** 2, axis=1)
            weights = mu * np.exp(-squared_distance / (8.0 * t))
            denominator = float(weights.sum())
            if denominator <= 0:
                raise FloatingPointError("all mean-shift weights underflowed")
            updated = (weights @ points) / denominator
            if np.linalg.norm(updated - current) <= tolerance * max(1.0, np.linalg.norm(current)):
                current = updated
                break
            current = updated
        else:
            raise RuntimeError("mean-shift did not converge; increase mean_shift_max_iter")
        duplicate = next(
            (
                index
                for index, mode in enumerate(modes)
                if np.linalg.norm(current - mode) <= merge_tolerance
            ),
            None,
        )
        if duplicate is None:
            modes.append(current)
            seed_for_mode.append(int(seed_index))
    modes = np.asarray(modes, dtype=float)
    mode_values = _scaled_diffusion_energy(modes, points, mu, t)
    order = np.argsort(mode_values, kind="stable")
    return modes[order], mode_values[order], np.asarray(seed_for_mode, dtype=int)[order]


def _representatives(points, labels_array, n_components, n_per_component, random_state):
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            "epsilon-neighborhood representatives require scikit-learn"
        ) from exc

    references = []
    for component in range(1, int(n_components) + 1):
        members = points[labels_array == component]
        if members.shape[0] == 0:
            continue
        k = min(int(n_per_component), members.shape[0])
        if k == 1:
            centers = np.mean(members, axis=0, keepdims=True)
            assignments = np.zeros(members.shape[0], dtype=int)
        else:
            model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(members)
            centers = model.cluster_centers_
            assignments = model.labels_
        for cluster_index in range(k):
            cluster = members[assignments == cluster_index]
            if cluster.shape[0]:
                distances = np.linalg.norm(cluster - centers[cluster_index], axis=1)
                references.append(cluster[int(np.argmin(distances))])
    if not references:
        return np.empty((0, points.shape[1]))
    references = np.asarray(references)
    _, unique_indices = np.unique(np.round(references, 12), axis=0, return_index=True)
    return references[np.sort(unique_indices)]


# %% [markdown]
# ## 10. Low-dimensional selector: a grid in R^2 or R^3
#
# For d=2 or d=3, the selector evaluates the scaled diffusion-Frechet function
# on a bounding-box grid, detects and continuously refines its local minima,
# and forms the epsilon sublevel component around every minimum.  Connected
# components of their union are summarized by KMeans representatives that are
# snapped back to actual grid points in the selected set.

# %%
def _select_grid(
    points,
    mu,
    t,
    *,
    method,
    epsilon,
    relative_epsilon,
    grid_shape,
    padding,
    n_representatives,
    max_modes,
    merge_tolerance,
    random_state,
    chunk_size,
):
    d = points.shape[1]
    if d not in {2, 3}:
        raise ValueError("the grid backend is available only for d=2 or d=3")
    axes, candidates, grid_shape, padding_used = _grid_candidates(
        points, grid_shape, padding
    )
    values = _scaled_diffusion_energy(candidates, points, mu, t, chunk_size)
    in_data_box = _inside_data_box(candidates, points)
    modes, mode_values = _grid_modes(
        points,
        mu,
        t,
        axes,
        values,
        grid_shape,
        max_modes,
        merge_tolerance,
    )

    if method == "global_minimum":
        references = modes[:1]
        selected = np.zeros(candidates.shape[0], dtype=bool)
        component_labels = np.zeros(candidates.shape[0], dtype=int)
        band = None
        epsilon_used = epsilon
        relative_used = relative_epsilon
        span = float(np.ptp(values[in_data_box] if np.any(in_data_box) else values))
    else:
        band, epsilon_used, relative_used, span = _energy_band(
            values, epsilon, relative_epsilon, d, t, span_mask=in_data_box
        )
        union = np.zeros(grid_shape, dtype=bool)
        shaped_values = values.reshape(grid_shape)
        structure_mode = generate_binary_structure(d, d)
        for mode, mode_value in zip(modes, mode_values):
            sublevel = shaped_values < mode_value + band
            sublevel_labels, _ = label(sublevel, structure=structure_mode)
            index = tuple(int(np.argmin(np.abs(axes[j] - mode[j]))) for j in range(d))
            component = int(sublevel_labels[index])
            if component:
                union |= sublevel_labels == component
        union_labels, count = label(
            union,
            structure=generate_binary_structure(d, max(d - 1, 1)),
        )
        selected = union.ravel()
        component_labels = union_labels.ravel()
        references = _representatives(
            candidates,
            component_labels,
            count,
            n_representatives,
            random_state,
        )
        if references.shape[0] == 0:
            raise ValueError(
                "no epsilon-neighborhood grid cells were found; increase epsilon "
                "or grid resolution"
            )
    return ReferenceSelectionResult(
        reference_points=references,
        t=t,
        backend="grid",
        method=method,
        modes=modes,
        mode_values=mode_values,
        candidate_points=candidates,
        candidate_values=values,
        selected_mask=selected,
        component_labels=component_labels,
        epsilon=epsilon_used,
        relative_epsilon=relative_used,
        diagnostics={
            "grid_shape": grid_shape,
            "padding": padding_used,
            "padding_mode": "relative" if padding is None else "absolute",
            "scaled_band": band,
            "candidate_value_span": span,
            "span_measured_inside_data_box": bool(np.any(in_data_box)),
            "n_modes": modes.shape[0],
            "n_selected_candidates": int(np.count_nonzero(selected)),
            "n_components": int(np.max(component_labels, initial=0)),
        },
    )


# %% [markdown]
# ## 11. Higher-dimensional selector: data candidates and a kNN graph
#
# A Cartesian grid is exponential in d.  The kNN backend therefore evaluates
# the diffusion-Frechet function only at the observations.  kNN-local minima
# seed Gaussian mean-shift, which refines them to continuous minima.  For every
# refined minimum, the data-point sublevel set is restricted to its kNN
# connected component.  Components of the union are represented by points
# selected from the observed cloud.

# %%
def _select_knn(
    points,
    mu,
    t,
    *,
    method,
    epsilon,
    relative_epsilon,
    n_neighbors,
    n_representatives,
    max_modes,
    merge_tolerance,
    mean_shift_tolerance,
    mean_shift_max_iter,
    random_state,
    chunk_size,
):
    values = _scaled_diffusion_energy(points, points, mu, t, chunk_size)
    graph, neighbor_indices = _knn_graph(points, n_neighbors)
    tolerance = 1e-12 * max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    local = np.all(
        values[:, None] <= values[neighbor_indices] + tolerance,
        axis=1,
    )
    seeds = np.flatnonzero(local)
    if seeds.size == 0:
        seeds = np.array([int(np.argmin(values))])
    seeds = seeds[np.argsort(values[seeds], kind="stable")]
    if max_modes is not None:
        seeds = seeds[: int(max_modes)]
    modes, mode_values, mode_seed_indices = _mean_shift_modes(
        seeds,
        points,
        mu,
        t,
        mean_shift_tolerance,
        mean_shift_max_iter,
        merge_tolerance,
    )

    if method == "global_minimum":
        references = modes[:1]
        selected = np.zeros(points.shape[0], dtype=bool)
        union_labels = np.zeros(points.shape[0], dtype=int)
        band = None
        epsilon_used = epsilon
        relative_used = relative_epsilon
        span = float(np.ptp(values))
    else:
        band, epsilon_used, relative_used, span = _energy_band(
            values, epsilon, relative_epsilon, points.shape[1], t
        )
        selected = np.zeros(points.shape[0], dtype=bool)
        for seed_index in mode_seed_indices:
            # Connectivity lives on the observed candidates, so its discrete
            # sublevel component is anchored at the candidate minimum.  The
            # continuously refined mode can have a lower value than every
            # observed point; using that lower value here could make a valid
            # graph neighborhood appear empty solely from sampling resolution.
            sublevel = values < values[int(seed_index)] + band
            indices = np.flatnonzero(sublevel)
            if indices.size == 0 or int(seed_index) not in indices:
                continue
            subgraph = graph[indices][:, indices]
            _, sublabels = connected_components(subgraph, directed=False)
            seed_position = int(np.flatnonzero(indices == int(seed_index))[0])
            selected[indices[sublabels == sublabels[seed_position]]] = True
        selected_indices = np.flatnonzero(selected)
        union_labels = np.zeros(points.shape[0], dtype=int)
        if selected_indices.size:
            count, labels_selected = connected_components(
                graph[selected_indices][:, selected_indices],
                directed=False,
            )
            union_labels[selected_indices] = labels_selected + 1
        else:
            count = 0
        references = _representatives(
            points,
            union_labels,
            count,
            n_representatives,
            random_state,
        )
        if references.shape[0] == 0:
            raise ValueError(
                "no kNN epsilon-neighborhood points were found; increase "
                "relative_epsilon or n_neighbors"
            )
    return ReferenceSelectionResult(
        reference_points=references,
        t=t,
        backend="knn",
        method=method,
        modes=modes,
        mode_values=mode_values,
        candidate_points=points.copy(),
        candidate_values=values,
        selected_mask=selected,
        component_labels=union_labels,
        epsilon=epsilon_used,
        relative_epsilon=relative_used,
        diagnostics={
            "n_neighbors": min(int(n_neighbors), points.shape[0] - 1),
            "scaled_band": band,
            "candidate_value_span": span,
            "n_discrete_seeds": int(seeds.size),
            "mode_seed_indices": mode_seed_indices,
            "n_modes": modes.shape[0],
            "n_selected_candidates": int(np.count_nonzero(selected)),
            "n_components": int(np.max(union_labels, initial=0)),
        },
    )


# %% [markdown]
# ## 12. Dimension-aware public reference selector
#
# Four user-facing methods are available:
#
# * ``'grid'``: every point of a Cartesian grid (d=2 or d=3 only);
# * ``'global_minimum'``: the best refined diffusion-Frechet minimum;
# * ``'epsilon_neighborhood'``: representatives of epsilon neighborhoods of
#   the retained local diffusion-Frechet minima;
# * ``'custom'``: reference points supplied directly by the user.
#
# For the two diffusion-Frechet methods, ``backend='auto'`` uses the continuous
# grid construction when d=2 or d=3 and the data-candidate/kNN construction
# when d>3.
#
# An absolute ``epsilon`` is in the units of V_t.  The default
# ``relative_epsilon=0.30`` is a fraction of the V_t range over the candidates
# inside the data's bounding box (on the grid backend; on the kNN backend the
# candidates are the data themselves) and is invariant both to the common
# heat-kernel normalizer and to the grid padding.  It is therefore preferable
# for heat-time sweeps and high-dimensional data.  In a sweep,
# ``time_strategy='per_t'`` reselects references at every t, while
# ``time_strategy='fixed'`` selects them once at ``reference_time``.

# %%
def _static_reference_result(
    points,
    method,
    *,
    custom_reference_points,
    grid_shape,
    padding,
):
    """Construct the non-Frechet ``grid`` and ``custom`` selections."""
    d = points.shape[1]
    if method == "grid":
        if d not in {2, 3}:
            raise ValueError("method='grid' is available only for d=2 or d=3")
        if custom_reference_points is not None:
            raise ValueError("custom_reference_points is used only with method='custom'")
        _, references, resolved_shape, padding_used = _grid_candidates(
            points, grid_shape, padding
        )
        backend = "grid"
        diagnostics = {
            "grid_shape": resolved_shape,
            "padding": padding_used,
            "padding_mode": "relative" if padding is None else "absolute",
            "uses_diffusion_frechet": False,
        }
    else:
        if custom_reference_points is None:
            raise ValueError(
                "custom_reference_points is required when method='custom'"
            )
        references = validate_reference_points(custom_reference_points, d).copy()
        backend = "custom"
        diagnostics = {"uses_diffusion_frechet": False}

    n_references = references.shape[0]
    return ReferenceSelectionResult(
        reference_points=references,
        t=None,
        backend=backend,
        method=method,
        modes=np.empty((0, d)),
        mode_values=np.empty(0),
        candidate_points=references.copy(),
        candidate_values=np.full(n_references, np.nan),
        selected_mask=np.ones(n_references, dtype=bool),
        component_labels=np.zeros(n_references, dtype=int),
        epsilon=None,
        relative_epsilon=None,
        diagnostics=diagnostics,
    )


def select_reference_points(
    data_points,
    t=None,
    *,
    mu=None,
    method="epsilon_neighborhood",
    custom_reference_points=None,
    backend="auto",
    epsilon=None,
    relative_epsilon=None,
    grid_shape=None,
    padding=None,
    n_neighbors=20,
    n_representatives=10,
    max_modes=10,
    merge_tolerance=None,
    mean_shift_tolerance=1e-10,
    mean_shift_max_iter=1000,
    random_state=0,
    chunk_size=4096,
):
    """Choose references by grid, Frechet minima, epsilon regions, or input.

    ``padding`` controls how far the candidate grid (``method='grid'`` and
    the grid backend) extends beyond the data.  ``None``, the default, pads
    every axis by 10% of that axis's data range, so the grid is invariant to
    the units of the coordinates.  A float is an absolute length in data
    units; pass ``padding=0.2`` to reproduce the earlier fixed default.

    With ``relative_epsilon``, the band is a fraction of the range of the
    scaled diffusion-Frechet function over the candidates that lie inside
    the data's bounding box.  Candidates in the padded margin are still
    eligible for selection, but they do not set the scale of the band, so
    the selected regions do not change when the padding does.
    """
    points, weights = validate_weighted_point_cloud(data_points, mu)
    methods = {"grid", "global_minimum", "epsilon_neighborhood", "custom"}
    if method not in methods:
        raise ValueError(
            "method must be 'grid', 'global_minimum', "
            "'epsilon_neighborhood', or 'custom'"
        )
    if method in {"grid", "custom"}:
        return _static_reference_result(
            points,
            method,
            custom_reference_points=custom_reference_points,
            grid_shape=grid_shape,
            padding=padding,
        )
    if custom_reference_points is not None:
        raise ValueError("custom_reference_points is used only with method='custom'")

    t = _positive_float(t, "t")
    if backend == "auto":
        backend = "grid" if points.shape[1] in {2, 3} else "knn"
    if backend not in {"grid", "knn"}:
        raise ValueError("backend must be 'auto', 'grid', or 'knn'")
    if not isinstance(n_representatives, (int, np.integer)) or int(n_representatives) < 1:
        raise ValueError("n_representatives must be a positive integer")
    if max_modes is not None and (
        not isinstance(max_modes, (int, np.integer)) or int(max_modes) < 1
    ):
        raise ValueError("max_modes must be None or a positive integer")
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    if (
        not isinstance(mean_shift_max_iter, (int, np.integer))
        or int(mean_shift_max_iter) < 1
    ):
        raise ValueError("mean_shift_max_iter must be a positive integer")

    extent = float(np.max(np.ptp(points, axis=0), initial=0.0))
    scale = max(1.0, extent, np.sqrt(t))
    if merge_tolerance is None:
        merge_tolerance = 1e-6 * scale
    merge_tolerance = _positive_float(merge_tolerance, "merge_tolerance")

    common = dict(
        method=method,
        epsilon=epsilon,
        relative_epsilon=relative_epsilon,
        n_representatives=int(n_representatives),
        max_modes=max_modes,
        merge_tolerance=merge_tolerance,
        random_state=int(random_state),
        chunk_size=int(chunk_size),
    )
    if backend == "grid":
        return _select_grid(
            points,
            weights,
            t,
            grid_shape=grid_shape,
            padding=padding,
            **common,
        )
    return _select_knn(
        points,
        weights,
        t,
        n_neighbors=n_neighbors,
        mean_shift_tolerance=_positive_float(
            mean_shift_tolerance, "mean_shift_tolerance"
        ),
        mean_shift_max_iter=int(mean_shift_max_iter),
        **common,
    )


def select_reference_points_over_time(
    data_points,
    t_values,
    *,
    mu=None,
    method="epsilon_neighborhood",
    time_strategy="per_t",
    reference_time=None,
    **kwargs,
):
    """Select references per heat time or once at a fixed reference time."""
    times = np.asarray(list(t_values), dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("t_values must be a nonempty sequence of positive finite values")
    if np.unique(times).size != times.size:
        raise ValueError("t_values must not contain duplicates")
    if time_strategy not in {"per_t", "fixed"}:
        raise ValueError("time_strategy must be 'per_t' or 'fixed'")

    if method in {"grid", "custom"}:
        if reference_time is not None:
            raise ValueError(
                "reference_time is used only for fixed diffusion-Frechet selection"
            )
        selected = select_reference_points(
            data_points, mu=mu, method=method, **kwargs
        )
        return {float(t): selected for t in times}

    if time_strategy == "per_t":
        if reference_time is not None:
            raise ValueError("reference_time must be omitted when time_strategy='per_t'")
        return {
            float(t): select_reference_points(
                data_points, float(t), mu=mu, method=method, **kwargs
            )
            for t in times
        }

    if reference_time is None:
        raise ValueError(
            "reference_time is required when time_strategy='fixed'"
        )
    selected = select_reference_points(
        data_points,
        _positive_float(reference_time, "reference_time"),
        mu=mu,
        method=method,
        **kwargs,
    )
    return {float(t): selected for t in times}


# %% [markdown]
# ## 13. Nearly degenerate eigenspaces
#
# When adjacent eigenvalues are nearly equal, their individual axes can rotate
# within the common eigenspace.  The magnitude of a score block is invariant
# under that rotation; a two-dimensional block also has a useful polar angle.

# %%
def degenerate_blocks(eigenvalues, ratio_threshold=1.2, max_size=None):
    """Group consecutive near-equal eigenvalues by complete linkage.

    A candidate joins a block only when the ratio of the block's first and
    last eigenvalues remains below ``ratio_threshold``.  This avoids chaining
    a slowly decaying spectrum into one artificial large block.
    """
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if values.size == 0:
        return []
    ratio_threshold = float(ratio_threshold)
    if ratio_threshold <= 1:
        raise ValueError("ratio_threshold must exceed one")
    blocks, current = [], [0]
    tiny = np.finfo(float).tiny
    for index in range(1, values.size):
        span = values[current[0]] / max(values[index], tiny)
        room = max_size is None or len(current) < int(max_size)
        if span < ratio_threshold and room:
            current.append(index)
        else:
            blocks.append(tuple(current))
            current = [index]
    blocks.append(tuple(current))
    return blocks


def block_magnitude(scores, block, *, rms=False):
    """Return the rotation-invariant magnitude of selected score columns."""
    chosen = np.asarray(scores, dtype=float)[:, list(block)]
    magnitude = np.sqrt(np.sum(chosen * chosen, axis=1))
    return magnitude / np.sqrt(chosen.shape[1]) if rms else magnitude


def block_angle(scores, block):
    """Return ``atan2`` for a two-component score block."""
    block = tuple(block)
    if len(block) != 2:
        raise ValueError("block_angle requires exactly two component indices")
    chosen = np.asarray(scores, dtype=float)[:, list(block)]
    return np.arctan2(chosen[:, 1], chosen[:, 0])


def block_summary(result: RiePCAResult, ratio_threshold=1.2):
    """Summarize each spectral block and its invariant score magnitude."""
    summaries = []
    for block in degenerate_blocks(result.eigenvalues, ratio_threshold):
        indices = list(block)
        summaries.append({
            "indices": indices,
            "pcs": [index + 1 for index in indices],
            "size": len(indices),
            "eigenvalues": result.eigenvalues[indices],
            "variance_share": float(np.sum(result.explained_variance_ratio[indices])),
            "magnitude": block_magnitude(result.scores, indices),
            "angle": block_angle(result.scores, indices) if len(indices) == 2 else None,
            "label": (
                f"PC{indices[0] + 1}-PC{indices[-1] + 1}"
                if len(indices) > 1
                else f"PC{indices[0] + 1}"
            ),
        })
    return summaries


def check_block_invariance(result, block, n_trials=5, seed=0, atol=1e-10):
    """Rotate a score block randomly and verify that its magnitude is unchanged.
    ``atol`` is RELATIVE to the largest baseline magnitude in the block.  The
    scores inherit the heat-kernel magnitude ``(4*pi*t)^(-d/2)/(2t)``, so the
    round-off of an exactly orthogonal rotation grows with that scale while an
    absolute tolerance does not: a fixed ``atol`` rejects results that are
    correct to machine precision whenever t is small or the coordinates are.
    """
    
    indices = list(block)
    if not indices:
        raise ValueError("block must contain at least one component index")
    rng = np.random.default_rng(seed)
    baseline = block_magnitude(result.scores, indices)
    scale = max(float(np.max(np.abs(baseline), initial=0.0)), 1e-300)
    worst_magnitude_change = 0.0
    worst_axis_change = 0.0
    for _ in range(int(n_trials)):
        rotation, _ = np.linalg.qr(rng.normal(size=(len(indices), len(indices))))
        rotated = result.scores.copy()
        rotated[:, indices] = result.scores[:, indices] @ rotation
        worst_magnitude_change = max(
            worst_magnitude_change,
            float(np.max(np.abs(block_magnitude(rotated, indices) - baseline))),
        )
        worst_axis_change = max(
            worst_axis_change,
            float(np.max(np.abs(rotated[:, indices] - result.scores[:, indices]))),
        )
    if worst_magnitude_change >= atol * scale:
        raise AssertionError(
            f"block magnitude changed by {worst_magnitude_change:.3e}, "
            f"which is {worst_magnitude_change / scale:.3e} of the block "
            f"magnitude and exceeds atol={atol:.3e}"
        )
    return {
        "magnitude_change": worst_magnitude_change,
        "axis_change": worst_axis_change,
        "relative_magnitude_change": worst_magnitude_change / scale,
    }


# %% [markdown]
# ## 14. Optional plotting helpers
#
# Matplotlib is imported only inside these functions, so it is not required to
# fit RiePCA or select references.  These small helpers cover the diagnostics
# used by the accompanying experiment notebooks.

# %%
_PC_COLORS = ["#3B6FD4", "#E4572E", "#1B9E77", "#9467BD"]
_PC_MARKERS = ["o", "s", "^", "D"]
_INK, _MUTED = "#1a1a1a", "#6b6b6b"


def _finish_plot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=_INK)
    ax.set_ylabel(ylabel, color=_INK)
    ax.set_title(title, color=_INK, fontsize=11)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize=9)


def plot_eigenvalues(
    result,
    ax=None,
    top_k=None,
    log=True,
    mark_blocks=True,
    ratio_threshold=1.2,
    title=None,
):
    """Plot the retained spectrum and optionally shade degenerate blocks."""
    import matplotlib.pyplot as plt

    values = np.asarray(result.eigenvalues, dtype=float)
    if log:
        values = values[values > 0]
    if top_k is not None:
        values = values[: int(top_k)]
    indices = np.arange(1, values.size + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    if mark_blocks:
        for block in degenerate_blocks(result.eigenvalues, ratio_threshold):
            visible = [index for index in block if index < values.size]
            if len(visible) > 1:
                ax.axvspan(
                    visible[0] + 0.6,
                    visible[-1] + 1.4,
                    color=_MUTED,
                    alpha=0.10,
                    linewidth=0,
                )
    ax.plot(indices, values, "-", color=_MUTED, linewidth=1.2)
    ax.scatter(indices, values, s=42, color=_PC_COLORS[0], linewidths=0, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(indices)
    _finish_plot(ax, "component $k$", r"$\lambda_k$", title or rf"eigenvalues, $t={result.t:g}$")
    return ax


def plot_variance_explained(result, ax=None, top_k=10, annotate=True, title=None):
    """Plot retained explained-variance ratios."""
    import matplotlib.pyplot as plt

    ratios = np.asarray(result.explained_variance_ratio)[: int(top_k)]
    indices = np.arange(1, ratios.size + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    ax.bar(indices, ratios, color=_PC_COLORS[0], width=0.68, linewidth=0)
    if annotate:
        for index, value in zip(indices, ratios):
            if value >= 0.01:
                ax.annotate(
                    f"{value:.1%}",
                    (index, value),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=8,
                    color=_MUTED,
                )
    ax.set_xticks(indices)
    ax.set_ylim(0, min(1.0, float(ratios.max()) * 1.2) if ratios.size else 1.0)
    _finish_plot(
        ax,
        "component $k$",
        "explained variance ratio",
        title or rf"variance explained, $t={result.t:g}$",
    )
    return ax


def plot_cumulative_variance(result, ax=None, top_k=10, threshold=0.9, title=None):
    """Plot cumulative retained explained variance."""
    import matplotlib.pyplot as plt

    cumulative = np.asarray(result.cumulative_explained_variance_ratio)[: int(top_k)]
    indices = np.arange(1, cumulative.size + 1)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    ax.axhline(1.0, color=_MUTED, linestyle="--", linewidth=0.8)
    if threshold is not None:
        ax.axhline(
            threshold,
            color=_PC_COLORS[1],
            linestyle=":",
            linewidth=1.2,
            label=f"{threshold:.0%}",
        )
        ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.plot(indices, cumulative, "-", color=_MUTED, linewidth=1.2)
    ax.scatter(indices, cumulative, s=42, color=_PC_COLORS[2], linewidths=0, zorder=3)
    ax.set_xticks(indices)
    ax.set_ylim(0, 1.08)
    _finish_plot(
        ax,
        "number of components",
        "cumulative variance",
        title or rf"cumulative variance, $t={result.t:g}$",
    )
    return ax


def plot_spectrum_across_t(results, ax=None, top_k=10, title=None):
    """Plot one eigenvalue curve per heat time using a sequential color map."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    items = list(results.items()) if isinstance(results, Mapping) else [
        (result.t, result) for result in results
    ]
    items.sort(key=lambda item: item[0])
    if not items:
        raise ValueError("results must not be empty")
    ax = ax or plt.subplots(figsize=(7.0, 4.6))[1]
    times = [float(item[0]) for item in items]
    lower, upper = min(times), max(times)
    if upper <= lower:
        upper = lower * 1.001 + 1e-12
    norm = (
        colors.LogNorm(lower, upper)
        if lower > 0 and upper / lower > 5
        else colors.Normalize(lower, upper)
    )
    color_map = plt.get_cmap("viridis")
    for t, result in items:
        values = np.asarray(result.eigenvalues)
        values = values[values > 0][: int(top_k)]
        if values.size:
            ax.plot(
                np.arange(1, values.size + 1),
                values,
                "o-",
                markersize=4,
                linewidth=1.6,
                color=color_map(0.12 + 0.78 * float(norm(t))),
            )
    ax.set_yscale("log")
    scalar = cm.ScalarMappable(norm=norm, cmap=color_map)
    colorbar = ax.figure.colorbar(scalar, ax=ax, pad=0.02)
    colorbar.set_label("heat time $t$")
    _finish_plot(ax, "component $k$", r"$\lambda_k$", title or "spectrum across heat times")
    return ax


def plot_block_magnitude(
    result,
    target=None,
    ratio_threshold=1.2,
    ax=None,
    target_label="",
    title=None,
):
    """Plot invariant magnitudes of multi-axis spectral blocks."""
    import matplotlib.pyplot as plt

    blocks = [
        summary
        for summary in block_summary(result, ratio_threshold)
        if summary["size"] > 1
    ]
    if not blocks:
        blocks = block_summary(result, ratio_threshold)[:2]
    if not blocks:
        raise ValueError("the result has no retained components")
    if ax is None:
        _, axes = plt.subplots(1, len(blocks), figsize=(4.4 * len(blocks), 3.9), squeeze=False)
        ax = axes.ravel()
    axes = np.atleast_1d(ax)
    for index, summary in enumerate(blocks[: axes.size]):
        current = axes[index]
        color = _PC_COLORS[index % len(_PC_COLORS)]
        marker = _PC_MARKERS[index % len(_PC_MARKERS)]
        if target is None:
            current.hist(summary["magnitude"], bins=40, color=color, alpha=0.85, linewidth=0)
            _finish_plot(
                current,
                rf"$\|\alpha_{{{summary['label']}}}\|$",
                "count",
                f"{summary['label']}   {summary['variance_share']:.1%} of variance",
            )
        else:
            current.scatter(
                np.asarray(target),
                summary["magnitude"],
                s=9,
                color=color,
                marker=marker,
                alpha=0.6,
                linewidths=0,
            )
            _finish_plot(
                current,
                target_label or "target",
                r"$\sqrt{\sum_k \alpha_k^2}$",
                f"{summary['label']}   {summary['variance_share']:.1%} of variance",
            )
    if title:
        axes[0].figure.suptitle(title, color=_INK)
    return axes
