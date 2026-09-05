# %% [markdown]
# # Readable RiePCA on a Riemannian manifold
#
# This file is both an importable Python module and a cell-by-cell explanation
# of the construction. Editors that understand ``# %% [markdown]`` markers
# can display these comments as Markdown cells.
#
# The central idea is to turn each observation into a vector field sampled at
# selected reference points. We center those fields, define their inner products
# with the Riemannian metric, construct the covariance operator, and then apply 
# PCA in the Hilbert space of vector fields.

# %%
""" Readable, self-contained RiePCA for data on a Riemannian manifold.

The code is intended for paper notebooks.  It requires only NumPy, SciPy,
and a Geomstats-style manifold object ``space`` with ``space.metric``.

Conventions
-----------
* ``data_points[i]`` is an observation ``y_i`` with probability ``mu[i]``.
* ``reference_points[a]`` is a point ``x_a`` where every field is sampled.
* At a selected reference point, all field values lie in the same tangent space 
and may therefore be averaged and compared with the Riemannian metric.
* Let (M, g) be a Riemannian manifold of bounded geometry. The inner product of 
two vector fields on M is defined by

      <U, V>_R = sum_a omega[a] g_{x_a}(U_a, V_a),

  where U_a := U(a) and V_a := V(a). The default volume measure is ``omega[a] = 1``, 
  which is the counting measure on the reference set.
* The default field is the gradient of a geodesic Gaussian.  It is the exact
  Euclidean heat-kernel gradient, but only a short-time heat-kernel model on a
  general curved manifold.  A custom exact field builder may be supplied.
* Covariance and projection are both centered with respect to ``mu``.
* Components are ordered largest-first, so column/component 0 is PC1.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import eigh


# %% [markdown]
# ## 1. Output of the construction
#
# A principal component is not one tangent vector at one point. It is a
# sampled vector field
#
# $$E_k=(E_k(x_1),\ldots,E_k(x_r)),\qquad E_k(x_a)\in T_{x_a}\mathcal M.$$
#
# The last axis of ``pc_fields`` indexes these principal fields. The scalar
# ``scores[i, k]`` is the projection of centered observation field $i$ onto
# principal field $k$.

# %%
@dataclass
class RiePCAResult:
    """RiePCA output at one time; the last axis of ``pc_fields`` is the PC."""

    t: float
    field_model: str
    eigenvalues: np.ndarray
    pc_fields: np.ndarray
    scores: np.ndarray
    mean_field: np.ndarray
    total_variance: float
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray
    rank_capacity: int
    mu: np.ndarray
    reference_points: np.ndarray
    reference_volumes: np.ndarray
    field_gram: np.ndarray
    dual_gram: np.ndarray
    dual_eigenvectors: np.ndarray
    raw_fields: np.ndarray
    centered_fields: np.ndarray

    @property
    def n_components(self):
        return self.eigenvalues.size


# %% [markdown]
# ## 2. The two sets of weights
#
# There are two conceptually different weights:
#
# * ``mu[i]`` is the probability of observation $y_i$ and is normalized to
#   sum to one.
# * ``reference_volumes[a]`` weights the field inner product at reference
#   $x_a$. For now, the default is one at every reference, giving counting
#   measure on the finite reference set.

# %%
def _measure(mu, n):
    """Return ``n`` nonnegative observation probabilities that sum to one."""
    if mu is None:
        return np.full(n, 1.0 / n)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if (
        mu.shape != (n,)
        or not np.all(np.isfinite(mu))
        or np.any(mu < 0)
        or mu.sum() <= 0
    ):
        raise ValueError(f"mu must be {n} nonnegative weights with positive sum")
    return mu / mu.sum()


def _reference_volumes(reference_volumes, n_references):
    """Return direct-sum reference weights; the default is counting measure."""
    if reference_volumes is None:
        return np.ones(n_references)
    volumes = np.asarray(reference_volumes, dtype=float).reshape(-1)
    if (
        volumes.shape != (n_references,)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes < 0)
        or volumes.sum() <= 0
    ):
        raise ValueError(
            "reference_volumes must be nonnegative weights with positive sum"
        )
    return volumes.copy()


# %% [markdown]
# ## 3. Manifold information supplied by Geomstats
#
# The readable helper asks only for a manifold object ``space`` with a metric.
# It uses three metric operations:
#
# $$d(x,y)^2,\qquad \operatorname{Log}_x(y),\qquad
# g_x(u,v).$$
#
# In Geomstats these are ``metric.squared_dist``, ``metric.log``, and
# ``metric.inner_product``.

# %%
def _validate_points(data_points, reference_points, space):
    """Validate point-array shapes and return the manifold metric."""
    points = np.asarray(data_points, dtype=float)
    references = np.asarray(reference_points, dtype=float)
    if points.ndim < 2 or points.shape[0] < 2:
        raise ValueError("data_points must have shape (n, *point_shape), n >= 2")
    if references.ndim < 2 or references.shape[0] < 1:
        raise ValueError(
            "reference_points must have shape (r, *point_shape), r >= 1"
        )
    if points.shape[1:] != references.shape[1:]:
        raise ValueError(
            "data_points and reference_points have different point shapes"
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(references)):
        raise ValueError("data_points and reference_points must be finite")
    metric = getattr(space, "metric", None)
    if metric is None:
        raise TypeError("space must be a Geomstats-style manifold with space.metric")
    return points, references, metric


def _stacked(batched_call, single_call, items, expected_rows):
    """Evaluate a metric on a batch, falling back to a loop if unsupported.

    Geomstats metrics broadcast one base point against a stack of points, which
    turns O(n r) or O(n^2 r) Python-level metric calls into O(r) of them. Not
    every backend does, so a batched call whose shape is wrong is discarded and
    the readable per-item loop is used instead. The two paths are numerically
    identical; only the call count differs.
    """
    try:
        array = np.atleast_1d(np.asarray(batched_call(), dtype=float))
        if array.shape[0] == expected_rows:
            return array
    except Exception:
        pass
    return np.stack(
        [np.asarray(single_call(item), dtype=float) for item in items],
        axis=0,
    )


def _count_cut_locus_pairs(metric, base_point, distance_squared, logs, rtol=1e-6):
    """Count observations that reach the cut locus of one reference point.

    Beyond the injectivity radius the logarithm map is not uniquely defined,
    and backends do not agree on what to return: Geomstats hands back an
    exactly zero tangent vector at the antipode of a sphere, so the field
    silently vanishes instead of raising. Two independent detectors are used,
    because a metric need not expose an injectivity radius at all:

    * the geometric test, ``d(x, y) >= (1 - rtol) * injectivity_radius(x)``;
    * the backend-agnostic test, a zero logarithm at a strictly positive
      distance, which no well-defined logarithm map can produce.
    """
    n_observations = distance_squared.shape[0]
    flagged = np.zeros(n_observations, dtype=bool)

    try:
        radius = float(np.asarray(metric.injectivity_radius(base_point)).reshape(-1)[0])
    except Exception:
        radius = float("inf")
    if np.isfinite(radius):
        limit = (1.0 - float(rtol)) * radius
        flagged |= distance_squared >= limit * limit

    magnitudes = np.max(np.abs(logs.reshape(n_observations, -1)), axis=1)
    flagged |= (distance_squared > 0.0) & (magnitudes == 0.0)
    return int(np.count_nonzero(flagged))


# %% [markdown]
# ## 4. Constructing an observation field
#
# For each observation $y_i$ and each reference point $x_a$, the default field
# is
#
# $$F_i(x_a)=\frac{1}{2t}(4\pi t)^{-q/2}
# \exp\!\left(-\frac{d(x_a,y_i)^2}{4t}\right)
# \operatorname{Log}_{x_a}(y_i).$$
#
# This formula is the Euclidean heat-kernel gradient when the manifold is
# Euclidean. On a curved manifold it is a geodesic-Gaussian, or short-time
# heat-kernel model. Calling it a geodesic Gaussian prevents us from claiming
# that it is the exact heat kernel on an arbitrary manifold.

# %%
def geodesic_gaussian_gradient_fields(
    data_points,
    reference_points,
    t,
    space,
    *,
    manifold_dimension=None,
):
    r"""Construct the default short-time geodesic-Gaussian gradient fields.

    For observation ``y`` and reference ``x``, this returns

    .. math::

        F_y(x) = \frac{1}{2t}(4\pi t)^{-q/2}
        \exp\!\left[-\frac{d(x,y)^2}{4t}\right]\operatorname{Log}_x(y).

    This is the exact heat-kernel gradient in Euclidean space.  On a general
    curved manifold it is a geodesic-Gaussian / short-time model, not the
    exact heat-kernel gradient.  Data/reference pairs at the cut locus should
    be avoided because the logarithm map is not uniquely defined there.

    Returns an array with shape ``(n_observations, n_references, *tangent_shape)``.
    """
    points, references, metric = _validate_points(
        data_points, reference_points, space
    )
    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("t must be finite and positive")

    if manifold_dimension is None:
        manifold_dimension = getattr(space, "dim", None)
    if (
        isinstance(manifold_dimension, (bool, np.bool_))
        or not isinstance(manifold_dimension, (int, np.integer))
        or int(manifold_dimension) < 1
    ):
        raise ValueError(
            "manifold_dimension must be a positive integer or available as space.dim"
        )
    q = int(manifold_dimension)

    n_observations = points.shape[0]
    log_normalization = -(q / 2.0) * np.log(4.0 * np.pi * t)
    columns = []
    n_at_cut_locus = 0
    for x in references:
        distance_squared = _stacked(
            lambda: metric.squared_dist(x, points),
            lambda y: metric.squared_dist(x, y),
            points,
            n_observations,
        ).reshape(n_observations)
        if not np.all(np.isfinite(distance_squared)) or np.any(
            distance_squared < -1e-12
        ):
            raise ValueError("the metric returned an invalid squared distance")
        distance_squared = np.maximum(distance_squared, 0.0)

        logs = _stacked(
            lambda: metric.log(points, base_point=x),
            lambda y: metric.log(y, base_point=x),
            points,
            n_observations,
        )
        if not np.all(np.isfinite(logs)):
            raise ValueError("the metric returned a non-finite logarithm vector")

        n_at_cut_locus += _count_cut_locus_pairs(metric, x, distance_squared, logs)

        log_factor = (
            log_normalization - np.log(2.0 * t) - distance_squared / (4.0 * t)
        )
        factor = np.exp(log_factor)
        if not np.all(np.isfinite(factor)):
            raise ValueError(
                "the geodesic-Gaussian factor overflowed; use a larger t"
            )
        columns.append(factor.reshape((-1,) + (1,) * (logs.ndim - 1)) * logs)

    fields = np.stack(columns, axis=1)
    if n_at_cut_locus:
        warnings.warn(
            f"{n_at_cut_locus} observation/reference pair(s) sit at or beyond "
            "the injectivity radius, where Log_x(y) is not uniquely defined "
            "(Geomstats returns an exactly zero logarithm at the antipode of a "
            "sphere, so those fields silently vanish). Move those reference "
            "points or reduce t so the kernel damps the pairs.",
            RuntimeWarning,
            stacklevel=2,
        )
    if fields.size and np.max(np.abs(fields)) == 0:
        warnings.warn(
            "all geodesic-Gaussian fields are zero; consider a different t",
            RuntimeWarning,
            stacklevel=2,
        )
    return fields


# %% [markdown]
# ## 5. Inner product between sampled vector fields
#
# At a fixed reference $x_a$, both $F_i(x_a)$ and $F_j(x_a)$ belong to the
# same tangent space $T_{x_a}\mathcal M$, so Geomstats can evaluate
# $g_{x_a}(F_i(x_a),F_j(x_a))$.
#
# We never compare tangent vectors based at different references. Instead, we
# add the scalar pointwise inner products:
#
# $$G_{ij}=\langle F_i,F_j\rangle_R
# =\sum_a\omega_a g_{x_a}(F_i(x_a),F_j(x_a)).$$
#
# This is why the construction does not require parallel transport.

# %%
def field_inner_product_gram(
    fields,
    reference_points,
    space,
    reference_volumes=None,
):
    r"""Return the Gram matrix of sampled fields in the direct-sum metric.

    If ``fields[i, a]`` belongs to ``T_{x_a} M``, the returned matrix is

    .. math::

        G_{ij} = \sum_a \omega_a
        g_{x_a}(\mathrm{fields}_{i,a},\mathrm{fields}_{j,a}).

    No tangent vectors based at different reference points are compared.
    """
    fields = np.asarray(fields, dtype=float)
    references = np.asarray(reference_points, dtype=float)
    if fields.ndim < 3 or fields.shape[1] != references.shape[0]:
        raise ValueError(
            "fields must have shape (n_fields, n_references, *tangent_shape)"
        )
    if not np.all(np.isfinite(fields)):
        raise ValueError("fields must be finite")
    metric = getattr(space, "metric", None)
    if metric is None:
        raise TypeError("space must be a Geomstats-style manifold with space.metric")
    volumes = _reference_volumes(reference_volumes, references.shape[0])

    n_fields = fields.shape[0]
    gram = np.zeros((n_fields, n_fields))
    for a, base in enumerate(references):
        # Every field value in this block lives in the same tangent space
        # T_{x_a} M, so the metric can compare a whole column at once.
        block = fields[:, a]
        for i in range(n_fields):
            gram[:, i] += volumes[a] * _stacked(
                lambda: metric.inner_product(block, block[i], base_point=base),
                lambda v: metric.inner_product(v, block[i], base_point=base),
                block,
                n_fields,
            ).reshape(n_fields)
    return 0.5 * (gram + gram.T)


# %% [markdown]
# ## 6. The weighted dual covariance Gram matrix
#
# Let $\widetilde F_i=F_i-\sum_j\mu_jF_j$ be the centered fields, and let
# $G_{ij}=\langle\widetilde F_i,\widetilde F_j\rangle_R$. The covariance
# operator on the field space is
#
# $$\widehat\Sigma h=\sum_i\mu_i
# \langle\widetilde F_i,h\rangle_R\widetilde F_i.$$
#
# Rather than writing a coordinate matrix for this operator, we diagonalize
# the symmetric observation-space matrix
#
# $$\Gamma=D_{\sqrt\mu}GD_{\sqrt\mu}.$$
#
# Its nonzero eigenvalues are exactly the nonzero eigenvalues of
# $\widehat\Sigma$.

# %%
def weighted_dual_gram(field_gram, mu):
    r"""Return ``Gamma = D_sqrt(mu) G D_sqrt(mu)``.

    ``Gamma`` acts on observation coefficients.  Its nonzero eigenvalues are
    those of the covariance operator on the sampled field space.
    """
    field_gram = np.asarray(field_gram, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if field_gram.shape != (mu.size, mu.size):
        raise ValueError("field_gram and mu have incompatible shapes")
    sqrt_mu = np.sqrt(mu)
    gram = sqrt_mu[:, None] * field_gram * sqrt_mu[None, :]
    return 0.5 * (gram + gram.T)


# %% [markdown]
# ## 7. Complete RiePCA calculation
#
# The main function keeps the mathematical order visible:
#
# 1. construct one field per observation;
# 2. subtract the $\mu$-weighted mean field;
# 3. construct the Riemannian field Gram matrix $G$;
# 4. construct $\Gamma=D_{\sqrt\mu}GD_{\sqrt\mu}$;
# 5. compute its eigenpairs with ``scipy.linalg.eigh``;
# 6. reconstruct orthonormal principal fields;
# 7. project the centered observation fields onto them.
#
# If $\Gamma q_k=\lambda_kq_k$, then the reconstructed principal field is
#
# $$E_k=\frac{1}{\sqrt{\lambda_k}}
# \sum_i\sqrt{\mu_i}\,q_{ik}\widetilde F_i.$$
#
# If $c_{jk}=\sqrt{\mu_j}q_{jk}/\sqrt{\lambda_k}$, then the projection scores
# can be computed directly from the field Gram matrix:
#
# $$\langle\widetilde F_i,E_k\rangle_R
# =\sum_jG_{ij}c_{jk}=(Gc)_{ik}.$$

# %%
def riepca_manifold(
    data_points,
    reference_points,
    space,
    *,
    t,
    mu=None,
    reference_volumes=None,
    manifold_dimension=None,
    field_builder=None,
    k_keep=10,
    eig_rtol=1e-10,
    eig_atol=0.0,
):
    r"""Run centered RiePCA on one manifold at one time.

    By default, fields are produced by
    :func:`geodesic_gaussian_gradient_fields`.  To use an exact heat-kernel
    gradient, pass a callable with signature

    ``field_builder(data_points, reference_points, t, space)``

    returning shape ``(n_observations, n_references, *tangent_shape)`` with
    every ``field[i, a]`` in ``T_{reference_points[a]} M``.
    """
    points, references, _ = _validate_points(
        data_points,
        reference_points,
        space,
    )
    n_observations = points.shape[0]
    mu = _measure(mu, n_observations)
    volumes = _reference_volumes(reference_volumes, references.shape[0])

    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("t must be finite and positive")
    if not isinstance(k_keep, (int, np.integer)) or int(k_keep) < 1:
        raise ValueError("k_keep must be a positive integer")
    eig_rtol = float(eig_rtol)
    eig_atol = float(eig_atol)
    if eig_rtol < 0 or eig_atol < 0:
        raise ValueError("eig_rtol and eig_atol must be nonnegative")

    # 1. Construct one sampled tangent field for each observation.
    if field_builder is None:
        raw_fields = geodesic_gaussian_gradient_fields(
            points,
            references,
            t,
            space,
            manifold_dimension=manifold_dimension,
        )
        field_model = "geodesic_gaussian"
    else:
        raw_fields = np.asarray(
            field_builder(points, references, t, space),
            dtype=float,
        )
        field_model = getattr(field_builder, "__name__", "custom")

    expected_prefix = (n_observations, references.shape[0])
    if raw_fields.ndim < 3 or raw_fields.shape[:2] != expected_prefix:
        raise ValueError(
            "field_builder must return shape "
            "(n_observations, n_references, *tangent_shape)"
        )
    if not np.all(np.isfinite(raw_fields)):
        raise ValueError("field_builder returned non-finite fields")

    # 2. Center the fields over observations using the probability mu.
    mean_field = np.zeros_like(raw_fields[0])
    for i in range(n_observations):
        mean_field += mu[i] * raw_fields[i]
    centered_fields = raw_fields - mean_field[None, ...]

    # 3. Use the Riemannian metric at each reference to compare fields.
    field_gram = field_inner_product_gram(
        centered_fields,
        references,
        space,
        volumes,
    )

    # 4. Insert observation weights in the symmetric dual covariance Gram.
    dual_gram = weighted_dual_gram(field_gram, mu)
    total_variance = float(np.trace(dual_gram))

    # 5. Eigendecompose the symmetric positive-semidefinite dual Gram matrix.
    all_values, all_vectors = eigh(dual_gram, check_finite=False)
    all_values = all_values[::-1]
    all_vectors = all_vectors[:, ::-1]
    scale = max(float(np.max(np.abs(all_values))), 1.0)
    if np.any(all_values < -1e-10 * scale):
        raise ValueError("the dual Gram matrix is not positive semidefinite")
    all_values = np.maximum(all_values, 0.0)

    rank_capacity = max(int(np.count_nonzero(mu)) - 1, 0)
    if all_values.size and all_values[0] > 0:
        cutoff = max(eig_rtol * all_values[0], eig_atol)
        n_positive = int(np.count_nonzero(all_values > cutoff))
    else:
        n_positive = 0
    k = min(int(k_keep), rank_capacity, n_positive)
    eigenvalues = all_values[:k]
    # Copy: the sign convention below writes in place, and a slice of a
    # reversed view would flip the eigenvectors inside ``eigh``'s own output.
    dual_eigenvectors = all_vectors[:, :k].copy()

    tangent_shape = raw_fields.shape[2:]
    if k:
        # 6. Reconstruct metric-orthonormal principal vector fields.
        synthesis_coefficients = (
            np.sqrt(mu)[:, None] * dual_eigenvectors
        ) / np.sqrt(eigenvalues)[None, :]
        principal_field_list = []
        for j in range(k):
            field = np.zeros_like(centered_fields[0])
            for i in range(n_observations):
                field += synthesis_coefficients[i, j] * centered_fields[i]
            principal_field_list.append(field)
        pc_fields = np.stack(principal_field_list, axis=-1)

        # Pin arbitrary eigenvector signs so repeated runs plot consistently.
        flat = pc_fields.reshape(-1, k)
        anchors = np.argmax(np.abs(flat), axis=0)
        signs = np.sign(flat[anchors, np.arange(k)])
        signs[signs == 0] = 1.0
        pc_fields *= signs.reshape((1,) * (pc_fields.ndim - 1) + (-1,))
        synthesis_coefficients *= signs[None, :]
        dual_eigenvectors *= signs[None, :]

        # 7. Project the centered observation fields onto the PCs.
        #    G[i,j] = <f_i,f_j>, so G @ coefficients gives <f_i,E_k>.
        scores = field_gram @ synthesis_coefficients
    else:
        pc_fields = np.empty((references.shape[0],) + tangent_shape + (0,))
        scores = np.empty((n_observations, 0))

    ratio = (
        eigenvalues / total_variance
        if total_variance > 0
        else np.full(eigenvalues.shape, np.nan)
    )

    return RiePCAResult(
        t=t,
        field_model=field_model,
        eigenvalues=eigenvalues,
        pc_fields=pc_fields,
        scores=scores,
        mean_field=mean_field,
        total_variance=total_variance,
        explained_variance_ratio=ratio,
        cumulative_explained_variance_ratio=np.cumsum(ratio),
        rank_capacity=rank_capacity,
        mu=mu,
        reference_points=references.copy(),
        reference_volumes=volumes,
        field_gram=field_gram,
        dual_gram=dual_gram,
        dual_eigenvectors=dual_eigenvectors,
        raw_fields=raw_fields,
        centered_fields=centered_fields,
    )


# %% [markdown]
# ## 8. Mathematical checks
#
# ``check_riepca`` verifies the defining PCA identities. Every identity is
# stated in **dimensionless** form, which matters more than it looks. The
# fields carry the magnitude $(4\pi t)^{-q/2}/2t$ of the kernel, so $\lambda_1$
# ranges over many orders of magnitude as $t$ varies: on $S^8$ at $t=0.005$ it
# is around $3\times10^9$. Comparing a raw score against zero with a fixed
# absolute tolerance therefore demands cancellation to fifteen significant
# figures, and rejects results that are correct to machine precision. Dividing
# the scores by $\sqrt{\lambda_k}$ removes the scale.
#
# With whitened scores $z_{ik}=\mathrm{score}_{ik}/\sqrt{\lambda_k}$:
#
# $$\langle E_j,E_k\rangle_R=\delta_{jk},\qquad
# \sum_i\mu_iz_{ik}=0,\qquad
# \sum_i\mu_iz_{ij}z_{ik}=\delta_{jk}.$$
#
# Two further identities are checked because they exercise code the three
# above do not. The eigenfield relation confirms that $E_k$ really is an
# eigenvector of the covariance operator, not merely an orthonormal field:
#
# $$\widehat\Sigma E_k
# =\sum_i\mu_i\langle\widetilde F_i,E_k\rangle_R\widetilde F_i
# =\lambda_kE_k.$$
#
# And the total variance is recomputed straight from the metric,
# $\sum_i\mu_i\|\widetilde F_i\|_R^2$, rather than being read back off
# ``dual_gram``. Comparing ``total_variance`` with ``trace(dual_gram)`` would
# be vacuous: that trace is its definition.

# %%
def check_riepca(result, space, rtol=1e-8, atol=1e-8):
    """Verify the defining PCA identities in scale-free form.

    ``atol`` applies to the dimensionless identities and ``rtol`` to the one
    comparison that carries physical scale, the total variance. Both are
    meaningful at any heat time; an absolute tolerance on raw scores is not.
    """
    k = result.n_components
    metric = getattr(space, "metric", None)
    if metric is None:
        raise TypeError("space must be a Geomstats-style manifold with space.metric")

    # The centering step must have removed the mu-weighted mean exactly.
    residual_mean = np.tensordot(result.mu, result.centered_fields, axes=(0, 0))
    field_scale = max(float(np.max(np.abs(result.centered_fields), initial=0.0)), 1e-300)
    np.testing.assert_allclose(residual_mean / field_scale, 0.0, atol=atol)

    # The total variance, recomputed from the metric alone.
    direct_total = 0.0
    for a, base in enumerate(result.reference_points):
        block = result.centered_fields[:, a]
        squared_norms = _stacked(
            lambda: metric.inner_product(block, block, base_point=base),
            lambda v: metric.inner_product(v, v, base_point=base),
            block,
            block.shape[0],
        ).reshape(-1)
        direct_total += result.reference_volumes[a] * float(result.mu @ squared_norms)
    np.testing.assert_allclose(result.total_variance, direct_total, rtol=rtol)

    if k == 0:
        return True

    # The principal fields are orthonormal in the direct-sum metric.
    component_fields = np.moveaxis(result.pc_fields, -1, 0)
    component_gram = field_inner_product_gram(
        component_fields,
        result.reference_points,
        space,
        result.reference_volumes,
    )
    np.testing.assert_allclose(component_gram, np.eye(k), atol=atol)

    # Whitened scores are centered and have identity covariance.
    whitened = result.scores / np.sqrt(result.eigenvalues)[None, :]
    np.testing.assert_allclose(result.mu @ whitened, np.zeros(k), atol=atol)
    np.testing.assert_allclose(
        (whitened * result.mu[:, None]).T @ whitened,
        np.eye(k),
        atol=atol,
    )

    # Each principal field is an eigenfield: Sigma-hat E_k = lambda_k E_k.
    residuals = np.stack(
        [
            np.tensordot(
                result.mu * whitened[:, j],
                result.centered_fields,
                axes=(0, 0),
            )
            / np.sqrt(result.eigenvalues[j])
            - result.pc_fields[..., j]
            for j in range(k)
        ],
        axis=0,
    )
    residual_gram = field_inner_product_gram(
        residuals,
        result.reference_points,
        space,
        result.reference_volumes,
    )
    residual_norms = np.sqrt(np.maximum(np.diag(residual_gram), 0.0))
    np.testing.assert_allclose(residual_norms, np.zeros(k), atol=atol)

    # Retained variance cannot exceed the total.
    assert float(np.sum(result.eigenvalues)) <= result.total_variance * (1.0 + rtol)
    return True


# %% [markdown]
# ## 9. An exact heat-kernel field, on the two-sphere
#
# The default field is a geodesic-Gaussian *model*. On $S^2$ the true heat
# kernel is known in closed form as a Legendre series,
#
# $$k_t(x,y)=\sum_{l\ge0}e^{-l(l+1)t}\frac{2l+1}{4\pi}P_l(u),
# \qquad u=\langle x,y\rangle,$$
#
# so the model can be replaced by the exact gradient and the two compared.
# Since $k_t$ depends on $x$ only through $u$, and the gradient of $u$ on the
# sphere is the tangential projection $y-ux$,
#
# $$\nabla_xk_t(x,y)=\Big(\sum_{l\ge1}e^{-l(l+1)t}\frac{2l+1}{4\pi}
# P_l'(u)\Big)(y-ux),\qquad P_l'=C_{l-1}^{(3/2)},$$
#
# the Gegenbauer form of $P_l'$ being used because the recurrence for the
# Legendre derivative is unstable near $u=\pm1$.
#
# Pass this as ``field_builder`` to check how far the geodesic-Gaussian model
# is from the truth. On $S^2$ the relative difference is $\approx t/3$: about
# 3.5% at $t=0.1$ and 0.2% at $t=0.005$.

# %%
def sphere_heat_kernel_gradient_fields(
    data_points,
    reference_points,
    t,
    space,
    *,
    series_tol=1e-14,
):
    r"""Exact heat-kernel gradient fields on the unit two-sphere.

    Drop-in ``field_builder`` for :func:`riepca_manifold`.  Unlike the default
    geodesic Gaussian, this is the true ``grad_x k_t(x, y)`` -- but only for
    ``Hypersphere(dim=2)`` embedded in ``R^3``, because it relies on the
    Legendre expansion of the sphere's heat kernel.

    The series is truncated where ``exp(-l(l+1)t) < series_tol``, so the term
    count grows like ``1/sqrt(t)``; very small heat times are expensive.
    """
    from scipy.special import eval_gegenbauer

    points, references, _ = _validate_points(data_points, reference_points, space)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("this builder is specific to S^2 embedded in R^3")
    if getattr(space, "dim", 2) != 2:
        raise ValueError("this builder is specific to Hypersphere(dim=2)")
    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("t must be finite and positive")

    cosine = np.clip(points @ references.T, -1.0, 1.0)
    degree_max = int(np.ceil(0.5 * (np.sqrt(1.0 - 4.0 * np.log(series_tol) / t) - 1.0)))
    if degree_max > 4000:
        raise ValueError(
            f"the Legendre series needs {degree_max} terms at t={t}; use a "
            "larger heat time or a looser series_tol"
        )

    derivative = np.zeros_like(cosine)
    for degree in range(1, degree_max + 1):
        derivative += (
            np.exp(-degree * (degree + 1) * t)
            * (2 * degree + 1)
            / (4.0 * np.pi)
            * eval_gegenbauer(degree - 1, 1.5, cosine)
        )

    tangential = points[:, None, :] - cosine[:, :, None] * references[None, :, :]
    return derivative[:, :, None] * tangential


# %% [markdown]
# ## 10. Typical notebook use
#
# The standalone file accepts the Geomstats manifold itself; no package
# wrapper is needed. For example, on the two-sphere:
#
# ```python
# from geomstats.geometry.hypersphere import Hypersphere
# from riepca_manifold import riepca_manifold, check_riepca
#
# sphere = Hypersphere(dim=2)
# result = riepca_manifold(
#     data_points=points,          # shape (n, 3), with unit-length rows
#     reference_points=references,# shape (r, 3), with unit-length rows
#     space=sphere,
#     t=0.1,
#     mu=mu,
#     k_keep=3,
# )
# check_riepca(result, sphere)
#
# eigenvalues = result.eigenvalues
# principal_fields = result.pc_fields
# scores = result.scores
# ```
#
# With $r$ sphere references and $k$ retained components,
# ``result.pc_fields`` has shape ``(r, 3, k)``. The last axis is zero-indexed,
# so ``result.pc_fields[a, :, j]`` is PC $j+1$ evaluated at reference $x_a$,
# and ``result.pc_fields[:, :, 0]`` is the whole leading principal field.
# (``[a, :, k]`` would be out of bounds: the valid indices stop at $k-1$.)
