"""Causal EWMA-based online PCA.

Maintains EWMA estimates of mean and covariance, eigendecomposes at
configurable intervals, and projects input vectors onto the top-k PCs.
All updates are strictly causal: the projection at time t uses only
data from times <= t-1 (EWMA state is updated *after* projection).

The EWMA mean/cov update is the hot path and is numba-accelerated.
Eigendecomposition uses np.linalg.eigh (d=4 -> trivially cheap).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _ewma_mean_cov_update(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    alpha: float,
) -> None:
    """In-place EWMA update of mean and covariance.

    Args:
        x: Input vector, shape (d,).
        mu: Running mean, shape (d,). Modified in-place.
        cov: Running covariance, shape (d, d). Modified in-place.
        alpha: Smoothing factor (= 1 - 2^(-1/hl)).
    """
    d = len(x)
    delta = np.empty(d)
    for i in range(d):
        delta[i] = x[i] - mu[i]
        mu[i] += alpha * delta[i]
    for i in range(d):
        for j in range(i, d):
            cov[i, j] = (1.0 - alpha) * cov[i, j] + alpha * delta[i] * delta[j]
            cov[j, i] = cov[i, j]


@njit(cache=True)
def _project(x: np.ndarray, mu: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Project centered x onto loadings.

    Args:
        x: Input vector, shape (d,).
        mu: Running mean, shape (d,).
        loadings: Eigenvector matrix, shape (d, k). Columns are PCs.

    Returns:
        Scores, shape (k,).
    """
    d = loadings.shape[0]
    k = loadings.shape[1]
    out = np.zeros(k)
    for j in range(k):
        s = 0.0
        for i in range(d):
            s += (x[i] - mu[i]) * loadings[i, j]
        out[j] = s
    return out


class EWMAPca:
    """Online PCA via EWMA covariance tracking.

    Args:
        n_dim: Number of input features.
        halflife: EWMA half-life in steps.
        burn: Burn-in length — accumulates buffer, seeds EWMA from
            sample mean/cov before producing output.
        n_components: Number of PCs to retain (default 1).
        refit_interval: Re-eigendecompose every N steps (default 1).
            For d<=8 there is no reason to skip steps.
    """

    def __init__(
        self,
        n_dim: int,
        halflife: float,
        burn: int = 500,
        n_components: int = 1,
        refit_interval: int = 1,
    ) -> None:
        self.n_dim = n_dim
        self.halflife = halflife
        self.burn = max(burn, 2)
        self.n_components = n_components
        self.refit_interval = max(refit_interval, 1)
        self._alpha = 1.0 - 0.5 ** (1.0 / halflife)

        self._buf: np.ndarray | None = np.zeros((self.burn, n_dim))
        self._n = 0
        self._steps_since_refit = 0

        self._mu = np.zeros(n_dim)
        self._cov = np.eye(n_dim)
        self._loadings = np.zeros((n_dim, n_components))
        self._explained_var = np.zeros(n_components)
        self._warm = False

    @property
    def warm(self) -> bool:
        """True once burn-in has elapsed and loadings are available."""
        return self._warm

    @property
    def loadings(self) -> np.ndarray:
        """Current PC loadings, shape (n_dim, n_components)."""
        return self._loadings

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of variance explained by each retained PC."""
        total = np.trace(self._cov)
        if total < 1e-15:
            return np.zeros(self.n_components)
        return self._explained_var / total

    def step(self, x: np.ndarray) -> np.ndarray:
        """Causal step: project x using *current* loadings, then update state.

        Args:
            x: Input vector, shape (n_dim,).

        Returns:
            PC scores, shape (n_components,). Zeros during burn-in.
        """
        if self._n < self.burn:
            assert self._buf is not None
            self._buf[self._n] = x
            self._n += 1
            if self._n == self.burn:
                self._seed()
            return np.zeros(self.n_components)

        scores = _project(x, self._mu, self._loadings)

        _ewma_mean_cov_update(x, self._mu, self._cov, self._alpha)
        self._n += 1
        self._steps_since_refit += 1

        if self._steps_since_refit >= self.refit_interval:
            self._refit()

        return scores

    def _seed(self) -> None:
        """Seed EWMA mean/cov from burn-in buffer and compute initial loadings."""
        assert self._buf is not None
        buf = self._buf
        self._mu = np.mean(buf, axis=0)
        centered = buf - self._mu
        self._cov = (centered.T @ centered) / max(self.burn - 1, 1)
        self._buf = None
        self._refit()
        self._warm = True

    def _refit(self) -> None:
        """Eigendecompose current covariance and update loadings."""
        eigvals, eigvecs = np.linalg.eigh(self._cov)
        idx = np.argsort(eigvals)[::-1][:self.n_components]
        self._loadings = eigvecs[:, idx]
        self._explained_var = eigvals[idx]
        self._steps_since_refit = 0
