"""Tests for causal_pca — causal EWMA-based online PCA."""

from __future__ import annotations

import numpy as np
from causal_pca import EWMAPca, _ewma_mean_cov_update, _project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_correlated(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate n samples of d correlated features with known structure."""
    rng = np.random.default_rng(seed)
    L = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1):
            L[i, j] = 1.0 / (1 + abs(i - j))
    return rng.standard_normal((n, d)) @ L.T


def _run_through(pca: EWMAPca, data: np.ndarray) -> np.ndarray:
    """Feed all rows through pca.step, return last score."""
    s = np.zeros(pca.n_components)
    for t in range(len(data)):
        s = pca.step(data[t])
    return s


# ---------------------------------------------------------------------------
# Kernel: _ewma_mean_cov_update
# ---------------------------------------------------------------------------

class TestEwmaMeanCovUpdate:
    def test_instant_adapt(self):
        """With alpha=1, EWMA snaps to the observation."""
        mu = np.zeros(3)
        cov = np.zeros((3, 3))
        x = np.array([1.0, 2.0, 3.0])
        _ewma_mean_cov_update(x, mu, cov, alpha=1.0)
        np.testing.assert_allclose(mu, x)

    def test_inplace(self):
        mu = np.zeros(2)
        cov = np.zeros((2, 2))
        mu_id, cov_id = id(mu), id(cov)
        _ewma_mean_cov_update(np.array([4.0, -1.0]), mu, cov, alpha=0.5)
        assert id(mu) == mu_id and id(cov) == cov_id
        assert not np.allclose(mu, 0.0)

    def test_symmetry(self):
        d = 4
        mu, cov = np.zeros(d), np.zeros((d, d))
        rng = np.random.default_rng(7)
        for _ in range(200):
            _ewma_mean_cov_update(rng.standard_normal(d), mu, cov, alpha=0.02)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)

    def test_positive_semidefinite(self):
        """Covariance must stay PSD after many updates."""
        d = 5
        mu, cov = np.zeros(d), np.eye(d) * 0.01
        rng = np.random.default_rng(13)
        for _ in range(2000):
            _ewma_mean_cov_update(rng.standard_normal(d), mu, cov, alpha=0.01)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > -1e-12)

    def test_converges_to_true_mean(self):
        """Given iid N(target, I), EWMA mean converges to target."""
        d = 3
        target = np.array([5.0, -3.0, 1.0])
        mu, cov = np.zeros(d), np.zeros((d, d))
        rng = np.random.default_rng(42)
        for _ in range(50_000):
            x = rng.standard_normal(d) + target
            _ewma_mean_cov_update(x, mu, cov, alpha=0.005)
        np.testing.assert_allclose(mu, target, atol=0.15)

    def test_converges_to_true_cov(self):
        """Given iid N(0, Sigma), EWMA cov converges to Sigma."""
        d = 3
        L = np.array([[2.0, 0, 0], [0.5, 1.5, 0], [0.1, -0.3, 1.0]])
        true_cov = L @ L.T
        mu, cov = np.zeros(d), np.eye(d)
        rng = np.random.default_rng(99)
        for _ in range(100_000):
            x = L @ rng.standard_normal(d)
            _ewma_mean_cov_update(x, mu, cov, alpha=0.002)
        np.testing.assert_allclose(cov, true_cov, atol=0.25)


# ---------------------------------------------------------------------------
# Kernel: _project
# ---------------------------------------------------------------------------

class TestProject:
    def test_identity_loadings(self):
        x = np.array([3.0, 1.0])
        mu = np.array([1.0, 1.0])
        scores = _project(x, mu, np.eye(2))
        np.testing.assert_allclose(scores, [2.0, 0.0])

    def test_single_component(self):
        x = np.array([1.0, 0.0, 0.0])
        loadings = np.array([[1.0], [0.0], [0.0]])
        scores = _project(x, np.zeros(3), loadings)
        assert scores.shape == (1,)
        np.testing.assert_allclose(scores, [1.0])

    def test_orthogonal_projection(self):
        """Projecting onto orthonormal basis recovers full centered vector."""
        x = np.array([3.0, 4.0])
        mu = np.array([1.0, 1.0])
        loadings = np.eye(2)
        scores = _project(x, mu, loadings)
        reconstructed = scores @ loadings.T
        np.testing.assert_allclose(reconstructed, x - mu)


# ---------------------------------------------------------------------------
# EWMAPca: lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_burn_in_returns_zeros(self):
        pca = EWMAPca(n_dim=3, halflife=50, burn=10)
        for t in range(10):
            s = pca.step(np.ones(3) * t)
            np.testing.assert_array_equal(s, [0.0])

    def test_warm_after_burn(self):
        pca = EWMAPca(n_dim=3, halflife=50, burn=10)
        for t in range(10):
            pca.step(np.ones(3) * t)
        assert pca.warm

    def test_not_warm_before_burn(self):
        pca = EWMAPca(n_dim=3, halflife=50, burn=10)
        for t in range(9):
            pca.step(np.ones(3) * t)
        assert not pca.warm

    def test_burn_buffer_freed(self):
        """Internal buffer should be released after burn-in."""
        pca = EWMAPca(n_dim=3, halflife=50, burn=10)
        for t in range(10):
            pca.step(np.ones(3) * t)
        assert pca._buf is None

    def test_minimum_burn(self):
        """burn < 2 should be clamped to 2."""
        pca = EWMAPca(n_dim=2, halflife=50, burn=0)
        assert pca.burn == 2

    def test_seed_matches_sample_stats(self):
        """After burn-in, internal mean/cov should match buffer sample stats."""
        d, burn = 3, 50
        pca = EWMAPca(n_dim=d, halflife=200, burn=burn)
        rng = np.random.default_rng(0)
        buf = rng.standard_normal((burn, d))
        for t in range(burn):
            pca.step(buf[t])
        expected_mu = np.mean(buf, axis=0)
        centered = buf - expected_mu
        expected_cov = (centered.T @ centered) / (burn - 1)
        np.testing.assert_allclose(pca._mu, expected_mu, atol=1e-12)
        np.testing.assert_allclose(pca._cov, expected_cov, atol=1e-12)


# ---------------------------------------------------------------------------
# EWMAPca: shapes
# ---------------------------------------------------------------------------

class TestShapes:
    def test_output_shape_single(self):
        pca = EWMAPca(n_dim=4, halflife=100, burn=20, n_components=1)
        s = _run_through(pca, _synth_correlated(100, 4))
        assert s.shape == (1,)

    def test_output_shape_multi(self):
        pca = EWMAPca(n_dim=4, halflife=100, burn=20, n_components=2)
        s = _run_through(pca, _synth_correlated(100, 4))
        assert s.shape == (2,)

    def test_loadings_shape(self):
        pca = EWMAPca(n_dim=4, halflife=100, burn=20, n_components=2)
        _run_through(pca, _synth_correlated(100, 4))
        assert pca.loadings.shape == (4, 2)

    def test_n_components_equals_n_dim(self):
        """Edge case: retaining all components."""
        d = 3
        pca = EWMAPca(n_dim=d, halflife=100, burn=20, n_components=d)
        _run_through(pca, _synth_correlated(100, d))
        assert pca.loadings.shape == (d, d)
        assert pca.explained_variance_ratio.shape == (d,)


# ---------------------------------------------------------------------------
# EWMAPca: statistical properties
# ---------------------------------------------------------------------------

class TestStatistical:
    def test_pc1_captures_dominant_direction(self):
        d = 4
        pca = EWMAPca(n_dim=d, halflife=200, burn=50, n_components=1)
        rng = np.random.default_rng(99)
        scales = np.array([10.0, 1.0, 1.0, 1.0])
        _run_through(pca, rng.standard_normal((2000, d)) * scales)
        assert np.argmax(np.abs(pca.loadings[:, 0])) == 0

    def test_explained_variance_ordered(self):
        """PC1 explains more variance than PC2."""
        pca = EWMAPca(n_dim=4, halflife=200, burn=50, n_components=3)
        _run_through(pca, _synth_correlated(1000, 4))
        evr = pca.explained_variance_ratio
        assert evr[0] >= evr[1] >= evr[2]

    def test_explained_variance_sums_below_one(self):
        pca = EWMAPca(n_dim=4, halflife=200, burn=50, n_components=2)
        _run_through(pca, _synth_correlated(500, 4))
        assert pca.explained_variance_ratio.sum() <= 1.0 + 1e-10

    def test_loadings_unit_norm(self):
        pca = EWMAPca(n_dim=4, halflife=100, burn=30, n_components=2)
        _run_through(pca, _synth_correlated(200, 4))
        for k in range(2):
            np.testing.assert_allclose(
                np.linalg.norm(pca.loadings[:, k]), 1.0, atol=1e-10
            )

    def test_loadings_orthogonal(self):
        """Retained PCs must be mutually orthogonal."""
        pca = EWMAPca(n_dim=4, halflife=100, burn=30, n_components=3)
        _run_through(pca, _synth_correlated(500, 4))
        G = pca.loadings.T @ pca.loadings
        np.testing.assert_allclose(G, np.eye(3), atol=1e-10)

    def test_explained_variance_zero_cov(self):
        """Constant input → zero covariance → zero explained ratio."""
        pca = EWMAPca(n_dim=2, halflife=50, burn=10)
        for t in range(100):
            pca.step(np.array([1.0, 1.0]))
        evr = pca.explained_variance_ratio
        np.testing.assert_allclose(evr, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# EWMAPca: causality
# ---------------------------------------------------------------------------

class TestCausality:
    def test_no_lookahead(self):
        """Score at t must not depend on data at t+1.

        Feed the same prefix followed by different futures — scores at
        the last shared step must be identical.
        """
        d = 3
        n_shared = 100
        rng = np.random.default_rng(0)
        shared = rng.standard_normal((n_shared, d))
        future_a = rng.standard_normal((50, d))
        future_b = rng.standard_normal((50, d)) * 5.0
        full_a = np.vstack([shared, future_a])
        full_b = np.vstack([shared, future_b])

        s_ref = _run_through(
            EWMAPca(n_dim=d, halflife=50, burn=20), shared
        )
        s_a: np.ndarray = np.zeros(1)
        pca_a = EWMAPca(n_dim=d, halflife=50, burn=20)
        for t in range(n_shared):
            s_a = pca_a.step(full_a[t])

        s_b: np.ndarray = np.zeros(1)
        pca_b = EWMAPca(n_dim=d, halflife=50, burn=20)
        for t in range(n_shared):
            s_b = pca_b.step(full_b[t])

        np.testing.assert_allclose(float(s_ref[0]), float(s_a[0]))
        np.testing.assert_allclose(float(s_ref[0]), float(s_b[0]))

    def test_step_order_matters(self):
        """Permuting input order must change scores — confirms state dependence."""
        d = 3
        rng = np.random.default_rng(5)
        data = rng.standard_normal((200, d))
        perm = np.concatenate([data[100:], data[:100]])

        pca_fwd = EWMAPca(n_dim=d, halflife=50, burn=20)
        pca_rev = EWMAPca(n_dim=d, halflife=50, burn=20)
        s_fwd = _run_through(pca_fwd, data)
        s_rev = _run_through(pca_rev, perm)
        assert not np.allclose(s_fwd, s_rev)

    def test_project_before_update(self):
        """Score at step t must use loadings from before the t-th update.

        Manually verify: snapshot loadings, step, confirm score matches
        projection with pre-step loadings.
        """
        d = 3
        pca = EWMAPca(n_dim=d, halflife=50, burn=20)
        rng = np.random.default_rng(88)
        data = rng.standard_normal((50, d))
        for t in range(50):
            pca.step(data[t])

        x_test = rng.standard_normal(d)
        loadings_before = pca.loadings.copy()
        mu_before = pca._mu.copy()
        score = pca.step(x_test)

        expected = _project(x_test, mu_before, loadings_before)
        np.testing.assert_allclose(score, expected)


# ---------------------------------------------------------------------------
# EWMAPca: non-stationarity tracking
# ---------------------------------------------------------------------------

class TestNonStationary:
    def test_tracks_regime_shift(self):
        """After a variance regime shift, PC1 should rotate to the new
        dominant direction within O(halflife) steps.
        """
        d = 3
        hl = 200
        pca = EWMAPca(n_dim=d, halflife=hl, burn=50, n_components=1)
        rng = np.random.default_rng(77)

        # regime 1: dim 0 dominant
        scales_1 = np.array([10.0, 1.0, 1.0])
        for _ in range(2000):
            pca.step(rng.standard_normal(d) * scales_1)
        assert np.argmax(np.abs(pca.loadings[:, 0])) == 0

        # regime 2: dim 2 dominant
        scales_2 = np.array([1.0, 1.0, 10.0])
        for _ in range(4000):
            pca.step(rng.standard_normal(d) * scales_2)
        assert np.argmax(np.abs(pca.loadings[:, 0])) == 2

    def test_ewma_forgets_old_data(self):
        """EWMA covariance should forget stale observations — short HL
        adapts faster than long HL after a shift.
        """
        d = 2
        rng = np.random.default_rng(33)
        pca_fast = EWMAPca(n_dim=d, halflife=50, burn=20)
        pca_slow = EWMAPca(n_dim=d, halflife=2000, burn=20)

        # train both on dim-0 dominant
        for _ in range(5000):
            x = rng.standard_normal(d) * np.array([10.0, 1.0])
            pca_fast.step(x)
            pca_slow.step(x)

        # shift to dim-1 dominant, run for ~4x fast HL but <0.2x slow HL
        for _ in range(200):
            x = rng.standard_normal(d) * np.array([1.0, 10.0])
            pca_fast.step(x)
            pca_slow.step(x)

        # fast should have rotated, slow should still point at dim 0
        assert np.argmax(np.abs(pca_fast.loadings[:, 0])) == 1
        assert np.argmax(np.abs(pca_slow.loadings[:, 0])) == 0


# ---------------------------------------------------------------------------
# EWMAPca: refit interval
# ---------------------------------------------------------------------------

class TestRefitInterval:
    def test_fewer_updates_with_high_interval(self):
        d = 3
        data = _synth_correlated(200, d, seed=11)
        pca_fast = EWMAPca(n_dim=d, halflife=50, burn=20, refit_interval=1)
        pca_slow = EWMAPca(n_dim=d, halflife=50, burn=20, refit_interval=50)

        loadings_fast, loadings_slow = [], []
        for t in range(200):
            pca_fast.step(data[t])
            pca_slow.step(data[t])
            if t > 20:
                loadings_fast.append(pca_fast.loadings.copy())
                loadings_slow.append(pca_slow.loadings.copy())

        changes_fast = sum(
            1 for i in range(1, len(loadings_fast))
            if not np.allclose(loadings_fast[i], loadings_fast[i - 1])
        )
        changes_slow = sum(
            1 for i in range(1, len(loadings_slow))
            if not np.allclose(loadings_slow[i], loadings_slow[i - 1])
        )
        assert changes_slow < changes_fast

    def test_refit_interval_one_is_default(self):
        pca = EWMAPca(n_dim=2, halflife=50)
        assert pca.refit_interval == 1

    def test_refit_interval_clamped(self):
        pca = EWMAPca(n_dim=2, halflife=50, refit_interval=0)
        assert pca.refit_interval == 1
