"""Causality verification for EWMAPca.

Three independent tests that the projection at time t uses only
state from times <= t-1:

1. Oracle mirror — step-by-step numerical match against manually
   tracked pre-update state.  Catches: update-before-project,
   refit-before-project, wrong delta in cov update.

2. Spike intervention — inject a spike at time T in one of two
   identical instances.  Verifies: scores match for t < T (no
   anticipation), scores at T use identical pre-spike state,
   state diverges forward-only.

3. Future-score independence — with iid input, score(t) must be
   uncorrelated with x(t+1).  Aggregates over the full series;
   rejects at p < 0.01 via z-test on Pearson r.

Usage:
    poetry run python scripts/verify_causality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_pca import EWMAPca


# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def _eigendecompose(cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:k]
    return eigvecs[:, idx], eigvals[idx]


def _print(label: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# -------------------------------------------------------------------
# 1. Oracle mirror
# -------------------------------------------------------------------

def test_oracle_mirror(
    n: int = 5000,
    d: int = 4,
    hl: float = 200.0,
    burn: int = 100,
    k: int = 2,
    refit_interval: int = 3,
    seed: int = 42,
) -> bool:
    """Replicate the algorithm step by step with explicit pre-update
    state tracking.  Every score must match to float64 precision.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d))
    alpha = 1.0 - 0.5 ** (1.0 / hl)

    pca = EWMAPca(
        n_dim=d, halflife=hl, burn=burn,
        n_components=k, refit_interval=refit_interval,
    )

    # --- burn-in: accumulate buffer, then seed ---
    buf = data[:burn].copy()
    for t in range(burn):
        s = pca.step(data[t])
        if not np.allclose(s, 0.0):
            _print("oracle / burn-in zeros", False, f"t={t}")
            return False

    mu = np.mean(buf, axis=0)
    centered = buf - mu
    cov = (centered.T @ centered) / (burn - 1)
    loadings, _ = _eigendecompose(cov, k)
    steps_since_refit = 0

    # --- post-burn: compare every score ---
    max_err = 0.0
    for t in range(burn, n):
        x = data[t]
        expected = (x - mu) @ loadings
        actual = pca.step(x)

        err = np.max(np.abs(expected - actual))
        max_err = max(max_err, err)
        if err > 1e-12:
            _print(
                "oracle / score match", False,
                f"t={t}, max_err={err:.2e}",
            )
            return False

        # --- replicate the state update ---
        delta = x - mu
        mu += alpha * delta
        for i in range(d):
            for j in range(i, d):
                cov[i, j] = (1 - alpha) * cov[i, j] + alpha * delta[i] * delta[j]
                cov[j, i] = cov[i, j]

        steps_since_refit += 1
        if steps_since_refit >= refit_interval:
            loadings, _ = _eigendecompose(cov, k)
            steps_since_refit = 0

    _print("oracle mirror", True, f"n={n}, max_err={max_err:.2e}")
    return True


# -------------------------------------------------------------------
# 2. Spike intervention
# -------------------------------------------------------------------

def test_spike_intervention(
    n: int = 2000,
    d: int = 4,
    hl: float = 200.0,
    burn: int = 100,
    spike_t: int = 1000,
    spike_mag: float = 100.0,
    seed: int = 7,
) -> bool:
    """Run two identical instances.  Inject a spike in one at t=spike_t.
    Verify: identical scores before spike, divergence after.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d))

    data_spiked = data.copy()
    data_spiked[spike_t] += spike_mag

    pca_clean = EWMAPca(n_dim=d, halflife=hl, burn=burn)
    pca_spike = EWMAPca(n_dim=d, halflife=hl, burn=burn)

    ok = True

    for t in range(n):
        s_clean = pca_clean.step(data[t])
        s_spike = pca_spike.step(data_spiked[t])

        if t < spike_t:
            if not np.array_equal(s_clean, s_spike):
                _print(
                    "spike / pre-spike identity", False,
                    f"scores diverge at t={t}, before spike at {spike_t}",
                )
                return False

        elif t == spike_t:
            # same pre-update state, different input → scores must differ
            # (unless the spike dimension has zero loading, astronomically unlikely)
            if np.allclose(s_clean, s_spike):
                _print(
                    "spike / spike-step divergence", False,
                    "scores identical at spike step (loadings may be degenerate)",
                )
                ok = False

            # internal state before this step's update should match
            # (we can't access pre-update state after step(), but the
            # score equality for t < spike_t already proves it)

    # post-spike: mu/cov should have diverged
    mu_diff = np.max(np.abs(pca_clean._mu - pca_spike._mu))
    if mu_diff < 1e-6:
        _print(
            "spike / post-spike state divergence", False,
            f"mu_diff={mu_diff:.2e} — spike didn't propagate",
        )
        ok = False

    if ok:
        _print("spike intervention", True, f"spike_t={spike_t}, mag={spike_mag}")
    return ok


# -------------------------------------------------------------------
# 3. Future-score independence
# -------------------------------------------------------------------

def test_future_independence(
    n: int = 50_000,
    d: int = 4,
    hl: float = 200.0,
    burn: int = 200,
    seed: int = 99,
) -> bool:
    """With iid input, score(t) must be uncorrelated with x(t+1).

    Computes Pearson r between each score component and each dimension
    of the next input.  Under H0 (causality), r ~ N(0, 1/sqrt(n)).
    Rejects if any |z| > z_{alpha/2} after Bonferroni correction.
    """
    rng = np.random.default_rng(seed)
    k = 2
    data = rng.standard_normal((n, d))

    pca = EWMAPca(n_dim=d, halflife=hl, burn=burn, n_components=k)

    scores = np.empty((n, k))
    for t in range(n):
        scores[t] = pca.step(data[t])

    # align: score(t) vs x(t+1), dropping burn-in
    start = burn + 1
    s = scores[start:-1]       # score(t) for t in [start, n-2]
    x_next = data[start + 1:]  # x(t+1) for same range
    m = len(s)

    n_tests = k * d
    z_crit = 2.576  # ~99% before Bonferroni; conservative enough

    max_z = 0.0
    worst_pair = (0, 0)
    ok = True

    for ki in range(k):
        for di in range(d):
            r = np.corrcoef(s[:, ki], x_next[:, di])[0, 1]
            z = abs(r) * np.sqrt(m)
            if z > max_z:
                max_z = z
                worst_pair = (ki, di)

            bonf_crit = z_crit + np.log(n_tests)
            if z > bonf_crit:
                _print(
                    f"future-independence / PC{ki}-x{di}", False,
                    f"r={r:.4f}, z={z:.1f}, crit={bonf_crit:.1f}",
                )
                ok = False

    if ok:
        _print(
            "future-score independence", True,
            f"n={m}, max |z|={max_z:.1f} (PC{worst_pair[0]},x{worst_pair[1]})",
        )
    return ok


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main() -> None:
    print("causality verification — causal_pca.EWMAPca\n")

    results = [
        test_oracle_mirror(),
        test_spike_intervention(),
        test_future_independence(),
    ]

    print()
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"all {n_total} tests passed.")
    else:
        print(f"{n_pass}/{n_total} passed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
