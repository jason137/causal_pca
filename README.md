# causal_pca

Online PCA via EWMA covariance tracking. Numba-accelerated, strictly
causal, dependency-light (`numpy` + `numba`).

The projection at time *t* uses only state estimated from times ≤ *t-1*.
The EWMA covariance is updated *after* the projection is emitted — so a
score never sees its own observation. This makes the component output
safe to use as a predictive feature: no contemporaneous leakage, no
lookahead.

## Why EWMA covariance

Batch PCA assumes a single stationary covariance. Streaming data
(order flow, sensor arrays, factor returns) drifts: the dominant
direction at noon is not the dominant direction at close. An
exponentially-weighted covariance forgets stale observations at a
configurable half-life, so the leading components track the *current*
regime rather than the time-averaged one.

## Usage

```python
import numpy as np
from causal_pca import EWMAPca

pca = EWMAPca(n_dim=4, halflife=200, burn=100, n_components=2)

for x_t in stream:                # x_t: shape (4,)
    score = pca.step(x_t)         # shape (2,); zeros during burn-in
    # ... use score as a causal feature ...

pca.loadings                      # (4, 2) current PC directions
pca.explained_variance_ratio      # (2,) fraction of variance per PC
```

`step()` accumulates a buffer for `burn` steps, seeds the EWMA mean/cov
from its sample statistics, then switches to recursive updates. Set
`refit_interval > 1` to re-eigendecompose less often (negligible for
small `n_dim`; eigendecomposition is O(d³) but d is typically ≤ 8).

## Design

- **Hot path** (`_ewma_mean_cov_update`, `_project`) is numba `njit`,
  updating only the upper triangle of the covariance and mirroring.
- **Eigendecomposition** uses `np.linalg.eigh` (symmetric, real
  spectrum) at `refit_interval` cadence.
- **Causality** is structural: `step()` projects with the *current*
  loadings, then updates state. No configuration can break it.

## Tests

```bash
poetry install
poetry run pytest tests/              # 33 unit tests
poetry run python scripts/verify_causality.py
```

`tests/` covers kernel convergence to true mean/covariance, PSD
preservation, loading orthogonality/unit-norm, explained-variance
ordering, and non-stationary regime tracking.

`scripts/verify_causality.py` is a standalone causal-correctness harness:

1. **Oracle mirror** — replays the algorithm with explicit pre-update
   state; every score matches to float64 precision (~1e-16).
2. **Spike intervention** — injects a shock into one of two identical
   streams; scores are bit-identical before the shock, divergent after.
3. **Future-score independence** — over 50k iid samples, the correlation
   between score(*t*) and input(*t+1*) is statistically zero across all
   component/dimension pairs.

