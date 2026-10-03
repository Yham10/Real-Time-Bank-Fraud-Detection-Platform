import numpy as np
from scipy import stats

# Scores are extremely skewed (most ≈ 0), so quantile bins would be unstable.
SCORE_EDGES = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95])


def make_bins(reference, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges computed on the reference distribution."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    qs = np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(qs)          # ties collapse into fewer bins


def bin_proportions(values, edges: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    idx = np.searchsorted(edges, v, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(expected, eps, None)
    a = np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(reference, current) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    return float(stats.ks_2samp(ref[~np.isnan(ref)], cur[~np.isnan(cur)]).statistic)