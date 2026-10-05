import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "drift"))

import numpy as np
from drift_math import make_bins, bin_proportions, psi, ks_statistic

def test_psi_identical_is_near_zero():
    ref = np.random.randn(3000)
    bins = make_bins(ref, n_bins=10)
    p_ref = bin_proportions(ref, bins)
    p_cur = bin_proportions(ref, bins)
    assert psi(p_ref, p_cur) < 0.01
    assert ks_statistic(ref, ref) < 0.01

def test_psi_shifted_is_high():
    ref = np.random.randn(3000)
    cur = ref + 3.0
    bins = make_bins(ref, n_bins=10)
    assert psi(bin_proportions(ref, bins), bin_proportions(cur, bins)) > 0.25
    assert ks_statistic(ref, cur) > 0.5

def test_random_3000_vs_sequential_logic():
    # reproduces your baseline check: random vs sequential
    ref = np.random.randn(10000)
    random_cur = np.random.randn(3000)
    seq_cur = np.random.randn(3000) + 1.5 # simulate time slice shift
    bins = make_bins(ref)
    psi_random = psi(bin_proportions(ref, bins), bin_proportions(random_cur, bins))
    psi_seq = psi(bin_proportions(ref, bins), bin_proportions(seq_cur, bins))
    assert psi_random < 0.1
    assert psi_seq > 0.25