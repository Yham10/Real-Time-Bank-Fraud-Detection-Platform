import numpy as np, pandas as pd


def _rows(features, n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, len(features))), columns=features)


def test_score_shape_and_range(streaming, model, scaler):
    df = _rows(streaming.ALL_FEATURES)
    p = streaming.score_features(df, model, scaler)
    assert p.shape == (10,)
    assert np.all((p >= 0) & (p <= 1))


def test_scoring_is_column_order_invariant(streaming, model, scaler):
    """Serving may receive columns in any order; score_features must reorder."""
    df = _rows(streaming.ALL_FEATURES)
    shuffled = df[list(reversed(streaming.ALL_FEATURES))]
    np.testing.assert_allclose(
        streaming.score_features(df, model, scaler),
        streaming.score_features(shuffled, model, scaler),
    )


def test_scoring_does_not_mutate_input(streaming, model, scaler):
    df = _rows(streaming.ALL_FEATURES)
    before = df.copy()
    streaming.score_features(df, model, scaler)
    pd.testing.assert_frame_equal(df, before)