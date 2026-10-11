import numpy as np
import pandas as pd
import pytest
import shap


def _high_conf_rows(features, n=5, seed=1):
    """Fake high-confidence transactions: random feature values + fake transaction_ids."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(features))), columns=features)
    df.insert(0, 'transaction_id', [f'TXN-{i:08d}' for i in range(n)])
    return df


@pytest.fixture(scope='module')
def explainer(model):
    """Same TreeExplainer streaming_job.py builds once at startup."""
    return shap.TreeExplainer(model)


def test_empty_high_conf_returns_empty_dict(streaming, model, scaler, explainer):
    empty = pd.DataFrame(columns=['transaction_id'] + streaming.ALL_FEATURES)
    result = streaming.explain_high_confidence(
        empty, model, scaler, explainer,
        streaming.ALL_FEATURES, streaming.SCALED_FEATURES
    )
    assert result == {}


def test_explain_returns_entry_per_transaction(streaming, model, scaler, explainer):
    df = _high_conf_rows(streaming.ALL_FEATURES, n=5)
    result = streaming.explain_high_confidence(
        df, model, scaler, explainer,
        streaming.ALL_FEATURES, streaming.SCALED_FEATURES, top_k=3
    )
    assert set(result.keys()) == set(df['transaction_id'])
    for reasons in result.values():
        assert len(reasons) == 3
        assert all(isinstance(v, float) for v in reasons.values())
        assert all(k in streaming.ALL_FEATURES for k in reasons.keys())


def test_explain_respects_top_k(streaming, model, scaler, explainer):
    df = _high_conf_rows(streaming.ALL_FEATURES, n=2)
    result = streaming.explain_high_confidence(
        df, model, scaler, explainer,
        streaming.ALL_FEATURES, streaming.SCALED_FEATURES, top_k=1
    )
    for reasons in result.values():
        assert len(reasons) == 1


def test_explain_sorted_by_absolute_value(streaming, model, scaler, explainer):
    """Top reasons must be ranked by impact magnitude, not raw signed value."""
    df = _high_conf_rows(streaming.ALL_FEATURES, n=3)
    result = streaming.explain_high_confidence(
        df, model, scaler, explainer,
        streaming.ALL_FEATURES, streaming.SCALED_FEATURES, top_k=5
    )
    for reasons in result.values():
        abs_values = [abs(v) for v in reasons.values()]
        assert abs_values == sorted(abs_values, reverse=True)


def test_explain_does_not_mutate_input(streaming, model, scaler, explainer):
    df = _high_conf_rows(streaming.ALL_FEATURES, n=3)
    before = df.copy()
    streaming.explain_high_confidence(
        df, model, scaler, explainer,
        streaming.ALL_FEATURES, streaming.SCALED_FEATURES
    )
    pd.testing.assert_frame_equal(df, before)