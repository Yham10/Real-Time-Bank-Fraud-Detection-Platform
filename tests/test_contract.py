V = [f'V{i}' for i in range(1, 29)]


def test_metadata_feature_order(metadata):
    assert metadata['features'] == ['Time'] + V + ['Amount']       # bug #2
    assert metadata['n_features'] == 30
    assert metadata['scaled_features'] == ['Amount', 'Time']


def test_scaler_fitted_on_both_columns(scaler, metadata):           # bug #1
    assert list(scaler.feature_names_in_) == metadata['scaled_features']
    assert scaler.n_features_in_ == 2


def test_booster_matches_metadata(model, metadata):                  # bug #2
    assert list(model.get_booster().feature_names) == metadata['features']


def test_streaming_reads_same_contract(streaming, metadata):
    assert streaming.ALL_FEATURES == metadata['features']
    assert streaming.SCALED_FEATURES == metadata['scaled_features']


def test_model_manager_boots(streaming):
    mm = streaming.ModelManager(streaming.MODEL_PATH, streaming.SCALER_PATH).load()
    assert mm._model is not None and mm._scaler is not None


import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pytest

def test_model_manager_rejects_broken_scaler(monkeypatch, streaming, artifacts, tmp_path):
    # force local load, not MLflow - you have MLFLOW_TRACKING_URI set in PowerShell
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    monkeypatch.setattr(streaming, "MLFLOW_TRACKING_URI", None, raising=False)

    bad = StandardScaler().fit(pd.DataFrame({'Time': [0.0, 1.0, 2.0]}))
    p = tmp_path / 'bad_scaler.pkl'
    joblib.dump(bad, p)
    
    with pytest.raises(RuntimeError, match='Scaler fitted on'):
        streaming.ModelManager(
            model_path=str(artifacts / "fraud_model.pkl"),
            scaler_path=str(p),
            metadata_path=str(artifacts / "metadata.json")
        ).load()