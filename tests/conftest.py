import os, sys, json, subprocess, importlib
from pathlib import Path
import numpy as np, pandas as pd, joblib, pytest

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_creditcard(path: Path, n=4000, fraud_rate=0.02, seed=42):
    """Same schema/column order as Kaggle creditcard.csv, with a learnable signal."""
    rng = rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    df = pd.DataFrame({'Time': rng.uniform(0, 172800, n)})
    for i in range(1, 29):
        df[f'V{i}'] = rng.normal(0, 1, n)
    df['V14'] -= 3 * y                      # fraud signal
    df['V17'] -= 2 * y
    df['Amount'] = rng.lognormal(3, 1.2, n)
    df['Class'] = y
    df.to_csv(path, index=False)


@pytest.fixture(scope='session')
def artifacts(tmp_path_factory):
    """Run the REAL train.py on synthetic data → real pkl + metadata in a temp dir."""
    work = tmp_path_factory.mktemp('artifacts')
    csv = work / 'creditcard.csv'
    _synthetic_creditcard(csv)
    env = {**os.environ, 'DATA_PATH': str(csv), 'MODEL_OUTPUT_DIR': str(work)}
    subprocess.run([sys.executable, str(ROOT / 'model' / 'train.py')],
                   check=True, env=env, cwd=ROOT)
    return work


@pytest.fixture(scope='session')
def metadata(artifacts):
    return json.loads((artifacts / 'metadata.json').read_text())


@pytest.fixture(scope='session')
def model(artifacts):
    return joblib.load(artifacts / 'fraud_model.pkl')


@pytest.fixture(scope='session')
def scaler(artifacts):
    return joblib.load(artifacts / 'scaler.pkl')


@pytest.fixture(scope='session')
def streaming(artifacts):
    """Import spark/streaming_job.py with env pointing at the temp artifacts."""
    os.environ['METADATA_PATH'] = str(artifacts / 'metadata.json')
    os.environ['MODEL_PATH']    = str(artifacts / 'fraud_model.pkl')
    os.environ['SCALER_PATH']   = str(artifacts / 'scaler.pkl')
    sys.path.insert(0, str(ROOT / 'spark'))
    return importlib.import_module('streaming_job')


@pytest.fixture(scope='session')
def producer_mod():
    sys.path.insert(0, str(ROOT / 'producer'))
    return importlib.import_module('producer')