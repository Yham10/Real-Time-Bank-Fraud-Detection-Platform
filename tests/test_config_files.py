import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_json_valid_with_uid():
    d = json.loads((ROOT / 'monitoring/grafana/provisioning/dashboards/fraud_dashboard.json').read_text())
    assert d['uid'] == 'fraud-detection'
    assert d['panels'], 'dashboard has no panels'


def test_committed_metadata_is_consistent():
    m = json.loads((ROOT / 'model/metadata.json').read_text())
    assert m['features'] == ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    assert m['scaled_features'] == ['Amount', 'Time']