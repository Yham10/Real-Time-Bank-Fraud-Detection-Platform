import pandas as pd


def test_message_covers_spark_schema(producer_mod, streaming):
    row = pd.Series({'Time': 1.0, 'Amount': 12.5, 'Class': 0,
                     **{f'V{i}': 0.1 * i for i in range(1, 29)}})
    msg = producer_mod.build_message(row, 42)
    schema_fields = {f.name for f in streaming.get_transaction_schema().fields}
    missing = schema_fields - msg.keys()
    assert not missing, f"producer message missing fields Spark expects: {missing}"
    assert msg['transaction_id'] == 'TXN-00000042'
    assert all(isinstance(msg[f'V{i}'], float) for i in range(1, 29))
    assert msg['is_fraud_ground_truth'] in (0, 1)