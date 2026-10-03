import os, time, json, requests
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowSkipException

PROM_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_DIR = os.getenv("MODEL_OUTPUT_DIR", "/model")
COOLDOWN_MIN = int(os.getenv("RETRAIN_COOLDOWN_MIN", "20"))

def check_drift(**ctx):
    conf = ctx["dag_run"].conf or {}
    force = conf.get("force", False)
    triggered_by = conf.get("triggered_by", "manual")
    print(f"Triggered by {triggered_by}, conf={conf}")
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": "drift_max_psi"}, timeout=10)
        r.raise_for_status()
        res = r.json()["data"]["result"]
        if not res:
            raise AirflowSkipException("No drift_max_psi metric yet")
        psi = float(res[0]["value"][1])
        print(f"Current drift_max_psi={psi}")
        if not force and psi < 0.25:
            raise AirflowSkipException(f"PSI {psi} < 0.25, skipping")
        # Save PSI in XCom, XCom allows tasks to exchange small pieces of information.
        ctx["ti"].xcom_push(key="psi", value=psi)
        return psi
    except AirflowSkipException:
        raise
    except Exception as e:
        if not force:
            raise AirflowSkipException(f"Prometheus query failed: {e}")
        return 0.0

def check_cooldown(**ctx):
    path = f"{MODEL_DIR}/last_retrain.txt"
    if not os.path.exists(path):
        return True
    try:
        last = float(open(path).read().strip())
        elapsed = (time.time() - last) / 60
        if elapsed < COOLDOWN_MIN and not (ctx["dag_run"].conf or {}).get("force"):
            raise AirflowSkipException(f"Last retrain {elapsed:.1f}m ago < {COOLDOWN_MIN}m cooldown")
    except AirflowSkipException:
        raise
    except Exception:
        pass
    return True

def write_retrain_marker(**ctx):
    open(f"{MODEL_DIR}/last_retrain.txt", "w").write(str(time.time()))
    return True

with DAG(
    dag_id="retrain_on_drift",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fraud", "drift", "mlops"],
) as dag:

    check = PythonOperator(task_id="check_drift", python_callable=check_drift)
    cooldown = PythonOperator(task_id="check_cooldown", python_callable=check_cooldown)
    
    train = BashOperator(
        task_id="train",
        bash_command="python /opt/airflow/model/train.py",
        env={
            "MLFLOW_TRACKING_URI": MLFLOW_URI,
            "MODEL_OUTPUT_DIR": MODEL_DIR,
            "DATA_PATH": os.getenv("DATA_PATH", "/data/creditcard.csv"),
            "MLFLOW_MODEL_NAME": "fraud-xgb",
            "MLFLOW_MODEL_ALIAS": "production",
        }
    )

    mark = PythonOperator(task_id="mark_retrain_time", python_callable=write_retrain_marker)

    # restart spark + drift-monitor so they pick up new model + new reference_sample.parquet
    deploy = BashOperator(
        task_id="deploy",
        bash_command="docker restart fraud-spark drift-monitor || true && echo 'restarted'"
    )

    check >> cooldown >> train >> mark >> deploy