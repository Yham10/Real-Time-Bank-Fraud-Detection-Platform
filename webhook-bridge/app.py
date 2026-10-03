from flask import Flask, request, jsonify
import requests, os, logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080/api/v1/dags/retrain_on_drift/dagRuns")
USER = os.getenv("AIRFLOW_USER", "airflow")
PASS = os.getenv("AIRFLOW_PASS", "airflow")

@app.route("/grafana-webhook", methods=["POST"])
def hook():
    data = request.get_json(force=True, silent=True) or {}
    app.logger.info(f"Grafana payload firing: {data.get('status')}")
    try:
        r = requests.post(AIRFLOW_URL, auth=(USER, PASS), json={"conf": {"triggered_by": "grafana", "payload": data}}, timeout=10)
        app.logger.info(f"Airflow {r.status_code}: {r.text}")
        return jsonify({"airflow": r.status_code}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health(): return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)  