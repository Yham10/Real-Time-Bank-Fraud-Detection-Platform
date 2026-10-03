import os, json, time, logging, threading
from collections import deque

import numpy as np
import pandas as pd
import psycopg2
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Gauge

from drift_math import (make_bins, bin_proportions, psi,
                        ks_statistic, SCORE_EDGES)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("drift-monitor")

# ── CONFIG ────────────────────────────────────────────────────
BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
TOPIC         = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "raw_transactions")
REFERENCE     = os.getenv("REFERENCE_PATH", "/model/reference_sample.parquet")
METADATA      = os.getenv("METADATA_PATH", "/model/metadata.json")
WINDOW_SIZE   = int(os.getenv("DRIFT_WINDOW_SIZE", "3000"))
MIN_SAMPLES   = int(os.getenv("DRIFT_MIN_SAMPLES", "1000"))
INTERVAL      = int(os.getenv("DRIFT_CHECK_INTERVAL", "30"))
METRICS_PORT  = int(os.getenv("METRICS_PORT", "8002"))
PSI_ALERT     = 0.25

PG = dict(host=os.getenv("POSTGRES_HOST", "postgres"),
          port=os.getenv("POSTGRES_PORT", "5432"),
          dbname=os.getenv("POSTGRES_DB", "fraud_detection"),
          user=os.getenv("POSTGRES_USER", "fraud_user"),
          password=os.getenv("POSTGRES_PASSWORD", "fraud_pass"),
          connect_timeout=5)

# ── METRICS ───────────────────────────────────────────────────
g_psi      = Gauge("drift_psi", "PSI per feature vs training reference", ["feature"])
g_ks       = Gauge("drift_ks_statistic", "KS statistic per feature", ["feature"])
g_max_psi  = Gauge("drift_max_psi", "Max PSI across monitored features")
g_n_drift  = Gauge("drift_features_drifted", f"Features with PSI > {PSI_ALERT}")
g_score    = Gauge("drift_score_psi", "PSI of fraud_probability distribution")
g_window   = Gauge("drift_window_size", "Transactions currently in the window")
g_last_run = Gauge("drift_last_run_timestamp_seconds", "Last successful check")

# ── REFERENCE ─────────────────────────────────────────────────
with open(METADATA) as f:
    FEATURES = [c for c in json.load(f)["features"] if c != "Time"]

ref_df = pd.read_parquet(REFERENCE)
missing = [c for c in FEATURES + ["fraud_probability"] if c not in ref_df.columns]
if missing:
    raise RuntimeError(f"Reference sample missing columns: {missing}. Re-run train.py")

REF_EDGES = {f: make_bins(ref_df[f]) for f in FEATURES}
import os
REF_MTIME = os.path.getmtime(REFERENCE)

def reload_reference_if_changed():
    global ref_df, REF_EDGES, REF_PROPS, REF_SCORE_PROPS, REF_MTIME
    try:
        mtime = os.path.getmtime(REFERENCE)
        if mtime != REF_MTIME:
            log.info(f"Reference file changed, reloading {REFERENCE}")
            new_ref = pd.read_parquet(REFERENCE)
            new_edges = {f: make_bins(new_ref[f]) for f in FEATURES}
            new_props = {f: bin_proportions(new_ref[f], new_edges[f]) for f in FEATURES}
            new_score_props = bin_proportions(new_ref["fraud_probability"], SCORE_EDGES) if "fraud_probability" in new_ref.columns else REF_SCORE_PROPS
            ref_df, REF_EDGES, REF_PROPS, REF_SCORE_PROPS = new_ref, new_edges, new_props, new_score_props
            REF_MTIME = mtime
            log.info(f"Reference reloaded: {len(ref_df)} rows")
    except Exception as e:
        log.warning(f"Failed to reload reference: {e}")
REF_PROPS = {f: bin_proportions(ref_df[f], REF_EDGES[f]) for f in FEATURES}
REF_SCORE_PROPS = bin_proportions(ref_df["fraud_probability"], SCORE_EDGES)
log.info(f"Reference loaded: {len(ref_df)} rows, {len(FEATURES)} features")

# ── KAFKA WINDOW ──────────────────────────────────────────────
window = deque(maxlen=WINDOW_SIZE)
lock = threading.Lock()


def consume():
    while True:
        try:
            consumer = KafkaConsumer(
                TOPIC, bootstrap_servers=BOOTSTRAP,
                group_id="drift-monitor",
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            )
            log.info("Kafka consumer connected")
            for msg in consumer:
                row = {f: msg.value.get(f) for f in FEATURES}
                with lock:
                    window.append(row)
        except Exception:
            log.exception("Consumer error, retrying in 5s")
            time.sleep(5)


def fetch_recent_scores(n: int) -> np.ndarray:
    conn = psycopg2.connect(**PG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT fraud_probability FROM transactions "
                        "ORDER BY id DESC LIMIT %s", (n,))
            return np.array([r[0] for r in cur.fetchall()], dtype=float)
    finally:
        conn.close()


# ── CHECK ─────────────────────────────────────────────────────
def run_check():
    reload_reference_if_changed()
    with lock:
        rows = list(window)
    g_window.set(len(rows))
    if len(rows) < MIN_SAMPLES:
        log.info(f"Window {len(rows)}/{MIN_SAMPLES}, waiting for more data")
        return

    cur_df = pd.DataFrame(rows).astype(float)
    psis = {}
    for f in FEATURES:
        cur = cur_df[f].to_numpy()
        psis[f] = psi(REF_PROPS[f], bin_proportions(cur, REF_EDGES[f]))
        g_psi.labels(f).set(psis[f])
        g_ks.labels(f).set(ks_statistic(ref_df[f], cur))

    top = max(psis, key=psis.get)
    g_max_psi.set(psis[top])
    g_n_drift.set(sum(v > PSI_ALERT for v in psis.values()))

    try:
        scores = fetch_recent_scores(len(rows))
        if len(scores) >= MIN_SAMPLES:
            g_score.set(psi(REF_SCORE_PROPS, bin_proportions(scores, SCORE_EDGES)))
    except Exception as e:
        log.warning(f"Score drift skipped: {e}")

    g_last_run.set(time.time())
    log.info(f"n={len(rows)} | max PSI={psis[top]:.4f} ({top}) | "
             f"drifted={sum(v > PSI_ALERT for v in psis.values())}")


def main():
    start_http_server(METRICS_PORT)
    log.info(f"Metrics on :{METRICS_PORT}")
    threading.Thread(target=consume, daemon=True).start()
    while True:
        time.sleep(INTERVAL)
        try:
            run_check()
        except Exception:
            log.exception("Drift check failed")


if __name__ == "__main__":
    main()