import os
import json
import time
import random
import logging
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from dotenv import load_dotenv
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC_TRANSACTIONS', 'raw_transactions')
TRANSACTIONS_PER_SECOND = int(os.getenv('TRANSACTIONS_PER_SECOND', 100))
DATASET_PATH = os.getenv('DATASET_PATH', '../data/creditcard.csv')
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 8000))
SAMPLING_MODE = os.getenv('PRODUCER_SAMPLING_MODE', 'random').lower() # random | sequential
DRIFT_CONFIG_PATH = os.getenv('DRIFT_CONFIG_PATH', '/tmp/drift.json')

# ── PROMETHEUS ────────────────────────────────────────────────
transactions_sent = Counter('producer_transactions_sent_total', 'Total sent to Kafka')
fraud_sent = Counter('producer_fraud_sent_total', 'Total fraud sent')
send_duration = Histogram('producer_send_duration_seconds', 'Time to send one msg', buckets=[.001, .005, .01, .025, .05, .1, .25, .5])
kafka_errors = Counter('producer_kafka_errors_total', 'Total Kafka send errors')
current_rate = Gauge('producer_current_rate_tps', 'Current tps')
drift_active = Gauge('producer_drift_injection_active', '1 if drift injection enabled')
drift_shift_gauge = Gauge('producer_drift_shift', 'Current shift', ['feature'])
drift_scale_gauge = Gauge('producer_drift_scale', 'Current scale', ['feature'])
drift_fraud_mult = Gauge('producer_drift_fraud_multiplier', 'Current fraud rate multiplier')

# ── DRIFT CONTROLLER ──────────────────────────────────────────
class DriftController:
    def __init__(self, path: str):
        self.path = Path(path)
        self.config = {"enabled": False, "shifts": {}, "scales": {}, "fraud_multiplier": 1.0}
        self._last_mtime = 0
        self.refresh(force=True)

    def refresh(self, force=False):
        if not self.path.exists():
            if self.config["enabled"]:
                log.info("🔵 Drift injection DISABLED (config file removed)")
                self.config = {"enabled": False, "shifts": {}, "scales": {}, "fraud_multiplier": 1.0}
                drift_active.set(0)
                drift_fraud_mult.set(1.0)
            return
        try:
            mtime = self.path.stat().st_mtime
            if not force and mtime == self._last_mtime:
                return
            self._last_mtime = mtime
            with open(self.path) as f:
                data = json.load(f)
            self.config["enabled"] = bool(data.get("enabled", True))
            self.config["shifts"] = dict(data.get("shifts", {}))
            self.config["scales"] = dict(data.get("scales", {}))
            self.config["fraud_multiplier"] = float(data.get("fraud_multiplier", 1.0))
            
            drift_active.set(1 if self.config["enabled"] else 0)
            drift_fraud_mult.set(self.config["fraud_multiplier"])
            for feat, val in self.config["shifts"].items():
                drift_shift_gauge.labels(feat).set(float(val))
            for feat, val in self.config["scales"].items():
                drift_scale_gauge.labels(feat).set(float(val))
            
            log.info(f"🔴 Drift config loaded: {self.config}")
        except Exception as e:
            log.error(f"Failed to load drift config {self.path}: {e}")

    def is_enabled(self): return self.config["enabled"]
    def get_shifts(self): return self.config["shifts"]
    def get_scales(self): return self.config["scales"]
    def get_fraud_multiplier(self): return self.config["fraud_multiplier"]

def create_producer(retries=10, wait=5):
    for attempt in range(1, retries+1):
        try:
            log.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS} (attempt {attempt}/{retries})")
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                acks='all', retries=3, retry_backoff_ms=500,
                batch_size=16384, linger_ms=10, compression_type='gzip',
                request_timeout_ms=30000, max_block_ms=60000,
            )
            log.info("✅ Connected to Kafka")
            return producer
        except NoBrokersAvailable:
            log.warning(f"Kafka not ready, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Could not connect after {retries} attempts")

def load_dataset(path):
    log.info(f"📂 Loading dataset from {path}")
    df = pd.read_csv(path)
    log.info(f"   Total: {len(df):,} | Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    return df

def build_message(row, transaction_id, drift_ctrl: DriftController):
    # Apply drift on a copy
    shifts = drift_ctrl.get_shifts() if drift_ctrl.is_enabled() else {}
    scales = drift_ctrl.get_scales() if drift_ctrl.is_enabled() else {}

    amount = float(row['Amount'])
    if 'Amount' in scales: amount *= float(scales['Amount'])
    if 'Amount' in shifts: amount += float(shifts['Amount'])

    v_values = {}
    for i in range(1, 29):
        k = f'V{i}'
        v = float(row[k])
        if k in scales: v *= float(scales[k])
        if k in shifts: v += float(shifts[k])
        v_values[k] = v

    message = {
        'transaction_id': f'TXN-{transaction_id:08d}',
        'timestamp': time.time(),
        'timestamp_iso': pd.Timestamp.now().isoformat(),
        'Time': float(row['Time']),
        'Amount': amount,
        **v_values,
        'is_fraud_ground_truth': int(row['Class']),
        'merchant_id': f'MERCHANT-{random.randint(1, 1000):04d}',
        'card_last_four': f'{random.randint(1000, 9999)}',
        'country': random.choice(['US','UK','FR','DE','ES','IT','BR','AU','CA','JP']),
    }
    return message

def run_producer():
    start_http_server(PROMETHEUS_PORT)
    log.info(f"📊 Prometheus on :{PROMETHEUS_PORT} | Sampling: {SAMPLING_MODE} | Drift config: {DRIFT_CONFIG_PATH}")

    df = load_dataset(DATASET_PATH)
    legit_df = df[df['Class']==0]
    fraud_df = df[df['Class']==1]
    base_fraud_rate = len(fraud_df)/len(df)

    producer = create_producer()
    drift_ctrl = DriftController(DRIFT_CONFIG_PATH)

    sleep_time = 1.0 / TRANSACTIONS_PER_SECOND
    transaction_id = 0
    loop_count = 0
    start_time = time.time()
    last_log_time = start_time
    last_drift_check = 0
    seq_idx = 0

    log.info(f"🚀 Starting | Topic: {KAFKA_TOPIC} | Speed: {TRANSACTIONS_PER_SECOND} tps")

    while True:
        loop_count += 1
        if SAMPLING_MODE == 'sequential':
            log.info(f"🔄 Loop {loop_count} sequential")
        
        while True:
            if SAMPLING_MODE == 'sequential' and seq_idx >= len(df):
                seq_idx = 0
                break

            # refresh drift config every ~1s
            if time.time() - last_drift_check > 1.0:
                drift_ctrl.refresh()
                last_drift_check = time.time()

            # pick row
            if SAMPLING_MODE == 'random':
                mult = drift_ctrl.get_fraud_multiplier() if drift_ctrl.is_enabled() else 1.0
                desired = min(0.5, base_fraud_rate * mult)
                if random.random() < desired and len(fraud_df) > 0:
                    row = fraud_df.sample(n=1).iloc[0]
                else:
                    row = legit_df.sample(n=1).iloc[0]
            else:
                row = df.iloc[seq_idx]
                seq_idx += 1

            transaction_id += 1
            message = build_message(row, transaction_id, drift_ctrl)
            key = transaction_id

            send_start = time.time()
            try:
                producer.send(topic=KAFKA_TOPIC, key=key, value=message).add_errback(lambda e: (kafka_errors.inc(), log.error(f"Send error: {e}")))
                send_duration.observe(time.time() - send_start)
                transactions_sent.inc()
                if message['is_fraud_ground_truth'] == 1:
                    fraud_sent.inc()
            except Exception as e:
                kafka_errors.inc()
                log.error(f"Error sending: {e}")

            now = time.time()
            if now - last_log_time >= 10:
                actual_tps = transaction_id / (now - start_time)
                current_rate.set(actual_tps)
                log.info(f"📈 Sent: {transaction_id:,} | {actual_tps:.1f} tps | Loop: {loop_count} | Drift: {'ON' if drift_ctrl.is_enabled() else 'OFF'}")
                last_log_time = now

            time.sleep(sleep_time)

            if SAMPLING_MODE == 'sequential':
                continue
            # random mode is infinite, no inner break
        producer.flush()

if __name__ == '__main__':
    try:
        run_producer()
    except KeyboardInterrupt:
        log.info("⛔ Stopped by user")