import os
import json
import time
import random
import logging
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from dotenv import load_dotenv
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge
)

# ── LOGGING SETUP ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── LOAD CONFIG ───────────────────────────────────────────────
load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    'KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'
)
KAFKA_TOPIC              = os.getenv(
    'KAFKA_TOPIC_TRANSACTIONS', 'raw_transactions'
)
TRANSACTIONS_PER_SECOND  = int(os.getenv(
    'TRANSACTIONS_PER_SECOND', 100
))
DATASET_PATH             = os.getenv(
    'DATASET_PATH', '../data/creditcard.csv'
)
PROMETHEUS_PORT          = int(os.getenv(
    'PROMETHEUS_PORT', 8000
))

# ── PROMETHEUS METRICS ────────────────────────────────────────
transactions_sent = Counter(
    'producer_transactions_sent_total',
    'Total number of transactions sent to Kafka'
)

fraud_sent = Counter(
    'producer_fraud_sent_total',
    'Total number of fraud transactions sent'
)

send_duration = Histogram(
    'producer_send_duration_seconds',
    'Time taken to send one message to Kafka',
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5]
)

kafka_errors = Counter(
    'producer_kafka_errors_total',
    'Total number of Kafka send errors'
)

current_rate = Gauge(
    'producer_current_rate_tps',
    'Current transactions per second being sent'
)


# ── KAFKA CONNECTION ──────────────────────────────────────────
def create_producer(retries: int = 10,
                    wait: int = 5) -> KafkaProducer:
    """
    Create Kafka producer with retry logic
    Kafka might not be ready immediately when container starts
    """
    for attempt in range(1, retries + 1):
        try:
            log.info(
                f"Connecting to Kafka at "
                f"{KAFKA_BOOTSTRAP_SERVERS} "
                f"(attempt {attempt}/{retries})"
            )
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,

                # Serialize Python dict → JSON bytes
                value_serializer=lambda v: (
                    json.dumps(v).encode('utf-8')
                ),

                # Add key for partitioning
                key_serializer=lambda k: (
                    str(k).encode('utf-8')
                ),

                # Reliability settings
                acks='all',            # wait for all replicas
                retries=3,             # retry on failure
                retry_backoff_ms=500,

                # Performance settings
                batch_size=16384,      # batch up to 16KB
                linger_ms=10,          # wait 10ms to fill batch
                compression_type='gzip',  # compress messages

                # Timeouts
                request_timeout_ms=30000,
                max_block_ms=60000,
            )
            log.info("✅ Connected to Kafka successfully")
            return producer

        except NoBrokersAvailable:
            log.warning(
                f"   Kafka not ready yet. "
                f"Waiting {wait} seconds..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not connect to Kafka after {retries} attempts"
    )


# ── DATA LOADER ───────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """Load and prepare the dataset"""
    log.info(f"📂 Loading dataset from {path}")
    df = pd.read_csv(path)

    total = len(df)
    fraud = df['Class'].sum()
    legit = total - fraud

    log.info(f"   Total transactions : {total:,}")
    log.info(f"   Legitimate         : {legit:,} "
             f"({legit/total*100:.2f}%)")
    log.info(f"   Fraud              : {fraud:,} "
             f"({fraud/total*100:.3f}%)")

    return df


# ── MESSAGE BUILDER ───────────────────────────────────────────
def build_message(row: pd.Series,
                  transaction_id: int) -> dict:
    """
    Convert a DataFrame row into a Kafka message
    Add metadata useful for the streaming job
    """
    message = {
        # Unique transaction identifier
        'transaction_id': f'TXN-{transaction_id:08d}',

        # Timestamp when transaction "happened"
        'timestamp': time.time(),
        'timestamp_iso': pd.Timestamp.now().isoformat(),

        # Original features from dataset
        'Time'  : float(row['Time']),
        'Amount': float(row['Amount']),

        # V1 to V28 (PCA features from bank)
        **{
            f'V{i}': float(row[f'V{i}'])
            for i in range(1, 29)
        },

        # Ground truth label (for validation only)
        # In real life this would NOT be in the message
        'is_fraud_ground_truth': int(row['Class']),

        # Simulated metadata
        'merchant_id'   : f'MERCHANT-{random.randint(1, 1000):04d}',
        'card_last_four': f'{random.randint(1000, 9999)}',
        'country'       : random.choice([
            'US', 'UK', 'FR', 'DE', 'ES',
            'IT', 'BR', 'AU', 'CA', 'JP'
        ]),
    }
    return message


# ── DELIVERY CALLBACKS ────────────────────────────────────────
def on_send_success(record_metadata):
    """Called when message is successfully sent"""
    pass  # metrics already tracked in main loop


def on_send_error(exception):
    """Called when message fails to send"""
    kafka_errors.inc()
    log.error(f"❌ Failed to send message: {exception}")


# ── MAIN PRODUCER LOOP ────────────────────────────────────────
def run_producer():
    """Main loop: read CSV rows → send to Kafka"""

    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    log.info(f"📊 Prometheus metrics on port {PROMETHEUS_PORT}")

    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Connect to Kafka
    producer = create_producer()

    # Calculate sleep time between messages
    sleep_time = 1.0 / TRANSACTIONS_PER_SECOND

    log.info(f"\n🚀 Starting producer")
    log.info(f"   Topic      : {KAFKA_TOPIC}")
    log.info(f"   Speed      : {TRANSACTIONS_PER_SECOND} txn/sec")
    log.info(f"   Sleep time : {sleep_time*1000:.1f}ms between msgs")
    log.info(f"   Dataset    : {len(df):,} transactions")
    log.info(f"   Loop mode  : ON (will restart after last row)\n")

    transaction_id  = 0
    loop_count      = 0
    start_time      = time.time()
    last_log_time   = start_time

    # Loop forever (restart from beginning when CSV ends)
    while True:
        loop_count += 1
        log.info(f"🔄 Starting loop {loop_count} over dataset")

        for idx, row in df.iterrows():
            transaction_id += 1

            # Build the message
            message = build_message(row, transaction_id)

            # Use transaction_id as key
            # → same ID always goes to same partition
            key = transaction_id

            # Send to Kafka (async)
            send_start = time.time()
            try:
                producer.send(
                    topic=KAFKA_TOPIC,
                    key=key,
                    value=message
                ).add_callback(
                    on_send_success
                ).add_errback(
                    on_send_error
                )

                # Track metrics
                send_elapsed = time.time() - send_start
                send_duration.observe(send_elapsed)
                transactions_sent.inc()

                if message['is_fraud_ground_truth'] == 1:
                    fraud_sent.inc()

            except Exception as e:
                kafka_errors.inc()
                log.error(f"Error sending message: {e}")

            # Log progress every 10 seconds
            now = time.time()
            if now - last_log_time >= 10:
                elapsed    = now - start_time
                actual_tps = transaction_id / elapsed

                current_rate.set(actual_tps)

                log.info(
                    f"📈 Sent: {transaction_id:,} txns | "
                    f"Speed: {actual_tps:.1f} tps | "
                    f"Elapsed: {elapsed:.0f}s | "
                    f"Loop: {loop_count}"
                )
                last_log_time = now

            # Control the send rate
            time.sleep(sleep_time)

        log.info(
            f"✅ Finished loop {loop_count}. "
            f"Total sent: {transaction_id:,}"
        )

        # Flush before restarting
        producer.flush()


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == '__main__':
    try:
        run_producer()
    except KeyboardInterrupt:
        log.info("\n⛔ Producer stopped by user")
    except Exception as e:
        log.error(f"💥 Fatal error: {e}")
        raise