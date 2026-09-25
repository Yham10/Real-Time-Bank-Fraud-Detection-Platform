import os
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
    IntegerType, LongType,
    TimestampType
)
from pyspark.sql.functions import pandas_udf, col

from kafka import KafkaProducer
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge
)
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────
load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    'KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'
)
KAFKA_TOPIC_INPUT  = os.getenv(
    'KAFKA_TOPIC_TRANSACTIONS', 'raw_transactions'
)
KAFKA_TOPIC_ALERTS = os.getenv(
    'KAFKA_TOPIC_ALERTS', 'fraud_alerts'
)
MODEL_PATH         = os.getenv(
    'MODEL_PATH', '/model/fraud_model.pkl'
)
SCALER_PATH        = os.getenv(
    'SCALER_PATH', '/model/scaler.pkl'
)
FRAUD_THRESHOLD    = float(os.getenv(
    'FRAUD_THRESHOLD', '0.5'
))
BATCH_INTERVAL     = int(os.getenv(
    'SPARK_BATCH_INTERVAL', '5'
))
PROMETHEUS_PORT    = int(os.getenv(
    'PROMETHEUS_PORT', '8001'
))
POSTGRES_HOST      = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT      = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB        = os.getenv('POSTGRES_DB', 'fraud_detection')
POSTGRES_USER      = os.getenv('POSTGRES_USER', 'fraud_user')
POSTGRES_PASSWORD  = os.getenv('POSTGRES_PASSWORD', 'fraud_pass')

# ── FEATURE COLUMNS ───────────────────────────────────────────
V_FEATURES   = [f'V{i}' for i in range(1, 29)]
ALL_FEATURES = ['Time'] + V_FEATURES + ['Amount'] # 30 features

# ── PROMETHEUS METRICS ────────────────────────────────────────
transactions_processed = Counter(
    'spark_transactions_processed_total',
    'Total transactions processed by Spark'
)
fraud_detected = Counter(
    'spark_fraud_detected_total',
    'Total fraud transactions detected'
)
inference_duration = Histogram(
    'spark_inference_duration_seconds',
    'Time to run ML inference on a batch',
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0]
)
batch_size_metric = Histogram(
    'spark_batch_size_transactions',
    'Number of transactions per micro-batch',
    buckets=[1, 5, 10, 50, 100, 500, 1000, 5000]
)
batch_processing_time = Histogram(
    'spark_batch_processing_time_seconds',
    'Total time to process one micro-batch',
    buckets=[.1, .5, 1.0, 2.5, 5.0, 10.0, 30.0]
)
fraud_score_distribution = Histogram(
    'spark_fraud_score_distribution',
    'Distribution of fraud probability scores',
    buckets=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
)
current_fraud_rate = Gauge(
    'spark_current_fraud_rate',
    'Fraud rate in the last batch (0.0 to 1.0)'
)
db_write_errors = Counter(
    'spark_db_write_errors_total',
    'Total database write errors'
)
kafka_alert_errors = Counter(
    'spark_kafka_alert_errors_total',
    'Total Kafka alert send errors'
)


# ════════════════════════════════════════════════════════════════
# MODEL MANAGER
# ════════════════════════════════════════════════════════════════
class ModelManager:
    """
    Loads and manages the ML model.
    Used on the driver to load model once,
    then contents are broadcast to workers.
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self._model      = None
        self._scaler     = None

    def load(self):
        """Load model and scaler from disk"""
        log.info(f"🤖 Loading model from {self.model_path}")
        self._model  = joblib.load(self.model_path)
        self._scaler = joblib.load(self.scaler_path)
        log.info("✅ Model loaded successfully")
        return self


# ════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ════════════════════════════════════════════════════════════════
class DatabaseManager:
    """Handles PostgreSQL connections and writes"""

    def __init__(self):
        self.conn   = None
        self.cursor = None

    def connect(self, retries: int = 10, wait: int = 5):
        """Connect to PostgreSQL with retry logic"""
        for attempt in range(1, retries + 1):
            try:
                log.info(
                    f"🗄️  Connecting to PostgreSQL "
                    f"(attempt {attempt}/{retries})"
                )
                self.conn = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    dbname=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    connect_timeout=10
                )
                self.cursor = self.conn.cursor()
                log.info("✅ Connected to PostgreSQL")
                return self
            except Exception as e:
                log.warning(
                    f"   DB not ready: {e}. "
                    f"Waiting {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError("Could not connect to PostgreSQL")

    def create_tables(self):
        """Create tables if they don't exist"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id                   SERIAL PRIMARY KEY,
                transaction_id       VARCHAR(20) UNIQUE,
                processed_at         TIMESTAMP DEFAULT NOW(),
                amount               DOUBLE PRECISION,
                fraud_probability    DOUBLE PRECISION,
                is_fraud_predicted   BOOLEAN,
                is_fraud_ground_truth INTEGER,
                merchant_id          VARCHAR(20),
                card_last_four       VARCHAR(4),
                country              VARCHAR(2),
                processing_time_ms   DOUBLE PRECISION
            );

            CREATE TABLE IF NOT EXISTS fraud_alerts (
                id                SERIAL PRIMARY KEY,
                transaction_id    VARCHAR(20),
                alerted_at        TIMESTAMP DEFAULT NOW(),
                fraud_probability DOUBLE PRECISION,
                amount            DOUBLE PRECISION,
                merchant_id       VARCHAR(20),
                country           VARCHAR(2)
            );

            CREATE TABLE IF NOT EXISTS batch_metrics (
                id              SERIAL PRIMARY KEY,
                batch_id        BIGINT,
                processed_at    TIMESTAMP DEFAULT NOW(),
                batch_size      INTEGER,
                fraud_count     INTEGER,
                fraud_rate      DOUBLE PRECISION,
                processing_ms   DOUBLE PRECISION
            );
        """)
        self.conn.commit()
        log.info("✅ Database tables ready")

    def insert_transactions(self, records: list):
        """Batch insert transactions"""
        if not records:
            return
        try:
            execute_batch(
                self.cursor,
                """
                INSERT INTO transactions (
                    transaction_id,
                    amount,
                    fraud_probability,
                    is_fraud_predicted,
                    is_fraud_ground_truth,
                    merchant_id,
                    card_last_four,
                    country,
                    processing_time_ms
                ) VALUES (
                    %(transaction_id)s,
                    %(amount)s,
                    %(fraud_probability)s,
                    %(is_fraud_predicted)s,
                    %(is_fraud_ground_truth)s,
                    %(merchant_id)s,
                    %(card_last_four)s,
                    %(country)s,
                    %(processing_time_ms)s
                )
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                records,
                page_size=500
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            db_write_errors.inc()
            log.error(f"❌ DB insert error: {e}")

    def insert_batch_metric(self, batch_id: int,
                            batch_size: int,
                            fraud_count: int,
                            processing_ms: float):
        """Record batch-level metrics"""
        try:
            fraud_rate = (
                fraud_count / batch_size
                if batch_size > 0 else 0
            )
            self.cursor.execute(
                """
                INSERT INTO batch_metrics
                (batch_id, batch_size, fraud_count,
                 fraud_rate, processing_ms)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (batch_id, batch_size, fraud_count,
                 fraud_rate, processing_ms)
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            log.error(f"❌ Batch metric insert error: {e}")


# ════════════════════════════════════════════════════════════════
# ALERT PRODUCER
# ════════════════════════════════════════════════════════════════
class AlertProducer:
    """Sends fraud alerts back to Kafka"""

    def __init__(self):
        self.producer = None

    def connect(self, retries: int = 10, wait: int = 5):
        for attempt in range(1, retries + 1):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: (
                        json.dumps(v).encode('utf-8')
                    )
                )
                log.info("✅ Alert producer connected to Kafka")
                return self
            except Exception as e:
                log.warning(
                    f"Alert producer not ready: {e}. "
                    f"Waiting {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError(
            "Could not connect alert producer to Kafka"
        )

    def send_alert(self, transaction: dict):
        """Send fraud alert to Kafka topic"""
        try:
            alert = {
                'alert_id'         : f"ALERT-{time.time_ns()}",
                'transaction_id'   : transaction['transaction_id'],
                'timestamp'        : time.time(),
                'fraud_probability': transaction['fraud_probability'],
                'amount'           : transaction['amount'],
                'merchant_id'      : transaction['merchant_id'],
                'country'          : transaction['country'],
                'card_last_four'   : transaction['card_last_four'],
                'severity'         : self._get_severity(
                    transaction['fraud_probability']
                )
            }
            self.producer.send(KAFKA_TOPIC_ALERTS, value=alert)
        except Exception as e:
            kafka_alert_errors.inc()
            log.error(f"❌ Alert send error: {e}")

    @staticmethod
    def _get_severity(probability: float) -> str:
        if probability >= 0.9:
            return 'CRITICAL'
        elif probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'


# ════════════════════════════════════════════════════════════════
# KAFKA MESSAGE SCHEMA
# ════════════════════════════════════════════════════════════════
def get_transaction_schema() -> StructType:
    """
    Define the schema for messages coming from Kafka.
    Spark uses this to parse JSON efficiently.
    """
    v_fields = [
        StructField(f'V{i}', DoubleType(), True)
        for i in range(1, 29)
    ]

    return StructType([
        StructField('transaction_id',
                    StringType(), False),
        StructField('timestamp',
                    DoubleType(), True),
        StructField('timestamp_iso',
                    StringType(), True),
        StructField('Time',
                    DoubleType(), True),
        StructField('Amount',
                    DoubleType(), True),
        *v_fields,
        StructField('is_fraud_ground_truth',
                    IntegerType(), True),
        StructField('merchant_id',
                    StringType(), True),
        StructField('card_last_four',
                    StringType(), True),
        StructField('country',
                    StringType(), True),
    ])


# ════════════════════════════════════════════════════════════════
# SPARK SESSION
# ════════════════════════════════════════════════════════════════
def create_spark_session() -> SparkSession:
    """Create and configure Spark session"""

    jars = ','.join([
        '/app/spark-sql-kafka.jar',
        '/app/kafka-clients.jar',
        '/app/spark-token-provider-kafka.jar',
        '/app/commons-pool2.jar',
    ])

    spark = (
        SparkSession.builder
        .appName('FraudDetectionStreaming')
        .config('spark.jars', jars)
        .config('spark.streaming.stopGracefullyOnShutdown', 'true')
        .config('spark.sql.streaming.checkpointLocation',
                '/tmp/spark-checkpoints')
        .config('spark.sql.shuffle.partitions', '4')
        .config('spark.default.parallelism', '4')
        .config('spark.driver.memory', '2g')
        .config('spark.executor.memory', '2g')
        .config('spark.ui.enabled', 'true')
        .config('spark.ui.port', '4040')
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel('WARN')

    log.info("✅ Spark session created")
    log.info(f"   Spark version : {spark.version}")
    log.info(f"   Spark UI      : http://localhost:4040")

    return spark


# ════════════════════════════════════════════════════════════════
# PANDAS UDF FACTORY
# ════════════════════════════════════════════════════════════════
def make_fraud_udf(model_broadcast):
    """
    Returns a Pandas UDF that runs XGBoost inference.
    Wraps the broadcast so workers access their local
    cached copy instead of reloading from disk each call.
    """

    @pandas_udf(DoubleType())
    def predict_fraud_proba(
        time   : pd.Series,
        amount : pd.Series,
        v1     : pd.Series, v2 : pd.Series, v3 : pd.Series,
        v4     : pd.Series, v5 : pd.Series, v6 : pd.Series,
        v7     : pd.Series, v8 : pd.Series, v9 : pd.Series,
        v10    : pd.Series, v11: pd.Series, v12: pd.Series,
        v13    : pd.Series, v14: pd.Series, v15: pd.Series,
        v16    : pd.Series, v17: pd.Series, v18: pd.Series,
        v19    : pd.Series, v20: pd.Series, v21: pd.Series,
        v22    : pd.Series, v23: pd.Series, v24: pd.Series,
        v25    : pd.Series, v26: pd.Series, v27: pd.Series,
        v28    : pd.Series
    ) -> pd.Series:

        # Each worker reads from its local broadcast cache
        model  = model_broadcast.value['model']
        scaler = model_broadcast.value['scaler']

        # Rebuild feature DataFrame on the worker
        features = pd.DataFrame({
            'Time'  : time,   'Amount': amount,
            'V1'    : v1,     'V2'    : v2,
            'V3'    : v3,     'V4'    : v4,
            'V5'    : v5,     'V6'    : v6,
            'V7'    : v7,     'V8'    : v8,
            'V9'    : v9,     'V10'   : v10,
            'V11'   : v11,    'V12'   : v12,
            'V13'   : v13,    'V14'   : v14,
            'V15'   : v15,    'V16'   : v16,
            'V17'   : v17,    'V18'   : v18,
            'V19'   : v19,    'V20'   : v20,
            'V21'   : v21,    'V22'   : v22,
            'V23'   : v23,    'V24'   : v24,
            'V25'   : v25,    'V26'   : v26,
            'V27'   : v27,    'V28'   : v28,
        })

        # Fix: scale both columns together with one transform
        features[['Amount', 'Time']] = scaler.transform(
            features[['Amount', 'Time']]
        )

        probabilities = model.predict_proba(
            features[ALL_FEATURES]
        )[:, 1]

        return pd.Series(probabilities)

    return predict_fraud_proba


# ════════════════════════════════════════════════════════════════
# STREAMING SETUP
# ════════════════════════════════════════════════════════════════
def run_streaming_with_udf(spark, model_broadcast,
                            db_manager, alert_producer):
    """
    Builds the streaming pipeline.
    UDF handles inference distributed across workers.
    foreachBatch only handles I/O (DB + Kafka alerts).
    """

    schema    = get_transaction_schema()
    fraud_udf = make_fraud_udf(model_broadcast)

    # ── READ FROM KAFKA ───────────────────────────────────────
    raw_stream = (
        spark.readStream
        .format('kafka')
        .option('kafka.bootstrap.servers', KAFKA_BOOTSTRAP_SERVERS)
        .option('subscribe', KAFKA_TOPIC_INPUT)
        .option('startingOffsets', 'latest')
        .option('maxOffsetsPerTrigger', 10000)
        .option('failOnDataLoss', 'false')
        .load()
    )

    # ── PARSE JSON ────────────────────────────────────────────
    parsed_stream = (
        raw_stream
        .select(
            F.from_json(
                F.col('value').cast('string'),
                schema
            ).alias('data'),
            F.col('timestamp').alias('kafka_timestamp'),
            F.col('partition'),
            F.col('offset')
        )
        .select('data.*', 'kafka_timestamp', 'partition', 'offset')
        .filter(F.col('transaction_id').isNotNull())
    )

    # ── APPLY UDF (distributed inference) ────────────────────
    # Scoring happens here on workers, NOT in foreachBatch
    scored_stream = (
        parsed_stream
        .withColumn(
            'fraud_probability',
            fraud_udf(
                col('Time'),  col('Amount'),
                col('V1'),    col('V2'),    col('V3'),
                col('V4'),    col('V5'),    col('V6'),
                col('V7'),    col('V8'),    col('V9'),
                col('V10'),   col('V11'),   col('V12'),
                col('V13'),   col('V14'),   col('V15'),
                col('V16'),   col('V17'),   col('V18'),
                col('V19'),   col('V20'),   col('V21'),
                col('V22'),   col('V23'),   col('V24'),
                col('V25'),   col('V26'),   col('V27'),
                col('V28')
            )
        )
        .withColumn(
            'is_fraud_predicted',
            F.col('fraud_probability') >= FRAUD_THRESHOLD
        )
    )

    log.info("✅ Stream + UDF pipeline defined")
    log.info(f"   Batch interval  : {BATCH_INTERVAL} seconds")
    log.info(f"   Fraud threshold : {FRAUD_THRESHOLD}")

    # ── START STREAMING QUERY ─────────────────────────────────
    query = (
        scored_stream.writeStream
        .foreachBatch(
            lambda df, batch_id: process_batch_udf(
                df, batch_id, db_manager, alert_producer
            )
        )
        .trigger(processingTime=f'{BATCH_INTERVAL} seconds')
        .option(
            'checkpointLocation',
            '/tmp/spark-checkpoints/fraud-streaming'
        )
        .start()
    )

    return query


# ════════════════════════════════════════════════════════════════
# BATCH PROCESSOR  (I/O only — no inference here)
# ════════════════════════════════════════════════════════════════
def process_batch_udf(batch_df, batch_id: int,
                      db_manager: DatabaseManager,
                      alert_producer: AlertProducer):
    """
    Called by Spark for each micro-batch.
    Inference already done by UDF on workers.
    This function only handles:
      - Prometheus metrics update
      - Fraud alerts → Kafka
      - All transactions → PostgreSQL
    """
    batch_start = time.time()

    if batch_df.isEmpty():
        log.debug(f"Batch {batch_id}: empty, skipping")
        return

    # Small collect to driver — only for I/O, not for inference
    pdf        = batch_df.toPandas()
    batch_size = len(pdf)

    fraud_count = int(pdf['is_fraud_predicted'].sum())
    fraud_rate  = fraud_count / batch_size if batch_size > 0 else 0

    log.info(
        f"\n{'='*50}\n"
        f"📦 Batch {batch_id} | {batch_size} transactions | "
        f"{fraud_count} fraud ({fraud_rate*100:.2f}%)"
    )

    # ── PROMETHEUS ────────────────────────────────────────────
    transactions_processed.inc(batch_size)
    fraud_detected.inc(fraud_count)
    batch_size_metric.observe(batch_size)
    current_fraud_rate.set(fraud_rate)

    for score in pdf['fraud_probability']:
        fraud_score_distribution.observe(float(score))

    # ── FRAUD ALERTS → KAFKA ──────────────────────────────────
    fraud_rows = pdf[pdf['is_fraud_predicted'] == True]
    for _, row in fraud_rows.iterrows():
        alert_producer.send_alert({
            'transaction_id'   : row['transaction_id'],
            'fraud_probability': float(row['fraud_probability']),
            'amount'           : float(row['Amount']),
            'merchant_id'      : row['merchant_id'],
            'country'          : row['country'],
            'card_last_four'   : row['card_last_four'],
        })

    if fraud_count > 0:
        log.info(f"   🚨 Sent {fraud_count} alerts to Kafka")

    # ── LOG HIGH CONFIDENCE FRAUD ─────────────────────────────
    high_confidence = pdf[pdf['fraud_probability'] >= 0.9]
    if len(high_confidence) > 0:
        log.info(f"   ⚠️  HIGH CONFIDENCE fraud:")
        for _, row in high_confidence.iterrows():
            log.info(
                f"      {row['transaction_id']} | "
                f"Amount: ${row['Amount']:.2f} | "
                f"Score: {row['fraud_probability']:.4f} | "
                f"Country: {row['country']}"
            )

    # ── ALL TRANSACTIONS → POSTGRESQL ─────────────────────────
    batch_elapsed_ms = (time.time() - batch_start) * 1000

    records = [
        {
            'transaction_id'       : row['transaction_id'],
            'amount'               : float(row['Amount']),
            'fraud_probability'    : float(row['fraud_probability']),
            'is_fraud_predicted'   : bool(row['is_fraud_predicted']),
            'is_fraud_ground_truth': int(row['is_fraud_ground_truth']),
            'merchant_id'          : row['merchant_id'],
            'card_last_four'       : row['card_last_four'],
            'country'              : row['country'],
            'processing_time_ms'   : batch_elapsed_ms,
        }
        for _, row in pdf.iterrows()
    ]

    db_manager.insert_transactions(records)

    # ── BATCH METRICS → POSTGRESQL ────────────────────────────
    batch_processing_time.observe(batch_elapsed_ms / 1000)
    db_manager.insert_batch_metric(
        batch_id, batch_size, fraud_count, batch_elapsed_ms
    )

    log.info(
        f"   Total batch time : {batch_elapsed_ms:.1f}ms\n"
        f"{'='*50}"
    )


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("FRAUD DETECTION STREAMING JOB STARTING")
    log.info("=" * 60)

    # ── START PROMETHEUS ──────────────────────────────────────
    start_http_server(PROMETHEUS_PORT)
    log.info(f"📊 Prometheus metrics on port {PROMETHEUS_PORT}")

    # ── LOAD MODEL ON DRIVER ──────────────────────────────────
    model_manager = ModelManager(MODEL_PATH, SCALER_PATH).load()

    # ── CONNECT TO POSTGRES ───────────────────────────────────
    db_manager = DatabaseManager().connect()
    db_manager.create_tables()

    # ── CONNECT ALERT PRODUCER ────────────────────────────────
    alert_producer = AlertProducer().connect()

    # ── CREATE SPARK SESSION ──────────────────────────────────
    spark = create_spark_session()

    # ── BROADCAST MODEL TO ALL WORKERS ───────────────────────
    # Sent once from driver → cached on each worker in memory
    # UDF reads from local cache, not from disk
    model_broadcast = spark.sparkContext.broadcast({
        'model' : model_manager._model,
        'scaler': model_manager._scaler,
    })
    log.info("📡 Model broadcast to all workers")

    # ── BUILD + START STREAMING PIPELINE ─────────────────────
    query = run_streaming_with_udf(
        spark, model_broadcast, db_manager, alert_producer
    )

    log.info("🚀 Streaming query started")
    log.info(f"   Query ID : {query.id}")
    log.info("   Waiting for data from Kafka...\n")

    # ── WAIT FOR TERMINATION ──────────────────────────────────
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        log.info("\n⛔ Stopping streaming job...")
        query.stop()
        spark.stop()
        log.info("✅ Streaming job stopped cleanly")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == '__main__':
    main()