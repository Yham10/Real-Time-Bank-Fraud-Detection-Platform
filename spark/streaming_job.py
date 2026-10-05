import os
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
    IntegerType
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
KAFKA_TOPIC_INPUT = os.getenv(
    'KAFKA_TOPIC_TRANSACTIONS', 'raw_transactions'
)
KAFKA_TOPIC_ALERTS = os.getenv(
    'KAFKA_TOPIC_ALERTS', 'fraud_alerts'
)

MODEL_PATH = os.getenv(
    'MODEL_PATH', '/model/fraud_model.pkl'
)
SCALER_PATH = os.getenv(
    'SCALER_PATH', '/model/scaler.pkl'
)

FRAUD_THRESHOLD = float(os.getenv(
    'FRAUD_THRESHOLD', '0.5'
))
BATCH_INTERVAL = int(os.getenv(
    'SPARK_BATCH_INTERVAL', '5'
))
PROMETHEUS_PORT = int(os.getenv(
    'PROMETHEUS_PORT', '8001'
))

POSTGRES_HOST = os.getenv(
    'POSTGRES_HOST', 'postgres'
)
POSTGRES_PORT = os.getenv(
    'POSTGRES_PORT', '5432'
)
POSTGRES_DB = os.getenv(
    'POSTGRES_DB', 'fraud_detection'
)
POSTGRES_USER = os.getenv(
    'POSTGRES_USER', 'fraud_user'
)
POSTGRES_PASSWORD = os.getenv(
    'POSTGRES_PASSWORD', 'fraud_pass'
)


# ── MLFLOW CONFIG ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_MODEL_NAME = os.getenv(
    'MLFLOW_MODEL_NAME', 'fraud-xgb'
)
MLFLOW_MODEL_ALIAS = os.getenv(
    'MLFLOW_MODEL_ALIAS', 'production'
)


# ── FEATURE CONTRACT ──────────────────────────────────────────
# Single source of truth = metadata.json written by model/train.py.
# Hardcoded list is only a fallback for local runs without artifacts.

METADATA_PATH = os.getenv(
    'METADATA_PATH', '/model/metadata.json'
)

V_FEATURES = [f'V{i}' for i in range(1, 29)]

_FALLBACK_FEATURES = (
    ['Time'] + V_FEATURES + ['Amount']
)

_FALLBACK_SCALED = [
    'Amount',
    'Time'
]


def load_feature_contract(path: str):
    """Read feature order + scaled columns from training metadata."""
    try:
        with open(path) as f:
            meta = json.load(f)

        features = list(meta['features'])
        scaled = list(
            meta.get(
                'scaled_features',
                _FALLBACK_SCALED
            )
        )

        log.info(
            f"📜 Feature contract loaded from {path} "
            f"({len(features)} features, scaled={scaled})"
        )

        return features, scaled

    except Exception as e:
        log.warning(
            f"⚠️ Could not read contract from {path} "
            f"({e}). Using fallback."
        )

        return (
            _FALLBACK_FEATURES,
            _FALLBACK_SCALED
        )


ALL_FEATURES, SCALED_FEATURES = load_feature_contract(
    METADATA_PATH
)


# ── PROMETHEUS METRICS ────────────────────────────────────────

transactions_processed = Counter(
    'spark_transactions_processed_total',
    'Total transactions processed by Spark'
)

fraud_detected = Counter(
    'spark_fraud_detected_total',
    'Total fraud transactions detected'
)

batch_processing_time = Histogram(
    'spark_batch_processing_time_seconds',
    'Total time to process one micro-batch',
    buckets=[
        .5, 1, 1.5, 2, 2.5, 3,
        3.5, 4, 4.5, 5, 7.5, 10
    ]
)

batch_size_metric = Histogram(
    'spark_batch_size_transactions',
    'Number of transactions per micro-batch',
    buckets=[
        1, 5, 10, 50, 100,
        500, 1000, 5000
    ]
)

fraud_score_distribution = Histogram(
    'spark_fraud_score_distribution',
    'Distribution of fraud probability scores',
    buckets=[
        .0, .1, .2, .3, .4,
        .5, .6, .7, .8, .9, 1.0
    ]
)

current_fraud_rate = Gauge(
    'spark_current_fraud_rate',
    'Fraud rate in the last batch'
)

db_write_errors = Counter(
    'spark_db_write_errors_total',
    'Total database write errors'
)

kafka_alert_errors = Counter(
    'spark_kafka_alert_errors_total',
    'Total Kafka alert send errors'
)

inference_duration = Histogram(
    'spark_inference_duration_seconds',
    'Time to materialize + score a batch',
    buckets=[
        .25, .5, .75, 1,
        1.25, 1.5, 2,
        2.5, 3, 4, 5, 10
    ]
)

model_info = Gauge(
    'spark_model_info',
    'Currently loaded model',
    ['source', 'name', 'version']
)

shap_explanations_total = Counter(
    'spark_shap_explanations_total',
    'Total SHAP explanations computed'
)

shap_duration = Histogram(
    'spark_shap_duration_seconds',
    'Time to compute SHAP per batch',
    buckets=[
        .01, .05, .1, .25,
        .5, 1, 2
    ]
)


# ═══════════════════════════════════════════════════════════════
# MODEL MANAGER
# ═══════════════════════════════════════════════════════════════

class ModelManager:

    def __init__(
        self,
        model_path,
        scaler_path,
        metadata_path=METADATA_PATH
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path

        self._model = None
        self._scaler = None
        self.features = None
        self.scaled = None
        self.source = None
        self.version = None

    def _load_from_mlflow(self):

        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        client = MlflowClient()

        mv = client.get_model_version_by_alias(
            MLFLOW_MODEL_NAME,
            MLFLOW_MODEL_ALIAS
        )

        log.info(
            f"🤖 MLflow: {MLFLOW_MODEL_NAME} "
            f"v{mv.version} "
            f"(@{MLFLOW_MODEL_ALIAS})"
        )

        self._model = mlflow.sklearn.load_model(
            f"models:/{MLFLOW_MODEL_NAME}@"
            f"{MLFLOW_MODEL_ALIAS}"
        )

        local = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id,
            dst_path='/tmp/mlflow-artifacts'
        )

        self._scaler = joblib.load(
            os.path.join(local, 'scaler.pkl')
        )

        with open(
            os.path.join(local, 'metadata.json')
        ) as f:
            meta = json.load(f)

        self.features = list(
            meta['features']
        )

        self.scaled = list(
            meta['scaled_features']
        )

        self.source = 'mlflow'
        self.version = str(mv.version)

    def _load_from_disk(self):

        log.info(
            f"🤖 Local: {self.model_path}"
        )

        self._model = joblib.load(
            self.model_path
        )

        self._scaler = joblib.load(
            self.scaler_path
        )

        self.features, self.scaled = (
            load_feature_contract(
                self.metadata_path
            )
        )

        self.source = 'local'
        self.version = 'pkl'

    def load(self):

        if MLFLOW_TRACKING_URI:

            try:
                self._load_from_mlflow()

            except Exception as e:

                log.warning(
                    f"⚠️ MLflow load failed ({e}); "
                    f"fallback to local"
                )

                self._load_from_disk()

        else:
            self._load_from_disk()

        self._validate_contract()

        self._model.set_params(
            n_jobs=1
        )

        model_info.labels(
            self.source,
            MLFLOW_MODEL_NAME,
            self.version
        ).set(1)

        log.info(
            f"✅ Model ready — "
            f"source={self.source} "
            f"version={self.version} "
            f"features={len(self.features)} "
            f"scaled={self.scaled}"
        )

        return self

    def _validate_contract(self):

        errors = []

        fitted = list(
            getattr(
                self._scaler,
                'feature_names_in_',
                []
            )
        )

        if fitted != self.scaled:
            errors.append(
                f"Scaler fitted on {fitted}, "
                f"contract scales {self.scaled}"
            )

        booster = list(
            self._model.get_booster().feature_names
            or []
        )

        if booster != self.features:

            errors.append(
                f"Feature order mismatch "
                f"model:{booster} "
                f"contract:{self.features}"
            )

        if len(self.features) != 30:

            errors.append(
                f"Contract has "
                f"{len(self.features)} features, "
                f"expected 30"
            )

        if errors:

            raise RuntimeError(
                "❌ Model contract validation failed:\n  - "
                + "\n  - ".join(errors)
            )


# ═══════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════

class DatabaseManager:

    def __init__(self):
        self.conn = None
        self.cursor = None

    def connect(self, retries=10, wait=5):

        for attempt in range(
            1,
            retries + 1
        ):

            try:

                log.info(
                    f"🗄️ Connecting to PostgreSQL "
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

                log.info(
                    "✅ Connected to PostgreSQL"
                )

                return self

            except Exception as e:

                log.warning(
                    f"   DB not ready: {e}. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)

        raise RuntimeError(
            "Could not connect to PostgreSQL"
        )

    def create_tables(self):

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(20) UNIQUE,
                processed_at TIMESTAMP DEFAULT NOW(),
                amount DOUBLE PRECISION,
                fraud_probability DOUBLE PRECISION,
                is_fraud_predicted BOOLEAN,
                is_fraud_ground_truth INTEGER,
                merchant_id VARCHAR(20),
                card_last_four VARCHAR(4),
                country VARCHAR(2),
                processing_time_ms DOUBLE PRECISION,
                top_reasons TEXT
            );

            CREATE TABLE IF NOT EXISTS fraud_alerts (
                id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(20),
                alerted_at TIMESTAMP DEFAULT NOW(),
                fraud_probability DOUBLE PRECISION,
                amount DOUBLE PRECISION,
                merchant_id VARCHAR(20),
                country VARCHAR(2),
                top_reasons TEXT
            );

            CREATE TABLE IF NOT EXISTS batch_metrics (
                id SERIAL PRIMARY KEY,
                batch_id BIGINT,
                processed_at TIMESTAMP DEFAULT NOW(),
                batch_size INTEGER,
                fraud_count INTEGER,
                fraud_rate DOUBLE PRECISION,
                processing_ms DOUBLE PRECISION
            );
        """)

        # For existing databases
        self.cursor.execute(
            "ALTER TABLE transactions "
            "ADD COLUMN IF NOT EXISTS top_reasons TEXT;"
        )

        self.cursor.execute(
            "ALTER TABLE fraud_alerts "
            "ADD COLUMN IF NOT EXISTS top_reasons TEXT;"
        )

        self.conn.commit()

        log.info(
            "✅ Database tables ready"
        )

    def insert_transactions(self, records: list):

        if not records:
            return

        try:

            execute_batch(self.cursor, """
                INSERT INTO transactions (transaction_id, amount, fraud_probability, is_fraud_predicted, is_fraud_ground_truth, merchant_id, card_last_four, country, processing_time_ms, top_reasons)
                VALUES (%(transaction_id)s, %(amount)s, %(fraud_probability)s, %(is_fraud_predicted)s, %(is_fraud_ground_truth)s, %(merchant_id)s, %(card_last_four)s, %(country)s, %(processing_time_ms)s, %(top_reasons)s)
                ON CONFLICT (transaction_id) DO UPDATE SET 
                top_reasons = EXCLUDED.top_reasons, 
                fraud_probability = EXCLUDED.fraud_probability, 
                is_fraud_predicted = EXCLUDED.is_fraud_predicted,
                processed_at = NOW(),
                amount = EXCLUDED.amount
                """, records, page_size=500)

            self.conn.commit()

        except Exception as e:

            self.conn.rollback()

            db_write_errors.inc()

            log.error(
                f"❌ DB insert error: {e}"
            )

    def insert_batch_metric(
        self,
        batch_id,
        batch_size,
        fraud_count,
        processing_ms
    ):

        try:

            fraud_rate = (
                fraud_count / batch_size
                if batch_size > 0
                else 0
            )

            self.cursor.execute(
                """
                INSERT INTO batch_metrics (
                    batch_id,
                    batch_size,
                    fraud_count,
                    fraud_rate,
                    processing_ms
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    batch_id,
                    batch_size,
                    fraud_count,
                    fraud_rate,
                    processing_ms
                )
            )

            self.conn.commit()

        except Exception as e:

            self.conn.rollback()

            log.error(
                f"❌ Batch metric insert error: {e}"
            )


# ═══════════════════════════════════════════════════════════════
# ALERT PRODUCER
# ═══════════════════════════════════════════════════════════════

class AlertProducer:

    def __init__(self):
        self.producer = None

    def connect(self, retries=10, wait=5):

        for attempt in range(
            1,
            retries + 1
        ):

            try:

                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v:
                        json.dumps(v).encode('utf-8')
                )

                log.info(
                    "✅ Alert producer connected to Kafka"
                )

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

    def send_alert(
        self,
        transaction: dict,
        top_reasons=None
    ):

        try:

            alert = {
                'alert_id':
                    f"ALERT-{time.time_ns()}",

                'transaction_id':
                    transaction['transaction_id'],

                'timestamp':
                    time.time(),

                'fraud_probability':
                    transaction['fraud_probability'],

                'amount':
                    transaction['amount'],

                'merchant_id':
                    transaction['merchant_id'],

                'country':
                    transaction['country'],

                'card_last_four':
                    transaction['card_last_four'],

                'severity':
                    self._get_severity(
                        transaction[
                            'fraud_probability'
                        ]
                    ),

                'top_reasons':
                    top_reasons or {}
            }

            self.producer.send(
                KAFKA_TOPIC_ALERTS,
                value=alert
            )

        except Exception as e:

            kafka_alert_errors.inc()

            log.error(
                f"❌ Alert send error: {e}"
            )

    @staticmethod
    def _get_severity(p: float) -> str:

        if p >= 0.9:
            return 'CRITICAL'

        elif p >= 0.7:
            return 'HIGH'

        elif p >= 0.5:
            return 'MEDIUM'

        else:
            return 'LOW'


# ═══════════════════════════════════════════════════════════════
# KAFKA MESSAGE SCHEMA
# ═══════════════════════════════════════════════════════════════

def get_transaction_schema() -> StructType:

    v_fields = [
        StructField(
            f'V{i}',
            DoubleType(),
            True
        )
        for i in range(1, 29)
    ]

    return StructType([
        StructField(
            'transaction_id',
            StringType(),
            False
        ),
        StructField(
            'timestamp',
            DoubleType(),
            True
        ),
        StructField(
            'timestamp_iso',
            StringType(),
            True
        ),
        StructField(
            'Time',
            DoubleType(),
            True
        ),
        StructField(
            'Amount',
            DoubleType(),
            True
        ),
        *v_fields,
        StructField(
            'is_fraud_ground_truth',
            IntegerType(),
            True
        ),
        StructField(
            'merchant_id',
            StringType(),
            True
        ),
        StructField(
            'card_last_four',
            StringType(),
            True
        ),
        StructField(
            'country',
            StringType(),
            True
        )
    ])


# ═══════════════════════════════════════════════════════════════
# SPARK SESSION
# ═══════════════════════════════════════════════════════════════

def create_spark_session() -> SparkSession:

    jars = ','.join([
        '/app/spark-sql-kafka.jar',
        '/app/kafka-clients.jar',
        '/app/spark-token-provider-kafka.jar',
        '/app/commons-pool2.jar'
    ])

    spark = (
        SparkSession.builder
        .appName('FraudDetectionStreaming')
        .config('spark.jars', jars)
        .config(
            'spark.streaming.stopGracefullyOnShutdown',
            'true'
        )
        .config(
            'spark.sql.streaming.checkpointLocation',
            '/tmp/spark-checkpoints'
        )
        .config(
            'spark.sql.shuffle.partitions',
            '4'
        )
        .config(
            'spark.default.parallelism',
            '4'
        )
        .config(
            'spark.driver.memory',
            '2g'
        )
        .config(
            'spark.executor.memory',
            '2g'
        )
        .config(
            'spark.ui.enabled',
            'true'
        )
        .config(
            'spark.ui.port',
            '4040'
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        'WARN'
    )

    log.info(
        "✅ Spark session created"
    )

    log.info(
        f"   Spark version : {spark.version}"
    )

    log.info(
        "   Spark UI      : http://localhost:4040"
    )

    return spark


# ═══════════════════════════════════════════════════════════════
# FEATURE SCORING
# ═══════════════════════════════════════════════════════════════

def score_features(
    features: pd.DataFrame,
    model,
    scaler,
    feature_order=None,
    scaled_cols=None
) -> np.ndarray:

    feature_order = (
        feature_order
        or ALL_FEATURES
    )

    scaled_cols = (
        scaled_cols
        or SCALED_FEATURES
    )

    features = features.copy()

    features[scaled_cols] = (
        scaler.transform(
            features[scaled_cols]
        )
    )

    return model.predict_proba(
        features[feature_order]
    )[:, 1]


# ═══════════════════════════════════════════════════════════════
# SHAP EXPLANATIONS
# ═══════════════════════════════════════════════════════════════

def explain_high_confidence(
    high_df,
    model,
    scaler,
    explainer,
    features,
    scaled_cols,
    top_k=3
):

    if high_df.empty:
        return {}

    feat = high_df[
        features
    ].copy()

    feat[scaled_cols] = (
        scaler.transform(
            feat[scaled_cols]
        )
    )

    feat = feat[features]

    vals = explainer.shap_values(
        feat
    )

    if isinstance(vals, list):

        vals = (
            vals[1]
            if len(vals) > 1
            else vals[0]
        )

    result = {}

    for idx, (_, row) in enumerate(
        high_df.iterrows()
    ):

        row_shap = vals[idx]

        paired = list(
            zip(
                features,
                row_shap
            )
        )

        # Important:
        # rank by absolute SHAP impact,
        # not raw positive/negative value.
        top = sorted(
            paired,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]

        result[
            row['transaction_id']
        ] = {
            k: float(v)
            for k, v in top
        }

    return result


# ═══════════════════════════════════════════════════════════════
# PANDAS UDF FACTORY
# ═══════════════════════════════════════════════════════════════

def make_fraud_udf(model_broadcast):

    @pandas_udf(DoubleType())
    def predict_fraud_proba(
        time: pd.Series,
        amount: pd.Series,
        v1: pd.Series,
        v2: pd.Series,
        v3: pd.Series,
        v4: pd.Series,
        v5: pd.Series,
        v6: pd.Series,
        v7: pd.Series,
        v8: pd.Series,
        v9: pd.Series,
        v10: pd.Series,
        v11: pd.Series,
        v12: pd.Series,
        v13: pd.Series,
        v14: pd.Series,
        v15: pd.Series,
        v16: pd.Series,
        v17: pd.Series,
        v18: pd.Series,
        v19: pd.Series,
        v20: pd.Series,
        v21: pd.Series,
        v22: pd.Series,
        v23: pd.Series,
        v24: pd.Series,
        v25: pd.Series,
        v26: pd.Series,
        v27: pd.Series,
        v28: pd.Series
    ) -> pd.Series:

        model = (
            model_broadcast
            .value['model']
        )

        scaler = (
            model_broadcast
            .value['scaler']
        )

        features = pd.DataFrame({
            'Time': time,
            'Amount': amount,
            'V1': v1,
            'V2': v2,
            'V3': v3,
            'V4': v4,
            'V5': v5,
            'V6': v6,
            'V7': v7,
            'V8': v8,
            'V9': v9,
            'V10': v10,
            'V11': v11,
            'V12': v12,
            'V13': v13,
            'V14': v14,
            'V15': v15,
            'V16': v16,
            'V17': v17,
            'V18': v18,
            'V19': v19,
            'V20': v20,
            'V21': v21,
            'V22': v22,
            'V23': v23,
            'V24': v24,
            'V25': v25,
            'V26': v26,
            'V27': v27,
            'V28': v28
        })

        return pd.Series(
            score_features(
                features,
                model,
                scaler,
                model_broadcast
                    .value['features'],
                model_broadcast
                    .value['scaled']
            )
        )

    return predict_fraud_proba


# ═══════════════════════════════════════════════════════════════
# STREAMING SETUP
# ═══════════════════════════════════════════════════════════════

def run_streaming_with_udf(
    spark,
    model_broadcast,
    db_manager,
    alert_producer,
    explainer,
    model_manager
):

    schema = get_transaction_schema()

    fraud_udf = make_fraud_udf(
        model_broadcast
    )

    raw_stream = (
        spark.readStream
        .format('kafka')
        .option(
            'kafka.bootstrap.servers',
            KAFKA_BOOTSTRAP_SERVERS
        )
        .option(
            'subscribe',
            KAFKA_TOPIC_INPUT
        )
        .option(
            'startingOffsets',
            'latest'
        )
        .option(
            'maxOffsetsPerTrigger',
            10000
        )
        .option(
            'failOnDataLoss',
            'false'
        )
        .load()
    )

    parsed_stream = (
        raw_stream
        .select(
            F.from_json(
                F.col('value').cast('string'),
                schema
            ).alias('data'),
            F.col('timestamp').alias(
                'kafka_timestamp'
            ),
            F.col('partition'),
            F.col('offset')
        )
        .select(
            'data.*',
            'kafka_timestamp',
            'partition',
            'offset'
        )
        .filter(
            F.col(
                'transaction_id'
            ).isNotNull()
        )
    )

    scored_stream = (
        parsed_stream
        .withColumn(
            'fraud_probability',
            fraud_udf(
                col('Time'),
                col('Amount'),
                col('V1'),
                col('V2'),
                col('V3'),
                col('V4'),
                col('V5'),
                col('V6'),
                col('V7'),
                col('V8'),
                col('V9'),
                col('V10'),
                col('V11'),
                col('V12'),
                col('V13'),
                col('V14'),
                col('V15'),
                col('V16'),
                col('V17'),
                col('V18'),
                col('V19'),
                col('V20'),
                col('V21'),
                col('V22'),
                col('V23'),
                col('V24'),
                col('V25'),
                col('V26'),
                col('V27'),
                col('V28')
            )
        )
        .withColumn(
            'is_fraud_predicted',
            F.col(
                'fraud_probability'
            ) >= FRAUD_THRESHOLD
        )
    )

    log.info(
        "✅ Stream + UDF pipeline defined"
    )

    log.info(
        f"   Batch interval  : "
        f"{BATCH_INTERVAL} seconds"
    )

    log.info(
        f"   Fraud threshold : "
        f"{FRAUD_THRESHOLD}"
    )

    query = (
        scored_stream
        .writeStream
        .foreachBatch(
            lambda df, batch_id:
                process_batch_udf(
                    df,
                    batch_id,
                    db_manager,
                    alert_producer,
                    explainer,
                    model_manager
                )
        )
        .trigger(
            processingTime=
            f'{BATCH_INTERVAL} seconds'
        )
        .option(
            'checkpointLocation',
            '/tmp/spark-checkpoints/'
            'fraud-streaming'
        )
        .start()
    )

    return query


# ═══════════════════════════════════════════════════════════════
# BATCH PROCESSOR
# ═══════════════════════════════════════════════════════════════

def process_batch_udf(
    batch_df,
    batch_id: int,
    db_manager: DatabaseManager,
    alert_producer: AlertProducer,
    explainer,
    model_manager
):

    batch_start = time.time()

    if batch_df.isEmpty():

        log.debug(
            f"Batch {batch_id}: "
            f"empty, skipping"
        )

        return

    inference_start = time.time()

    pdf = batch_df.toPandas()

    inference_duration.observe(
        time.time() - inference_start
    )

    batch_size = len(pdf)

    fraud_count = int(
        pdf[
            'is_fraud_predicted'
        ].sum()
    )

    fraud_rate = (
        fraud_count / batch_size
        if batch_size > 0
        else 0
    )

    log.info(
        f"\n{'=' * 50}\n"
        f"📦 Batch {batch_id} | "
        f"{batch_size} transactions | "
        f"{fraud_count} fraud "
        f"({fraud_rate * 100:.2f}%)"
    )

    # ── PROMETHEUS ────────────────────────────────────────────

    transactions_processed.inc(
        batch_size
    )

    fraud_detected.inc(
        fraud_count
    )

    batch_size_metric.observe(
        batch_size
    )

    current_fraud_rate.set(
        fraud_rate
    )

    for score in pdf[
        'fraud_probability'
    ]:

        fraud_score_distribution.observe(
            float(score)
        )

    # ── SHAP FOR HIGH CONFIDENCE ──────────────────────────────

    high_conf = pdf[
        pdf['fraud_probability'] >= 0.9
    ]

    explanations = {}

    if not high_conf.empty:

        try:

            t0 = time.time()

            explanations = (
                explain_high_confidence(
                    high_conf,
                    model_manager._model,
                    model_manager._scaler,
                    explainer,
                    model_manager.features,
                    model_manager.scaled,
                    top_k=3
                )
            )

            shap_duration.observe(
                time.time() - t0
            )

            shap_explanations_total.inc(
                len(explanations)
            )

            log.info(
                f"   🔍 SHAP computed for "
                f"{len(explanations)} "
                f"high-confidence frauds"
            )

        except Exception as e:

            log.warning(
                f"   SHAP failed: {e}"
            )

    # ── FRAUD ALERTS → KAFKA ──────────────────────────────────

    fraud_rows = pdf[
        pdf['is_fraud_predicted'] == True
    ]

    for _, row in fraud_rows.iterrows():

        alert_producer.send_alert(
            {
                'transaction_id':
                    row['transaction_id'],

                'fraud_probability':
                    float(
                        row['fraud_probability']
                    ),

                'amount':
                    float(row['Amount']),

                'merchant_id':
                    row['merchant_id'],

                'country':
                    row['country'],

                'card_last_four':
                    row['card_last_four']
            },
            top_reasons=
                explanations.get(
                    row['transaction_id']
                )
        )

    if fraud_count > 0:

        log.info(
            f"   🚨 Sent "
            f"{fraud_count} alerts to Kafka"
        )

    # ── HIGH CONFIDENCE LOGGING ───────────────────────────────

    high_display = pdf[
        pdf['fraud_probability'] >= 0.9
    ]

    if len(high_display) > 0:

        log.info(
            "   ⚠️ HIGH CONFIDENCE fraud:"
        )

        for _, row in (
            high_display.iterrows()
        ):

            reasons = explanations.get(
                row['transaction_id'],
                {}
            )

            reason_str = (
                ", ".join(
                    [
                        f"{k}={v:.3f}"
                        for k, v in reasons.items()
                    ]
                )
                if reasons
                else "no SHAP"
            )

            log.info(
                f"      "
                f"{row['transaction_id']} | "
                f"${row['Amount']:.2f} | "
                f"Score: "
                f"{row['fraud_probability']:.4f} | "
                f"{row['country']} | "
                f"Reasons: {reason_str}"
            )

    # ── ALL TRANSACTIONS → POSTGRESQL ─────────────────────────

    batch_elapsed_ms = (
        time.time() - batch_start
    ) * 1000

    records = [
        {
            'transaction_id':
                row['transaction_id'],

            'amount':
                float(row['Amount']),

            'fraud_probability':
                float(
                    row['fraud_probability']
                ),

            'is_fraud_predicted':
                bool(
                    row['is_fraud_predicted']
                ),

            'is_fraud_ground_truth':
                int(
                    row['is_fraud_ground_truth']
                ),

            'merchant_id':
                row['merchant_id'],

            'card_last_four':
                row['card_last_four'],

            'country':
                row['country'],

            'processing_time_ms':
                batch_elapsed_ms,

            'top_reasons':
                json.dumps(
                    explanations.get(
                        row['transaction_id']
                    )
                )
                if explanations.get(
                    row['transaction_id']
                )
                else None
        }

        for _, row in pdf.iterrows()
    ]

    db_manager.insert_transactions(
        records
    )

    # ── BATCH METRICS → POSTGRESQL ────────────────────────────

    batch_processing_time.observe(
        batch_elapsed_ms / 1000
    )

    db_manager.insert_batch_metric(
        batch_id,
        batch_size,
        fraud_count,
        batch_elapsed_ms
    )

    log.info(
        f"   Total batch time : "
        f"{batch_elapsed_ms:.1f}ms\n"
        f"{'=' * 50}"
    )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():

    log.info("=" * 60)
    log.info(
        "FRAUD DETECTION STREAMING JOB STARTING"
    )
    log.info("=" * 60)

    # ── LOAD MODEL ────────────────────────────────────────────

    model_manager = (
        ModelManager(
            MODEL_PATH,
            SCALER_PATH
        ).load()
    )

    import shap

    log.info(
        "🔍 Creating SHAP explainer..."
    )

    explainer = shap.TreeExplainer(
        model_manager._model
    )

    log.info(
        "✅ SHAP explainer ready"
    )

    # ── CONNECT TO POSTGRES ──────────────────────────────────

    db_manager = (
        DatabaseManager()
        .connect()
    )

    db_manager.create_tables()

    # ── CONNECT ALERT PRODUCER ───────────────────────────────

    alert_producer = (
        AlertProducer()
        .connect()
    )

    # ── CREATE SPARK SESSION ─────────────────────────────────

    spark = create_spark_session()

    # ── BROADCAST MODEL + CONTRACT ───────────────────────────

    model_broadcast = (
        spark.sparkContext.broadcast(
            {
                'model':
                    model_manager._model,

                'scaler':
                    model_manager._scaler,

                'features':
                    model_manager.features,

                'scaled':
                    model_manager.scaled
            }
        )
    )

    log.info(
        "📡 Model broadcast to all workers"
    )

    # ── PROMETHEUS ────────────────────────────────────────────

    start_http_server(
        PROMETHEUS_PORT
    )

    log.info(
        f"📊 Prometheus metrics on "
        f"port {PROMETHEUS_PORT}"
    )

    # ── START STREAMING ──────────────────────────────────────

    query = run_streaming_with_udf(
        spark,
        model_broadcast,
        db_manager,
        alert_producer,
        explainer,
        model_manager
    )

    log.info(
        "🚀 Streaming query started"
    )

    log.info(
        f"   Query ID : {query.id}"
    )

    try:

        query.awaitTermination()

    except KeyboardInterrupt:

        log.info(
            "⛔ Stopping streaming job..."
        )

        query.stop()
        spark.stop()

        log.info(
            "✅ Stopped cleanly"
        )


# ── ENTRY POINT ───────────────────────────────────────────────

if __name__ == '__main__':
    main()