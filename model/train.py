import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import json
import os

DATA_PATH = os.getenv('DATA_PATH', 'data/creditcard.csv')
OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', 'model')

import sklearn, xgboost
from datetime import datetime, timezone
from sklearn.metrics import precision_recall_fscore_support

print("=" * 60)
print("FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# ── 2. FEATURES & TARGET ──────────────────────────────────────
print("\nPreparing features...")
feature_cols = [col for col in df.columns if col != 'Class']
X = df[feature_cols]
y = df['Class']

# Scale Amount and Time (V1-V28 already scaled by bank)
scaler = StandardScaler()
X = X.copy()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

print(f"   Features: {len(feature_cols)}")
print(f"   Feature names: {feature_cols}")

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserve fraud ratio in both splits
)

print(f"   Train size: {len(X_train)}")
print(f"   Test size:  {len(X_test)}")
print(f"   Train fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
print(f"   Test fraud:  {y_test.sum()} ({y_test.mean()*100:.3f}%)")

# ── 4. HANDLE CLASS IMBALANCE ─────────────────────────────────
print("\nHandling class imbalance...")
fraud_count   = y_train.sum()
legit_count   = len(y_train) - fraud_count
scale_pos_weight = legit_count / fraud_count

print(f"   Legitimate: {legit_count}")
print(f"   Fraud:      {fraud_count}")
print(f"   scale_pos_weight: {scale_pos_weight:.1f}")

# ── 5. TRAIN MODEL ────────────────────────────────────────────
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # handle imbalance
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("   Training complete ✅")

# ── 6. EVALUATE ───────────────────────────────────────────────
print("\nEvaluating model...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)

auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"\n   AUC-ROC Score: {auc_roc:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred,
                           target_names=['Legitimate', 'Fraud']))

cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
print(f"   TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
print(f"   FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
print(f"\n   False Positive Rate: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.2f}%")
print(f"   False Negative Rate: {cm[1,0]/(cm[1,0]+cm[1,1])*100:.2f}%")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────
print("\nTop 10 Most Important Features:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance.head(10).iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"   {row['feature']:10s}: {bar} {row['importance']:.4f}")

# ── 8. SAVE MODEL + CONTRACT ──────────────────────────────────
print("\nSaving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(model, os.path.join(OUTPUT_DIR, 'fraud_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary'
)

metadata = {
    'model_type'      : 'XGBClassifier',
    'trained_at'      : datetime.now(timezone.utc).isoformat(),
    # ── CONTRACT (consumed by spark/streaming_job.py) ──
    'features'        : feature_cols,          # ORDER MATTERS
    'scaled_features' : ['Amount', 'Time'],    # order the scaler was fitted with
    'n_features'      : len(feature_cols),
    'threshold'       : 0.5,
    # ── METRICS ──
    'metrics': {
        'auc_roc'  : round(float(auc_roc), 4),
        'precision': round(float(precision), 4),
        'recall'   : round(float(recall), 4),
        'f1'       : round(float(f1), 4),
        'fpr'      : round(float(cm[0,1] / (cm[0,0] + cm[0,1])), 6),
    },
    # ── TRAINING SETUP ──
    'n_estimators'    : 100,
    'max_depth'       : 6,
    'learning_rate'   : 0.1,
    'scale_pos_weight': round(float(scale_pos_weight), 2),
    'train_size'      : int(len(X_train)),
    'test_size'       : int(len(X_test)),
    'fraud_rate_train': round(float(y_train.mean()), 6),
    # ── ENVIRONMENT (reproducibility) ──
    'versions': {
        'xgboost'     : xgboost.__version__,
        'scikit-learn': sklearn.__version__,
        'pandas'      : pd.__version__,
        'numpy'       : np.__version__,
    },
}

with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# ── 9. TRAIN-TIME CONTRACT CHECK ──────────────────────────────
# Same checks the streaming job runs at startup. Catch skew here first.
print("\nValidating saved artifacts against contract...")
_scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
_model = joblib.load(os.path.join(OUTPUT_DIR, 'fraud_model.pkl'))
assert list(_scaler.feature_names_in_) == metadata['scaled_features'], \
    f"scaler fitted on {list(_scaler.feature_names_in_)}"
assert list(_model.get_booster().feature_names) == metadata['features'], \
    "model feature order != metadata['features']"
print("   Contract OK ✅")

print(f"   Model saved  → {os.path.join(OUTPUT_DIR, 'fraud_model.pkl')}")
print(f"   Scaler saved → {os.path.join(OUTPUT_DIR, 'scaler.pkl')}")
print(f"   Metadata     → {os.path.join(OUTPUT_DIR, 'metadata.json')}")