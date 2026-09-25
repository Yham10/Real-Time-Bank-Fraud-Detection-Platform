import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)
from xgboost import XGBClassifier
import joblib
import json
import os

print("=" * 60)
print("FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv('data/creditcard.csv')
print(f"   Shape: {df.shape}")
print(f"   Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# ── 2. FEATURES & TARGET ──────────────────────────────────────
print("\n🔧 Preparing features...")
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
print("\n✂️  Splitting data...")
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
print("\n⚖️  Handling class imbalance...")
fraud_count   = y_train.sum()
legit_count   = len(y_train) - fraud_count
scale_pos_weight = legit_count / fraud_count

print(f"   Legitimate: {legit_count}")
print(f"   Fraud:      {fraud_count}")
print(f"   scale_pos_weight: {scale_pos_weight:.1f}")

# ── 5. TRAIN MODEL ────────────────────────────────────────────
print("\n🤖 Training XGBoost model...")
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
print("\n📊 Evaluating model...")
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
print("\n🏆 Top 10 Most Important Features:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance.head(10).iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"   {row['feature']:10s}: {bar} {row['importance']:.4f}")

# ── 8. SAVE MODEL ─────────────────────────────────────────────
print("\n💾 Saving model...")
os.makedirs('model', exist_ok=True)

# Save model
joblib.dump(model, 'model/fraud_model.pkl')

# Save scaler (needed for inference)
joblib.dump(scaler, 'model/scaler.pkl')

# Save metadata
metadata = {
    'auc_roc': round(auc_roc, 4),
    'features': feature_cols,
    'n_estimators': 100,
    'threshold': 0.5,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'fraud_rate_train': round(float(y_train.mean()), 6),
    'scale_pos_weight': round(float(scale_pos_weight), 2)
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   Model saved  → model/fraud_model.pkl")
print(f"   Scaler saved → model/scaler.pkl")
print(f"   Metadata     → model/metadata.json")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print(f"   Final AUC-ROC: {auc_roc:.4f}")
print("=" * 60)