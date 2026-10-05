import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from xgboost import XGBClassifier
import joblib, json, os, hashlib
import sklearn, xgboost
from datetime import datetime, timezone

DATA_PATH = os.getenv('DATA_PATH', 'data/creditcard.csv')
OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', 'model')

def file_md5(path, n=12):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:n]

print("="*60)
print("FRAUD DETECTION MODEL TRAINING")
print("="*60)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

print("\nPreparing features...")
feature_cols = [c for c in df.columns if c != 'Class']
X = df[feature_cols]
y = df['Class']
scaler = StandardScaler()
X = X.copy()
X[['Amount','Time']] = scaler.fit_transform(X[['Amount','Time']])
print(f"   Features: {len(feature_cols)}")

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)} Test: {len(X_test)}")

print("\nHandling class imbalance...")
fraud_count = y_train.sum()
legit_count = len(y_train)-fraud_count
scale_pos_weight = legit_count/fraud_count
print(f"   scale_pos_weight: {scale_pos_weight:.1f}")

print("\nTraining XGBoost...")
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='auc', random_state=42, n_jobs=-1)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("   Training complete ✅")

print("\nEvaluating...")
y_pred_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_pred_proba>=0.5).astype(int)
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"   AUC-ROC: {auc_roc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Legit','Fraud']))
cm = confusion_matrix(y_test, y_pred)
print(f"   TN:{cm[0,0]} FP:{cm[0,1]} FN:{cm[1,0]} TP:{cm[1,1]}")

# ── REFERENCE SAMPLE ──
print("\nSaving reference sample...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
ref_idx = X_test.sample(n=min(10000, len(X_test)), random_state=42).index
reference = df.loc[ref_idx, feature_cols].copy()
reference['fraud_probability'] = model.predict_proba(X_test.loc[ref_idx])[:,1]
reference.to_parquet(os.path.join(OUTPUT_DIR, 'reference_sample.parquet'), index=False)
print(f"   Reference {len(reference)} rows -> {OUTPUT_DIR}/reference_sample.parquet")

print("\nTop 10 Important:")
imp = pd.DataFrame({'feature':feature_cols, 'importance':model.feature_importances_}).sort_values('importance', ascending=False)
for _, r in imp.head(10).iterrows():
    print(f"   {r['feature']}: {r['importance']:.4f}")

# ── SHAP ──
shap_path = os.path.join(OUTPUT_DIR, 'shap_summary.png')
try:
    import shap
    import matplotlib.pyplot as plt
    print("\nGenerating SHAP summary...")
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, plot_size=(10,6))
    plt.tight_layout()
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   SHAP saved -> {shap_path}")
except Exception as e:
    print(f"   SHAP failed (optional): {e}")

# ── SAVE + METADATA ──
print("\nSaving model...")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
metadata = {
    'model_type':'XGBClassifier',
    'trained_at':datetime.now(timezone.utc).isoformat(),
    'features':feature_cols,
    'scaled_features':['Amount','Time'],
    'n_features':len(feature_cols),
    'threshold':0.5,
    'metrics':{'auc_roc':round(float(auc_roc),4),'precision':round(float(precision),4),'recall':round(float(recall),4),'f1':round(float(f1),4),'fpr':round(float(cm[0,1]/(cm[0,0]+cm[0,1])),6)},
    'n_estimators':100,'max_depth':6,'learning_rate':0.1,'scale_pos_weight':round(float(scale_pos_weight),2),
    'train_size':int(len(X_train)),'test_size':int(len(X_test)),'fraud_rate_train':round(float(y_train.mean()),6),
    'versions':{'xgboost':xgboost.__version__,'scikit-learn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, os.path.join(OUTPUT_DIR, 'fraud_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
with open(os.path.join(OUTPUT_DIR, 'metadata.json'),'w') as f:
    json.dump(metadata, f, indent=2)

# contract check
_scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
_model = joblib.load(os.path.join(OUTPUT_DIR, 'fraud_model.pkl'))
assert list(_scaler.feature_names_in_) == metadata['scaled_features']
assert list(_model.get_booster().feature_names) == metadata['features']
print("   Contract OK ✅")

# ── MLFLOW (optional) ──
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')
MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME','fraud-xgb')
MODEL_ALIAS = os.getenv('MLFLOW_MODEL_ALIAS','production')
if MLFLOW_URI:
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment('fraud-detection')
    with mlflow.start_run() as run:
        mlflow.log_params({'n_estimators':100,'max_depth':6,'learning_rate':0.1,'scale_pos_weight':round(float(scale_pos_weight),2),'threshold':0.5})
        mlflow.log_metrics(metadata['metrics'])
        mlflow.set_tags({'features_hash': hashlib.md5(",".join(feature_cols).encode()).hexdigest()[:12], 'n_features': len(feature_cols), 'data_hash': file_md5(DATA_PATH) if os.path.exists(DATA_PATH) else 'no-data'})
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'metadata.json'))
        if os.path.exists(os.path.join(OUTPUT_DIR, 'reference_sample.parquet')):
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'reference_sample.parquet'))
        if os.path.exists(shap_path):
            mlflow.log_artifact(shap_path)
        mlflow.sklearn.log_model(model, artifact_path='model', registered_model_name=MODEL_NAME)
        run_id = run.info.run_id
    client = MlflowClient()
    new_version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id==run_id)
    new_auc = metadata['metrics']['auc_roc']
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        current_auc = client.get_run(current.run_id).data.metrics.get('auc_roc',0.0)
    except Exception:
        current, current_auc = None, None
    if current is None or new_auc >= current_auc:
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, str(new_version))
        print(f"\n🚀 {MODEL_NAME} v{new_version} -> @{MODEL_ALIAS}")
    else:
        print(f"\n⏸️  v{new_version} NOT promoted: {new_auc} < {current_auc}")
else:
    print("\nℹ️  MLFLOW_TRACKING_URI not set — skipped tracking/registry")