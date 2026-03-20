"""
train_lgbm.py v3 — Treino LightGBM otimizado

Mudanças v3:
  - Early stopping
  - Hiperparâmetros ajustados para mais dados
  - Calibração com sigmoid (compatível com qualquer versão sklearn)
  - Análise de erros (FP/FN) salva em CSV
  - Métricas Recall@Top50 e Recall@Top100
  - Distribuição de scores por classe
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")

INPUT_DATA = os.path.join(DADOS_DIR, "base_mvp_model_ready.csv")

MODEL_PATH = os.path.join(ARTEFACT_DIR, "model_lightgbm.joblib")
MODEL_CALIBRATED_PATH = os.path.join(ARTEFACT_DIR, "model_lightgbm_calibrated.joblib")
METRICS_PATH = os.path.join(ARTEFACT_DIR, "metricas_lightgbm.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTEFACT_DIR, "feature_importance_lightgbm.csv")
PREDICTIONS_TEST_PATH = os.path.join(ARTEFACT_DIR, "predicoes_teste_lightgbm.csv")
SHAP_PATH = os.path.join(ARTEFACT_DIR, "shap_summary.csv")
ERROR_ANALYSIS_PATH = os.path.join(ARTEFACT_DIR, "error_analysis.csv")
SCORE_DISTRIBUTION_PATH = os.path.join(ARTEFACT_DIR, "score_distribution.csv")

os.makedirs(ARTEFACT_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def precision_at_k(y_true, y_score, k_ratio=0.01):
    y_true, y_score = np.array(y_true), np.array(y_score)
    k = max(1, int(len(y_true) * k_ratio))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / k)


def recall_at_k(y_true, y_score, k_ratio=0.01):
    y_true, y_score = np.array(y_true), np.array(y_score)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    k = max(1, int(len(y_true) * k_ratio))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / positives)


def recall_at_top_n(y_true, y_score, n):
    y_true, y_score = np.array(y_true), np.array(y_score)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    k = min(n, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / positives)


def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else None,
        "average_precision": float(average_precision_score(y_true, y_prob)) if has_both else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision_at_1pct": precision_at_k(y_true, y_prob, 0.01),
        "recall_at_1pct": recall_at_k(y_true, y_prob, 0.01),
        "precision_at_5pct": precision_at_k(y_true, y_prob, 0.05),
        "recall_at_5pct": recall_at_k(y_true, y_prob, 0.05),
        "recall_at_top_50": recall_at_top_n(y_true, y_prob, 50),
        "recall_at_top_100": recall_at_top_n(y_true, y_prob, 100),
    }


def find_best_threshold_by_f1(y_true, y_prob):
    best_threshold, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return float(best_threshold), float(best_f1)


def print_split_summary(name, df, label_col="is_fraud"):
    n = len(df)
    n_fraud = int(df[label_col].sum())
    date_min = df["event_datetime"].min()
    date_max = df["event_datetime"].max()
    print(f"  {name}: {n} rows | {n_fraud} fraudes ({n_fraud/n*100:.2f}%) | "
          f"{date_min} → {date_max}")


# =========================================================
# LOAD DATA
# =========================================================
print("=" * 70)
print("TREINO LightGBM v3 — Otimizado")
print("=" * 70)

print("\nLendo base model-ready...")
df = pd.read_csv(INPUT_DATA)
df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
df = df.sort_values("event_datetime").reset_index(drop=True)
df = df[df["is_fraud"].notna()].copy()
df["is_fraud"] = df["is_fraud"].astype(int)

print(f"Shape: {df.shape}")
print(f"Fraudes: {df['is_fraud'].sum()} | Normais: {(df['is_fraud'] == 0).sum()}")
print(f"Proporção fraude: {df['is_fraud'].mean()*100:.3f}%")


# =========================================================
# SPLIT TEMPORAL
# =========================================================
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print("\nSplit temporal:")
print_split_summary("Treino", df_train)
print_split_summary("Validação", df_val)
print_split_summary("Teste", df_test)

drop_cols = ["transaction_id", "customer_id", "event_datetime", "source_dataset", "is_fraud"]
feature_cols = [c for c in df.columns if c not in drop_cols]

# CORREÇÃO: resetar índices para consistência entre DataFrames
X_train = df_train[feature_cols].reset_index(drop=True)
y_train = df_train["is_fraud"].reset_index(drop=True)

X_val = df_val[feature_cols].reset_index(drop=True)
y_val = df_val["is_fraud"].reset_index(drop=True)

X_test = df_test[feature_cols].reset_index(drop=True)
y_test = df_test["is_fraud"].reset_index(drop=True)


# =========================================================
# CLASS IMBALANCE
# =========================================================
n_pos = int(y_train.sum())
n_neg = len(y_train) - n_pos
if n_pos == 0:
    raise ValueError("Sem positivos no treino.")
scale_pos_weight = n_neg / n_pos
print(f"\nscale_pos_weight: {scale_pos_weight:.2f} ({n_pos} pos / {n_neg} neg)")


# =========================================================
# MODEL
# =========================================================
print("\nTreinando LightGBM v3...")
model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1500,
    learning_rate=0.01,
    num_leaves=63,
    max_depth=7,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    callbacks=[
        early_stopping(stopping_rounds=100, verbose=True),
        log_evaluation(period=100),
    ],
)

best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") else -1
print(f"  Best iteration: {best_iter}")


# =========================================================
# CALIBRATION — compatível com qualquer versão sklearn
# =========================================================
print("\nCalibrando probabilidades (sigmoid)...")
use_calibrated = False
y_test_prob_cal = None

try:
    # Tentar cv="prefit" primeiro (sklearn >= 1.0, < 1.6)
    calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_val, y_val)
    use_calibrated = True
except (TypeError, ValueError):
    try:
        # Fallback: usar cv=2 com treino+val concatenados
        print("  cv='prefit' não suportado, usando cv=2...")
        X_cal = pd.concat([X_train, X_val], ignore_index=True)
        y_cal = pd.concat([y_train, y_val], ignore_index=True)
        calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv=2)
        calibrated_model.fit(X_cal, y_cal)
        use_calibrated = True
    except Exception as e2:
        print(f"  ⚠ Calibração falhou completamente: {e2}")

if use_calibrated:
    joblib.dump(calibrated_model, MODEL_CALIBRATED_PATH)
    print("  ✓ Calibração OK")


# =========================================================
# THRESHOLD SEARCH
# =========================================================
y_val_prob = model.predict_proba(X_val)[:, 1]
best_threshold, best_val_f1 = find_best_threshold_by_f1(y_val, y_val_prob)
print(f"\nMelhor threshold (F1 val): {best_threshold:.4f} → F1={best_val_f1:.4f}")


# =========================================================
# EVALUATION
# =========================================================
print("\nAvaliando...")
y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

if use_calibrated:
    y_test_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
else:
    y_test_prob_cal = y_test_prob

metrics = {
    "train_threshold_0_5": evaluate_metrics(y_train, y_train_prob, 0.5),
    "val_threshold_0_5": evaluate_metrics(y_val, y_val_prob, 0.5),
    "test_threshold_0_5": evaluate_metrics(y_test, y_test_prob, 0.5),
    "train_best_threshold": evaluate_metrics(y_train, y_train_prob, best_threshold),
    "val_best_threshold": evaluate_metrics(y_val, y_val_prob, best_threshold),
    "test_best_threshold": evaluate_metrics(y_test, y_test_prob, best_threshold),
}
if use_calibrated:
    metrics["test_calibrated_0_5"] = evaluate_metrics(y_test, y_test_prob_cal, 0.5)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
    "importance_split": model.booster_.feature_importance(importance_type="split"),
}).sort_values("importance_gain", ascending=False).reset_index(drop=True)


# =========================================================
# SHAP
# =========================================================
print("\nCalculando SHAP...")
shap_importance = None
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_importance.to_csv(SHAP_PATH, index=False)
    print(f"  SHAP salvo. Top 10:")
    print(shap_importance.head(10).to_string(index=False))
except ImportError:
    print("  ⚠ shap não instalado")
except Exception as e:
    print(f"  ⚠ Erro SHAP: {e}")


# =========================================================
# ERROR ANALYSIS — CORREÇÃO: usar índices consistentes
# =========================================================
print("\nAnálise de erros...")

# Resetar índices do df_test também para consistência
df_test_reset = df_test.reset_index(drop=True)

pred_test = pd.DataFrame({
    "transaction_id": df_test_reset["transaction_id"].values,
    "customer_id": df_test_reset["customer_id"].values,
    "event_datetime": df_test_reset["event_datetime"].values,
    "is_fraud": y_test.values,
    "score_fraude": y_test_prob,
    "score_fraude_calibrated": y_test_prob_cal,
    "pred_0_5": (y_test_prob >= 0.5).astype(int),
    "pred_best": (y_test_prob >= best_threshold).astype(int),
})

# Identificar erros
fn_mask = (pred_test["is_fraud"] == 1) & (pred_test["pred_0_5"] == 0)
fp_mask = (pred_test["is_fraud"] == 0) & (pred_test["pred_0_5"] == 1)

errors = pred_test[fn_mask | fp_mask].copy()
errors["error_type"] = np.where(errors["is_fraud"] == 1, "FALSE_NEGATIVE", "FALSE_POSITIVE")

# Adicionar features — agora os índices batem
error_features = [
    "vl_pix", "vl_mediana_pix_trimestre", "qt_total_pix_trimestre",
    "nr_idade", "qt_tempo_relacionamento_mes", "topaz_risk_score",
    "topaz_score_filled", "tx_count_prev_30m", "burst_30m_flag",
    "first_receiver_flag", "rule_score_raw", "vl_pix_over_1000_flag",
    "is_first_tx_trimestre", "hour", "day_of_week",
    "latencia_rede_ms_final", "app_version_minor",
]

for feat in error_features:
    if feat in X_test.columns:
        errors[feat] = X_test.loc[errors.index, feat].values

errors.to_csv(ERROR_ANALYSIS_PATH, index=False)
print(f"  {len(errors)} erros salvos em: {ERROR_ANALYSIS_PATH}")
print(f"    FN (fraudes perdidas): {fn_mask.sum()}")
print(f"    FP (falsos alarmes):   {fp_mask.sum()}")

if len(errors) > 0:
    print("\n  Detalhes dos erros:")
    for _, row in errors.iterrows():
        etype = row["error_type"]
        score = row["score_fraude"]
        vlpix = row.get("vl_pix", "?")
        rules = row.get("rule_score_raw", "?")
        idade = row.get("nr_idade", "?")
        tx_id = str(row["transaction_id"])[:35]
        print(f"    [{etype:15s}] score={score:.4f} vl_pix={vlpix:>10} "
              f"idade={idade} rules={rules} tx={tx_id}...")


# =========================================================
# SCORE DISTRIBUTION
# =========================================================
print("\nDistribuição de scores por classe...")
score_dist = pd.DataFrame({"class": y_test.values, "score": y_test_prob})

dist_stats = score_dist.groupby("class")["score"].describe()
print(dist_stats.to_string())

score_dist.to_csv(SCORE_DISTRIBUTION_PATH, index=False)


# =========================================================
# SAVE ALL
# =========================================================
pred_test.to_csv(PREDICTIONS_TEST_PATH, index=False)

metadata = {
    "model_type": "LightGBM",
    "version": "v3_optimized",
    "n_train": int(len(df_train)),
    "n_val": int(len(df_val)),
    "n_test": int(len(df_test)),
    "n_features": int(len(feature_cols)),
    "positives_train": n_pos,
    "negatives_train": n_neg,
    "fraud_ratio_train": float(n_pos / (n_pos + n_neg)),
    "scale_pos_weight": float(scale_pos_weight),
    "best_iteration": int(best_iter),
    "best_threshold_val_f1": float(best_threshold),
    "best_val_f1": float(best_val_f1),
    "calibrated": use_calibrated,
    "n_errors_fn": int(fn_mask.sum()),
    "n_errors_fp": int(fp_mask.sum()),
    "metrics": metrics,
    "top_20_features_gain": feature_importance.head(20).to_dict(orient="records"),
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
joblib.dump(model, MODEL_PATH)


# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("RESULTADOS v3")
print("=" * 70)

print(f"\nTeste @ threshold 0.5:")
for k, v in metrics["test_threshold_0_5"].items():
    print(f"  {k}: {v}")

print(f"\nTeste @ best threshold ({best_threshold:.4f}):")
for k, v in metrics["test_best_threshold"].items():
    print(f"  {k}: {v}")

if use_calibrated and "test_calibrated_0_5" in metrics:
    print(f"\nTeste calibrado @ 0.5:")
    for k, v in metrics["test_calibrated_0_5"].items():
        print(f"  {k}: {v}")

print(f"\nTop 20 features (gain):")
print(feature_importance.head(20).to_string(index=False))

print(f"\nArtefatos salvos:")
print(f"  Modelo:            {MODEL_PATH}")
if use_calibrated:
    print(f"  Modelo calibrado:  {MODEL_CALIBRATED_PATH}")
print(f"  Métricas:          {METRICS_PATH}")
print(f"  Feature importance:{FEATURE_IMPORTANCE_PATH}")
print(f"  Predições teste:   {PREDICTIONS_TEST_PATH}")
print(f"  Análise de erros:  {ERROR_ANALYSIS_PATH}")
print(f"  Score distribution:{SCORE_DISTRIBUTION_PATH}")
if shap_importance is not None:
    print(f"  SHAP summary:      {SHAP_PATH}")
print("=" * 70)
