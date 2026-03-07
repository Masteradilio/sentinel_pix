""" 
O que esse script faz:

leitura da base em /dados/base_mvp_model_ready.csv
split temporal
treino supervisionado
tratamento de desbalanceamento
métricas de desempenho
feature importance
artefato do modelo salvo

que ele vai salvar:

/backend/artefatos/model_lightgbm.joblib
/backend/artefatos/metricas_lightgbm.json
/backend/artefatos/feature_importance_lightgbm.csv

"""


import os
import json
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")

INPUT_DATA = os.path.join(DADOS_DIR, "base_mvp_model_ready.csv")

MODEL_PATH = os.path.join(ARTEFACT_DIR, "model_lightgbm.joblib")
METRICS_PATH = os.path.join(ARTEFACT_DIR, "metricas_lightgbm.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTEFACT_DIR, "feature_importance_lightgbm.csv")
PREDICTIONS_TEST_PATH = os.path.join(ARTEFACT_DIR, "predicoes_teste_lightgbm.csv")

os.makedirs(ARTEFACT_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def precision_at_k(y_true, y_score, k_ratio=0.01):
    """
    Precision nos top K% maiores scores.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n = len(y_true)
    k = max(1, int(n * k_ratio))

    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / k)


def recall_at_k(y_true, y_score, k_ratio=0.01):
    """
    Recall capturado nos top K% maiores scores.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    positives = y_true.sum()
    if positives == 0:
        return 0.0

    n = len(y_true)
    k = max(1, int(n * k_ratio))

    idx = np.argsort(-y_score)[:k]
    captured = y_true[idx].sum()
    return float(captured / positives)


def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision_at_1pct": precision_at_k(y_true, y_prob, 0.01),
        "recall_at_1pct": recall_at_k(y_true, y_prob, 0.01),
        "precision_at_5pct": precision_at_k(y_true, y_prob, 0.05),
        "recall_at_5pct": recall_at_k(y_true, y_prob, 0.05),
        "precision_at_10pct": precision_at_k(y_true, y_prob, 0.10),
        "recall_at_10pct": recall_at_k(y_true, y_prob, 0.10),
    }
    return metrics


def find_best_threshold_by_f1(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1 = -1

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return float(best_threshold), float(best_f1)


# =========================================================
# LOAD DATA
# =========================================================
print("Lendo base model-ready...")
df = pd.read_csv(INPUT_DATA)

# parse datetime
df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")

# ordenar por tempo
df = df.sort_values("event_datetime").reset_index(drop=True)

# remover linhas sem label
df = df[df["is_fraud"].notna()].copy()
df["is_fraud"] = df["is_fraud"].astype(int)

print(f"Shape da base: {df.shape}")
print(f"Fraudes: {df['is_fraud'].sum()} | Não fraudes: {(df['is_fraud'] == 0).sum()}")


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
print(f"Treino: {df_train.shape}")
print(f"Validação: {df_val.shape}")
print(f"Teste: {df_test.shape}")

# colunas não usadas no treino
drop_cols = [
    "transaction_id",
    "customer_id",
    "event_datetime",
    "source_dataset",
    "join_status_mobile",
    "is_fraud",
]

feature_cols = [c for c in df.columns if c not in drop_cols]

X_train = df_train[feature_cols].copy()
y_train = df_train["is_fraud"].copy()

X_val = df_val[feature_cols].copy()
y_val = df_val["is_fraud"].copy()

X_test = df_test[feature_cols].copy()
y_test = df_test["is_fraud"].copy()


# =========================================================
# CLASS IMBALANCE
# =========================================================
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos

if n_pos == 0:
    raise ValueError("Não há exemplos positivos no conjunto de treino.")

scale_pos_weight = n_neg / n_pos

print("\nBalanceamento:")
print(f"Positivos treino: {n_pos}")
print(f"Negativos treino: {n_neg}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# =========================================================
# MODEL
# =========================================================
print("\nTreinando LightGBM...")

model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric="auc"
)


# =========================================================
# VALIDATION THRESHOLD SEARCH
# =========================================================
y_val_prob = model.predict_proba(X_val)[:, 1]
best_threshold, best_val_f1 = find_best_threshold_by_f1(y_val, y_val_prob)

print(f"\nMelhor threshold na validação (por F1): {best_threshold:.4f}")
print(f"Melhor F1 validação: {best_val_f1:.4f}")


# =========================================================
# EVALUATION
# =========================================================
print("\nAvaliando desempenho...")

y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "train_threshold_0_5": evaluate_metrics(y_train, y_train_prob, threshold=0.5),
    "val_threshold_0_5": evaluate_metrics(y_val, y_val_prob, threshold=0.5),
    "test_threshold_0_5": evaluate_metrics(y_test, y_test_prob, threshold=0.5),

    "train_best_threshold": evaluate_metrics(y_train, y_train_prob, threshold=best_threshold),
    "val_best_threshold": evaluate_metrics(y_val, y_val_prob, threshold=best_threshold),
    "test_best_threshold": evaluate_metrics(y_test, y_test_prob, threshold=best_threshold),
}


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
    "importance_split": model.booster_.feature_importance(importance_type="split"),
})

feature_importance = feature_importance.sort_values(
    by="importance_gain",
    ascending=False
).reset_index(drop=True)


# =========================================================
# SAVE PREDICTIONS TEST
# =========================================================
pred_test_df = pd.DataFrame({
    "transaction_id": df_test["transaction_id"].values,
    "customer_id": df_test["customer_id"].values,
    "event_datetime": df_test["event_datetime"].values,
    "is_fraud": y_test.values,
    "score_fraude": y_test_prob,
    "pred_0_5": (y_test_prob >= 0.5).astype(int),
    "pred_best_threshold": (y_test_prob >= best_threshold).astype(int),
})

pred_test_df.to_csv(PREDICTIONS_TEST_PATH, index=False)


# =========================================================
# SAVE METRICS / MODEL / IMPORTANCE
# =========================================================
metadata = {
    "model_type": "LightGBM",
    "n_train": int(len(df_train)),
    "n_val": int(len(df_val)),
    "n_test": int(len(df_test)),
    "n_features": int(len(feature_cols)),
    "positives_train": int(n_pos),
    "negatives_train": int(n_neg),
    "scale_pos_weight": float(scale_pos_weight),
    "best_threshold_val_f1": float(best_threshold),
    "best_val_f1": float(best_val_f1),
    "metrics": metrics,
    "top_20_features_gain": feature_importance.head(20).to_dict(orient="records"),
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
joblib.dump(model, MODEL_PATH)


# =========================================================
# PRINT SUMMARY
# =========================================================
print("\n================ RESULTADOS ================\n")

print("Teste @ threshold 0.5")
for k, v in metrics["test_threshold_0_5"].items():
    print(f"{k}: {v}")

print("\nTeste @ best threshold")
for k, v in metrics["test_best_threshold"].items():
    print(f"{k}: {v}")

print("\nTop 20 features por importance_gain:")
print(feature_importance.head(20).to_string(index=False))

print("\nArtefatos salvos:")
print(f"Modelo: {MODEL_PATH}")
print(f"Métricas: {METRICS_PATH}")
print(f"Feature importance: {FEATURE_IMPORTANCE_PATH}")
print(f"Predições teste: {PREDICTIONS_TEST_PATH}")
