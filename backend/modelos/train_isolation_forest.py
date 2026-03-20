"""
train_isolation_forest.py v3 — IF Especializado em primeiras transações

Estratégia:
  - Treinar APENAS com primeiras transações do trimestre (is_first_tx_trimestre=1)
  - Aprender o perfil "normal" de uma primeira transação
  - Score só usado quando LGBM está na zona cinzenta E é primeira tx
  - Foco em features absolutas (valor, idade, relacionamento)
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")

INPUT_DATA = os.path.join(DADOS_DIR, "base_mvp_model_ready.csv")
LGBM_PREDICTIONS_PATH = os.path.join(ARTEFACT_DIR, "predicoes_teste_lightgbm.csv")

MODEL_PATH = os.path.join(ARTEFACT_DIR, "model_isolation_forest.joblib")
SCALER_PATH = os.path.join(ARTEFACT_DIR, "scaler_isolation_forest.joblib")
METRICS_PATH = os.path.join(ARTEFACT_DIR, "metricas_isolation_forest.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTEFACT_DIR, "feature_importance_isolation_forest.csv")
PREDICTIONS_TEST_PATH = os.path.join(ARTEFACT_DIR, "predicoes_teste_isolation_forest.csv")
ERROR_ANALYSIS_PATH = os.path.join(ARTEFACT_DIR, "error_analysis_isolation_forest.csv")
SCORE_DISTRIBUTION_PATH = os.path.join(ARTEFACT_DIR, "score_distribution_isolation_forest.csv")
COMPLEMENTARITY_PATH = os.path.join(ARTEFACT_DIR, "complementarity_analysis.csv")
SHAP_PATH = os.path.join(ARTEFACT_DIR, "shap_summary_isolation_forest.csv")
IF_CONFIG_PATH = os.path.join(ARTEFACT_DIR, "isolation_forest_config.json")

os.makedirs(ARTEFACT_DIR, exist_ok=True)
RANDOM_STATE = 42


# =========================================================
# FEATURES — focadas no perfil de "primeira transação"
# =========================================================
IF_FEATURES = [
    # Valor absoluto (sem ratios — não há histórico)
    "vl_pix",
    "log_vl_pix",

    # Perfil do cliente
    "nr_idade",
    "qt_tempo_relacionamento_mes",

    # Sinais externos
    "topaz_score_filled",
    "rule_score_raw",

    # Latência (padrão de device/rede)
    "latencia_rede_ms_final",
    "latencia_host_ratio",
    "tempo_processamento_host_ms",

    # Hora (fraudes tendem a horários específicos)
    "hour",
]


# =========================================================
# HELPERS
# =========================================================
def anomaly_score_percentile(raw_scores, ref_scores=None):
    """
    Converte decision_function para percentil [0,1] baseado na
    distribuição de referência (treino).
    Quanto mais alto = mais anômalo.
    """
    inverted = -raw_scores
    if ref_scores is None:
        ref = inverted
    else:
        ref = -ref_scores
    # Percentile rank
    scores = np.array([np.mean(ref <= v) for v in inverted])
    return scores


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


def evaluate_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else None,
        "average_precision": float(average_precision_score(y_true, y_score)) if has_both else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision_at_1pct": precision_at_k(y_true, y_score, 0.01),
        "recall_at_1pct": recall_at_k(y_true, y_score, 0.01),
        "precision_at_5pct": precision_at_k(y_true, y_score, 0.05),
        "recall_at_5pct": recall_at_k(y_true, y_score, 0.05),
        "recall_at_top_50": recall_at_top_n(y_true, y_score, 50),
        "recall_at_top_100": recall_at_top_n(y_true, y_score, 100),
    }


def find_best_threshold_by_f1(y_true, y_score):
    best_threshold, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return float(best_threshold), float(best_f1)


def print_split_summary(name, df, label_col="is_fraud"):
    n = len(df)
    n_fraud = int(df[label_col].sum())
    n_first = int(df["is_first_tx_trimestre"].sum()) if "is_first_tx_trimestre" in df.columns else "?"
    print(f"  {name}: {n} rows | {n_fraud} fraudes | {n_first} primeiras tx")


def compute_feature_importance_permutation(model, scaler, X, y, feature_names,
                                           ref_raw, n_repeats=5):
    X_scaled = scaler.transform(X)
    base_raw = model.decision_function(X_scaled)
    base_scores = anomaly_score_percentile(base_raw, ref_raw)
    base_ap = average_precision_score(y, base_scores)

    importances = []
    rng = np.random.RandomState(RANDOM_STATE)
    for i, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_scaled.copy()
            rng.shuffle(X_perm[:, i])
            perm_raw = model.decision_function(X_perm)
            perm_scores = anomaly_score_percentile(perm_raw, ref_raw)
            perm_ap = average_precision_score(y, perm_scores)
            drops.append(base_ap - perm_ap)
        importances.append({
            "feature": feat,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })

    return pd.DataFrame(importances).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)


# =========================================================
# LOAD DATA
# =========================================================
print("=" * 70)
print("TREINO Isolation Forest v3 — Especializado em primeiras tx")
print("=" * 70)

print("\nLendo base model-ready...")
df = pd.read_csv(INPUT_DATA)
df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
df = df.sort_values("event_datetime").reset_index(drop=True)
df = df[df["is_fraud"].notna()].copy()
df["is_fraud"] = df["is_fraud"].astype(int)

# Garantir que hour existe
if "hour" not in df.columns and "event_datetime" in df.columns:
    df["hour"] = df["event_datetime"].dt.hour

print(f"Shape: {df.shape}")
print(f"Fraudes: {df['is_fraud'].sum()} | Normais: {(df['is_fraud'] == 0).sum()}")

# Verificar features
available = [f for f in IF_FEATURES if f in df.columns]
missing_feats = [f for f in IF_FEATURES if f not in df.columns]
if missing_feats:
    print(f"  ⚠ Features não encontradas: {missing_feats}")
IF_FEATURES = available
print(f"\n  Features: {len(IF_FEATURES)}: {IF_FEATURES}")

# Stats de primeiras tx
first_tx_mask = df["is_first_tx_trimestre"] == 1
print(f"\n  Primeiras tx no trimestre: {first_tx_mask.sum()} ({first_tx_mask.mean()*100:.1f}%)")
print(f"  Fraudes em primeiras tx:   {df.loc[first_tx_mask, 'is_fraud'].sum()}")
print(f"  Fraudes em tx regulares:   {df.loc[~first_tx_mask, 'is_fraud'].sum()}")


# =========================================================
# SPLIT TEMPORAL
# =========================================================
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy().reset_index(drop=True)
df_val = df.iloc[train_end:val_end].copy().reset_index(drop=True)
df_test = df.iloc[val_end:].copy().reset_index(drop=True)

print("\nSplit temporal:")
print_split_summary("Treino", df_train)
print_split_summary("Validação", df_val)
print_split_summary("Teste", df_test)


# =========================================================
# PREPARE DATA — treinar apenas com primeiras tx normais
# =========================================================
print("\nPreparando dados...")

# Treino: apenas primeiras tx normais
train_first_normal = df_train[
    (df_train["is_first_tx_trimestre"] == 1) & (df_train["is_fraud"] == 0)
].copy()
print(f"  Treino (primeiras tx normais): {len(train_first_normal)} rows")

X_train_first = train_first_normal[IF_FEATURES].copy()

# Validação e teste: TODAS as transações (para avaliar no contexto geral)
X_val = df_val[IF_FEATURES].copy()
X_test = df_test[IF_FEATURES].copy()
y_val = df_val["is_fraud"].values.copy()
y_test = df_test["is_fraud"].values.copy()

# Mediana do treino
medians = X_train_first.median()
X_train_first = X_train_first.fillna(medians)
X_val = X_val.fillna(medians)
X_test = X_test.fillna(medians)

# Escalar
print("  Escalando (RobustScaler)...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_first)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Referência para percentil: decision_function do treino
# (será calculado depois do fit)


# =========================================================
# HYPERPARAMETER SEARCH
# =========================================================
print("\nBuscando hiperparâmetros...")

param_grid = []
for n_est in [300, 500, 800]:
    for max_samp in [0.7, 0.8, "auto"]:
        for max_feat in [0.7, 0.8, 1.0]:
            for contam in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
                param_grid.append({
                    "n_estimators": n_est,
                    "max_samples": max_samp,
                    "max_features": max_feat,
                    "contamination": contam,
                })

best_params = None
best_val_ap = -1
best_model = None

for params in param_grid:
    iforest = IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        contamination=params["contamination"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train_scaled)

    # Referência para percentil
    ref_raw = iforest.decision_function(X_train_scaled)
    raw_val = iforest.decision_function(X_val_scaled)
    scores_val = anomaly_score_percentile(raw_val, ref_raw)

    ap = average_precision_score(y_val, scores_val)

    if ap > best_val_ap:
        best_val_ap = ap
        best_params = params
        best_model = iforest

print(f"\n  Melhor config: {best_params}")
print(f"  Melhor AP (validação): {best_val_ap:.4f}")

model = best_model

# Referência final para percentil
ref_raw_train = model.decision_function(X_train_scaled)


# =========================================================
# SCORES — todos os splits
# =========================================================
raw_val = model.decision_function(X_val_scaled)
anomaly_scores_val = anomaly_score_percentile(raw_val, ref_raw_train)

raw_test = model.decision_function(X_test_scaled)
anomaly_scores_test = anomaly_score_percentile(raw_test, ref_raw_train)

raw_train_all = model.decision_function(
    scaler.transform(df_train[IF_FEATURES].fillna(medians))
)
anomaly_scores_train = anomaly_score_percentile(raw_train_all, ref_raw_train)
y_train_all = df_train["is_fraud"].values


# =========================================================
# THRESHOLD SEARCH
# =========================================================
best_threshold, best_val_f1 = find_best_threshold_by_f1(y_val, anomaly_scores_val)
print(f"\n  Melhor threshold (F1 val): {best_threshold:.4f} → F1={best_val_f1:.4f}")


# =========================================================
# EVALUATION
# =========================================================
print("\nAvaliando no dataset COMPLETO...")

metrics = {}
for split, y_true, scores in [
    ("train", y_train_all, anomaly_scores_train),
    ("val", y_val, anomaly_scores_val),
    ("test", y_test, anomaly_scores_test),
]:
    metrics[f"{split}_threshold_0_5"] = evaluate_metrics(y_true, scores, 0.5)
    metrics[f"{split}_best_threshold"] = evaluate_metrics(y_true, scores, best_threshold)

    m = metrics[f"{split}_threshold_0_5"]
    print(f"\n  {split} @ 0.5:")
    if m['roc_auc']:
        print(f"    ROC-AUC: {m['roc_auc']:.4f}")
    if m['average_precision']:
        print(f"    AP:      {m['average_precision']:.4f}")
    print(f"    F1:      {m['f1']:.4f}")
    print(f"    P/R:     {m['precision']:.4f} / {m['recall']:.4f}")
    print(f"    R@Top50: {m['recall_at_top_50']:.4f}")
    print(f"    R@Top100:{m['recall_at_top_100']:.4f}")


# =========================================================
# EVALUATE ONLY ON FIRST-TX SUBSET (onde o IF deve brilhar)
# =========================================================
print("\n" + "=" * 70)
print("Avaliando APENAS em primeiras tx do trimestre...")
print("=" * 70)

first_mask_test = df_test["is_first_tx_trimestre"].values == 1
if first_mask_test.sum() > 0 and y_test[first_mask_test].sum() > 0:
    m_first = evaluate_metrics(
        y_test[first_mask_test],
        anomaly_scores_test[first_mask_test],
        0.5
    )
    bt_first, bf1_first = find_best_threshold_by_f1(
        y_test[first_mask_test],
        anomaly_scores_test[first_mask_test]
    )
    m_first_best = evaluate_metrics(
        y_test[first_mask_test],
        anomaly_scores_test[first_mask_test],
        bt_first,
    )
    metrics["test_first_tx_0_5"] = m_first
    metrics["test_first_tx_best"] = m_first_best

    n_first = first_mask_test.sum()
    n_fraud_first = y_test[first_mask_test].sum()
    print(f"  Subset: {n_first} tx | {n_fraud_first} fraudes")
    print(f"\n  @ threshold 0.5:")
    print(f"    ROC-AUC: {m_first['roc_auc']:.4f}" if m_first['roc_auc'] else "")
    print(f"    AP:      {m_first['average_precision']:.4f}" if m_first['average_precision'] else "")
    print(f"    F1:      {m_first['f1']:.4f}")
    print(f"    P/R:     {m_first['precision']:.4f} / {m_first['recall']:.4f}")

    print(f"\n  @ best threshold ({bt_first:.4f}):")
    print(f"    F1:      {m_first_best['f1']:.4f}")
    print(f"    P/R:     {m_first_best['precision']:.4f} / {m_first_best['recall']:.4f}")
else:
    print("  ⚠ Insuficiente para avaliar")


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
print("\nCalculando feature importance...")
np.random.seed(RANDOM_STATE)

feat_imp = compute_feature_importance_permutation(
    model, scaler,
    X_test.values, y_test,
    IF_FEATURES,
    ref_raw=ref_raw_train,
    n_repeats=5,
)
feat_imp.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
print(feat_imp.to_string(index=False))


# =========================================================
# SHAP
# =========================================================
print("\nCalculando SHAP...")
shap_importance = None
try:
    import shap
    n_shap = min(2000, len(X_test))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled[:n_shap])

    shap_importance = pd.DataFrame({
        "feature": IF_FEATURES,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_importance.to_csv(SHAP_PATH, index=False)
    print(shap_importance.to_string(index=False))
except ImportError:
    print("  ⚠ shap não instalado")
except Exception as e:
    print(f"  ⚠ Erro SHAP: {e}")


# =========================================================
# PREDICTIONS TEST
# =========================================================
pred_test = pd.DataFrame({
    "transaction_id": df_test["transaction_id"].values,
    "customer_id": df_test["customer_id"].values,
    "event_datetime": df_test["event_datetime"].values,
    "is_fraud": y_test,
    "is_first_tx_trimestre": df_test["is_first_tx_trimestre"].values,
    "anomaly_score": anomaly_scores_test,
    "anomaly_raw": raw_test,
    "pred_0_5": (anomaly_scores_test >= 0.5).astype(int),
    "pred_best": (anomaly_scores_test >= best_threshold).astype(int),
})
pred_test.to_csv(PREDICTIONS_TEST_PATH, index=False)


# =========================================================
# ERROR ANALYSIS
# =========================================================
print("\nAnálise de erros...")

fn_mask = (pred_test["is_fraud"] == 1) & (pred_test["pred_best"] == 0)
fp_mask = (pred_test["is_fraud"] == 0) & (pred_test["pred_best"] == 1)

errors = pred_test[fn_mask | fp_mask].copy()
errors["error_type"] = np.where(errors["is_fraud"] == 1, "FALSE_NEGATIVE", "FALSE_POSITIVE")

for feat in IF_FEATURES:
    if feat in X_test.columns:
        errors[feat] = X_test.loc[errors.index, feat].values

errors.to_csv(ERROR_ANALYSIS_PATH, index=False)
print(f"  {len(errors)} erros (best threshold)")
print(f"    FN: {fn_mask.sum()}")
print(f"    FP: {fp_mask.sum()}")


# =========================================================
# SCORE DISTRIBUTION
# =========================================================
print("\nDistribuição de scores por classe...")
score_dist = pd.DataFrame({
    "class": y_test,
    "score": anomaly_scores_test,
    "is_first_tx": df_test["is_first_tx_trimestre"].values,
})

print("\n  GERAL:")
print(score_dist.groupby("class")["score"].describe().to_string())

print("\n  APENAS primeiras tx:")
first_only = score_dist[score_dist["is_first_tx"] == 1]
if len(first_only) > 0:
    print(first_only.groupby("class")["score"].describe().to_string())

score_dist.to_csv(SCORE_DISTRIBUTION_PATH, index=False)


# =========================================================
# COMPLEMENTARITY — com ensemble condicional
# =========================================================
print("\nAnálise de complementaridade...")

complementarity = None
try:
    lgbm_preds = pd.read_csv(LGBM_PREDICTIONS_PATH)

    comp = pred_test[[
        "transaction_id", "is_fraud", "anomaly_score", "is_first_tx_trimestre"
    ]].merge(
        lgbm_preds[["transaction_id", "score_fraude"]].rename(
            columns={"score_fraude": "lgbm_score"}
        ),
        on="transaction_id",
        how="inner",
    )

    # Ensemble CONDICIONAL: IF só contribui para primeiras tx com LGBM na zona cinzenta
    lgbm_s = comp["lgbm_score"].values
    if_s = comp["anomaly_score"].values
    is_first = comp["is_first_tx_trimestre"].values

    # Máscara: primeira tx E lgbm incerto (0.05 < score < 0.6)
    use_if = (is_first == 1) & (lgbm_s > 0.05) & (lgbm_s < 0.6)

    # Peso do IF: 0.30 quando ativo, 0 quando não
    if_weight = np.where(use_if, 0.30, 0.0)
    lgbm_weight = 1.0 - if_weight

    comp["ensemble_conditional"] = lgbm_weight * lgbm_s + if_weight * if_s
    comp["if_active"] = use_if.astype(int)
    comp["if_weight"] = if_weight

    # Ensemble boost condicional
    if_bonus = np.where(use_if & (if_s > 0.8), 0.20, 0.0)
    comp["ensemble_boost_cond"] = np.clip(lgbm_s + if_bonus, 0, 1)

    # Métricas
    comp_metrics = {}
    score_cols = ["lgbm_score", "anomaly_score", "ensemble_conditional",
                  "ensemble_boost_cond"]

    for score_col in score_cols:
        y_true = comp["is_fraud"].values
        y_score = comp[score_col].values
        bt, bf1 = find_best_threshold_by_f1(y_true, y_score)

        comp_metrics[score_col] = {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "recall_at_1pct": recall_at_k(y_true, y_score, 0.01),
            "recall_at_5pct": recall_at_k(y_true, y_score, 0.05),
            "recall_at_top_50": recall_at_top_n(y_true, y_score, 50),
            "recall_at_top_100": recall_at_top_n(y_true, y_score, 100),
            "best_threshold": bt,
            "best_f1": bf1,
        }

    print("\n  Comparação:")
    print(f"  {'Modelo':<24s} {'ROC-AUC':>8s} {'AP':>8s} {'R@1%':>8s} "
          f"{'R@Top100':>8s} {'Best F1':>8s}")
    print("  " + "-" * 72)
    for name, m in comp_metrics.items():
        print(f"  {name:<24s} {m['roc_auc']:8.4f} {m['average_precision']:8.4f} "
              f"{m['recall_at_1pct']:8.4f} {m['recall_at_top_100']:8.4f} "
              f"{m['best_f1']:8.4f}")

    # FN do LGBM
    lgbm_fn = comp[(comp["is_fraud"] == 1) & (comp["lgbm_score"] < 0.5)]
    if len(lgbm_fn) > 0:
        print(f"\n  Fraudes que LightGBM perde (score < 0.5): {len(lgbm_fn)}")
        for _, row in lgbm_fn.iterrows():
            is_f = "1ªTX" if row["is_first_tx_trimestre"] == 1 else "REG"
            active = "IF ATIVO" if row["if_active"] == 1 else "IF OFF"
            print(f"    [{is_f}] [{active}] lgbm={row['lgbm_score']:.4f} "
                  f"if={row['anomaly_score']:.4f} "
                  f"cond={row['ensemble_conditional']:.4f} "
                  f"boost={row['ensemble_boost_cond']:.4f}")

    # Quantos normais o IF estraga?
    lgbm_correct = comp[(comp["is_fraud"] == 0) & (comp["lgbm_score"] < 0.1)]
    if len(lgbm_correct) > 0:
        n_damaged_cond = (lgbm_correct["ensemble_conditional"] > 0.3).sum()
        n_damaged_boost = (lgbm_correct["ensemble_boost_cond"] > 0.3).sum()
        print(f"\n  Normais com LGBM<0.1 ({len(lgbm_correct)} tx):")
        print(f"    ensemble_conditional prejudica: {n_damaged_cond}")
        print(f"    ensemble_boost_cond prejudica:  {n_damaged_boost}")

    comp.to_csv(COMPLEMENTARITY_PATH, index=False)
    complementarity = comp_metrics

except FileNotFoundError:
    print("  ⚠ Predições LightGBM não encontradas")
except Exception as e:
    print(f"  ⚠ Erro: {e}")
    import traceback
    traceback.print_exc()


# =========================================================
# SAVE ALL
# =========================================================
metadata = {
    "model_type": "IsolationForest",
    "version": "v3_first_tx_specialist",
    "strategy": "Trained only on first-tx-of-trimester normals, percentile scoring",
    "n_features": len(IF_FEATURES),
    "features": IF_FEATURES,
    "best_params": {k: str(v) if not isinstance(v, (int, float)) else v
                    for k, v in best_params.items()},
    "best_val_ap": float(best_val_ap),
    "best_threshold_val_f1": float(best_threshold),
    "best_val_f1": float(best_val_f1),
    "n_train_first_normal": int(len(train_first_normal)),
    "n_val": int(len(df_val)),
    "n_test": int(len(df_test)),
    "metrics": metrics,
    "complementarity": complementarity,
}

if_config = {
    "features": IF_FEATURES,
    "medians": {k: float(v) for k, v in medians.to_dict().items()},
    "best_threshold": float(best_threshold),
    "best_params": {k: str(v) if not isinstance(v, (int, float)) else v
                    for k, v in best_params.items()},
    "ensemble_strategy": "conditional",
    "ensemble_params": {
        "condition": "is_first_tx_trimestre=1 AND lgbm_score in (0.05, 0.6)",
        "if_weight_when_active": 0.30,
        "boost_threshold": 0.80,
        "boost_amount": 0.20,
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

with open(IF_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(if_config, f, ensure_ascii=False, indent=2)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# Salvar referência do treino para percentil em produção
np.save(os.path.join(ARTEFACT_DIR, "if_ref_raw_train.npy"), ref_raw_train)


# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("RESULTADOS — Isolation Forest v3 (first-tx specialist)")
print("=" * 70)

print(f"\nConfig: {best_params}")
print(f"Features: {len(IF_FEATURES)}")
print(f"Treino (primeiras tx normais): {len(train_first_normal)}")

print(f"\nTeste completo @ 0.5:")
for k, v in metrics["test_threshold_0_5"].items():
    print(f"  {k}: {v}")

if "test_first_tx_0_5" in metrics:
    print(f"\nTeste APENAS primeiras tx @ 0.5:")
    for k, v in metrics["test_first_tx_0_5"].items():
        print(f"  {k}: {v}")

print(f"\nArtefatos salvos:")
for path in [MODEL_PATH, SCALER_PATH, METRICS_PATH, IF_CONFIG_PATH,
             FEATURE_IMPORTANCE_PATH, PREDICTIONS_TEST_PATH,
             ERROR_ANALYSIS_PATH, SCORE_DISTRIBUTION_PATH,
             COMPLEMENTARITY_PATH]:
    print(f"  {path}")
if shap_importance is not None:
    print(f"  {SHAP_PATH}")
print("=" * 70)
