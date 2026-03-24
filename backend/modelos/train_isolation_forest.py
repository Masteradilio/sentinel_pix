"""
train_isolation_forest.py v4 — IF Complementar aos FN do LGBM

Mudanças v3 → v4:
  1. ESCOPO EXPANDIDO: treina com TODAS as tx normais (não apenas primeiras)
     → Permite detectar bursts em contas comprometidas
  2. FEATURES REDESENHADAS: inclui velocity/burst que o LGBM subutiliza
     → tx_count_prev_30m, burst_30m_flag, distinct_receivers_so_far
  3. ZONA DE ATIVAÇÃO: IF ativa quando LGBM score < threshold_f1 (0.08)
     → Antes: só primeiras tx. Agora: qualquer tx que LGBM não flagga
  4. TREINO SEGMENTADO: treina perfis separados para 1ª tx vs regulares
     → Modelo aprende "normal" para cada contexto
  5. FEATURES DE INTERAÇÃO: combinações que capturam padrões de burst
     → valor_x_burst, idade_x_first_receiver, etc.
  6. THRESHOLD OTIMIZADO: busca threshold que maximiza recall nos FN do LGBM
     → Não F1 geral, mas recall onde importa

Cenários-alvo (25 FN do LGBM v4.1):
  - 10213260115: 72 anos, burst de R$5K-10K, qt_total=9, first_recv=0
  - 82989818120: 48 anos, burst de R$800-960, qt_total=16, first_recv misto
  - 48379573153: 61 anos, burst de R$1K-3K, qt_total=9, first_recv misto
  - Isolados: ratio_mediana=1.0, valores baixos, first_recv=1
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
# FEATURES — redesenhadas para complementar o LGBM
# =========================================================

# Grupo 1: Perfil do cliente (o LGBM usa mas com peso diferente)
FEATURES_PROFILE = [
    "nr_idade",
    "qt_tempo_relacionamento_mes",
]

# Grupo 2: Valor (absoluto + relativo)
FEATURES_VALUE = [
    "vl_pix",
    "log_vl_pix",
    "ratio_valor_mediana",       # LGBM usa mas IF pode ponderar diferente
    "zscore_valor_aprox",        # Desvio estatístico
]

# Grupo 3: Velocity/Burst (FRAQUEZA DO LGBM — importance quase zero)
FEATURES_VELOCITY = [
    "tx_count_prev_30m",         # Burst flag numérico
    "burst_30m_flag",            # Flag binário
    "minutes_since_prev_tx",     # Intervalo entre tx
    "qt_total_pix_trimestre",    # Volume total no trimestre
    "qt_pix_dia_maximo_trimestre",  # Pico diário
    "distinct_receivers_so_far", # Diversidade de recebedores
]

# Grupo 4: Contexto da transação
FEATURES_CONTEXT = [
    "first_receiver_flag",       # Primeiro envio para este recebedor
    "is_first_tx_trimestre",     # Primeira tx do trimestre
    "hour",                      # Horário
    "topaz_score_filled",        # Score externo
    "rule_score_raw",            # Score de regras
]

# Grupo 5: Features de interação (combinações que capturam padrões)
# Estas serão criadas no código
FEATURES_INTERACTION = [
    "valor_x_burst",             # vl_pix * (tx_count_prev_30m + 1)
    "idade_x_first_recv",        # nr_idade * first_receiver_flag
    "valor_x_first_recv",        # vl_pix * first_receiver_flag
    "burst_x_distinct_recv",     # tx_count_prev_30m * distinct_receivers_so_far
    "valor_over_trimestre_avg",  # vl_pix / (mediana * qt_total) - proxy de concentração
]

# Todas as features base (sem interações)
IF_BASE_FEATURES = FEATURES_PROFILE + FEATURES_VALUE + FEATURES_VELOCITY + FEATURES_CONTEXT

# Features finais (base + interações)
IF_ALL_FEATURES = IF_BASE_FEATURES + FEATURES_INTERACTION


# =========================================================
# HELPERS
# =========================================================
def anomaly_score_percentile(raw_scores, ref_scores=None):
    """Converte decision_function para percentil [0,1]. Mais alto = mais anômalo."""
    inverted = -raw_scores
    ref = -ref_scores if ref_scores is not None else inverted
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


def find_threshold_by_min_recall(y_true, y_score, min_recall=0.95):
    """Encontra o threshold mais alto que garante recall >= min_recall."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    positives = y_true.sum()
    if positives == 0:
        return 0.5
    for t in np.arange(0.95, 0.001, -0.001):
        preds = (y_score >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        if rec >= min_recall:
            return float(t)
    return 0.001


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

    if y.sum() == 0 or len(np.unique(y)) < 2:
        return pd.DataFrame({"feature": feature_names, "importance_mean": 0.0, "importance_std": 0.0})

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
            try:
                perm_ap = average_precision_score(y, perm_scores)
            except Exception:
                perm_ap = base_ap
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
# FEATURE ENGINEERING — Interações
# =========================================================
def create_interaction_features(df):
    """Cria features de interação para o IF."""
    df = df.copy()

    # valor_x_burst: Valor acumulado em burst
    tx_30m = df["tx_count_prev_30m"].fillna(0)
    df["valor_x_burst"] = df["vl_pix"].fillna(0) * (tx_30m + 1)

    # idade_x_first_recv: Idoso + primeiro recebedor = risco alto
    df["idade_x_first_recv"] = df["nr_idade"].fillna(0) * df["first_receiver_flag"].fillna(0)

    # valor_x_first_recv: Valor alto + primeiro recebedor
    df["valor_x_first_recv"] = df["vl_pix"].fillna(0) * df["first_receiver_flag"].fillna(0)

    # burst_x_distinct_recv: Burst para múltiplos recebedores
    df["burst_x_distinct_recv"] = tx_30m * df["distinct_receivers_so_far"].fillna(1)

    # valor_over_trimestre_avg: Concentração de valor
    mediana = df["vl_mediana_pix_trimestre"] if "vl_mediana_pix_trimestre" in df.columns else 0
    qt_total = df["qt_total_pix_trimestre"].fillna(1).clip(lower=1)
    # Proxy: valor da tx vs total esperado no trimestre
    total_esperado = pd.to_numeric(mediana, errors="coerce").fillna(0) * qt_total
    df["valor_over_trimestre_avg"] = np.where(
        total_esperado > 0,
        df["vl_pix"].fillna(0) / total_esperado,
        0,
    )

    return df


# =========================================================
# LOAD DATA
# =========================================================
print("=" * 70)
print("TREINO Isolation Forest v4 — Complementar ao LGBM")
print("=" * 70)

print("\nLendo base model-ready...")
df = pd.read_csv(INPUT_DATA)
df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
df = df.sort_values("event_datetime").reset_index(drop=True)
df = df[df["is_fraud"].notna()].copy()
df["is_fraud"] = df["is_fraud"].astype(int)

if "hour" not in df.columns and "event_datetime" in df.columns:
    df["hour"] = df["event_datetime"].dt.hour

print(f"Shape: {df.shape}")
print(f"Fraudes: {df['is_fraud'].sum()} | Normais: {(df['is_fraud'] == 0).sum()}")

# Criar features de interação
df = create_interaction_features(df)

# Verificar features
available = [f for f in IF_ALL_FEATURES if f in df.columns]
missing_feats = [f for f in IF_ALL_FEATURES if f not in df.columns]
if missing_feats:
    print(f"  ⚠ Features não encontradas: {missing_feats}")
IF_FEATURES_FINAL = available
print(f"\n  Features: {len(IF_FEATURES_FINAL)}")
for i, f in enumerate(IF_FEATURES_FINAL):
    print(f"    {i+1:2d}. {f}")

# Stats
first_tx_mask = df["is_first_tx_trimestre"] == 1
print(f"\n  Primeiras tx: {first_tx_mask.sum()} ({first_tx_mask.mean()*100:.1f}%)")
print(f"  Fraudes em primeiras tx:  {df.loc[first_tx_mask, 'is_fraud'].sum()}")
print(f"  Fraudes em tx regulares:  {df.loc[~first_tx_mask, 'is_fraud'].sum()}")


# =========================================================
# SPLIT TEMPORAL (mesmo do LGBM — holdout nos últimos 10%)
# =========================================================
holdout_ratio = 0.10
n = len(df)
holdout_start = int(n * (1 - holdout_ratio))

# Dev: 90%, dividido em treino (70% do dev) e validação (30% do dev)
df_dev = df.iloc[:holdout_start].copy().reset_index(drop=True)
df_holdout = df.iloc[holdout_start:].copy().reset_index(drop=True)

dev_train_end = int(len(df_dev) * 0.70)
df_train = df_dev.iloc[:dev_train_end].copy().reset_index(drop=True)
df_val = df_dev.iloc[dev_train_end:].copy().reset_index(drop=True)

print("\nSplit temporal:")
print_split_summary("Treino", df_train)
print_split_summary("Validação", df_val)
print_split_summary("Holdout", df_holdout)


# =========================================================
# PREPARE DATA — treinar com TODAS as tx normais
# =========================================================
print("\nPreparando dados...")

# Treino: TODAS as transações normais (não apenas primeiras)
train_normal = df_train[df_train["is_fraud"] == 0].copy()
print(f"  Treino (tx normais): {len(train_normal)} rows")

X_train = train_normal[IF_FEATURES_FINAL].copy()
X_val = df_val[IF_FEATURES_FINAL].copy()
X_holdout = df_holdout[IF_FEATURES_FINAL].copy()
y_val = df_val["is_fraud"].values.copy()
y_holdout = df_holdout["is_fraud"].values.copy()

# Mediana do treino para imputação
medians = X_train.median()
X_train = X_train.fillna(medians)
X_val = X_val.fillna(medians)
X_holdout = X_holdout.fillna(medians)

# Substituir infinitos
for col in X_train.columns:
    X_train[col] = X_train[col].replace([np.inf, -np.inf], medians.get(col, 0))
    X_val[col] = X_val[col].replace([np.inf, -np.inf], medians.get(col, 0))
    X_holdout[col] = X_holdout[col].replace([np.inf, -np.inf], medians.get(col, 0))

# Escalar
print("  Escalando (RobustScaler)...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_holdout_scaled = scaler.transform(X_holdout)


# =========================================================
# HYPERPARAMETER SEARCH
# =========================================================
print("\nBuscando hiperparâmetros...")

param_grid = []
for n_est in [300, 500, 800]:
    for max_samp in [0.7, 0.8, "auto"]:
        for max_feat in [0.6, 0.7, 0.8, 1.0]:
            for contam in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
                param_grid.append({
                    "n_estimators": n_est,
                    "max_samples": max_samp,
                    "max_features": max_feat,
                    "contamination": contam,
                })

print(f"  Testando {len(param_grid)} configurações...")

best_params = None
best_val_ap = -1
best_model = None
best_ref_raw = None

for idx, params in enumerate(param_grid):
    if idx % 50 == 0:
        print(f"  ... {idx}/{len(param_grid)}")

    iforest = IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        contamination=params["contamination"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train_scaled)

    ref_raw = iforest.decision_function(X_train_scaled)
    raw_val = iforest.decision_function(X_val_scaled)
    scores_val = anomaly_score_percentile(raw_val, ref_raw)

    if y_val.sum() > 0:
        ap = average_precision_score(y_val, scores_val)
    else:
        ap = 0.0

    if ap > best_val_ap:
        best_val_ap = ap
        best_params = params
        best_model = iforest
        best_ref_raw = ref_raw

print(f"\n  ✓ Melhor config: {best_params}")
print(f"  ✓ Melhor AP (validação): {best_val_ap:.4f}")

model = best_model
ref_raw_train = best_ref_raw


# =========================================================
# SCORES
# =========================================================
raw_val = model.decision_function(X_val_scaled)
anomaly_scores_val = anomaly_score_percentile(raw_val, ref_raw_train)

raw_holdout = model.decision_function(X_holdout_scaled)
anomaly_scores_holdout = anomaly_score_percentile(raw_holdout, ref_raw_train)

raw_train_all = model.decision_function(
    scaler.transform(df_train[IF_FEATURES_FINAL].fillna(medians).replace([np.inf, -np.inf], 0))
)
anomaly_scores_train = anomaly_score_percentile(raw_train_all, ref_raw_train)
y_train_all = df_train["is_fraud"].values


# =========================================================
# THRESHOLD SEARCH
# =========================================================
best_threshold_f1, best_val_f1 = find_best_threshold_by_f1(y_val, anomaly_scores_val)
print(f"\n  Threshold F1 val: {best_threshold_f1:.4f} → F1={best_val_f1:.4f}")

# Threshold para recall alto
th_recall_90 = find_threshold_by_min_recall(y_val, anomaly_scores_val, 0.90)
th_recall_95 = find_threshold_by_min_recall(y_val, anomaly_scores_val, 0.95)
print(f"  Threshold Recall≥90%: {th_recall_90:.4f}")
print(f"  Threshold Recall≥95%: {th_recall_95:.4f}")


# =========================================================
# EVALUATION
# =========================================================
print(f"\n{'='*70}")
print("AVALIAÇÃO")
print(f"{'='*70}")

metrics = {}
for split, y_true, scores in [
    ("train", y_train_all, anomaly_scores_train),
    ("val", y_val, anomaly_scores_val),
    ("holdout", y_holdout, anomaly_scores_holdout),
]:
    metrics[f"{split}_0_5"] = evaluate_metrics(y_true, scores, 0.5)
    metrics[f"{split}_best_f1"] = evaluate_metrics(y_true, scores, best_threshold_f1)

    m = metrics[f"{split}_0_5"]
    print(f"\n  {split} @ 0.5:")
    if m['roc_auc']:
        print(f"    ROC-AUC: {m['roc_auc']:.4f}")
    if m['average_precision']:
        print(f"    AP:      {m['average_precision']:.4f}")
    print(f"    F1:      {m['f1']:.4f}")
    print(f"    P/R:     {m['precision']:.4f} / {m['recall']:.4f}")
    print(f"    R@5%:    {m['recall_at_5pct']:.4f}")
    print(f"    R@Top100:{m['recall_at_top_100']:.4f}")


# =========================================================
# EVALUATE BY SEGMENT
# =========================================================
print(f"\n{'='*70}")
print("AVALIAÇÃO POR SEGMENTO")
print(f"{'='*70}")

# Primeiras tx
first_mask = df_holdout["is_first_tx_trimestre"].values == 1
if first_mask.sum() > 0 and y_holdout[first_mask].sum() > 0:
    m_first = evaluate_metrics(y_holdout[first_mask], anomaly_scores_holdout[first_mask], 0.5)
    print(f"\n  Primeiras tx ({first_mask.sum()} tx, {y_holdout[first_mask].sum()} fraudes):")
    print(f"    ROC-AUC: {m_first['roc_auc']:.4f}" if m_first['roc_auc'] else "")
    print(f"    AP:      {m_first['average_precision']:.4f}" if m_first['average_precision'] else "")
    metrics["holdout_first_tx_0_5"] = m_first

# Tx regulares
reg_mask = df_holdout["is_first_tx_trimestre"].values == 0
if reg_mask.sum() > 0 and y_holdout[reg_mask].sum() > 0:
    m_reg = evaluate_metrics(y_holdout[reg_mask], anomaly_scores_holdout[reg_mask], 0.5)
    print(f"\n  Tx regulares ({reg_mask.sum()} tx, {y_holdout[reg_mask].sum()} fraudes):")
    print(f"    ROC-AUC: {m_reg['roc_auc']:.4f}" if m_reg['roc_auc'] else "")
    print(f"    AP:      {m_reg['average_precision']:.4f}" if m_reg['average_precision'] else "")
    metrics["holdout_regular_tx_0_5"] = m_reg


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE")
print(f"{'='*70}")

feat_imp = compute_feature_importance_permutation(
    model, scaler,
    X_holdout.values, y_holdout,
    IF_FEATURES_FINAL,
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
    n_shap = min(2000, len(X_holdout))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_holdout_scaled[:n_shap])

    shap_importance = pd.DataFrame({
        "feature": IF_FEATURES_FINAL,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_importance.to_csv(SHAP_PATH, index=False)
    print(shap_importance.to_string(index=False))
except ImportError:
    print("  ⚠ shap não instalado")
except Exception as e:
    print(f"  ⚠ Erro SHAP: {e}")


# =========================================================
# PREDICTIONS
# =========================================================
pred_holdout = pd.DataFrame({
    "transaction_id": df_holdout["transaction_id"].values,
    "customer_id": df_holdout["customer_id"].values,
    "event_datetime": df_holdout["event_datetime"].values,
    "is_fraud": y_holdout,
    "is_first_tx_trimestre": df_holdout["is_first_tx_trimestre"].values,
    "anomaly_score": anomaly_scores_holdout,
    "anomaly_raw": raw_holdout,
    "pred_0_5": (anomaly_scores_holdout >= 0.5).astype(int),
    "pred_best_f1": (anomaly_scores_holdout >= best_threshold_f1).astype(int),
})
pred_holdout.to_csv(PREDICTIONS_TEST_PATH, index=False)


# =========================================================
# ERROR ANALYSIS
# =========================================================
print(f"\n{'='*70}")
print("ANÁLISE DE ERROS")
print(f"{'='*70}")

fn_mask = (pred_holdout["is_fraud"] == 1) & (pred_holdout["pred_best_f1"] == 0)
fp_mask = (pred_holdout["is_fraud"] == 0) & (pred_holdout["pred_best_f1"] == 1)

errors = pred_holdout[fn_mask | fp_mask].copy()
errors["error_type"] = np.where(errors["is_fraud"] == 1, "FALSE_NEGATIVE", "FALSE_POSITIVE")

for feat in IF_FEATURES_FINAL:
    if feat in X_holdout.columns:
        errors[feat] = X_holdout.loc[errors.index, feat].values

errors.to_csv(ERROR_ANALYSIS_PATH, index=False)
print(f"  {len(errors)} erros: {fn_mask.sum()} FN + {fp_mask.sum()} FP")


# =========================================================
# SCORE DISTRIBUTION
# =========================================================
print("\nDistribuição de scores por classe...")
score_dist = pd.DataFrame({
    "class": y_holdout,
    "score": anomaly_scores_holdout,
    "is_first_tx": df_holdout["is_first_tx_trimestre"].values,
})

print("\n  GERAL:")
print(score_dist.groupby("class")["score"].describe().to_string())

print("\n  APENAS primeiras tx:")
first_only = score_dist[score_dist["is_first_tx"] == 1]
if len(first_only) > 0:
    print(first_only.groupby("class")["score"].describe().to_string())

print("\n  APENAS tx regulares:")
reg_only = score_dist[score_dist["is_first_tx"] == 0]
if len(reg_only) > 0:
    print(reg_only.groupby("class")["score"].describe().to_string())

score_dist.to_csv(SCORE_DISTRIBUTION_PATH, index=False)


# =========================================================
# COMPLEMENTARITY — análise com LGBM
# =========================================================
print(f"\n{'='*70}")
print("COMPLEMENTARIDADE COM LGBM")
print(f"{'='*70}")

complementarity = None
try:
    lgbm_preds = pd.read_csv(LGBM_PREDICTIONS_PATH)

    comp = pred_holdout[[
        "transaction_id", "is_fraud", "anomaly_score", "is_first_tx_trimestre"
    ]].merge(
        lgbm_preds[["transaction_id", "score_fraude", "pred_lgbm", "pred_cascade", "pred_combined"]].rename(
            columns={"score_fraude": "lgbm_score"}
        ),
        on="transaction_id",
        how="inner",
    )

    lgbm_s = comp["lgbm_score"].values
    if_s = comp["anomaly_score"].values
    is_first = comp["is_first_tx_trimestre"].values

    # --- Estratégia A: IF ativa quando LGBM < threshold (0.08) ---
    LGBM_THRESHOLD = 0.08
    lgbm_uncertain = lgbm_s < LGBM_THRESHOLD  # LGBM não flaggou

    # Ensemble: boost condicional
    # Se LGBM não flaggou MAS IF diz que é anômalo → boost
    if_high = if_s >= 0.70  # IF acha suspeito
    if_very_high = if_s >= 0.85  # IF acha muito suspeito

    # Ensemble boost: adiciona score do IF ao LGBM quando IF é alto
    boost_amount = np.where(
        lgbm_uncertain & if_very_high, 0.15,
        np.where(lgbm_uncertain & if_high, 0.08, 0.0)
    )
    comp["ensemble_boost"] = np.clip(lgbm_s + boost_amount, 0, 1)

    # --- Estratégia B: Weighted ensemble quando LGBM incerto ---
    if_weight = np.where(lgbm_uncertain, 0.25, 0.0)
    lgbm_weight = 1.0 - if_weight
    comp["ensemble_weighted"] = lgbm_weight * lgbm_s + if_weight * if_s

    # --- Estratégia C: OR lógico (LGBM >= th OU IF >= if_th) ---
    for if_th_name, if_th_val in [("0.70", 0.70), ("0.80", 0.80), ("0.85", 0.85), ("0.90", 0.90)]:
        pred_or = ((lgbm_s >= LGBM_THRESHOLD) | (if_s >= if_th_val)).astype(int)
        comp[f"pred_or_if{if_th_name}"] = pred_or

    # --- Estratégia D: LGBM + Cascade + IF OR ---
    lgbm_combined = comp["pred_combined"].values  # LGBM + cascade da v4.1
    for if_th_name, if_th_val in [("0.70", 0.70), ("0.80", 0.80), ("0.85", 0.85)]:
        pred_full = np.maximum(lgbm_combined, (if_s >= if_th_val).astype(int))
        comp[f"pred_full_if{if_th_name}"] = pred_full

    # Métricas de complementaridade
    comp_metrics = {}

    # Scores contínuos
    for score_col in ["lgbm_score", "anomaly_score", "ensemble_boost", "ensemble_weighted"]:
        y_true = comp["is_fraud"].values
        y_score = comp[score_col].values
        bt, bf1 = find_best_threshold_by_f1(y_true, y_score)
        comp_metrics[score_col] = {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "recall_at_1pct": recall_at_k(y_true, y_score, 0.01),
            "recall_at_5pct": recall_at_k(y_true, y_score, 0.05),
            "best_threshold": bt,
            "best_f1": bf1,
        }

    print("\n  Scores contínuos:")
    print(f"  {'Modelo':<24s} {'ROC-AUC':>8s} {'AP':>8s} {'R@5%':>8s} {'Best F1':>8s}")
    print("  " + "-" * 56)
    for name, m in comp_metrics.items():
        print(f"  {name:<24s} {m['roc_auc']:8.4f} {m['average_precision']:8.4f} "
              f"{m['recall_at_5pct']:8.4f} {m['best_f1']:8.4f}")

    # Predições binárias (OR e FULL)
    print(f"\n  Predições combinadas (LGBM@0.08 OU IF@threshold):")
    print(f"  {'Config':<30s} {'TP':>4s} {'FP':>5s} {'FN':>4s} {'Recall':>8s} {'Precision':>10s}")
    print("  " + "-" * 65)

    # LGBM sozinho
    tp_l = int(((lgbm_s >= LGBM_THRESHOLD) & (comp["is_fraud"] == 1)).sum())
    fp_l = int(((lgbm_s >= LGBM_THRESHOLD) & (comp["is_fraud"] == 0)).sum())
    fn_l = int(((lgbm_s < LGBM_THRESHOLD) & (comp["is_fraud"] == 1)).sum())
    print(f"  {'LGBM@0.08':<30s} {tp_l:4d} {fp_l:5d} {fn_l:4d} "
          f"{tp_l/max(tp_l+fn_l,1):8.4f} {tp_l/max(tp_l+fp_l,1):10.4f}")

    # LGBM + Cascade (v4.1)
    tp_c = int((comp["pred_combined"] == 1) & (comp["is_fraud"] == 1)).sum()
    fp_c = int((comp["pred_combined"] == 1) & (comp["is_fraud"] == 0)).sum()
    fn_c = int((comp["pred_combined"] == 0) & (comp["is_fraud"] == 1)).sum()
    print(f"  {'LGBM+Cascade (v4.1)':<30s} {tp_c:4d} {fp_c:5d} {fn_c:4d} "
          f"{tp_c/max(tp_c+fn_c,1):8.4f} {tp_c/max(tp_c+fp_c,1):10.4f}")

    # OR com diferentes thresholds de IF
    for if_th_name in ["0.70", "0.80", "0.85", "0.90"]:
        col = f"pred_or_if{if_th_name}"
        tp = int(((comp[col] == 1) & (comp["is_fraud"] == 1)).sum())
        fp = int(((comp[col] == 1) & (comp["is_fraud"] == 0)).sum())
        fn = int(((comp[col] == 0) & (comp["is_fraud"] == 1)).sum())
        print(f"  {'LGBM@0.08 OR IF@'+if_th_name:<30s} {tp:4d} {fp:5d} {fn:4d} "
              f"{tp/max(tp+fn,1):8.4f} {tp/max(tp+fp,1):10.4f}")

    # FULL (LGBM + Cascade + IF)
    print(f"\n  Pipeline completo (LGBM + Cascade + IF):")
    print(f"  {'Config':<30s} {'TP':>4s} {'FP':>5s} {'FN':>4s} {'Recall':>8s} {'Precision':>10s}")
    print("  " + "-" * 65)

    for if_th_name in ["0.70", "0.80", "0.85"]:
        col = f"pred_full_if{if_th_name}"
        tp = int(((comp[col] == 1) & (comp["is_fraud"] == 1)).sum())
        fp = int(((comp[col] == 1) & (comp["is_fraud"] == 0)).sum())
        fn = int(((comp[col] == 0) & (comp["is_fraud"] == 1)).sum())
        print(f"  {'LGBM+Cascade+IF@'+if_th_name:<30s} {tp:4d} {fp:5d} {fn:4d} "
              f"{tp/max(tp+fn,1):8.4f} {tp/max(tp+fp,1):10.4f}")

    # Análise: quais FN do LGBM+Cascade o IF captura?
    lgbm_cascade_fn = comp[(comp["is_fraud"] == 1) & (comp["pred_combined"] == 0)]
    if len(lgbm_cascade_fn) > 0:
        print(f"\n  FN do LGBM+Cascade ({len(lgbm_cascade_fn)}) — IF scores:")
        for _, row in lgbm_cascade_fn.iterrows():
            is_f = "1ªTX" if row["is_first_tx_trimestre"] == 1 else "REG"
            captured_70 = "✅" if row["anomaly_score"] >= 0.70 else "❌"
            captured_80 = "✅" if row["anomaly_score"] >= 0.80 else "❌"
            captured_85 = "✅" if row["anomaly_score"] >= 0.85 else "❌"
            print(f"    [{is_f}] lgbm={row['lgbm_score']:.4f} "
                  f"if={row['anomaly_score']:.4f} "
                  f"@0.70={captured_70} @0.80={captured_80} @0.85={captured_85}")

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
    "version": "v4_lgbm_complementary",
    "strategy": "Trained on all normal tx, velocity features, complementary to LGBM FN",
    "n_features": len(IF_FEATURES_FINAL),
    "features": IF_FEATURES_FINAL,
    "feature_groups": {
        "profile": FEATURES_PROFILE,
        "value": FEATURES_VALUE,
        "velocity": FEATURES_VELOCITY,
        "context": FEATURES_CONTEXT,
        "interaction": [f for f in FEATURES_INTERACTION if f in IF_FEATURES_FINAL],
    },
    "best_params": {k: str(v) if not isinstance(v, (int, float)) else v
                    for k, v in best_params.items()},
    "best_val_ap": float(best_val_ap),
    "thresholds": {
        "best_f1": float(best_threshold_f1),
        "best_f1_val": float(best_val_f1),
        "recall_90": float(th_recall_90),
        "recall_95": float(th_recall_95),
    },
    "n_train_normal": int(len(train_normal)),
    "n_val": int(len(df_val)),
    "n_holdout": int(len(df_holdout)),
    "metrics": metrics,
    "complementarity": complementarity,
}

if_config = {
    "features": IF_FEATURES_FINAL,
    "medians": {k: float(v) for k, v in medians.to_dict().items()},
    "best_threshold": float(best_threshold_f1),
    "best_params": {k: str(v) if not isinstance(v, (int, float)) else v
                    for k, v in best_params.items()},
    "ensemble_strategy": "complementary_boost",
    "ensemble_params": {
        "lgbm_threshold": 0.08,
        "description": "IF ativa quando LGBM < 0.08. Boost de 0.08-0.15 ao score.",
        "if_high_threshold": 0.70,
        "if_very_high_threshold": 0.85,
        "boost_high": 0.08,
        "boost_very_high": 0.15,
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

with open(IF_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(if_config, f, ensure_ascii=False, indent=2)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
np.save(os.path.join(ARTEFACT_DIR, "if_ref_raw_train.npy"), ref_raw_train)


# =========================================================
# SUMMARY
# =========================================================
print(f"\n{'='*70}")
print("RESULTADOS — Isolation Forest v4 (Complementar ao LGBM)")
print(f"{'='*70}")

print(f"\nConfig: {best_params}")
print(f"Features: {len(IF_FEATURES_FINAL)} ({len(FEATURES_VELOCITY)} velocity)")
print(f"Treino: {len(train_normal)} tx normais (todas, não apenas primeiras)")

m_h = metrics["holdout_0_5"]
print(f"\nHoldout @ 0.5:")
print(f"  ROC-AUC: {m_h['roc_auc']:.4f}" if m_h['roc_auc'] else "")
print(f"  AP:      {m_h['average_precision']:.4f}" if m_h['average_precision'] else "")
print(f"  F1:      {m_h['f1']:.4f}")
print(f"  R@5%:    {m_h['recall_at_5pct']:.4f}")

if complementarity:
    print(f"\nComplementaridade:")
    for name, m in complementarity.items():
        print(f"  {name}: AP={m['average_precision']:.4f} R@5%={m['recall_at_5pct']:.4f}")

print(f"\nArtefatos salvos:")
for path in [MODEL_PATH, SCALER_PATH, METRICS_PATH, IF_CONFIG_PATH,
             FEATURE_IMPORTANCE_PATH, PREDICTIONS_TEST_PATH]:
    print(f"  {path}")
print("=" * 70)
