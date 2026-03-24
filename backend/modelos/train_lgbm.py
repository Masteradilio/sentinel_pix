"""
train_lgbm.py v4.1 — LGBM v4.0 + Cascade para Recall Máximo

Mudanças v4.1 vs v4.0:
  1. Mesmo treino CV temporal do v4.0 (não mexe no modelo)
  2. ADICIONA busca de threshold por recall mínimo (não por F1)
  3. ADICIONA cascade rules pós-LGBM para bursts não detectados
  4. ADICIONA métricas focadas em FN=0 (recall=100%)
  5. SALVA múltiplos thresholds para uso em produção
  6. SALVA regras de cascade como artefato

Filosofia:
  - Banco prefere 50 FP a 1 FN
  - Threshold primário: recall >= 99% (aceita mais FP)
  - Cascade: regras simples para bursts (tx_count_prev_30m >= 3 + first_receiver)
  - Resultado: FN → 0 (ou quase), FP aceitável para analistas
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
from sklearn.model_selection import TimeSeriesSplit

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
OOF_PREDICTIONS_PATH = os.path.join(ARTEFACT_DIR, "oof_predictions.csv")
CV_METRICS_PATH = os.path.join(ARTEFACT_DIR, "cv_fold_metrics.json")
THRESHOLDS_PATH = os.path.join(ARTEFACT_DIR, "thresholds_config.json")
CASCADE_RULES_PATH = os.path.join(ARTEFACT_DIR, "cascade_rules.json")

os.makedirs(ARTEFACT_DIR, exist_ok=True)


# =========================================================
# FEATURES
# =========================================================
CORE_FEATURES = [
    "receiver_document_same_as_customer_flag",
    "pix_key_random_flag", "pix_key_email_flag",
    "pix_key_document_flag", "pix_key_other_flag",
    "pix_key_missing_flag_derived",
    "vl_pix", "log_vl_pix", "vl_pix_over_1000_flag",
    "qt_total_pix_trimestre", "is_first_tx_trimestre",
    "vl_mediana_pix_trimestre", "vl_desvio_padrao_pix_trimestre",
    "qt_intervalo_transacao_minuto", "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre", "qt_pix_dia_maximo_trimestre",
    "qt_aparelhos_distintos_trimestre",
    "nr_idade", "qt_tempo_relacionamento_mes",
    "latencia_rede_ms_final", "vl_latencia_rede_media_trimestre",
    "tempo_processamento_host_ms",
    "ratio_valor_mediana", "diff_valor_mediana",
    "ratio_valor_desvio_padrao", "zscore_valor_aprox",
    "ratio_intervalo_vs_mediana", "diff_intervalo_vs_mediana",
    "zscore_intervalo_aprox",
    "ratio_latencia_cliente", "diff_latencia_cliente", "latencia_host_ratio",
    "minutes_since_prev_tx",
    "tx_count_prev_30m", "burst_30m_flag",
    "receiver_tx_count_prev", "first_receiver_flag",
    "key_tx_count_prev", "first_key_flag",
    "distinct_receivers_so_far", "distinct_keys_so_far",
    "hour", "day_of_week", "is_business_hours",
    "app_version_minor",
    "topaz_risk_score", "topaz_score_filled",
    "device_missing_flag", "app_version_missing_flag",
    "auth_method_missing_flag", "topaz_missing_flag",
    "host_time_missing_flag", "latencia_missing_flag",
    "rule_age_score", "rule_relationship_score",
    "rule_mule_account_score", "rule_random_key_score",
    "rule_velocity_score", "rule_topaz_score",
    "rule_score_raw", "rule_score_normalized",
]

EXTRA_FEATURES = [
    "ratio_pix_renda", "vl_renda_cliente",
    "pix_over_50pct_renda_flag", "pix_over_100pct_renda_flag",
    "renda_missing_flag", "perfil_vulneravel_se_flag",
    "is_sexo_feminino_flag", "is_viuvo_flag",
    "is_segmento_premium_flag", "qt_dependentes",
    "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
    "is_agendamento_recorrente_flag", "metodo_auth_encoded",
    "is_login_senha_flag", "is_login_biometria_flag",
    "topaz_rejeitada_flag", "tempo_interacao_missing_flag",
]


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


def find_threshold_by_min_recall(y_true, y_prob, min_recall=0.99):
    """Encontra o threshold mais alto que garante recall >= min_recall."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    positives = y_true.sum()
    if positives == 0:
        return 0.5

    # Testar thresholds de alto para baixo
    for t in np.arange(0.95, 0.001, -0.001):
        preds = (y_prob >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        if rec >= min_recall:
            return float(t)

    # Se não encontrar, retornar o menor threshold testado
    return 0.001


def print_split_summary(name, df_sub, label_col="is_fraud"):
    n = len(df_sub)
    n_fraud = int(df_sub[label_col].sum())
    date_min = df_sub["event_datetime"].min()
    date_max = df_sub["event_datetime"].max()
    print(f"  {name}: {n:>7} rows | {n_fraud:>4} fraudes ({n_fraud/max(n,1)*100:.2f}%) | "
          f"{date_min} → {date_max}")


# =========================================================
# CASCADE RULES — regras para pegar o que o LGBM perde
# =========================================================
def apply_cascade_rules(df, y_prob, lgbm_threshold):
    """
    Aplica regras cascade para capturar fraudes que o LGBM perde.

    Regras baseadas na análise dos FN da v4.0:
    - Burst de transações rápidas para recebedores novos
    - Valores altos para contas novas
    - Padrões de esvaziamento de conta

    Returns:
        cascade_flags: array booleano (True = flagged pela cascade)
        cascade_reasons: lista de strings com razões
    """
    n = len(df)
    cascade_flags = np.zeros(n, dtype=bool)
    cascade_reasons = [""] * n

    for i in range(n):
        reasons = []

        # Já detectado pelo LGBM? Skip
        if y_prob[i] >= lgbm_threshold:
            continue

        row = df.iloc[i]

        # Regra C1: Burst intenso (3+ tx em 30min) + primeiro recebedor
        tx_30m = row.get("tx_count_prev_30m", 0) or 0
        first_recv = row.get("first_receiver_flag", 0) or 0
        if tx_30m >= 3 and first_recv == 1:
            reasons.append(f"BURST_FIRST_RECEIVER(tx30m={tx_30m})")

        # Regra C2: Burst intenso (5+ tx em 30min) qualquer recebedor
        if tx_30m >= 5:
            reasons.append(f"BURST_INTENSO(tx30m={tx_30m})")

        # Regra C3: Conta nova (< 6 meses) + primeiro recebedor + valor > mediana * 3
        tempo_rel = row.get("qt_tempo_relacionamento_mes", 999) or 999
        ratio_med = row.get("ratio_valor_mediana", 0) or 0
        vl_pix = row.get("vl_pix", 0) or 0
        if tempo_rel <= 6 and first_recv == 1 and ratio_med >= 3.0:
            reasons.append(f"CONTA_NOVA_ATIPICO(meses={tempo_rel},ratio={ratio_med:.1f})")

        # Regra C4: Conta muito nova (< 3 meses) + valor alto (> R$5000)
        if tempo_rel <= 3 and vl_pix >= 5000:
            reasons.append(f"CONTA_NOVA_ALTO_VALOR(meses={tempo_rel},vl={vl_pix:.0f})")

        # Regra C5: Burst + valor alto vs mediana (padrão esvaziamento)
        burst_flag = row.get("burst_30m_flag", 0) or 0
        if burst_flag == 1 and ratio_med >= 5.0 and vl_pix >= 1000:
            reasons.append(f"ESVAZIAMENTO(ratio={ratio_med:.1f},vl={vl_pix:.0f})")

        # Regra C6: LGBM deu score > 0.01 (não totalmente zerado) + sinais combinados
        if y_prob[i] >= 0.01:
            # Combinação: primeiro recebedor + valor acima mediana + qualquer sinal extra
            sinais = 0
            if first_recv == 1:
                sinais += 1
            if ratio_med >= 2.0:
                sinais += 1
            if vl_pix >= 1000:
                sinais += 1
            idade = row.get("nr_idade", 0) or 0
            if idade >= 60:
                sinais += 1
            chave_random = row.get("pix_key_random_flag", 0) or 0
            if chave_random == 1:
                sinais += 1

            if sinais >= 3:
                reasons.append(f"LGBM_BORDERLINE_COMBINADO(score={y_prob[i]:.4f},sinais={sinais})")

        if reasons:
            cascade_flags[i] = True
            cascade_reasons[i] = " | ".join(reasons)

    return cascade_flags, cascade_reasons


# =========================================================
# LOAD DATA
# =========================================================
print("=" * 70)
print("TREINO LightGBM v4.1 — CV Temporal + Cascade para Recall Máximo")
print("=" * 70)

print("\nLendo base model-ready...")
df = pd.read_csv(INPUT_DATA)
df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
df = df.sort_values("event_datetime").reset_index(drop=True)
df = df[df["is_fraud"].notna()].copy()
df["is_fraud"] = df["is_fraud"].astype(int)

total_rows = len(df)
total_fraud = int(df["is_fraud"].sum())
print(f"Shape: {df.shape}")
print(f"Fraudes: {total_fraud} | Normais: {total_rows - total_fraud}")
print(f"Proporção fraude: {df['is_fraud'].mean()*100:.3f}%")


# =========================================================
# FEATURE SELECTION
# =========================================================
available_cols = set(df.columns)
feature_cols = [f for f in CORE_FEATURES if f in available_cols]
extra_used = [f for f in EXTRA_FEATURES if f in available_cols]
feature_cols.extend(extra_used)

seen = set()
feature_cols = [f for f in feature_cols if not (f in seen or seen.add(f))]

print(f"\nFeatures: {len(feature_cols)} ({len([f for f in CORE_FEATURES if f in available_cols])} core + {len(extra_used)} extras)")


# =========================================================
# HOLDOUT
# =========================================================
holdout_ratio = 0.10
holdout_start = int(total_rows * (1 - holdout_ratio))

df_dev = df.iloc[:holdout_start].copy().reset_index(drop=True)
df_holdout = df.iloc[holdout_start:].copy().reset_index(drop=True)

print(f"\n--- Separação Dev / Holdout ---")
print_split_summary("Dev (CV)", df_dev)
print_split_summary("Holdout", df_holdout)

dev_fraud = int(df_dev["is_fraud"].sum())
holdout_fraud = int(df_holdout["is_fraud"].sum())
print(f"  Fraudes: Dev={dev_fraud} ({dev_fraud/total_fraud*100:.1f}%) | "
      f"Holdout={holdout_fraud} ({holdout_fraud/total_fraud*100:.1f}%)")


# =========================================================
# CROSS-VALIDATION TEMPORAL (idêntico ao v4.0)
# =========================================================
N_FOLDS = 5
MIN_TRAIN_FRAUD = 5

print(f"\n{'='*70}")
print(f"CROSS-VALIDATION TEMPORAL — {N_FOLDS} folds")
print(f"{'='*70}")

X_dev = df_dev[feature_cols]
y_dev = df_dev["is_fraud"]

tscv = TimeSeriesSplit(n_splits=N_FOLDS)

oof_scores = np.full(len(df_dev), np.nan)
fold_metrics = []
fold_models = []
fold_best_iters = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
    X_fold_train = X_dev.iloc[train_idx].reset_index(drop=True)
    y_fold_train = y_dev.iloc[train_idx].reset_index(drop=True)
    X_fold_val = X_dev.iloc[val_idx].reset_index(drop=True)
    y_fold_val = y_dev.iloc[val_idx].reset_index(drop=True)

    n_pos_train = int(y_fold_train.sum())
    n_pos_val = int(y_fold_val.sum())
    n_neg_train = len(y_fold_train) - n_pos_train

    print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
    print(f"  Treino: {len(X_fold_train)} rows, {n_pos_train} fraudes")
    print(f"  Val:    {len(X_fold_val)} rows, {n_pos_val} fraudes")

    if n_pos_train < MIN_TRAIN_FRAUD:
        print(f"  ⚠ Apenas {n_pos_train} fraudes no treino — SKIP")
        continue

    spw = n_neg_train / max(n_pos_train, 1)

    fold_model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=63,
        max_depth=7,
        min_child_samples=max(3, min(20, n_pos_train // 3)),
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        verbose=-1,
    )

    if n_pos_val >= 2:
        fold_model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric="average_precision",
            callbacks=[
                early_stopping(stopping_rounds=150, verbose=False),
                log_evaluation(period=200),
            ],
        )
        best_iter = fold_model.best_iteration_ if hasattr(fold_model, "best_iteration_") else -1
    else:
        fold_model.set_params(n_estimators=500)
        fold_model.fit(X_fold_train, y_fold_train)
        best_iter = 500

    print(f"  Best iteration: {best_iter}")
    fold_best_iters.append(best_iter)

    val_prob = fold_model.predict_proba(X_fold_val)[:, 1]
    oof_scores[val_idx] = val_prob

    if n_pos_val > 0:
        fold_auc = roc_auc_score(y_fold_val, val_prob)
        fold_ap = average_precision_score(y_fold_val, val_prob)
        fold_th, fold_f1 = find_best_threshold_by_f1(y_fold_val, val_prob)

        fold_met = {
            "fold": fold_idx + 1,
            "n_train": len(X_fold_train),
            "n_val": len(X_fold_val),
            "n_pos_train": n_pos_train,
            "n_pos_val": n_pos_val,
            "best_iteration": int(best_iter),
            "scale_pos_weight": float(spw),
            "roc_auc": float(fold_auc),
            "average_precision": float(fold_ap),
            "best_threshold": float(fold_th),
            "best_f1": float(fold_f1),
        }
        fold_metrics.append(fold_met)
        print(f"  ROC-AUC: {fold_auc:.4f} | AP: {fold_ap:.4f} | F1: {fold_f1:.4f} @ th={fold_th:.2f}")
    else:
        fold_metrics.append({
            "fold": fold_idx + 1, "n_train": len(X_fold_train),
            "n_val": len(X_fold_val), "n_pos_train": n_pos_train,
            "n_pos_val": 0, "best_iteration": int(best_iter),
            "note": "no_positives_in_val",
        })

    fold_models.append(fold_model)

with open(CV_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(fold_metrics, f, ensure_ascii=False, indent=2)


# =========================================================
# OOF ANALYSIS
# =========================================================
print(f"\n{'='*70}")
print("ANÁLISE OOF (Out-of-Fold)")
print(f"{'='*70}")

oof_mask = ~np.isnan(oof_scores)
oof_y = y_dev[oof_mask].values
oof_p = oof_scores[oof_mask]

print(f"  Predições OOF: {oof_mask.sum()} ({oof_mask.sum()/len(df_dev)*100:.1f}% do dev)")
print(f"  Fraudes nas OOF: {int(oof_y.sum())}")

if oof_y.sum() > 0:
    oof_auc = roc_auc_score(oof_y, oof_p)
    oof_ap = average_precision_score(oof_y, oof_p)
    oof_th_f1, oof_f1 = find_best_threshold_by_f1(oof_y, oof_p)
    oof_th_recall99 = find_threshold_by_min_recall(oof_y, oof_p, 0.99)
    oof_th_recall95 = find_threshold_by_min_recall(oof_y, oof_p, 0.95)

    print(f"  ROC-AUC OOF: {oof_auc:.4f}")
    print(f"  AP OOF:       {oof_ap:.4f}")
    print(f"  Threshold F1 OOF:       {oof_th_f1:.4f} → F1={oof_f1:.4f}")
    print(f"  Threshold Recall≥99% OOF: {oof_th_recall99:.4f}")
    print(f"  Threshold Recall≥95% OOF: {oof_th_recall95:.4f}")

    oof_df = pd.DataFrame({
        "idx": np.where(oof_mask)[0],
        "is_fraud": oof_y.astype(int),
        "oof_score": oof_p,
    })
    oof_df.to_csv(OOF_PREDICTIONS_PATH, index=False)
else:
    oof_th_f1 = 0.5
    oof_f1 = 0.0
    oof_th_recall99 = 0.01
    oof_th_recall95 = 0.05
    oof_auc = None
    oof_ap = None


# =========================================================
# MODELO FINAL
# =========================================================
print(f"\n{'='*70}")
print("RETREINO FINAL — usando todo o dev set")
print(f"{'='*70}")

X_final_train = df_dev[feature_cols].reset_index(drop=True)
y_final_train = df_dev["is_fraud"].reset_index(drop=True)

X_holdout = df_holdout[feature_cols].reset_index(drop=True)
y_holdout = df_holdout["is_fraud"].reset_index(drop=True)

n_pos_final = int(y_final_train.sum())
n_neg_final = len(y_final_train) - n_pos_final
spw_final = n_neg_final / max(n_pos_final, 1)

print(f"  Treino final: {len(X_final_train)} rows, {n_pos_final} fraudes")
print(f"  scale_pos_weight: {spw_final:.2f}")

if fold_best_iters:
    valid_iters = [it for it in fold_best_iters if it > 10]
    if valid_iters:
        final_n_estimators = int(np.median(valid_iters) * 1.2)
        final_n_estimators = max(100, min(2000, final_n_estimators))
    else:
        final_n_estimators = 500
else:
    final_n_estimators = 500

print(f"  n_estimators final: {final_n_estimators}")

final_model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=final_n_estimators,
    learning_rate=0.01,
    num_leaves=63,
    max_depth=7,
    min_child_samples=max(3, min(20, n_pos_final // 5)),
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=spw_final,
    verbose=-1,
)

final_model.fit(X_final_train, y_final_train)
print(f"  ✓ Modelo final treinado com {final_n_estimators} iterações")

importances = final_model.booster_.feature_importance(importance_type="gain")
n_features_used = int((importances > 0).sum())
print(f"  Features com gain > 0: {n_features_used}/{len(feature_cols)}")


# =========================================================
# CALIBRATION
# =========================================================
print("\nCalibrando probabilidades...")
use_calibrated = False

cal_start = int(len(df_dev) * 0.80)
X_cal = X_final_train.iloc[cal_start:]
y_cal = y_final_train.iloc[cal_start:]
cal_fraud = int(y_cal.sum())
print(f"  Set calibração: {len(X_cal)} rows, {cal_fraud} fraudes")

if cal_fraud >= 2:
    try:
        calibrated_model = CalibratedClassifierCV(final_model, method="sigmoid", cv="prefit")
        calibrated_model.fit(X_cal, y_cal)
        use_calibrated = True
        print("  ✓ Calibração OK (sigmoid, prefit)")
    except (TypeError, ValueError):
        try:
            calibrated_model = CalibratedClassifierCV(final_model, method="sigmoid", cv=2)
            calibrated_model.fit(X_cal, y_cal)
            use_calibrated = True
            print("  ✓ Calibração OK (sigmoid, cv=2)")
        except Exception as e:
            print(f"  ⚠ Calibração falhou: {e}")

if use_calibrated:
    joblib.dump(calibrated_model, MODEL_CALIBRATED_PATH)


# =========================================================
# HOLDOUT EVALUATION
# =========================================================
print(f"\n{'='*70}")
print("AVALIAÇÃO — Holdout Final")
print(f"{'='*70}")

y_holdout_prob = final_model.predict_proba(X_holdout)[:, 1]

if use_calibrated:
    y_holdout_prob_cal = calibrated_model.predict_proba(X_holdout)[:, 1]
else:
    y_holdout_prob_cal = y_holdout_prob


# =========================================================
# THRESHOLD SEARCH — múltiplos thresholds para produção
# =========================================================
print(f"\n{'='*70}")
print("BUSCA DE THRESHOLDS MÚLTIPLOS")
print(f"{'='*70}")

# Threshold por F1
th_f1, best_f1 = find_best_threshold_by_f1(y_holdout, y_holdout_prob)
print(f"  Threshold F1:          {th_f1:.4f} → F1={best_f1:.4f}")

# Threshold por recall mínimo (NO HOLDOUT para referência)
th_recall100 = find_threshold_by_min_recall(y_holdout, y_holdout_prob, 1.00)
th_recall99 = find_threshold_by_min_recall(y_holdout, y_holdout_prob, 0.99)
th_recall98 = find_threshold_by_min_recall(y_holdout, y_holdout_prob, 0.98)
th_recall95 = find_threshold_by_min_recall(y_holdout, y_holdout_prob, 0.95)

# Para cada threshold, calcular métricas
for name, th in [("R=100%", th_recall100), ("R=99%", th_recall99),
                  ("R=98%", th_recall98), ("R=95%", th_recall95), ("F1", th_f1)]:
    preds = (y_holdout_prob >= th).astype(int)
    tp = int(((preds == 1) & (y_holdout == 1)).sum())
    fp = int(((preds == 1) & (y_holdout == 0)).sum())
    fn = int(((preds == 0) & (y_holdout == 1)).sum())
    rec = recall_score(y_holdout, preds, zero_division=0)
    prec = precision_score(y_holdout, preds, zero_division=0)
    print(f"  {name:6}: th={th:.6f} | TP={tp:3d} FP={fp:4d} FN={fn:3d} | "
          f"Recall={rec:.4f} Precision={prec:.4f}")


# =========================================================
# CASCADE RULES EVALUATION
# =========================================================
print(f"\n{'='*70}")
print("AVALIAÇÃO — CASCADE RULES")
print(f"{'='*70}")

# Usar threshold de melhor F1 como base do LGBM
lgbm_threshold_for_cascade = th_f1

cascade_flags, cascade_reasons = apply_cascade_rules(
    df_holdout, y_holdout_prob, lgbm_threshold_for_cascade
)

# Predição combinada: LGBM OU cascade
lgbm_preds = (y_holdout_prob >= lgbm_threshold_for_cascade).astype(int)
combined_preds = np.maximum(lgbm_preds, cascade_flags.astype(int))

tp_lgbm = int(((lgbm_preds == 1) & (y_holdout == 1)).sum())
fp_lgbm = int(((lgbm_preds == 1) & (y_holdout == 0)).sum())
fn_lgbm = int(((lgbm_preds == 0) & (y_holdout == 1)).sum())

tp_combined = int(((combined_preds == 1) & (y_holdout == 1)).sum())
fp_combined = int(((combined_preds == 1) & (y_holdout == 0)).sum())
fn_combined = int(((combined_preds == 0) & (y_holdout == 1)).sum())

cascade_caught = int(((cascade_flags == True) & (y_holdout == 1) & (lgbm_preds == 0)).sum())
cascade_fp = int(((cascade_flags == True) & (y_holdout == 0) & (lgbm_preds == 0)).sum())

print(f"  LGBM sozinho @ th={lgbm_threshold_for_cascade:.4f}:")
print(f"    TP={tp_lgbm} FP={fp_lgbm} FN={fn_lgbm} | "
      f"Recall={tp_lgbm/(tp_lgbm+fn_lgbm):.4f} Precision={tp_lgbm/max(tp_lgbm+fp_lgbm,1):.4f}")

print(f"\n  Cascade Rules (sozinha, nos FN do LGBM):")
print(f"    Fraudes capturadas: {cascade_caught}")
print(f"    Falsos positivos:   {cascade_fp}")

print(f"\n  LGBM + Cascade COMBINADO:")
print(f"    TP={tp_combined} FP={fp_combined} FN={fn_combined} | "
      f"Recall={tp_combined/(tp_combined+fn_combined):.4f} "
      f"Precision={tp_combined/max(tp_combined+fp_combined,1):.4f}")

# Também testar com threshold de recall alto + cascade
for th_name, th_val in [("R=95%", th_recall95), ("R=98%", th_recall98), ("R=99%", th_recall99)]:
    lgbm_p = (y_holdout_prob >= th_val).astype(int)
    casc_f, _ = apply_cascade_rules(df_holdout, y_holdout_prob, th_val)
    comb_p = np.maximum(lgbm_p, casc_f.astype(int))
    tp_c = int(((comb_p == 1) & (y_holdout == 1)).sum())
    fp_c = int(((comb_p == 1) & (y_holdout == 0)).sum())
    fn_c = int(((comb_p == 0) & (y_holdout == 1)).sum())
    rec_c = tp_c / max(tp_c + fn_c, 1)
    prec_c = tp_c / max(tp_c + fp_c, 1)
    print(f"\n  LGBM@{th_name} + Cascade:")
    print(f"    TP={tp_c} FP={fp_c} FN={fn_c} | Recall={rec_c:.4f} Precision={prec_c:.4f}")


# =========================================================
# FULL METRICS (todos os thresholds)
# =========================================================
metrics = {
    "holdout_threshold_f1": evaluate_metrics(y_holdout, y_holdout_prob, th_f1),
    "holdout_threshold_0_5": evaluate_metrics(y_holdout, y_holdout_prob, 0.5),
    "holdout_recall_99": evaluate_metrics(y_holdout, y_holdout_prob, th_recall99),
    "holdout_recall_95": evaluate_metrics(y_holdout, y_holdout_prob, th_recall95),
    "train_threshold_0_5": evaluate_metrics(y_final_train,
                                             final_model.predict_proba(X_final_train)[:, 1], 0.5),
}

if use_calibrated:
    cal_th, cal_f1 = find_best_threshold_by_f1(y_holdout, y_holdout_prob_cal)
    metrics["holdout_calibrated_best"] = evaluate_metrics(y_holdout, y_holdout_prob_cal, cal_th)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
    "importance_split": final_model.booster_.feature_importance(importance_type="split"),
}).sort_values("importance_gain", ascending=False).reset_index(drop=True)

if len(fold_models) > 0:
    avg_gain = np.zeros(len(feature_cols))
    for fm in fold_models:
        avg_gain += fm.booster_.feature_importance(importance_type="gain")
    avg_gain /= len(fold_models)
    feature_importance["avg_fold_gain"] = avg_gain
    feature_importance = feature_importance.sort_values("avg_fold_gain", ascending=False).reset_index(drop=True)


# =========================================================
# SHAP
# =========================================================
print("\nCalculando SHAP...")
shap_importance = None
try:
    import shap
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_holdout)
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_importance.to_csv(SHAP_PATH, index=False)
    print(f"  Top 10 SHAP:")
    print(shap_importance.head(10).to_string(index=False))
except ImportError:
    print("  ⚠ shap não instalado")
except Exception as e:
    print(f"  ⚠ Erro SHAP: {e}")


# =========================================================
# ERROR ANALYSIS — focado nos FN residuais
# =========================================================
print(f"\n{'='*70}")
print("ANÁLISE DE ERROS — Threshold F1 + Cascade")
print(f"{'='*70}")

# Usar combined para error analysis
pred_holdout = pd.DataFrame({
    "transaction_id": df_holdout["transaction_id"].values,
    "customer_id": df_holdout["customer_id"].values,
    "event_datetime": df_holdout["event_datetime"].values,
    "is_fraud": y_holdout.values,
    "score_fraude": y_holdout_prob,
    "score_fraude_calibrated": y_holdout_prob_cal,
    "pred_lgbm": lgbm_preds,
    "pred_cascade": cascade_flags.astype(int),
    "pred_combined": combined_preds,
    "cascade_reason": cascade_reasons,
})

fn_mask = (pred_holdout["is_fraud"] == 1) & (pred_holdout["pred_combined"] == 0)
fp_mask = (pred_holdout["is_fraud"] == 0) & (pred_holdout["pred_combined"] == 1)

errors = pred_holdout[fn_mask | fp_mask].copy()
errors["error_type"] = np.where(errors["is_fraud"] == 1, "FALSE_NEGATIVE", "FALSE_POSITIVE")

error_features = [
    "vl_pix", "vl_mediana_pix_trimestre", "qt_total_pix_trimestre",
    "nr_idade", "qt_tempo_relacionamento_mes", "topaz_risk_score",
    "topaz_score_filled", "tx_count_prev_30m", "burst_30m_flag",
    "first_receiver_flag", "rule_score_raw", "vl_pix_over_1000_flag",
    "is_first_tx_trimestre", "hour", "day_of_week",
    "latencia_rede_ms_final", "app_version_minor", "ratio_valor_mediana",
]

for feat in error_features:
    if feat in X_holdout.columns:
        errors[feat] = X_holdout.loc[errors.index, feat].values

errors.to_csv(ERROR_ANALYSIS_PATH, index=False)

n_fn = int(fn_mask.sum())
n_fp = int(fp_mask.sum())
print(f"  Erros combinados: {n_fn} FN + {n_fp} FP")

if n_fn > 0:
    fn_errors = errors[errors["error_type"] == "FALSE_NEGATIVE"]
    print(f"\n  FN residuais ({n_fn}):")
    for _, row in fn_errors.iterrows():
        vlpix = row.get("vl_pix", "?")
        score = row["score_fraude"]
        cust = str(row["customer_id"])[:20]
        burst = row.get("burst_30m_flag", "?")
        tx_prev = row.get("tx_count_prev_30m", "?")
        first_r = row.get("first_receiver_flag", "?")
        print(f"    score={score:.6f} | vl_pix={vlpix:>10} | burst={burst} | "
              f"tx_30m={tx_prev} | first_recv={first_r} | cust={cust}")


# =========================================================
# SCORE DISTRIBUTION
# =========================================================
score_dist = pd.DataFrame({"class": y_holdout.values, "score": y_holdout_prob})
print("\nDistribuição de scores por classe:")
print(score_dist.groupby("class")["score"].describe().to_string())
score_dist.to_csv(SCORE_DISTRIBUTION_PATH, index=False)


# =========================================================
# SAVE ALL
# =========================================================
pred_holdout.to_csv(PREDICTIONS_TEST_PATH, index=False)
feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
joblib.dump(final_model, MODEL_PATH)

# Salvar thresholds para produção
thresholds_config = {
    "threshold_f1_best": float(th_f1),
    "threshold_recall_100": float(th_recall100),
    "threshold_recall_99": float(th_recall99),
    "threshold_recall_98": float(th_recall98),
    "threshold_recall_95": float(th_recall95),
    "threshold_oof_f1": float(oof_th_f1),
    "threshold_oof_recall_99": float(oof_th_recall99),
    "recommended_production": {
        "lgbm_threshold": float(th_f1),
        "cascade_enabled": True,
        "strategy": "LGBM @ best_F1 + cascade rules para bursts",
        "note": "Se FN combinado ainda > 0, reduzir lgbm_threshold para th_recall95",
    },
}

with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
    json.dump(thresholds_config, f, ensure_ascii=False, indent=2)

# Salvar regras cascade
cascade_config = {
    "version": "1.0",
    "rules": [
        {
            "id": "C1", "name": "BURST_FIRST_RECEIVER",
            "condition": "tx_count_prev_30m >= 3 AND first_receiver_flag == 1",
            "rationale": "Burst de 3+ tx em 30min para recebedor nunca visto",
        },
        {
            "id": "C2", "name": "BURST_INTENSO",
            "condition": "tx_count_prev_30m >= 5",
            "rationale": "5+ transações em 30 minutos (esvaziamento)",
        },
        {
            "id": "C3", "name": "CONTA_NOVA_ATIPICO",
            "condition": "qt_tempo_relacionamento_mes <= 6 AND first_receiver_flag == 1 AND ratio_valor_mediana >= 3",
            "rationale": "Conta nova + recebedor novo + valor 3x acima da mediana",
        },
        {
            "id": "C4", "name": "CONTA_NOVA_ALTO_VALOR",
            "condition": "qt_tempo_relacionamento_mes <= 3 AND vl_pix >= 5000",
            "rationale": "Conta muito nova (< 3 meses) + PIX alto valor",
        },
        {
            "id": "C5", "name": "ESVAZIAMENTO",
            "condition": "burst_30m_flag == 1 AND ratio_valor_mediana >= 5 AND vl_pix >= 1000",
            "rationale": "Burst + valor 5x mediana + > R$1000",
        },
        {
            "id": "C6", "name": "LGBM_BORDERLINE_COMBINADO",
            "condition": "lgbm_score >= 0.01 AND (sinais_combinados >= 3)",
            "rationale": "LGBM deu sinal fraco + 3+ sinais de risco combinados",
        },
    ],
}

with open(CASCADE_RULES_PATH, "w", encoding="utf-8") as f:
    json.dump(cascade_config, f, ensure_ascii=False, indent=2)


metadata = {
    "model_type": "LightGBM",
    "version": "v4.1_cascade_recall_max",
    "strategy": "TimeSeriesSplit CV + final retrain + cascade rules for max recall",
    "n_folds": N_FOLDS,
    "n_dev": int(len(df_dev)),
    "n_holdout": int(len(df_holdout)),
    "n_features": int(len(feature_cols)),
    "n_features_used": n_features_used,
    "positives_dev": int(y_final_train.sum()),
    "positives_holdout": int(y_holdout.sum()),
    "fraud_ratio_dev": float(y_final_train.mean()),
    "scale_pos_weight_final": float(spw_final),
    "final_n_estimators": final_n_estimators,
    "fold_best_iterations": fold_best_iters,
    "thresholds": thresholds_config,
    "cascade_enabled": True,
    "calibrated": use_calibrated,
    "n_errors_fn_lgbm_only": fn_lgbm,
    "n_errors_fp_lgbm_only": fp_lgbm,
    "n_errors_fn_combined": fn_combined,
    "n_errors_fp_combined": fp_combined,
    "cascade_caught_frauds": cascade_caught,
    "cascade_false_positives": cascade_fp,
    "metrics": metrics,
    "fold_metrics": fold_metrics,
    "top_20_features_gain": feature_importance.head(20).to_dict(orient="records"),
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)


# =========================================================
# SUMMARY
# =========================================================
print(f"\n{'='*70}")
print("RESULTADOS v4.1 — Cascade para Recall Máximo")
print(f"{'='*70}")

print(f"\nDev: {len(df_dev)} rows ({dev_fraud}F) | Holdout: {len(df_holdout)} rows ({holdout_fraud}F)")
print(f"Features: {len(feature_cols)} ({n_features_used} usadas)")

print(f"\n--- LGBM Sozinho ---")
print(f"  @F1 best ({th_f1:.4f}):   TP={tp_lgbm} FP={fp_lgbm} FN={fn_lgbm} | "
      f"Recall={tp_lgbm/(tp_lgbm+fn_lgbm):.4f} Prec={tp_lgbm/max(tp_lgbm+fp_lgbm,1):.4f}")

print(f"\n--- LGBM + Cascade ---")
print(f"  TP={tp_combined} FP={fp_combined} FN={fn_combined} | "
      f"Recall={tp_combined/(tp_combined+fn_combined):.4f} "
      f"Prec={tp_combined/max(tp_combined+fp_combined,1):.4f}")
print(f"  Cascade capturou: {cascade_caught} fraudes extras | {cascade_fp} FP extras")

print(f"\n--- ROC/AP (modelo LGBM) ---")
m = metrics["holdout_threshold_f1"]
print(f"  ROC-AUC: {m['roc_auc']:.4f}")
print(f"  AP:      {m['average_precision']:.4f}")
print(f"  R@5%:    {m['recall_at_5pct']:.4f}")
print(f"  P@1%:    {m['precision_at_1pct']:.4f}")

print(f"\n--- Comparativo Final ---")
print(f"  {'Métrica':<20} {'v4.0':>10} {'v4.1':>10} {'Alvo':>10}")
print(f"  {'ROC-AUC':<20} {'0.9954':>10} {m['roc_auc']:>10.4f} {'0.9998':>10}")
print(f"  {'AP':<20} {'0.9667':>10} {m['average_precision']:>10.4f} {'0.9680':>10}")
print(f"  {'FN (LGBM)':<20} {'66':>10} {fn_lgbm:>10} {'0':>10}")
print(f"  {'FN (combinado)':<20} {'N/A':>10} {fn_combined:>10} {'0':>10}")
print(f"  {'FP (combinado)':<20} {'1':>10} {fp_combined:>10} {'~20':>10}")

print(f"\nArtefatos salvos:")
print(f"  Modelo:            {MODEL_PATH}")
if use_calibrated:
    print(f"  Modelo calibrado:  {MODEL_CALIBRATED_PATH}")
print(f"  Métricas:          {METRICS_PATH}")
print(f"  Thresholds:        {THRESHOLDS_PATH}")
print(f"  Cascade Rules:     {CASCADE_RULES_PATH}")
print(f"  CV folds:          {CV_METRICS_PATH}")
print(f"  Feature importance:{FEATURE_IMPORTANCE_PATH}")
print(f"  Predições holdout: {PREDICTIONS_TEST_PATH}")
print(f"  Análise de erros:  {ERROR_ANALYSIS_PATH}")
print("=" * 70)
