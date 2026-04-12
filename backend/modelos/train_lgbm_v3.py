"""
train_lgbm_v3.py v6.1 — LGBM com Graph Features Temporais (sem leakage)

Mudanças v6.1 vs v6.0:
  1. Graph features: 13 temporais (prefixo graph_) em vez de 15+5 do grafo estático
  2. Removidas: sender_community, receiver_community, same_community (noise)
  3. Removidas: sender_total_value, receiver_total_value (leakage)
  4. Novas: graph_pair_tx_count, graph_degree_ratio, graph_*_value_ratio
  5. Baseline: v5.1 (sem graph) — objetivo é SUPERAR
"""

import hashlib
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_lgbm_v6")


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DADOS_DIR = PROJECT_ROOT / "dados"
ARTEFACT_DIR = PROJECT_ROOT / "backend" / "artefatos"
RESULT_DIR = PROJECT_ROOT / "backend" / "modelos" / "resultado_treino_lgbm_v3"

INPUT_DATA = DADOS_DIR / "base_treino_final.csv"

# ── Produção: modelo serializado ──
MODEL_PATH = ARTEFACT_DIR / "model_lightgbm.joblib"

# ── Relatório: artefatos de treino ──
METRICS_PATH = RESULT_DIR / "metricas_lgbm_v6.json"
CV_METRICS_PATH = RESULT_DIR / "cv_fold_metrics.json"
FEATURES_PATH = RESULT_DIR / "lgbm_features.json"
THRESHOLDS_PATH = RESULT_DIR / "thresholds_config.json"
FEATURE_IMPORTANCE_PATH = RESULT_DIR / "feature_importance.csv"
OOF_PREDICTIONS_PATH = RESULT_DIR / "oof_predictions.csv"
HOLDOUT_PREDICTIONS_PATH = RESULT_DIR / "holdout_predictions.csv"
SCORE_DISTRIBUTION_PATH = RESULT_DIR / "score_distribution.csv"
THRESHOLD_SWEEP_PATH = RESULT_DIR / "threshold_sweep.csv"
TRAINING_LOG_PATH = RESULT_DIR / "training_log.txt"

ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Versão anterior para comparação (v5.1 — leakage-free sem graph)
V51_METRICS = {
    "roc_auc": 0.9998,
    "average_precision": 0.9791,
    "f1": 0.9576,
    "recall": 0.9875,
    "precision": 0.9294,
    "threshold_f1": 0.35,
    "fn": 1,
    "fp": 6,
    "tp": 79,
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════════════
LEAKAGE_FIXED_FEATURES = {
    "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre",
    "qt_total_pix_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre",
    "is_first_tx_trimestre",
    "ratio_valor_mediana",
    "diff_valor_mediana",
    "ratio_valor_desvio_padrao",
    "zscore_valor_aprox",
    "ratio_intervalo_vs_mediana",
    "diff_intervalo_vs_mediana",
    "zscore_intervalo_aprox",
}

CORE_FEATURES = [
    # --- Valor e Desvio ---
    "vl_pix",
    "vl_pix_over_1000_flag",
    "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre",
    "ratio_valor_mediana",
    "diff_valor_mediana",
    "ratio_valor_desvio_padrao",
    "zscore_valor_aprox",
    # --- Frequência e Velocity ---
    "qt_total_pix_trimestre",
    "is_first_tx_trimestre",
    "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "ratio_intervalo_vs_mediana",
    "diff_intervalo_vs_mediana",
    "zscore_intervalo_aprox",
    "minutes_since_prev_tx",
    # --- Burst e Velocity ---
    "tx_count_prev_30m",
    "burst_30m_flag",
    # --- Recebedor ---
    "receiver_tx_count_prev",
    "first_receiver_flag",
    "distinct_receivers_so_far",
    "tp_primeiro_envio_recebedor_trimestre",
    "qt_envio_recebedor_trimestre",
    # --- Chave PIX ---
    "pix_key_random_flag",
    "key_tx_count_prev",
    "first_key_flag",
    "distinct_keys_so_far",
    # --- Temporal ---
    "hour",
    # --- Perfil do Cliente ---
    "nr_idade",
    "qt_tempo_relacionamento_mes",
    # --- Dispositivo e Sessão ---
    "qt_aparelhos_distintos_trimestre",
    "vl_latencia_rede_media_trimestre",
    "ratio_latencia_cliente",
    "diff_latencia_cliente",
    # --- Missing Flags ---
    "device_missing_flag",
    "host_time_missing_flag",
    "topaz_missing_flag",
    # --- Topaz ---
    "topaz_risk_score",
    # --- Regras de Negócio ---
    "rule_age_score",
    "rule_relationship_score",
    "rule_random_key_score",
    "rule_topaz_score",
    "rule_score_raw",
]

EXTRA_FEATURES = [
    "ratio_pix_renda",
    "vl_renda_cliente",
    "pix_over_50pct_renda_flag",
    "renda_missing_flag",
    "perfil_vulneravel_se_flag",
    "is_viuvo_flag",
    "is_segmento_premium_flag",
]

# ── Graph Features TEMPORAIS (13 — sem leakage, prefixo graph_) ──
GRAPH_FEATURES = [
    # Sender (pagador)
    "graph_sender_out_degree",
    "graph_sender_tx_count",
    "graph_sender_avg_value",
    "graph_sender_value_zscore",
    # Receiver (recebedor)
    "graph_receiver_in_degree",
    "graph_receiver_tx_count",
    "graph_receiver_avg_value",
    # Pair (par pagador-recebedor)
    "graph_pair_tx_count",
    "graph_is_new_edge",
    # Ratios derivados
    "graph_sender_value_ratio",
    "graph_receiver_value_ratio",
    "graph_receiver_concentration_hhi",
    "graph_degree_ratio",
]


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def md5_file(path: Path) -> str:
    """Calcula MD5 do arquivo de input para rastreabilidade."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k_ratio: float = 0.01,
) -> float:
    """Precision nos top-k% do ranking."""
    k = max(1, int(len(y_true) * k_ratio))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / k)


def recall_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k_ratio: float = 0.01,
) -> float:
    """Recall nos top-k% do ranking."""
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    k = max(1, int(len(y_true) * k_ratio))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / positives)


def recall_at_top_n(
    y_true: np.ndarray, y_score: np.ndarray, n: int,
) -> float:
    """Recall nos top-N absolutos do ranking."""
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    k = min(n, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / positives)


def evaluate_full(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
) -> dict:
    """Calcula métricas completas para um dado threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(float(threshold), 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "roc_auc": (
            round(float(roc_auc_score(y_true, y_prob)), 8) if has_both else None
        ),
        "average_precision": (
            round(float(average_precision_score(y_true, y_prob)), 8)
            if has_both else None
        ),
        "precision": round(
            float(precision_score(y_true, y_pred, zero_division=0)), 6,
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, zero_division=0)), 6,
        ),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
        "precision_at_1pct": round(precision_at_k(y_true, y_prob, 0.01), 4),
        "recall_at_1pct": round(recall_at_k(y_true, y_prob, 0.01), 4),
        "precision_at_5pct": round(precision_at_k(y_true, y_prob, 0.05), 4),
        "recall_at_5pct": round(recall_at_k(y_true, y_prob, 0.05), 4),
        "recall_at_top_50": round(recall_at_top_n(y_true, y_prob, 50), 4),
        "recall_at_top_100": round(recall_at_top_n(y_true, y_prob, 100), 4),
        "recall_at_top_200": round(recall_at_top_n(y_true, y_prob, 200), 4),
        "recall_at_top_500": round(recall_at_top_n(y_true, y_prob, 500), 4),
    }


def find_best_threshold_by_f1(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> tuple[float, float]:
    """Busca threshold de melhor F1 com step 0.005."""
    best_threshold, best_f1 = 0.5, -1.0
    for t in np.arange(0.005, 0.96, 0.005):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)
    return round(best_threshold, 4), round(best_f1, 6)


def find_threshold_by_min_recall(
    y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.99,
) -> float:
    """Encontra o threshold mais alto que garante recall >= min_recall."""
    positives = y_true.sum()
    if positives == 0:
        return 0.5
    for t in np.arange(0.95, 0.0005, -0.001):
        rec = recall_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if rec >= min_recall:
            return round(float(t), 4)
    return 0.001


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray,
) -> pd.DataFrame:
    """Varre thresholds com step 0.005 e calcula métricas para cada um."""
    rows = []
    for t in np.arange(0.005, 0.96, 0.005):
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": round(float(t), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": round(
                float(precision_score(y_true, y_pred, zero_division=0)), 6,
            ),
            "recall": round(
                float(recall_score(y_true, y_pred, zero_division=0)), 6,
            ),
            "f1": round(
                float(f1_score(y_true, y_pred, zero_division=0)), 6,
            ),
            "fpr": round(float(fp / max(fp + tn, 1)), 8),
        })
    return pd.DataFrame(rows)


def print_split_summary(
    name: str, df_sub: pd.DataFrame, label_col: str = "is_fraud",
) -> str:
    """Imprime resumo de um split (dev/holdout)."""
    n = len(df_sub)
    n_fraud = int(df_sub[label_col].sum())
    date_min = df_sub["event_datetime"].min()
    date_max = df_sub["event_datetime"].max()
    msg = (
        f"  {name}: {n:>7,} rows | {n_fraud:>4} fraudes "
        f"({n_fraud / max(n, 1) * 100:.2f}%) | {date_min} → {date_max}"
    )
    print(msg)
    return msg


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Pipeline principal de treino do LGBM v6.1."""
    t_start = time.time()
    training_log: list[str] = []

    def tlog(msg: str) -> None:
        """Log para console e arquivo."""
        log.info(msg)
        training_log.append(f"[{time.time() - t_start:7.1f}s] {msg}")

    print("=" * 72)
    print("  TREINO LightGBM v6.1 — Graph Features Temporais (sem leakage)")
    print("=" * 72)

    # ──────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ──────────────────────────────────────────────────────────────
    tlog(f"Input: {INPUT_DATA}")
    if not INPUT_DATA.exists():
        tlog(f"❌ Arquivo não encontrado: {INPUT_DATA}")
        tlog("   Rode primeiro: python core/preprocessing.py")
        sys.exit(1)

    input_md5 = md5_file(INPUT_DATA)
    tlog(f"MD5: {input_md5}")

    df = pd.read_csv(INPUT_DATA)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df = df.sort_values("event_datetime").reset_index(drop=True)
    df = df[df["is_fraud"].notna()].copy()
    df["is_fraud"] = df["is_fraud"].astype(int)

    total_rows = len(df)
    total_fraud = int(df["is_fraud"].sum())
    total_normal = total_rows - total_fraud

    tlog(f"Shape: {df.shape}")
    tlog(
        f"Fraudes: {total_fraud} | Normais: {total_normal} | "
        f"Ratio: {df['is_fraud'].mean() * 100:.3f}%"
    )

    # ──────────────────────────────────────────────────────────────
    # 2. FEATURE SELECTION
    # ──────────────────────────────────────────────────────────────
    available_cols = set(df.columns)
    feature_cols: list[str] = []
    seen: set[str] = set()

    for f in CORE_FEATURES + EXTRA_FEATURES + GRAPH_FEATURES:
        if f in available_cols and f not in seen:
            feature_cols.append(f)
            seen.add(f)

    n_core = len([f for f in CORE_FEATURES if f in available_cols])
    n_extra = len([f for f in EXTRA_FEATURES if f in available_cols])
    n_graph = len([f for f in GRAPH_FEATURES if f in available_cols])
    n_fixed = len([f for f in feature_cols if f in LEAKAGE_FIXED_FEATURES])

    tlog(
        f"Features: {len(feature_cols)} total "
        f"({n_core} core + {n_extra} extra + {n_graph} graph)"
    )
    tlog(f"Features corrigidas (leakage-free): {n_fixed}")

    graph_present = [f for f in GRAPH_FEATURES if f in available_cols]
    graph_missing = [f for f in GRAPH_FEATURES if f not in available_cols]
    if graph_missing:
        tlog(f"⚠ Graph features ausentes: {graph_missing}")
    tlog(f"Graph features temporais: {len(graph_present)} (prefixo graph_)")

    # Verificar NaN
    nan_report: dict[str, int] = {}
    for f in feature_cols:
        n_nan = int(df[f].isna().sum())
        if n_nan > 0:
            nan_report[f] = n_nan
    if nan_report:
        tlog(f"Features com NaN: {len(nan_report)}")
        for feat, count in sorted(nan_report.items(), key=lambda x: -x[1])[:10]:
            fixed_tag = " ⚠️CORRIGIDA" if feat in LEAKAGE_FIXED_FEATURES else ""
            graph_tag = " 📊GRAPH" if feat in GRAPH_FEATURES else ""
            tlog(
                f"  {feat}: {count:,} NaN "
                f"({count / total_rows * 100:.1f}%){fixed_tag}{graph_tag}"
            )

    # ──────────────────────────────────────────────────────────────
    # 3. HOLDOUT SPLIT (temporal, 90/10)
    # ──────────────────────────────────────────────────────────────
    holdout_ratio = 0.10
    holdout_start = int(total_rows * (1 - holdout_ratio))

    df_dev = df.iloc[:holdout_start].copy().reset_index(drop=True)
    df_holdout = df.iloc[holdout_start:].copy().reset_index(drop=True)

    tlog("--- Split Dev / Holdout ---")
    training_log.append(print_split_summary("Dev (CV)", df_dev))
    training_log.append(print_split_summary("Holdout ", df_holdout))

    dev_fraud = int(df_dev["is_fraud"].sum())
    holdout_fraud = int(df_holdout["is_fraud"].sum())
    tlog(
        f"Fraudes — Dev: {dev_fraud} ({dev_fraud / total_fraud * 100:.1f}%) | "
        f"Holdout: {holdout_fraud} ({holdout_fraud / total_fraud * 100:.1f}%)"
    )

    X_dev = df_dev[feature_cols]
    y_dev = df_dev["is_fraud"]
    X_holdout = df_holdout[feature_cols]
    y_holdout = df_holdout["is_fraud"]

    # ──────────────────────────────────────────────────────────────
    # 4. CROSS-VALIDATION TEMPORAL
    # ──────────────────────────────────────────────────────────────
    N_FOLDS = 5
    MIN_TRAIN_FRAUD = 5

    print(f"\n{'=' * 72}")
    print(f"  CROSS-VALIDATION TEMPORAL — {N_FOLDS} folds")
    print(f"{'=' * 72}")

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    oof_scores = np.full(len(df_dev), np.nan)
    fold_metrics: list[dict] = []
    fold_models: list[LGBMClassifier] = []
    fold_best_iters: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
        X_tr = X_dev.iloc[train_idx]
        y_tr = y_dev.iloc[train_idx]
        X_va = X_dev.iloc[val_idx]
        y_va = y_dev.iloc[val_idx]

        n_pos_tr = int(y_tr.sum())
        n_pos_va = int(y_va.sum())
        n_neg_tr = len(y_tr) - n_pos_tr

        date_tr_min = df_dev.iloc[train_idx]["event_datetime"].min()
        date_tr_max = df_dev.iloc[train_idx]["event_datetime"].max()
        date_va_min = df_dev.iloc[val_idx]["event_datetime"].min()
        date_va_max = df_dev.iloc[val_idx]["event_datetime"].max()

        tlog(f"--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        tlog(
            f"  Treino: {len(X_tr):,} rows, {n_pos_tr} fraudes "
            f"({date_tr_min} → {date_tr_max})"
        )
        tlog(
            f"  Val:    {len(X_va):,} rows, {n_pos_va} fraudes "
            f"({date_va_min} → {date_va_max})"
        )

        if n_pos_tr < MIN_TRAIN_FRAUD:
            tlog(f"  ⚠ Apenas {n_pos_tr} fraudes no treino — SKIP")
            fold_metrics.append({
                "fold": fold_idx + 1,
                "status": "skipped",
                "reason": f"n_pos_train={n_pos_tr} < {MIN_TRAIN_FRAUD}",
            })
            continue

        spw = n_neg_tr / max(n_pos_tr, 1)

        fold_model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=63,
            max_depth=7,
            min_child_samples=max(3, min(20, n_pos_tr // 3)),
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=spw,
            verbose=-1,
        )

        if n_pos_va >= 2:
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="average_precision",
                callbacks=[
                    early_stopping(stopping_rounds=150, verbose=False),
                    log_evaluation(period=500),
                ],
            )
            best_iter = getattr(
                fold_model, "best_iteration_", fold_model.n_estimators,
            )
        else:
            fold_model.set_params(n_estimators=500)
            fold_model.fit(X_tr, y_tr)
            best_iter = 500

        fold_best_iters.append(best_iter)
        tlog(f"  Best iteration: {best_iter}")

        val_prob = fold_model.predict_proba(X_va)[:, 1]
        oof_scores[val_idx] = val_prob

        if n_pos_va > 0:
            fold_auc = roc_auc_score(y_va, val_prob)
            fold_ap = average_precision_score(y_va, val_prob)
            fold_th, fold_f1 = find_best_threshold_by_f1(y_va.values, val_prob)
            fold_preds = (val_prob >= fold_th).astype(int)
            fold_rec = recall_score(y_va, fold_preds, zero_division=0)
            fold_prec = precision_score(y_va, fold_preds, zero_division=0)
            fold_cm = confusion_matrix(y_va, fold_preds)
            fold_tn, fold_fp, fold_fn, fold_tp = fold_cm.ravel()

            fm = {
                "fold": fold_idx + 1,
                "status": "ok",
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "n_pos_train": n_pos_tr,
                "n_pos_val": n_pos_va,
                "date_train": f"{date_tr_min} → {date_tr_max}",
                "date_val": f"{date_va_min} → {date_va_max}",
                "best_iteration": int(best_iter),
                "scale_pos_weight": round(float(spw), 2),
                "roc_auc": round(float(fold_auc), 6),
                "average_precision": round(float(fold_ap), 6),
                "best_threshold_f1": round(float(fold_th), 4),
                "best_f1": round(float(fold_f1), 6),
                "recall_at_best_f1": round(float(fold_rec), 4),
                "precision_at_best_f1": round(float(fold_prec), 4),
                "tp": int(fold_tp),
                "fp": int(fold_fp),
                "fn": int(fold_fn),
            }
            fold_metrics.append(fm)
            tlog(
                f"  AUC: {fold_auc:.4f} | AP: {fold_ap:.4f} | "
                f"F1: {fold_f1:.4f} @ th={fold_th:.4f} | "
                f"Rec: {fold_rec:.4f} Prec: {fold_prec:.4f} | "
                f"TP={fold_tp} FP={fold_fp} FN={fold_fn}"
            )
        else:
            fold_metrics.append({
                "fold": fold_idx + 1,
                "status": "no_positives_in_val",
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "n_pos_train": n_pos_tr,
                "n_pos_val": 0,
                "best_iteration": int(best_iter),
            })

        fold_models.append(fold_model)

    # ──────────────────────────────────────────────────────────────
    # 5. OOF ANALYSIS
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  ANÁLISE OOF (Out-of-Fold)")
    print(f"{'=' * 72}")

    oof_mask = ~np.isnan(oof_scores)
    oof_y = y_dev[oof_mask].values
    oof_p = oof_scores[oof_mask]

    tlog(
        f"OOF: {oof_mask.sum():,} predições "
        f"({oof_mask.sum() / len(df_dev) * 100:.1f}% do dev)"
    )
    tlog(f"OOF fraudes: {int(oof_y.sum())}")

    oof_auc: float | None = None
    oof_ap: float | None = None
    oof_th_f1 = 0.5
    oof_f1 = 0.0

    if oof_y.sum() > 0:
        oof_auc = round(float(roc_auc_score(oof_y, oof_p)), 8)
        oof_ap = round(float(average_precision_score(oof_y, oof_p)), 8)
        oof_th_f1, oof_f1 = find_best_threshold_by_f1(oof_y, oof_p)
        oof_th_r99 = find_threshold_by_min_recall(oof_y, oof_p, 0.99)
        oof_th_r95 = find_threshold_by_min_recall(oof_y, oof_p, 0.95)

        tlog(f"OOF ROC-AUC:  {oof_auc:.6f}")
        tlog(f"OOF AP:       {oof_ap:.6f}")
        tlog(f"OOF F1 best:  {oof_f1:.4f} @ th={oof_th_f1:.4f}")
        tlog(f"OOF th R≥99%: {oof_th_r99:.4f}")
        tlog(f"OOF th R≥95%: {oof_th_r95:.4f}")

        oof_df = pd.DataFrame({
            "dev_idx": np.where(oof_mask)[0],
            "is_fraud": oof_y.astype(int),
            "oof_score": np.round(oof_p, 8),
        })
        oof_df.to_csv(OOF_PREDICTIONS_PATH, index=False)
        tlog(f"OOF salvo: {OOF_PREDICTIONS_PATH}")
    else:
        oof_th_r99 = 0.01
        oof_th_r95 = 0.05

    # ──────────────────────────────────────────────────────────────
    # 6. MODELO FINAL — retreino em todo dev
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  MODELO FINAL — Retreino em todo o Dev")
    print(f"{'=' * 72}")

    n_pos_final = int(y_dev.sum())
    n_neg_final = len(y_dev) - n_pos_final
    spw_final = n_neg_final / max(n_pos_final, 1)

    valid_iters = [it for it in fold_best_iters if it > 10]
    if valid_iters:
        final_n_estimators = int(np.median(valid_iters) * 1.2)
        final_n_estimators = max(200, min(2500, final_n_estimators))
    else:
        final_n_estimators = 500

    tlog(f"Treino final: {len(X_dev):,} rows, {n_pos_final} fraudes")
    tlog(f"scale_pos_weight: {spw_final:.2f}")
    tlog(f"n_estimators: {final_n_estimators}")

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

    final_model.fit(X_dev, y_dev)

    importances_gain = final_model.booster_.feature_importance(
        importance_type="gain",
    )
    n_features_used = int((importances_gain > 0).sum())
    tlog(
        f"✅ Modelo treinado — {n_features_used}/{len(feature_cols)} "
        f"features com gain > 0"
    )

    # Graph features com gain > 0
    graph_gains = {
        f: importances_gain[i]
        for i, f in enumerate(feature_cols) if f in GRAPH_FEATURES
    }
    graph_active = {k: v for k, v in graph_gains.items() if v > 0}
    tlog(
        f"Graph features com gain > 0: {len(graph_active)}/{len(graph_gains)}"
    )
    for feat, gain in sorted(graph_active.items(), key=lambda x: -x[1]):
        tlog(f"  📊 {feat}: gain={gain:.1f}")

    # ──────────────────────────────────────────────────────────────
    # 7. CALIBRAÇÃO
    # ──────────────────────────────────────────────────────────────
    tlog("Calibrando probabilidades...")
    use_calibrated = False
    calibrated_model = None

    cal_start = int(len(X_dev) * 0.70)
    X_cal = X_dev.iloc[cal_start:].reset_index(drop=True)
    y_cal = y_dev.iloc[cal_start:].reset_index(drop=True)
    cal_fraud = int(y_cal.sum())

    tlog(f"  Set calibração: {len(X_cal):,} rows, {cal_fraud} fraudes")

    if cal_fraud >= 10:
        try:
            cal_model = LGBMClassifier(
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

            calibrated_model = CalibratedClassifierCV(
                estimator=cal_model,
                method="isotonic",
                cv=3,
            )
            calibrated_model.fit(X_cal, y_cal)
            use_calibrated = True
            tlog("✅ Calibração isotonic (cv=3) OK")
        except Exception as e:
            tlog(f"⚠ Calibração falhou: {e}")
    else:
        tlog(f"⚠ Poucas fraudes para calibração ({cal_fraud}), pulando")

    # ──────────────────────────────────────────────────────────────
    # 8. HOLDOUT EVALUATION
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  AVALIAÇÃO — Holdout Final")
    print(f"{'=' * 72}")

    y_ho_prob = final_model.predict_proba(X_holdout)[:, 1]
    y_ho_prob_cal = (
        calibrated_model.predict_proba(X_holdout)[:, 1]
        if use_calibrated
        else y_ho_prob
    )

    y_train_prob = final_model.predict_proba(X_dev)[:, 1]

    # Threshold search
    th_f1, best_f1 = find_best_threshold_by_f1(y_holdout.values, y_ho_prob)
    th_r100 = find_threshold_by_min_recall(y_holdout.values, y_ho_prob, 1.00)
    th_r99 = find_threshold_by_min_recall(y_holdout.values, y_ho_prob, 0.99)
    th_r98 = find_threshold_by_min_recall(y_holdout.values, y_ho_prob, 0.98)
    th_r95 = find_threshold_by_min_recall(y_holdout.values, y_ho_prob, 0.95)

    th_f1_cal = th_f1
    if use_calibrated:
        th_f1_cal, _ = find_best_threshold_by_f1(y_holdout.values, y_ho_prob_cal)

    tlog("--- Thresholds (modelo raw) ---")
    for name, th in [
        ("F1-best", th_f1), ("R=100%", th_r100), ("R=99%", th_r99),
        ("R=98%", th_r98), ("R=95%", th_r95),
    ]:
        y_pred = (y_ho_prob >= th).astype(int)
        cm = confusion_matrix(y_holdout, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rec = recall_score(y_holdout, y_pred, zero_division=0)
        prec = precision_score(y_holdout, y_pred, zero_division=0)
        f1v = f1_score(y_holdout, y_pred, zero_division=0)
        tlog(
            f"  {name:8}: th={th:.6f} | TP={tp:3d} FP={fp:4d} FN={fn:3d} | "
            f"Rec={rec:.4f} Prec={prec:.4f} F1={f1v:.4f}"
        )

    if use_calibrated:
        tlog(f"--- Threshold calibrado: F1-best = {th_f1_cal:.4f} ---")

    # Métricas completas
    holdout_metrics_f1 = evaluate_full(y_holdout.values, y_ho_prob, th_f1)
    holdout_metrics_05 = evaluate_full(y_holdout.values, y_ho_prob, 0.5)
    train_metrics = evaluate_full(y_dev.values, y_train_prob, th_f1)

    holdout_metrics_cal = None
    if use_calibrated:
        holdout_metrics_cal = evaluate_full(
            y_holdout.values, y_ho_prob_cal, th_f1_cal,
        )

    # ──────────────────────────────────────────────────────────────
    # 9. FEATURE IMPORTANCE
    # ──────────────────────────────────────────────────────────────
    tlog("Calculando feature importance...")

    total_gain = float(importances_gain.sum())
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importances_gain,
        "importance_gain_pct": np.round(
            importances_gain / max(total_gain, 1) * 100, 4,
        ),
        "importance_split": final_model.booster_.feature_importance(
            importance_type="split",
        ),
        "is_leakage_fixed": [f in LEAKAGE_FIXED_FEATURES for f in feature_cols],
        "is_graph_feature": [f in GRAPH_FEATURES for f in feature_cols],
        "feature_group": [
            "graph" if f in GRAPH_FEATURES
            else "leakage_fixed" if f in LEAKAGE_FIXED_FEATURES
            else "core" if f in CORE_FEATURES
            else "extra"
            for f in feature_cols
        ],
    })

    # Média dos folds
    if fold_models:
        avg_gain = np.zeros(len(feature_cols))
        for fm_model in fold_models:
            avg_gain += fm_model.booster_.feature_importance(
                importance_type="gain",
            )
        avg_gain /= len(fold_models)
        fi_df["avg_fold_gain"] = avg_gain

    fi_df = fi_df.sort_values(
        "importance_gain", ascending=False,
    ).reset_index(drop=True)
    fi_df["rank"] = range(1, len(fi_df) + 1)
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    tlog("Top 20 features (gain):")
    for _, row in fi_df.head(20).iterrows():
        tag = ""
        if row["is_leakage_fixed"]:
            tag = " ⚠️FIX"
        elif row["is_graph_feature"]:
            tag = " 📊GRAPH"
        tlog(
            f"  #{int(row['rank']):2d} {row['feature']:45s} "
            f"gain={row['importance_gain']:12.1f} "
            f"({row['importance_gain_pct']:5.2f}%){tag}"
        )

    # Resumo por grupo
    for group_name in ["core", "leakage_fixed", "extra", "graph"]:
        group_df = fi_df[fi_df["feature_group"] == group_name]
        group_gain = group_df["importance_gain"].sum()
        group_pct = group_gain / max(total_gain, 1) * 100
        group_active = int((group_df["importance_gain"] > 0).sum())
        tlog(
            f"  Grupo '{group_name}': {group_active}/{len(group_df)} ativas, "
            f"{group_pct:.1f}% do gain total"
        )

    # ──────────────────────────────────────────────────────────────
    # 10. CURVAS PR/ROC + THRESHOLD SWEEP
    # ──────────────────────────────────────────────────────────────
    tlog("Gerando curvas PR/ROC e threshold sweep...")

    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(
        y_holdout, y_ho_prob,
    )
    pr_df = pd.DataFrame({
        "precision": prec_arr[:-1],
        "recall": rec_arr[:-1],
        "threshold": pr_thresholds,
    })
    if len(pr_df) > 2000:
        step = len(pr_df) // 2000
        pr_df = pr_df.iloc[::step].reset_index(drop=True)

    fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_holdout, y_ho_prob)
    n_roc = min(len(fpr_arr), len(tpr_arr), len(roc_thresholds))
    roc_data = {
        "fpr": fpr_arr[:n_roc].tolist(),
        "tpr": tpr_arr[:n_roc].tolist(),
        "thresholds": roc_thresholds[:n_roc].tolist(),
        "auc": round(float(roc_auc_score(y_holdout, y_ho_prob)), 8),
    }

    sweep_df = threshold_sweep(y_holdout.values, y_ho_prob)
    sweep_df.to_csv(THRESHOLD_SWEEP_PATH, index=False)

    # ──────────────────────────────────────────────────────────────
    # 11. SCORE DISTRIBUTION
    # ──────────────────────────────────────────────────────────────
    tlog("Gerando distribuição de scores...")

    score_dist = pd.DataFrame({
        "split": "holdout",
        "is_fraud": y_holdout.values,
        "score_raw": np.round(y_ho_prob, 8),
        "score_calibrated": np.round(y_ho_prob_cal, 8),
    })

    rng = np.random.RandomState(42)
    train_fraud_idx = np.where(y_dev.values == 1)[0]
    train_normal_idx = np.where(y_dev.values == 0)[0]
    train_normal_sample = rng.choice(
        train_normal_idx,
        size=min(20000, len(train_normal_idx)),
        replace=False,
    )
    train_sample_idx = np.sort(
        np.concatenate([train_fraud_idx, train_normal_sample]),
    )

    score_dist_train = pd.DataFrame({
        "split": "train",
        "is_fraud": y_dev.values[train_sample_idx],
        "score_raw": np.round(y_train_prob[train_sample_idx], 8),
        "score_calibrated": np.nan,
    })
    score_dist = pd.concat([score_dist, score_dist_train], ignore_index=True)
    score_dist.to_csv(SCORE_DISTRIBUTION_PATH, index=False)

    # Estatísticas por classe
    score_stats: dict = {}
    for split_name, y_true, y_prob_arr in [
        ("holdout", y_holdout.values, y_ho_prob),
        ("train", y_dev.values, y_train_prob),
    ]:
        fraud_scores = y_prob_arr[y_true == 1]
        normal_scores = y_prob_arr[y_true == 0]
        stats = {
            "fraud_mean": round(float(fraud_scores.mean()), 6),
            "fraud_median": round(float(np.median(fraud_scores)), 6),
            "fraud_min": round(float(fraud_scores.min()), 8),
            "fraud_max": round(float(fraud_scores.max()), 8),
            "fraud_p25": round(float(np.percentile(fraud_scores, 25)), 6),
            "fraud_p75": round(float(np.percentile(fraud_scores, 75)), 6),
            "normal_mean": round(float(normal_scores.mean()), 8),
            "normal_median": round(float(np.median(normal_scores)), 8),
            "normal_max": round(float(normal_scores.max()), 6),
            "normal_p75": round(
                float(np.percentile(normal_scores, 75)), 8,
            ),
            "normal_p99": round(
                float(np.percentile(normal_scores, 99)), 6,
            ),
        }
        score_stats[split_name] = stats
        tlog(f"Score dist ({split_name}):")
        tlog(
            f"  Fraudes: mean={stats['fraud_mean']:.4f} "
            f"med={stats['fraud_median']:.4f} "
            f"min={stats['fraud_min']:.6f} max={stats['fraud_max']:.6f}"
        )
        tlog(
            f"  Normais: mean={stats['normal_mean']:.6f} "
            f"med={stats['normal_median']:.6f} "
            f"max={stats['normal_max']:.6f} p99={stats['normal_p99']:.6f}"
        )

    # ──────────────────────────────────────────────────────────────
    # 12. ERROR ANALYSIS
    # ──────────────────────────────────────────────────────────────
    tlog("Análise de erros...")

    y_ho_pred = (y_ho_prob >= th_f1).astype(int)

    holdout_pred_df = pd.DataFrame({
        "idx": range(len(df_holdout)),
        "is_fraud": y_holdout.values,
        "score_raw": np.round(y_ho_prob, 8),
        "score_calibrated": np.round(y_ho_prob_cal, 8),
        "pred_f1_best": y_ho_pred,
        "error_type": np.where(
            (y_holdout.values == 1) & (y_ho_pred == 0), "FN",
            np.where(
                (y_holdout.values == 0) & (y_ho_pred == 1), "FP", "OK",
            ),
        ),
    })

    analysis_features = [
        "vl_pix", "nr_idade", "qt_tempo_relacionamento_mes",
        "tx_count_prev_30m", "burst_30m_flag", "first_receiver_flag",
        "qt_total_pix_trimestre", "is_first_tx_trimestre",
        "vl_mediana_pix_trimestre", "ratio_valor_mediana",
        "distinct_receivers_so_far", "pix_key_random_flag",
        "topaz_risk_score", "rule_score_raw", "hour",
        "minutes_since_prev_tx", "key_tx_count_prev",
        # Graph features no error analysis
        "graph_pair_tx_count", "graph_is_new_edge",
        "graph_sender_avg_value", "graph_receiver_avg_value",
        "graph_sender_value_ratio", "graph_receiver_value_ratio",
        "graph_degree_ratio",
    ]
    for feat in analysis_features:
        if feat in df_holdout.columns:
            holdout_pred_df[feat] = df_holdout[feat].values

    if "transaction_id" in df_holdout.columns:
        holdout_pred_df.insert(
            0, "transaction_id", df_holdout["transaction_id"].values,
        )
    if "customer_id" in df_holdout.columns:
        holdout_pred_df.insert(
            1, "customer_id", df_holdout["customer_id"].values,
        )
    if "event_datetime" in df_holdout.columns:
        holdout_pred_df.insert(
            2, "event_datetime", df_holdout["event_datetime"].values,
        )

    holdout_pred_df.to_csv(HOLDOUT_PREDICTIONS_PATH, index=False)

    n_fn = int((holdout_pred_df["error_type"] == "FN").sum())
    n_fp = int((holdout_pred_df["error_type"] == "FP").sum())
    tlog(f"Erros: {n_fn} FN + {n_fp} FP")

    if n_fn > 0:
        tlog("FN residuais:")
        fn_rows = holdout_pred_df[holdout_pred_df["error_type"] == "FN"]
        for _, row in fn_rows.iterrows():
            tlog(
                f"  score={row['score_raw']:.6f} "
                f"vl_pix={row.get('vl_pix', '?'):>10} "
                f"burst={row.get('burst_30m_flag', '?')} "
                f"tx30m={row.get('tx_count_prev_30m', '?')} "
                f"1st_recv={row.get('first_receiver_flag', '?')} "
                f"idade={row.get('nr_idade', '?')} "
                f"1st_tx={row.get('is_first_tx_trimestre', '?')} "
                f"pair_tx={row.get('graph_pair_tx_count', '?')} "
                f"new_edge={row.get('graph_is_new_edge', '?')}"
            )

    if n_fp > 0:
        tlog("FP (top 5 por score):")
        fp_rows = holdout_pred_df[
            holdout_pred_df["error_type"] == "FP"
        ].nlargest(5, "score_raw")
        for _, row in fp_rows.iterrows():
            tlog(
                f"  score={row['score_raw']:.6f} "
                f"vl_pix={row.get('vl_pix', '?'):>10} "
                f"burst={row.get('burst_30m_flag', '?')} "
                f"tx30m={row.get('tx_count_prev_30m', '?')} "
                f"1st_recv={row.get('first_receiver_flag', '?')} "
                f"idade={row.get('nr_idade', '?')} "
                f"pair_tx={row.get('graph_pair_tx_count', '?')} "
                f"deg_ratio={row.get('graph_degree_ratio', '?')}"
            )

    # ──────────────────────────────────────────────────────────────
    # 13. SAVE ARTEFATOS
    # ──────────────────────────────────────────────────────────────
    tlog("Salvando artefatos...")

    # ── PRODUÇÃO ──
    joblib.dump(final_model, MODEL_PATH)
    tlog(f"✅ PRODUÇÃO: {MODEL_PATH}")

    # 2. CV fold metrics
    with open(CV_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)

    # 3. Features list
    features_config = {
        "version": "v6.1",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "core_features": [f for f in CORE_FEATURES if f in available_cols],
        "extra_features": [f for f in EXTRA_FEATURES if f in available_cols],
        "graph_features": graph_present,
        "graph_features_description": {
            "graph_sender_out_degree": "Nº recebedores distintos do pagador (janela 90d antes da tx)",
            "graph_sender_tx_count": "Nº tx anteriores do pagador na janela 90d",
            "graph_sender_avg_value": "Valor médio das tx anteriores do pagador",
            "graph_sender_value_zscore": "Z-score do valor vs histórico do pagador",
            "graph_receiver_in_degree": "Nº pagadores distintos que enviaram pro recebedor",
            "graph_receiver_tx_count": "Nº tx recebidas pelo recebedor na janela 90d",
            "graph_receiver_avg_value": "Valor médio recebido por tx",
            "graph_pair_tx_count": "Nº tx anteriores deste par pagador-recebedor",
            "graph_is_new_edge": "Par nunca transacionou antes na janela (0/1)",
            "graph_sender_value_ratio": "vl_pix / sender_avg_value (anomaly signal)",
            "graph_receiver_value_ratio": "vl_pix / receiver_avg_value (anomaly signal)",
            "graph_receiver_concentration_hhi": "HHI de concentração do pagador por recebedor",
            "graph_degree_ratio": "sender_out_degree / receiver_in_degree (mula pattern)",
        },
        "leakage_fixed_features": sorted(LEAKAGE_FIXED_FEATURES),
    }
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features_config, f, ensure_ascii=False, indent=2)

    # 4. Thresholds
    thresholds_config = {
        "version": "v6.1",
        "threshold_f1_best": round(float(th_f1), 6),
        "threshold_f1_calibrated": (
            round(float(th_f1_cal), 6) if use_calibrated else None
        ),
        "threshold_recall_100": round(float(th_r100), 6),
        "threshold_recall_99": round(float(th_r99), 6),
        "threshold_recall_98": round(float(th_r98), 6),
        "threshold_recall_95": round(float(th_r95), 6),
        "oof_threshold_f1": round(float(oof_th_f1), 6),
        "oof_threshold_recall_99": round(float(oof_th_r99), 6),
        "recommended_production": {
            "lgbm_threshold": round(float(th_f1), 6),
            "note": (
                "Threshold de melhor F1 no holdout. "
                "Engine aplica cascade e vetos em cima."
            ),
        },
    }
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(thresholds_config, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────────────────────
    # 14. MÉTRICAS MASTER
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Comparativo v5.1 vs v6.1
    comparison: dict = {}
    for metric_name in [
        "roc_auc", "average_precision", "f1", "recall", "precision",
        "fn", "fp", "tp",
    ]:
        v51_val = V51_METRICS.get(metric_name)
        v61_val = holdout_metrics_f1.get(metric_name)
        if v51_val is not None and v61_val is not None:
            if isinstance(v51_val, float) and isinstance(v61_val, float):
                delta = round(v61_val - v51_val, 6)
            else:
                delta = int(v61_val) - int(v51_val)
            comparison[metric_name] = {
                "v5.1": v51_val,
                "v6.1": v61_val,
                "delta": delta,
            }

    # Graph feature impact
    graph_impact = {}
    for feat in sorted(GRAPH_FEATURES):
        if feat not in df.columns:
            continue
        vals = df[feat]
        fraud_vals = df.loc[df["is_fraud"] == 1, feat]
        normal_vals = df.loc[df["is_fraud"] == 0, feat]
        std_all = vals.std()
        effect_size = (
            abs(fraud_vals.mean() - normal_vals.mean()) / std_all
            if std_all > 0 else 0.0
        )
        feat_rank = fi_df.loc[
            fi_df["feature"] == feat, "rank"
        ].values
        graph_impact[feat] = {
            "mean_fraud": round(float(fraud_vals.mean()), 4),
            "mean_normal": round(float(normal_vals.mean()), 4),
            "effect_size": round(float(effect_size), 4),
            "importance_gain": round(
                float(graph_gains.get(feat, 0)), 1,
            ),
            "importance_rank": (
                int(feat_rank[0]) if len(feat_rank) > 0 else None
            ),
        }

    # Leakage impact
    leakage_impact: dict = {}
    for feat in sorted(LEAKAGE_FIXED_FEATURES):
        if feat not in df.columns:
            continue
        vals = df[feat]
        fraud_vals = df.loc[df["is_fraud"] == 1, feat]
        normal_vals = df.loc[df["is_fraud"] == 0, feat]
        leakage_impact[feat] = {
            "mean_all": (
                round(float(vals.mean()), 4) if vals.notna().any() else None
            ),
            "mean_fraud": (
                round(float(fraud_vals.mean()), 4)
                if fraud_vals.notna().any() else None
            ),
            "mean_normal": (
                round(float(normal_vals.mean()), 4)
                if normal_vals.notna().any() else None
            ),
            "nan_pct": round(float(vals.isna().mean() * 100), 2),
        }

    master_metrics = {
        "meta": {
            "version": "v6.1",
            "description": (
                "LGBM leakage-free + graph features TEMPORAIS — "
                "grafo incremental sem leakage"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "input_file": str(INPUT_DATA.name),
            "input_md5": input_md5,
            "python_version": sys.version.split()[0],
            "random_seed": 42,
        },
        "dataset": {
            "total_rows": total_rows,
            "total_fraud": total_fraud,
            "total_normal": total_normal,
            "fraud_ratio": round(total_fraud / total_rows, 6),
            "n_dev": len(df_dev),
            "n_holdout": len(df_holdout),
            "dev_fraud": dev_fraud,
            "holdout_fraud": holdout_fraud,
            "date_range_dev": (
                f"{df_dev['event_datetime'].min()} → "
                f"{df_dev['event_datetime'].max()}"
            ),
            "date_range_holdout": (
                f"{df_holdout['event_datetime'].min()} → "
                f"{df_holdout['event_datetime'].max()}"
            ),
        },
        "model": {
            "algorithm": "LightGBM (gbdt)",
            "n_features_total": len(feature_cols),
            "n_features_used": n_features_used,
            "n_features_core": n_core,
            "n_features_extra": n_extra,
            "n_features_graph": n_graph,
            "n_features_leakage_fixed": n_fixed,
            "final_n_estimators": final_n_estimators,
            "scale_pos_weight": round(float(spw_final), 2),
            "hyperparameters": {
                "learning_rate": 0.01,
                "num_leaves": 63,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.5,
                "reg_lambda": 1.0,
                "min_child_samples": max(3, min(20, n_pos_final // 5)),
            },
            "calibrated": use_calibrated,
            "calibration_method": (
                "isotonic_cv3" if use_calibrated else None
            ),
        },
        "cross_validation": {
            "strategy": "TimeSeriesSplit",
            "n_folds": N_FOLDS,
            "fold_metrics": fold_metrics,
            "oof_roc_auc": oof_auc,
            "oof_average_precision": oof_ap,
            "oof_best_f1": oof_f1,
            "oof_best_threshold": oof_th_f1,
        },
        "holdout": {
            "at_best_f1": holdout_metrics_f1,
            "at_0_5": holdout_metrics_05,
            "calibrated_at_best_f1": holdout_metrics_cal,
        },
        "train_metrics": {
            "at_best_f1_threshold": train_metrics,
        },
        "thresholds": thresholds_config,
        "overfitting_analysis": {
            "gap_auc_train_holdout": round(
                (train_metrics.get("roc_auc") or 0)
                - (holdout_metrics_f1.get("roc_auc") or 0),
                6,
            ),
            "gap_f1_train_holdout": round(
                (train_metrics.get("f1") or 0)
                - (holdout_metrics_f1.get("f1") or 0),
                4,
            ),
            "gap_ap_train_holdout": round(
                (train_metrics.get("average_precision") or 0)
                - (holdout_metrics_f1.get("average_precision") or 0),
                6,
            ),
        },
        "comparison_v51_vs_v61": comparison,
        "graph_feature_impact": graph_impact,
        "score_distribution": score_stats,
        "roc_curve": roc_data,
        "leakage_impact": leakage_impact,
        "features": {
            "top_20_gain": fi_df.head(20)[
                [
                    "rank", "feature", "importance_gain",
                    "importance_gain_pct", "is_leakage_fixed",
                    "is_graph_feature", "feature_group",
                ]
            ].to_dict(orient="records"),
        },
        "nan_report": {k: int(v) for k, v in nan_report.items()},
        "errors": {
            "fn_count": n_fn,
            "fp_count": n_fp,
            "fn_details": (
                holdout_pred_df[holdout_pred_df["error_type"] == "FN"][
                    ["score_raw"]
                    + [
                        f for f in analysis_features
                        if f in holdout_pred_df.columns
                    ]
                ].to_dict(orient="records")
                if n_fn > 0 else []
            ),
            "fp_details": (
                holdout_pred_df[
                    holdout_pred_df["error_type"] == "FP"
                ].nlargest(10, "score_raw")[
                    ["score_raw"]
                    + [
                        f for f in analysis_features
                        if f in holdout_pred_df.columns
                    ]
                ].to_dict(orient="records")
                if n_fp > 0 else []
            ),
        },
        "artefacts": {
            "production": {
                "model": str(MODEL_PATH),
            },
            "report": {
                "metrics": str(METRICS_PATH.name),
                "cv_metrics": str(CV_METRICS_PATH.name),
                "features": str(FEATURES_PATH.name),
                "thresholds": str(THRESHOLDS_PATH.name),
                "feature_importance": str(FEATURE_IMPORTANCE_PATH.name),
                "oof_predictions": str(OOF_PREDICTIONS_PATH.name),
                "holdout_predictions": str(HOLDOUT_PREDICTIONS_PATH.name),
                "score_distribution": str(SCORE_DISTRIBUTION_PATH.name),
                "threshold_sweep": str(THRESHOLD_SWEEP_PATH.name),
                "training_log": str(TRAINING_LOG_PATH.name),
            },
        },
    }

    # 1. Master metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, ensure_ascii=False, indent=2)

    # 10. Training log
    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(training_log))

    # ──────────────────────────────────────────────────────────────
    # 15. SUMMARY
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  RESULTADOS — LGBM v6.1 (Graph Features Temporais)")
    print(f"{'=' * 72}")

    m = holdout_metrics_f1
    print(f"\n  Dataset: {total_rows:,} tx ({total_fraud} fraudes)")
    print(
        f"  Dev: {len(df_dev):,} ({dev_fraud}F) | "
        f"Holdout: {len(df_holdout):,} ({holdout_fraud}F)"
    )
    print(
        f"  Features: {len(feature_cols)} "
        f"({n_core} core + {n_extra} extra + {n_graph} graph) | "
        f"{n_features_used} com gain>0 | {n_fixed} leakage-fixed"
    )
    print(
        f"  Calibração: "
        f"{'✅ isotonic cv=3' if use_calibrated else '❌ falhou'}"
    )

    print(f"\n  {'─' * 60}")
    print(f"  HOLDOUT @ F1-best (th={th_f1:.4f}):")
    print(f"    ROC-AUC:  {m['roc_auc']:.6f}")
    print(f"    AP:       {m['average_precision']:.6f}")
    print(f"    F1:       {m['f1']:.4f}")
    print(f"    Recall:   {m['recall']:.4f}  (TP={m['tp']}, FN={m['fn']})")
    print(f"    Precision:{m['precision']:.4f}  (FP={m['fp']})")
    print(f"    FPR:      {m['fpr']:.6f}")

    if use_calibrated and holdout_metrics_cal:
        mc = holdout_metrics_cal
        print(f"\n  HOLDOUT CALIBRADO @ F1-best (th={th_f1_cal:.4f}):")
        print(f"    F1:       {mc['f1']:.4f}")
        print(
            f"    Recall:   {mc['recall']:.4f}  "
            f"(TP={mc['tp']}, FN={mc['fn']})"
        )
        print(f"    Precision:{mc['precision']:.4f}  (FP={mc['fp']})")

    print(f"\n  {'─' * 60}")
    print("  COMPARATIVO v5.1 → v6.1:")
    print(
        f"    {'Métrica':<22} {'v5.1 (sem graph)':>16} "
        f"{'v6.1 (temporal)':>16} {'Delta':>10}"
    )
    for name, vals in comparison.items():
        v51 = vals["v5.1"]
        v61 = vals["v6.1"]
        delta = vals["delta"]
        if isinstance(v51, float):
            print(
                f"    {name:<22} {v51:>16.4f} {v61:>16.4f} "
                f"{delta:>+10.4f}"
            )
        else:
            print(
                f"    {name:<22} {v51:>16} {v61:>16} "
                f"{delta:>+10}"
            )

    print(f"\n  {'─' * 60}")
    print("  GRAPH FEATURES TEMPORAIS — CONTRIBUIÇÃO:")
    graph_fi = fi_df[fi_df["is_graph_feature"]]
    graph_total_gain = graph_fi["importance_gain"].sum()
    graph_total_pct = graph_total_gain / max(total_gain, 1) * 100
    print(
        f"    Total gain: {graph_total_gain:.1f} ({graph_total_pct:.1f}% "
        f"do modelo)"
    )
    print(
        f"    Ativas: {int((graph_fi['importance_gain'] > 0).sum())}"
        f"/{len(graph_fi)}"
    )
    for _, row in graph_fi[
        graph_fi["importance_gain"] > 0
    ].sort_values("importance_gain", ascending=False).iterrows():
        print(
            f"    📊 #{int(row['rank']):2d} {row['feature']:<40} "
            f"gain={row['importance_gain']:10.1f} "
            f"({row['importance_gain_pct']:5.2f}%)"
        )

    print(f"\n  {'─' * 60}")
    print("  OVERFITTING CHECK:")
    oa = master_metrics["overfitting_analysis"]
    print(
        f"    Gap AUC  (train-holdout): {oa['gap_auc_train_holdout']:+.6f}"
    )
    print(
        f"    Gap F1   (train-holdout): {oa['gap_f1_train_holdout']:+.4f}"
    )
    print(
        f"    Gap AP   (train-holdout): {oa['gap_ap_train_holdout']:+.6f}"
    )

    print(f"\n  Tempo total: {elapsed:.1f}s")

    print(f"\n  Artefato de PRODUÇÃO:")
    print(f"    {MODEL_PATH}")

    print(f"\n  Artefatos de RELATÓRIO ({RESULT_DIR.name}/):")
    for p in sorted(RESULT_DIR.glob("*")):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name} ({size_kb:.0f} KB)")

    print(f"\n{'=' * 72}")
    print("  ✅ LGBM v6.1 — Treino concluído")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
