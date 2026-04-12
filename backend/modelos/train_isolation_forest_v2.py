"""
train_isolation_forest_v3.py — IF Otimizado: Features Limpas + Separação Melhorada

Mudanças v2 → v3:
  1. FEATURES: 22 → 12 (removidas 10 com importance negativa/NaN alto)
  2. HYPERPARAMS: fixos do v2 (800 trees, 0.8 samples, 0.7 features)
     Apenas contamination é testado (3 valores)
  3. SEPARAÇÃO: log_vl_pix + treino segmentado (só normais regulares no fit)
  4. NOVAS FEATURES: log_vl_pix, tx_count_prev_1h (se disponível)
  5. TREINO SEGMENTADO: fit em normais COM histórico (não 1ªTX)
     → IF aprende "comportamento normal estabelecido"
     → 1ªTX são intrinsecamente anômalas → scores mais altos → melhor separação
"""

import os
import sys
import json
import time
import hashlib
import logging
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_DIR = os.path.join(PROJECT_ROOT, "dados")
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "backend", "artefatos")
REPORT_DIR = os.path.join(PROJECT_ROOT, "backend", "modelos", "resultado_treino_if")

INPUT_DATA = os.path.join(DADOS_DIR, "base_treino_final.csv")
LGBM_HOLDOUT_PATH = os.path.join(
    PROJECT_ROOT, "backend", "modelos", "resultado_treino_lgbm", "holdout_predictions.csv"
)

# Produção
MODEL_PATH = os.path.join(ARTEFACT_DIR, "model_isolation_forest.joblib")
SCALER_PATH = os.path.join(ARTEFACT_DIR, "scaler_isolation_forest.joblib")
REF_RAW_PATH = os.path.join(ARTEFACT_DIR, "if_ref_raw_train.npy")

# Relatório (10 artefatos)
METRICS_PATH = os.path.join(REPORT_DIR, "metricas_if.json")
CONFIG_PATH = os.path.join(REPORT_DIR, "isolation_forest_config.json")
FEAT_IMP_PATH = os.path.join(REPORT_DIR, "feature_importance.csv")
HOLDOUT_PREDS_PATH = os.path.join(REPORT_DIR, "holdout_predictions.csv")
SCORE_DIST_PATH = os.path.join(REPORT_DIR, "score_distribution.csv")
ERROR_ANALYSIS_PATH = os.path.join(REPORT_DIR, "error_analysis.csv")
COMPLEMENTARITY_PATH = os.path.join(REPORT_DIR, "complementarity_analysis.csv")
HYPERPARAM_PATH = os.path.join(REPORT_DIR, "contamination_search.csv")
THRESHOLD_SWEEP_PATH = os.path.join(REPORT_DIR, "threshold_sweep.csv")
TRAINING_LOG_PATH = os.path.join(REPORT_DIR, "training_log.txt")

os.makedirs(ARTEFACT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Hiperparâmetros fixos do v2 (melhores encontrados)
FIXED_PARAMS = {
    "n_estimators": 800,
    "max_samples": 0.8,
    "max_features": 0.7,
}
# Só contamination é buscado
CONTAMINATION_GRID = [0.005, 0.01, 0.02, 0.03]


# ─── Tee logger ───────────────────────────────────────────
class TeeLogger:
    """Duplica stdout para arquivo."""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


tee = TeeLogger(TRAINING_LOG_PATH)
sys.stdout = tee


# ═══════════════════════════════════════════════════════════
# FEATURES v3 — Apenas as 12 com importance positiva + log_vl_pix
# ═══════════════════════════════════════════════════════════

# Removidas (importance negativa ou NaN alto):
#   qt_total_pix_trimestre (-0.049), distinct_receivers_so_far (-0.045),
#   is_first_tx_trimestre (-0.015), ratio_valor_desvio_padrao (-0.012),
#   ratio_valor_mediana (-0.010), zscore_valor_aprox (-0.010),
#   valor_over_trimestre_avg (-0.007), qt_pix_dia_maximo_trimestre (-0.004),
#   hour (-0.004), valor_x_first_recv (-0.001)

FEATURES_CORE = [
    # Velocity/Burst — top performers (imp > 0.07)
    "burst_30m_flag",           # #1: 0.0785
    "tx_count_prev_30m",        # #2: 0.0727

    # Context — strong signal (imp > 0.01)
    "first_receiver_flag",      # #3: 0.0331
    "topaz_risk_score",         # #6: 0.0175
    "rule_score_raw",           # #10: 0.0044

    # Profile (imp > 0.003)
    "qt_tempo_relacionamento_mes",  # #7: 0.0171
    "nr_idade",                 # #11: 0.0034

    # Value (imp > 0.015)
    "vl_pix",                   # #8: 0.0158

    # Timing
    "minutes_since_prev_tx",    # #12: 0.0029
]

FEATURES_INTERACTION = [
    # Interações com importance positiva
    "burst_x_distinct_recv",    # #4: 0.0331
    "valor_x_burst",            # #5: 0.0184
    "idade_x_first_recv",       # #9: 0.0103
]

# Feature nova: log do valor (melhora separação em valores extremos)
FEATURES_ENGINEERED = [
    "log_vl_pix",               # log1p(vl_pix) — comprime outliers
]

IF_FEATURES_V3 = FEATURES_CORE + FEATURES_INTERACTION + FEATURES_ENGINEERED


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def anomaly_score_percentile(
    raw_scores: np.ndarray,
    ref_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Converte decision_function → percentil [0,1]. Mais alto = mais anômalo."""
    inverted = -raw_scores
    ref = -ref_scores if ref_scores is not None else inverted
    return np.array([np.mean(ref <= v) for v in inverted])


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_ratio: float) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    k = max(1, int(len(y_true) * k_ratio))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_ratio: float) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    k = max(1, int(len(y_true) * k_ratio))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / positives)


def recall_at_top_n(y_true: np.ndarray, y_score: np.ndarray, n: int) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    k = min(n, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / positives)


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else None,
        "average_precision": float(average_precision_score(y_true, y_score)) if has_both else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "precision_at_1pct": precision_at_k(y_true, y_score, 0.01),
        "recall_at_1pct": recall_at_k(y_true, y_score, 0.01),
        "precision_at_5pct": precision_at_k(y_true, y_score, 0.05),
        "recall_at_5pct": recall_at_k(y_true, y_score, 0.05),
        "recall_at_top_50": recall_at_top_n(y_true, y_score, 50),
        "recall_at_top_100": recall_at_top_n(y_true, y_score, 100),
        "recall_at_top_200": recall_at_top_n(y_true, y_score, 200),
    }


def find_best_threshold_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    step: float = 0.005,
) -> tuple[float, float]:
    best_th, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, step):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, t
    return float(best_th), float(best_f1)


def find_threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: float,
) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if y_true.sum() == 0:
        return 0.5
    for t in np.arange(0.95, 0.001, -0.001):
        rec = recall_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if rec >= min_recall:
            return float(t)
    return 0.001


def create_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de interação e engineered para IF v3."""
    df = df.copy()
    tx_30m = df["tx_count_prev_30m"].fillna(0)

    # Interações (mantidas do v2 — importance positiva)
    df["burst_x_distinct_recv"] = tx_30m * df["distinct_receivers_so_far"].fillna(1)
    df["valor_x_burst"] = df["vl_pix"].fillna(0) * (tx_30m + 1)
    df["idade_x_first_recv"] = df["nr_idade"].fillna(0) * df["first_receiver_flag"].fillna(0)

    # Nova: log do valor — comprime outliers, melhora separação
    df["log_vl_pix"] = np.log1p(df["vl_pix"].fillna(0).clip(lower=0))

    return df


def compute_permutation_importance(
    model: IsolationForest,
    scaler: RobustScaler,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    ref_raw: np.ndarray,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Feature importance via permutation (drop em AP)."""
    X_scaled = scaler.transform(X)
    base_raw = model.decision_function(X_scaled)
    base_scores = anomaly_score_percentile(base_raw, ref_raw)

    if y.sum() == 0 or len(np.unique(y)) < 2:
        return pd.DataFrame({
            "feature": feature_names,
            "importance_mean": 0.0,
            "importance_std": 0.0,
        })

    base_ap = average_precision_score(y, base_scores)
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []

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
        rows.append({
            "feature": feat,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

t0 = time.time()

print("=" * 72)
print("  TREINO Isolation Forest v3 — Features Limpas + Separação Otimizada")
print("=" * 72)

# ─── Load ──────────────────────────────────────────────────
log.info(f"Input: {INPUT_DATA}")
input_md5 = file_md5(INPUT_DATA)
log.info(f"MD5: {input_md5}")

df = pd.read_csv(INPUT_DATA)
df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
df = df.sort_values("event_datetime").reset_index(drop=True)
df = df[df["is_fraud"].notna()].copy()
df["is_fraud"] = df["is_fraud"].astype(int)

if "hour" not in df.columns:
    df["hour"] = df["event_datetime"].dt.hour

log.info(f"Shape: {df.shape}")
log.info(f"Fraudes: {df['is_fraud'].sum()} | Normais: {(df['is_fraud'] == 0).sum()}")

# ─── Feature engineering v3 ───────────────────────────────
df = create_features_v3(df)

available = [f for f in IF_FEATURES_V3 if f in df.columns]
missing = [f for f in IF_FEATURES_V3 if f not in df.columns]
if missing:
    log.warning(f"Features não encontradas: {missing}")
IF_FEATURES_FINAL = available
log.info(f"\nFeatures v3: {len(IF_FEATURES_FINAL)} (v2 tinha 22)")
log.info("Removidas por importance negativa:")
log.info("  qt_total_pix_trimestre, distinct_receivers_so_far, is_first_tx_trimestre,")
log.info("  ratio_valor_desvio_padrao, ratio_valor_mediana, zscore_valor_aprox,")
log.info("  valor_over_trimestre_avg, qt_pix_dia_maximo_trimestre, hour, valor_x_first_recv")
log.info("Adicionada: log_vl_pix (log1p do valor)")
log.info("")
for i, f in enumerate(IF_FEATURES_FINAL):
    log.info(f"  {i + 1:2d}. {f}")


# ─── Split temporal IDÊNTICO ao LGBM v5.1 ─────────────────
holdout_ratio = 0.10
n = len(df)
holdout_start = int(n * (1 - holdout_ratio))

df_dev = df.iloc[:holdout_start].copy().reset_index(drop=True)
df_holdout = df.iloc[holdout_start:].copy().reset_index(drop=True)

dev_train_end = int(len(df_dev) * 0.70)
df_train = df_dev.iloc[:dev_train_end].copy().reset_index(drop=True)
df_val = df_dev.iloc[dev_train_end:].copy().reset_index(drop=True)

log.info("\n--- Split Temporal ---")
log.info(
    f"  Treino: {len(df_train)} rows | {df_train['is_fraud'].sum()} fraudes | "
    f"{(df_train['is_first_tx_trimestre'] == 1).sum()} 1ªTX"
)
log.info(
    f"  Val:    {len(df_val)} rows | {df_val['is_fraud'].sum()} fraudes | "
    f"{(df_val['is_first_tx_trimestre'] == 1).sum()} 1ªTX"
)
log.info(
    f"  Holdout:{len(df_holdout)} rows | {df_holdout['is_fraud'].sum()} fraudes | "
    f"{(df_holdout['is_first_tx_trimestre'] == 1).sum()} 1ªTX"
)
log.info(
    f"  Dev date range: {df_dev['event_datetime'].min()} → {df_dev['event_datetime'].max()}"
)
log.info(
    f"  Holdout date range: {df_holdout['event_datetime'].min()} → {df_holdout['event_datetime'].max()}"
)


# ─── Preparar dados ──────────────────────────────────────
# MUDANÇA v3: treinar apenas com normais que TÊM histórico (não 1ªTX)
# Isso faz o IF aprender "comportamento normal estabelecido"
# Consequência: 1ªTX serão mais anômalas → fraudes em 1ªTX ficam com score mais alto
train_all_normal = df_train[df_train["is_fraud"] == 0].copy()
train_regular_normal = train_all_normal[train_all_normal["is_first_tx_trimestre"] == 0].copy()

n_first_tx_excluded = len(train_all_normal) - len(train_regular_normal)
log.info(f"\n--- Treino Segmentado (v3) ---")
log.info(f"  Normais totais: {len(train_all_normal)}")
log.info(f"  Normais regulares (com histórico): {len(train_regular_normal)}")
log.info(f"  1ªTX excluídas do fit: {n_first_tx_excluded}")
log.info(f"  Ratio: {len(train_regular_normal)/len(train_all_normal)*100:.1f}% das normais")

# Se houver poucas regulares, fallback para todas
if len(train_regular_normal) < 1000:
    log.warning("Poucas tx regulares — usando todas as normais")
    train_for_fit = train_all_normal
    train_strategy = "all_normal"
else:
    train_for_fit = train_regular_normal
    train_strategy = "regular_normal_only"

log.info(f"  Estratégia de treino: {train_strategy}")

X_train = train_for_fit[IF_FEATURES_FINAL].copy()
X_val = df_val[IF_FEATURES_FINAL].copy()
X_holdout = df_holdout[IF_FEATURES_FINAL].copy()
y_val = df_val["is_fraud"].values.copy()
y_holdout = df_holdout["is_fraud"].values.copy()

# NaN report
nan_counts = X_train.isna().sum()
nan_feats = nan_counts[nan_counts > 0]
if len(nan_feats) > 0:
    log.info("\nFeatures com NaN no treino:")
    for feat, cnt in nan_feats.items():
        log.info(f"  {feat}: {cnt} ({cnt / len(X_train) * 100:.1f}%)")
else:
    log.info("\n✅ Zero NaN nas features v3 (features com NaN alto foram removidas)")

# Imputação
medians = X_train.median()
X_train = X_train.fillna(medians)
X_val = X_val.fillna(medians)
X_holdout = X_holdout.fillna(medians)

# Infinitos
for col in X_train.columns:
    med = medians.get(col, 0)
    X_train[col] = X_train[col].replace([np.inf, -np.inf], med)
    X_val[col] = X_val[col].replace([np.inf, -np.inf], med)
    X_holdout[col] = X_holdout[col].replace([np.inf, -np.inf], med)

# Escalar
log.info("Escalando (RobustScaler)...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_holdout_scaled = scaler.transform(X_holdout)


# ═══════════════════════════════════════════════════════════
# CONTAMINATION SEARCH (hiperparams fixos, só contamination varia)
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  CONTAMINATION SEARCH (params fixos do v2)")
print(f"{'=' * 72}")

log.info(f"Params fixos: {FIXED_PARAMS}")
log.info(f"Contamination grid: {CONTAMINATION_GRID}")

search_results = []
best_contam = None
best_val_ap = -1.0
best_model = None
best_ref_raw = None

for contam in CONTAMINATION_GRID:
    log.info(f"\n  contamination={contam}...")

    iforest = IsolationForest(
        n_estimators=FIXED_PARAMS["n_estimators"],
        max_samples=FIXED_PARAMS["max_samples"],
        max_features=FIXED_PARAMS["max_features"],
        contamination=contam,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train_scaled)

    ref_raw = iforest.decision_function(X_train_scaled)
    raw_val = iforest.decision_function(X_val_scaled)
    scores_val = anomaly_score_percentile(raw_val, ref_raw)

    ap = average_precision_score(y_val, scores_val) if y_val.sum() > 0 else 0.0
    auc = roc_auc_score(y_val, scores_val) if y_val.sum() > 0 else 0.0
    r_5 = recall_at_k(y_val, scores_val, 0.05)
    r_top100 = recall_at_top_n(y_val, scores_val, 100)

    # Score distribution na validação
    fraud_scores = scores_val[y_val == 1]
    normal_scores = scores_val[y_val == 0]
    gap = np.median(fraud_scores) - np.median(normal_scores) if len(fraud_scores) > 0 else 0

    search_results.append({
        "contamination": contam,
        "val_ap": ap,
        "val_auc": auc,
        "val_recall_5pct": r_5,
        "val_recall_top100": r_top100,
        "fraud_median": float(np.median(fraud_scores)) if len(fraud_scores) > 0 else None,
        "normal_median": float(np.median(normal_scores)),
        "normal_p75": float(np.percentile(normal_scores, 75)),
        "normal_p99": float(np.percentile(normal_scores, 99)),
        "median_gap": gap,
    })

    log.info(f"    AP={ap:.4f} AUC={auc:.4f} R@5%={r_5:.4f} R@Top100={r_top100:.4f}")
    log.info(
        f"    Fraud med={np.median(fraud_scores):.4f} | "
        f"Normal med={np.median(normal_scores):.4f} p75={np.percentile(normal_scores, 75):.4f} "
        f"p99={np.percentile(normal_scores, 99):.4f} | Gap={gap:.4f}"
    )

    if ap > best_val_ap:
        best_val_ap = ap
        best_contam = contam
        best_model = iforest
        best_ref_raw = ref_raw

search_df = pd.DataFrame(search_results).sort_values("val_ap", ascending=False)
search_df.to_csv(HYPERPARAM_PATH, index=False)

best_params = {**FIXED_PARAMS, "contamination": best_contam}
log.info(f"\n✅ Melhor contamination: {best_contam}")
log.info(f"✅ Melhor AP (validação): {best_val_ap:.4f}")
log.info(f"✅ Config final: {best_params}")

model = best_model
ref_raw_train = best_ref_raw


# ═══════════════════════════════════════════════════════════
# SCORES
# ═══════════════════════════════════════════════════════════
raw_val = model.decision_function(X_val_scaled)
scores_val = anomaly_score_percentile(raw_val, ref_raw_train)

raw_holdout = model.decision_function(X_holdout_scaled)
scores_holdout = anomaly_score_percentile(raw_holdout, ref_raw_train)

raw_train_all = model.decision_function(
    scaler.transform(
        df_train[IF_FEATURES_FINAL]
        .fillna(medians)
        .replace([np.inf, -np.inf], 0)
    )
)
scores_train = anomaly_score_percentile(raw_train_all, ref_raw_train)
y_train_all = df_train["is_fraud"].values


# ═══════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════
best_th_f1, best_f1_val = find_best_threshold_f1(y_val, scores_val, step=0.005)
th_recall_90 = find_threshold_for_recall(y_val, scores_val, 0.90)
th_recall_95 = find_threshold_for_recall(y_val, scores_val, 0.95)

log.info(f"\nThreshold F1 val: {best_th_f1:.4f} → F1={best_f1_val:.4f}")
log.info(f"Threshold Recall≥90%: {th_recall_90:.4f}")
log.info(f"Threshold Recall≥95%: {th_recall_95:.4f}")


# ═══════════════════════════════════════════════════════════
# THRESHOLD SWEEP (holdout)
# ═══════════════════════════════════════════════════════════
sweep_rows = []
for t in np.arange(0.05, 0.96, 0.005):
    y_pred = (scores_holdout >= t).astype(int)
    cm = confusion_matrix(y_holdout, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    sweep_rows.append({
        "threshold": round(float(t), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "fpr": round(fp / max(fp + tn, 1), 6),
    })
pd.DataFrame(sweep_rows).to_csv(THRESHOLD_SWEEP_PATH, index=False)


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  AVALIAÇÃO")
print(f"{'=' * 72}")

all_metrics = {}
for split, y_true, scores in [
    ("train", y_train_all, scores_train),
    ("val", y_val, scores_val),
    ("holdout", y_holdout, scores_holdout),
]:
    m_05 = evaluate_at_threshold(y_true, scores, 0.5)
    m_f1 = evaluate_at_threshold(y_true, scores, best_th_f1)
    all_metrics[f"{split}_at_0.5"] = m_05
    all_metrics[f"{split}_at_best_f1"] = m_f1

    log.info(f"\n  {split} @ 0.5:")
    if m_05["roc_auc"]:
        log.info(f"    ROC-AUC: {m_05['roc_auc']:.4f}")
    if m_05["average_precision"]:
        log.info(f"    AP:      {m_05['average_precision']:.4f}")
    log.info(f"    F1:      {m_05['f1']:.4f}  (TP={m_05['tp']} FP={m_05['fp']} FN={m_05['fn']})")
    log.info(f"    R@5%:    {m_05['recall_at_5pct']:.4f}")
    log.info(f"    R@Top100:{m_05['recall_at_top_100']:.4f}")
    log.info(f"    R@Top200:{m_05['recall_at_top_200']:.4f}")


# ─── Avaliação por segmento ───────────────────────────────
print(f"\n{'=' * 72}")
print("  AVALIAÇÃO POR SEGMENTO (Holdout)")
print(f"{'=' * 72}")

first_mask = df_holdout["is_first_tx_trimestre"].values == 1
reg_mask = ~first_mask

for seg_name, seg_mask in [("first_tx", first_mask), ("regular_tx", reg_mask)]:
    n_seg = seg_mask.sum()
    n_fraud_seg = y_holdout[seg_mask].sum()
    if n_seg > 0 and n_fraud_seg > 0:
        m_seg = evaluate_at_threshold(y_holdout[seg_mask], scores_holdout[seg_mask], 0.5)
        all_metrics[f"holdout_{seg_name}_at_0.5"] = m_seg
        log.info(f"\n  {seg_name} ({n_seg} tx, {n_fraud_seg} fraudes):")
        if m_seg["roc_auc"]:
            log.info(f"    ROC-AUC: {m_seg['roc_auc']:.4f}")
        if m_seg["average_precision"]:
            log.info(f"    AP:      {m_seg['average_precision']:.4f}")
        log.info(f"    F1:      {m_seg['f1']:.4f}  (TP={m_seg['tp']} FP={m_seg['fp']} FN={m_seg['fn']})")
        log.info(f"    R@5%:    {m_seg['recall_at_5pct']:.4f}")
        log.info(f"    R@Top100:{m_seg['recall_at_top_100']:.4f}")


# ═══════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  FEATURE IMPORTANCE (permutation)")
print(f"{'=' * 72}")

feat_imp = compute_permutation_importance(
    model, scaler,
    X_holdout.values, y_holdout,
    IF_FEATURES_FINAL,
    ref_raw=ref_raw_train,
    n_repeats=5,
)
feat_imp["rank"] = range(1, len(feat_imp) + 1)
feat_imp.to_csv(FEAT_IMP_PATH, index=False)
log.info("\n" + feat_imp.to_string(index=False))

# Verificar se alguma feature ficou negativa
negative_feats = feat_imp[feat_imp["importance_mean"] < 0]
if len(negative_feats) > 0:
    log.warning(f"\n⚠ {len(negative_feats)} features com importance negativa no v3:")
    for _, row in negative_feats.iterrows():
        log.warning(f"  {row['feature']}: {row['importance_mean']:.6f}")
else:
    log.info("\n✅ Todas as features com importance ≥ 0 — limpeza funcionou!")


# ═══════════════════════════════════════════════════════════
# HOLDOUT PREDICTIONS
# ═══════════════════════════════════════════════════════════
pred_holdout = pd.DataFrame({
    "transaction_id": df_holdout["transaction_id"].values,
    "customer_id": df_holdout["customer_id"].values,
    "event_datetime": df_holdout["event_datetime"].values,
    "is_fraud": y_holdout,
    "is_first_tx_trimestre": df_holdout["is_first_tx_trimestre"].values,
    "anomaly_score": scores_holdout,
    "anomaly_raw": raw_holdout,
    "pred_at_0_5": (scores_holdout >= 0.5).astype(int),
    "pred_at_best_f1": (scores_holdout >= best_th_f1).astype(int),
    "vl_pix": df_holdout["vl_pix"].values,
    "nr_idade": df_holdout["nr_idade"].values,
    "tx_count_prev_30m": df_holdout["tx_count_prev_30m"].values,
    "burst_30m_flag": df_holdout["burst_30m_flag"].values,
    "first_receiver_flag": df_holdout["first_receiver_flag"].values,
})

pred_holdout["error_type"] = "OK"
fn_mask = (pred_holdout["is_fraud"] == 1) & (pred_holdout["pred_at_best_f1"] == 0)
fp_mask = (pred_holdout["is_fraud"] == 0) & (pred_holdout["pred_at_best_f1"] == 1)
pred_holdout.loc[fn_mask, "error_type"] = "FN"
pred_holdout.loc[fp_mask, "error_type"] = "FP"

pred_holdout.to_csv(HOLDOUT_PREDS_PATH, index=False)


# ═══════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  ANÁLISE DE ERROS")
print(f"{'=' * 72}")

errors = pred_holdout[pred_holdout["error_type"] != "OK"].copy()
for feat in IF_FEATURES_FINAL:
    if feat in df_holdout.columns and feat not in errors.columns:
        errors[feat] = df_holdout.loc[errors.index, feat].values

errors.to_csv(ERROR_ANALYSIS_PATH, index=False)
log.info(f"Erros: {fn_mask.sum()} FN + {fp_mask.sum()} FP = {len(errors)} total")

if fn_mask.sum() > 0:
    log.info("FN residuais:")
    for _, row in pred_holdout[fn_mask].head(10).iterrows():
        log.info(
            f"  score={row['anomaly_score']:.4f} vl_pix={row['vl_pix']:>10.2f} "
            f"burst={int(row['burst_30m_flag'])} 1st_recv={int(row['first_receiver_flag'])} "
            f"idade={int(row['nr_idade'])} 1st_tx={int(row['is_first_tx_trimestre'])}"
        )


# ═══════════════════════════════════════════════════════════
# SCORE DISTRIBUTION — Comparativo v2 vs v3
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  SCORE DISTRIBUTION")
print(f"{'=' * 72}")

score_dist = pd.DataFrame({
    "split": "holdout",
    "is_fraud": y_holdout,
    "anomaly_score": scores_holdout,
    "is_first_tx": df_holdout["is_first_tx_trimestre"].values,
})
score_dist.to_csv(SCORE_DIST_PATH, index=False)

log.info("\nDistribuição de scores (holdout):")
for cls_name, cls_val in [("Fraudes", 1), ("Normais", 0)]:
    s = scores_holdout[y_holdout == cls_val]
    if len(s) > 0:
        log.info(
            f"  {cls_name}: mean={s.mean():.4f} med={np.median(s):.4f} "
            f"min={s.min():.6f} max={s.max():.6f} "
            f"p25={np.percentile(s, 25):.4f} p75={np.percentile(s, 75):.4f}"
        )

# Comparar com v2 (hardcoded para referência)
V2_FRAUD_MEDIAN = 0.9970
V2_NORMAL_MEDIAN = 0.6542
V2_NORMAL_P75 = 0.9148

v3_fraud_med = float(np.median(scores_holdout[y_holdout == 1])) if y_holdout.sum() > 0 else 0
v3_normal_med = float(np.median(scores_holdout[y_holdout == 0]))
v3_normal_p75 = float(np.percentile(scores_holdout[y_holdout == 0], 75))

log.info(f"\n  Comparativo v2 → v3:")
log.info(f"    Fraud median:  {V2_FRAUD_MEDIAN:.4f} → {v3_fraud_med:.4f} ({v3_fraud_med - V2_FRAUD_MEDIAN:+.4f})")
log.info(f"    Normal median: {V2_NORMAL_MEDIAN:.4f} → {v3_normal_med:.4f} ({v3_normal_med - V2_NORMAL_MEDIAN:+.4f})")
log.info(f"    Normal P75:    {V2_NORMAL_P75:.4f} → {v3_normal_p75:.4f} ({v3_normal_p75 - V2_NORMAL_P75:+.4f})")
log.info(f"    Gap (fraud_med - normal_p75): {V2_FRAUD_MEDIAN - V2_NORMAL_P75:.4f} → {v3_fraud_med - v3_normal_p75:.4f}")

# Score por segmento
log.info("\n  Por segmento (holdout):")
for seg_name, seg_val in [("1ªTX", 1), ("Regular", 0)]:
    seg_m = df_holdout["is_first_tx_trimestre"].values == seg_val
    for cls_name, cls_val in [("Fraude", 1), ("Normal", 0)]:
        mask = seg_m & (y_holdout == cls_val)
        s = scores_holdout[mask]
        if len(s) > 0:
            log.info(
                f"    {seg_name} {cls_name} ({len(s)}): "
                f"med={np.median(s):.4f} p25={np.percentile(s, 25):.4f} p75={np.percentile(s, 75):.4f}"
            )


# ═══════════════════════════════════════════════════════════
# COMPLEMENTARIDADE COM LGBM v5.1
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  COMPLEMENTARIDADE COM LGBM v5.1")
print(f"{'=' * 72}")

complementarity = None
comp_metrics = {}

try:
    lgbm_preds = pd.read_csv(LGBM_HOLDOUT_PATH)
    log.info(f"LGBM holdout predictions: {len(lgbm_preds)} rows")

    comp = pred_holdout[["transaction_id", "is_fraud", "anomaly_score", "is_first_tx_trimestre"]].merge(
        lgbm_preds[["transaction_id", "score_raw"]].rename(columns={"score_raw": "lgbm_score"}),
        on="transaction_id",
        how="inner",
    )
    log.info(f"Match: {len(comp)} tx")

    lgbm_s = comp["lgbm_score"].values
    if_s = comp["anomaly_score"].values
    y_comp = comp["is_fraud"].values

    LGBM_THRESHOLD = 0.27

    # ─── Estratégia A: IF boost quando LGBM < threshold ───
    lgbm_below = lgbm_s < LGBM_THRESHOLD
    if_high = if_s >= 0.70
    if_very_high = if_s >= 0.85

    boost = np.where(
        lgbm_below & if_very_high, 0.15,
        np.where(lgbm_below & if_high, 0.08, 0.0),
    )
    comp["ensemble_boost"] = np.clip(lgbm_s + boost, 0, 1)

    # ─── Estratégia B: Weighted ensemble ───────────────────
    if_weight = np.where(lgbm_below, 0.25, 0.0)
    lgbm_weight = 1.0 - if_weight
    comp["ensemble_weighted"] = lgbm_weight * lgbm_s + if_weight * if_s

    # ─── Estratégia C: OR lógico ───────────────────────────
    for if_th in [0.70, 0.80, 0.85, 0.90, 0.95]:
        col = f"pred_or_if{if_th:.2f}"
        comp[col] = ((lgbm_s >= LGBM_THRESHOLD) | (if_s >= if_th)).astype(int)

    # ─── Métricas contínuas ────────────────────────────────
    log.info(f"\n  {'Modelo':<24s} {'ROC-AUC':>8s} {'AP':>8s} {'R@5%':>8s} {'Best F1':>8s}")
    log.info("  " + "-" * 56)

    for score_col in ["lgbm_score", "anomaly_score", "ensemble_boost", "ensemble_weighted"]:
        y_s = comp[score_col].values
        bt, bf = find_best_threshold_f1(y_comp, y_s)
        auc = roc_auc_score(y_comp, y_s) if len(np.unique(y_comp)) > 1 else 0.0
        ap = average_precision_score(y_comp, y_s) if len(np.unique(y_comp)) > 1 else 0.0
        r5 = recall_at_k(y_comp, y_s, 0.05)
        comp_metrics[score_col] = {
            "roc_auc": float(auc),
            "average_precision": float(ap),
            "recall_at_5pct": float(r5),
            "best_threshold": float(bt),
            "best_f1": float(bf),
        }
        log.info(f"  {score_col:<24s} {auc:8.4f} {ap:8.4f} {r5:8.4f} {bf:8.4f}")

    # ─── Predições binárias (OR) ───────────────────────────
    log.info(f"\n  {'Config':<35s} {'TP':>4s} {'FP':>5s} {'FN':>4s} {'Recall':>8s} {'Prec':>8s}")
    log.info("  " + "-" * 64)

    tp_l = int(((lgbm_s >= LGBM_THRESHOLD) & (y_comp == 1)).sum())
    fp_l = int(((lgbm_s >= LGBM_THRESHOLD) & (y_comp == 0)).sum())
    fn_l = int(((lgbm_s < LGBM_THRESHOLD) & (y_comp == 1)).sum())
    log.info(
        f"  {'LGBM@0.27 (solo)':<35s} {tp_l:4d} {fp_l:5d} {fn_l:4d} "
        f"{tp_l / max(tp_l + fn_l, 1):8.4f} {tp_l / max(tp_l + fp_l, 1):8.4f}"
    )

    for if_th in [0.70, 0.80, 0.85, 0.90, 0.95]:
        col = f"pred_or_if{if_th:.2f}"
        tp = int(((comp[col] == 1) & (y_comp == 1)).sum())
        fp = int(((comp[col] == 1) & (y_comp == 0)).sum())
        fn = int(((comp[col] == 0) & (y_comp == 1)).sum())
        log.info(
            f"  {f'LGBM@0.27 OR IF@{if_th}':<35s} {tp:4d} {fp:5d} {fn:4d} "
            f"{tp / max(tp + fn, 1):8.4f} {tp / max(tp + fp, 1):8.4f}"
        )

    # ─── FN do LGBM que IF captura ────────────────────────
    lgbm_fn = comp[(y_comp == 1) & (lgbm_s < LGBM_THRESHOLD)]
    if len(lgbm_fn) > 0:
        log.info(f"\n  FN do LGBM ({len(lgbm_fn)}) — scores IF v3:")
        for _, row in lgbm_fn.iterrows():
            seg = "1ªTX" if row["is_first_tx_trimestre"] == 1 else "REG"
            c70 = "✅" if row["anomaly_score"] >= 0.70 else "❌"
            c85 = "✅" if row["anomaly_score"] >= 0.85 else "❌"
            c90 = "✅" if row["anomaly_score"] >= 0.90 else "❌"
            c95 = "✅" if row["anomaly_score"] >= 0.95 else "❌"
            log.info(
                f"    [{seg}] lgbm={row['lgbm_score']:.4f} if={row['anomaly_score']:.4f} "
                f"@0.70={c70} @0.85={c85} @0.90={c90} @0.95={c95}"
            )

    comp.to_csv(COMPLEMENTARITY_PATH, index=False)
    complementarity = comp_metrics

except FileNotFoundError:
    log.warning("Predições LGBM v5.1 não encontradas — pulando complementaridade")
except Exception as e:
    log.error(f"Erro na complementaridade: {e}")
    import traceback
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════
# SAVE ARTEFATOS
# ═══════════════════════════════════════════════════════════
elapsed = time.time() - t0

metadata = {
    "meta": {
        "version": "v3",
        "description": "IF v3 — features limpas (12), treino segmentado (regular_normal), log_vl_pix",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "input_file": INPUT_DATA,
        "input_md5": input_md5,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "random_seed": RANDOM_STATE,
    },
    "dataset": {
        "total_rows": len(df),
        "total_fraud": int(df["is_fraud"].sum()),
        "n_train": len(df_train),
        "n_train_fit": len(train_for_fit),
        "n_train_normal_total": len(train_all_normal),
        "n_train_normal_regular": len(train_regular_normal),
        "n_train_first_tx_excluded": n_first_tx_excluded,
        "train_strategy": train_strategy,
        "n_val": len(df_val),
        "n_holdout": len(df_holdout),
        "train_fraud": int(df_train["is_fraud"].sum()),
        "val_fraud": int(df_val["is_fraud"].sum()),
        "holdout_fraud": int(df_holdout["is_fraud"].sum()),
    },
    "model": {
        "algorithm": "IsolationForest (sklearn)",
        "n_features": len(IF_FEATURES_FINAL),
        "features": IF_FEATURES_FINAL,
        "features_removed_from_v2": [
            "qt_total_pix_trimestre", "distinct_receivers_so_far",
            "is_first_tx_trimestre", "ratio_valor_desvio_padrao",
            "ratio_valor_mediana", "zscore_valor_aprox",
            "valor_over_trimestre_avg", "qt_pix_dia_maximo_trimestre",
            "hour", "valor_x_first_recv",
        ],
        "features_added_in_v3": ["log_vl_pix"],
        "best_params": best_params,
        "best_val_ap": float(best_val_ap),
        "contamination_search": search_results,
    },
    "thresholds": {
        "best_f1": float(best_th_f1),
        "best_f1_score": float(best_f1_val),
        "recall_90": float(th_recall_90),
        "recall_95": float(th_recall_95),
    },
    "metrics": all_metrics,
    "complementarity": complementarity,
    "score_distribution": {
        "holdout": {
            "fraud_mean": float(scores_holdout[y_holdout == 1].mean()) if y_holdout.sum() > 0 else None,
            "fraud_median": float(np.median(scores_holdout[y_holdout == 1])) if y_holdout.sum() > 0 else None,
            "fraud_min": float(scores_holdout[y_holdout == 1].min()) if y_holdout.sum() > 0 else None,
            "fraud_max": float(scores_holdout[y_holdout == 1].max()) if y_holdout.sum() > 0 else None,
            "normal_mean": float(scores_holdout[y_holdout == 0].mean()),
            "normal_median": float(np.median(scores_holdout[y_holdout == 0])),
            "normal_p75": float(np.percentile(scores_holdout[y_holdout == 0], 75)),
            "normal_max": float(scores_holdout[y_holdout == 0].max()),
            "normal_p99": float(np.percentile(scores_holdout[y_holdout == 0], 99)),
        },
        "v2_comparison": {
            "v2_fraud_median": V2_FRAUD_MEDIAN,
            "v2_normal_median": V2_NORMAL_MEDIAN,
            "v2_normal_p75": V2_NORMAL_P75,
            "v3_fraud_median": v3_fraud_med,
            "v3_normal_median": v3_normal_med,
            "v3_normal_p75": v3_normal_p75,
        },
    },
    "artefacts": {
        "production": {
            "model": MODEL_PATH,
            "scaler": SCALER_PATH,
            "ref_raw": REF_RAW_PATH,
        },
        "report": {
            "metrics": os.path.basename(METRICS_PATH),
            "config": os.path.basename(CONFIG_PATH),
            "feature_importance": os.path.basename(FEAT_IMP_PATH),
            "holdout_predictions": os.path.basename(HOLDOUT_PREDS_PATH),
            "score_distribution": os.path.basename(SCORE_DIST_PATH),
            "error_analysis": os.path.basename(ERROR_ANALYSIS_PATH),
            "complementarity": os.path.basename(COMPLEMENTARITY_PATH),
            "contamination_search": os.path.basename(HYPERPARAM_PATH),
            "threshold_sweep": os.path.basename(THRESHOLD_SWEEP_PATH),
            "training_log": os.path.basename(TRAINING_LOG_PATH),
        },
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

# ─── IF Config (usado pelo Engine) ────────────────────────
if_config = {
    "version": "v3",
    "features": IF_FEATURES_FINAL,
    "medians": {k: float(v) for k, v in medians.to_dict().items()},
    "best_threshold": float(best_th_f1),
    "best_params": best_params,
    "train_strategy": train_strategy,
    "ensemble_strategy": "complementary_boost",
    "ensemble_params": {
        "lgbm_threshold": 0.27,
        "description": "IF v3 ativa quando LGBM < 0.27. Boost em anomalias extremas.",
        "if_high_threshold": 0.90,
        "if_very_high_threshold": 0.95,
        "boost_high": 0.05,
        "boost_very_high": 0.10,
    },
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(if_config, f, ensure_ascii=False, indent=2)

# ─── Modelos ──────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
np.save(REF_RAW_PATH, ref_raw_train)


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  RESULTADOS — Isolation Forest v3 (Features Limpas + Segmentado)")
print(f"{'=' * 72}")

m_h = all_metrics.get("holdout_at_0.5", {})
m_hf1 = all_metrics.get("holdout_at_best_f1", {})

print(f"""
  Dataset: {len(df):,} tx ({df['is_fraud'].sum()} fraudes)
  Treino (fit): {len(train_for_fit):,} normais regulares ({train_strategy})
  Validação: {len(df_val):,} ({df_val['is_fraud'].sum()} fraudes)
  Holdout: {len(df_holdout):,} ({df_holdout['is_fraud'].sum()} fraudes)
  Features: {len(IF_FEATURES_FINAL)} (v2: 22 → v3: {len(IF_FEATURES_FINAL)})
  Config: {best_params}

  ────────────────────────────────────────────────────────
  HOLDOUT @ 0.5:
    ROC-AUC:  {m_h.get('roc_auc', 'N/A')}
    AP:       {m_h.get('average_precision', 'N/A')}
    F1:       {m_h.get('f1', 'N/A')}
    Recall:   {m_h.get('recall', 'N/A')}  (TP={m_h.get('tp', '?')}, FN={m_h.get('fn', '?')})
    Precision:{m_h.get('precision', 'N/A')}  (FP={m_h.get('fp', '?')})
    R@5%:     {m_h.get('recall_at_5pct', 'N/A')}
    R@Top100: {m_h.get('recall_at_top_100', 'N/A')}
    R@Top200: {m_h.get('recall_at_top_200', 'N/A')}

  HOLDOUT @ Best F1 (th={best_th_f1:.4f}):
    F1:       {m_hf1.get('f1', 'N/A')}
    Recall:   {m_hf1.get('recall', 'N/A')}  (TP={m_hf1.get('tp', '?')}, FN={m_hf1.get('fn', '?')})
    Precision:{m_hf1.get('precision', 'N/A')}  (FP={m_hf1.get('fp', '?')})

  Score Distribution (v2 → v3):
    Fraud median:  {V2_FRAUD_MEDIAN:.4f} → {v3_fraud_med:.4f}
    Normal median: {V2_NORMAL_MEDIAN:.4f} → {v3_normal_med:.4f}
    Normal P75:    {V2_NORMAL_P75:.4f} → {v3_normal_p75:.4f}

  Tempo total: {elapsed:.1f}s
""")

print("  Artefatos PRODUÇÃO:")
for path in [MODEL_PATH, SCALER_PATH, REF_RAW_PATH]:
    size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
    print(f"    {os.path.basename(path)} ({size_kb:.0f} KB)")

print("\n  Artefatos RELATÓRIO (resultado_treino_if/):")
for path in [
    METRICS_PATH, CONFIG_PATH, FEAT_IMP_PATH, HOLDOUT_PREDS_PATH,
    SCORE_DIST_PATH, ERROR_ANALYSIS_PATH, COMPLEMENTARITY_PATH,
    HYPERPARAM_PATH, THRESHOLD_SWEEP_PATH, TRAINING_LOG_PATH,
]:
    size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
    print(f"    {os.path.basename(path)} ({size_kb:.0f} KB)")

print(f"\n{'=' * 72}")
print("  ✅ Isolation Forest v3 — Treino concluído")
print(f"{'=' * 72}")

# Restaurar stdout
sys.stdout = tee.terminal
tee.close()
