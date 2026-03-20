"""
analise_scores_fraude.py — Análise de Scores das Fraudes + Calibração de Limiares
==================================================================================

Este script:
  1. Passa TODAS as fraudes pelo pipeline e analisa seus scores
  2. Passa uma amostra de normais para comparar
  3. Testa 3 abordagens de scoring:
     A) Score RAW do LGBM
     B) Percentile Score (0-100)
     C) Reescala Híbrida
  4. Encontra limiares ótimos para as faixas:
     - 0-59.99: APROVAR (deixar passar)
     - 60-84.99: CONFIRMAR (autenticação adicional)
     - 85-100: BLOQUEAR (analista humano)
  5. Gera dashboard comparativo

Uso:
  python analise_scores_fraude.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "backend").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"
RELATORIO_DIR = PROJECT_ROOT / "relatorio"
RELATORIO_DIR.mkdir(exist_ok=True)

# =========================================================
# ESTILO
# =========================================================
plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#1a1d23",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#ffffff",
        "text.color": "#ffffff",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "grid.color": "#333333",
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)

C = {
    "green": "#00d4aa",
    "red": "#ff6b6b",
    "yellow": "#ffd93d",
    "blue": "#6c5ce7",
    "orange": "#ff9f43",
    "cyan": "#4ecdc4",
    "white": "#ffffff",
    "gray": "#888888",
}


# =========================================================
# 1. CARREGAR DADOS E MODELOS
# =========================================================
def load_all():
    """Carrega modelos e dados de teste."""
    print("\n" + "=" * 70)
    print("  1. CARREGANDO ARTEFATOS E DADOS")
    print("=" * 70)

    # Modelos
    lgbm_cal = joblib.load(ARTEFATOS_DIR / "model_lightgbm_calibrated.joblib")
    lgbm_raw = joblib.load(ARTEFATOS_DIR / "model_lightgbm.joblib")
    print(f"  ✅ LGBM Calibrado: {type(lgbm_cal).__name__}")
    print(f"  ✅ LGBM Raw: {type(lgbm_raw).__name__}")

    # IF
    if_model = joblib.load(ARTEFATOS_DIR / "model_isolation_forest.joblib")
    if_scaler = joblib.load(ARTEFATOS_DIR / "scaler_isolation_forest.joblib")
    with open(ARTEFATOS_DIR / "isolation_forest_config.json", "r") as f:
        if_config = json.load(f)
    if_ref = np.load(ARTEFATOS_DIR / "if_ref_raw_train.npy")
    print(f"  ✅ Isolation Forest + Scaler + Config + Ref Scores")

    # Features
    with open(ARTEFATOS_DIR / "lgbm_features.json", "r") as f:
        lgbm_features = json.load(f)
    print(f"  ✅ LGBM Features: {len(lgbm_features)}")

    # Dados de teste
    X_test = pd.read_csv(ARTEFATOS_DIR / "X_test.csv")
    y_test = pd.read_csv(ARTEFATOS_DIR / "y_test.csv").iloc[:, 0]
    print(f"  ✅ X_test: {X_test.shape} | Fraudes: {y_test.sum()} | Normais: {(y_test == 0).sum()}")

    return {
        "lgbm_cal": lgbm_cal,
        "lgbm_raw": lgbm_raw,
        "if_model": if_model,
        "if_scaler": if_scaler,
        "if_config": if_config,
        "if_ref": if_ref,
        "lgbm_features": lgbm_features,
        "X_test": X_test,
        "y_test": y_test,
    }


# =========================================================
# 2. CALCULAR TODOS OS SCORES
# =========================================================
def calculate_all_scores(data: Dict) -> pd.DataFrame:
    """Calcula scores de todas as abordagens para todo o dataset."""
    print("\n" + "=" * 70)
    print("  2. CALCULANDO SCORES (TODAS AS ABORDAGENS)")
    print("=" * 70)

    X_test = data["X_test"]
    y_test = data["y_test"]
    lgbm_features = data["lgbm_features"]

    X_lgbm = X_test[lgbm_features].fillna(0)
    n = len(X_test)

    # ─── Score A: LGBM Raw ───
    score_raw = data["lgbm_raw"].predict_proba(X_lgbm)[:, 1]
    print(f"  Score Raw   → min={score_raw.min():.6f}, max={score_raw.max():.6f}, "
          f"mean={score_raw.mean():.6f}")

    # ─── Score B: LGBM Calibrado ───
    score_cal = data["lgbm_cal"].predict_proba(X_lgbm)[:, 1]
    print(f"  Score Cal   → min={score_cal.min():.6f}, max={score_cal.max():.6f}, "
          f"mean={score_cal.mean():.6f}")

    # ─── Isolation Forest (para primeiras tx) ───
    if_config = data["if_config"]
    if_features = if_config.get("features", [])
    if_medians = if_config.get("medians", {})
    is_first = X_test["is_first_tx_trimestre"].values if "is_first_tx_trimestre" in X_test.columns else np.zeros(n)

    if_scores = np.zeros(n)
    first_mask = is_first.astype(bool)
    n_first = first_mask.sum()

    if n_first > 0:
        X_if = pd.DataFrame(index=X_test[first_mask].index)
        for feat in if_features:
            if feat in X_test.columns:
                X_if[feat] = X_test.loc[first_mask, feat].values
            else:
                X_if[feat] = if_medians.get(feat, 0)
        for feat in if_features:
            X_if[feat] = X_if[feat].fillna(if_medians.get(feat, 0))

        X_if_scaled = data["if_scaler"].transform(X_if[if_features])
        raw_if = data["if_model"].decision_function(X_if_scaled)

        if data["if_ref"] is not None and len(data["if_ref"]) > 0:
            percentiles_if = np.array([np.mean(data["if_ref"] <= s) for s in raw_if])
        else:
            percentiles_if = 1.0 / (1.0 + np.exp(raw_if * 5))
        percentiles_if = np.clip(percentiles_if, 0, 1)
        first_idx = np.where(first_mask)[0]
        if_scores[first_idx] = percentiles_if

    # ─── Score C: Percentile Score ───
    # Calcula o percentil de cada score raw em relação a todos os scores
    score_percentile = np.array([np.mean(score_raw <= s) * 100 for s in score_raw])
    print(f"  Score Pctil → min={score_percentile.min():.2f}, max={score_percentile.max():.2f}")

    # ─── Score D: Reescala Híbrida (mapeamento não-linear) ───
    # Mapear score raw para escala 0-100 usando pontos de ancoragem
    # Baseado nos quantis dos scores normais vs fraudes
    score_hybrid = _rescale_hybrid(score_raw, score_cal)
    print(f"  Score Hybr  → min={score_hybrid.min():.2f}, max={score_hybrid.max():.2f}")

    # ─── Score E: Score Raw * 100 (simples) ───
    score_raw_100 = np.clip(score_raw * 100, 0, 100)
    print(f"  Score R*100 → min={score_raw_100.min():.2f}, max={score_raw_100.max():.2f}")

    # ─── Score F: Ensemble com IF (usando raw) ───
    # IF contribui apenas para primeiras tx na zona cinzenta
    score_ensemble_raw = score_raw.copy()
    if_active_mask = first_mask & (score_raw >= 0.03) & (score_raw <= 0.70)
    if if_active_mask.any():
        score_ensemble_raw[if_active_mask] = (
            0.75 * score_raw[if_active_mask]
            + 0.25 * if_scores[if_active_mask]
        )
    score_ensemble_100 = np.clip(score_ensemble_raw * 100, 0, 100)

    # Montar DataFrame
    results = pd.DataFrame({
        "y_true": y_test.values,
        "is_first_tx": is_first.astype(int),
        "if_score": if_scores,

        # Scores originais
        "score_raw": score_raw,
        "score_calibrated": score_cal,

        # 4 abordagens de escala 0-100
        "A_raw_x100": score_raw_100,
        "B_percentile": score_percentile,
        "C_hybrid": score_hybrid,
        "D_ensemble_x100": score_ensemble_100,
    })

    # Rule scores
    if "rule_score_raw" in X_test.columns:
        results["rule_score_raw"] = X_test["rule_score_raw"].values

    return results


def _rescale_hybrid(score_raw: np.ndarray, score_cal: np.ndarray) -> np.ndarray:
    """
    Reescala não-linear para 0-100.

    Âncoras:
      raw 0.0001 → 0 (transação claramente normal)
      raw 0.001  → 20
      raw 0.01   → 40
      raw 0.10   → 60
      raw 0.50   → 75
      raw 0.90   → 90
      raw 0.99   → 98
      raw 0.999+ → 100
    """
    anchors_raw = np.array([0.0, 0.0001, 0.001, 0.01, 0.05, 0.10, 0.30, 0.50, 0.80, 0.95, 0.999, 1.0])
    anchors_out = np.array([0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 70.0, 78.0, 88.0, 95.0, 99.0, 100.0])

    result = np.interp(score_raw, anchors_raw, anchors_out)
    return np.clip(result, 0, 100)


# =========================================================
# 3. ANALISAR SCORES DAS FRAUDES
# =========================================================
def analyze_fraud_scores(results: pd.DataFrame) -> Dict:
    """Análise detalhada dos scores das fraudes."""
    print("\n" + "=" * 70)
    print("  3. ANÁLISE DOS SCORES DAS FRAUDES")
    print("=" * 70)

    fraud_mask = results["y_true"] == 1
    normal_mask = results["y_true"] == 0
    n_fraud = fraud_mask.sum()
    n_normal = normal_mask.sum()

    score_cols = ["A_raw_x100", "B_percentile", "C_hybrid", "D_ensemble_x100"]
    analysis = {}

    for col in score_cols:
        fraud_scores = results.loc[fraud_mask, col].values
        normal_scores = results.loc[normal_mask, col].values

        stats = {
            "fraud_min": float(np.min(fraud_scores)),
            "fraud_p5": float(np.percentile(fraud_scores, 5)),
            "fraud_p10": float(np.percentile(fraud_scores, 10)),
            "fraud_p25": float(np.percentile(fraud_scores, 25)),
            "fraud_median": float(np.median(fraud_scores)),
            "fraud_p75": float(np.percentile(fraud_scores, 75)),
            "fraud_p95": float(np.percentile(fraud_scores, 95)),
            "fraud_max": float(np.max(fraud_scores)),
            "fraud_mean": float(np.mean(fraud_scores)),
            "normal_max": float(np.max(normal_scores)),
            "normal_p999": float(np.percentile(normal_scores, 99.9)),
            "normal_p99": float(np.percentile(normal_scores, 99)),
            "normal_p95": float(np.percentile(normal_scores, 95)),
            "normal_median": float(np.median(normal_scores)),
            "gap": float(np.min(fraud_scores) - np.max(normal_scores)),
        }
        analysis[col] = stats

        print(f"\n  ── {col} ──")
        print(f"  Fraudes ({n_fraud} tx):")
        print(f"    Min:     {stats['fraud_min']:8.2f}")
        print(f"    P5:      {stats['fraud_p5']:8.2f}")
        print(f"    P10:     {stats['fraud_p10']:8.2f}")
        print(f"    P25:     {stats['fraud_p25']:8.2f}")
        print(f"    Mediana: {stats['fraud_median']:8.2f}")
        print(f"    P75:     {stats['fraud_p75']:8.2f}")
        print(f"    P95:     {stats['fraud_p95']:8.2f}")
        print(f"    Max:     {stats['fraud_max']:8.2f}")
        print(f"  Normais (top):")
        print(f"    Max:     {stats['normal_max']:8.2f}")
        print(f"    P99.9:   {stats['normal_p999']:8.2f}")
        print(f"    P99:     {stats['normal_p99']:8.2f}")
        print(f"  GAP (min fraude - max normal): {stats['gap']:+.2f}")

    return analysis


# =========================================================
# 4. TESTAR FAIXAS DE DECISÃO
# =========================================================
def test_decision_bands(results: pd.DataFrame) -> Dict:
    """Testa as faixas de decisão propostas pelo Adilio."""
    print("\n" + "=" * 70)
    print("  4. TESTANDO FAIXAS DE DECISÃO")
    print("=" * 70)

    y_true = results["y_true"].values
    n_fraud = (y_true == 1).sum()
    n_normal = (y_true == 0).sum()

    # Faixas desejadas:
    #   0 - 59.99  → APROVAR
    #   60 - 84.99 → CONFIRMAR (2FA)
    #   85 - 100   → BLOQUEAR (analista)

    score_cols = ["A_raw_x100", "B_percentile", "C_hybrid", "D_ensemble_x100"]
    band_results = {}

    print(f"\n  Faixas: APROVAR [0-60) | CONFIRMAR [60-85) | BLOQUEAR [85-100]")
    print(f"  Fraudes: {n_fraud} | Normais: {n_normal}")
    print(f"\n  {'Abordagem':<20} {'Recall':>8} {'Prec.':>8} {'F1':>8} "
          f"{'FN':>5} {'FP':>6} {'APROV':>8} {'CONF':>8} {'BLOQ':>8}")
    print(f"  {'─' * 95}")

    for col in score_cols:
        scores = results[col].values

        decisions = np.full(len(scores), "APROVAR", dtype=object)
        decisions[scores >= 60] = "CONFIRMAR"
        decisions[scores >= 85] = "BLOQUEAR"

        # BLOQUEAR = fraude detectada (para métricas principais)
        y_pred_bloq = (scores >= 85).astype(int)

        # CONFIRMAR + BLOQUEAR = qualquer ação
        y_pred_any = (scores >= 60).astype(int)

        tp = int(((y_pred_bloq == 1) & (y_true == 1)).sum())
        fp = int(((y_pred_bloq == 1) & (y_true == 0)).sum())
        fn = int(((y_pred_bloq == 0) & (y_true == 1)).sum())

        recall_b = tp / n_fraud if n_fraud > 0 else 0
        prec_b = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_b = 2 * prec_b * recall_b / (prec_b + recall_b) if (prec_b + recall_b) > 0 else 0

        tp_any = int(((y_pred_any == 1) & (y_true == 1)).sum())
        fp_any = int(((y_pred_any == 1) & (y_true == 0)).sum())
        fn_any = int(((y_pred_any == 0) & (y_true == 1)).sum())

        recall_any = tp_any / n_fraud if n_fraud > 0 else 0

        n_aprov = int((decisions == "APROVAR").sum())
        n_conf = int((decisions == "CONFIRMAR").sum())
        n_bloq = int((decisions == "BLOQUEAR").sum())

        fraud_in_aprov = int(y_true[decisions == "APROVAR"].sum())
        fraud_in_conf = int(y_true[decisions == "CONFIRMAR"].sum())
        fraud_in_bloq = int(y_true[decisions == "BLOQUEAR"].sum())

        print(f"  {col:<20} {recall_b:>7.1%} {prec_b:>7.1%} {f1_b:>7.4f} "
              f"{fn:>5} {fp:>6} {n_aprov:>7,} {n_conf:>7,} {n_bloq:>7,}")
        print(f"  {'':20} {'Recall ≥60:':>8} {recall_any:>6.1%}   "
              f"FN≥60:{fn_any:>3}  "
              f"Fraude: {fraud_in_aprov}|{fraud_in_conf}|{fraud_in_bloq}")

        band_results[col] = {
            "bloq_recall": round(recall_b, 4),
            "bloq_precision": round(prec_b, 4),
            "bloq_f1": round(f1_b, 4),
            "bloq_fn": fn,
            "bloq_fp": fp,
            "any_recall": round(recall_any, 4),
            "any_fn": fn_any,
            "any_fp": fp_any,
            "n_aprovar": n_aprov,
            "n_confirmar": n_conf,
            "n_bloquear": n_bloq,
            "fraud_in_aprovar": fraud_in_aprov,
            "fraud_in_confirmar": fraud_in_conf,
            "fraud_in_bloquear": fraud_in_bloq,
        }

    return band_results


# =========================================================
# 5. ENCONTRAR LIMIARES ÓTIMOS
# =========================================================
def find_optimal_thresholds(results: pd.DataFrame) -> Dict:
    """Encontra limiares ótimos para cada abordagem."""
    print("\n" + "=" * 70)
    print("  5. LIMIARES ÓTIMOS")
    print("=" * 70)

    y_true = results["y_true"].values
    score_cols = ["A_raw_x100", "B_percentile", "C_hybrid", "D_ensemble_x100"]
    optimal = {}

    for col in score_cols:
        scores = results[col].values
        fraud_scores = scores[y_true == 1]
        normal_scores = scores[y_true == 0]

        # Threshold = menor score de fraude (sua ideia)
        min_fraud = float(np.min(fraud_scores))

        # Threshold com margem de segurança (P5 das fraudes)
        p5_fraud = float(np.percentile(fraud_scores, 5))

        # Threshold que maximiza F1
        best_f1 = 0
        best_t_f1 = 0
        for t in np.arange(0, 100, 0.5):
            pred = (scores >= t).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t_f1 = t

        # Threshold com FPR < 0.1%
        max_fp_allowed = int(len(normal_scores) * 0.001)
        sorted_normal = np.sort(normal_scores)[::-1]
        t_fpr_01 = float(sorted_normal[min(max_fp_allowed, len(sorted_normal) - 1)])

        optimal[col] = {
            "min_fraud_score": round(min_fraud, 2),
            "p5_fraud_score": round(p5_fraud, 2),
            "best_f1_threshold": round(best_t_f1, 2),
            "best_f1_value": round(best_f1, 4),
            "fpr_01_threshold": round(t_fpr_01, 2),
        }

        # Recall com faixas usando min_fraud como BLOQUEAR
        t_bloq = max(min_fraud - 5, 0)  # 5 pontos de margem
        t_conf = max(t_bloq - 15, 0)  # CONFIRMAR 15 pontos abaixo

        pred_bloq = (scores >= t_bloq).astype(int)
        pred_conf = (scores >= t_conf).astype(int)

        recall_bloq = recall_score(y_true, pred_bloq, zero_division=0)
        recall_conf = recall_score(y_true, pred_conf, zero_division=0)
        fp_bloq = int(((pred_bloq == 1) & (y_true == 0)).sum())
        fp_conf = int(((pred_conf == 1) & (y_true == 0)).sum())

        optimal[col]["suggested_bloquear"] = round(t_bloq, 1)
        optimal[col]["suggested_confirmar"] = round(t_conf, 1)
        optimal[col]["recall_at_bloquear"] = round(recall_bloq, 4)
        optimal[col]["recall_at_confirmar"] = round(recall_conf, 4)
        optimal[col]["fp_at_bloquear"] = fp_bloq
        optimal[col]["fp_at_confirmar"] = fp_conf

        print(f"\n  ── {col} ──")
        print(f"  Menor score de fraude:    {min_fraud:.2f}")
        print(f"  P5 das fraudes:           {p5_fraud:.2f}")
        print(f"  Melhor F1 threshold:      {best_t_f1:.2f} (F1={best_f1:.4f})")
        print(f"  Sugestão BLOQUEAR:        ≥ {t_bloq:.1f} (recall={recall_bloq:.1%}, FP={fp_bloq})")
        print(f"  Sugestão CONFIRMAR:        ≥ {t_conf:.1f} (recall={recall_conf:.1%}, FP={fp_conf})")

    return optimal


# =========================================================
# 6. DASHBOARD COMPARATIVO
# =========================================================
def plot_comparison_dashboard(results: pd.DataFrame, analysis: Dict, optimal: Dict):
    """Gera dashboard comparando as abordagens."""
    print("\n" + "=" * 70)
    print("  6. GERANDO DASHBOARD COMPARATIVO")
    print("=" * 70)

    fraud_mask = results["y_true"] == 1
    normal_mask = results["y_true"] == 0
    score_cols = ["A_raw_x100", "B_percentile", "C_hybrid", "D_ensemble_x100"]
    titles = ["A) Raw × 100", "B) Percentile", "C) Híbrido", "D) Ensemble × 100"]

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle(
        "Comparação de Abordagens de Scoring — Escala 0-100\n"
        f"Faixas: APROVAR [0-60) | CONFIRMAR [60-75) | BLOQUEAR [75-100]",
        fontsize=18, fontweight="bold", color=C["green"], y=1.02,
    )

    for i, (col, title) in enumerate(zip(score_cols, titles)):
        fraud_scores = results.loc[fraud_mask, col].values
        normal_scores = results.loc[normal_mask, col].values

        # ─── Linha 1: Distribuição ───
        ax = axes[0, i]

        # Histograma normais (só top 5% para ver melhor)
        p95_normal = np.percentile(normal_scores, 95)
        normal_top = normal_scores[normal_scores >= p95_normal]

        ax.hist(normal_top, bins=50, alpha=0.6, color=C["green"],
                label=f"Normais top 5%\n(n={len(normal_top):,})", density=True)
        ax.hist(fraud_scores, bins=30, alpha=0.8, color=C["red"],
                label=f"Fraudes\n(n={len(fraud_scores)})", density=True)

        # Faixas
        ax.axvspan(0, 60, alpha=0.05, color=C["green"])
        ax.axvspan(60, 75, alpha=0.08, color=C["yellow"])
        ax.axvspan(75, 100, alpha=0.08, color=C["red"])

        ax.axvline(60, color=C["yellow"], ls="--", lw=2, alpha=0.8)
        ax.axvline(75, color=C["red"], ls="--", lw=2, alpha=0.8)

        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlabel("Score (0-100)")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0, 105)

        # Métricas no canto
        stats = analysis[col]
        opt = optimal[col]
        text = (
            f"Fraud min: {stats['fraud_min']:.1f}\n"
            f"Fraud P10: {stats['fraud_p10']:.1f}\n"
            f"Normal max: {stats['normal_max']:.1f}\n"
            f"GAP: {stats['gap']:+.1f}"
        )
        ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=8,
                va="top", ha="right", color=C["gray"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d23", edgecolor="#333"))

        # ─── Linha 2: Scatter fraudes ───
        ax2 = axes[1, i]

        # Plotar cada fraude como ponto
        fraud_idx = np.arange(len(fraud_scores))
        colors_fraud = np.where(fraud_scores >= 75, C["red"],
                       np.where(fraud_scores >= 60, C["yellow"], C["green"]))

        ax2.scatter(fraud_idx, sorted(fraud_scores), c=[
            C["red"] if s >= 75 else C["yellow"] if s >= 60 else C["green"]
            for s in sorted(fraud_scores)
        ], s=40, zorder=5, edgecolors="white", linewidths=0.5)

        ax2.axhline(60, color=C["yellow"], ls="--", lw=2, alpha=0.8, label="CONFIRMAR (60)")
        ax2.axhline(75, color=C["red"], ls="--", lw=2, alpha=0.8, label="BLOQUEAR (75)")
        ax2.axhspan(0, 60, alpha=0.05, color=C["green"])
        ax2.axhspan(60, 75, alpha=0.08, color=C["yellow"])
        ax2.axhspan(75, 100, alpha=0.08, color=C["red"])

        n_bloq = (fraud_scores >= 75).sum()
        n_conf = ((fraud_scores >= 60) & (fraud_scores < 75)).sum()
        n_aprov = (fraud_scores < 60).sum()

        ax2.set_title(
            f"BLOQ: {n_bloq} ({n_bloq/len(fraud_scores)*100:.0f}%) | "
            f"CONF: {n_conf} ({n_conf/len(fraud_scores)*100:.0f}%) | "
            f"APROV: {n_aprov} ({n_aprov/len(fraud_scores)*100:.0f}%)",
            fontsize=11,
        )
        ax2.set_xlabel(f"Fraudes (ordenadas por score)")
        ax2.set_ylabel("Score (0-100)")
        ax2.set_ylim(-5, 105)
        ax2.legend(fontsize=8)

    plt.tight_layout()
    output = RELATORIO_DIR / "comparacao_scores.png"
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Dashboard salvo: {output}")


# =========================================================
# 7. GERAR RELATÓRIO DE RECOMENDAÇÃO
# =========================================================
def generate_recommendation(analysis: Dict, optimal: Dict, band_results: Dict):
    """Gera recomendação final."""
    print("\n" + "=" * 70)
    print("  7. RECOMENDAÇÃO FINAL")
    print("=" * 70)

    # Encontrar melhor abordagem
    # Critério: máximo recall no BLOQUEAR com FP mínimo
    best_col = None
    best_recall = 0
    best_fp = float("inf")

    for col, res in band_results.items():
        # Preferir: mais fraudes bloqueadas, menos falsos positivos
        score = res["bloq_recall"] * 1000 - res["bloq_fp"]
        if res["bloq_recall"] > best_recall or (res["bloq_recall"] == best_recall and res["bloq_fp"] < best_fp):
            best_recall = res["bloq_recall"]
            best_fp = res["bloq_fp"]
            best_col = col

    print(f"\n  🏆 MELHOR ABORDAGEM: {best_col}")
    print(f"  ─────────────────────────────────")

    res = band_results[best_col]
    opt = optimal[best_col]
    stats = analysis[best_col]

    print(f"""
  Com as faixas 0-60 / 60-75 / 75-100:
    BLOQUEAR (≥75):  {res['fraud_in_bloquear']}/{res['fraud_in_bloquear'] + res['fraud_in_confirmar'] + res['fraud_in_aprovar']} fraudes ({res['bloq_recall']:.1%} recall)
    CONFIRMAR (60-75): {res['fraud_in_confirmar']} fraudes adicionais
    APROVAR (<60):   {res['fraud_in_aprovar']} fraudes escapam

  Total detectado (≥60): {res['any_recall']:.1%} recall
  Falsos positivos ≥75:  {res['bloq_fp']:,}
  Falsos positivos ≥60:  {res['any_fp']:,}

  Limiares sugeridos (baseados nas fraudes):
    BLOQUEAR:  ≥ {opt['suggested_bloquear']:.0f}  (recall={opt['recall_at_bloquear']:.1%})
    CONFIRMAR: ≥ {opt['suggested_confirmar']:.0f}  (recall={opt['recall_at_confirmar']:.1%})
    """)

    # Salvar config
    config = {
        "abordagem_recomendada": best_col,
        "faixas": {
            "aprovar": {"min": 0, "max": 60, "acao": "Liberar automaticamente"},
            "confirmar": {"min": 60, "max": 75, "acao": "Exigir 2FA / biometria"},
            "bloquear": {"min": 75, "max": 100, "acao": "Parar para analista humano"},
        },
        "metricas_com_faixas_padrao": band_results[best_col],
        "limiares_otimizados": optimal[best_col],
        "estatisticas_fraudes": analysis[best_col],
        "todas_abordagens": {
            col: {
                "band": band_results[col],
                "optimal": optimal[col],
                "stats": analysis[col],
            }
            for col in band_results
        },
    }

    config_path = RELATORIO_DIR / "config_scoring_recomendado.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✅ Config salva: {config_path}")

    return config


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n")
    print("█" * 70)
    print("  ANÁLISE DE SCORES + CALIBRAÇÃO DE LIMIARES")
    print("  Sistema Anomalia PIX")
    print("█" * 70)

    t0 = time.time()

    # 1. Carregar
    data = load_all()

    # 2. Calcular scores
    results = calculate_all_scores(data)

    # 3. Analisar fraudes
    analysis = analyze_fraud_scores(results)

    # 4. Testar faixas
    band_results = test_decision_bands(results)

    # 5. Limiares ótimos
    optimal = find_optimal_thresholds(results)

    # 6. Dashboard
    plot_comparison_dashboard(results, analysis, optimal)

    # 7. Recomendação
    config = generate_recommendation(analysis, optimal, band_results)

    # 8. Salvar resultados detalhados
    results_path = RELATORIO_DIR / "scores_detalhados_todas_abordagens.csv"
    results.to_csv(results_path, index=False)
    print(f"\n  ✅ Scores detalhados: {results_path}")

    elapsed = time.time() - t0
    print(f"\n  ⏱️  Tempo total: {elapsed:.1f}s")

    print(f"\n\n{'█' * 70}")
    print("  RESUMO")
    print(f"{'█' * 70}")
    print(f"""
  📊 Abordagens testadas: 4
  🏆 Recomendada: {config['abordagem_recomendada']}
  
  📁 Arquivos gerados:
     📊 {RELATORIO_DIR}/comparacao_scores.png
     📋 {RELATORIO_DIR}/config_scoring_recomendado.json
     📑 {RELATORIO_DIR}/scores_detalhados_todas_abordagens.csv
  
  👉 PRÓXIMO PASSO:
     Analise o dashboard e o JSON.
     Com base nos resultados, decidimos qual abordagem usar
     e atualizamos o pipeline_inferencia.py.
    """)


if __name__ == "__main__":
    main()
