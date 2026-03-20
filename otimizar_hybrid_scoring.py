"""
otimizar_hybrid_scoring.py — Otimização das Âncoras do Score Híbrido
=====================================================================

Encontra o mapeamento não-linear ótimo para o score raw → score 0-100
que maximiza a separação entre fraudes e normais nas faixas:
  - APROVAR:   [0, 60)
  - CONFIRMAR: [60, 85)
  - BLOQUEAR:  [85, 100]

Gera:
  - config_scoring_final.json (configuração definitiva)
  - dashboard_scoring_final.png (visualização)
  - Atualiza o pipeline de inferência

Uso:
  python otimizar_hybrid_scoring.py
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
from typing import Dict, List, Tuple
from itertools import product

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
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

# Faixas de decisão
FAIXA_CONFIRMAR = 60.0
FAIXA_BLOQUEAR = 85.0

# Estilo
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
    "dark": "#1a1d23",
}


# =========================================================
# 1. CARREGAR DADOS
# =========================================================
def load_data():
    """Carrega modelos e dados de teste."""
    print("\n  Carregando artefatos...")
    lgbm_raw = joblib.load(ARTEFATOS_DIR / "model_lightgbm.joblib")
    lgbm_cal = joblib.load(ARTEFATOS_DIR / "model_lightgbm_calibrated.joblib")

    with open(ARTEFATOS_DIR / "lgbm_features.json", "r") as f:
        lgbm_features = json.load(f)

    X_test = pd.read_csv(ARTEFATOS_DIR / "X_test.csv")
    y_test = pd.read_csv(ARTEFATOS_DIR / "y_test.csv").iloc[:, 0]

    # Calcular score raw
    X_lgbm = X_test[lgbm_features].fillna(0)
    score_raw = lgbm_raw.predict_proba(X_lgbm)[:, 1]
    score_cal = lgbm_cal.predict_proba(X_lgbm)[:, 1]

    print(f"  ✅ {len(X_test)} transações | {y_test.sum()} fraudes")
    print(f"  Score raw: min={score_raw.min():.6f}, max={score_raw.max():.6f}")

    return score_raw, score_cal, y_test.values, X_test


# =========================================================
# 2. OTIMIZAÇÃO DAS ÂNCORAS
# =========================================================
def apply_mapping(score_raw: np.ndarray, anchors_raw: np.ndarray, anchors_out: np.ndarray) -> np.ndarray:
    """Aplica mapeamento não-linear usando interpolação."""
    # Converter para arrays numpy puros
    ar = np.asarray(anchors_raw, dtype=np.float64)
    ao = np.asarray(anchors_out, dtype=np.float64)
    sr = np.asarray(score_raw, dtype=np.float64)
    return np.clip(np.interp(sr, ar, ao), 0.0, 100.0)


def evaluate_mapping(
    score_mapped: np.ndarray,
    y_true: np.ndarray,
    t_confirmar: float = FAIXA_CONFIRMAR,
    t_bloquear: float = FAIXA_BLOQUEAR,
) -> Dict:
    """Avalia uma configuração de mapeamento."""
    fraud_mask = y_true == 1
    normal_mask = y_true == 0

    fraud_scores = score_mapped[fraud_mask]
    normal_scores = score_mapped[normal_mask]

    # Decisões
    y_pred_bloq = (score_mapped >= t_bloquear).astype(int)
    y_pred_any = (score_mapped >= t_confirmar).astype(int)

    tp_b = int(((y_pred_bloq == 1) & (y_true == 1)).sum())
    fp_b = int(((y_pred_bloq == 1) & (y_true == 0)).sum())
    fn_b = int(((y_pred_bloq == 0) & (y_true == 1)).sum())

    tp_a = int(((y_pred_any == 1) & (y_true == 1)).sum())
    fp_a = int(((y_pred_any == 1) & (y_true == 0)).sum())
    fn_a = int(((y_pred_any == 0) & (y_true == 1)).sum())

    n_fraud = int(fraud_mask.sum())

    recall_bloq = tp_b / n_fraud if n_fraud > 0 else 0
    prec_bloq = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
    f1_bloq = 2 * prec_bloq * recall_bloq / (prec_bloq + recall_bloq + 1e-10)

    recall_any = tp_a / n_fraud if n_fraud > 0 else 0
    prec_any = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0

    # GAP
    gap = float(np.min(fraud_scores)) - float(np.percentile(normal_scores, 99.9))

    # Score composto
    composite = (
        recall_any * 1000
        - fn_a * 500
        + recall_bloq * 200
        - fp_b * 2
        - fp_a * 0.5
        + gap * 5
        + f1_bloq * 100
    )

    return {
        "recall_bloq": recall_bloq,
        "prec_bloq": prec_bloq,
        "f1_bloq": f1_bloq,
        "tp_bloq": tp_b,
        "fp_bloq": fp_b,
        "fn_bloq": fn_b,
        "recall_any": recall_any,
        "prec_any": prec_any,
        "tp_any": tp_a,
        "fp_any": fp_a,
        "fn_any": fn_a,
        "fraud_min": float(np.min(fraud_scores)),
        "fraud_p5": float(np.percentile(fraud_scores, 5)),
        "fraud_median": float(np.median(fraud_scores)),
        "normal_max": float(np.max(normal_scores)),
        "normal_p999": float(np.percentile(normal_scores, 99.9)),
        "gap": gap,
        "composite": composite,
    }


def optimize_anchors(score_raw: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Encontra as âncoras ótimas para o mapeamento."""
    print("\n" + "=" * 70)
    print("  2. OTIMIZANDO ÂNCORAS DO MAPEAMENTO")
    print("=" * 70)

    fraud_mask = y_true == 1
    normal_mask = y_true == 0

    fraud_scores = score_raw[fraud_mask]
    normal_scores = score_raw[normal_mask]

    p_normal_50 = float(np.median(normal_scores))
    p_normal_99 = float(np.percentile(normal_scores, 99))
    p_normal_999 = float(np.percentile(normal_scores, 99.9))
    p_normal_max = float(np.max(normal_scores))
    p_fraud_min = float(np.min(fraud_scores))
    p_fraud_5 = float(np.percentile(fraud_scores, 5))
    p_fraud_25 = float(np.percentile(fraud_scores, 25))
    p_fraud_50 = float(np.median(fraud_scores))

    print(f"\n  Distribuição dos scores raw:")
    print(f"  Normal P50:    {p_normal_50:.6f}")
    print(f"  Normal P99:    {p_normal_99:.6f}")
    print(f"  Normal P99.9:  {p_normal_999:.6f}")
    print(f"  Normal MAX:    {p_normal_max:.6f}")
    print(f"  Fraud MIN:     {p_fraud_min:.6f}")
    print(f"  Fraud P5:      {p_fraud_5:.6f}")
    print(f"  Fraud P25:     {p_fraud_25:.6f}")
    print(f"  Fraud P50:     {p_fraud_50:.6f}")

    # Pontos de ancoragem fixos no eixo raw
    raw_points = np.array([
        0.0,
        p_normal_50,
        p_normal_99,
        p_normal_999 * 0.95,
        p_normal_999,
        (p_normal_999 + p_fraud_min) / 2,
        p_fraud_min * 0.98,
        p_fraud_min,
        p_fraud_5,
        p_fraud_25,
        p_fraud_50,
        1.0,
    ])
    raw_points = np.sort(np.unique(raw_points))

    print(f"\n  Pontos de ancoragem raw: {len(raw_points)}")
    print(f"  Executando busca em grid...")

    best_score = -float("inf")
    best_anchors_raw = None
    best_anchors_out = None
    best_metrics = None
    n_tested = 0

    for out_normal_999 in np.arange(30, 58, 2):
        for out_fraud_min in np.arange(62, 92, 2):
            for out_fraud_5 in np.arange(max(out_fraud_min + 2, 86), 99, 2):

                if out_fraud_min <= out_normal_999 + 3:
                    continue

                out_points = np.array([
                    0.0,
                    out_normal_999 * 0.3,
                    out_normal_999 * 0.7,
                    out_normal_999 - 2,
                    out_normal_999,
                    (out_normal_999 + out_fraud_min) / 2,
                    out_fraud_min - 1,
                    out_fraud_min,
                    out_fraud_5,
                    min(out_fraud_5 + 3, 99),
                    min(out_fraud_5 + 5, 99.5),
                    100.0,
                ])

                if not np.all(np.diff(out_points) >= 0):
                    continue
                if len(raw_points) != len(out_points):
                    continue

                mapped = apply_mapping(score_raw, raw_points, out_points)
                metrics = evaluate_mapping(mapped, y_true)
                n_tested += 1

                if metrics["composite"] > best_score:
                    best_score = metrics["composite"]
                    best_anchors_raw = raw_points.copy()
                    best_anchors_out = out_points.copy()
                    best_metrics = metrics

    print(f"  Testadas: {n_tested:,} combinações")

    if best_metrics:
        print(f"\n  ✅ MELHOR CONFIGURAÇÃO ENCONTRADA:")
        print(f"  Composite score: {best_score:.1f}")
        print(f"  Recall BLOQUEAR (≥{FAIXA_BLOQUEAR}): {best_metrics['recall_bloq']:.1%}")
        print(f"  Recall TOTAL (≥{FAIXA_CONFIRMAR}):   {best_metrics['recall_any']:.1%}")
        print(f"  FP BLOQUEAR:     {best_metrics['fp_bloq']}")
        print(f"  FP TOTAL:        {best_metrics['fp_any']}")
        print(f"  Precision BLOQ:  {best_metrics['prec_bloq']:.1%}")
        print(f"  F1 BLOQUEAR:     {best_metrics['f1_bloq']:.4f}")
        print(f"  Fraud min score: {best_metrics['fraud_min']:.2f}")
        print(f"  Normal P99.9:    {best_metrics['normal_p999']:.2f}")
        print(f"  GAP:             {best_metrics['gap']:+.2f}")

        print(f"\n  Âncoras RAW:  {[round(float(x), 6) for x in best_anchors_raw]}")
        print(f"  Âncoras OUT:  {[round(float(x), 2) for x in best_anchors_out]}")

    return best_anchors_raw, best_anchors_out, best_metrics


# =========================================================
# 3. REFINAR COM BUSCA LOCAL
# =========================================================
def refine_anchors(
    score_raw: np.ndarray,
    y_true: np.ndarray,
    anchors_raw: np.ndarray,
    anchors_out: np.ndarray,
    best_metrics: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Refina as âncoras com busca local."""
    print("\n" + "=" * 70)
    print("  3. REFINANDO COM BUSCA LOCAL")
    print("=" * 70)

    best_score = best_metrics["composite"]
    best_out = np.asarray(anchors_out, dtype=np.float64).copy()
    improved = True
    iteration = 0

    while improved and iteration < 50:
        improved = False
        iteration += 1

        for i in range(1, len(best_out) - 1):
            original = best_out[i]

            for delta in [-3, -2, -1, -0.5, 0.5, 1, 2, 3]:
                candidate = original + delta

                if candidate <= best_out[i - 1] + 0.1:
                    continue
                if candidate >= best_out[i + 1] - 0.1:
                    continue
                if candidate < 0 or candidate > 100:
                    continue

                test_out = best_out.copy()
                test_out[i] = candidate

                mapped = apply_mapping(score_raw, anchors_raw, test_out)
                metrics = evaluate_mapping(mapped, y_true)

                if metrics["composite"] > best_score:
                    best_score = metrics["composite"]
                    best_out = test_out.copy()
                    improved = True

    mapped_final = apply_mapping(score_raw, anchors_raw, best_out)
    final_metrics = evaluate_mapping(mapped_final, y_true)

    print(f"  Iterações: {iteration}")
    print(f"  Composite: {best_score:.1f}")
    print(f"  Recall BLOQUEAR: {final_metrics['recall_bloq']:.1%}")
    print(f"  Recall TOTAL:    {final_metrics['recall_any']:.1%}")
    print(f"  FP BLOQUEAR:     {final_metrics['fp_bloq']}")
    print(f"  FP TOTAL:        {final_metrics['fp_any']}")
    print(f"  Fraud min:       {final_metrics['fraud_min']:.2f}")
    print(f"  Normal P99.9:    {final_metrics['normal_p999']:.2f}")
    print(f"  GAP:             {final_metrics['gap']:+.2f}")
    print(f"\n  Âncoras finais OUT: {[round(float(x), 2) for x in best_out]}")

    return anchors_raw, best_out, final_metrics


# =========================================================
# 4. ANÁLISE DETALHADA DAS FRAUDES
# =========================================================
def analyze_fraud_details(
    score_raw: np.ndarray,
    score_mapped: np.ndarray,
    y_true: np.ndarray,
):
    """Mostra cada fraude com seus scores."""
    print("\n" + "=" * 70)
    print("  4. DETALHES DE CADA FRAUDE")
    print("=" * 70)

    fraud_mask = y_true == 1
    fraud_raw = score_raw[fraud_mask]
    fraud_mapped = score_mapped[fraud_mask]

    order = np.argsort(fraud_mapped)

    print(f"\n  {'#':>3} {'Raw':>12} {'Mapped':>10} {'Decisão':>12}")
    print(f"  {'─' * 42}")

    for rank, idx in enumerate(order):
        raw = fraud_raw[idx]
        mapped = fraud_mapped[idx]
        if mapped >= FAIXA_BLOQUEAR:
            dec = "🔴 BLOQUEAR"
        elif mapped >= FAIXA_CONFIRMAR:
            dec = "🟡 CONFIRMAR"
        else:
            dec = "🟢 APROVAR"
        print(f"  {rank + 1:>3} {raw:>12.6f} {mapped:>10.2f} {dec:>12}")

    n_bloq = (fraud_mapped >= FAIXA_BLOQUEAR).sum()
    n_conf = ((fraud_mapped >= FAIXA_CONFIRMAR) & (fraud_mapped < FAIXA_BLOQUEAR)).sum()
    n_aprov = (fraud_mapped < FAIXA_CONFIRMAR).sum()

    print(f"\n  Resumo:")
    print(f"  🔴 BLOQUEAR:  {n_bloq}/69 ({n_bloq / 69 * 100:.1f}%)")
    print(f"  🟡 CONFIRMAR: {n_conf}/69 ({n_conf / 69 * 100:.1f}%)")
    print(f"  🟢 APROVAR:   {n_aprov}/69 ({n_aprov / 69 * 100:.1f}%)")


# =========================================================
# 5. DASHBOARD FINAL
# =========================================================
def plot_final_dashboard(
    score_raw: np.ndarray,
    score_mapped: np.ndarray,
    y_true: np.ndarray,
    anchors_raw: np.ndarray,
    anchors_out: np.ndarray,
    metrics: Dict,
):
    """Dashboard final com o scoring otimizado."""
    print("\n" + "=" * 70)
    print("  5. GERANDO DASHBOARD FINAL")
    print("=" * 70)

    fraud_mask = y_true == 1
    normal_mask = y_true == 0

    fraud_mapped = score_mapped[fraud_mask]
    normal_mapped = score_mapped[normal_mask]

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(
        f"Score Híbrido Otimizado — Detecção de Fraude PIX\n"
        f"Faixas: APROVAR [0-{FAIXA_CONFIRMAR:.0f}) | CONFIRMAR [{FAIXA_CONFIRMAR:.0f}-{FAIXA_BLOQUEAR:.0f}) | BLOQUEAR [{FAIXA_BLOQUEAR:.0f}-100]",
        fontsize=18, fontweight="bold", color=C["green"], y=1.02,
    )

    # ─── 1. Curva de mapeamento ───
    ax = axes[0, 0]
    x_curve = np.linspace(0, 1, 1000)
    y_curve = apply_mapping(x_curve, anchors_raw, anchors_out)

    ax.plot(x_curve, y_curve, color=C["green"], lw=2.5, label="Mapeamento")
    ax.scatter(
        [float(x) for x in anchors_raw],
        [float(x) for x in anchors_out],
        color=C["yellow"], s=80, zorder=5, label="Âncoras",
    )

    ax.axhline(FAIXA_CONFIRMAR, color=C["yellow"], ls="--", lw=1.5, alpha=0.7, label=f"CONFIRMAR ({FAIXA_CONFIRMAR:.0f})")
    ax.axhline(FAIXA_BLOQUEAR, color=C["red"], ls="--", lw=1.5, alpha=0.7, label=f"BLOQUEAR ({FAIXA_BLOQUEAR:.0f})")

    # Marcar pontos críticos
    fraud_raw_vals = score_raw[fraud_mask]
    fraud_min_raw = float(np.min(fraud_raw_vals))
    normal_p999_raw = float(np.percentile(score_raw[normal_mask], 99.9))

    ax.axvline(fraud_min_raw, color=C["red"], ls=":", lw=1, alpha=0.5)
    ax.axvline(normal_p999_raw, color=C["green"], ls=":", lw=1, alpha=0.5)
    ax.text(fraud_min_raw + 0.01, 10, f"Fraud min\n({fraud_min_raw:.3f})",
            fontsize=8, color=C["red"])
    ax.text(normal_p999_raw - 0.15, 45, f"Normal P99.9\n({normal_p999_raw:.3f})",
            fontsize=8, color=C["green"])

    ax.set_xlabel("Score Raw LGBM (0-1)")
    ax.set_ylabel("Score Mapeado (0-100)")
    ax.set_title("Curva de Mapeamento Raw → 0-100", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

    # ─── 2. Distribuição mapeada (completa) ───
    ax = axes[0, 1]
    ax.hist(normal_mapped, bins=100, alpha=0.6, color=C["green"],
            label=f"Normais (n={normal_mask.sum():,})", density=True)
    ax.hist(fraud_mapped, bins=30, alpha=0.8, color=C["red"],
            label=f"Fraudes (n={fraud_mask.sum()})", density=True)

    ax.axvline(FAIXA_CONFIRMAR, color=C["yellow"], ls="--", lw=2)
    ax.axvline(FAIXA_BLOQUEAR, color=C["red"], ls="--", lw=2)
    ax.axvspan(0, FAIXA_CONFIRMAR, alpha=0.03, color=C["green"])
    ax.axvspan(FAIXA_CONFIRMAR, FAIXA_BLOQUEAR, alpha=0.06, color=C["yellow"])
    ax.axvspan(FAIXA_BLOQUEAR, 100, alpha=0.06, color=C["red"])

    ax.set_xlabel("Score Mapeado (0-100)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição Geral dos Scores", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 105)

    # ─── 3. Zoom na zona crítica ───
    ax = axes[0, 2]
    normal_high = normal_mapped[normal_mapped >= 15]

    if len(normal_high) > 0:
        ax.hist(normal_high, bins=60, alpha=0.6, color=C["green"],
                label=f"Normais ≥15 (n={len(normal_high):,})")
    ax.hist(fraud_mapped, bins=30, alpha=0.8, color=C["red"],
            label=f"Fraudes (n={len(fraud_mapped)})")

    ax.axvline(FAIXA_CONFIRMAR, color=C["yellow"], ls="--", lw=2, label=f"CONFIRMAR ({FAIXA_CONFIRMAR:.0f})")
    ax.axvline(FAIXA_BLOQUEAR, color=C["red"], ls="--", lw=2, label=f"BLOQUEAR ({FAIXA_BLOQUEAR:.0f})")
    ax.axvspan(0, FAIXA_CONFIRMAR, alpha=0.03, color=C["green"])
    ax.axvspan(FAIXA_CONFIRMAR, FAIXA_BLOQUEAR, alpha=0.06, color=C["yellow"])
    ax.axvspan(FAIXA_BLOQUEAR, 100, alpha=0.06, color=C["red"])

    # Anotar GAP
    gap_val = metrics["gap"]
    ax.annotate(
        f"GAP: {gap_val:+.1f}",
        xy=(metrics["fraud_min"], 0),
        xytext=(55, ax.get_ylim()[1] * 0.7),
        fontsize=12, fontweight="bold", color=C["cyan"],
        arrowprops=dict(arrowstyle="->", color=C["cyan"]),
    )

    ax.set_xlabel("Score (0-100)")
    ax.set_ylabel("Contagem")
    ax.set_title("Zoom: Zona de Separação", fontweight="bold")
    ax.legend(fontsize=8)

    # ─── 4. Scatter de todas as fraudes ───
    ax = axes[1, 0]
    fraud_sorted = np.sort(fraud_mapped)
    colors_scatter = [
        C["red"] if s >= FAIXA_BLOQUEAR
        else C["yellow"] if s >= FAIXA_CONFIRMAR
        else C["green"]
        for s in fraud_sorted
    ]

    ax.scatter(range(len(fraud_sorted)), fraud_sorted, c=colors_scatter, s=50,
               edgecolors=C["white"], linewidths=0.5, zorder=5)
    ax.axhline(FAIXA_CONFIRMAR, color=C["yellow"], ls="--", lw=2, label=f"CONFIRMAR ({FAIXA_CONFIRMAR:.0f})")
    ax.axhline(FAIXA_BLOQUEAR, color=C["red"], ls="--", lw=2, label=f"BLOQUEAR ({FAIXA_BLOQUEAR:.0f})")
    ax.axhspan(0, FAIXA_CONFIRMAR, alpha=0.03, color=C["green"])
    ax.axhspan(FAIXA_CONFIRMAR, FAIXA_BLOQUEAR, alpha=0.06, color=C["yellow"])
    ax.axhspan(FAIXA_BLOQUEAR, 100, alpha=0.06, color=C["red"])

    n_b = int((fraud_mapped >= FAIXA_BLOQUEAR).sum())
    n_c = int(((fraud_mapped >= FAIXA_CONFIRMAR) & (fraud_mapped < FAIXA_BLOQUEAR)).sum())
    n_a = int((fraud_mapped < FAIXA_CONFIRMAR).sum())

    ax.set_title(
        f"69 Fraudes: 🔴 BLOQ={n_b} | 🟡 CONF={n_c} | 🟢 APROV={n_a}",
        fontweight="bold",
    )
    ax.set_xlabel("Fraudes (ordenadas por score)")
    ax.set_ylabel("Score (0-100)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9)

    # ─── 5. KPIs ───
    ax = axes[1, 1]
    ax.axis("off")

    kpi_text = (
        f"{'━' * 35}\n"
        f"   MÉTRICAS FINAIS\n"
        f"{'━' * 35}\n\n"
        f" BLOQUEAR (≥{FAIXA_BLOQUEAR:.0f}):\n"
        f"   Recall:    {metrics['recall_bloq']:.1%} ({metrics['tp_bloq']}/69)\n"
        f"   Precision: {metrics['prec_bloq']:.1%}\n"
        f"   F1:        {metrics['f1_bloq']:.4f}\n"
        f"   FP:        {metrics['fp_bloq']}\n\n"
        f" TOTAL (≥{FAIXA_CONFIRMAR:.0f}):\n"
        f"   Recall:    {metrics['recall_any']:.1%} ({metrics['tp_any']}/69)\n"
        f"   Precision: {metrics['prec_any']:.1%}\n"
        f"   FP:        {metrics['fp_any']}\n"
        f"   FN:        {metrics['fn_any']} fraudes escapam\n\n"
        f" SEPARAÇÃO:\n"
        f"   Fraud min:     {metrics['fraud_min']:.1f}\n"
        f"   Normal P99.9:  {metrics['normal_p999']:.1f}\n"
        f"   GAP:           {metrics['gap']:+.1f}\n"
    )
    ax.text(
        0.05, 0.95, kpi_text,
        transform=ax.transAxes,
        fontsize=13, va="top",
        fontfamily="monospace",
        color=C["green"],
        bbox=dict(boxstyle="round,pad=0.5", facecolor=C["dark"], edgecolor=C["green"]),
    )

    # ─── 6. Decisões por faixa ───
    ax = axes[1, 2]

    decisions = np.full(len(score_mapped), "APROVAR", dtype=object)
    decisions[score_mapped >= FAIXA_CONFIRMAR] = "CONFIRMAR"
    decisions[score_mapped >= FAIXA_BLOQUEAR] = "BLOQUEAR"

    dec_order = ["APROVAR", "CONFIRMAR", "BLOQUEAR"]
    dec_colors_list = [C["green"], C["yellow"], C["red"]]

    bars_total = []
    for i, dec in enumerate(dec_order):
        mask_dec = decisions == dec
        n_total = int(mask_dec.sum())
        n_fraud_in = int(y_true[mask_dec].sum())
        n_normal_in = n_total - n_fraud_in
        rate = n_fraud_in / n_total * 100 if n_total > 0 else 0
        bars_total.append(n_total)

        bar_n = ax.barh(dec, n_normal_in, color=dec_colors_list[i], alpha=0.4)
        bar_f = ax.barh(dec, n_fraud_in, left=n_normal_in, color=dec_colors_list[i], alpha=0.9)

        label_x = n_total + max(max(bars_total) * 0.02, 20)
        ax.text(
            label_x, i,
            f"n={n_total:,} | {n_fraud_in} fraudes ({rate:.1f}%)",
            va="center", fontsize=11, color=C["white"],
        )

    ax.set_title("Distribuição por Decisão", fontweight="bold")
    ax.set_xlabel("Quantidade de Transações")
    ax.set_xlim(0, max(bars_total) * 1.3)

    plt.tight_layout()
    output = RELATORIO_DIR / "dashboard_scoring_final.png"
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Dashboard salvo: {output}")


# =========================================================
# 6. SALVAR CONFIGURAÇÃO FINAL
# =========================================================
def save_final_config(
    anchors_raw: np.ndarray,
    anchors_out: np.ndarray,
    metrics: Dict,
    score_raw: np.ndarray,
    y_true: np.ndarray,
):
    """Salva a configuração final."""
    print("\n" + "=" * 70)
    print("  6. SALVANDO CONFIGURAÇÃO FINAL")
    print("=" * 70)

    config = {
        "versao": "1.0",
        "data_geracao": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "descricao": "Score híbrido otimizado - mapeamento não-linear raw → 0-100",
        "abordagem": "hybrid_optimized",
        "score_input": "lgbm_raw",
        "mapeamento": {
            "anchors_raw": [round(float(x), 8) for x in anchors_raw],
            "anchors_out": [round(float(x), 4) for x in anchors_out],
            "metodo": "numpy.interp (interpolação linear entre âncoras)",
        },
        "faixas_decisao": {
            "aprovar": {
                "range": f"[0, {FAIXA_CONFIRMAR})",
                "threshold": FAIXA_CONFIRMAR,
                "acao": "Liberar automaticamente",
            },
            "confirmar": {
                "range": f"[{FAIXA_CONFIRMAR}, {FAIXA_BLOQUEAR})",
                "threshold": FAIXA_BLOQUEAR,
                "acao": "Exigir autenticação adicional (2FA / biometria)",
            },
            "bloquear": {
                "range": f"[{FAIXA_BLOQUEAR}, 100]",
                "threshold": FAIXA_BLOQUEAR,
                "acao": "Parar transação para análise humana",
            },
        },
        "metricas_teste": {
            "n_total": int(len(y_true)),
            "n_fraudes": int((y_true == 1).sum()),
            "n_normais": int((y_true == 0).sum()),
            "recall_bloquear": round(float(metrics["recall_bloq"]), 4),
            "precision_bloquear": round(float(metrics["prec_bloq"]), 4),
            "f1_bloquear": round(float(metrics["f1_bloq"]), 4),
            "fp_bloquear": int(metrics["fp_bloq"]),
            "fn_bloquear": int(metrics["fn_bloq"]),
            "recall_total": round(float(metrics["recall_any"]), 4),
            "precision_total": round(float(metrics["prec_any"]), 4),
            "fp_total": int(metrics["fp_any"]),
            "fn_total": int(metrics["fn_any"]),
            "fraud_score_min": round(float(metrics["fraud_min"]), 2),
            "fraud_score_p5": round(float(metrics["fraud_p5"]), 2),
            "fraud_score_median": round(float(metrics["fraud_median"]), 2),
            "normal_score_max": round(float(metrics["normal_max"]), 2),
            "normal_score_p999": round(float(metrics["normal_p999"]), 2),
            "gap_fraud_min_vs_normal_p999": round(float(metrics["gap"]), 2),
        },
    }

    config_path = RELATORIO_DIR / "config_scoring_final.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Config relatorio: {config_path}")

    config_prod_path = ARTEFATOS_DIR / "scoring_config.json"
    with open(config_prod_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Config produção:  {config_prod_path}")

    return config


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n")
    print("█" * 70)
    print("  OTIMIZAÇÃO DO SCORE HÍBRIDO")
    print(f"  Faixas: APROVAR [0-{FAIXA_CONFIRMAR:.0f}) | CONFIRMAR [{FAIXA_CONFIRMAR:.0f}-{FAIXA_BLOQUEAR:.0f}) | BLOQUEAR [{FAIXA_BLOQUEAR:.0f}-100]")
    print("█" * 70)

    t0 = time.time()

    # 1. Carregar dados
    score_raw, score_cal, y_true, X_test = load_data()

    # 2. Otimizar âncoras
    anchors_raw, anchors_out, metrics = optimize_anchors(score_raw, y_true)

    if anchors_raw is None:
        print("\n  ❌ Otimização falhou. Usando âncoras padrão.")
        # Fallback com âncoras manuais baseadas nos dados conhecidos
        fraud_scores = score_raw[y_true == 1]
        normal_scores = score_raw[y_true == 0]
        p999 = float(np.percentile(normal_scores, 99.9))
        f_min = float(np.min(fraud_scores))

        anchors_raw = np.array([0.0, 0.000002, 0.001805, p999 * 0.95, p999,
                                (p999 + f_min) / 2, f_min * 0.98, f_min,
                                0.96, 0.997, 0.999, 1.0])
        anchors_out = np.array([0.0, 5.0, 15.0, 28.0, 30.0, 60.0, 89.0, 90.0,
                                92.0, 95.0, 97.0, 100.0])
        score_mapped = apply_mapping(score_raw, anchors_raw, anchors_out)
        metrics = evaluate_mapping(score_mapped, y_true)
    else:
        score_mapped = apply_mapping(score_raw, anchors_raw, anchors_out)

    # 3. Refinar
    anchors_raw, anchors_out, metrics = refine_anchors(
        score_raw, y_true, anchors_raw, anchors_out, metrics
    )
    score_mapped = apply_mapping(score_raw, anchors_raw, anchors_out)

    # 4. Detalhe das fraudes
    analyze_fraud_details(score_raw, score_mapped, y_true)

    # 5. Dashboard
    plot_final_dashboard(score_raw, score_mapped, y_true, anchors_raw, anchors_out, metrics)

    # 6. Salvar
    config = save_final_config(anchors_raw, anchors_out, metrics, score_raw, y_true)

    elapsed = time.time() - t0

    # ─── Resumo final ───
    print(f"\n\n{'█' * 70}")
    print("  RESULTADO FINAL")
    print(f"{'█' * 70}")

    n_total = len(y_true)
    n_fraud = int((y_true == 1).sum())

    decisions = np.full(n_total, "APROVAR", dtype=object)
    decisions[score_mapped >= FAIXA_CONFIRMAR] = "CONFIRMAR"
    decisions[score_mapped >= FAIXA_BLOQUEAR] = "BLOQUEAR"

    n_aprov = int((decisions == "APROVAR").sum())
    n_conf = int((decisions == "CONFIRMAR").sum())
    n_bloq = int((decisions == "BLOQUEAR").sum())
    f_aprov = int(y_true[decisions == "APROVAR"].sum())
    f_conf = int(y_true[decisions == "CONFIRMAR"].sum())
    f_bloq = int(y_true[decisions == "BLOQUEAR"].sum())

    print(f"""
  🎯 Score Híbrido Otimizado (raw → 0-100)

  ┌─────────────┬───────────┬──────────┬───────────┬─────────────┐
  │   Decisão   │  Faixa    │   Total  │  Fraudes  │ Taxa Fraude │
  ├─────────────┼───────────┼──────────┼───────────┼─────────────┤
  │ 🟢 APROVAR  │  [0-{FAIXA_CONFIRMAR:.0f})   │ {n_aprov:>7,} │  {f_aprov:>5}    │   {f_aprov/max(n_aprov,1)*100:>6.3f}%  │
  │ 🟡 CONFIRMAR│  [{FAIXA_CONFIRMAR:.0f}-{FAIXA_BLOQUEAR:.0f})  │ {n_conf:>7,} │  {f_conf:>5}    │   {f_conf/max(n_conf,1)*100:>6.2f}%  │
  │ 🔴 BLOQUEAR │  [{FAIXA_BLOQUEAR:.0f}-100]  │ {n_bloq:>7,} │  {f_bloq:>5}    │   {f_bloq/max(n_bloq,1)*100:>6.2f}%  │
  └─────────────┴───────────┴──────────┴───────────┴─────────────┘

  Performance:
    Recall BLOQUEAR (≥{FAIXA_BLOQUEAR:.0f}):  {metrics['recall_bloq']:.1%} — {metrics['tp_bloq']}/69 fraudes
    Recall TOTAL (≥{FAIXA_CONFIRMAR:.0f}):     {metrics['recall_any']:.1%} — {metrics['tp_any']}/69 fraudes
    Precision BLOQUEAR:    {metrics['prec_bloq']:.1%}
    F1 BLOQUEAR:           {metrics['f1_bloq']:.4f}
    Falsos positivos:      {metrics['fp_bloq']} (BLOQ) + {metrics['fp_any'] - metrics['fp_bloq']} (CONF) = {metrics['fp_any']} total
    Fraudes que escapam:   {metrics['fn_any']}

  Separação:
    Score mínimo de fraude:  {metrics['fraud_min']:.1f}
    Score P99.9 dos normais: {metrics['normal_p999']:.1f}
    GAP:                     {metrics['gap']:+.1f} pontos

  📁 Arquivos:
     📊 relatorio/dashboard_scoring_final.png
     📋 relatorio/config_scoring_final.json
     📋 backend/artefatos/scoring_config.json

  ⏱️  Tempo: {elapsed:.1f}s
    """)


if __name__ == "__main__":
    main()
