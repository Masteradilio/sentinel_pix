"""
avaliar_se_retroativo.py — Frente 1: Validação Retroativa do Módulo de Engenharia Social

Objetivo:
  Rodar o SocialEngineeringDetector em TODAS as transações do dataset
  e medir sua efetividade isolada + complementaridade com o LGBM.

Métricas geradas:
  1. Confusion matrix do SE em múltiplos thresholds de se_score
  2. Precision/Recall/F1 por padrão individual
  3. Taxa de ativação de cada indicador (fraude vs normal)
  4. Overlap entre padrões (co-ocorrência)
  5. Complementaridade: SE detectou algo que LGBM não detectou?
  6. Information Gain de cada indicador para discriminar fraude

Outputs:
  - relatorio/se_validacao_metricas.json
  - relatorio/se_validacao_por_padrao.csv
  - relatorio/se_validacao_indicadores.csv
  - relatorio/se_validacao_overlap.csv
  - relatorio/se_validacao_complementaridade.csv
  - relatorio/se_validacao_detalhado.csv
  - relatorio/se_validacao_relatorio.html

Uso:
  python avaliar_se_retroativo.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================
# PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "backend" else SCRIPT_DIR

# Tentar detectar raiz do projeto
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]:
    if (candidate / "backend").exists() and (candidate / "dados").exists():
        PROJECT_ROOT = candidate
        break

BACKEND_DIR = PROJECT_ROOT / "backend"
DADOS_DIR = PROJECT_ROOT / "dados"
RELATORIO_DIR = PROJECT_ROOT / "relatorio"
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"
CORE_DIR = BACKEND_DIR / "core"

RELATORIO_DIR.mkdir(parents=True, exist_ok=True)

# Garantir imports
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import do SE detector
from core.social_engineering import SocialEngineeringDetector, SEAnalysisResult

# =========================================================
# CONFIG
# =========================================================
INPUT_DATA = DADOS_DIR / "base_mvp_model_ready_optimized.csv"

# Outputs
OUT_METRICAS = RELATORIO_DIR / "se_validacao_metricas.json"
OUT_POR_PADRAO = RELATORIO_DIR / "se_validacao_por_padrao.csv"
OUT_INDICADORES = RELATORIO_DIR / "se_validacao_indicadores.csv"
OUT_OVERLAP = RELATORIO_DIR / "se_validacao_overlap.csv"
OUT_COMPLEMENTARIDADE = RELATORIO_DIR / "se_validacao_complementaridade.csv"
OUT_DETALHADO = RELATORIO_DIR / "se_validacao_detalhado.csv"
OUT_HTML = RELATORIO_DIR / "se_validacao_relatorio.html"

# Se existir predições do LGBM, usar para complementaridade
LGBM_PREDICTIONS = ARTEFATOS_DIR / "predicoes_teste_lightgbm.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("se_validacao")


# =========================================================
# HELPERS
# =========================================================
def safe_float(val: Any, default: float = 0.0) -> float:
    """Converte valor para float de forma segura."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        v = float(val)
        return default if np.isnan(v) or np.isinf(v) else v
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Converte valor para int de forma segura."""
    return int(safe_float(val, float(default)))


def row_to_features(row: pd.Series) -> Dict[str, Any]:
    """Converte uma linha do DataFrame em dict para o SE detector."""
    result = {}
    for col in row.index:
        val = row[col]
        if isinstance(val, (np.integer,)):
            result[col] = int(val)
        elif isinstance(val, (np.floating,)):
            result[col] = float(val) if not np.isnan(val) else None
        elif isinstance(val, (np.bool_,)):
            result[col] = bool(val)
        elif isinstance(val, pd.Timestamp):
            result[col] = val.to_pydatetime()
        elif pd.isna(val):
            result[col] = None
        else:
            result[col] = val
    return result


def information_gain(y: np.ndarray, indicator: np.ndarray) -> float:
    """
    Calcula Information Gain de um indicador binário para prever fraude.

    IG(Y|X) = H(Y) - H(Y|X)
    """
    n = len(y)
    if n == 0:
        return 0.0

    # Entropia de Y
    p1 = y.mean()
    p0 = 1 - p1
    if p1 == 0 or p1 == 1:
        return 0.0
    h_y = -(p1 * np.log2(p1) + p0 * np.log2(p0))

    # Entropia condicional H(Y|X)
    h_y_given_x = 0.0
    for x_val in [0, 1]:
        mask = indicator == x_val
        n_x = mask.sum()
        if n_x == 0:
            continue
        p1_x = y[mask].mean()
        p0_x = 1 - p1_x
        if p1_x == 0 or p1_x == 1:
            h_x = 0.0
        else:
            h_x = -(p1_x * np.log2(p1_x) + p0_x * np.log2(p0_x))
        h_y_given_x += (n_x / n) * h_x

    return h_y - h_y_given_x


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calcula precision, recall, F1."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 70)
    print("FRENTE 1 — Validação Retroativa do Módulo de Engenharia Social")
    print("=" * 70)

    # ─── 1. Carregar dados ──────────────────────────────────
    print(f"\n[1/7] Carregando dados de {INPUT_DATA}...")
    if not INPUT_DATA.exists():
        logger.error(f"Arquivo não encontrado: {INPUT_DATA}")
        sys.exit(1)

    df = pd.read_csv(INPUT_DATA, low_memory=False)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")

    n_total = len(df)
    n_fraud = int(df["is_fraud"].sum())
    n_normal = n_total - n_fraud
    print(f"  Total: {n_total:,} transações")
    print(f"  Fraudes: {n_fraud:,} ({n_fraud/n_total*100:.2f}%)")
    print(f"  Normais: {n_normal:,}")

    # ─── 2. Rodar SE em todas as transações ─────────────────
    print(f"\n[2/7] Rodando SocialEngineeringDetector em {n_total:,} transações...")
    detector = SocialEngineeringDetector()

    results = []
    t0 = time.perf_counter()
    log_interval = max(1, n_total // 20)

    for idx in range(n_total):
        row = df.iloc[idx]
        features = row_to_features(row)

        se_result: SEAnalysisResult = detector.detect_from_pipeline(features)

        pattern_names = [p.pattern_name for p in se_result.patterns]
        pattern_severities = [p.severity for p in se_result.patterns]
        pattern_scores = [p.score for p in se_result.patterns]

        # Coletar indicadores ativos
        active_indicators = {
            k: v for k, v in se_result.active_indicators.items() if v
        }

        results.append({
            "idx": idx,
            "transaction_id": features.get("transaction_id", ""),
            "customer_id": features.get("customer_id", ""),
            "is_fraud": int(row.get("is_fraud", 0)),
            "se_score": se_result.se_score,
            "n_patterns": len(se_result.patterns),
            "patterns": "|".join(pattern_names),
            "severities": "|".join(pattern_severities),
            "pattern_scores": "|".join(str(s) for s in pattern_scores),
            "worst_pattern": se_result.worst_pattern.pattern_name if se_result.worst_pattern else "",
            "worst_severity": se_result.worst_pattern.severity if se_result.worst_pattern else "",
            "n_active_indicators": len(active_indicators),
            "active_indicators": "|".join(sorted(active_indicators.keys())),
            # Features-chave para análise
            "vl_pix": safe_float(features.get("vl_pix")),
            "nr_idade": safe_int(features.get("nr_idade")),
            "pix_key_random_flag": safe_int(features.get("pix_key_random_flag")),
            "first_receiver_flag": safe_int(features.get("first_receiver_flag")),
            "burst_30m_flag": safe_int(features.get("burst_30m_flag")),
            "qt_tempo_relacionamento_mes": safe_float(
                features.get("qt_tempo_relacionamento_mes")
            ),
        })

        if (idx + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / rate
            print(
                f"  {idx+1:>7,}/{n_total:,} "
                f"({(idx+1)/n_total*100:.0f}%) | "
                f"{rate:.0f} tx/s | ETA {eta:.0f}s"
            )

    elapsed_total = time.perf_counter() - t0
    print(f"  ✓ Concluído em {elapsed_total:.1f}s ({n_total/elapsed_total:.0f} tx/s)")

    df_results = pd.DataFrame(results)

    # Salvar detalhado
    df_results.to_csv(OUT_DETALHADO, index=False)
    print(f"  Salvo: {OUT_DETALHADO}")

    # ─── 3. Métricas globais do SE ──────────────────────────
    print(f"\n[3/7] Calculando métricas globais...")

    fraud_mask = df_results["is_fraud"] == 1
    normal_mask = df_results["is_fraud"] == 0

    # SE ativou (se_score > 0) = detectou algum padrão
    se_any = df_results["se_score"] > 0
    se_fraud_any = (se_any & fraud_mask).sum()
    se_normal_any = (se_any & normal_mask).sum()

    print(f"\n  SE ativou em:")
    print(f"    Fraudes:  {se_fraud_any}/{n_fraud} ({se_fraud_any/n_fraud*100:.1f}%)")
    print(f"    Normais:  {se_normal_any}/{n_normal} ({se_normal_any/n_normal*100:.2f}%)")

    # Métricas por threshold de se_score
    thresholds_se = [0, 10, 15, 20, 25, 30, 40, 50, 60, 80]
    threshold_metrics = []

    print(f"\n  {'Threshold':>10} {'TP':>5} {'FP':>6} {'FN':>5} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FPR':>8}")
    print(f"  {'-'*62}")

    for th in thresholds_se:
        se_positive = df_results["se_score"] > th
        tp = int((se_positive & fraud_mask).sum())
        fp = int((se_positive & normal_mask).sum())
        fn = int((~se_positive & fraud_mask).sum())
        tn = int((~se_positive & normal_mask).sum())

        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        fpr = fp / max(fp + tn, 1)

        threshold_metrics.append({
            "threshold": th,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 6),
        })

        print(
            f"  {th:>10} {tp:>5} {fp:>6} {fn:>5} "
            f"{rec:>8.4f} {prec:>10.4f} {f1:>8.4f} {fpr:>8.6f}"
        )

    # ─── 4. Métricas por padrão ─────────────────────────────
    print(f"\n[4/7] Analisando cada padrão individualmente...")

    all_patterns = set()
    for pstr in df_results["patterns"]:
        if pstr:
            all_patterns.update(pstr.split("|"))
    all_patterns.discard("")

    pattern_metrics = []
    for pattern_name in sorted(all_patterns):
        # Transações onde este padrão ativou
        pattern_mask = df_results["patterns"].str.contains(pattern_name, na=False)
        tp = int((pattern_mask & fraud_mask).sum())
        fp = int((pattern_mask & normal_mask).sum())
        fn = n_fraud - tp  # Fraudes que este padrão NÃO detectou
        tn = n_normal - fp

        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        fpr = fp / max(fp + tn, 1)
        total_ativacoes = int(pattern_mask.sum())
        pct_fraudes = tp / max(total_ativacoes, 1) * 100

        pattern_metrics.append({
            "pattern": pattern_name,
            "total_ativacoes": total_ativacoes,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 6),
            "pct_fraudes_quando_ativa": round(pct_fraudes, 2),
        })

    df_patterns = pd.DataFrame(pattern_metrics).sort_values("tp", ascending=False)
    df_patterns.to_csv(OUT_POR_PADRAO, index=False)

    print(f"\n  {'Padrão':<35} {'Ativ':>5} {'TP':>4} {'FP':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'-'*75}")
    for _, row in df_patterns.iterrows():
        print(
            f"  {row['pattern']:<35} {row['total_ativacoes']:>5} "
            f"{row['tp']:>4} {row['fp']:>5} "
            f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f}"
        )

    # ─── 5. Análise de indicadores ──────────────────────────
    print(f"\n[5/7] Analisando indicadores individuais...")

    # Coletar todos os indicadores que apareceram
    all_indicators: Counter = Counter()
    indicator_fraud: Counter = Counter()
    indicator_normal: Counter = Counter()

    for _, row in df_results.iterrows():
        indicators = row["active_indicators"]
        if not indicators:
            continue
        ind_list = indicators.split("|")
        is_fraud = row["is_fraud"] == 1

        for ind in ind_list:
            all_indicators[ind] += 1
            if is_fraud:
                indicator_fraud[ind] += 1
            else:
                indicator_normal[ind] += 1

    # Calcular métricas por indicador
    indicator_metrics = []
    for ind_name in sorted(all_indicators.keys()):
        total = all_indicators[ind_name]
        in_fraud = indicator_fraud.get(ind_name, 0)
        in_normal = indicator_normal.get(ind_name, 0)

        # Taxas
        rate_fraud = in_fraud / max(n_fraud, 1)
        rate_normal = in_normal / max(n_normal, 1)
        lift = rate_fraud / max(rate_normal, 1e-9)

        # Information Gain (binário: indicador ativo ou não)
        ind_binary = np.zeros(n_total)
        # Reconstruir array binário do indicador
        for idx_r, row in df_results.iterrows():
            inds = row["active_indicators"]
            if inds and ind_name in inds.split("|"):
                ind_binary[idx_r] = 1

        ig = information_gain(
            df_results["is_fraud"].values.astype(float),
            ind_binary,
        )

        indicator_metrics.append({
            "indicator": ind_name,
            "total_ativacoes": total,
            "em_fraudes": in_fraud,
            "em_normais": in_normal,
            "rate_fraude": round(rate_fraud, 4),
            "rate_normal": round(rate_normal, 6),
            "lift": round(lift, 2),
            "information_gain": round(ig, 6),
        })

    df_indicators = pd.DataFrame(indicator_metrics).sort_values(
        "lift", ascending=False
    )
    df_indicators.to_csv(OUT_INDICADORES, index=False)

    print(f"\n  Top 20 indicadores por Lift (fraude vs normal):")
    print(f"  {'Indicador':<40} {'Fraude%':>8} {'Normal%':>8} {'Lift':>7} {'IG':>8}")
    print(f"  {'-'*75}")
    for _, row in df_indicators.head(20).iterrows():
        print(
            f"  {row['indicator']:<40} "
            f"{row['rate_fraude']*100:>7.1f}% "
            f"{row['rate_normal']*100:>7.3f}% "
            f"{row['lift']:>7.1f} "
            f"{row['information_gain']:>8.6f}"
        )

    # ─── 6. Análise de overlap entre padrões ────────────────
    print(f"\n[6/7] Analisando overlap entre padrões...")

    # Matriz de co-ocorrência
    pattern_list = sorted(all_patterns)
    n_patterns = len(pattern_list)
    cooccurrence = np.zeros((n_patterns, n_patterns), dtype=int)

    for _, row in df_results.iterrows():
        pstr = row["patterns"]
        if not pstr:
            continue
        active = set(pstr.split("|"))
        for i, p1 in enumerate(pattern_list):
            if p1 not in active:
                continue
            for j, p2 in enumerate(pattern_list):
                if p2 in active:
                    cooccurrence[i, j] += 1

    df_overlap = pd.DataFrame(
        cooccurrence, index=pattern_list, columns=pattern_list
    )
    df_overlap.to_csv(OUT_OVERLAP)

    # Calcular Jaccard similarity para pares com overlap
    print(f"\n  Pares de padrões com overlap significativo:")
    print(f"  {'Par':<65} {'Co-oc':>6} {'Jaccard':>8}")
    print(f"  {'-'*82}")

    overlap_pairs = []
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            co = cooccurrence[i, j]
            if co == 0:
                continue
            union = cooccurrence[i, i] + cooccurrence[j, j] - co
            jaccard = co / max(union, 1)
            overlap_pairs.append({
                "pattern_a": pattern_list[i],
                "pattern_b": pattern_list[j],
                "co_occurrence": co,
                "jaccard": round(jaccard, 4),
            })

    overlap_pairs.sort(key=lambda x: x["jaccard"], reverse=True)
    for pair in overlap_pairs[:15]:
        label = f"{pair['pattern_a']} × {pair['pattern_b']}"
        print(f"  {label:<65} {pair['co_occurrence']:>6} {pair['jaccard']:>8.4f}")

    # ─── 7. Complementaridade com LGBM ──────────────────────
    print(f"\n[7/7] Analisando complementaridade com LGBM...")

    complementarity_data = None
    if LGBM_PREDICTIONS.exists():
        try:
            lgbm_preds = pd.read_csv(LGBM_PREDICTIONS)

            # Merge com resultados SE
            comp = df_results[["transaction_id", "is_fraud", "se_score", "n_patterns", "patterns"]].merge(
                lgbm_preds[["transaction_id", "score_fraude", "pred_lgbm", "pred_combined"]],
                on="transaction_id",
                how="inner",
            )

            n_comp = len(comp)
            print(f"  Merge com LGBM: {n_comp:,} transações")

            comp_fraud = comp[comp["is_fraud"] == 1]
            n_comp_fraud = len(comp_fraud)

            # LGBM detectou (pred_combined = LGBM + cascade)
            lgbm_detected = comp_fraud["pred_combined"] == 1
            lgbm_missed = ~lgbm_detected

            # SE detectou (se_score > 0)
            se_detected = comp_fraud["se_score"] > 0

            # Complementaridade
            both = (lgbm_detected & se_detected).sum()
            only_lgbm = (lgbm_detected & ~se_detected).sum()
            only_se = (~lgbm_detected & se_detected).sum()
            neither = (~lgbm_detected & ~se_detected).sum()

            print(f"\n  Complementaridade nas {n_comp_fraud} fraudes (holdout):")
            print(f"    LGBM + SE detectaram:     {both}")
            print(f"    Só LGBM detectou:         {only_lgbm}")
            print(f"    Só SE detectou:           {only_se}")
            print(f"    Nenhum detectou:          {neither}")

            # Detalhar FN do LGBM que SE pegou
            fn_lgbm_se_caught = comp_fraud[lgbm_missed & se_detected]
            if len(fn_lgbm_se_caught) > 0:
                print(f"\n  🎯 SE capturou {len(fn_lgbm_se_caught)} fraudes que LGBM perdeu:")
                for _, r in fn_lgbm_se_caught.iterrows():
                    print(
                        f"    TX {r['transaction_id']} | "
                        f"LGBM={r['score_fraude']:.4f} | "
                        f"SE={r['se_score']:.0f} | "
                        f"Padrões: {r['patterns']}"
                    )
            else:
                print(f"\n  SE não capturou nenhuma fraude adicional vs LGBM+Cascade")

            # Para normais: SE ativou indevidamente?
            comp_normal = comp[comp["is_fraud"] == 0]
            se_fp = (comp_normal["se_score"] > 0).sum()
            lgbm_fp = (comp_normal["pred_combined"] == 1).sum()
            both_fp = ((comp_normal["se_score"] > 0) & (comp_normal["pred_combined"] == 1)).sum()

            print(f"\n  Falsos positivos (transações normais):")
            print(f"    SE FP (score > 0):        {se_fp} ({se_fp/len(comp_normal)*100:.2f}%)")
            print(f"    LGBM+Cascade FP:          {lgbm_fp} ({lgbm_fp/len(comp_normal)*100:.2f}%)")
            print(f"    Ambos FP:                 {both_fp}")

            complementarity_data = {
                "n_holdout_fraud": n_comp_fraud,
                "both_detected": int(both),
                "only_lgbm": int(only_lgbm),
                "only_se": int(only_se),
                "neither": int(neither),
                "se_incremental_frauds": int(only_se),
                "se_fp_count": int(se_fp),
                "se_fp_rate": round(se_fp / max(len(comp_normal), 1), 6),
                "lgbm_fp_count": int(lgbm_fp),
            }

            # Salvar complementaridade detalhada
            comp.to_csv(OUT_COMPLEMENTARIDADE, index=False)

        except Exception as e:
            logger.warning(f"Erro na análise de complementaridade: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠ Predições LGBM não encontradas em {LGBM_PREDICTIONS}")
        print(f"    Complementaridade não será calculada")

    # ─── Consolidar métricas ────────────────────────────────
    print(f"\n{'='*70}")
    print("CONSOLIDAÇÃO")
    print(f"{'='*70}")

    # Score distribution
    fraud_scores = df_results.loc[fraud_mask, "se_score"]
    normal_scores = df_results.loc[normal_mask, "se_score"]

    metricas_consolidadas = {
        "data_geracao": datetime.now().isoformat(),
        "dataset": str(INPUT_DATA),
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_normal": n_normal,
        "se_version": "2.1",
        "n_patterns_catalogados": 12,
        "n_indicadores_catalogados": len(detector.INDICATORS),
        "ativacao_global": {
            "fraudes_com_se_ativo": int(se_fraud_any),
            "fraudes_com_se_ativo_pct": round(se_fraud_any / n_fraud * 100, 2),
            "normais_com_se_ativo": int(se_normal_any),
            "normais_com_se_ativo_pct": round(se_normal_any / n_normal * 100, 4),
        },
        "score_distribution": {
            "fraud": {
                "mean": round(float(fraud_scores.mean()), 2),
                "median": round(float(fraud_scores.median()), 2),
                "std": round(float(fraud_scores.std()), 2),
                "min": round(float(fraud_scores.min()), 2),
                "max": round(float(fraud_scores.max()), 2),
                "pct_zero": round(float((fraud_scores == 0).mean() * 100), 2),
                "pct_gt_20": round(float((fraud_scores > 20).mean() * 100), 2),
                "pct_gt_40": round(float((fraud_scores > 40).mean() * 100), 2),
                "pct_gt_60": round(float((fraud_scores > 60).mean() * 100), 2),
            },
            "normal": {
                "mean": round(float(normal_scores.mean()), 2),
                "median": round(float(normal_scores.median()), 2),
                "std": round(float(normal_scores.std()), 2),
                "min": round(float(normal_scores.min()), 2),
                "max": round(float(normal_scores.max()), 2),
                "pct_zero": round(float((normal_scores == 0).mean() * 100), 2),
                "pct_gt_20": round(float((normal_scores > 20).mean() * 100), 2),
                "pct_gt_40": round(float((normal_scores > 40).mean() * 100), 2),
                "pct_gt_60": round(float((normal_scores > 60).mean() * 100), 2),
            },
        },
        "threshold_metrics": threshold_metrics,
        "pattern_metrics": df_patterns.to_dict(orient="records"),
        "top_indicators_by_lift": df_indicators.head(20).to_dict(orient="records"),
        "overlap_pairs_top10": overlap_pairs[:10],
        "complementarity": complementarity_data,
    }

    with open(OUT_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas_consolidadas, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Métricas salvas: {OUT_METRICAS}")

    # ─── Gerar HTML ─────────────────────────────────────────
    _generate_html_report(
        metricas_consolidadas,
        df_patterns,
        df_indicators,
        overlap_pairs,
    )

    # ─── Resumo Final ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESUMO — Validação Retroativa do Módulo SE")
    print(f"{'='*70}")

    print(f"\n  Dataset: {n_total:,} tx ({n_fraud} fraudes)")
    print(f"\n  SE ativou em:")
    print(f"    {se_fraud_any}/{n_fraud} fraudes ({se_fraud_any/n_fraud*100:.1f}%)")
    print(f"    {se_normal_any}/{n_normal} normais ({se_normal_any/n_normal*100:.2f}%)")
    print(f"\n  Score médio:")
    print(f"    Fraudes: {fraud_scores.mean():.1f} (mediana {fraud_scores.median():.0f})")
    print(f"    Normais: {normal_scores.mean():.1f} (mediana {normal_scores.median():.0f})")

    if complementarity_data:
        print(f"\n  Complementaridade (holdout):")
        print(f"    SE capturou {complementarity_data['se_incremental_frauds']} fraudes extras vs LGBM")
        print(f"    SE FP: {complementarity_data['se_fp_count']} ({complementarity_data['se_fp_rate']*100:.2f}%)")

    print(f"\n  Artefatos gerados:")
    for p in [OUT_METRICAS, OUT_POR_PADRAO, OUT_INDICADORES, OUT_OVERLAP,
              OUT_COMPLEMENTARIDADE, OUT_DETALHADO, OUT_HTML]:
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.name} ({size_kb:.0f} KB)")

    print(f"\n{'='*70}")


# =========================================================
# HTML REPORT GENERATOR
# =========================================================
def _generate_html_report(
    metricas: Dict,
    df_patterns: pd.DataFrame,
    df_indicators: pd.DataFrame,
    overlap_pairs: List[Dict],
):
    """Gera relatório HTML com os resultados da validação."""

    ativacao = metricas["ativacao_global"]
    score_fraud = metricas["score_distribution"]["fraud"]
    score_normal = metricas["score_distribution"]["normal"]
    comp = metricas.get("complementarity")

    # Tabela de thresholds
    th_rows = ""
    for m in metricas["threshold_metrics"]:
        th_rows += f"""
        <tr>
            <td>{m['threshold']}</td>
            <td class="highlight">{m['tp']}</td>
            <td class="warning">{m['fp']}</td>
            <td>{m['fn']}</td>
            <td>{m['recall']:.4f}</td>
            <td>{m['precision']:.4f}</td>
            <td>{m['f1']:.4f}</td>
            <td>{m['fpr']:.6f}</td>
        </tr>"""

    # Tabela de padrões
    pat_rows = ""
    for _, p in df_patterns.iterrows():
        prec_color = "#00d4aa" if p["precision"] > 0.5 else "#ffd93d" if p["precision"] > 0.1 else "#ff6b6b"
        pat_rows += f"""
        <tr>
            <td><strong>{p['pattern']}</strong></td>
            <td>{p['total_ativacoes']}</td>
            <td class="highlight">{p['tp']}</td>
            <td class="warning">{p['fp']}</td>
            <td style="color:{prec_color}">{p['precision']:.4f}</td>
            <td>{p['recall']:.4f}</td>
            <td>{p['f1']:.4f}</td>
            <td>{p['pct_fraudes_quando_ativa']:.1f}%</td>
        </tr>"""

    # Tabela de indicadores (top 25)
    ind_rows = ""
    for _, ind in df_indicators.head(25).iterrows():
        lift_color = "#00d4aa" if ind["lift"] > 10 else "#ffd93d" if ind["lift"] > 3 else "#ccc"
        ind_rows += f"""
        <tr>
            <td>{ind['indicator']}</td>
            <td>{ind['total_ativacoes']:,}</td>
            <td>{ind['em_fraudes']}</td>
            <td>{ind['rate_fraude']*100:.1f}%</td>
            <td>{ind['rate_normal']*100:.3f}%</td>
            <td style="color:{lift_color};font-weight:bold">{ind['lift']:.1f}x</td>
            <td>{ind['information_gain']:.6f}</td>
        </tr>"""

    # Complementaridade
    comp_html = ""
    if comp:
        comp_html = f"""
    <div class="section">
        <h2>🔗 Complementaridade SE × LGBM (Holdout)</h2>
        <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="kpi"><div class="value" style="color:#00d4aa">{comp['both_detected']}</div>
                <div class="label">Ambos detectaram</div></div>
            <div class="kpi"><div class="value" style="color:#6c5ce7">{comp['only_lgbm']}</div>
                <div class="label">Só LGBM</div></div>
            <div class="kpi"><div class="value" style="color:#ff9f43">{comp['only_se']}</div>
                <div class="label">Só SE</div></div>
            <div class="kpi"><div class="value" style="color:#ff6b6b">{comp['neither']}</div>
                <div class="label">Nenhum</div></div>
        </div>
        <div class="callout {'success' if comp['se_incremental_frauds'] > 0 else 'orange'}">
            <strong>Valor incremental do SE:</strong>
            {f"O SE capturou {comp['se_incremental_frauds']} fraude(s) que o LGBM+Cascade não detectou." if comp['se_incremental_frauds'] > 0
             else "No holdout, o LGBM+Cascade já capturou todas as fraudes. O SE funciona como rede de segurança para cenários futuros."}
        </div>
    </div>"""

    # Overlap
    overlap_rows = ""
    for pair in overlap_pairs[:10]:
        jaccard_color = "#ff6b6b" if pair["jaccard"] > 0.7 else "#ffd93d" if pair["jaccard"] > 0.4 else "#ccc"
        overlap_rows += f"""
        <tr>
            <td>{pair['pattern_a']}</td>
            <td>{pair['pattern_b']}</td>
            <td>{pair['co_occurrence']}</td>
            <td style="color:{jaccard_color};font-weight:bold">{pair['jaccard']:.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validação Retroativa — Módulo de Engenharia Social</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0e1117; color: #e0e0e0;
               padding: 30px 50px; line-height: 1.6; }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 3px solid #ff9f43;
                   padding-bottom: 20px; }}
        .header h1 {{ color: #ff9f43; font-size: 28px; }}
        .header .subtitle {{ color: #888; font-size: 14px; margin-top: 8px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
                     margin-bottom: 30px; }}
        .kpi {{ background: #1a1d23; border-radius: 12px; padding: 20px; text-align: center;
                border: 2px solid #333; }}
        .kpi .value {{ font-size: 36px; font-weight: 800; }}
        .kpi .label {{ font-size: 12px; color: #888; margin-top: 6px; text-transform: uppercase; }}
        .kpi .detail {{ font-size: 11px; color: #666; margin-top: 4px; }}
        .section {{ background: #1a1d23; border-radius: 12px; padding: 24px; margin-bottom: 20px;
                    border: 1px solid #2a2d33; }}
        .section h2 {{ color: #ff9f43; margin-bottom: 16px; font-size: 18px;
                       border-bottom: 1px solid #333; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
        th {{ color: #ff9f43; font-size: 11px; text-transform: uppercase; padding: 10px 12px;
              text-align: left; border-bottom: 2px solid #333; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #2a2d33; }}
        tr:hover {{ background: rgba(255,159,67,0.05); }}
        .highlight {{ color: #00d4aa; font-weight: bold; }}
        .warning {{ color: #ffd93d; }}
        .danger {{ color: #ff6b6b; font-weight: bold; }}
        .callout {{ background: rgba(255,159,67,0.08); border-left: 4px solid #ff9f43;
                    padding: 14px 18px; margin: 14px 0; border-radius: 0 8px 8px 0; }}
        .callout.success {{ border-left-color: #00d4aa; background: rgba(0,212,170,0.08); }}
        .callout.orange {{ border-left-color: #ff9f43; }}
        .footer {{ text-align: center; color: #555; font-size: 11px; margin-top: 40px;
                   padding-top: 16px; border-top: 1px solid #333; }}
    </style>
</head>
<body>

<div class="header">
    <h1>🎭 Validação Retroativa — Módulo de Engenharia Social v2.1</h1>
    <p class="subtitle">
        Frente 1 do Plano de Melhoria | Gerado em {metricas['data_geracao'][:19]} |
        {metricas['n_total']:,} transações ({metricas['n_fraud']} fraudes)
    </p>
</div>

<!-- KPIs -->
<div class="kpi-grid">
    <div class="kpi" style="border-color: #ff9f43;">
        <div class="value" style="color: #ff9f43;">{ativacao['fraudes_com_se_ativo_pct']}%</div>
        <div class="label">Fraudes com SE Ativo</div>
        <div class="detail">{ativacao['fraudes_com_se_ativo']} de {metricas['n_fraud']}</div>
    </div>
    <div class="kpi" style="border-color: #ffd93d;">
        <div class="value" style="color: #ffd93d;">{ativacao['normais_com_se_ativo_pct']}%</div>
        <div class="label">FPR do SE</div>
        <div class="detail">{ativacao['normais_com_se_ativo']} normais flaggadas</div>
    </div>
    <div class="kpi" style="border-color: #6c5ce7;">
        <div class="value" style="color: #6c5ce7;">{score_fraud['mean']}</div>
        <div class="label">Score Médio Fraudes</div>
        <div class="detail">vs {score_normal['mean']} em normais</div>
    </div>
</div>

<!-- Score Distribution -->
<div class="section">
    <h2>📊 Distribuição do SE Score</h2>
    <table>
        <tr><th>Métrica</th><th>Fraudes</th><th>Normais</th><th>Separação</th></tr>
        <tr><td>Média</td><td class="highlight">{score_fraud['mean']}</td>
            <td>{score_normal['mean']}</td>
            <td>{score_fraud['mean'] - score_normal['mean']:.1f} pts</td></tr>
        <tr><td>Mediana</td><td>{score_fraud['median']}</td>
            <td>{score_normal['median']}</td><td>—</td></tr>
        <tr><td>% com score = 0</td><td>{score_fraud['pct_zero']}%</td>
            <td>{score_normal['pct_zero']}%</td><td>—</td></tr>
        <tr><td>% com score &gt; 20</td><td class="highlight">{score_fraud['pct_gt_20']}%</td>
            <td>{score_normal['pct_gt_20']}%</td><td>—</td></tr>
        <tr><td>% com score &gt; 40</td><td class="highlight">{score_fraud['pct_gt_40']}%</td>
            <td>{score_normal['pct_gt_40']}%</td><td>—</td></tr>
        <tr><td>% com score &gt; 60</td><td>{score_fraud['pct_gt_60']}%</td>
            <td>{score_normal['pct_gt_60']}%</td><td>—</td></tr>
    </table>
</div>

<!-- Threshold Analysis -->
<div class="section">
    <h2>🎯 Performance do SE por Threshold</h2>
    <p style="color:#888;margin-bottom:12px;">
        Se usássemos o SE como detector isolado, qual seria a performance em cada threshold?
    </p>
    <table>
        <tr><th>Threshold</th><th>TP</th><th>FP</th><th>FN</th>
            <th>Recall</th><th>Precision</th><th>F1</th><th>FPR</th></tr>
        {th_rows}
    </table>
</div>

<!-- Per-Pattern -->
<div class="section">
    <h2>🎭 Performance por Padrão de Golpe</h2>
    <table>
        <tr><th>Padrão</th><th>Ativações</th><th>TP</th><th>FP</th>
            <th>Precision</th><th>Recall</th><th>F1</th><th>% Fraude</th></tr>
        {pat_rows}
    </table>
</div>

<!-- Indicators -->
<div class="section">
    <h2>🔍 Top 25 Indicadores por Poder Discriminativo (Lift)</h2>
    <p style="color:#888;margin-bottom:12px;">
        Lift = (taxa em fraudes) / (taxa em normais). Lift alto = indicador discriminativo.
    </p>
    <table>
        <tr><th>Indicador</th><th>Ativações</th><th>Em Fraudes</th>
            <th>Taxa Fraude</th><th>Taxa Normal</th><th>Lift</th><th>Info Gain</th></tr>
        {ind_rows}
    </table>
</div>

<!-- Overlap -->
<div class="section">
    <h2>🔄 Overlap entre Padrões (Jaccard Similarity)</h2>
    <p style="color:#888;margin-bottom:12px;">
        Jaccard &gt; 0.7 = padrões quase idênticos (candidatos a merge).
        Jaccard &lt; 0.3 = padrões independentes (bom).
    </p>
    <table>
        <tr><th>Padrão A</th><th>Padrão B</th><th>Co-ocorrência</th><th>Jaccard</th></tr>
        {overlap_rows}
    </table>
</div>

{comp_html}

<div class="section">
    <h2>📋 Conclusões e Próximos Passos</h2>
    <div class="callout orange">
        <strong>Este relatório é a Frente 1 do plano de melhoria do módulo SE.</strong><br>
        Com estes dados, as próximas frentes são:<br>
        • <strong>Frente 2:</strong> Calibrar min_score de cada padrão usando curvas precision-recall<br>
        • <strong>Frente 3:</strong> Desduplicar padrões com Jaccard &gt; 0.7<br>
        • <strong>Frente 4:</strong> Documentar cada regra com evidência + referência regulatória
    </div>
</div>

<div class="footer">
    Frente 1 — Validação Retroativa do Módulo SE v2.1 |
    {metricas['n_total']:,} transações analisadas |
    Gerado automaticamente por avaliar_se_retroativo.py
</div>

</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Relatório HTML: {OUT_HTML}")


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    main()
