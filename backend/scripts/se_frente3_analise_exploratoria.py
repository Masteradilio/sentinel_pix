"""
scripts/se_frente3_analise_exploratoria.py — Frente 3: Análise Exploratória

Objetivo: Entender onde o SE v3.1 falha para propor melhorias cirúrgicas.

Análises:
  1. Profiling das 155 fraudes não detectadas (se_score=0)
  2. Profiling dos 303 FP do COACAO_FISICA
  3. Candidatos a novos required / padrões
  4. Combinações de indicadores inexplorados

Dependências:
  - base_mvp_model_ready_optimized.csv
  - SocialEngineeringDetector v3.1

Saídas:
  - se_frente3_fraudes_invisiveis.csv
  - se_frente3_fp_coacao.csv
  - se_frente3_candidatos.json
  - se_frente3_relatorio.html
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent          # backend/scripts/
BACKEND_DIR = SCRIPT_DIR.parent                        # backend/
PROJECT_ROOT = BACKEND_DIR.parent                      # rebuild_pix/
DADOS_DIR = PROJECT_ROOT / "dados"
RELATORIO_DIR = BACKEND_DIR / "relatorio"
RELATORIO_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))

from core.social_engineering import SocialEngineeringDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuração ──
DATA_PATH = DADOS_DIR / "base_mvp_model_ready_optimized.csv"
TOP_N_COMBINACOES = 30
MIN_SUPPORT_FRAUDES = 5  # mínimo de fraudes para considerar combinação relevante


def load_data() -> pd.DataFrame:
    """Carrega dataset e faz validações básicas."""
    print("=" * 70)
    print("FRENTE 3 — Análise Exploratória para Melhoria do SE v3.1")
    print("=" * 70)

    print(f"\n[1/8] Carregando dados de {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    n_fraud = int(df["is_fraud"].sum())
    n_normal = len(df) - n_fraud
    print(f"  {len(df):,} transações ({n_fraud} fraudes, {n_normal:,} normais)")
    return df


def run_se_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Roda SE detector em todo o dataset, retorna DataFrame enriquecido."""
    print("\n[2/8] Rodando SocialEngineeringDetector v3.1...")
    detector = SocialEngineeringDetector()

    results: list[dict] = []
    n = len(df)
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        features = row.to_dict()
        result = detector.detect_from_pipeline(features)

        rec = {
            "idx": i,
            "transaction_id": features.get("transaction_id", features.get("cd_pix", "")),
            "is_fraud": int(row["is_fraud"]),
            "se_score": result.se_score,
            "n_patterns": len(result.patterns),
            "patterns": "|".join(p.pattern_name for p in result.patterns),
            "pattern_scores": "|".join(str(p.score) for p in result.patterns),
            "n_active_indicators": sum(1 for v in result.active_indicators.values() if v),
        }

        # Guardar cada indicador como coluna booleana
        for ind_name, ind_val in result.active_indicators.items():
            rec[f"ind_{ind_name}"] = int(ind_val)

        results.append(rec)

        if (i + 1) % (n // 10) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"   {i + 1:,}/{n:,} ({100 * (i + 1) / n:.0f}%) | {rate:.0f} tx/s")

    elapsed = time.time() - t0
    print(f"  ✓ Concluído em {elapsed:.1f}s ({n / elapsed:.0f} tx/s)")

    df_results = pd.DataFrame(results)
    return df_results


def merge_features(df_original: pd.DataFrame, df_se: pd.DataFrame) -> pd.DataFrame:
    """Merge features originais com resultados do SE."""
    print("\n[3/8] Fazendo merge de features originais com resultados SE...")

    feature_cols = [
        "vl_pix", "nr_idade", "pix_key_random_flag", "first_receiver_flag",
        "burst_30m_flag", "tx_count_prev_30m", "qt_intervalo_transacao_minuto",
        "qt_tempo_relacionamento_mes", "qt_pix_dia_maximo_trimestre",
        "qt_total_pix_trimestre", "qt_envio_recebedor_trimestre",
        "is_first_tx_trimestre", "hour", "day_of_week",
        "is_segmento_premium_flag", "pix_over_100pct_renda_flag",
        "pix_over_50pct_renda_flag", "renda_missing_flag",
        "is_sexo_feminino_flag", "is_viuvo_flag",
        "perfil_vulneravel_se_flag", "distinct_receivers_so_far",
        "vl_mediana_pix_trimestre", "ratio_valor_mediana",
        "is_login_senha_flag", "is_agendamento_recorrente_flag",
    ]

    # Só pegar colunas que existem E que não estão já no df_se
    se_cols = set(df_se.columns)
    existing_cols = [c for c in feature_cols if c in df_original.columns and c not in se_cols]
    missing_cols = [c for c in feature_cols if c not in df_original.columns]
    dupl_cols = [c for c in feature_cols if c in df_original.columns and c in se_cols]

    if missing_cols:
        print(f"  ⚠ Colunas ausentes (ignoradas): {missing_cols}")
    if dupl_cols:
        print(f"  ⚠ Colunas já no SE (ignoradas para evitar duplicata): {dupl_cols}")

    df_features = df_original[existing_cols].reset_index(drop=True)
    df_merged = pd.concat([df_se.reset_index(drop=True), df_features], axis=1)

    # Garantia extra: remover colunas duplicadas se houver
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    print(f"  ✓ Merge OK: {len(df_merged):,} rows × {len(df_merged.columns)} cols")
    return df_merged



def analyze_invisible_frauds(df: pd.DataFrame) -> pd.DataFrame:
    """Analisa as fraudes não detectadas pelo SE (score=0)."""
    print("\n[4/8] Analisando fraudes invisíveis ao SE (score=0)...")

    fraud_detected = df[(df["is_fraud"] == 1) & (df["se_score"] > 0)]
    fraud_invisible = df[(df["is_fraud"] == 1) & (df["se_score"] == 0)]
    normals = df[df["is_fraud"] == 0]

    n_det = len(fraud_detected)
    n_inv = len(fraud_invisible)
    print(f"  Fraudes detectadas:   {n_det}")
    print(f"  Fraudes invisíveis:   {n_inv}")

    # Profiling comparativo
    print(f"\n  {'Métrica':<40} {'Detectadas':>12} {'Invisíveis':>12} {'Normais':>12}")
    print("  " + "-" * 78)

    profile_cols = {
        "vl_pix": ("Valor PIX (mediana)", "median"),
        "nr_idade": ("Idade (mediana)", "median"),
        "burst_30m_flag": ("% com burst_30m", "mean_pct"),
        "tx_count_prev_30m": ("tx_count_prev_30m (média)", "mean"),
        "pix_key_random_flag": ("% chave aleatória", "mean_pct"),
        "first_receiver_flag": ("% primeiro recebedor", "mean_pct"),
        "is_first_tx_trimestre": ("% primeira tx trimestre", "mean_pct"),
        "qt_intervalo_transacao_minuto": ("Intervalo tx (mediana min)", "median"),
        "qt_tempo_relacionamento_mes": ("Tempo relação (mediana meses)", "median"),
        "is_segmento_premium_flag": ("% segmento premium", "mean_pct"),
        "pix_over_100pct_renda_flag": ("% PIX > 100% renda", "mean_pct"),
        "renda_missing_flag": ("% renda desconhecida", "mean_pct"),
        "distinct_receivers_so_far": ("Recebedores distintos (média)", "mean"),
    }

    profile_data: list[dict] = []
    for col, (label, agg) in profile_cols.items():
        if col not in df.columns:
            continue

        row: dict = {"metrica": label}
        for name, subset in [("detectadas", fraud_detected), ("invisiveis", fraud_invisible), ("normais", normals)]:
            if len(subset) == 0:
                row[name] = "N/A"
                continue
            col_data = subset[col]
            # Se retornou DataFrame (colunas duplicadas), pegar primeira
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            series = pd.to_numeric(col_data, errors="coerce").dropna()
            series = pd.to_numeric(subset[col], errors="coerce").dropna()
            if agg == "median":
                val = series.median()
                row[name] = f"{val:,.1f}"
            elif agg == "mean":
                val = series.mean()
                row[name] = f"{val:,.2f}"
            elif agg == "mean_pct":
                val = series.mean() * 100
                row[name] = f"{val:.1f}%"
            else:
                val = series.mean()
                row[name] = f"{val:,.2f}"

        profile_data.append(row)
        print(f"  {row['metrica']:<40} {row['detectadas']:>12} {row['invisiveis']:>12} {row['normais']:>12}")

    # Indicadores ativos nas fraudes invisíveis
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    print(f"\n  Indicadores mais comuns nas {n_inv} fraudes invisíveis:")
    print(f"  {'Indicador':<40} {'Invisíveis':>12} {'Detectadas':>12} {'Ratio':>8}")
    print("  " + "-" * 64)

    ind_stats: list[dict] = []
    for col in ind_cols:
        ind_name = col.replace("ind_", "")
        rate_inv = fraud_invisible[col].mean() if n_inv > 0 else 0
        rate_det = fraud_detected[col].mean() if n_det > 0 else 0
        ratio = rate_inv / rate_det if rate_det > 0 else float("inf")
        ind_stats.append({
            "indicator": ind_name,
            "rate_invisible": rate_inv,
            "rate_detected": rate_det,
            "ratio": ratio,
            "count_invisible": int(fraud_invisible[col].sum()),
            "count_detected": int(fraud_detected[col].sum()),
        })

    ind_stats.sort(key=lambda x: x["rate_invisible"], reverse=True)
    for s in ind_stats[:15]:
        ratio_str = f"{s['ratio']:.2f}" if s['ratio'] != float("inf") else "∞"
        print(
            f"  {s['indicator']:<40} "
            f"{s['rate_invisible'] * 100:>10.1f}% "
            f"{s['rate_detected'] * 100:>10.1f}% "
            f"{ratio_str:>8}"
        )

    # Salvar CSV
    out_path = RELATORIO_DIR / "se_frente3_fraudes_invisiveis.csv"
    fraud_invisible.to_csv(out_path, index=False)
    print(f"\n  Salvo: {out_path} ({len(fraud_invisible)} rows)")

    return fraud_invisible


def analyze_coacao_fp(df: pd.DataFrame) -> pd.DataFrame:
    """Analisa os FP do COACAO_FISICA — o maior gerador de FP."""
    print("\n[5/8] Analisando FP do COACAO_FISICA...")

    # Identificar FP e TP do COACAO_FISICA
    has_coacao = df["patterns"].str.contains("COACAO_FISICA", na=False)
    coacao_tp = df[has_coacao & (df["is_fraud"] == 1)]
    coacao_fp = df[has_coacao & (df["is_fraud"] == 0)]

    print(f"  COACAO_FISICA TP: {len(coacao_tp)}")
    print(f"  COACAO_FISICA FP: {len(coacao_fp)}")

    # Profiling TP vs FP
    print(f"\n  {'Métrica':<40} {'TP':>12} {'FP':>12}")
    print("  " + "-" * 66)

    compare_cols = {
        "vl_pix": ("Valor PIX (mediana)", "median"),
        "nr_idade": ("Idade (mediana)", "median"),
        "ind_burst_30m": ("% com burst_30m", "mean_pct"),
        "ind_burst_intenso": ("% com burst_intenso", "mean_pct"),
        "ind_multiplos_pix_rapidos": ("% com multiplos_pix_rapidos", "mean_pct"),
        "ind_valor_absoluto_alto": ("% com valor ≥ 5000", "mean_pct"),
        "ind_valor_absoluto_muito_alto": ("% com valor ≥ 10000", "mean_pct"),
        "ind_primeira_tx_trimestre": ("% primeira tx trimestre", "mean_pct"),
        "ind_chave_aleatoria": ("% chave aleatória", "mean_pct"),
        "ind_multiplos_recebedores_distintos": ("% múltiplos recebedores", "mean_pct"),
        "ind_renda_incompativel": ("% renda incompatível", "mean_pct"),
        "ind_renda_desconhecida_valor_alto": ("% renda desc + valor alto", "mean_pct"),
        "ind_recebedor_nunca_visto": ("% recebedor nunca visto", "mean_pct"),
    }

    for col, (label, agg) in compare_cols.items():
        if col not in df.columns:
            continue
        tp_series = pd.to_numeric(coacao_tp[col], errors="coerce").dropna()
        fp_series = pd.to_numeric(coacao_fp[col], errors="coerce").dropna()

        if agg == "median":
            tp_val, fp_val = tp_series.median(), fp_series.median()
            print(f"  {label:<40} {tp_val:>12,.1f} {fp_val:>12,.1f}")
        elif agg == "mean_pct":
            tp_val, fp_val = tp_series.mean() * 100, fp_series.mean() * 100
            print(f"  {label:<40} {tp_val:>11.1f}% {fp_val:>11.1f}%")

    # Indicadores que diferenciam TP de FP
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    print(f"\n  Indicadores que mais diferenciam TP vs FP no COACAO_FISICA:")
    print(f"  {'Indicador':<40} {'TP rate':>10} {'FP rate':>10} {'Δ':>10} {'Lift TP/FP':>10}")
    print("  " + "-" * 82)

    diff_stats: list[dict] = []
    for col in ind_cols:
        ind_name = col.replace("ind_", "")
        tp_rate = coacao_tp[col].mean() if len(coacao_tp) > 0 else 0
        fp_rate = coacao_fp[col].mean() if len(coacao_fp) > 0 else 0
        delta = tp_rate - fp_rate
        lift = tp_rate / fp_rate if fp_rate > 0 else float("inf")
        diff_stats.append({
            "indicator": ind_name,
            "tp_rate": tp_rate,
            "fp_rate": fp_rate,
            "delta": delta,
            "lift_tp_fp": lift,
        })

    diff_stats.sort(key=lambda x: x["delta"], reverse=True)
    for s in diff_stats[:15]:
        lift_str = f"{s['lift_tp_fp']:.2f}" if s['lift_tp_fp'] != float("inf") else "∞"
        print(
            f"  {s['indicator']:<40} "
            f"{s['tp_rate'] * 100:>9.1f}% "
            f"{s['fp_rate'] * 100:>9.1f}% "
            f"{s['delta'] * 100:>+9.1f}% "
            f"{lift_str:>10}"
        )

    # Salvar CSV
    out_path = RELATORIO_DIR / "se_frente3_fp_coacao.csv"
    coacao_fp.to_csv(out_path, index=False)
    print(f"\n  Salvo: {out_path} ({len(coacao_fp)} rows)")

    return coacao_fp


def analyze_indicator_combinations(df: pd.DataFrame) -> dict:
    """Testa combinações de 2-3 indicadores como candidatos a novos required."""
    print("\n[6/8] Testando combinações de indicadores como candidatos a required...")

    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    frauds = df[df["is_fraud"] == 1]
    normals = df[df["is_fraud"] == 0]

    n_fraud = len(frauds)
    n_normal = len(normals)

    # Indicadores com lift > 3x para usar como candidatos
    high_lift_indicators: list[str] = []
    for col in ind_cols:
        rate_fraud = frauds[col].mean()
        rate_normal = normals[col].mean()
        if rate_normal > 0:
            lift = rate_fraud / rate_normal
        else:
            lift = float("inf") if rate_fraud > 0 else 0
        if lift >= 3.0 and rate_fraud >= 0.02:  # lift >= 3x e ativa em >= 2% das fraudes
            high_lift_indicators.append(col)

    print(f"  Indicadores com Lift ≥ 3x e ≥ 2% fraudes: {len(high_lift_indicators)}")
    for col in high_lift_indicators:
        ind_name = col.replace("ind_", "")
        rate_f = frauds[col].mean() * 100
        rate_n = normals[col].mean() * 100
        lift = rate_f / rate_n if rate_n > 0 else float("inf")
        print(f"    {ind_name:<35} fraude={rate_f:5.1f}%  normal={rate_n:5.2f}%  lift={lift:.1f}x")

    # Testar pares
    print(f"\n  Testando pares de indicadores...")
    pair_results: list[dict] = []

    for i, col_a in enumerate(high_lift_indicators):
        for col_b in high_lift_indicators[i + 1:]:
            # Ambos ativos
            mask_fraud = (frauds[col_a] == 1) & (frauds[col_b] == 1)
            mask_normal = (normals[col_a] == 1) & (normals[col_b] == 1)

            tp = int(mask_fraud.sum())
            fp = int(mask_normal.sum())

            if tp < MIN_SUPPORT_FRAUDES:
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / n_fraud
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / n_normal

            pair_results.append({
                "type": "pair",
                "indicators": f"{col_a.replace('ind_', '')} + {col_b.replace('ind_', '')}",
                "tp": tp,
                "fp": fp,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 6),
                "total_ativacoes": tp + fp,
            })

    pair_results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n  Top {TOP_N_COMBINACOES} pares por F1:")
    print(f"  {'Combinação':<55} {'TP':>5} {'FP':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FPR':>10}")
    print("  " + "-" * 104)

    for r in pair_results[:TOP_N_COMBINACOES]:
        print(
            f"  {r['indicators']:<55} "
            f"{r['tp']:>5} "
            f"{r['fp']:>6} "
            f"{r['precision']:>8.1%} "
            f"{r['recall']:>8.1%} "
            f"{r['f1']:>8.4f} "
            f"{r['fpr']:>10.4%}"
        )

    # Testar trincas (top indicadores apenas para não explodir)
    print(f"\n  Testando trincas de indicadores (top 8 por lift)...")
    top8 = high_lift_indicators[:8]
    triple_results: list[dict] = []

    for i, col_a in enumerate(top8):
        for j, col_b in enumerate(top8[i + 1:], i + 1):
            for col_c in top8[j + 1:]:
                mask_fraud = (frauds[col_a] == 1) & (frauds[col_b] == 1) & (frauds[col_c] == 1)
                mask_normal = (normals[col_a] == 1) & (normals[col_b] == 1) & (normals[col_c] == 1)

                tp = int(mask_fraud.sum())
                fp = int(mask_normal.sum())

                if tp < MIN_SUPPORT_FRAUDES:
                    continue

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / n_fraud
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                fpr = fp / n_normal

                triple_results.append({
                    "type": "triple",
                    "indicators": (
                        f"{col_a.replace('ind_', '')} + "
                        f"{col_b.replace('ind_', '')} + "
                        f"{col_c.replace('ind_', '')}"
                    ),
                    "tp": tp,
                    "fp": fp,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "fpr": round(fpr, 6),
                    "total_ativacoes": tp + fp,
                })

    triple_results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n  Top {TOP_N_COMBINACOES} trincas por F1:")
    print(f"  {'Combinação':<75} {'TP':>5} {'FP':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 106)

    for r in triple_results[:TOP_N_COMBINACOES]:
        print(
            f"  {r['indicators']:<75} "
            f"{r['tp']:>5} "
            f"{r['fp']:>6} "
            f"{r['precision']:>8.1%} "
            f"{r['recall']:>8.1%} "
            f"{r['f1']:>8.4f}"
        )

    all_candidates = pair_results + triple_results
    return {"pairs": pair_results[:TOP_N_COMBINACOES], "triples": triple_results[:TOP_N_COMBINACOES]}


def analyze_coacao_refinement(df: pd.DataFrame) -> dict:
    """Testa variantes do COACAO_FISICA para reduzir FP."""
    print("\n[7/8] Testando variantes do COACAO_FISICA...")

    frauds = df[df["is_fraud"] == 1]
    normals = df[df["is_fraud"] == 0]
    n_fraud = len(frauds)
    n_normal = len(normals)

    # Required atual: intervalo_muito_curto + pix_acima_1000
    base_required_fraud = (frauds["ind_intervalo_muito_curto"] == 1) & (frauds["ind_pix_acima_1000"] == 1)
    base_required_normal = (normals["ind_intervalo_muito_curto"] == 1) & (normals["ind_pix_acima_1000"] == 1)

    variants: list[dict] = []

    # Variante 1: Adicionar burst_30m como required
    for extra_name, extra_col in [
        ("+ burst_30m", "ind_burst_30m"),
        ("+ burst_intenso", "ind_burst_intenso"),
        ("+ valor_absoluto_alto (≥5k)", "ind_valor_absoluto_alto"),
        ("+ valor_absoluto_muito_alto (≥10k)", "ind_valor_absoluto_muito_alto"),
        ("+ primeira_tx_trimestre", "ind_primeira_tx_trimestre"),
        ("+ multiplos_pix_rapidos", "ind_multiplos_pix_rapidos"),
        ("+ chave_aleatoria", "ind_chave_aleatoria"),
    ]:
        if extra_col not in df.columns:
            continue

        mask_fraud = base_required_fraud & (frauds[extra_col] == 1)
        mask_normal = base_required_normal & (normals[extra_col] == 1)

        tp = int(mask_fraud.sum())
        fp = int(mask_normal.sum())

        if tp == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_fraud
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        variants.append({
            "variant": f"COACAO required {extra_name}",
            "required": f"intervalo_muito_curto + pix_acima_1000 {extra_name}",
            "tp": tp,
            "fp": fp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fp_reduction_vs_current": 303 - fp,
            "tp_loss_vs_current": 122 - tp,
        })

    # Variante: Trocar pix_acima_1000 por valor_absoluto_alto
    mask_f_v5k = (frauds["ind_intervalo_muito_curto"] == 1) & (frauds["ind_valor_absoluto_alto"] == 1)
    mask_n_v5k = (normals["ind_intervalo_muito_curto"] == 1) & (normals["ind_valor_absoluto_alto"] == 1)
    tp_v5k = int(mask_f_v5k.sum())
    fp_v5k = int(mask_n_v5k.sum())
    if tp_v5k > 0:
        prec = tp_v5k / (tp_v5k + fp_v5k) if (tp_v5k + fp_v5k) > 0 else 0
        rec = tp_v5k / n_fraud
        f1_ = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        variants.append({
            "variant": "COACAO required swap: pix≥5k em vez de pix≥1k",
            "required": "intervalo_muito_curto + valor_absoluto_alto (≥5k)",
            "tp": tp_v5k,
            "fp": fp_v5k,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_, 4),
            "fp_reduction_vs_current": 303 - fp_v5k,
            "tp_loss_vs_current": 122 - tp_v5k,
        })

    print(f"\n  {'Variante':<55} {'TP':>5} {'FP':>6} {'Prec':>8} {'F1':>8} {'ΔFP':>6} {'ΔTP':>6}")
    print("  " + "-" * 98)
    print(f"  {'ATUAL (intervalo_curto + pix≥1k, ms=5)':<55} {'122':>5} {'303':>6} {'28.7%':>8} {'0.313':>8} {'---':>6} {'---':>6}")
    print("  " + "-" * 98)

    for v in sorted(variants, key=lambda x: x["f1"], reverse=True):
        print(
            f"  {v['variant']:<55} "
            f"{v['tp']:>5} "
            f"{v['fp']:>6} "
            f"{v['precision']:>8.1%} "
            f"{v['f1']:>8.4f} "
            f"{v['fp_reduction_vs_current']:>+6} "
            f"{v['tp_loss_vs_current']:>+6}"
        )

    return {"variants": variants}


def generate_report(
    df: pd.DataFrame,
    fraud_invisible: pd.DataFrame,
    candidates: dict,
    coacao_variants: dict,
) -> None:
    """Gera relatório HTML consolidado."""
    print("\n[8/8] Gerando relatório e artefatos...")

    # Salvar candidatos JSON
    out_json = RELATORIO_DIR / "se_frente3_candidatos.json"
    report_data = {
        "data_geracao": datetime.now().isoformat(),
        "n_fraudes_invisiveis": int(((df["is_fraud"] == 1) & (df["se_score"] == 0)).sum())
    if "se_score" in df.columns else len(fraud_invisible),

        "n_fp_coacao": int(
            ((df["patterns"].str.contains("COACAO_FISICA", na=False)) & (df["is_fraud"] == 0)).sum()
        ),
        "top_pair_candidates": candidates.get("pairs", [])[:10],
        "top_triple_candidates": candidates.get("triples", [])[:10],
        "coacao_variants": coacao_variants.get("variants", []),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Salvo: {out_json}")

    # HTML report
    out_html = RELATORIO_DIR / "se_frente3_relatorio.html"

    html_parts: list[str] = []
    html_parts.append("<!DOCTYPE html><html><head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("<title>Frente 3 — Análise Exploratória SE v3.1</title>")
    html_parts.append("<style>")
    html_parts.append("""
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2em; background: #f5f5f5; }
        h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 0.3em; }
        h2 { color: #16213e; margin-top: 2em; }
        h3 { color: #0f3460; }
        table { border-collapse: collapse; margin: 1em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
        th { background: #16213e; color: white; }
        td:first-child, th:first-child { text-align: left; }
        tr:nth-child(even) { background: #f8f8f8; }
        .highlight { background: #fff3cd; font-weight: bold; }
        .good { color: #28a745; }
        .bad { color: #dc3545; }
        .metric-box { display: inline-block; background: white; padding: 1em 2em; margin: 0.5em;
                       border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        .metric-box .number { font-size: 2em; font-weight: bold; color: #e94560; }
        .metric-box .label { font-size: 0.9em; color: #666; }
        pre { background: #1a1a2e; color: #e0e0e0; padding: 1em; border-radius: 4px; overflow-x: auto; }
    """)
    html_parts.append("</style></head><body>")

    html_parts.append("<h1>🔬 Frente 3 — Análise Exploratória SE v3.1</h1>")
    html_parts.append(f"<p>Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

    # KPIs
    n_inv = len(fraud_invisible)
    n_fp_coacao = report_data["n_fp_coacao"]
    html_parts.append("<div>")
    html_parts.append(f"<div class='metric-box'><div class='number'>{n_inv}</div><div class='label'>Fraudes invisíveis ao SE</div></div>")
    html_parts.append(f"<div class='metric-box'><div class='number'>{n_fp_coacao}</div><div class='label'>FP do COACAO_FISICA</div></div>")
    html_parts.append("</div>")

    # Tabela de variantes COACAO
    html_parts.append("<h2>Variantes do COACAO_FISICA</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Variante</th><th>TP</th><th>FP</th><th>Precision</th><th>F1</th><th>ΔFP</th><th>ΔTP</th></tr>")
    html_parts.append("<tr class='highlight'><td>ATUAL (ms=5)</td><td>122</td><td>303</td><td>28.7%</td><td>0.313</td><td>—</td><td>—</td></tr>")
    for v in sorted(coacao_variants.get("variants", []), key=lambda x: x["f1"], reverse=True):
        fp_class = "good" if v["fp_reduction_vs_current"] > 0 else "bad"
        tp_class = "bad" if v["tp_loss_vs_current"] < -10 else ""
        html_parts.append(
            f"<tr><td>{v['variant']}</td>"
            f"<td>{v['tp']}</td>"
            f"<td>{v['fp']}</td>"
            f"<td>{v['precision']:.1%}</td>"
            f"<td>{v['f1']:.4f}</td>"
            f"<td class='{fp_class}'>{v['fp_reduction_vs_current']:+d}</td>"
            f"<td class='{tp_class}'>{v['tp_loss_vs_current']:+d}</td></tr>"
        )
    html_parts.append("</table>")

    # Tabela de pares
    html_parts.append("<h2>Top Pares de Indicadores (candidatos a required)</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Combinação</th><th>TP</th><th>FP</th><th>Precision</th><th>Recall</th><th>F1</th><th>FPR</th></tr>")
    for r in candidates.get("pairs", [])[:20]:
        html_parts.append(
            f"<tr><td>{r['indicators']}</td>"
            f"<td>{r['tp']}</td>"
            f"<td>{r['fp']}</td>"
            f"<td>{r['precision']:.1%}</td>"
            f"<td>{r['recall']:.1%}</td>"
            f"<td>{r['f1']:.4f}</td>"
            f"<td>{r['fpr']:.4%}</td></tr>"
        )
    html_parts.append("</table>")

    # Tabela de trincas
    html_parts.append("<h2>Top Trincas de Indicadores</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Combinação</th><th>TP</th><th>FP</th><th>Precision</th><th>Recall</th><th>F1</th></tr>")
    for r in candidates.get("triples", [])[:20]:
        html_parts.append(
            f"<tr><td>{r['indicators']}</td>"
            f"<td>{r['tp']}</td>"
            f"<td>{r['fp']}</td>"
            f"<td>{r['precision']:.1%}</td>"
            f"<td>{r['recall']:.1%}</td>"
            f"<td>{r['f1']:.4f}</td></tr>"
        )
    html_parts.append("</table>")

    html_parts.append("</body></html>")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"  Salvo: {out_html}")


def print_summary(candidates: dict, coacao_variants: dict) -> None:
    """Imprime resumo final."""
    print("\n" + "=" * 70)
    print("RESUMO — Frente 3: Análise Exploratória")
    print("=" * 70)

    # Melhor variante COACAO
    variants = coacao_variants.get("variants", [])
    if variants:
        best = max(variants, key=lambda x: x["f1"])
        print(f"\n  Melhor variante COACAO_FISICA:")
        print(f"    {best['variant']}")
        print(f"    TP={best['tp']}, FP={best['fp']}, Precision={best['precision']:.1%}, F1={best['f1']:.4f}")
        print(f"    vs atual: ΔFP={best['fp_reduction_vs_current']:+d}, ΔTP={best['tp_loss_vs_current']:+d}")

    # Melhores candidatos a novos padrões
    pairs = candidates.get("pairs", [])
    if pairs:
        print(f"\n  Top 5 pares candidatos a required:")
        for r in pairs[:5]:
            print(f"    {r['indicators']:<55} Prec={r['precision']:.1%} F1={r['f1']:.4f} TP={r['tp']} FP={r['fp']}")

    triples = candidates.get("triples", [])
    if triples:
        print(f"\n  Top 5 trincas candidatas:")
        for r in triples[:5]:
            print(f"    {r['indicators']:<70} Prec={r['precision']:.1%} F1={r['f1']:.4f}")

    print(f"\n  Artefatos:")
    for f in RELATORIO_DIR.glob("se_frente3_*"):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.0f} KB)")

    print("\n" + "=" * 70)


def main() -> None:
    """Pipeline principal."""
    # 1. Carregar dados
    df = load_data()

    # 2. Rodar SE
    df_se = run_se_detector(df)

    # 3. Merge features
    df_merged = merge_features(df, df_se)

    # 4. Analisar fraudes invisíveis
    fraud_invisible = analyze_invisible_frauds(df_merged)

    # 5. Analisar FP do COACAO
    analyze_coacao_fp(df_merged)

    # 6. Testar combinações de indicadores
    candidates = analyze_indicator_combinations(df_merged)

    # 7. Testar variantes COACAO
    coacao_variants = analyze_coacao_refinement(df_merged)

    # 8. Gerar relatório
    generate_report(df_merged, fraud_invisible, candidates, coacao_variants)

    # 9. Resumo
    print_summary(candidates, coacao_variants)


if __name__ == "__main__":
    main()
