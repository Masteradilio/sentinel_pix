"""
experimentos/exp_001_threshold_final/run_exp_001.py

EXP-001: Ajuste do Threshold Final do Score (77 -> 62)

ESTRATÉGIA DE EXECUÇÃO (eficiente):
  1. Processa o sample UMA única vez via PipelineOrquestrador (salva score_final).
  2. Aplica múltiplos thresholds POST-HOC sobre score_final (4 variantes).
  3. Roda sweep fino 40..90 step=1 (granularidade maior que FASE 0).
  4. Valida em sample independente (seed=123).
  5. Gera relatório executivo em markdown.

Artefatos gerados (máximo 5):
  01_tabela_comparativa.csv      — métricas das 4 variantes lado a lado
  02_threshold_sweep_fino.csv    — sweep completo 40..90
  03_analise_fp_fn.json          — deep-dive qualitativo em FP/FN
  04_validacao_cruzada.json      — resultado em seed=123
  05_conclusao_executiva.md      — relatório humano-readable

Uso:
    python experimentos/exp_001_threshold_final/run_exp_001.py
    python experimentos/exp_001_threshold_final/run_exp_001.py --sample 6000 --workers 4
    python experimentos/exp_001_threshold_final/run_exp_001.py --skip-validation

Autor: AI Engineer + Adilio
Data: 2026-04-17
Fase: 1 (Otimização Cirúrgica)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Garantir que o pacote experimentos/ está no path
EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimentos.utils_experimentos import (
    apply_threshold_to_predictions,
    compute_metrics,
    get_experiment_output_dir,
    get_logger,
    load_dataset,
    print_section,
    process_dataframe_via_orquestrador,
    safe_json_dump,
    stratified_sample,
)

EXP_ID = "EXP-001"
EXP_TITLE = "Ajuste do Threshold Final (77 -> 62)"
CONFIG_PATH = EXP_DIR / "config_variantes.json"

logger = get_logger("EXP-001")


# =========================================================
# LÓGICA CENTRAL: avaliar todas as variantes a partir de UM processamento
# =========================================================
def evaluate_all_variants(
    predictions_df: pd.DataFrame,
    baseline_cfg: dict,
    variantes_cfg: list[dict],
) -> pd.DataFrame:
    """Aplica baseline + N variantes de threshold sobre o MESMO predictions_df.

    Returns:
        DataFrame com uma linha por variante + colunas de métricas.
    """
    y_true = predictions_df["is_fraud"].values.astype(int)
    rows = []

    # Baseline usa a coluna `decisao` original (que já foi gerada com threshold=77)
    y_pred_baseline = predictions_df["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values
    m_base = compute_metrics(y_true, y_pred_baseline, baseline_cfg["label"])

    rows.append({
        "variante_id": "BASELINE",
        "threshold_confirmar": baseline_cfg["threshold_confirmar"],
        "threshold_bloquear": baseline_cfg["threshold_bloquear"],
        "label": baseline_cfg["label"],
        **m_base.to_dict(),
    })

    # Variantes — aplicadas post-hoc via re-threshold sobre score_final
    for var in variantes_cfg:
        decisao_var = apply_threshold_to_predictions(
            predictions_df,
            threshold_confirmar=var["threshold_confirmar"],
            threshold_bloquear=var["threshold_bloquear"],
        )
        y_pred_var = decisao_var.isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values
        m_var = compute_metrics(y_true, y_pred_var, var["label"])

        rows.append({
            "variante_id": var["id"],
            "threshold_confirmar": var["threshold_confirmar"],
            "threshold_bloquear": var["threshold_bloquear"],
            "label": var["label"],
            **m_var.to_dict(),
        })

    df_result = pd.DataFrame(rows)

    # Delta em relação ao baseline
    baseline_f1 = df_result.loc[df_result["variante_id"] == "BASELINE", "F1"].iloc[0]
    baseline_rec = df_result.loc[df_result["variante_id"] == "BASELINE", "Recall"].iloc[0]
    baseline_prec = df_result.loc[df_result["variante_id"] == "BASELINE", "Precision"].iloc[0]

    df_result["delta_F1"] = (df_result["F1"] - baseline_f1).round(6)
    df_result["delta_Recall"] = (df_result["Recall"] - baseline_rec).round(6)
    df_result["delta_Precision"] = (df_result["Precision"] - baseline_prec).round(6)

    return df_result


# =========================================================
# THRESHOLD SWEEP FINO (granularidade 1pt)
# =========================================================
def run_fine_threshold_sweep(
    predictions_df: pd.DataFrame,
    t_min: int,
    t_max: int,
    step: int,
) -> pd.DataFrame:
    """Sweep de threshold com granularidade fina sobre score_final."""
    y_true = predictions_df["is_fraud"].values.astype(int)
    rows = []

    for t in range(t_min, t_max + 1, step):
        decisao = apply_threshold_to_predictions(
            predictions_df,
            threshold_confirmar=t,
            threshold_bloquear=90,
        )
        y_pred = decisao.isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values
        m = compute_metrics(y_true, y_pred, f"t={t}")

        rows.append({
            "threshold": t,
            "TP": m.tp,
            "FP": m.fp,
            "FN": m.fn,
            "TN": m.tn,
            "Precision": round(m.precision, 6),
            "Recall": round(m.recall, 6),
            "F1": round(m.f1, 6),
            "FPR": round(m.fpr, 8),
        })

    return pd.DataFrame(rows)


# =========================================================
# ANÁLISE QUALITATIVA DE FP/FN
# =========================================================
def analyze_fp_fn_differences(
    predictions_df: pd.DataFrame,
    baseline_th: int,
    winner_th: int,
) -> dict:
    """Compara FP/FN entre baseline e variante vencedora.

    Identifica:
      - Quantas fraudes foram recuperadas (FN -> TP)
      - Quantos FP novos surgiram (TN -> FP)
      - Perfil dos novos FP (idade, valor, PJ?)
      - Perfil dos FN recuperados
    """
    y_true = predictions_df["is_fraud"].values.astype(int)

    # Decisões em cada configuração
    dec_baseline = apply_threshold_to_predictions(predictions_df, baseline_th, 90)
    dec_winner = apply_threshold_to_predictions(predictions_df, winner_th, 90)

    flagged_baseline = dec_baseline.isin(["CONFIRMAR", "BLOQUEAR"]).values
    flagged_winner = dec_winner.isin(["CONFIRMAR", "BLOQUEAR"]).values

    # FN recuperados: era fraude E não era flagged E agora é flagged
    mask_fn_recovered = (y_true == 1) & (~flagged_baseline) & (flagged_winner)
    # FP novos: não era fraude E não era flagged E agora é flagged
    mask_fp_new = (y_true == 0) & (~flagged_baseline) & (flagged_winner)

    df_fn_rec = predictions_df[mask_fn_recovered].copy()
    df_fp_new = predictions_df[mask_fp_new].copy()

    cols_interesse = [
        "transaction_id", "customer_id", "vl_pix", "nr_idade",
        "qt_tempo_relacionamento_mes", "is_first_tx_trimestre",
        "first_receiver_flag", "burst_30m_flag", "pix_key_random_flag",
        "perfil_vulneravel_se_flag", "lgbm_raw", "if_percentile",
        "se_score", "beh_score", "score_final", "decisao", "veto_aplicado",
    ]
    cols_presentes = [c for c in cols_interesse if c in predictions_df.columns]

    # Agregações para perfil
    def _profile(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n": 0}
        return {
            "n": len(df),
            "valor_pix": {
                "min": float(df["vl_pix"].min()) if "vl_pix" in df else None,
                "max": float(df["vl_pix"].max()) if "vl_pix" in df else None,
                "median": float(df["vl_pix"].median()) if "vl_pix" in df else None,
                "mean": float(df["vl_pix"].mean()) if "vl_pix" in df else None,
            },
            "idade": {
                "min": int(df["nr_idade"].min()) if "nr_idade" in df else None,
                "max": int(df["nr_idade"].max()) if "nr_idade" in df else None,
                "median": float(df["nr_idade"].median()) if "nr_idade" in df else None,
                "idosos_60plus": int((df["nr_idade"] >= 60).sum()) if "nr_idade" in df else 0,
                "jovens_abaixo_25": int((df["nr_idade"] < 25).sum()) if "nr_idade" in df else 0,
            },
            "score_final": {
                "min": float(df["score_final"].min()) if "score_final" in df else None,
                "max": float(df["score_final"].max()) if "score_final" in df else None,
                "median": float(df["score_final"].median()) if "score_final" in df else None,
            },
            "flags": {
                "first_receiver": int(df.get("first_receiver_flag", pd.Series(dtype=int)).sum()),
                "burst_30m": int(df.get("burst_30m_flag", pd.Series(dtype=int)).sum()),
                "pix_key_random": int(df.get("pix_key_random_flag", pd.Series(dtype=int)).sum()),
                "perfil_vulneravel": int(df.get("perfil_vulneravel_se_flag", pd.Series(dtype=int)).sum()),
            },
        }

    # Top 10 casos (amostra para inspeção humana)
    def _top_cases(df: pd.DataFrame, n: int = 10, sort_col: str = "vl_pix") -> list:
        if df.empty:
            return []
        df_sorted = df[cols_presentes].sort_values(sort_col, ascending=False).head(n)
        return df_sorted.to_dict(orient="records")

    return {
        "baseline_threshold": baseline_th,
        "winner_threshold": winner_th,
        "fn_recuperados": {
            "total": int(mask_fn_recovered.sum()),
            "perfil": _profile(df_fn_rec),
            "casos_top10_por_valor": _top_cases(df_fn_rec, 10, "vl_pix"),
        },
        "fp_novos": {
            "total": int(mask_fp_new.sum()),
            "perfil": _profile(df_fp_new),
            "casos_top10_por_score": _top_cases(df_fp_new, 10, "score_final"),
            "alertas": _check_fp_alerts(df_fp_new),
        },
    }


def _check_fp_alerts(df_fp_new: pd.DataFrame) -> list[str]:
    """Checa alertas críticos no perfil dos novos FP."""
    alerts = []
    if df_fp_new.empty:
        return alerts

    n = len(df_fp_new)

    # Alerta 1: concentração desproporcional em idosos
    if "nr_idade" in df_fp_new.columns:
        idosos = (df_fp_new["nr_idade"] >= 65).sum()
        if idosos / max(n, 1) > 0.30:
            alerts.append(
                f"ATENCAO: {idosos}/{n} ({idosos/n:.0%}) novos FP sao idosos (65+). "
                f"Avaliar impacto em perfil vulneravel."
            )

    # Alerta 2: FP de alto valor
    if "vl_pix" in df_fp_new.columns:
        alto_valor = (df_fp_new["vl_pix"] >= 5000).sum()
        if alto_valor > 0:
            alerts.append(
                f"INFO: {alto_valor} novos FP com valor >= R$5.000. "
                f"Revisar manualmente antes de deploy."
            )

    # Alerta 3: FP concentrados em perfil vulnerável
    if "perfil_vulneravel_se_flag" in df_fp_new.columns:
        vuln = df_fp_new["perfil_vulneravel_se_flag"].sum()
        if vuln / max(n, 1) > 0.20:
            alerts.append(
                f"ATENCAO: {vuln}/{n} ({vuln/n:.0%}) novos FP em perfil vulneravel. "
                f"Pode indicar bias do threshold mais agressivo."
            )

    return alerts


# =========================================================
# VALIDAÇÃO CRUZADA
# =========================================================
def run_cross_validation(
    df_full: pd.DataFrame,
    sample_n: int,
    seed_validacao: int,
    winner_cfg: dict,
    baseline_cfg: dict,
    workers: int,
) -> dict:
    """Processa sample independente e verifica se direção do resultado se mantém."""
    logger.info(f"Validação cruzada: gerando sample com seed={seed_validacao}")
    df_val = stratified_sample(df_full, n=sample_n, seed=seed_validacao, logger=logger)

    logger.info(f"Processando {len(df_val):,} tx (validação)...")
    preds_val = process_dataframe_via_orquestrador(df_val, workers=workers, logger=logger)

    y_true = preds_val["is_fraud"].values.astype(int)

    # Baseline
    dec_base = preds_val["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values
    m_base = compute_metrics(y_true, dec_base, f"Baseline @ seed={seed_validacao}")

    # Vencedor
    dec_winner = apply_threshold_to_predictions(
        preds_val,
        threshold_confirmar=winner_cfg["threshold_confirmar"],
        threshold_bloquear=winner_cfg["threshold_bloquear"],
    )
    y_pred_winner = dec_winner.isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values
    m_winner = compute_metrics(y_true, y_pred_winner, f"Winner @ seed={seed_validacao}")

    direcao_confirmada = m_winner.f1 > m_base.f1

    return {
        "seed": seed_validacao,
        "sample_size": len(df_val),
        "n_fraudes": int(y_true.sum()),
        "baseline": m_base.to_dict(),
        "winner": m_winner.to_dict(),
        "delta_F1": round(m_winner.f1 - m_base.f1, 6),
        "delta_Recall": round(m_winner.recall - m_base.recall, 6),
        "direcao_confirmada": bool(direcao_confirmada),
        "interpretacao": (
            "VALIDADO: F1 do vencedor > F1 do baseline no sample independente."
            if direcao_confirmada else
            "ATENCAO: direcao NAO confirmada. Revisar antes de deploy."
        ),
    }


# =========================================================
# CRITÉRIOS DE ACEITAÇÃO
# =========================================================
def evaluate_acceptance_criteria(
    winner_metrics: dict,
    baseline_metrics: dict,
    criterios: dict,
) -> dict:
    """Avalia se a variante vencedora atende aos critérios de aceitação do EXP-001."""
    delta_f1 = winner_metrics["F1"] - baseline_metrics["F1"]

    checks = {
        "f1_margem_minima": {
            "criterio": f"delta_F1 >= {criterios['f1_min_margem']}",
            "valor_obtido": round(delta_f1, 6),
            "threshold": criterios["f1_min_margem"],
            "passou": bool(delta_f1 >= criterios["f1_min_margem"]),
        },
        "recall_minimo": {
            "criterio": f"Recall >= {criterios['recall_min']:.0%}",
            "valor_obtido": winner_metrics["Recall"],
            "threshold": criterios["recall_min"],
            "passou": bool(winner_metrics["Recall"] >= criterios["recall_min"]),
        },
        "fpr_maximo": {
            "criterio": f"FPR <= {criterios['fpr_max']:.2%}",
            "valor_obtido": winner_metrics["FPR"],
            "threshold": criterios["fpr_max"],
            "passou": bool(winner_metrics["FPR"] <= criterios["fpr_max"]),
        },
        "precision_minima": {
            "criterio": f"Precision >= {criterios['precision_min']:.0%}",
            "valor_obtido": winner_metrics["Precision"],
            "threshold": criterios["precision_min"],
            "passou": bool(winner_metrics["Precision"] >= criterios["precision_min"]),
        },
    }

    todos_passaram = all(c["passou"] for c in checks.values())

    return {
        "aprovado": todos_passaram,
        "checks": checks,
    }


# =========================================================
# RELATÓRIO EXECUTIVO EM MARKDOWN
# =========================================================
def generate_executive_report(
    df_comparativa: pd.DataFrame,
    df_sweep: pd.DataFrame,
    fp_fn_analysis: dict,
    cross_val: dict | None,
    acceptance: dict,
    winner_cfg: dict,
    config: dict,
    elapsed_total: float,
) -> str:
    """Gera relatório markdown consolidando todos os achados."""

    winner_row = df_comparativa[df_comparativa["variante_id"] == winner_cfg["id"]].iloc[0]
    baseline_row = df_comparativa[df_comparativa["variante_id"] == "BASELINE"].iloc[0]

    # Encontrar melhor threshold no sweep fino
    best_sweep_idx = df_sweep["F1"].idxmax()
    best_sweep = df_sweep.loc[best_sweep_idx]

    status_emoji = "✅ APROVADO" if acceptance["aprovado"] else "❌ REPROVADO"
    val_status = ""
    if cross_val:
        val_status = "✅ CONFIRMADA" if cross_val["direcao_confirmada"] else "⚠️ NÃO CONFIRMADA"

    report = f"""# EXP-001 — Conclusão Executiva

> **Experimento:** {EXP_TITLE}
> **Status:** {status_emoji}
> **Validação cruzada:** {val_status or "N/A (skipped)"}
> **Tempo total:** {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)

---

## 1. Resumo do Resultado

| Métrica | Baseline (t=77) | Vencedor ({winner_cfg['label']}) | Delta |
|---|---:|---:|---:|
| **TP** | {baseline_row['TP']:.0f} | **{winner_row['TP']:.0f}** | **+{winner_row['TP']-baseline_row['TP']:.0f}** |
| **FP** | {baseline_row['FP']:.0f} | **{winner_row['FP']:.0f}** | **+{winner_row['FP']-baseline_row['FP']:.0f}** |
| **FN** | {baseline_row['FN']:.0f} | **{winner_row['FN']:.0f}** | **{winner_row['FN']-baseline_row['FN']:+.0f}** |
| **Precision** | {baseline_row['Precision']:.2%} | **{winner_row['Precision']:.2%}** | {winner_row['delta_Precision']:+.4f} |
| **Recall** | {baseline_row['Recall']:.2%} | **{winner_row['Recall']:.2%}** | {winner_row['delta_Recall']:+.4f} |
| **F1** | {baseline_row['F1']:.4f} | **{winner_row['F1']:.4f}** | {winner_row['delta_F1']:+.4f} |
| **FPR** | {baseline_row['FPR']:.4%} | **{winner_row['FPR']:.4%}** | — |

---

## 2. Comparativo de Variantes

{df_comparativa[['variante_id', 'threshold_confirmar', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']].to_markdown(index=False, floatfmt='.4f')}

---

## 3. Threshold Sweep Fino

**Melhor F1 no sweep:** F1={best_sweep['F1']:.4f} @ threshold={int(best_sweep['threshold'])} (TP={int(best_sweep['TP'])}, FP={int(best_sweep['FP'])}, FN={int(best_sweep['FN'])})

{df_sweep[df_sweep['threshold'].between(55, 80)][['threshold', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']].to_markdown(index=False, floatfmt='.4f')}

---

## 4. Análise dos FN Recuperados

**Total recuperado:** {fp_fn_analysis['fn_recuperados']['total']} fraudes

"""

    if fp_fn_analysis["fn_recuperados"]["total"] > 0:
        perfil_fn = fp_fn_analysis["fn_recuperados"]["perfil"]
        report += f"""**Perfil das fraudes recuperadas:**
- Valor mediano: R$ {perfil_fn['valor_pix']['median']:,.2f}
- Valor máximo: R$ {perfil_fn['valor_pix']['max']:,.2f}
- Idade mediana: {perfil_fn['idade']['median']:.0f} anos
- Idosos (60+): {perfil_fn['idade']['idosos_60plus']}
- Jovens (<25): {perfil_fn['idade']['jovens_abaixo_25']}
- Score final mediano: {perfil_fn['score_final']['median']:.1f}
- Com first_receiver_flag: {perfil_fn['flags']['first_receiver']}
- Com perfil vulneravel: {perfil_fn['flags']['perfil_vulneravel']}

"""
    else:
        report += "_(nenhuma fraude recuperada nesta variante)_\n\n"

    report += "---\n\n## 5. Análise dos FP Novos\n\n"
    report += f"**Total novos FP:** {fp_fn_analysis['fp_novos']['total']}\n\n"

    if fp_fn_analysis["fp_novos"]["total"] > 0:
        perfil_fp = fp_fn_analysis["fp_novos"]["perfil"]
        report += f"""**Perfil dos novos FP:**
- Valor mediano: R$ {perfil_fp['valor_pix']['median']:,.2f}
- Valor máximo: R$ {perfil_fp['valor_pix']['max']:,.2f}
- Idade mediana: {perfil_fp['idade']['median']:.0f} anos
- Idosos (60+): {perfil_fp['idade']['idosos_60plus']}
- Score final mediano: {perfil_fp['score_final']['median']:.1f}

"""
        if fp_fn_analysis["fp_novos"]["alertas"]:
            report += "**Alertas de segurança:**\n"
            for a in fp_fn_analysis["fp_novos"]["alertas"]:
                report += f"- ⚠️ {a}\n"
            report += "\n"
        else:
            report += "_✅ Sem alertas de segurança no perfil dos novos FP._\n\n"

    # Validação cruzada
    if cross_val:
        report += "---\n\n## 6. Validação Cruzada\n\n"
        report += f"""- **Sample:** {cross_val['sample_size']:,} tx (seed={cross_val['seed']})
- **Fraudes no sample:** {cross_val['n_fraudes']}
- **F1 baseline:** {cross_val['baseline']['F1']:.4f}
- **F1 vencedor:** {cross_val['winner']['F1']:.4f}
- **Delta F1:** {cross_val['delta_F1']:+.4f}
- **Delta Recall:** {cross_val['delta_Recall']:+.4f}
- **Interpretação:** {cross_val['interpretacao']}

"""

    # Critérios
    report += "---\n\n## 7. Critérios de Aceitação\n\n"
    report += "| Critério | Valor obtido | Threshold | Status |\n|---|---:|---:|:---:|\n"
    for name, check in acceptance["checks"].items():
        status = "✅" if check["passou"] else "❌"
        report += f"| {check['criterio']} | {check['valor_obtido']:.4f} | {check['threshold']:.4f} | {status} |\n"

        # Recomendação
    report += "\n---\n\n## 8. Recomendação Final\n\n"

    winner_id = winner_cfg["id"]
    winner_th = winner_cfg["threshold_confirmar"]

    if acceptance["aprovado"] and (cross_val is None or cross_val["direcao_confirmada"]):
        # Cenário 1: APROVAR
        recomendacao_linhas = [
            f"### ✅ APROVAR — Deploy da variante **{winner_id}** (threshold={winner_th})",
            "",
            "**Próximos passos:**",
            "",
            "1. Atualizar `backend/artefatos/scoring_config.json`:",
            "   ```json",
            "   {",
            f'     "score_final_threshold_confirmar": {winner_th}',
            "   }",
            "   ```",
            "2. Incrementar `engine_version`: 3.0.5 → 3.0.6",
            "3. Criar PR com link para este relatório",
            "4. Monitorar métricas por 48h pós-deploy",
            "5. Seguir para o próximo experimento (EXP-004 — Rate Limiting)",
            "",
        ]
        report += "\n".join(recomendacao_linhas)

    elif acceptance["aprovado"] and cross_val and not cross_val["direcao_confirmada"]:
        # Cenário 2: APROVADO COM RESSALVA
        recomendacao_linhas = [
            "### ⚠️ APROVADO COM RESSALVA — validação cruzada não confirmou direção",
            "",
            "Critérios primários atendidos, mas sample independente mostrou comportamento divergente.",
            "",
            "**Recomendação:** rodar EXP-001 em sample maior (12k-20k) antes de deploy.",
            "",
            "**Investigar:**",
            "- Se há viés no sample principal (seed=42)",
            "- Se os FN/FP recuperados são de perfis específicos que não aparecem em seed=123",
            "- Aumentar n para reduzir variância estatística",
            "",
        ]
        report += "\n".join(recomendacao_linhas)

    else:
        # Cenário 3: REPROVAR
        failed = [name for name, c in acceptance["checks"].items() if not c["passou"]]
        recomendacao_linhas = [
            "### ❌ REPROVAR — critérios não atendidos",
            "",
            f"**Critérios reprovados:** {', '.join(failed)}",
            "",
            "**Recomendação:**",
            "",
            "1. Investigar causa raiz antes de mexer em threshold",
            "2. Considerar avançar direto para EXP-004 (Rate Limiting) que ataca FN diferentes",
            "3. Reavaliar premissas da FASE 0 sobre o threshold sweep",
            "",
        ]
        report += "\n".join(recomendacao_linhas)

    # Metadata de execução
    metadata_linhas = [
        "---",
        "",
        "## 9. Metadata de Execução",
        "",
        f"- **Experimento:** {EXP_ID}",
        f"- **Sample size:** {config['sample']['n']:,} tx",
        f"- **Seed principal:** {config['sample']['seed_principal']}",
        f"- **Seed validação:** {config['sample']['seed_validacao']}",
        f"- **Tempo total:** {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)",
        "",
        "---",
        "",
        "*Gerado automaticamente por `experimentos/exp_001_threshold_final/run_exp_001.py`*",
        "",
    ]
    report += "\n" + "\n".join(metadata_linhas)

    return report


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """Orquestra a execução completa do EXP-001."""
    parser = argparse.ArgumentParser(
        description=f"{EXP_ID}: {EXP_TITLE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Override do sample size (default: lê de config_variantes.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Workers paralelos (1=sequencial, recomendado 2-4)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Pular validação cruzada em seed=123 (economiza ~metade do tempo)",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # -----------------------------------------------------
    # SETUP
    # -----------------------------------------------------
    print_section(f"{EXP_ID} — {EXP_TITLE}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    sample_n = args.sample if args.sample else config["sample"]["n"]
    seed_principal = config["sample"]["seed_principal"]

    output_dir = get_experiment_output_dir(EXP_ID)
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Sample size: {sample_n:,} | Seed principal: {seed_principal}")

    # -----------------------------------------------------
    # 1. CARREGAR E SAMPLEAR
    # -----------------------------------------------------
    print_section("1. Carregar dataset e gerar sample principal")
    df_full = load_dataset()
    logger.info(
        f"Dataset completo: {len(df_full):,} tx | {df_full['is_fraud'].sum()} fraudes"
    )

    df_sample = stratified_sample(
        df_full, n=sample_n, seed=seed_principal, logger=logger
    )

    # -----------------------------------------------------
    # 2. PROCESSAR UMA VEZ (base para todas as variantes)
    # -----------------------------------------------------
    print_section("2. Processar sample via PipelineOrquestrador (uma unica vez)")
    predictions_df = process_dataframe_via_orquestrador(
        df_sample,
        workers=args.workers,
        logger=logger,
    )

    # -----------------------------------------------------
    # 3. AVALIAR TODAS AS VARIANTES (post-hoc)
    # -----------------------------------------------------
    print_section("3. Avaliar baseline + 3 variantes (post-hoc)")
    df_comparativa = evaluate_all_variants(
        predictions_df,
        baseline_cfg=config["baseline"],
        variantes_cfg=config["variantes"],
    )
    print(df_comparativa[[
        "variante_id", "threshold_confirmar", "TP", "FP", "FN",
        "Precision", "Recall", "F1", "delta_F1",
    ]].to_string(index=False))

    # Vencedor = maior F1 entre as variantes (não conta baseline)
    df_variants_only = df_comparativa[df_comparativa["variante_id"] != "BASELINE"]
    winner_id = df_variants_only.loc[df_variants_only["F1"].idxmax(), "variante_id"]
    winner_cfg_full = next(v for v in config["variantes"] if v["id"] == winner_id)
    logger.info(
        f"🏆 Variante vencedora: {winner_id} "
        f"(threshold={winner_cfg_full['threshold_confirmar']})"
    )

    # -----------------------------------------------------
    # 4. SWEEP FINO
    # -----------------------------------------------------
    print_section("4. Threshold sweep fino (granularidade 1pt)")
    sweep_cfg = config["sweep_fino"]
    df_sweep = run_fine_threshold_sweep(
        predictions_df,
        t_min=sweep_cfg["threshold_min"],
        t_max=sweep_cfg["threshold_max"],
        step=sweep_cfg["step"],
    )
    best_sweep = df_sweep.loc[df_sweep["F1"].idxmax()]
    logger.info(
        f"Melhor F1 no sweep: {best_sweep['F1']:.4f} "
        f"@ threshold={int(best_sweep['threshold'])} "
        f"(TP={int(best_sweep['TP'])}, FP={int(best_sweep['FP'])}, "
        f"FN={int(best_sweep['FN'])})"
    )

    # -----------------------------------------------------
    # 5. ANÁLISE QUALITATIVA FP/FN
    # -----------------------------------------------------
    print_section("5. Analise qualitativa de FP/FN (baseline vs vencedor)")
    fp_fn_analysis = analyze_fp_fn_differences(
        predictions_df,
        baseline_th=config["baseline"]["threshold_confirmar"],
        winner_th=winner_cfg_full["threshold_confirmar"],
    )
    logger.info(
        f"FN recuperados: {fp_fn_analysis['fn_recuperados']['total']} | "
        f"FP novos: {fp_fn_analysis['fp_novos']['total']}"
    )
    if fp_fn_analysis["fp_novos"]["alertas"]:
        for a in fp_fn_analysis["fp_novos"]["alertas"]:
            logger.warning(a)

    # -----------------------------------------------------
    # 6. VALIDAÇÃO CRUZADA (opcional)
    # -----------------------------------------------------
    cross_val_result = None
    if not args.skip_validation:
        print_section("6. Validacao cruzada (seed=123)")
        cross_val_result = run_cross_validation(
            df_full=df_full,
            sample_n=sample_n,
            seed_validacao=config["sample"]["seed_validacao"],
            winner_cfg=winner_cfg_full,
            baseline_cfg=config["baseline"],
            workers=args.workers,
        )
        logger.info(cross_val_result["interpretacao"])
    else:
        logger.warning("Validacao cruzada SKIPPED (--skip-validation)")

    # -----------------------------------------------------
    # 7. CRITÉRIOS DE ACEITAÇÃO
    # -----------------------------------------------------
    print_section("7. Avaliar criterios de aceitacao")
    winner_row = df_comparativa[
        df_comparativa["variante_id"] == winner_id
    ].iloc[0].to_dict()
    baseline_row = df_comparativa[
        df_comparativa["variante_id"] == "BASELINE"
    ].iloc[0].to_dict()

    acceptance = evaluate_acceptance_criteria(
        winner_metrics=winner_row,
        baseline_metrics=baseline_row,
        criterios=config["criterios_aceitacao"],
    )
    status_str = "APROVADO" if acceptance["aprovado"] else "REPROVADO"
    logger.info(f"Resultado final: {status_str}")
    for name, check in acceptance["checks"].items():
        passou = "OK" if check["passou"] else "FAIL"
        logger.info(
            f"  [{passou}] {check['criterio']} -> obtido={check['valor_obtido']}"
        )

    # -----------------------------------------------------
    # 8. SALVAR 5 ARTEFATOS
    # -----------------------------------------------------
    print_section("8. Salvar artefatos (maximo 5)")
    elapsed_total = time.perf_counter() - t_start

    # Artefato 1: Tabela comparativa
    path1 = output_dir / "01_tabela_comparativa.csv"
    df_comparativa.to_csv(path1, index=False)
    logger.info(f"[1/5] {path1.name}")

    # Artefato 2: Sweep fino
    path2 = output_dir / "02_threshold_sweep_fino.csv"
    df_sweep.to_csv(path2, index=False)
    logger.info(f"[2/5] {path2.name}")

    # Artefato 3: Análise FP/FN
    path3 = output_dir / "03_analise_fp_fn.json"
    safe_json_dump(fp_fn_analysis, path3)
    logger.info(f"[3/5] {path3.name}")

    # Artefato 4: Validação cruzada (se houver)
    if cross_val_result:
        path4 = output_dir / "04_validacao_cruzada.json"
        safe_json_dump(cross_val_result, path4)
        logger.info(f"[4/5] {path4.name}")
    else:
        logger.info("[4/5] Pulado (--skip-validation)")

    # Artefato 5: Relatório executivo
    report_md = generate_executive_report(
        df_comparativa=df_comparativa,
        df_sweep=df_sweep,
        fp_fn_analysis=fp_fn_analysis,
        cross_val=cross_val_result,
        acceptance=acceptance,
        winner_cfg=winner_cfg_full,
        config=config,
        elapsed_total=elapsed_total,
    )
    path5 = output_dir / "05_conclusao_executiva.md"
    path5.write_text(report_md, encoding="utf-8")
    logger.info(f"[5/5] {path5.name}")

    # -----------------------------------------------------
    # 9. RESUMO
    # -----------------------------------------------------
    print_section("RESUMO FINAL")
    resumo_linhas = [
        "",
        f"  Experimento:    {EXP_ID}",
        f"  Vencedor:       {winner_id} (threshold={winner_cfg_full['threshold_confirmar']})",
        f"  F1:             {winner_row['F1']:.4f} "
        f"(baseline: {baseline_row['F1']:.4f}, delta: {winner_row['delta_F1']:+.4f})",
        f"  Recall:         {winner_row['Recall']:.2%} (baseline: {baseline_row['Recall']:.2%})",
        f"  Precision:      {winner_row['Precision']:.2%} (baseline: {baseline_row['Precision']:.2%})",
        f"  FN recuperados: {fp_fn_analysis['fn_recuperados']['total']}",
        f"  FP novos:       {fp_fn_analysis['fp_novos']['total']}",
        f"  Status:         {status_str}",
        f"  Tempo total:    {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)",
        "",
        f"  Artefatos em: {output_dir}",
        "    -> Comece lendo: 05_conclusao_executiva.md",
        "",
    ]
    print("\n".join(resumo_linhas))


if __name__ == "__main__":
    main()
 