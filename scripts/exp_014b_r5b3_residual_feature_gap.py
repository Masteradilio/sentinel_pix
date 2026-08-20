#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R5B3 — diagnóstico dos resíduos pós R5B2-FROZEN.

Compara:
  1. normais ainda em BLOQUEAR vs fraudes em BLOQUEAR
  2. fraudes ainda em APROVAR vs normais em APROVAR

O objetivo é orientar a próxima rodada de feature engineering/regras sem retreino.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R5B3-RESIDUAL-FEATURE-GAP"
SOURCE_EXPERIMENT = "EXP-014B-R5B2-FROZEN"

NUMERIC_COLS = [
    "vl_pix",
    "lgbm_raw",
    "score_final",
    "se_score",
    "beh_score",
    "qtd_pix_pagador_7d",
    "qtd_pix_pagador_30d",
    "qtd_pix_pagador_90d",
    "qtd_pix_pagador_180d",
    "valor_total_pagador_90d",
    "valor_total_pagador_180d",
    "max_qtd_pix_dia_pagador_7d",
    "max_qtd_pix_dia_pagador_30d",
    "valor_maximo_pix_pagador_180d",
    "soma_recebedores_distintos_dia_180d",
    "qtd_pix_mesmo_recebedor_7d",
    "qtd_pix_mesmo_recebedor_30d",
    "qtd_pix_mesmo_recebedor_90d",
    "qtd_pix_mesmo_recebedor_180d",
    "valor_total_para_recebedor_30d",
    "valor_total_para_recebedor_90d",
    "valor_total_para_recebedor_180d",
    "dias_desde_primeiro_envio_recebedor",
    "dias_desde_ultima_transacao_recebedor",
    "valor_medio_para_recebedor_180d",
    "ratio_valor_pix_vs_max_recebedor_180d",
    "qtd_pix_recebidos_30d",
    "qtd_pix_recebidos_90d",
    "qtd_pix_recebidos_180d",
    "valor_total_recebido_30d",
    "valor_total_recebido_90d",
    "valor_total_recebido_180d",
    "soma_pagadores_distintos_dia_recebedor_180d",
    "max_qtd_pix_recebidos_dia_180d",
    "ratio_valor_media_pagador_90d",
    "ratio_valor_maximo_pagador_180d",
    "mbk_completeness_score",
]

CATEGORICAL_COLS = [
    "temporal_split",
    "periodo_dia",
    "value_band",
    "ds_tipo_chave_norm",
    "sample_strategy",
    "mbk_available_flag",
    "first_receiver_flag_real",
    "is_recebedor_recorrente_180d_str",
    "qtd_pix_mesmo_recebedor_7d_bin",
    "dias_desde_ultima_transacao_recebedor_bin",
    "ratio_valor_pix_vs_max_recebedor_180d_bin",
    "module_quiet",
    "lgbm_bin",
    "score_bin",
    "ratio_bin",
    "qtd_rec_bin",
    "valor_rec_bin",
]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric_contrast(df: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series, label_a: str, label_b: str) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        a = pd.to_numeric(df.loc[mask_a, col], errors="coerce")
        b = pd.to_numeric(df.loc[mask_b, col], errors="coerce")
        if a.notna().sum() == 0 and b.notna().sum() == 0:
            continue
        a_med = float(a.median()) if a.notna().any() else np.nan
        b_med = float(b.median()) if b.notna().any() else np.nan
        rows.append({
            "feature": col,
            f"{label_a}_n": int(a.notna().sum()),
            f"{label_b}_n": int(b.notna().sum()),
            f"{label_a}_missing_rate": round(float(a.isna().mean()), 6) if len(a) else np.nan,
            f"{label_b}_missing_rate": round(float(b.isna().mean()), 6) if len(b) else np.nan,
            f"{label_a}_median": a_med,
            f"{label_b}_median": b_med,
            "median_delta_a_minus_b": a_med - b_med if np.isfinite(a_med) and np.isfinite(b_med) else np.nan,
            f"{label_a}_p90": float(a.quantile(0.9)) if a.notna().any() else np.nan,
            f"{label_b}_p90": float(b.quantile(0.9)) if b.notna().any() else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_median_delta"] = out["median_delta_a_minus_b"].abs()
        out = out.sort_values("abs_median_delta", ascending=False)
    return out


def categorical_contrast(df: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series, label_a: str, label_b: str) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        a = df.loc[mask_a, col].fillna("<MISSING>").astype(str)
        b = df.loc[mask_b, col].fillna("<MISSING>").astype(str)
        a_counts = a.value_counts()
        b_counts = b.value_counts()
        values = set(a_counts.index).union(set(b_counts.index))
        for value in values:
            a_n = int(a_counts.get(value, 0))
            b_n = int(b_counts.get(value, 0))
            rows.append({
                "feature": col,
                "value": value,
                f"{label_a}_n": a_n,
                f"{label_b}_n": b_n,
                f"{label_a}_rate": round(float(a_n / max(len(a), 1)), 6),
                f"{label_b}_rate": round(float(b_n / max(len(b), 1)), 6),
                "rate_delta_a_minus_b": round(float(a_n / max(len(a), 1) - b_n / max(len(b), 1)), 6),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_rate_delta"] = out["rate_delta_a_minus_b"].abs()
        out = out.sort_values(["abs_rate_delta", f"{label_a}_n", f"{label_b}_n"], ascending=[False, False, False])
    return out


def top_cases(df: pd.DataFrame, mask: pd.Series, output_cols: list[str], n: int = 100) -> pd.DataFrame:
    cols = [c for c in output_cols if c in df.columns]
    return df.loc[mask, cols].head(n).copy()


def main() -> None:
    root = Path.cwd()
    src = root / "resultados" / "experimentos" / SOURCE_EXPERIMENT / "06_predictions_frozen.csv"
    out_dir = root / "resultados" / "experimentos" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src, low_memory=False)
    y = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    action = df["r5b2_frozen_decisao"].astype(str).str.upper()

    block_normal = (action == "BLOQUEAR") & (y == 0)
    block_fraud = (action == "BLOQUEAR") & (y == 1)
    approve_fraud = (action == "APROVAR") & (y == 1)
    approve_normal = (action == "APROVAR") & (y == 0)

    group_metrics = {
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "n_rows": int(len(df)),
        "groups": {
            "block_normal": int(block_normal.sum()),
            "block_fraud": int(block_fraud.sum()),
            "approve_fraud": int(approve_fraud.sum()),
            "approve_normal": int(approve_normal.sum()),
        },
    }

    block_num = numeric_contrast(df, block_normal, block_fraud, "block_normal", "block_fraud")
    block_cat = categorical_contrast(df, block_normal, block_fraud, "block_normal", "block_fraud")
    approve_num = numeric_contrast(df, approve_fraud, approve_normal, "approve_fraud", "approve_normal")
    approve_cat = categorical_contrast(df, approve_fraud, approve_normal, "approve_fraud", "approve_normal")

    case_cols = [
        "transaction_id",
        "temporal_split",
        "event_datetime",
        "is_fraud",
        "r5b2_frozen_decisao",
        "vl_pix",
        "lgbm_raw",
        "score_final",
        "se_score",
        "beh_score",
        "ds_tipo_chave_norm",
        "periodo_dia",
        "value_band",
        "first_receiver_flag_real",
        "qtd_pix_mesmo_recebedor_180d",
        "valor_total_para_recebedor_180d",
        "qtd_pix_recebidos_180d",
        "valor_total_recebido_180d",
        "ratio_valor_media_pagador_90d",
        "ratio_valor_maximo_pagador_180d",
    ]

    candidate_gaps = {
        "block_normal_vs_block_fraud": {
            "top_numeric_median_deltas": block_num.head(12).to_dict(orient="records"),
            "top_categorical_rate_deltas": block_cat.head(12).to_dict(orient="records"),
            "hypothesis": (
                "Normais ainda em BLOQUEAR devem ser separados por confiança/reputação "
                "do recebedor, relacionamento recorrente e baixa evidência supervisionada."
            ),
        },
        "approve_fraud_vs_approve_normal": {
            "top_numeric_median_deltas": approve_num.head(12).to_dict(orient="records"),
            "top_categorical_rate_deltas": approve_cat.head(12).to_dict(orient="records"),
            "hypothesis": (
                "Fraudes ainda em APROVAR exigem resgate por sinais fracos combinados ou "
                "novo score high-recall; a etapa R5B2 não tinha headroom de FP para isso."
            ),
        },
        "recommended_next_steps": [
            "Validar subconjunto produtivo das regras BLOQUEAR->CONFIRMAR com critérios de robustez por mês/split.",
            "Criar features explícitas de trust do recebedor e força do par pagador-recebedor para reduzir bloqueio indevido.",
            "Abrir experimento separado de resgate APROVAR->CONFIRMAR, pois 682 fraudes ainda ficam aprovadas.",
        ],
    }

    write_json(out_dir / "00_run_summary.json", group_metrics)
    block_num.to_csv(out_dir / "01_block_normal_vs_block_fraud_numeric.csv", index=False)
    block_cat.to_csv(out_dir / "02_block_normal_vs_block_fraud_categorical.csv", index=False)
    approve_num.to_csv(out_dir / "03_approve_fraud_vs_approve_normal_numeric.csv", index=False)
    approve_cat.to_csv(out_dir / "04_approve_fraud_vs_approve_normal_categorical.csv", index=False)
    write_json(out_dir / "05_candidate_feature_gaps.json", candidate_gaps)
    top_cases(df, block_normal, case_cols).to_csv(out_dir / "06_block_normal_residual_cases.csv", index=False)
    top_cases(df, approve_fraud, case_cols).to_csv(out_dir / "07_approve_fraud_residual_cases.csv", index=False)

    report = f"""# {EXPERIMENT} — Diagnóstico de resíduos pós R5B2

## Grupos críticos
- Normais ainda em BLOQUEAR: `{int(block_normal.sum())}`
- Fraudes em BLOQUEAR: `{int(block_fraud.sum())}`
- Fraudes ainda em APROVAR: `{int(approve_fraud.sum())}`
- Normais em APROVAR: `{int(approve_normal.sum())}`

## Principais diferenças — normal BLOQUEAR vs fraude BLOQUEAR
{block_num.head(10).to_markdown(index=False)}

{block_cat.head(10).to_markdown(index=False)}

## Principais diferenças — fraude APROVAR vs normal APROVAR
{approve_num.head(10).to_markdown(index=False)}

{approve_cat.head(10).to_markdown(index=False)}

## Próximas ações recomendadas
1. Congelar um subconjunto produtivo/robusto das regras `BLOQUEAR -> CONFIRMAR`.
2. Criar features de reputação do recebedor e força do relacionamento pagador-recebedor.
3. Rodar experimento separado para resgatar fraudes em `APROVAR`, pois ainda restam `{int(approve_fraud.sum())}` casos.
"""
    (out_dir / "08_exp014b_r5b3_residual_feature_gap_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(group_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
