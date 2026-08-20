#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-011A — Replay Baseline no Dataset v2 rolling 180d

Objetivo:
  Avaliar o motor atual (PipelineOrquestrador + DecisionEngine + SE + BEH)
  no dataset oficial hmo_ml.tb_pix_dataset_v2_180d_v1 exportado como CSV,
  sem retreinar nenhum modelo.

Entrada esperada:
  dados/hmo_ml_tb_pix_dataset_v2_180d_v1.csv

Saidas:
  resultados/experimentos/EXP-011A/
    00_run_summary.json
    01_metrics_by_split.csv
    02_confusion_matrix_by_split.csv
    03_false_negatives_holdout.csv
    04_false_positives_holdout.csv
    05_decision_distribution.csv
    06_rule_hits.csv
    07_recommendation.md
    08_predictions.csv
    09_schema_coverage.csv
    10_module_activation.csv
    11_metrics_global.json

Uso:
  python scripts/exp_011a_replay_baseline_dataset_v2.py --full --workers 4

Teste rapido:
  python scripts/exp_011a_replay_baseline_dataset_v2.py --sample 5000 --workers 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
SCRIPT_PATH = Path(__file__).resolve()

# Este script pode ficar em:
#   rebuild_pix/scripts/exp_011a...
# ou backend/scripts/exp_011a...
if (SCRIPT_PATH.parent.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_PATH.parent.parent
elif (SCRIPT_PATH.parent.parent.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_PATH.parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd()

BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
DADOS_DIR = PROJECT_ROOT / "dados"
RESULTS_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-011A"
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"

DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v2_180d_v1.csv"

# Ordem importa: CORE_DIR antes de BACKEND_DIR para imports usados internamente
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Reaproveita a simulacao E2E oficial, que usa o PipelineOrquestrador real
try:
    from backend.scripts.simular_pipeline_e2e_v2 import (
        compute_metrics,
        process_batch_parallel,
        process_batch_sequential,
        validate_module_activations,
    )
except Exception:
    from scripts.simular_pipeline_e2e_v2 import (
        compute_metrics,
        process_batch_parallel,
        process_batch_sequential,
        validate_module_activations,
    )

try:
    from core.pipeline_orquestrador import RAW_INPUT_COLUMNS
except Exception:
    RAW_INPUT_COLUMNS = []

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-011A | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-011A")


# =============================================================================
# Helpers
# =============================================================================
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stratified_sample_by_split(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Amostra para teste rapido:
      - sempre preserva todas as fraudes se n >= n_fraudes;
      - completa com normais por amostra aleatoria;
      - ordena temporalmente para preservar caches sequenciais do orquestrador.
    """
    fraud = df[df["is_fraud"] == 1].copy()
    normal = df[df["is_fraud"] == 0].copy()

    n_fraud = len(fraud)
    n_norm = max(0, min(n - min(n_fraud, n), len(normal)))

    if n >= n_fraud:
        fraud_sample = fraud
    else:
        fraud_sample = fraud.sample(n=n, random_state=seed)

    normal_sample = normal.sample(n=n_norm, random_state=seed) if n_norm > 0 else normal.iloc[0:0]
    out = pd.concat([fraud_sample, normal_sample], axis=0)
    out = out.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)
    return out


def normalize_text_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})


def coalesce_columns(df: pd.DataFrame, target: str, sources: list[str]) -> pd.DataFrame:
    """
    Preenche target com a primeira fonte nao nula.

    Patch EXP-011A-R1:
      pandas pode inferir/criar target como float64 quando a coluna nasce
      vazia, por exemplo ds_tipo_chave=np.nan. Ao tentar preencher esse
      target com strings vindas de ds_tipo_chave_norm, ocorre:
        TypeError: Invalid value '<StringArray>...' for dtype 'float64'

      Por isso, sempre que fazemos coalesce, forçamos target e source para
      dtype object antes da atribuicao. Isso torna a funcao segura para
      aliases textuais e numericos.
    """
    if target not in df.columns:
        df[target] = pd.Series(pd.NA, index=df.index, dtype="object")
    else:
        df[target] = df[target].astype("object")

    null_tokens = {"", "nan", "none", "null", "nat", "<na>"}

    for src in sources:
        if src not in df.columns:
            continue

        src_values = df[src].astype("object")
        target_as_text = df[target].astype("string").fillna("").str.strip().str.lower()
        mask = df[target].isna() | target_as_text.isin(null_tokens)

        df.loc[mask, target] = src_values.loc[mask].values

    return df


def adapt_dataset_v2_to_pipeline_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta o CSV do EXP-010G ao schema bruto esperado pelo PipelineOrquestrador.

    Importante:
      O PipelineOrquestrador cria as features derivadas internamente.
      Aqui apenas garantimos aliases e colunas de entrada bruta.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Aliases basicos
    df = coalesce_columns(df, "cd_pix", ["transaction_id"])
    df = coalesce_columns(df, "transaction_id", ["cd_pix"])
    df = coalesce_columns(df, "cd_cpf_pagador", ["customer_id"])
    df = coalesce_columns(df, "customer_id", ["cd_cpf_pagador"])
    df = coalesce_columns(df, "cd_cpf_cnpj_recebedor", ["counterparty_id"])
    df = coalesce_columns(df, "dt_pix", ["event_datetime"])
    df = coalesce_columns(df, "event_datetime", ["dt_pix"])
    df = coalesce_columns(df, "ds_tipo_chave", ["ds_tipo_chave_norm"])
    df = coalesce_columns(df, "cd_retorno", ["autcodret"])

    # Padronizacoes criticas
    df["transaction_id"] = normalize_text_key(df["transaction_id"])
    df["cd_pix"] = normalize_text_key(df["cd_pix"])
    df["customer_id"] = normalize_text_key(df["customer_id"])
    df["cd_cpf_pagador"] = normalize_text_key(df["cd_cpf_pagador"])
    df["cd_cpf_cnpj_recebedor"] = normalize_text_key(df["cd_cpf_cnpj_recebedor"])

    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["dt_pix"] = pd.to_datetime(df["dt_pix"], errors="coerce")
    df["data_pix"] = pd.to_datetime(df.get("data_pix", df["dt_pix"]), errors="coerce").dt.date

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    # Garantir colunas brutas esperadas. O orquestrador/preprocessing lida com NaN.
    for col in RAW_INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Campos usados nas analises do experimento
    for col in ["temporal_split", "dataset_role", "source_dataset", "sample_strategy", "sample_weight"]:
        if col not in df.columns:
            df[col] = np.nan

    # Ordenacao temporal é essencial porque o PipelineOrquestrador mantém cache sequencial
    df = df.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)
    return df


def sanitize_dataset_for_e2e_replay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitiza o dataset antes de chamar simular_pipeline_e2e_v2.process_single_tx.

    Motivo do patch EXP-011A-R2:
      simular_pipeline_e2e_v2.process_single_tx monta alguns campos de saída com
      int(row.get(...)), por exemplo nr_idade. Quando o CSV v2 não possui essas
      features cadastrais/históricas, elas entram como NaN; int(np.nan) gera:
        ValueError: cannot convert float NaN to integer

    Esta sanitização não treina nem altera o motor. Ela apenas transforma ausências
    em defaults seguros para permitir o replay baseline e, ao mesmo tempo, preserva
    um relatório de cobertura em 09_schema_coverage.csv.
    """
    df = df.copy()

    # Colunas que o wrapper E2E converte com int(...) no retorno analítico.
    int_defaults = {
        "nr_idade": 0,
        "qt_tempo_relacionamento_mes": 0,
        "is_first_tx_trimestre": 0,
        "first_receiver_flag": 0,
        "burst_30m_flag": 0,
        "pix_key_random_flag": 0,
        "perfil_vulneravel_se_flag": 0,
    }

    for col, default in int_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)

    # Colunas numéricas comuns do RAW schema. Mantemos NaN onde o pipeline consegue
    # lidar, exceto campos usados diretamente em comparações ou casts externos.
    numeric_safe_defaults = {
        "vl_pix": 0.0,
        "qt_total_pix_trimestre": 0.0,
        "vl_mediana_pix_trimestre": np.nan,
        "vl_desvio_padrao_pix_trimestre": np.nan,
        "qt_intervalo_transacao_minuto": np.nan,
        "qt_intervalo_mediana_trimestre": np.nan,
        "qt_intervalo_desvio_padrao_trimestre": np.nan,
        "qt_pix_dia_maximo_trimestre": 0.0,
        "latencia_rede_ms": np.nan,
        "vl_latencia_rede_media_trimestre": np.nan,
        "tempo_interacao_ms": np.nan,
        "vl_tempo_interacao_medio_trimestre": np.nan,
        "tempo_processamento_host_ms": np.nan,
        "topaz_risk_score": np.nan,
        "topaz_transacao_rejeitada": 0.0,
        "qt_aparelhos_distintos_trimestre": 0.0,
        "vl_renda_cliente": np.nan,
        "qt_dependentes": 0.0,
    }

    for col, default in numeric_safe_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not pd.isna(default):
            df[col] = df[col].fillna(default)

    # Texto: garantir strings ou NaN, sem objetos estranhos.
    text_defaults = {
        "ds_tipo_chave": "INFORMACAO_AUSENTE",
        "ds_chave_pix": np.nan,
        "device_name": np.nan,
        "app_version": np.nan,
        "ip_address": np.nan,
        "metodo_autenticacao": np.nan,
        "session_id": np.nan,
        "cd_retorno": np.nan,
        "is_agendamento_recorrente": np.nan,
        "ds_sexo": np.nan,
        "ds_estado_civil": np.nan,
        "ds_segmento": np.nan,
    }

    for col, default in text_defaults.items():
        if col not in df.columns:
            df[col] = default
        if pd.isna(default):
            df[col] = df[col].where(df[col].notna(), np.nan)
        else:
            df[col] = df[col].fillna(default)

    # IDs e datas obrigatórios para o replay.
    for col in ["transaction_id", "cd_pix", "customer_id", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor"]:
        if col in df.columns:
            df[col] = normalize_text_key(df[col])

    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["dt_pix"] = pd.to_datetime(df["dt_pix"], errors="coerce")

    return df


def build_schema_coverage(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Relatorio de cobertura das colunas que o PipelineOrquestrador espera.
    """
    rows = []
    expected = list(RAW_INPUT_COLUMNS) if RAW_INPUT_COLUMNS else []
    for col in expected:
        exists = col in input_df.columns
        non_null = int(input_df[col].notna().sum()) if exists else 0
        rows.append({
            "column": col,
            "exists": int(exists),
            "non_null": non_null,
            "total": len(input_df),
            "coverage_pct": round(100.0 * non_null / max(len(input_df), 1), 4),
        })

    # Tambem registrar colunas extras relevantes do dataset v2
    for col in [
        "temporal_split", "dataset_role", "source_dataset", "sample_strategy",
        "sample_weight", "mbk_available_flag", "window_start_date", "window_end_date",
    ]:
        if col in input_df.columns:
            rows.append({
                "column": col,
                "exists": 1,
                "non_null": int(input_df[col].notna().sum()),
                "total": len(input_df),
                "coverage_pct": round(100.0 * input_df[col].notna().sum() / max(len(input_df), 1), 4),
            })
    return pd.DataFrame(rows)


def normalize_predictions_join(input_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une predicoes ao dataset de entrada pelo indice de processamento.
    process_batch_* retorna a coluna idx local; como processamos df resetado,
    idx == indice da linha.
    """
    base = input_df.reset_index(drop=True).copy()
    pred = pred_df.copy()
    pred["idx"] = pd.to_numeric(pred["idx"], errors="coerce").astype("Int64")

    merged = base.reset_index().rename(columns={"index": "idx"}).merge(
        pred,
        on="idx",
        how="left",
        suffixes=("", "_pred"),
    )

    # Normalizar colunas principais vindas da predicao
    if "transaction_id_pred" in merged.columns:
        merged["transaction_id_replay"] = merged["transaction_id_pred"]
    if "customer_id_pred" in merged.columns:
        merged["customer_id_replay"] = merged["customer_id_pred"]

    return merged


def compute_metrics_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y_true = g["is_fraud"].astype(int).values
        y_pred = g["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values
        m = compute_metrics(y_true, y_pred, label="|".join(map(str, keys)))
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update({
            "n": len(g),
            "tp": m["TP"],
            "fp": m["FP"],
            "fn": m["FN"],
            "tn": m["TN"],
            "precision": m["Precision"],
            "recall": m["Recall"],
            "f1": m["F1"],
            "fpr": m["FPR"],
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def confusion_matrix_by_split(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for split, g in df.groupby("temporal_split", dropna=False):
        for decisao, gd in g.groupby("decisao", dropna=False):
            out.append({
                "temporal_split": split,
                "decisao": decisao,
                "qtd": len(gd),
                "fraudes": int(gd["is_fraud"].sum()),
                "normais": int((gd["is_fraud"] == 0).sum()),
                "taxa_fraude": round(float(gd["is_fraud"].mean()), 6) if len(gd) else 0.0,
            })
    return pd.DataFrame(out).sort_values(["temporal_split", "decisao"]).reset_index(drop=True)


def decision_distribution(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["temporal_split", "dataset_role", "sample_strategy", "decisao"]
    cols = [c for c in cols if c in df.columns]
    out = (
        df.groupby(cols, dropna=False)
        .agg(qtd=("is_fraud", "size"), fraudes=("is_fraud", "sum"))
        .reset_index()
    )
    out["normais"] = out["qtd"] - out["fraudes"]
    out["taxa_fraude"] = (out["fraudes"] / out["qtd"].clip(lower=1)).round(6)
    return out.sort_values(cols).reset_index(drop=True)


def rule_hits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if "cascade_rules" in df.columns:
        tmp = df[["is_fraud", "decisao", "cascade_rules"]].copy()
        tmp["rule"] = tmp["cascade_rules"].fillna("").astype(str)
        tmp = tmp[tmp["rule"].str.len() > 0]
        if not tmp.empty:
            for rule, g in tmp.groupby("rule"):
                rows.append({
                    "rule_type": "cascade_rules",
                    "rule": rule,
                    "qtd": len(g),
                    "fraudes": int(g["is_fraud"].sum()),
                    "normais": int((g["is_fraud"] == 0).sum()),
                    "precision": round(float(g["is_fraud"].mean()), 6),
                })

    for col in ["veto_reason", "veto_aplicado", "veto_suppressed_reason"]:
        if col in df.columns:
            tmp = df[["is_fraud", "decisao", col]].copy()
            tmp["rule"] = tmp[col].fillna("").astype(str).str.slice(0, 160)
            tmp = tmp[tmp["rule"].str.len() > 0]
            if not tmp.empty:
                for rule, g in tmp.groupby("rule"):
                    rows.append({
                        "rule_type": col,
                        "rule": rule,
                        "qtd": len(g),
                        "fraudes": int(g["is_fraud"].sum()),
                        "normais": int((g["is_fraud"] == 0).sum()),
                        "precision": round(float(g["is_fraud"].mean()), 6),
                    })

    if not rows:
        return pd.DataFrame(columns=["rule_type", "rule", "qtd", "fraudes", "normais", "precision"])

    return pd.DataFrame(rows).sort_values(["rule_type", "qtd"], ascending=[True, False]).reset_index(drop=True)


def module_activation_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("temporal_split", dropna=False):
        n = len(g)
        rows.append({
            "temporal_split": split,
            "n": n,
            "se_active": int((pd.to_numeric(g.get("se_score", 0), errors="coerce").fillna(0) > 0).sum()),
            "se_active_pct": round(100.0 * (pd.to_numeric(g.get("se_score", 0), errors="coerce").fillna(0) > 0).mean(), 4),
            "beh_active": int((pd.to_numeric(g.get("beh_score", 0), errors="coerce").fillna(0) > 0).sum()),
            "beh_active_pct": round(100.0 * (pd.to_numeric(g.get("beh_score", 0), errors="coerce").fillna(0) > 0).mean(), 4),
            "cascade_active": int((g.get("cascade_triggered", False) == True).sum()) if "cascade_triggered" in g.columns else 0,
            "cascade_active_pct": round(100.0 * ((g.get("cascade_triggered", False) == True).mean() if "cascade_triggered" in g.columns else 0), 4),
        })
    return pd.DataFrame(rows)


def recommendation_text(summary: dict[str, Any], metrics_split: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-011A — Recommendation")
    lines.append("")
    lines.append("## Status")
    if summary["n_errors"] == 0 and summary["critical_alerts"] == 0:
        lines.append("APROVADO_PARA_ANALISE_METRICA")
    else:
        lines.append("APROVADO_COM_ALERTAS_DE_EXECUCAO")
    lines.append("")
    lines.append("## Resumo")
    lines.append(f"- Linhas processadas: {summary['n_processed']}")
    lines.append(f"- Fraudes: {summary['n_fraud']}")
    lines.append(f"- Normais: {summary['n_normal']}")
    lines.append(f"- Erros de processamento: {summary['n_errors']}")
    lines.append(f"- Tempo total: {summary['elapsed_seconds']}s")
    lines.append("")
    lines.append("## Proxima decisao")
    lines.append(
        "Comparar as metricas do HOLDOUT com o baseline pos-C1. "
        "Se recall/F1 cairem por falta de features historicas, o proximo passo deve ser EXP-010G-R2 "
        "para enriquecer a tabela v2 com as features do treinamento atual antes de treinar modelo novo."
    )
    lines.append("")
    lines.append("## Metricas por split")
    lines.append("")
    if metrics_split.empty:
        lines.append("(sem metricas)")
    else:
        lines.append(metrics_split.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-011A — Replay baseline no dataset v2 rolling 180d",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="Processa dataset completo.")
    group.add_argument("--sample", type=int, help="Processa amostra estratificada de N linhas.")

    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="CSV de entrada.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Diretorio de saida.")
    parser.add_argument("--workers", type=int, default=1, help="Workers paralelos. Use 1 para sequencial.")
    parser.add_argument("--seed", type=int, default=42, help="Seed da amostra.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Log de progresso no modo sequencial.")
    parser.add_argument("--copy-input-to-dados", action="store_true", help="Copia o input para dados/ com nome padrao.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV de entrada nao encontrado: {input_path}")

    if args.copy_input_to_dados:
        safe_mkdir(DADOS_DIR)
        dst = DEFAULT_INPUT
        if input_path.resolve() != dst.resolve():
            shutil.copy2(input_path, dst)
            log.info("Input copiado para %s", dst)

    log.info("Input: %s", input_path)
    log.info("Output: %s", output_dir)
    log.info("MD5: %s", file_md5(input_path))

    # -------------------------------------------------------------------------
    # 1. Load + adaptacao de schema
    # -------------------------------------------------------------------------
    df_raw = pd.read_csv(input_path, low_memory=False)
    df = adapt_dataset_v2_to_pipeline_schema(df_raw)
    df = sanitize_dataset_for_e2e_replay(df)

    # Garantias minimas
    required = ["transaction_id", "cd_pix", "customer_id", "cd_cpf_pagador", "dt_pix", "event_datetime", "is_fraud"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise RuntimeError(f"Colunas obrigatorias ausentes apos adaptacao: {missing_required}")

    df = df[df["transaction_id"].notna()].copy()
    df = df[df["event_datetime"].notna()].copy()
    df = df.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)

    if args.sample:
        df = stratified_sample_by_split(df, args.sample, seed=args.seed)
        df = sanitize_dataset_for_e2e_replay(df)

    schema_df = build_schema_coverage(df)
    schema_df.to_csv(output_dir / "09_schema_coverage.csv", index=False)

    log.info(
        "Dataset EXP-011A: %s linhas | %s fraudes | periodo %s -> %s",
        f"{len(df):,}",
        int(df["is_fraud"].sum()),
        df["event_datetime"].min(),
        df["event_datetime"].max(),
    )

    # -------------------------------------------------------------------------
    # 2. Processamento pelo PipelineOrquestrador real
    # -------------------------------------------------------------------------
    if args.workers > 1:
        pred_df = process_batch_parallel(df, n_workers=args.workers)
    else:
        pred_df = process_batch_sequential(df, progress_every=args.progress_every)

    merged = normalize_predictions_join(df, pred_df)

    # -------------------------------------------------------------------------
    # 3. Guardrails e metricas
    # -------------------------------------------------------------------------
    alerts = validate_module_activations(pred_df)
    n_errors = int((pred_df.get("decisao", "") == "ERRO").sum()) if "decisao" in pred_df.columns else 0

    metrics_global = compute_metrics(
        merged["is_fraud"].astype(int).values,
        merged["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]).astype(int).values,
        label="GLOBAL",
    )

    metrics_by_split = compute_metrics_by_group(merged, ["temporal_split"])
    if "dataset_role" in merged.columns:
        metrics_by_split_role = compute_metrics_by_group(merged, ["temporal_split", "dataset_role"])
        metrics_by_split = pd.concat([metrics_by_split, metrics_by_split_role], ignore_index=True, sort=False)

    cm_split = confusion_matrix_by_split(merged)
    dist_decision = decision_distribution(merged)
    rules = rule_hits(merged)
    module_report = module_activation_report(merged)

    # FNs e FPs, com foco no HOLDOUT
    flagged = merged["decisao"].isin(["CONFIRMAR", "BLOQUEAR"])
    holdout = merged[merged["temporal_split"].astype(str).str.upper() == "HOLDOUT"].copy()
    fn_holdout = holdout[(holdout["is_fraud"] == 1) & (~holdout["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]))].copy()
    fp_holdout = holdout[(holdout["is_fraud"] == 0) & (holdout["decisao"].isin(["CONFIRMAR", "BLOQUEAR"]))].copy()

    # Ordenacao util
    if "vl_pix" in fn_holdout.columns:
        fn_holdout = fn_holdout.sort_values("vl_pix", ascending=False)
    if "score_final" in fp_holdout.columns:
        fp_holdout = fp_holdout.sort_values("score_final", ascending=False)

    # -------------------------------------------------------------------------
    # 4. Salvar artefatos
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - t0

    run_summary = {
        "experiment": "EXP-011A",
        "status": "DONE" if n_errors == 0 else "DONE_WITH_ERRORS",
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "output_dir": str(output_dir),
        "n_input_after_filter": int(len(df)),
        "n_processed": int(len(pred_df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "n_normal": int((df["is_fraud"] == 0).sum()),
        "n_errors": n_errors,
        "critical_alerts": len(alerts),
        "alerts": alerts,
        "workers": args.workers,
        "mode": "sample" if args.sample else "full",
        "sample_n": args.sample,
        "elapsed_seconds": round(elapsed, 2),
        "rows_per_second": round(len(df) / max(elapsed, 1e-9), 4),
        "period_min": str(df["event_datetime"].min()),
        "period_max": str(df["event_datetime"].max()),
        "pipeline_metrics_global": metrics_global,
    }

    safe_json_dump(run_summary, output_dir / "00_run_summary.json")
    metrics_by_split.to_csv(output_dir / "01_metrics_by_split.csv", index=False)
    cm_split.to_csv(output_dir / "02_confusion_matrix_by_split.csv", index=False)
    fn_holdout.to_csv(output_dir / "03_false_negatives_holdout.csv", index=False)
    fp_holdout.to_csv(output_dir / "04_false_positives_holdout.csv", index=False)
    dist_decision.to_csv(output_dir / "05_decision_distribution.csv", index=False)
    rules.to_csv(output_dir / "06_rule_hits.csv", index=False)
    merged.to_csv(output_dir / "08_predictions.csv", index=False)
    module_report.to_csv(output_dir / "10_module_activation.csv", index=False)
    safe_json_dump(metrics_global, output_dir / "11_metrics_global.json")

    rec = recommendation_text(run_summary, metrics_by_split)
    (output_dir / "07_recommendation.md").write_text(rec, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 5. Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXP-011A — REPLAY BASELINE DATASET V2")
    print("=" * 72)
    print(f"Input:        {input_path}")
    print(f"Output:       {output_dir}")
    print(f"Processadas:  {len(pred_df):,}")
    print(f"Fraudes:      {int(df['is_fraud'].sum()):,}")
    print(f"Normais:      {int((df['is_fraud'] == 0).sum()):,}")
    print(f"Erros:        {n_errors}")
    print(f"Tempo:        {elapsed/60:.1f} min")
    print("")
    print("Metricas globais:")
    print(json.dumps(metrics_global, indent=2, ensure_ascii=False))
    print("")
    print("Artefatos principais:")
    for name in [
        "00_run_summary.json",
        "01_metrics_by_split.csv",
        "02_confusion_matrix_by_split.csv",
        "03_false_negatives_holdout.csv",
        "04_false_positives_holdout.csv",
        "05_decision_distribution.csv",
        "06_rule_hits.csv",
        "07_recommendation.md",
        "08_predictions.csv",
        "09_schema_coverage.csv",
        "10_module_activation.csv",
        "11_metrics_global.json",
    ]:
        print(f"  - {output_dir / name}")

    if alerts:
        print("\nALERTAS:")
        for a in alerts:
            print(f"  - {a}")


if __name__ == "__main__":
    main()
