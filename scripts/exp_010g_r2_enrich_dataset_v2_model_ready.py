#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-010G-R2 — Enriquecimento model-ready do Dataset v2 rolling 180d

Objetivo:
  Ler o dataset unificado aprovado no EXP-010G:
      dados/hmo_ml_tb_pix_dataset_v2_180d_v1.csv

  Reaproveitar a lógica leakage-free do preprocessing.py para gerar um CSV
  enriquecido, com features históricas, sequenciais, rolling 90d e graph temporal,
  sem sobrescrever artefatos de produção como:
      backend/artefatos/preprocessing.joblib

Saídas:
  dados/hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv
  dados/base_treino_final_exp010g_r2.csv
  resultados/experimentos/EXP-010G-R2/
      00_run_summary.json
      01_schema_coverage_before.csv
      02_schema_coverage_after.csv
      03_feature_diagnostics.csv
      04_class_split_summary.csv

Uso:
  python scripts/exp_010g_r2_enrich_dataset_v2_model_ready.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================
SCRIPT_PATH = Path(__file__).resolve()

if (SCRIPT_PATH.parent.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_PATH.parent.parent
elif (SCRIPT_PATH.parent.parent.parent / "backend").exists():
    PROJECT_ROOT = SCRIPT_PATH.parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd()

BACKEND_DIR = PROJECT_ROOT / "backend"
CORE_DIR = BACKEND_DIR / "core"
DADOS_DIR = PROJECT_ROOT / "dados"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-010G-R2"

sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Reutiliza funcoes oficiais do preprocessing.py, mas NAO chama main(),
# para nao sobrescrever preprocessing.joblib nem base_treino_final.csv.
import preprocessing as pp  # type: ignore

DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v2_180d_v1.csv"
DEFAULT_OUTPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv"
DEFAULT_ALIAS_OUTPUT = DADOS_DIR / "base_treino_final_exp010g_r2.csv"

REQUIRED_RAW_COLS = [
    "cd_pix", "dt_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
    "ds_chave_pix", "ds_tipo_chave", "vl_pix",
    "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "device_name", "app_version", "ip_address",
    "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
    "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
    "tempo_processamento_host_ms", "metodo_autenticacao", "session_id",
    "cd_retorno", "topaz_risk_score", "topaz_transacao_rejeitada",
    "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
    "ds_sexo", "ds_estado_civil", "ds_segmento",
    "tp_primeiro_envio_recebedor_trimestre", "qt_envio_recebedor_trimestre",
    "vl_renda_cliente",
    "is_fraud", "source_dataset", "dt_carga",
]

TEXT_COLS = [
    "cd_pix", "cd_cpf_pagador", "cd_cpf_cnpj_recebedor",
    "ds_chave_pix", "ds_tipo_chave", "device_name", "app_version",
    "ip_address", "metodo_autenticacao", "session_id", "cd_retorno",
    "source_dataset", "ds_sexo", "ds_estado_civil", "ds_segmento",
]

NUM_COLS = [
    "vl_pix", "qt_total_pix_trimestre", "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre", "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre", "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre", "latencia_rede_ms",
    "vl_latencia_rede_media_trimestre", "tempo_interacao_ms",
    "vl_tempo_interacao_medio_trimestre", "tempo_processamento_host_ms",
    "topaz_risk_score", "topaz_transacao_rejeitada",
    "qt_aparelhos_distintos_trimestre", "nr_idade", "qt_tempo_relacionamento_mes",
    "is_fraud", "tp_primeiro_envio_recebedor_trimestre",
    "qt_envio_recebedor_trimestre", "vl_renda_cliente",
]

SENTINEL_COLS = [
    "latencia_rede_ms", "vl_latencia_rede_media_trimestre",
    "tempo_interacao_ms", "vl_tempo_interacao_medio_trimestre",
    "tempo_processamento_host_ms", "topaz_risk_score",
    "topaz_transacao_rejeitada",
]

METADATA_COLS = [
    "transaction_id", "temporal_split", "dataset_role", "sample_strategy",
    "sample_weight", "mbk_available_flag", "window_start_date", "window_end_date",
    "source_dataset_original", "dataset_created_at",
]

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-010G-R2 | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-010G-R2")


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


def coalesce(df: pd.DataFrame, target: str, sources: list[str]) -> pd.DataFrame:
    if target not in df.columns:
        df[target] = pd.NA
    df[target] = df[target].astype("object")
    null_tokens = {"", "nan", "none", "null", "nat", "<na>"}
    for src in sources:
        if src not in df.columns:
            continue
        src_values = df[src].astype("object")
        target_text = df[target].astype("string").fillna("").str.strip().str.lower()
        mask = df[target].isna() | target_text.isin(null_tokens)
        df.loc[mask, target] = src_values.loc[mask].values
    return df


def normalize_pix_key_type(x: Any) -> Any:
    """Converte labels canonicas do EXP-010G para formato esperado pelo preprocessing.py."""
    if pd.isna(x):
        return "Informação ausente"
    s = str(x).strip().upper()
    s = (
        s.replace("Ç", "C")
        .replace("Ã", "A")
        .replace("Á", "A")
        .replace("É", "E")
        .replace("_", " ")
    )
    if "CHAVE" in s and "ALEATORIA" in s:
        return "CHAVE ALEATORIA"
    if "DOCUMENTO" in s or "TELEFONE" in s:
        return "DOCUMENTO/TELEFONE"
    if s == "EMAIL":
        return "EMAIL"
    if "INFORMACAO" in s and "AUSENTE" in s:
        return "Informação ausente"
    if s in ("NAN", "NONE", "NULL", ""):
        return "Informação ausente"
    return "OUTROS"


def build_schema_coverage(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        exists = col in df.columns
        non_null = int(df[col].notna().sum()) if exists else 0
        rows.append({
            "column": col,
            "exists": int(exists),
            "non_null": non_null,
            "total": len(df),
            "coverage_pct": round(100.0 * non_null / max(len(df), 1), 4),
            "dtype": str(df[col].dtype) if exists else None,
        })
    return pd.DataFrame(rows)


def adapt_exp010g_dataset_to_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adapta hmo_ml_tb_pix_dataset_v2_180d_v1.csv para o schema bruto esperado
    pelas funcoes de preprocessing.py.
    """
    df = pp.standardize_columns(df)
    df = df.copy()

    # Preservar source_dataset original do EXP-010G antes de converter para normal/fraud.
    if "source_dataset" in df.columns:
        df["source_dataset_original"] = df["source_dataset"]
    else:
        df["source_dataset_original"] = pd.NA

    df = coalesce(df, "cd_pix", ["transaction_id"])
    df = coalesce(df, "transaction_id", ["cd_pix"])
    df = coalesce(df, "cd_cpf_pagador", ["customer_id"])
    df = coalesce(df, "customer_id", ["cd_cpf_pagador"])
    df = coalesce(df, "cd_cpf_cnpj_recebedor", ["counterparty_id"])
    df = coalesce(df, "dt_pix", ["event_datetime"])
    df = coalesce(df, "event_datetime", ["dt_pix"])
    df = coalesce(df, "ds_tipo_chave", ["ds_tipo_chave_norm"])
    df = coalesce(df, "cd_retorno", ["autcodret"])

    if "dt_carga" not in df.columns:
        df["dt_carga"] = pd.Timestamp.now()

    # source_dataset no preprocessing antigo espera normal/fraud.
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["source_dataset"] = np.where(df["is_fraud"] == 1, "fraud", "normal")

    # Tipo de chave no formato esperado por classify_key_flags().
    df["ds_tipo_chave"] = df["ds_tipo_chave"].apply(normalize_pix_key_type)

    for c in REQUIRED_RAW_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Conversões equivalentes ao load_and_prepare_pix().
    df = pp.clean_text_columns(df, [c for c in TEXT_COLS if c in df.columns])
    df = pp.safe_to_numeric(df, [c for c in NUM_COLS if c in df.columns])
    df = pp.safe_to_datetime(df, ["dt_pix", "dt_carga"])

    df["cd_pix"] = pp.normalize_transaction_key(df["cd_pix"])
    df["transaction_id"] = pp.normalize_transaction_key(df["transaction_id"])
    df["cd_cpf_pagador"] = df["cd_cpf_pagador"].astype("object")
    df["customer_id"] = df["customer_id"].astype("object")
    df["cd_cpf_cnpj_recebedor"] = df["cd_cpf_cnpj_recebedor"].astype("object")

    df = pp.replace_sentinels_with_nan(df, [c for c in SENTINEL_COLS if c in df.columns])
    df = pp.replace_zero_with_nan(df, ["vl_tempo_interacao_medio_trimestre"])

    # Remover registros sem chave/data, pois quebram as features temporais.
    df = df[df["cd_pix"].notna()].copy()
    df = df[df["dt_pix"].notna()].copy()

    metadata_cols = [c for c in METADATA_COLS if c in df.columns]
    metadata = df[metadata_cols].drop_duplicates("transaction_id", keep="first").copy()

    return df[REQUIRED_RAW_COLS + [c for c in metadata_cols if c not in REQUIRED_RAW_COLS]], metadata


def class_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["temporal_split", "dataset_role", "source_dataset", "is_fraud"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = (
        df.groupby(cols, dropna=False)
        .agg(qtd=("transaction_id", "size"))
        .reset_index()
        .sort_values(cols)
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-010G-R2 — Enriquecimento model-ready do Dataset v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV do dataset v2 do EXP-010G.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV enriquecido principal.")
    parser.add_argument("--alias-output", default=str(DEFAULT_ALIAS_OUTPUT), help="Copia alias para facilitar scripts.")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Diretorio de artefatos.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_path = Path(args.output)
    alias_output = Path(args.alias_output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alias_output.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    print("=" * 80)
    print("EXP-010G-R2 — Dataset v2 Feature Enrichment")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"MD5:    {file_md5(input_path)}")

    log.info("Carregando dataset v2...")
    raw0 = pd.read_csv(input_path, low_memory=False)
    coverage_before = build_schema_coverage(raw0, sorted(set(REQUIRED_RAW_COLS + list(raw0.columns))))
    coverage_before.to_csv(output_dir / "01_schema_coverage_before.csv", index=False)

    log.info("Adaptando schema EXP-010G -> RAW preprocessing...")
    raw, metadata = adapt_exp010g_dataset_to_raw(raw0)

    n0 = len(raw)
    n_fraud0 = int(raw["is_fraud"].sum())
    n_normal0 = int((raw["is_fraud"] == 0).sum())
    log.info("Base adaptada: %d linhas | %d fraudes | %d normais", n0, n_fraud0, n_normal0)

    # Deduplicação equivalente ao main do preprocessing.
    log.info("Deduplicando por cd_pix...")
    priority_cols = [
        "cd_cpf_cnpj_recebedor", "ds_chave_pix", "ds_tipo_chave",
        "device_name", "app_version", "ip_address", "latencia_rede_ms",
        "tempo_interacao_ms", "tempo_processamento_host_ms",
        "metodo_autenticacao", "session_id", "cd_retorno",
        "topaz_risk_score", "topaz_transacao_rejeitada",
        "ds_sexo", "ds_estado_civil", "ds_segmento",
        "tp_primeiro_envio_recebedor_trimestre",
    ]
    raw = pp.deduplicate_by_key(raw, "cd_pix", priority_cols)
    raw["transaction_id"] = raw["cd_pix"]

    log.info("Feature engineering completa...")
    df = pp.create_all_features(raw)

    log.info("Deduplicação final por transaction_id...")
    priority_final = [
        "latencia_rede_ms_final", "tempo_processamento_host_ms",
        "device_name", "app_version", "ip_address", "metodo_autenticacao",
        "cd_retorno", "topaz_risk_score", "cd_cpf_cnpj_recebedor",
        "ds_chave_pix", "ds_tipo_chave", "session_id",
        "tempo_interacao_ms", "ds_sexo", "ds_estado_civil",
    ]
    df = pp.deduplicate_by_key(df, "transaction_id", priority_final)

    log.info("Fix leakage temporal rolling 90d por CPF...")
    df = pp.fix_leakage_temporal(df)

    log.info("Graph features temporais incrementais...")
    df_graph = pp.compute_temporal_graph_features(df)
    df = df.merge(df_graph, on="transaction_id", how="left")
    graph_cols = [c for c in df_graph.columns if c != "transaction_id"]
    df[graph_cols] = df[graph_cols].fillna(0)

    log.info("Selecionando colunas finais...")
    df_features = pp.select_final_columns(df)

    # Reanexar metadata útil do EXP-010G.
    if not metadata.empty:
        df_features = df_features.merge(metadata, on="transaction_id", how="left", suffixes=("", "_meta"))

    # Garantir split/roles se vieram no input.
    if "temporal_split" not in df_features.columns:
        df_features["temporal_split"] = pd.NA
    if "dataset_role" not in df_features.columns:
        df_features["dataset_role"] = np.where(df_features["is_fraud"] == 1, "POSITIVE_FRAUD", "NEGATIVE_NORMAL")

    # Diagnóstico.
    id_cols = ["transaction_id", "customer_id", "event_datetime", "source_dataset", "is_fraud"]
    diag = pp.diagnose_features(df_features, id_cols)
    diag.to_csv(output_dir / "03_feature_diagnostics.csv", index=False)

    coverage_after = build_schema_coverage(df_features, sorted(df_features.columns))
    coverage_after.to_csv(output_dir / "02_schema_coverage_after.csv", index=False)

    summary_split = class_split_summary(df_features)
    summary_split.to_csv(output_dir / "04_class_split_summary.csv", index=False)

    # Salvar CSV enriquecido.
    df_features = df_features.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)
    df_features.to_csv(output_path, index=False)
    shutil.copy2(output_path, alias_output)

    elapsed = time.perf_counter() - t0

    run_summary = {
        "experiment": "EXP-010G-R2",
        "status": "DONE",
        "input_path": str(input_path),
        "input_md5": file_md5(input_path),
        "output_path": str(output_path),
        "alias_output": str(alias_output),
        "n_input": int(len(raw0)),
        "n_after_raw_adapt": int(n0),
        "n_output": int(len(df_features)),
        "n_fraud_output": int(df_features["is_fraud"].sum()),
        "n_normal_output": int((df_features["is_fraud"] == 0).sum()),
        "duplicated_transaction_id": int(len(df_features) - df_features["transaction_id"].nunique()),
        "dt_min": str(pd.to_datetime(df_features["event_datetime"], errors="coerce").min()),
        "dt_max": str(pd.to_datetime(df_features["event_datetime"], errors="coerce").max()),
        "n_columns": int(len(df_features.columns)),
        "n_graph_features": int(len(graph_cols)),
        "elapsed_seconds": round(elapsed, 2),
    }
    safe_json_dump(run_summary, output_dir / "00_run_summary.json")

    print("\n" + "=" * 80)
    print("EXP-010G-R2 CONCLUÍDO")
    print("=" * 80)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print("\nArtefatos:")
    print(f"  {output_path}")
    print(f"  {alias_output}")
    print(f"  {output_dir / '00_run_summary.json'}")
    print(f"  {output_dir / '01_schema_coverage_before.csv'}")
    print(f"  {output_dir / '02_schema_coverage_after.csv'}")
    print(f"  {output_dir / '03_feature_diagnostics.csv'}")
    print(f"  {output_dir / '04_class_split_summary.csv'}")


if __name__ == "__main__":
    main()
