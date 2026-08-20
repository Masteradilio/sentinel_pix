#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_v3_csv_splits.py

Une os 3 CSVs exportados do HUE por temporal_split e valida o arquivo final
do EXP-012C.

Motivo:
  O PowerShell Import-Csv usa vírgula como delimitador por padrão. Se o HUE
  exportou com ; ou tab, ele pode gerar um CSV final corrompido, sem colunas
  como event_datetime/dt_pix. Isso causa KeyError: 'event_datetime' no treino.

Uso, na raiz do projeto:
  python scripts\merge_v3_csv_splits.py

Entradas esperadas:
  dados\hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv
  dados\hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv
  dados\hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv

Saída:
  dados\hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DADOS = PROJECT_ROOT / "dados"

FILES = [
    DADOS / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv",
    DADOS / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv",
    DADOS / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv",
]

OUTPUT = DADOS / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"

REQUIRED_ANY_DATE = ["event_datetime", "dt_pix"]
REQUIRED_COLUMNS = ["transaction_id", "temporal_split", "is_fraud"]


def sniff_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    # Remove eventual linha "sep=;" do Excel/HUE.
    first_line = sample.splitlines()[0].strip().lower() if sample.splitlines() else ""
    if first_line.startswith("sep="):
        return first_line.split("=", 1)[1][:1] or ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # Heurística simples.
        header = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {sep: header.count(sep) for sep in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get)


def read_hue_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    sep = sniff_separator(path)
    print(f"[READ] {path.name} | sep={repr(sep)}")

    # Detecta e pula linha sep=; se existir.
    first_line = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0].strip().lower()
    skiprows = 1 if first_line.startswith("sep=") else 0

    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        dtype=str,
        encoding="utf-8-sig",
        skiprows=skiprows,
        on_bad_lines="warn",
    )

    df.columns = [str(c).replace("\ufeff", "").strip().strip('"').strip("'") for c in df.columns]

    # Caso o arquivo já tenha sido corrompido por Import-Csv/Export-Csv e tenha
    # virado uma única coluna com separadores dentro do nome.
    if len(df.columns) == 1:
        only_col = df.columns[0]
        if ";" in only_col or "\t" in only_col or "|" in only_col:
            raise RuntimeError(
                f"{path.name} parece ter sido reexportado como uma única coluna ({only_col[:120]}...). "
                "Use os CSVs originais baixados do HUE, não o CSV combinado pelo PowerShell."
            )

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Alguns exports podem vir com prefixo de tabela; remove somente se ocorrer.
    df.columns = [c.split(".")[-1].strip() for c in df.columns]

    # Aliases defensivos.
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]
    if "dt_pix" not in df.columns and "event_datetime" in df.columns:
        df["dt_pix"] = df["event_datetime"]
    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]

    return df


def validate_part(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: colunas obrigatórias ausentes: {missing}. Colunas encontradas: {list(df.columns)[:30]}")

    if not any(c in df.columns for c in REQUIRED_ANY_DATE):
        raise RuntimeError(
            f"{name}: nenhuma coluna de data encontrada entre {REQUIRED_ANY_DATE}. "
            f"Colunas encontradas: {list(df.columns)[:40]}"
        )


def main() -> None:
    parts = []
    for path in FILES:
        df = normalize_columns(read_hue_csv(path))
        validate_part(df, path.name)
        parts.append(df)
        print(
            f"[OK] {path.name}: rows={len(df):,} | cols={len(df.columns)} | "
            f"splits={df['temporal_split'].dropna().unique()[:5]}"
        )

    # Garante mesma lista de colunas, preservando a ordem do TRAIN e adicionando extras no fim.
    all_cols = []
    for df in parts:
        for c in df.columns:
            if c not in all_cols:
                all_cols.append(c)

    parts = [df.reindex(columns=all_cols) for df in parts]
    out = pd.concat(parts, ignore_index=True)

    out = normalize_columns(out)
    out["transaction_id"] = out["transaction_id"].astype(str).str.strip()

    n_rows = len(out)
    n_unique = out["transaction_id"].nunique(dropna=True)
    n_dup_rows = n_rows - n_unique
    n_fraud = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(0).astype(int).sum()

    print("=" * 80)
    print("[VALIDAÇÃO FINAL]")
    print(f"rows:                 {n_rows:,}")
    print(f"transaction_id únicos: {n_unique:,}")
    print(f"duplicados:            {n_dup_rows:,}")
    print(f"fraudes:               {int(n_fraud):,}")
    print("splits:")
    print(out["temporal_split"].value_counts(dropna=False).to_string())
    print("=" * 80)

    if n_rows != 113844:
        print(f"[WARN] Esperado 113.844 linhas pelo Hive, mas o CSV final tem {n_rows:,}.")
    if n_dup_rows != 0:
        dup_path = DADOS / "hmo_ml_tb_pix_dataset_v3_features_180d_v1_DUPLICADOS.csv"
        out[out.duplicated("transaction_id", keep=False)].to_csv(dup_path, index=False, encoding="utf-8")
        raise RuntimeError(f"Foram encontrados {n_dup_rows:,} duplicados. Arquivo de auditoria: {dup_path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"[DONE] CSV final salvo em: {OUTPUT}")


if __name__ == "__main__":
    main()
