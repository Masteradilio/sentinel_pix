#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audita divergencia entre homologacao E2E R5B17 e frozen R4G/R5B16."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R5B17-PIPELINE-HOMOLOGATION"
E2E_FRAUDS = OUT_DIR / "frauds_only" / "01_pipeline_predictions.csv"
FROZEN = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014B-R4G-FAST-FROZEN" / "06_predictions_frozen.csv"
SUMMARY = OUT_DIR / "05_frozen_alignment_audit.json"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    e2e = pd.read_csv(E2E_FRAUDS, low_memory=False)
    frozen = pd.read_csv(FROZEN, low_memory=False)
    if "transaction_id" not in e2e.columns or "transaction_id" not in frozen.columns:
        raise KeyError("transaction_id ausente em uma das entradas.")

    e2e = e2e.rename(columns={"decisao": "decisao_e2e"})
    joined = e2e.merge(
        frozen[["transaction_id", "r4g_fast_frozen_decisao_recommended"]],
        on="transaction_id",
        how="left",
    )
    joined["decisao_e2e"] = joined["decisao_e2e"].fillna("").astype(str).str.upper().str.strip()
    joined["r4g_fast_frozen_decisao_recommended"] = (
        joined["r4g_fast_frozen_decisao_recommended"].fillna("").astype(str).str.upper().str.strip()
    )

    approve_e2e = joined["decisao_e2e"].eq("APROVAR")
    frozen_intervenes = joined["r4g_fast_frozen_decisao_recommended"].isin(["CONFIRMAR", "BLOQUEAR"])
    crosstab = (
        joined.groupby(["decisao_e2e", "r4g_fast_frozen_decisao_recommended"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["decisao_e2e", "r4g_fast_frozen_decisao_recommended"])
    )
    crosstab.to_csv(OUT_DIR / "06_fraud_only_e2e_vs_frozen_crosstab.csv", index=False)

    summary = {
        "status": "FAIL_R5B17_E2E_FROZEN_ALIGNMENT_GAP",
        "e2e_fraud_rows": int(len(joined)),
        "e2e_approve_frauds": int(approve_e2e.sum()),
        "e2e_approve_frauds_that_frozen_would_intervene": int((approve_e2e & frozen_intervenes).sum()),
        "e2e_approve_frauds_that_frozen_would_block": int(
            (approve_e2e & joined["r4g_fast_frozen_decisao_recommended"].eq("BLOQUEAR")).sum()
        ),
        "e2e_approve_frauds_that_frozen_would_confirm": int(
            (approve_e2e & joined["r4g_fast_frozen_decisao_recommended"].eq("CONFIRMAR")).sum()
        ),
        "crosstab_csv": str((OUT_DIR / "06_fraud_only_e2e_vs_frozen_crosstab.csv").relative_to(PROJECT_ROOT)),
    }
    write_json(SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
