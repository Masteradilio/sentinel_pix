#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014A-0 — Prepare Expanded Scored Input

Por que este script existe:
  O EXP-014A validava uma entrada expandida já scoreada, mas essa entrada ainda
  não tinha sido criada. Este script é a etapa zero.

Objetivo:
  1. Procurar CSVs candidatos em dados/ e resultados/.
  2. Verificar qual deles tem volume expandido de fraudes.
  3. Verificar se já contém as colunas necessárias para validação congelada:
       - is_fraud
       - data/event_datetime
       - predição base ou final
       - features/bins das 10 regras EXP-013K
  4. Se o CSV já tiver predição base/final suficiente, criar:
       dados/exp014a_expanded_scored_input.csv
  5. Se não tiver, gerar relatório objetivo dizendo exatamente o que falta.

IMPORTANTE:
  Este script NÃO inventa score e NÃO retreina modelo.
  Para validação congelada, precisamos de uma predição base congelada, como:
     pred_STRICT_RECALL95_SAFE_ONLY
     exp013k_base_pred
     exp013h_frozen_pred
     exp013g_micro_pred
  ou então a predição final:
     exp013k_residual_fp_pred
     exp013l_frozen_pred

Uso recomendado:
  python scripts/exp_014a0_prepare_expanded_scored_input.py

Com input explícito:
  python scripts/exp_014a0_prepare_expanded_scored_input.py --input dados\\hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv

Forçar build se o contrato passar:
  python scripts/exp_014a0_prepare_expanded_scored_input.py --input dados\\arquivo.csv --build

Saídas:
  resultados/experimentos/EXP-014A-0/
    00_run_summary.json
    01_dataset_inventory.csv
    02_best_candidate_contract.json
    03_missing_requirements.md
    04_prepare_report.md
  dados/exp014a_expanded_scored_input.csv, se build for possível
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014A-0"
DEFAULT_TARGET = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"

BASE_PRED_COLS = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
    "pred_HIGH_RECALL_95",
]

FINAL_PRED_COLS = [
    "exp013k_residual_fp_pred",
    "exp013l_frozen_pred",
    "exp014a_frozen_pred",
]

DATE_COLS = ["event_datetime", "data_pix", "dt_pix"]

FEATURE_ALTERNATIVES = {
    "ds_tipo_chave_norm": [["ds_tipo_chave_norm"]],
    "first_receiver_flag_real": [["first_receiver_flag_real"]],
    "mbk_available_flag": [["mbk_available_flag"]],
    "value_band": [["value_band"]],
    "ratio_bin": [["ratio_bin"], ["ratio_valor_media_pagador_90d"]],
    "lgbm_bin": [["lgbm_bin"], ["lgbm_r4_score"], ["r4_score"], ["lgbm_mapped"], ["lgbm_raw"]],
    "score_bin": [["score_bin"], ["score_final"]],
    "vl_bin": [["vl_bin"], ["vl_pix"]],
    "if_bin": [["if_bin"], ["if_percentile"], ["if_percentile_x"], ["if_percentile_y"]],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_colnames(cols: list[str]) -> list[str]:
    return [str(c).strip().split(".")[-1] for c in cols]


def read_columns(path: Path) -> list[str]:
    try:
        df = pd.read_csv(path, nrows=0, low_memory=False)
        return normalize_colnames(list(df.columns))
    except Exception:
        return []


def quick_profile(path: Path, max_rows_for_count: int | None = None) -> dict[str, Any]:
    cols = read_columns(path)
    if not cols:
        return {
            "path": str(path),
            "filename": path.name,
            "readable": False,
            "n_rows": None,
            "n_frauds": None,
            "has_is_fraud": False,
            "has_date": False,
            "has_base_pred": False,
            "has_final_pred": False,
            "base_pred_cols": "",
            "final_pred_cols": "",
            "contract_score": 0,
            "error": "could_not_read_columns",
        }

    has_is_fraud = "is_fraud" in cols
    has_date = any(c in cols for c in DATE_COLS)
    base_pred = [c for c in BASE_PRED_COLS if c in cols]
    final_pred = [c for c in FINAL_PRED_COLS if c in cols]

    n_rows = None
    n_frauds = None
    error = None

    try:
        usecols = ["is_fraud"] if has_is_fraud else None
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        n_rows = int(len(df))
        if has_is_fraud:
            n_frauds = int(pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int).sum())
    except Exception as exc:
        error = str(exc)[:300]

    # Score only for inventory ranking.
    contract_score = 0
    contract_score += 10 if has_is_fraud else 0
    contract_score += 5 if has_date else 0
    contract_score += 20 if base_pred else 0
    contract_score += 15 if final_pred else 0
    contract_score += min(30, int((n_frauds or 0) / 20))

    for logical, alts in FEATURE_ALTERNATIVES.items():
        if any(all(a in cols for a in alt) for alt in alts):
            contract_score += 3

    return {
        "path": str(path),
        "filename": path.name,
        "readable": True,
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "has_is_fraud": has_is_fraud,
        "has_date": has_date,
        "has_base_pred": bool(base_pred),
        "has_final_pred": bool(final_pred),
        "base_pred_cols": "|".join(base_pred),
        "final_pred_cols": "|".join(final_pred),
        "contract_score": contract_score,
        "error": error,
    }


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def selected_rule_cols(policy: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for rule in policy.get("selected_rules", []) or []:
        params = {}
        raw = rule.get("params_json")
        if isinstance(raw, dict):
            params = raw
        else:
            try:
                params = json.loads(str(raw))
            except Exception:
                params = rule.get("params", {}) if isinstance(rule.get("params"), dict) else {}
        for c in params.get("combo_cols", []) or []:
            out.add(str(c))
    return out


def contract_for_file(path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    cols = read_columns(path)
    colset = set(cols)

    base_pred = [c for c in BASE_PRED_COLS if c in colset]
    final_pred = [c for c in FINAL_PRED_COLS if c in colset]

    missing = []
    warnings = []

    if "is_fraud" not in colset:
        missing.append("is_fraud")

    if not any(c in colset for c in DATE_COLS):
        missing.append("event_datetime_or_data_pix")

    if not base_pred and not final_pred:
        missing.append("base_or_final_prediction_column")

    needed_logical_cols = selected_rule_cols(policy)
    if not needed_logical_cols:
        warnings.append("policy_selected_rule_cols_not_detected; using default feature requirements")
        needed_logical_cols = set(FEATURE_ALTERNATIVES.keys())

    feature_status = []
    for logical in sorted(needed_logical_cols):
        alts = FEATURE_ALTERNATIVES.get(logical, [[logical]])
        ok = any(all(a in colset for a in alt) for alt in alts)
        feature_status.append({
            "logical_col": logical,
            "accepted_alternatives": alts,
            "ok": ok,
        })
        if not ok:
            missing.append(f"feature_or_bin:{logical}")

    build_possible = len(missing) == 0

    # Build mode: base is preferred; final is acceptable only for direct validation.
    if base_pred:
        build_mode = "base_prediction_available"
        selected_pred_col = base_pred[0]
    elif final_pred:
        build_mode = "final_prediction_direct_only"
        selected_pred_col = final_pred[0]
        warnings.append("only_final_prediction_found; EXP-014A must run with --allow-final-direct or use final direct mode")
    else:
        build_mode = "not_scored"
        selected_pred_col = None

    return {
        "path": str(path),
        "filename": path.name,
        "columns_count": len(cols),
        "has_is_fraud": "is_fraud" in colset,
        "has_date": any(c in colset for c in DATE_COLS),
        "base_pred_cols": base_pred,
        "final_pred_cols": final_pred,
        "selected_pred_col": selected_pred_col,
        "build_mode": build_mode,
        "feature_status": feature_status,
        "missing": missing,
        "warnings": warnings,
        "build_possible": build_possible,
    }


def find_csvs(search_roots: list[Path], max_files: int) -> list[Path]:
    out = []
    seen = set()

    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            if p.name.startswith("~"):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            if len(out) >= max_files:
                return out
    return out


def make_missing_report(contract: dict[str, Any], inventory: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014A-0 — Missing Requirements")
    lines.append("")
    lines.append("## Melhor candidato avaliado")
    lines.append(f"- Arquivo: `{contract.get('path')}`")
    lines.append(f"- Build possible: `{contract.get('build_possible')}`")
    lines.append(f"- Build mode: `{contract.get('build_mode')}`")
    lines.append("")

    if contract.get("missing"):
        lines.append("## Itens faltantes")
        for m in contract["missing"]:
            lines.append(f"- `{m}`")
    else:
        lines.append("Nenhum item obrigatório faltante.")

    if contract.get("warnings"):
        lines.append("")
        lines.append("## Avisos")
        for w in contract["warnings"]:
            lines.append(f"- `{w}`")

    lines.append("")
    lines.append("## Como resolver")
    if "base_or_final_prediction_column" in contract.get("missing", []):
        lines.append("O arquivo expandido ainda não está scoreado. Precisamos gerar uma coluna de predição congelada antes do EXP-014A.")
        lines.append("")
        lines.append("Opções válidas:")
        lines.append("- gerar `pred_STRICT_RECALL95_SAFE_ONLY`; ou")
        lines.append("- gerar `exp013k_base_pred`; ou")
        lines.append("- gerar direto `exp013k_residual_fp_pred` e depois rodar EXP-014A com `--allow-final-direct`.")
    if any(str(m).startswith("feature_or_bin:") for m in contract.get("missing", [])):
        lines.append("")
        lines.append("O arquivo não contém todas as features/bins usadas pelas 10 regras do EXP-013K. Traga as colunas brutas correspondentes ou calcule os bins antes.")

    lines.append("")
    lines.append("## Top candidatos encontrados")
    if inventory.empty:
        lines.append("Nenhum CSV encontrado.")
    else:
        show_cols = ["filename", "n_rows", "n_frauds", "has_base_pred", "has_final_pred", "contract_score", "path"]
        lines.append(inventory[show_cols].head(20).to_markdown(index=False))

    return "\n".join(lines)


def make_report(summary: dict[str, Any], contract: dict[str, Any], inventory: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014A-0 — Prepare Expanded Scored Input")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Target: `{summary['target_path']}`")
    lines.append("")
    lines.append("## Contrato do melhor arquivo")
    lines.append(f"- Arquivo: `{contract.get('path')}`")
    lines.append(f"- Build possible: `{contract.get('build_possible')}`")
    lines.append(f"- Build mode: `{contract.get('build_mode')}`")
    lines.append(f"- Selected pred col: `{contract.get('selected_pred_col')}`")
    lines.append("")
    lines.append("## Próximo comando")
    if summary["objective_status"].endswith("BUILT"):
        lines.append("Agora rode:")
        lines.append("")
        lines.append("```powershell")
        lines.append("python scripts\\exp_014a_expanded_frozen_validation.py")
        lines.append("```")
    else:
        lines.append("Ainda não rode o EXP-014A. Primeiro gere a predição base/final que está faltando, conforme `03_missing_requirements.md`.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=None, help="CSV expandido candidato. Se omitido, procura em dados/ e resultados/.")
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--build", action="store_true", help="Cria dados/exp014a_expanded_scored_input.csv se o contrato passar.")
    parser.add_argument("--max-files", type=int, default=300)
    args = parser.parse_args()

    t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(Path(args.policy_artifact))

    log("=" * 80)
    log("EXP-014A-0 — Prepare Expanded Scored Input")
    log("=" * 80)

    if args.input:
        candidates = [Path(args.input)]
    else:
        candidates = find_csvs([PROJECT_ROOT / "dados", PROJECT_ROOT / "resultados"], max_files=args.max_files)

    if not candidates:
        raise FileNotFoundError("Nenhum CSV encontrado em dados/ ou resultados/. Informe --input.")

    rows = []
    for p in candidates:
        rows.append(quick_profile(p))

    inventory = pd.DataFrame(rows).sort_values(["contract_score", "n_frauds", "n_rows"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
    inventory.to_csv(output_dir / "01_dataset_inventory.csv", index=False)

    # Best candidate: explicit input wins; otherwise ranked best.
    best_path = Path(args.input) if args.input else Path(inventory.iloc[0]["path"])
    contract = contract_for_file(best_path, policy)
    dump_json(contract, output_dir / "02_best_candidate_contract.json")

    missing_report = make_missing_report(contract, inventory)
    (output_dir / "03_missing_requirements.md").write_text(missing_report, encoding="utf-8")

    built = False
    target = Path(args.target)

    if args.build:
        if not contract["build_possible"]:
            raise RuntimeError("Contrato não passou; não posso criar input expandido. Veja 03_missing_requirements.md.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_path, target)
        built = True
        log(f"Arquivo criado: {target}")
    else:
        log("Build não solicitado. Use --build quando o contrato estiver OK.")

    objective_status = "DONE_CONTRACT_OK" if contract["build_possible"] else "DONE_CONTRACT_NOT_OK"
    if built:
        objective_status += "_BUILT"

    summary = {
        "experiment": "EXP-014A-0",
        "status": "DONE",
        "objective_status": objective_status,
        "best_candidate": str(best_path),
        "target_path": str(target),
        "build_requested": bool(args.build),
        "built": built,
        "contract": contract,
        "n_candidates_scanned": int(len(inventory)),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, contract, inventory)
    (output_dir / "04_prepare_report.md").write_text(report, encoding="utf-8")

    log("")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_dataset_inventory.csv",
        output_dir / "02_best_candidate_contract.json",
        output_dir / "03_missing_requirements.md",
        output_dir / "04_prepare_report.md",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
