#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014A-1 — Expanded Scoring Replay

Contexto:
  O EXP-014A-0 confirmou que o dataset expandido existe:
      dados/hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv
  com is_fraud, data e 1.465 fraudes, mas ainda NÃO está scoreado.

  Faltam:
    - base_or_final_prediction_column
    - if_bin / if_percentile
    - lgbm_bin / lgbm_r4_score
    - score_bin / score_final

Objetivo:
  Preparar dados/exp014a_expanded_scored_input.csv para o EXP-014A,
  mas sem inventar score.

O que este script faz:
  1. Carrega o dataset expandido.
  2. Procura artefatos de scoring já existentes no projeto.
  3. Tenta gerar lgbm_r4_score/lgbm_bin com artefato LGBM.
  4. Tenta gerar if_percentile/if_bin com artefato Isolation Forest.
  5. Verifica se existe score_final/score_bin ou artefato/pipeline capaz de gerar.
  6. Verifica se existe predição base/final congelada, ou se há pipeline capaz de gerar.
  7. Se o contrato ficar completo, grava:
       dados/exp014a_expanded_scored_input.csv
  8. Se não ficar completo, grava relatório objetivo com os artefatos/colunas que faltam.

IMPORTANTE:
  - Modo oficial default é conservador.
  - O script NÃO cria score_final a partir de LGBM por aproximação.
  - O script NÃO cria pred_STRICT_RECALL95_SAFE_ONLY por threshold inventado.
  - Existe um modo diagnóstico opcional, mas NÃO use para validar produção.

Uso:
  python scripts/exp_014a1_expanded_scoring_replay.py

Com input explícito:
  python scripts/exp_014a1_expanded_scoring_replay.py --input dados\\hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv

Se quiser passar artefatos diretamente:
  python scripts/exp_014a1_expanded_scoring_replay.py ^
    --lgbm-artifact backend\\artefatos_candidatos\\...\\model.pkl ^
    --if-artifact backend\\artefatos\\...\\isolation_forest.pkl

Apenas diagnóstico aproximado, NÃO oficial:
  python scripts/exp_014a1_expanded_scoring_replay.py --allow-diagnostic-approximations

Saídas:
  resultados/experimentos/EXP-014A-1/
    00_run_summary.json
    01_scoring_inventory.csv
    02_column_contract.json
    03_missing_artifacts.md
    04_scoring_report.md
    05_scored_preview.csv
  dados/exp014a_expanded_scored_input.csv, se contrato oficial passar
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"
DEFAULT_TARGET = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014A-1"
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

MODEL_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".sav", ".txt"}

FEATURE_FILE_NAMES = [
    "feature_columns.json",
    "features.json",
    "selected_features.json",
    "model_features.json",
    "feature_names.json",
    "cols_modelo.json",
    "columns.json",
    "feature_columns.txt",
    "features.txt",
]

LGBM_NAME_HINTS = ["lgbm", "lightgbm", "exp012", "exp013", "r4", "v3"]
IF_NAME_HINTS = ["isolation", "iforest", "isolation_forest", "if_"]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]
    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]

    if "is_fraud" in df.columns:
        df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df.reset_index(drop=True)


def pick_col(df: pd.DataFrame, names: str | list[str]) -> str | None:
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in df.columns:
            return n
    return None


def num(df: pd.DataFrame, names: str | list[str], default: float = 0.0) -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def qbin_series(s: pd.Series, name: str, bins: list[float]) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = []
    edges = [-np.inf] + bins + [np.inf]
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if np.isneginf(left):
            labels.append(f"{name}_LT_{right:g}")
        elif np.isposinf(right):
            labels.append(f"{name}_GE_{left:g}")
        else:
            labels.append(f"{name}_{left:g}_{right:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def ensure_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])

    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])

    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])

    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])

    return df


def load_json_like(path: Path) -> Any:
    try:
        if path.suffix.lower() == ".txt":
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_feature_list(obj: Any) -> list[str] | None:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [str(x) for x in obj if isinstance(x, (str, int, float))]
    if isinstance(obj, dict):
        for key in ["features", "feature_names", "feature_columns", "columns", "selected_features", "model_features"]:
            if key in obj:
                return flatten_feature_list(obj[key])
    return None


def discover_feature_names_near(model_path: Path) -> list[str] | None:
    dirs = [model_path.parent, model_path.parent.parent]
    for d in dirs:
        if not d.exists():
            continue
        for name in FEATURE_FILE_NAMES:
            p = d / name
            if p.exists():
                fl = flatten_feature_list(load_json_like(p))
                if fl:
                    return fl
    return None


def discover_artifacts(root: Path) -> pd.DataFrame:
    rows = []
    search_roots = [
        root / "backend" / "artefatos",
        root / "backend" / "artefatos_candidatos",
        root / "resultados" / "experimentos",
        root / "modelos",
        root / "artifacts",
    ]

    seen = set()
    for sr in search_roots:
        if not sr.exists():
            continue
        for p in sr.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in MODEL_EXTENSIONS:
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)

            low = str(p).lower()
            kind_score = 0
            lgbm_hint = any(h in low for h in LGBM_NAME_HINTS)
            if_hint = any(h in low for h in IF_NAME_HINTS)
            if lgbm_hint:
                kind_score += 10
            if if_hint:
                kind_score += 10

            rows.append({
                "path": str(p),
                "filename": p.name,
                "suffix": p.suffix,
                "size_bytes": p.stat().st_size,
                "lgbm_hint": lgbm_hint,
                "if_hint": if_hint,
                "score": kind_score + min(10, p.stat().st_size // 100000),
            })

    if not rows:
        return pd.DataFrame(columns=["path", "filename", "suffix", "size_bytes", "lgbm_hint", "if_hint", "score"])

    return pd.DataFrame(rows).sort_values(["score", "size_bytes"], ascending=[False, False]).reset_index(drop=True)


def load_model(path: Path) -> tuple[Any, str]:
    suffix = path.suffix.lower()

    if suffix in {".pkl", ".pickle", ".sav"}:
        with path.open("rb") as f:
            return pickle.load(f), "pickle"

    if suffix == ".joblib":
        try:
            import joblib
        except Exception as exc:
            raise RuntimeError(f"joblib não disponível para carregar {path}: {exc}")
        return joblib.load(path), "joblib"

    if suffix == ".txt":
        try:
            import lightgbm as lgb
        except Exception as exc:
            raise RuntimeError(f"lightgbm não disponível para carregar Booster txt {path}: {exc}")
        return lgb.Booster(model_file=str(path)), "lightgbm_booster_txt"

    raise RuntimeError(f"Extensão de modelo não suportada: {path}")


def infer_feature_names(model: Any, model_path: Path) -> list[str] | None:
    for attr in ["feature_names_in_", "feature_name_", "feature_names", "feature_name"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    val = None
            if val is not None:
                try:
                    out = [str(x) for x in list(val)]
                    if out:
                        return out
                except Exception:
                    pass

    # LightGBM Booster
    if hasattr(model, "feature_name"):
        try:
            out = [str(x) for x in model.feature_name()]
            if out:
                return out
        except Exception:
            pass

    return discover_feature_names_near(model_path)


def make_X(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing = [f for f in features if f not in df.columns]
    used = [f for f in features if f in df.columns]

    X = pd.DataFrame(index=df.index)

    for f in features:
        if f in df.columns:
            s = df[f]
            if pd.api.types.is_numeric_dtype(s):
                X[f] = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                # Deterministic diagnostic encoding only. Exact only if training pipeline has encoders inside model.
                X[f] = pd.factorize(s.astype("string").fillna("<MISSING>"))[0].astype(float)
        else:
            X[f] = 0.0

    return X, used, missing


def predict_score(model: Any, df: pd.DataFrame, model_path: Path, score_kind: str) -> dict[str, Any]:
    features = infer_feature_names(model, model_path)

    if features is None:
        # Last-resort: if model is a pipeline, it may accept the raw df directly.
        features = list(df.columns)

    X, used, missing = make_X(df, features)

    exact_features = len(missing) == 0

    try:
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(X)
            if isinstance(pred, list):
                pred = np.asarray(pred)
            if len(np.asarray(pred).shape) == 2 and np.asarray(pred).shape[1] > 1:
                score = np.asarray(pred)[:, 1]
            else:
                score = np.asarray(pred).reshape(-1)
            method = "predict_proba"
        elif hasattr(model, "predict"):
            pred = model.predict(X)
            score = np.asarray(pred).reshape(-1)
            method = "predict"
        elif hasattr(model, "decision_function"):
            pred = model.decision_function(X)
            score = np.asarray(pred).reshape(-1)
            method = "decision_function"
        elif hasattr(model, "score_samples"):
            pred = model.score_samples(X)
            score = np.asarray(pred).reshape(-1)
            method = "score_samples"
        else:
            raise RuntimeError("Modelo não tem predict_proba/predict/decision_function/score_samples.")
    except Exception as exc:
        # Try raw dataframe for sklearn pipeline with own preprocessing.
        try:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(df)
                score = np.asarray(pred)[:, 1] if len(np.asarray(pred).shape) == 2 and np.asarray(pred).shape[1] > 1 else np.asarray(pred).reshape(-1)
                method = "predict_proba_raw_df"
                exact_features = True
                missing = []
            elif hasattr(model, "predict"):
                pred = model.predict(df)
                score = np.asarray(pred).reshape(-1)
                method = "predict_raw_df"
                exact_features = True
                missing = []
            else:
                raise exc
        except Exception as exc2:
            raise RuntimeError(f"Falha ao scorear com {model_path}: {exc2}") from exc

    return {
        "score": score,
        "method": method,
        "features_count": len(features),
        "used_features_count": len(used),
        "missing_features_count": len(missing),
        "missing_features": missing[:200],
        "exact_features": exact_features,
    }


def choose_artifact(inventory: pd.DataFrame, kind: str, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    if inventory.empty:
        return None

    if kind == "lgbm":
        cand = inventory[inventory["lgbm_hint"] == True]
    elif kind == "if":
        cand = inventory[inventory["if_hint"] == True]
    else:
        cand = inventory

    if cand.empty:
        return None

    return Path(cand.iloc[0]["path"])


def percentile_from_scores(scores: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    return s.rank(pct=True).to_numpy(dtype=float)


def contract_status(df: pd.DataFrame) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("event_datetime_or_data_pix")
    if not any(c in df.columns for c in BASE_PRED_COLS + FINAL_PRED_COLS):
        missing.append("base_or_final_prediction_column")

    for logical, alts in {
        "if_bin": [["if_bin"], ["if_percentile"], ["if_percentile_x"], ["if_percentile_y"]],
        "lgbm_bin": [["lgbm_bin"], ["lgbm_r4_score"], ["r4_score"], ["lgbm_mapped"], ["lgbm_raw"]],
        "score_bin": [["score_bin"], ["score_final"]],
        "ratio_bin": [["ratio_bin"], ["ratio_valor_media_pagador_90d"]],
        "vl_bin": [["vl_bin"], ["vl_pix"]],
        "value_band": [["value_band"]],
        "ds_tipo_chave_norm": [["ds_tipo_chave_norm"]],
        "first_receiver_flag_real": [["first_receiver_flag_real"]],
        "mbk_available_flag": [["mbk_available_flag"]],
    }.items():
        if not any(all(c in df.columns for c in alt) for alt in alts):
            missing.append(f"feature_or_bin:{logical}")

    return {
        "contract_ok": len(missing) == 0,
        "missing": missing,
        "base_pred_cols_present": [c for c in BASE_PRED_COLS if c in df.columns],
        "final_pred_cols_present": [c for c in FINAL_PRED_COLS if c in df.columns],
    }


def make_missing_report(summary: dict[str, Any], scoring_steps: list[dict[str, Any]], contract: dict[str, Any]) -> str:
    lines = []
    lines.append("# EXP-014A-1 — Missing Artifacts / Columns")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Contract OK: `{contract['contract_ok']}`")
    lines.append("")
    lines.append("## Itens ainda faltantes")
    if contract["missing"]:
        for m in contract["missing"]:
            lines.append(f"- `{m}`")
    else:
        lines.append("Nenhum item faltante.")
    lines.append("")
    lines.append("## Passos de scoring tentados")
    if scoring_steps:
        for s in scoring_steps:
            lines.append(f"- `{s.get('step')}`: status=`{s.get('status')}`, artifact=`{s.get('artifact')}`, note=`{s.get('note')}`")
    else:
        lines.append("Nenhum scoring foi tentado.")
    lines.append("")
    lines.append("## Como resolver")
    lines.append("Para o EXP-014A oficial, precisamos gerar uma predição congelada real. Caminhos possíveis:")
    lines.append("")
    lines.append("1. Localizar o artefato/pipeline que gera `pred_STRICT_RECALL95_SAFE_ONLY` ou `exp013k_residual_fp_pred`.")
    lines.append("2. Localizar os artefatos que geram `lgbm_r4_score`, `if_percentile` e `score_final`.")
    lines.append("3. Reexecutar o pipeline E2E/shadow no dataset expandido para produzir essas colunas.")
    lines.append("")
    lines.append("Não use `--allow-diagnostic-approximations` para consolidar métricas oficiais; ele serve apenas para diagnóstico.")
    return "\n".join(lines)


def make_report(summary: dict[str, Any], contract: dict[str, Any], scoring_steps: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# EXP-014A-1 — Expanded Scoring Replay")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Input: `{summary['input_path']}`")
    lines.append(f"- Target: `{summary['target_path']}`")
    lines.append(f"- Built: `{summary['built']}`")
    lines.append("")
    lines.append("## Contrato final")
    lines.append(f"- Contract OK: `{contract['contract_ok']}`")
    lines.append(f"- Missing: `{contract['missing']}`")
    lines.append(f"- Base pred cols: `{contract['base_pred_cols_present']}`")
    lines.append(f"- Final pred cols: `{contract['final_pred_cols_present']}`")
    lines.append("")
    lines.append("## Passos de scoring")
    if scoring_steps:
        for s in scoring_steps:
            lines.append(f"- `{s.get('step')}`: {s.get('status')} — {s.get('note')}")
    else:
        lines.append("Nenhum.")
    lines.append("")
    lines.append("## Próximo passo")
    if summary["built"]:
        lines.append("Rodar:")
        lines.append("```powershell")
        lines.append("python scripts\\exp_014a_expanded_frozen_validation.py")
        lines.append("```")
    else:
        lines.append("Resolver os itens de `03_missing_artifacts.md` antes do EXP-014A.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lgbm-artifact", default=None)
    parser.add_argument("--if-artifact", default=None)
    parser.add_argument("--allow-diagnostic-approximations", action="store_true")
    parser.add_argument("--score-final-from-lgbm-diagnostic", action="store_true")
    parser.add_argument("--pred-from-score-final-threshold", type=float, default=None)
    parser.add_argument("--force-write-diagnostic", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    target_path = Path(args.target)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014A-1 — Expanded Scoring Replay")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Target: {target_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    n_rows = len(df)
    n_frauds = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None

    inventory = discover_artifacts(PROJECT_ROOT)
    inventory.to_csv(output_dir / "01_scoring_inventory.csv", index=False)

    scoring_steps = []

    # 1. Existing bins from raw columns.
    before_cols = set(df.columns)
    df = ensure_bins(df)
    added = sorted(set(df.columns) - before_cols)
    scoring_steps.append({
        "step": "derive_existing_bins_from_raw_columns",
        "status": "OK",
        "artifact": None,
        "note": f"added={added}",
    })

    # 2. LGBM score if missing.
    if "lgbm_bin" not in df.columns and not pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        art = choose_artifact(inventory, "lgbm", args.lgbm_artifact)
        if art is not None:
            try:
                model, load_method = load_model(art)
                pred = predict_score(model, df, art, "lgbm")
                df["lgbm_r4_score"] = pred["score"]
                df = ensure_bins(df)
                scoring_steps.append({
                    "step": "generate_lgbm_r4_score",
                    "status": "OK_EXACT_FEATURES" if pred["exact_features"] else "OK_WITH_MISSING_FEATURES_DIAGNOSTIC",
                    "artifact": str(art),
                    "note": {
                        "load_method": load_method,
                        "predict_method": pred["method"],
                        "features_count": pred["features_count"],
                        "missing_features_count": pred["missing_features_count"],
                        "missing_features": pred["missing_features"][:30],
                    },
                })
            except Exception as exc:
                scoring_steps.append({
                    "step": "generate_lgbm_r4_score",
                    "status": "FAILED",
                    "artifact": str(art),
                    "note": str(exc)[:1000],
                })
        else:
            scoring_steps.append({
                "step": "generate_lgbm_r4_score",
                "status": "SKIPPED_NO_ARTIFACT",
                "artifact": None,
                "note": "Não encontrei artefato LGBM por auto-discovery. Use --lgbm-artifact.",
            })

    # 3. IF percentile if missing.
    if "if_bin" not in df.columns and not pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        art = choose_artifact(inventory, "if", args.if_artifact)
        if art is not None:
            try:
                model, load_method = load_model(art)
                pred = predict_score(model, df, art, "if")
                raw = np.asarray(pred["score"]).reshape(-1)
                df["if_raw_score"] = raw
                df["if_percentile"] = percentile_from_scores(raw)
                df = ensure_bins(df)
                scoring_steps.append({
                    "step": "generate_if_percentile",
                    "status": "OK_EXACT_FEATURES" if pred["exact_features"] else "OK_WITH_MISSING_FEATURES_DIAGNOSTIC",
                    "artifact": str(art),
                    "note": {
                        "load_method": load_method,
                        "predict_method": pred["method"],
                        "features_count": pred["features_count"],
                        "missing_features_count": pred["missing_features_count"],
                        "missing_features": pred["missing_features"][:30],
                    },
                })
            except Exception as exc:
                scoring_steps.append({
                    "step": "generate_if_percentile",
                    "status": "FAILED",
                    "artifact": str(art),
                    "note": str(exc)[:1000],
                })
        else:
            scoring_steps.append({
                "step": "generate_if_percentile",
                "status": "SKIPPED_NO_ARTIFACT",
                "artifact": None,
                "note": "Não encontrei artefato Isolation Forest por auto-discovery. Use --if-artifact.",
            })

    # 4. score_final.
    if "score_bin" not in df.columns and "score_final" not in df.columns:
        if args.allow_diagnostic_approximations and args.score_final_from_lgbm_diagnostic and pick_col(df, ["lgbm_r4_score"]):
            df["score_final"] = (num(df, "lgbm_r4_score", 0.0) * 100.0).clip(0, 100)
            df = ensure_bins(df)
            scoring_steps.append({
                "step": "generate_score_final",
                "status": "DIAGNOSTIC_APPROXIMATION",
                "artifact": None,
                "note": "score_final = lgbm_r4_score * 100. NÃO usar para métrica oficial.",
            })
        else:
            scoring_steps.append({
                "step": "generate_score_final",
                "status": "MISSING",
                "artifact": None,
                "note": "Não há score_final/score_bin nem pipeline de DecisionEngine informado. Não vou inventar score.",
            })

    # 5. base/final prediction.
    if not any(c in df.columns for c in BASE_PRED_COLS + FINAL_PRED_COLS):
        if args.allow_diagnostic_approximations and args.pred_from_score_final_threshold is not None and "score_final" in df.columns:
            df["pred_STRICT_RECALL95_SAFE_ONLY"] = (num(df, "score_final", 0.0) >= float(args.pred_from_score_final_threshold)).astype(int)
            scoring_steps.append({
                "step": "generate_base_prediction",
                "status": "DIAGNOSTIC_APPROXIMATION",
                "artifact": None,
                "note": f"pred_STRICT_RECALL95_SAFE_ONLY = score_final >= {args.pred_from_score_final_threshold}. NÃO usar para métrica oficial.",
            })
        else:
            scoring_steps.append({
                "step": "generate_base_prediction",
                "status": "MISSING",
                "artifact": None,
                "note": "Não há coluna de predição base/final e não foi informado pipeline/threshold oficial.",
            })

    # Recompute final bins after all attempts.
    df = ensure_bins(df)
    contract = contract_status(df)
    dump_json(contract, output_dir / "02_column_contract.json")

    official_build_ok = contract["contract_ok"]

    # Require no diagnostic approximations for official build.
    has_diag = any(str(s.get("status", "")).startswith("DIAGNOSTIC") or "DIAGNOSTIC" in str(s.get("status", "")) for s in scoring_steps)
    if has_diag and not args.force_write_diagnostic:
        official_build_ok = False

    built = False
    if official_build_ok or (args.force_write_diagnostic and args.allow_diagnostic_approximations):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)
        built = True
        log(f"Arquivo criado: {target_path}")

    # Always write preview.
    preview_cols = []
    for c in ["transaction_id", "is_fraud", "event_datetime", "data_pix"] + BASE_PRED_COLS + FINAL_PRED_COLS + [
        "lgbm_r4_score", "lgbm_bin", "if_percentile", "if_bin", "score_final", "score_bin",
        "value_band", "ds_tipo_chave_norm", "first_receiver_flag_real", "mbk_available_flag", "vl_pix", "vl_bin", "ratio_valor_media_pagador_90d", "ratio_bin",
    ]:
        if c in df.columns and c not in preview_cols:
            preview_cols.append(c)
    df[preview_cols].head(1000).to_csv(output_dir / "05_scored_preview.csv", index=False)

    objective_status = "DONE_CONTRACT_OK" if contract["contract_ok"] else "DONE_CONTRACT_NOT_OK"
    if has_diag:
        objective_status += "_HAS_DIAGNOSTIC_APPROX"
    if built:
        objective_status += "_BUILT"

    summary = {
        "experiment": "EXP-014A-1",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "target_path": str(target_path),
        "n_rows": int(n_rows),
        "n_frauds": n_frauds,
        "built": built,
        "has_diagnostic_approximations": has_diag,
        "contract": contract,
        "scoring_steps": scoring_steps,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    missing_report = make_missing_report(summary, scoring_steps, contract)
    (output_dir / "03_missing_artifacts.md").write_text(missing_report, encoding="utf-8")

    report = make_report(summary, contract, scoring_steps)
    (output_dir / "04_scoring_report.md").write_text(report, encoding="utf-8")

    log("")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_scoring_inventory.csv",
        output_dir / "02_column_contract.json",
        output_dir / "03_missing_artifacts.md",
        output_dir / "04_scoring_report.md",
        output_dir / "05_scored_preview.csv",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
