#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014A-3 - Expanded Policy Assembly

Parte de dados/exp014a_lgbm_scored_partial.csv, criado pelo EXP-014A-2,
e tenta montar dados/exp014a_expanded_scored_input.csv para o EXP-014A.

Modo oficial:
  - nao inventa score_final;
  - nao inventa predicao base;
  - tenta gerar IF com model/preprocessor reais;
  - aplica as regras congeladas do EXP-013K apenas se houver base/final pred.

Modo diagnostico, NAO oficial:
  python scripts/exp_014a3_expanded_policy_assembly.py --allow-diagnostic-approximations --score-final-from-lgbm --base-pred-from-lgbm-threshold 0.001 --force-write-diagnostic
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
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

# Compatibilidade para artefatos antigos salvos como __main__.PixPreprocessor.
class PixPreprocessor:
    def __init__(self, *args, **kwargs):
        pass
    def transform(self, X):
        for attr in ["preprocessor", "pipeline", "transformer", "column_transformer", "scaler"]:
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "transform"):
                return obj.transform(X)
        feature_cols = None
        for attr in ["feature_columns", "features", "selected_features", "feature_names", "columns"]:
            val = getattr(self, attr, None)
            if val is not None:
                try:
                    feature_cols = [str(v) for v in list(val)]
                    break
                except Exception:
                    pass
        if feature_cols:
            out = pd.DataFrame(index=X.index)
            for c in feature_cols:
                out[c] = pd.to_numeric(X[c], errors="coerce") if c in X.columns else 0.0
            return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        return X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

sys.modules["__main__"].PixPreprocessor = PixPreprocessor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()
DEFAULT_INPUT = PROJECT_ROOT / "dados" / "exp014a_lgbm_scored_partial.csv"
DEFAULT_TARGET = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014A-3"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013K" / "12_policy_artifact.json"

BASE_PRED_COLS = [
    "pred_STRICT_RECALL95_SAFE_ONLY", "exp013k_base_pred", "exp013h_frozen_pred",
    "exp013g_micro_pred", "pred_HIGH_RECALL_95", "exp014a_lgbm_base_pred",
]
FINAL_PRED_COLS = ["exp013k_residual_fp_pred", "exp013l_frozen_pred", "exp014a_frozen_pred"]
MODEL_EXTS = {".joblib", ".pkl", ".pickle", ".sav"}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path) -> Any:
    if path.suffix.lower() == ".joblib":
        import joblib
        return joblib.load(path)
    with path.open("rb") as f:
        return pickle.load(f)


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
    edges = [-np.inf] + bins + [np.inf]
    labels = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if np.isneginf(left):
            labels.append(f"{name}_LT_{right:g}")
        elif np.isposinf(right):
            labels.append(f"{name}_GE_{left:g}")
        else:
            labels.append(f"{name}_{left:g}_{right:g}")
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True).astype("string").fillna(f"{name}_MISSING").astype(str)


def ensure_bins_and_guards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d"), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix"), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])
    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])
    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]), "if", [0.32, 0.5, 0.7, 0.85, 0.95])
    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final"), "score", [0.5, 1, 2, 3, 5, 10])

    se_score = num(df, ["se_score_x", "se_score_y", "se_score"])
    se_count = num(df, ["se_patterns_count", "se_pattern_count"])
    beh_score = num(df, ["beh_score", "behavioral_score"])
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"])
    runtime = num(df, "runtime_flagged")
    strong = (se_score >= 40) | (se_count >= 2) | (beh_score >= 25) | (beh_count >= 2) | (runtime >= 1)
    df["module_quiet"] = np.where(strong, "module_strong", "module_quiet")
    return df


def discover_if_artifacts(root: Path) -> tuple[Path | None, Path | None, list[dict[str, Any]]]:
    rows, models, preps = [], [], []
    for base in [root / "backend" / "artefatos", root / "backend" / "artefatos_candidatos", root / "resultados" / "experimentos"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in MODEL_EXTS:
                continue
            low = str(p).lower()
            rec = {"path": str(p), "filename": p.name, "is_if_model": ("isolation" in low or "iforest" in low), "is_preprocessor": ("preprocess" in low or "preprocessing" in low or "scaler" in low), "size_bytes": p.stat().st_size}
            rows.append(rec)
            if rec["is_if_model"]:
                models.append(p)
            if rec["is_preprocessor"]:
                preps.append(p)
    model = models[0] if models else None
    prep = None
    if model is not None:
        same = [p for p in preps if p.parent == model.parent]
        if same:
            prep = same[0]
    if prep is None and preps:
        prep = preps[0]
    return model, prep, rows


def prep_columns(prep: Any) -> list[str] | None:
    if hasattr(prep, "feature_names_in_"):
        try:
            return [str(x) for x in list(prep.feature_names_in_)]
        except Exception:
            pass
    for attr in ["feature_columns", "features", "selected_features", "feature_names", "columns"]:
        val = getattr(prep, attr, None)
        if val is not None:
            try:
                return [str(x) for x in list(val)]
            except Exception:
                pass
    if hasattr(prep, "transformers_"):
        cols = []
        try:
            for _, _, c in prep.transformers_:
                if isinstance(c, str):
                    cols.append(c)
                elif isinstance(c, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend([str(x) for x in list(c)])
            return sorted(set(cols)) if cols else None
        except Exception:
            pass
    return None


def transform_with_preprocessor(prep: Any, df: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    required = prep_columns(prep)
    missing = []
    if required:
        x = df.copy()
        missing = [c for c in required if c not in x.columns]
        for c in missing:
            x[c] = np.nan
        x = x[required]
    else:
        x = df.copy()
    try:
        return prep.transform(x), {"required_columns_count": None if required is None else len(required), "missing_columns_count": len(missing), "missing_columns": missing[:100], "transform_input": "required_columns" if required else "full_df"}
    except Exception as exc:
        if required is None:
            nums = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return nums.to_numpy(dtype=float), {"required_columns_count": None, "missing_columns_count": 0, "missing_columns": [], "transform_input": "numeric_only_fallback", "first_error": str(exc)[:500]}
        raise


def score_if(df: pd.DataFrame, model_path: Path, prep_path: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
    model = load_model(model_path)
    if prep_path is not None and prep_path.exists():
        prep = load_model(prep_path)
        x, info = transform_with_preprocessor(prep, df)
    else:
        x = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        info = {"transform_input": "numeric_only_no_preprocessor", "missing_columns_count": None}
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x)).reshape(-1)
        method = "decision_function"
    elif hasattr(model, "score_samples"):
        raw = np.asarray(model.score_samples(x)).reshape(-1)
        method = "score_samples"
    elif hasattr(model, "predict"):
        raw = np.asarray(model.predict(x)).reshape(-1)
        method = "predict"
    else:
        raise RuntimeError("IF model has no usable scoring method")
    pct = pd.Series(raw).rank(pct=True).to_numpy(dtype=float)
    info["method"] = method
    return pct, info


def parse_params(rule: dict[str, Any]) -> dict[str, Any]:
    raw = rule.get("params_json")
    if isinstance(raw, dict):
        return raw
    if raw:
        try:
            return json.loads(str(raw))
        except Exception:
            pass
    return rule.get("params", {}) if isinstance(rule.get("params"), dict) else {}


def selected_rules(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rules = policy.get("selected_rules", [])
    return rules if isinstance(rules, list) else []


def choose_base_col(df: pd.DataFrame) -> str | None:
    for c in BASE_PRED_COLS:
        if c in df.columns:
            return c
    return None


def choose_final_col(df: pd.DataFrame) -> str | None:
    for c in FINAL_PRED_COLS:
        if c in df.columns:
            return c
    return None


def apply_rule_mask(df: pd.DataFrame, rule: dict[str, Any], pred: np.ndarray) -> np.ndarray:
    params = parse_params(rule)
    cols = params.get("combo_cols", [])
    vals = params.get("combo_values", [])
    if not cols:
        cols, vals = [], []
        for part in str(rule.get("description", "")).split(" AND "):
            if "=" in part:
                c, v = part.split("=", 1)
                cols.append(c.strip())
                vals.append(v.strip())
    if not cols:
        raise RuntimeError(f"Nao consegui parsear regra: {rule}")
    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(cols, vals):
        if c not in df.columns:
            raise RuntimeError(f"Coluna da regra ausente: {c}")
        mask = mask & (df[c].astype(str) == str(v))
    return mask & (pred == 1)


def apply_policy(df: pd.DataFrame, policy: dict[str, Any], base_col: str) -> tuple[np.ndarray, pd.DataFrame]:
    pred = pd.to_numeric(df[base_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y = df["is_fraud"].to_numpy(dtype=int) if "is_fraud" in df.columns else np.zeros(len(df), dtype=int)
    rows = []
    for idx, rule in enumerate(selected_rules(policy)):
        mask = apply_rule_mask(df, rule, pred)
        rows.append({"rule_index": idx, "description": rule.get("description"), "tp_loss": int(((y == 1) & mask).sum()), "fp_removed": int(((y == 0) & mask).sum()), "n_removed": int(mask.sum())})
        pred[mask] = 0
    return pred, pd.DataFrame(rows)


def contract_status(df: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    missing = []
    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("event_datetime_or_data_pix")
    if choose_base_col(df) is None and choose_final_col(df) is None:
        missing.append("base_or_final_prediction_column")
    logical = set()
    for r in selected_rules(policy):
        logical.update(parse_params(r).get("combo_cols", []) or [])
    if not logical:
        logical = {"ds_tipo_chave_norm", "first_receiver_flag_real", "if_bin", "lgbm_bin", "mbk_available_flag", "ratio_bin", "score_bin", "value_band", "vl_bin"}
    for c in sorted(logical):
        if c not in df.columns:
            missing.append(f"feature_or_bin:{c}")
    return {"contract_ok": len(missing) == 0, "missing": missing, "base_pred_cols_present": [c for c in BASE_PRED_COLS if c in df.columns], "final_pred_cols_present": [c for c in FINAL_PRED_COLS if c in df.columns], "selected_base_col": choose_base_col(df), "selected_final_col": choose_final_col(df)}


def preview_cols(df: pd.DataFrame) -> list[str]:
    preferred = ["transaction_id", "is_fraud", "event_datetime", "data_pix", "lgbm_r4_score", "lgbm_bin", "if_percentile", "if_bin", "score_final", "score_bin", "exp014a_lgbm_base_pred", "exp014a_frozen_pred", "value_band", "ds_tipo_chave_norm", "first_receiver_flag_real", "mbk_available_flag", "module_quiet", "vl_pix", "vl_bin", "ratio_valor_media_pagador_90d", "ratio_bin"]
    return [c for c in preferred if c in df.columns]


def write_missing_report(path: Path, summary: dict[str, Any], contract: dict[str, Any], steps: list[dict[str, Any]]) -> None:
    lines = ["# EXP-014A-3 - Missing Requirements", "", f"- Status: `{summary['objective_status']}`", f"- Built: `{summary['built']}`", "", "## Faltantes"]
    if contract["missing"]:
        lines.extend([f"- `{m}`" for m in contract["missing"]])
    else:
        lines.append("Nenhum item faltante.")
    lines += ["", "## Etapas"]
    for s in steps:
        lines.append(f"- `{s.get('step')}`: status=`{s.get('status')}`, note=`{s.get('note')}`")
    lines += ["", "## Como resolver"]
    if "feature_or_bin:score_bin" in contract["missing"]:
        lines.append("- Falta score_final/score_bin. Precisa vir da logica oficial do DecisionEngine ou de um artefato que calcule score_final.")
    if "base_or_final_prediction_column" in contract["missing"]:
        lines.append("- Falta predicao base/final oficial. Precisa vir do replay do DecisionEngine ou de thresholds/politicas congeladas oficiais.")
    if "feature_or_bin:if_bin" in contract["missing"]:
        lines.append("- Falta if_percentile/if_bin. Informe --if-model e --if-preprocessor corretos.")
    lines.append("- Modo diagnostico nao deve ser usado para metricas oficiais.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], contract: dict[str, Any]) -> None:
    lines = ["# EXP-014A-3 - Expanded Policy Assembly", "", "## Resultado", f"- Status: `{summary['objective_status']}`", f"- Built: `{summary['built']}`", f"- Target: `{summary['target_path']}`", "", "## Contrato final", f"- Contract OK: `{contract['contract_ok']}`", f"- Missing: `{contract['missing']}`", f"- Base cols: `{contract['base_pred_cols_present']}`", f"- Final cols: `{contract['final_pred_cols_present']}`", "", "## Proximo passo"]
    if summary["built"]:
        lines += ["Rodar:", "```powershell", "python scripts\\exp_014a_expanded_frozen_validation.py --allow-final-direct", "```"]
    else:
        lines.append("Resolver `05_missing_requirements.md` antes do EXP-014A final.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--if-model", default=None)
    parser.add_argument("--if-preprocessor", default=None)
    parser.add_argument("--allow-diagnostic-approximations", action="store_true")
    parser.add_argument("--score-final-from-lgbm", action="store_true")
    parser.add_argument("--base-pred-from-lgbm-threshold", type=float, default=None)
    parser.add_argument("--force-write-diagnostic", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    target_path = Path(args.target)
    policy_path = Path(args.policy_artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014A-3 - Expanded Policy Assembly")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Target: {target_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input nao encontrado: {input_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy artifact nao encontrado: {policy_path}")

    policy = load_json(policy_path)
    dump_json(policy, output_dir / "policy_used.json")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    n_rows = int(len(df))
    n_frauds = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None

    before = contract_status(ensure_bins_and_guards(df), policy)
    dump_json(before, output_dir / "01_input_contract_before.json")

    steps = []
    before_cols = set(df.columns)
    df = ensure_bins_and_guards(df)
    steps.append({"step": "derive_bins_and_module_quiet", "status": "OK", "note": f"added={sorted(set(df.columns)-before_cols)}"})

    if "if_bin" not in df.columns and not pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        if_model = Path(args.if_model) if args.if_model else None
        if_prep = Path(args.if_preprocessor) if args.if_preprocessor else None
        if if_model is None or not if_model.exists():
            auto_model, auto_prep, inv = discover_if_artifacts(PROJECT_ROOT)
            pd.DataFrame(inv).to_csv(output_dir / "if_artifact_inventory.csv", index=False)
            if if_model is None:
                if_model = auto_model
            if if_prep is None:
                if_prep = auto_prep
        if if_model is not None and if_model.exists():
            try:
                pct, info = score_if(df, if_model, if_prep if if_prep and if_prep.exists() else None)
                df["if_percentile"] = pct
                df = ensure_bins_and_guards(df)
                steps.append({"step": "generate_if_percentile", "status": "OK_OFFICIAL_ATTEMPT", "note": {"model": str(if_model), "preprocessor": str(if_prep), **info}})
            except Exception as exc:
                steps.append({"step": "generate_if_percentile", "status": "FAILED", "note": f"{type(exc).__name__}: {str(exc)[:1000]}"})
        else:
            steps.append({"step": "generate_if_percentile", "status": "MISSING_NO_ARTIFACT", "note": "Nenhum artefato IF encontrado/informado."})

    if "score_bin" not in df.columns and "score_final" not in df.columns:
        if args.allow_diagnostic_approximations and args.score_final_from_lgbm:
            df["score_final"] = (num(df, "lgbm_r4_score") * 100.0).clip(0, 100)
            df = ensure_bins_and_guards(df)
            steps.append({"step": "generate_score_final", "status": "DIAGNOSTIC_APPROXIMATION", "note": "score_final=lgbm_r4_score*100. NAO oficial."})
        else:
            steps.append({"step": "generate_score_final", "status": "MISSING", "note": "Sem score_final/score_bin oficial."})

    if choose_base_col(df) is None and choose_final_col(df) is None:
        if args.allow_diagnostic_approximations and args.base_pred_from_lgbm_threshold is not None:
            df["exp014a_lgbm_base_pred"] = (num(df, "lgbm_r4_score") >= float(args.base_pred_from_lgbm_threshold)).astype(int)
            steps.append({"step": "generate_base_prediction", "status": "DIAGNOSTIC_APPROXIMATION", "note": f"exp014a_lgbm_base_pred=lgbm_r4_score>={args.base_pred_from_lgbm_threshold}. NAO oficial."})
        else:
            steps.append({"step": "generate_base_prediction", "status": "MISSING", "note": "Sem predicao base/final oficial."})

    rule_impact = pd.DataFrame()
    final_col = choose_final_col(df)
    base_col = choose_base_col(df)
    if final_col is not None:
        df["exp014a_frozen_pred"] = pd.to_numeric(df[final_col], errors="coerce").fillna(0).astype(int)
        steps.append({"step": "apply_exp013k_policy", "status": "USED_EXISTING_FINAL_PRED", "note": f"final_col={final_col}"})
    elif base_col is not None:
        try:
            df = ensure_bins_and_guards(df)
            pred, rule_impact = apply_policy(df, policy, base_col)
            df["exp014a_frozen_pred"] = pred
            steps.append({"step": "apply_exp013k_policy", "status": "OK", "note": f"base_col={base_col}, rules={len(rule_impact)}"})
        except Exception as exc:
            steps.append({"step": "apply_exp013k_policy", "status": "FAILED", "note": f"{type(exc).__name__}: {str(exc)[:1000]}"})
    else:
        steps.append({"step": "apply_exp013k_policy", "status": "SKIPPED_NO_BASE", "note": "Sem base/final prediction."})

    if "exp014a_frozen_pred" in df.columns:
        df["exp013k_residual_fp_pred"] = df["exp014a_frozen_pred"].astype(int)

    df = ensure_bins_and_guards(df)
    final_contract = contract_status(df, policy)
    dump_json(steps, output_dir / "02_assembly_steps.json")
    dump_json(final_contract, output_dir / "03_final_contract.json")
    rule_impact.to_csv(output_dir / "04_rule_application_impact.csv", index=False)

    has_diag = any("DIAGNOSTIC" in str(s.get("status", "")) for s in steps)
    official_ok = final_contract["contract_ok"] and not has_diag
    diagnostic_ok = final_contract["contract_ok"] and has_diag and args.force_write_diagnostic and args.allow_diagnostic_approximations

    built = False
    if official_ok or diagnostic_ok:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)
        built = True
        log(f"Arquivo criado: {target_path}")

    df[preview_cols(df)].head(1000).to_csv(output_dir / "07_scored_preview.csv", index=False)

    objective_status = "DONE_CONTRACT_OK" if final_contract["contract_ok"] else "DONE_CONTRACT_NOT_OK"
    if has_diag:
        objective_status += "_HAS_DIAGNOSTIC_APPROX"
    if built:
        objective_status += "_BUILT"

    summary = {"experiment": "EXP-014A-3", "status": "DONE", "objective_status": objective_status, "input_path": str(input_path), "target_path": str(target_path), "n_rows": n_rows, "n_frauds": n_frauds, "built": built, "has_diagnostic_approximations": has_diag, "before_contract": before, "final_contract": final_contract, "steps": steps, "output_dir": str(output_dir)}
    dump_json(summary, output_dir / "00_run_summary.json")
    write_missing_report(output_dir / "05_missing_requirements.md", summary, final_contract, steps)
    write_report(output_dir / "06_assembly_report.md", summary, final_contract)

    log("")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in ["00_run_summary.json", "01_input_contract_before.json", "02_assembly_steps.json", "03_final_contract.json", "04_rule_application_impact.csv", "05_missing_requirements.md", "06_assembly_report.md", "07_scored_preview.csv"]:
        log(f"  {output_dir / p}")


if __name__ == "__main__":
    main()
