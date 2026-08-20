#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014A-4 — Official Runtime Replay Builder

Motivação:
  O EXP-014A-3 ainda não conseguiu montar o input final porque faltaram:
    - base_or_final_prediction_column
    - if_bin
    - score_bin

  Além disso, o IF falhou porque foi aplicado sem o preprocessing correto:
    ValueError: X has 50 features, but IsolationForest is expecting 13 features.

Estratégia correta:
  Em vez de tentar inventar score_final/predição, este script usa o runtime real
  do projeto — simular_pipeline_e2e_v2 / PipelineOrquestrador / DecisionEngine —
  para gerar oficialmente:
    - score_final
    - decisao
    - if_percentile, se o runtime produzir
    - se_score / beh_score, se o runtime produzir
    - predição final direta: exp014a_frozen_pred = decisao em {CONFIRMAR, BLOQUEAR}

  Depois ele combina isso com o arquivo parcial LGBM já resolvido pelo EXP-014A-2:
    dados/exp014a_lgbm_scored_partial.csv

Saída principal:
    dados/exp014a_expanded_scored_input.csv

Uso rápido em amostra:
  python scripts/exp_014a4_official_runtime_replay.py --sample 5000 --workers 1

Uso full:
  python scripts/exp_014a4_official_runtime_replay.py --workers 4

Uso full com chunks menores:
  python scripts/exp_014a4_official_runtime_replay.py --workers 2 --chunk-size 5000

Depois, se criar o arquivo final:
  python scripts/exp_014a_expanded_frozen_validation.py --allow-final-direct

Observação:
  Esta etapa valida a política/predição final do runtime oficial em escala expandida.
  Ela não inventa score_final nem thresholds. Se quiser aplicar as regras EXP-013K
  sobre uma base específica, essa base ainda precisa existir como coluna
  pred_STRICT_RECALL95_SAFE_ONLY ou similar.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
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

DEFAULT_INPUT = PROJECT_ROOT / "dados" / "exp014a_lgbm_scored_partial.csv"
DEFAULT_TARGET = PROJECT_ROOT / "dados" / "exp014a_expanded_scored_input.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014A-4"

FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}

FINAL_PRED_COLS = [
    "exp014a_frozen_pred",
    "exp013k_residual_fp_pred",
    "exp013l_frozen_pred",
]

BASE_PRED_COLS = [
    "pred_STRICT_RECALL95_SAFE_ONLY",
    "exp013k_base_pred",
    "exp013h_frozen_pred",
    "exp013g_micro_pred",
    "pred_HIGH_RECALL_95",
]


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

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

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

    if "lgbm_bin" not in df.columns and pick_col(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"]):
        df["lgbm_bin"] = qbin_series(num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0), "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])

    if "if_bin" not in df.columns and pick_col(df, ["if_percentile", "if_percentile_x", "if_percentile_y"]):
        df["if_bin"] = qbin_series(num(df, ["if_percentile", "if_percentile_x", "if_percentile_y"], 0.0), "if", [0.32, 0.5, 0.7, 0.85, 0.95])

    if "score_bin" not in df.columns and "score_final" in df.columns:
        df["score_bin"] = qbin_series(num(df, "score_final", 0.0), "score", [0.5, 1, 2, 3, 5, 10])

    if "ratio_bin" not in df.columns and "ratio_valor_media_pagador_90d" in df.columns:
        df["ratio_bin"] = qbin_series(num(df, "ratio_valor_media_pagador_90d", 0.0), "ratio", [0.05, 0.1, 0.2, 0.5, 1, 2, 5])

    if "vl_bin" not in df.columns and "vl_pix" in df.columns:
        df["vl_bin"] = qbin_series(num(df, "vl_pix", 0.0), "vl", [20, 50, 100, 250, 500, 1000, 5000, 10000])

    se_score = num(df, ["se_score_x", "se_score_y", "se_score"], 0.0)
    se_count = num(df, ["se_patterns_count", "se_pattern_count"], 0.0)
    beh_score = num(df, ["beh_score", "behavioral_score"], 0.0)
    beh_count = num(df, ["beh_factors_count", "behavioral_risk_factor_count"], 0.0)
    runtime = num(df, "runtime_flagged", 0.0)

    module_strong = (
        (se_score >= 40)
        | (se_count >= 2)
        | (beh_score >= 25)
        | (beh_count >= 2)
        | (runtime >= 1)
    )
    df["module_quiet"] = np.where(module_strong, "module_strong", "module_quiet")

    return df


def setup_project_imports() -> list[str]:
    added = []
    candidates = [
        PROJECT_ROOT,
        PROJECT_ROOT / "backend",
        PROJECT_ROOT / "backend" / "core",
        PROJECT_ROOT / "backend" / "app",
        PROJECT_ROOT / "backend" / "app" / "services",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "experimentos",
    ]

    for p in candidates:
        if p.exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
                added.append(s)

    # Import real preprocessing module if available, so old joblibs can find PixPreprocessor.
    try:
        import preprocessing  # noqa: F401
    except Exception:
        pass

    return added


def find_module_file(filename: str) -> Path | None:
    for root in [PROJECT_ROOT / "backend", PROJECT_ROOT / "scripts", PROJECT_ROOT / "experimentos", PROJECT_ROOT]:
        if not root.exists():
            continue
        hits = list(root.rglob(filename))
        if hits:
            # prefer backend/core or backend
            hits.sort(key=lambda p: (0 if "backend" in str(p).lower() else 1, len(str(p))))
            return hits[0]
    return None


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Não consegui criar spec para {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_runtime_module() -> tuple[Any | None, str | None, str]:
    setup_project_imports()

    # 1) direct import.
    for name in ["simular_pipeline_e2e_v2", "utils_experimentos"]:
        try:
            return importlib.import_module(name), name, "direct_import"
        except Exception:
            pass

    # 2) file discovery.
    for filename, module_name in [
        ("simular_pipeline_e2e_v2.py", "simular_pipeline_e2e_v2"),
        ("utils_experimentos.py", "utils_experimentos"),
    ]:
        p = find_module_file(filename)
        if p is not None:
            try:
                return import_module_from_path(module_name, p), module_name, f"import_from_path:{p}"
            except Exception as exc:
                return None, module_name, f"failed_import_from_path:{p}:{exc}"

    return None, None, "not_found"


def process_with_runtime(df: pd.DataFrame, workers: int, engine_overrides: dict[str, Any] | None) -> pd.DataFrame:
    mod, name, mode = load_runtime_module()
    if mod is None:
        raise RuntimeError(f"Não encontrei runtime simular_pipeline_e2e_v2/utils_experimentos. mode={mode}")

    log(f"Runtime carregado: {name} ({mode})")

    if hasattr(mod, "process_dataframe_via_orquestrador"):
        return mod.process_dataframe_via_orquestrador(
            df,
            workers=workers,
            logger=None,
            engine_config_overrides=engine_overrides,
        )

    if workers and workers > 1 and hasattr(mod, "process_batch_parallel"):
        return mod.process_batch_parallel(
            df,
            n_workers=workers,
            engine_config_overrides=engine_overrides,
        )

    if hasattr(mod, "process_batch_sequential"):
        return mod.process_batch_sequential(
            df,
            engine_config_overrides=engine_overrides,
        )

    raise RuntimeError(f"Runtime {name} não possui funções esperadas.")


def merge_predictions(base: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    pred = normalize_columns(pred)

    # Prefer transaction_id merge.
    if "transaction_id" in base.columns and "transaction_id" in pred.columns:
        base2 = base.copy()
        pred2 = pred.drop_duplicates("transaction_id").copy()

        # Bring all prediction columns, suffix conflicts.
        cols_to_add = [c for c in pred2.columns if c != "transaction_id"]
        merged = base2.merge(pred2[["transaction_id"] + cols_to_add], on="transaction_id", how="left", suffixes=("", "_runtime"))

        # For important runtime cols, prefer runtime suffix if original exists.
        for c in ["score_final", "decisao", "if_percentile", "se_score", "beh_score", "lgbm_raw"]:
            cr = c + "_runtime"
            if cr in merged.columns:
                merged[c] = merged[cr].combine_first(merged[c]) if c in merged.columns else merged[cr]
        return merged

    # Fallback index-aligned.
    out = base.copy()
    for c in pred.columns:
        if c in out.columns:
            out[c + "_runtime"] = pred[c].values[: len(out)]
            if c in ["score_final", "decisao", "if_percentile", "se_score", "beh_score", "lgbm_raw"]:
                out[c] = out[c + "_runtime"]
        else:
            out[c] = pred[c].values[: len(out)]
    return out


def run_runtime_in_chunks(
    df: pd.DataFrame,
    workers: int,
    chunk_size: int,
    output_dir: Path,
    resume: bool,
    engine_overrides: dict[str, Any] | None,
) -> pd.DataFrame:
    chunks_dir = output_dir / "runtime_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    n = len(df)
    n_chunks = int(math.ceil(n / chunk_size))

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        chunk_path = chunks_dir / f"chunk_{i:05d}_{start}_{end}.csv"

        if resume and chunk_path.exists():
            log(f"[chunk {i+1}/{n_chunks}] usando checkpoint {chunk_path.name}")
            out = pd.read_csv(chunk_path, low_memory=False)
            outputs.append(out)
            continue

        log(f"[chunk {i+1}/{n_chunks}] processando linhas {start}:{end}")
        chunk = df.iloc[start:end].copy()
        t0 = time.perf_counter()
        out = process_with_runtime(chunk, workers=workers, engine_overrides=engine_overrides)
        elapsed = time.perf_counter() - t0
        out = normalize_columns(out)
        out.to_csv(chunk_path, index=False)
        log(f"[chunk {i+1}/{n_chunks}] OK em {elapsed:.1f}s -> {chunk_path.name}")
        outputs.append(out)

    return pd.concat(outputs, ignore_index=True)


def derive_final_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "decisao" in df.columns:
        df["exp014a_frozen_pred"] = df["decisao"].astype(str).str.upper().isin(FLAGGED_DECISIONS).astype(int)
        df["runtime_flagged"] = df["exp014a_frozen_pred"]
        # Alias accepted by old EXP-014A final direct mode.
        df["exp013k_residual_fp_pred"] = df["exp014a_frozen_pred"]
    elif "score_final" in df.columns:
        # Do NOT invent official threshold; this is only to avoid silent failure.
        # We intentionally do not create final pred from score_final unless user asks in a future script.
        pass

    return df


def contract_status(df: pd.DataFrame) -> dict[str, Any]:
    missing = []

    if "is_fraud" not in df.columns:
        missing.append("is_fraud")
    if not any(c in df.columns for c in ["event_datetime", "data_pix", "dt_pix"]):
        missing.append("event_datetime_or_data_pix")
    if not any(c in df.columns for c in FINAL_PRED_COLS + BASE_PRED_COLS):
        missing.append("base_or_final_prediction_column")

    requirements = {
        "if_bin": [["if_bin"], ["if_percentile"], ["if_percentile_x"], ["if_percentile_y"]],
        "lgbm_bin": [["lgbm_bin"], ["lgbm_r4_score"], ["r4_score"], ["lgbm_mapped"], ["lgbm_raw"]],
        "score_bin": [["score_bin"], ["score_final"]],
        "ratio_bin": [["ratio_bin"], ["ratio_valor_media_pagador_90d"]],
        "vl_bin": [["vl_bin"], ["vl_pix"]],
        "value_band": [["value_band"]],
        "ds_tipo_chave_norm": [["ds_tipo_chave_norm"]],
        "first_receiver_flag_real": [["first_receiver_flag_real"]],
        "mbk_available_flag": [["mbk_available_flag"]],
    }

    for logical, alternatives in requirements.items():
        if not any(all(c in df.columns for c in alt) for alt in alternatives):
            missing.append(f"feature_or_bin:{logical}")

    return {
        "contract_ok": len(missing) == 0,
        "missing": missing,
        "final_pred_cols_present": [c for c in FINAL_PRED_COLS if c in df.columns],
        "base_pred_cols_present": [c for c in BASE_PRED_COLS if c in df.columns],
        "has_score_final": "score_final" in df.columns,
        "has_if_percentile": any(c in df.columns for c in ["if_percentile", "if_percentile_x", "if_percentile_y"]),
        "has_decisao": "decisao" in df.columns,
    }


def safe_metric_preview(df: pd.DataFrame) -> dict[str, Any]:
    if "is_fraud" not in df.columns or "exp014a_frozen_pred" not in df.columns:
        return {}

    y = df["is_fraud"].astype(int).to_numpy()
    p = df["exp014a_frozen_pred"].astype(int).to_numpy()

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
    }


def make_report(summary: dict[str, Any], contract: dict[str, Any]) -> str:
    lines = []
    lines.append("# EXP-014A-4 — Official Runtime Replay Builder")
    lines.append("")
    lines.append("## Resultado")
    lines.append(f"- Status: `{summary['objective_status']}`")
    lines.append(f"- Built: `{summary['built']}`")
    lines.append(f"- Target: `{summary['target_path']}`")
    lines.append("")
    lines.append("## Contrato")
    lines.append(f"- Contract OK: `{contract['contract_ok']}`")
    lines.append(f"- Missing: `{contract['missing']}`")
    lines.append(f"- Final pred cols: `{contract['final_pred_cols_present']}`")
    lines.append(f"- Has score_final: `{contract['has_score_final']}`")
    lines.append(f"- Has if_percentile: `{contract['has_if_percentile']}`")
    lines.append("")
    if summary.get("metric_preview"):
        lines.append("## Preview de métricas runtime")
        for k, v in summary["metric_preview"].items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.append("## Próximo passo")
    if summary["built"]:
        lines.append("Rodar:")
        lines.append("```powershell")
        lines.append("python scripts\\exp_014a_expanded_frozen_validation.py --allow-final-direct")
        lines.append("```")
    else:
        lines.append("Resolver os faltantes em `02_contract_after.json` antes do EXP-014A final.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample", type=int, default=0, help="0 = full; >0 = amostra estratificada para teste.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-runtime", action="store_true", help="Apenas testa contrato/bins sem rodar runtime.")
    parser.add_argument("--engine-overrides-json", default=None)
    parser.add_argument("--write-even-if-contract-fails", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    target_path = Path(args.target)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014A-4 — Official Runtime Replay Builder")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Target: {target_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    original_rows = len(df)
    original_frauds = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None

    if args.sample and args.sample > 0 and args.sample < len(df):
        if "is_fraud" in df.columns and df["is_fraud"].nunique() > 1:
            fraud = df[df["is_fraud"] == 1]
            normal = df[df["is_fraud"] == 0]
            n_fraud = min(len(fraud), max(1, min(len(fraud), int(args.sample * 0.25))))
            n_normal = min(len(normal), args.sample - n_fraud)
            df = pd.concat([
                fraud.sample(n_fraud, random_state=42),
                normal.sample(n_normal, random_state=42),
            ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        log(f"Amostra ativa: {len(df)} linhas")

    df = ensure_bins(df)
    before_contract = contract_status(df)
    dump_json(before_contract, output_dir / "01_contract_before.json")

    engine_overrides = None
    if args.engine_overrides_json:
        engine_overrides = json.loads(args.engine_overrides_json)

    runtime_error = None
    runtime_shape = None

    if not args.no_runtime:
        try:
            predictions = run_runtime_in_chunks(
                df=df,
                workers=args.workers,
                chunk_size=args.chunk_size,
                output_dir=output_dir,
                resume=args.resume,
                engine_overrides=engine_overrides,
            )
            runtime_shape = list(predictions.shape)
            predictions.to_csv(output_dir / "runtime_predictions_raw.csv", index=False)
            df = merge_predictions(df, predictions)
        except Exception as exc:
            runtime_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
            log("ERRO NO RUNTIME:")
            log(str(runtime_error["message"]))

    df = derive_final_prediction(df)
    df = ensure_bins(df)

    after_contract = contract_status(df)
    dump_json(after_contract, output_dir / "02_contract_after.json")

    preview_cols = []
    for c in [
        "transaction_id", "is_fraud", "event_datetime", "data_pix",
        "decisao", "score_final", "score_bin",
        "if_percentile", "if_bin",
        "lgbm_r4_score", "lgbm_bin",
        "exp014a_frozen_pred", "exp013k_residual_fp_pred",
        "value_band", "ds_tipo_chave_norm", "first_receiver_flag_real",
        "mbk_available_flag", "module_quiet", "vl_pix", "vl_bin", "ratio_bin",
    ]:
        if c in df.columns and c not in preview_cols:
            preview_cols.append(c)

    df[preview_cols].head(1000).to_csv(output_dir / "03_scored_preview.csv", index=False)

    metric_preview = safe_metric_preview(df)
    built = False

    if after_contract["contract_ok"] or args.write_even_if_contract_fails:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)
        built = True
        log(f"Arquivo criado: {target_path}")

    objective_status = "DONE_CONTRACT_OK" if after_contract["contract_ok"] else "DONE_CONTRACT_NOT_OK"
    if runtime_error:
        objective_status += "_RUNTIME_FAILED"
    if built:
        objective_status += "_BUILT"

    summary = {
        "experiment": "EXP-014A-4",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "target_path": str(target_path),
        "original_rows": int(original_rows),
        "original_frauds": original_frauds,
        "active_rows": int(len(df)),
        "active_frauds": int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None,
        "sample": args.sample,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "runtime_shape": runtime_shape,
        "runtime_error": runtime_error,
        "before_contract": before_contract,
        "after_contract": after_contract,
        "built": built,
        "metric_preview": metric_preview,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, after_contract)
    (output_dir / "04_runtime_replay_report.md").write_text(report, encoding="utf-8")

    log("")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_contract_before.json",
        output_dir / "02_contract_after.json",
        output_dir / "03_scored_preview.csv",
        output_dir / "04_runtime_replay_report.md",
        output_dir / "runtime_predictions_raw.csv",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    import argparse
    main()
