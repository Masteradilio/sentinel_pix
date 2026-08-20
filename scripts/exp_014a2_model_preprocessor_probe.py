#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014A-2 — Model/Preprocessor Pair Probe

Por que existe:
  O EXP-014A-1 mostrou que:
    - o LGBM indicado foi carregado, mas com 49/52 features ausentes;
    - o IsolationForest foi carregado, mas falhou porque recebeu colunas/string erradas;
    - score_final e predição base/final ainda não existem.

  Isso significa que precisamos parar de apontar "um modelo solto" e localizar o
  PAR correto:
      preprocessor + model
  usado no experimento vencedor de LGBM.

Objetivo:
  1. Inventariar pares model/preprocessor.
  2. Testar cada par em uma amostra do dataset expandido.
  3. Identificar quais pares conseguem scorear sem features ausentes.
  4. Opcionalmente gerar um arquivo parcial com lgbm_r4_score/lgbm_bin quando
     um par exato for encontrado.

Este script NÃO gera score_final e NÃO gera predição final oficial.
Ele resolve a etapa de identificar/aplicar o LGBM corretamente.

Uso:
  python scripts/exp_014a2_model_preprocessor_probe.py

Com input explícito:
  python scripts/exp_014a2_model_preprocessor_probe.py --input dados\\hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv

Testar par específico:
  python scripts/exp_014a2_model_preprocessor_probe.py ^
    --model backend\\artefatos_candidatos\\exp012c_r4_lgbm_fp_squeeze\\model_lgbm_v3_r4_fp_squeeze_shadow.joblib ^
    --preprocessor backend\\artefatos_candidatos\\exp012c_r4_lgbm_fp_squeeze\\preprocessor_lgbm_v3_r4_fp_squeeze_shadow.joblib

Gerar arquivo parcial com score LGBM se o par passar:
  python scripts/exp_014a2_model_preprocessor_probe.py --build-lgbm-scored

Saídas:
  resultados/experimentos/EXP-014A-2/
    00_run_summary.json
    01_pair_inventory.csv
    02_pair_probe_results.csv
    03_recommended_pairs.md
    04_lgbm_scored_preview.csv
  dados/exp014a_lgbm_scored_partial.csv, se --build-lgbm-scored e par exato passar
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "dados").exists() else Path.cwd()

DEFAULT_INPUT = PROJECT_ROOT / "dados" / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-014A-2"
DEFAULT_TARGET = PROJECT_ROOT / "dados" / "exp014a_lgbm_scored_partial.csv"

MODEL_EXTS = {".joblib", ".pkl", ".pickle", ".sav"}
MODEL_HINTS = ["model", "lgbm", "lightgbm"]
PREP_HINTS = ["preprocessor", "preprocess", "preprocessing"]


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_obj(path: Path) -> Any:
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
    return df


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


def discover_files(root: Path) -> pd.DataFrame:
    roots = [
        root / "backend" / "artefatos_candidatos",
        root / "backend" / "artefatos",
        root / "resultados" / "experimentos",
    ]
    rows = []
    seen = set()

    for sr in roots:
        if not sr.exists():
            continue
        for p in sr.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in MODEL_EXTS:
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            low = p.name.lower()
            is_model = any(h in low for h in MODEL_HINTS) and not any(h in low for h in PREP_HINTS)
            is_prep = any(h in low for h in PREP_HINTS)
            rows.append({
                "path": str(p),
                "dir": str(p.parent),
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "is_model_like": is_model,
                "is_preprocessor_like": is_prep,
                "priority": (
                    100 if "exp012c_r4" in str(p).lower() else
                    90 if "exp012c_r3" in str(p).lower() else
                    80 if "exp012c" in str(p).lower() else
                    70 if "exp011" in str(p).lower() else
                    10
                ),
            })

    if not rows:
        return pd.DataFrame(columns=["path", "dir", "filename", "size_bytes", "is_model_like", "is_preprocessor_like", "priority"])

    return pd.DataFrame(rows).sort_values(["priority", "size_bytes"], ascending=[False, False]).reset_index(drop=True)


def build_pair_inventory(files: pd.DataFrame, explicit_model: str | None, explicit_prep: str | None) -> pd.DataFrame:
    rows = []

    if explicit_model and explicit_prep:
        rows.append({
            "pair_id": "explicit_pair",
            "model_path": str(Path(explicit_model)),
            "preprocessor_path": str(Path(explicit_prep)),
            "pair_source": "explicit",
            "priority": 1000,
        })
        return pd.DataFrame(rows)

    if files.empty:
        return pd.DataFrame(columns=["pair_id", "model_path", "preprocessor_path", "pair_source", "priority"])

    # Pair by same directory.
    for d, g in files.groupby("dir"):
        models = g[g["is_model_like"] == True]
        preps = g[g["is_preprocessor_like"] == True]
        if models.empty or preps.empty:
            continue

        for _, m in models.iterrows():
            for _, p in preps.iterrows():
                rows.append({
                    "pair_id": f"same_dir_{len(rows):04d}",
                    "model_path": m["path"],
                    "preprocessor_path": p["path"],
                    "pair_source": "same_dir",
                    "priority": max(int(m["priority"]), int(p["priority"])),
                })

    # Pair known model/preprocessor name families even if model-like detection misses.
    for d, g in files.groupby("dir"):
        preps = g[g["is_preprocessor_like"] == True]
        if preps.empty:
            continue
        models = g[(g["is_preprocessor_like"] == False) & (g["size_bytes"] > 10000)]
        for _, m in models.iterrows():
            for _, p in preps.iterrows():
                rows.append({
                    "pair_id": f"loose_same_dir_{len(rows):04d}",
                    "model_path": m["path"],
                    "preprocessor_path": p["path"],
                    "pair_source": "loose_same_dir",
                    "priority": max(int(m["priority"]), int(p["priority"])) - 1,
                })

    if not rows:
        return pd.DataFrame(columns=["pair_id", "model_path", "preprocessor_path", "pair_source", "priority"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["model_path", "preprocessor_path"])
    out = out.sort_values(["priority"], ascending=False).reset_index(drop=True)
    out["pair_id"] = [f"pair_{i:04d}" for i in range(len(out))]
    return out


def required_columns_from_preprocessor(prep: Any) -> list[str] | None:
    if hasattr(prep, "feature_names_in_"):
        try:
            return [str(x) for x in list(prep.feature_names_in_)]
        except Exception:
            pass

    # ColumnTransformer stores columns in transformers_
    if hasattr(prep, "transformers_"):
        cols = []
        try:
            for _, _, c in prep.transformers_:
                if c is None:
                    continue
                if isinstance(c, slice):
                    return None
                if isinstance(c, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend([str(x) for x in list(c)])
                elif isinstance(c, str):
                    cols.append(c)
            return sorted(set(cols)) if cols else None
        except Exception:
            pass

    return None


def safe_transform(prep: Any, df: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    required = required_columns_from_preprocessor(prep)
    missing = []
    X_in = df

    if required:
        missing = [c for c in required if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        X_in = df[required]

    try:
        X = prep.transform(X_in)
        return X, {
            "required_columns_count": None if required is None else len(required),
            "missing_columns_count": len(missing),
            "missing_columns": missing[:100],
            "transform_input": "required_columns" if required else "full_df",
        }
    except Exception as exc:
        # Try full df fallback.
        try:
            X = prep.transform(df)
            return X, {
                "required_columns_count": None if required is None else len(required),
                "missing_columns_count": len(missing),
                "missing_columns": missing[:100],
                "transform_input": "full_df_fallback",
                "first_error": str(exc)[:500],
            }
        except Exception as exc2:
            raise RuntimeError(f"preprocessor.transform failed: {exc2}") from exc


def safe_predict(model: Any, X: Any) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        arr = np.asarray(p)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return arr[:, 1].astype(float), "predict_proba_col1"
        return arr.reshape(-1).astype(float), "predict_proba_flat"
    if hasattr(model, "predict"):
        p = model.predict(X)
        return np.asarray(p).reshape(-1).astype(float), "predict"
    raise RuntimeError("model has no predict_proba/predict")


def probe_pair(pair: pd.Series, df_sample: pd.DataFrame) -> dict[str, Any]:
    model_path = Path(str(pair["model_path"]))
    prep_path = Path(str(pair["preprocessor_path"]))

    result = {
        "pair_id": pair["pair_id"],
        "model_path": str(model_path),
        "preprocessor_path": str(prep_path),
        "pair_source": pair["pair_source"],
        "priority": int(pair["priority"]),
        "status": "UNKNOWN",
        "score_method": None,
        "n_rows_scored": int(len(df_sample)),
        "required_columns_count": None,
        "missing_columns_count": None,
        "missing_columns": None,
        "roc_auc_sample": None,
        "average_precision_sample": None,
        "score_min": None,
        "score_mean": None,
        "score_max": None,
        "error": None,
    }

    try:
        prep = load_obj(prep_path)
        model = load_obj(model_path)
        X, prep_info = safe_transform(prep, df_sample.copy())
        score, method = safe_predict(model, X)

        result["status"] = "OK_EXACT" if prep_info["missing_columns_count"] == 0 else "OK_WITH_MISSING_COLUMNS"
        result["score_method"] = method
        result.update({
            "required_columns_count": prep_info["required_columns_count"],
            "missing_columns_count": prep_info["missing_columns_count"],
            "missing_columns": "|".join(prep_info["missing_columns"][:30]),
            "score_min": float(np.nanmin(score)),
            "score_mean": float(np.nanmean(score)),
            "score_max": float(np.nanmax(score)),
        })

        if "is_fraud" in df_sample.columns and df_sample["is_fraud"].nunique() > 1:
            y = df_sample["is_fraud"].to_numpy(dtype=int)
            try:
                result["roc_auc_sample"] = float(roc_auc_score(y, score))
            except Exception:
                pass
            try:
                result["average_precision_sample"] = float(average_precision_score(y, score))
            except Exception:
                pass

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = str(exc)[:1000]

    return result


def score_full_pair(model_path: Path, prep_path: Path, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    prep = load_obj(prep_path)
    model = load_obj(model_path)
    X, prep_info = safe_transform(prep, df.copy())
    score, method = safe_predict(model, X)
    return score, {"predict_method": method, **prep_info}


def make_recommended_pairs(results: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-014A-2 — Recommended Model/Preprocessor Pairs")
    lines.append("")
    if results.empty:
        lines.append("Nenhum par foi testado.")
        return "\n".join(lines)

    ok = results[results["status"].astype(str).str.startswith("OK")].copy()
    exact = ok[ok["missing_columns_count"].fillna(999999).astype(float) == 0].copy()

    lines.append("## Melhor leitura")
    if not exact.empty:
        best = exact.sort_values(["priority", "average_precision_sample", "roc_auc_sample"], ascending=[False, False, False]).iloc[0]
        lines.append("Existe pelo menos um par com colunas exatas. Use esse par para gerar `lgbm_r4_score`.")
        lines.append("")
        lines.append("```powershell")
        lines.append(
            "python scripts\\exp_014a2_model_preprocessor_probe.py "
            f"--model \"{best['model_path']}\" "
            f"--preprocessor \"{best['preprocessor_path']}\" "
            "--build-lgbm-scored"
        )
        lines.append("```")
    elif not ok.empty:
        best = ok.sort_values(["missing_columns_count", "priority"], ascending=[True, False]).iloc[0]
        lines.append("Nenhum par exato foi encontrado. O melhor par ainda tem colunas ausentes; não usar oficialmente.")
        lines.append(f"- missing_columns_count: `{best['missing_columns_count']}`")
        lines.append(f"- model: `{best['model_path']}`")
        lines.append(f"- preprocessor: `{best['preprocessor_path']}`")
    else:
        lines.append("Nenhum par conseguiu scorear.")

    lines.append("")
    lines.append("## Top resultados")
    show_cols = [
        "status", "missing_columns_count", "roc_auc_sample", "average_precision_sample",
        "model_path", "preprocessor_path", "error"
    ]
    lines.append(results[show_cols].head(20).to_markdown(index=False))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--model", default=None)
    parser.add_argument("--preprocessor", default=None)
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-lgbm-scored", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-014A-2 — Model/Preprocessor Pair Probe")
    log("=" * 80)
    log(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    n_rows = len(df)
    n_frauds = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else None

    if args.sample and len(df) > args.sample:
        # Stratified sample when label exists.
        if "is_fraud" in df.columns and df["is_fraud"].nunique() > 1:
            pos = df[df["is_fraud"] == 1]
            neg = df[df["is_fraud"] == 0]
            n_pos = min(len(pos), max(100, int(args.sample * 0.2)))
            n_neg = max(0, args.sample - n_pos)
            sample_df = pd.concat([
                pos.sample(n_pos, random_state=args.seed),
                neg.sample(min(len(neg), n_neg), random_state=args.seed),
            ], ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        else:
            sample_df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    else:
        sample_df = df.copy()

    files = discover_files(PROJECT_ROOT)
    pairs = build_pair_inventory(files, args.model, args.preprocessor)
    pairs.to_csv(output_dir / "01_pair_inventory.csv", index=False)

    results = []
    for _, pair in pairs.iterrows():
        log(f"Testing {pair['pair_id']}: {Path(pair['model_path']).name} + {Path(pair['preprocessor_path']).name}")
        results.append(probe_pair(pair, sample_df))

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["status", "missing_columns_count", "priority", "average_precision_sample"],
            ascending=[True, True, False, False],
            na_position="last",
        ).reset_index(drop=True)
    results_df.to_csv(output_dir / "02_pair_probe_results.csv", index=False)

    recommended = make_recommended_pairs(results_df)
    (output_dir / "03_recommended_pairs.md").write_text(recommended, encoding="utf-8")

    built = False
    build_note = None

    if args.build_lgbm_scored:
        if results_df.empty:
            raise RuntimeError("Nenhum resultado de probe para construir score.")

        ok_exact = results_df[
            (results_df["status"] == "OK_EXACT") &
            (results_df["missing_columns_count"].fillna(999999).astype(float) == 0)
        ].copy()

        if ok_exact.empty:
            raise RuntimeError("Nenhum par exato encontrado; não vou gerar score oficial.")

        best = ok_exact.sort_values(["priority", "average_precision_sample", "roc_auc_sample"], ascending=[False, False, False]).iloc[0]
        score, info = score_full_pair(Path(best["model_path"]), Path(best["preprocessor_path"]), df)
        df["lgbm_r4_score"] = score
        df["lgbm_bin"] = qbin_series(df["lgbm_r4_score"], "lgbm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1])

        target = Path(args.target)
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)

        preview_cols = [c for c in ["transaction_id", "is_fraud", "event_datetime", "data_pix", "lgbm_r4_score", "lgbm_bin", "vl_pix", "value_band", "ds_tipo_chave_norm"] if c in df.columns]
        df[preview_cols].head(1000).to_csv(output_dir / "04_lgbm_scored_preview.csv", index=False)
        built = True
        build_note = {"target": str(target), "best_pair": best.to_dict(), "score_info": info}
        log(f"Arquivo parcial criado: {target}")
    else:
        if not results_df.empty:
            best_ok = results_df[results_df["status"].astype(str).str.startswith("OK")].head(1)
            if not best_ok.empty:
                preview = sample_df.copy()
                try:
                    score, info = score_full_pair(Path(best_ok.iloc[0]["model_path"]), Path(best_ok.iloc[0]["preprocessor_path"]), preview)
                    preview["lgbm_probe_score"] = score
                    preview_cols = [c for c in ["transaction_id", "is_fraud", "event_datetime", "data_pix", "lgbm_probe_score", "vl_pix", "value_band", "ds_tipo_chave_norm"] if c in preview.columns]
                    preview[preview_cols].head(1000).to_csv(output_dir / "04_lgbm_scored_preview.csv", index=False)
                except Exception:
                    pass

    summary = {
        "experiment": "EXP-014A-2",
        "status": "DONE",
        "objective_status": "DONE_BUILT_LGBM_PARTIAL" if built else "DONE_PROBE_ONLY",
        "input_path": str(input_path),
        "n_rows": int(n_rows),
        "n_frauds": n_frauds,
        "n_pairs_tested": int(len(results_df)),
        "n_ok_exact": int(((results_df["status"] == "OK_EXACT") & (results_df["missing_columns_count"].fillna(999999).astype(float) == 0)).sum()) if not results_df.empty else 0,
        "n_ok_any": int(results_df["status"].astype(str).str.startswith("OK").sum()) if not results_df.empty else 0,
        "built": built,
        "build_note": build_note,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    log("")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_pair_inventory.csv",
        output_dir / "02_pair_probe_results.csv",
        output_dir / "03_recommended_pairs.md",
        output_dir / "04_lgbm_scored_preview.csv",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
