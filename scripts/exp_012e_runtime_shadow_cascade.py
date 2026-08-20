#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-012E — E2E Runtime Shadow da Cascata EXP-012D

Objetivo:
  Validar a cascata EXP-012D em modo shadow contra o PipelineOrquestrador /
  DecisionEngine reais, sem promover artefatos e sem alterar produção.

O que este script faz:
  1. Carrega o dataset v3 completo.
  2. Seleciona por padrão o HOLDOUT_LABEL_SAFE.
  3. Processa o mesmo sample no PipelineOrquestrador real.
  4. Lê as predições shadow do EXP-012D.
  5. Compara:
       - RUNTIME_BASELINE: decisão real do PipelineOrquestrador/DecisionEngine;
       - SHADOW_EXP012D: política campeã offline do EXP-012D;
       - UNION_BASELINE_OR_SHADOW;
       - INTERSECTION_BASELINE_AND_SHADOW.
  6. Gera análise de deltas: FNs recuperados, FPs adicionados,
     TPs perdidos e FPs removidos vs runtime baseline.
  7. Salva relatório e artefatos de auditoria.

Importante:
  - Este experimento NÃO troca artefatos.
  - Este experimento NÃO escreve scoring_config.
  - Este experimento NÃO promove modelo.
  - É um E2E shadow comparativo.

Uso recomendado:
  Smoke test:
    python scripts\\exp_012e_runtime_shadow_cascade.py --sample 2000 --workers 4

  Full HOLDOUT_LABEL_SAFE:
    python scripts\\exp_012e_runtime_shadow_cascade.py --workers 4

Entradas default:
  dados\\hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv
  resultados\\experimentos\\EXP-012D\\05_champion_predictions_holdout_label_safe.csv

Saídas:
  resultados\\experimentos\\EXP-012E\\
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / "backend").exists() else Path.cwd()

DADOS_DIR = PROJECT_ROOT / "dados"
OUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012E"

DEFAULT_INPUT = DADOS_DIR / "hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv"
DEFAULT_SHADOW_HOLDOUT_SAFE = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012D" / "05_champion_predictions_holdout_label_safe.csv"
DEFAULT_EXP012D_SUMMARY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-012D" / "00_run_summary.json"

FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | EXP-012E | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EXP-012E")


# =============================================================================
# Helpers
# =============================================================================
def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split(".")[-1] for c in df.columns]

    if "event_datetime" not in df.columns and "dt_pix" in df.columns:
        df["event_datetime"] = df["dt_pix"]
    if "transaction_id" not in df.columns and "cd_pix" in df.columns:
        df["transaction_id"] = df["cd_pix"]

    if "transaction_id" not in df.columns:
        raise RuntimeError("Coluna transaction_id/cd_pix ausente.")
    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna is_fraud ausente.")
    if "event_datetime" not in df.columns:
        raise RuntimeError("Coluna event_datetime/dt_pix ausente.")

    df["transaction_id"] = df["transaction_id"].astype("string").str.strip()
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    if "data_pix" in df.columns:
        df["data_pix"] = pd.to_datetime(df["data_pix"], errors="coerce")
    else:
        df["data_pix"] = df["event_datetime"].dt.normalize()

    df = df[df["event_datetime"].notna() & df["data_pix"].notna()].copy()
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    if "temporal_split" in df.columns:
        df["temporal_split"] = df["temporal_split"].astype(str).str.upper().str.strip()

    return df.sort_values(["event_datetime", "transaction_id"]).reset_index(drop=True)


def get_holdout_label_safe(df: pd.DataFrame) -> pd.DataFrame:
    if "temporal_split" not in df.columns:
        raise RuntimeError("temporal_split ausente; não é possível montar HOLDOUT_LABEL_SAFE.")

    holdout = df[df["temporal_split"] == "HOLDOUT"].copy()
    if holdout.empty:
        raise RuntimeError("Split HOLDOUT vazio.")

    max_fraud_dt = holdout.loc[holdout["is_fraud"] == 1, "data_pix"].max()
    if pd.isna(max_fraud_dt):
        raise RuntimeError("HOLDOUT não tem fraude confirmada.")

    return holdout[holdout["data_pix"] <= max_fraud_dt].copy().reset_index(drop=True)


def select_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split = split_name.upper()
    if split == "HOLDOUT_LABEL_SAFE":
        return get_holdout_label_safe(df)
    if split == "HOLDOUT_FULL":
        return df[df["temporal_split"] == "HOLDOUT"].copy().reset_index(drop=True)
    if split in {"TRAIN", "VALIDATION", "HOLDOUT"}:
        return df[df["temporal_split"] == split].copy().reset_index(drop=True)
    raise ValueError(f"Split não suportado: {split_name}")


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df.copy().reset_index(drop=True)

    fraud = df[df["is_fraud"] == 1].copy()
    normal = df[df["is_fraud"] == 0].copy()

    n_fraud = min(len(fraud), max(1, int(round(n * len(fraud) / max(len(df), 1)))))
    # Para E2E, mantenha todas as fraudes se couber.
    if len(fraud) <= n:
        n_fraud = len(fraud)

    n_normal = min(len(normal), max(0, n - n_fraud))
    rng = np.random.RandomState(seed)

    fraud_s = fraud.sample(n=n_fraud, random_state=seed) if n_fraud < len(fraud) else fraud
    normal_s = normal.sample(n=n_normal, random_state=seed) if n_normal < len(normal) else normal

    return pd.concat([fraud_s, normal_s], axis=0).sort_values("event_datetime").reset_index(drop=True)


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 8),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 8),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 8),
        "fpr": round(float(fp / max(fp + tn, 1)), 8),
    }


def add_metric_row(rows, label, y_true, y_pred, extra=None):
    row = {"policy": label}
    row.update(compute_metrics(y_true, y_pred))
    if extra:
        row.update(extra)
    rows.append(row)


def flagged_from_runtime(preds: pd.DataFrame) -> pd.Series:
    if "decisao" not in preds.columns:
        raise RuntimeError("Predições runtime não possuem coluna decisao.")
    return preds["decisao"].astype(str).str.upper().isin(FLAGGED_DECISIONS).astype(int)


def choose_shadow_pred_column(df: pd.DataFrame) -> str:
    preferred = ["exp012d_pred", "r4_pred", "lgbm_r4_pred", "shadow_pred"]
    for c in preferred:
        if c in df.columns:
            return c
    # Fallback: qualquer coluna terminando em _pred com valores 0/1.
    for c in df.columns:
        if c.lower().endswith("_pred"):
            vals = set(pd.to_numeric(df[c], errors="coerce").dropna().astype(int).unique().tolist())
            if vals.issubset({0, 1}):
                return c
    raise RuntimeError("Não encontrei coluna de predição shadow no arquivo EXP-012D.")


def load_shadow_predictions(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo shadow EXP-012D não encontrado: {path}")

    sh = normalize_columns(pd.read_csv(path, low_memory=False))
    pred_col = choose_shadow_pred_column(sh)
    sh[pred_col] = pd.to_numeric(sh[pred_col], errors="coerce").fillna(0).astype(int)
    return sh, pred_col


def setup_paths_for_runtime() -> None:
    candidates = [
        PROJECT_ROOT,
        PROJECT_ROOT / "experimentos",
        PROJECT_ROOT / "backend",
        PROJECT_ROOT / "backend" / "core",
        PROJECT_ROOT / "backend" / "scripts",
    ]
    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p))


def run_pipeline_runtime(df_sample: pd.DataFrame, workers: int, engine_config_overrides: dict[str, Any] | None = None) -> pd.DataFrame:
    setup_paths_for_runtime()

    try:
        from utils_experimentos import process_dataframe_via_orquestrador
    except Exception as exc:
        raise RuntimeError(
            "Falha ao importar experimentos/utils_experimentos.py. "
            "Confirme que o script está na raiz do projeto rebuild_pix."
        ) from exc

    return process_dataframe_via_orquestrador(
        df_sample,
        workers=workers,
        logger=log,
        engine_config_overrides=engine_config_overrides,
    )


def safe_merge_runtime_with_sample(sample: pd.DataFrame, runtime_preds: pd.DataFrame) -> pd.DataFrame:
    rt = runtime_preds.copy()
    rt.columns = [str(c).strip().split(".")[-1] for c in rt.columns]

    # Muitos scripts preservam transaction_id; se não, usar índice.
    if "transaction_id" in rt.columns:
        rt["transaction_id"] = rt["transaction_id"].astype("string").str.strip()
        keep_cols = [c for c in rt.columns if c not in sample.columns or c in {"transaction_id", "is_fraud"}]
        merged = sample.merge(rt[keep_cols], on="transaction_id", how="left", suffixes=("", "_runtime"))
    else:
        rt = rt.reset_index(drop=True)
        sample2 = sample.reset_index(drop=True)
        merged = pd.concat([sample2, rt.add_suffix("_runtime")], axis=1)
        if "decisao_runtime" in merged.columns and "decisao" not in merged.columns:
            merged["decisao"] = merged["decisao_runtime"]

    if "decisao" not in merged.columns and "decisao_runtime" in merged.columns:
        merged["decisao"] = merged["decisao_runtime"]

    return merged


def compare_policies(sample: pd.DataFrame, runtime_preds: pd.DataFrame, shadow: pd.DataFrame, shadow_pred_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rt_merged = safe_merge_runtime_with_sample(sample, runtime_preds)

    shadow_keep = ["transaction_id", shadow_pred_col]
    for optional in ["lgbm_r4_score", "r4_score", "if_percentile", "se_score", "behavioral_score", "external_any_signal"]:
        if optional in shadow.columns and optional not in shadow_keep:
            shadow_keep.append(optional)

    comp = rt_merged.merge(
        shadow[shadow_keep].drop_duplicates("transaction_id"),
        on="transaction_id",
        how="left",
    )

    comp[shadow_pred_col] = pd.to_numeric(comp[shadow_pred_col], errors="coerce").fillna(0).astype(int)

    y = comp["is_fraud"].astype(int).values
    runtime_flag = flagged_from_runtime(comp).values
    shadow_flag = comp[shadow_pred_col].astype(int).values

    union_flag = ((runtime_flag == 1) | (shadow_flag == 1)).astype(int)
    inter_flag = ((runtime_flag == 1) & (shadow_flag == 1)).astype(int)

    rows = []
    add_metric_row(rows, "RUNTIME_BASELINE", y, runtime_flag)
    add_metric_row(rows, "SHADOW_EXP012D", y, shadow_flag)
    add_metric_row(rows, "UNION_BASELINE_OR_SHADOW", y, union_flag)
    add_metric_row(rows, "INTERSECTION_BASELINE_AND_SHADOW", y, inter_flag)

    metrics_df = pd.DataFrame(rows)

    comp["runtime_flagged"] = runtime_flag
    comp["shadow_exp012d_flagged"] = shadow_flag
    comp["union_flagged"] = union_flag
    comp["intersection_flagged"] = inter_flag

    # Deltas shadow vs runtime.
    comp["shadow_recovers_runtime_fn"] = ((comp["is_fraud"] == 1) & (comp["runtime_flagged"] == 0) & (comp["shadow_exp012d_flagged"] == 1)).astype(int)
    comp["shadow_adds_fp_vs_runtime"] = ((comp["is_fraud"] == 0) & (comp["runtime_flagged"] == 0) & (comp["shadow_exp012d_flagged"] == 1)).astype(int)
    comp["shadow_loses_runtime_tp"] = ((comp["is_fraud"] == 1) & (comp["runtime_flagged"] == 1) & (comp["shadow_exp012d_flagged"] == 0)).astype(int)
    comp["shadow_removes_runtime_fp"] = ((comp["is_fraud"] == 0) & (comp["runtime_flagged"] == 1) & (comp["shadow_exp012d_flagged"] == 0)).astype(int)

    delta_summary = pd.DataFrame([{
        "shadow_recovers_runtime_fn": int(comp["shadow_recovers_runtime_fn"].sum()),
        "shadow_adds_fp_vs_runtime": int(comp["shadow_adds_fp_vs_runtime"].sum()),
        "shadow_loses_runtime_tp": int(comp["shadow_loses_runtime_tp"].sum()),
        "shadow_removes_runtime_fp": int(comp["shadow_removes_runtime_fp"].sum()),
        "runtime_flagged_total": int(comp["runtime_flagged"].sum()),
        "shadow_flagged_total": int(comp["shadow_exp012d_flagged"].sum()),
        "union_flagged_total": int(comp["union_flagged"].sum()),
        "intersection_flagged_total": int(comp["intersection_flagged"].sum()),
    }])

    return metrics_df, delta_summary, comp


def make_recommendation(summary: dict[str, Any], metrics_df: pd.DataFrame, delta_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-012E — E2E Runtime Shadow")
    lines.append("")
    lines.append("## Objetivo")
    lines.append("Comparar a cascata EXP-012D em modo shadow contra o PipelineOrquestrador/DecisionEngine real.")
    lines.append("")
    lines.append("## Métricas")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Deltas SHADOW vs RUNTIME")
    lines.append(delta_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Decisão sugerida")

    m = {row["policy"]: row for _, row in metrics_df.iterrows()}
    runtime = m.get("RUNTIME_BASELINE")
    shadow = m.get("SHADOW_EXP012D")

    if runtime is None or shadow is None:
        lines.append("INCONCLUSIVO: métricas principais ausentes.")
    else:
        shadow_recall_ok = float(shadow["recall"]) >= 0.95
        shadow_fp_better_than_runtime = int(shadow["fp"]) <= int(runtime["fp"])
        shadow_tp_not_lower = int(shadow["tp"]) >= int(runtime["tp"])

        if shadow_recall_ok and shadow_fp_better_than_runtime and shadow_tp_not_lower:
            lines.append("APROVAR_CANDIDATO_PARA_PATCH_SHADOW_CONFIGURAVEL.")
        elif shadow_recall_ok:
            lines.append("APROVAR_APENAS_COMO_SHADOW_DIAGNOSTICO: recall alto, mas ainda não melhora FP/TP contra runtime baseline.")
        else:
            lines.append("NÃO APROVAR: shadow não manteve recall mínimo no E2E runtime.")

    lines.append("")
    lines.append("## Próximo passo")
    lines.append("Se aprovado, criar EXP-012F patch configurável no DecisionEngine/PipelineOrquestrador; se não, ajustar política externa antes do patch.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--shadow-predictions", default=str(DEFAULT_SHADOW_HOLDOUT_SAFE))
    parser.add_argument("--exp012d-summary", default=str(DEFAULT_EXP012D_SUMMARY))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--split", default="HOLDOUT_LABEL_SAFE", choices=["HOLDOUT_LABEL_SAFE", "HOLDOUT_FULL", "HOLDOUT", "VALIDATION", "TRAIN"])
    parser.add_argument("--sample", type=int, default=None, help="Sample estratificado opcional.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-runtime", action="store_true", help="Não roda PipelineOrquestrador; usa apenas shadow e gera diagnóstico parcial.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    shadow_path = Path(args.shadow_predictions)
    exp012d_summary_path = Path(args.exp012d_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXP-012E — E2E Runtime Shadow da Cascata EXP-012D")
    print("=" * 80)
    print(f"Input:       {input_path}")
    print(f"Shadow:      {shadow_path}")
    print(f"Split:       {args.split}")
    print(f"Sample:      {args.sample}")
    print(f"Workers:     {args.workers}")
    print(f"Output dir:  {output_dir}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    split_df = select_split(df, args.split)
    sample = stratified_sample(split_df, args.sample, args.seed)

    log.info(
        "Sample selecionado: %d tx | fraudes=%d | normais=%d | %s -> %s",
        len(sample),
        int(sample["is_fraud"].sum()),
        int((sample["is_fraud"] == 0).sum()),
        sample["data_pix"].min().date(),
        sample["data_pix"].max().date(),
    )

    shadow, shadow_pred_col = load_shadow_predictions(shadow_path)

    # Garantir que shadow cobre o sample.
    missing_shadow = sorted(set(sample["transaction_id"]) - set(shadow["transaction_id"]))
    if missing_shadow:
        raise RuntimeError(
            f"Shadow EXP-012D não cobre {len(missing_shadow)} transações do sample. "
            "Para este script, use por padrão HOLDOUT_LABEL_SAFE ou gere shadow para o split escolhido."
        )

    runtime_preds = None
    runtime_error = None

    if not args.skip_runtime:
        try:
            runtime_preds = run_pipeline_runtime(sample, workers=args.workers)
            runtime_preds.to_csv(output_dir / "01_runtime_predictions.csv", index=False)
        except Exception as exc:
            runtime_error = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            dump_json(runtime_error, output_dir / "runtime_error.json")
            raise

    if args.skip_runtime:
        # Diagnóstico parcial: shadow sozinho.
        joined = sample.merge(shadow[["transaction_id", shadow_pred_col]], on="transaction_id", how="left")
        joined[shadow_pred_col] = pd.to_numeric(joined[shadow_pred_col], errors="coerce").fillna(0).astype(int)
        metrics_df = pd.DataFrame([{"policy": "SHADOW_EXP012D", **compute_metrics(joined["is_fraud"].values, joined[shadow_pred_col].values)}])
        delta_df = pd.DataFrame()
        comparison_df = joined
    else:
        metrics_df, delta_df, comparison_df = compare_policies(sample, runtime_preds, shadow, shadow_pred_col)

    metrics_df.to_csv(output_dir / "02_policy_metrics.csv", index=False)
    delta_df.to_csv(output_dir / "03_delta_shadow_vs_runtime.csv", index=False)
    comparison_df.to_csv(output_dir / "04_comparison_by_transaction.csv", index=False)

    # Auditorias úteis.
    if not delta_df.empty:
        comparison_df[comparison_df.get("shadow_recovers_runtime_fn", 0) == 1].to_csv(output_dir / "05_shadow_recovers_runtime_fn.csv", index=False)
        comparison_df[comparison_df.get("shadow_adds_fp_vs_runtime", 0) == 1].to_csv(output_dir / "06_shadow_adds_fp_vs_runtime.csv", index=False)
        comparison_df[comparison_df.get("shadow_loses_runtime_tp", 0) == 1].to_csv(output_dir / "07_shadow_loses_runtime_tp.csv", index=False)
        comparison_df[comparison_df.get("shadow_removes_runtime_fp", 0) == 1].to_csv(output_dir / "08_shadow_removes_runtime_fp.csv", index=False)

    exp012d_summary = None
    if exp012d_summary_path.exists():
        try:
            exp012d_summary = json.loads(exp012d_summary_path.read_text(encoding="utf-8"))
        except Exception:
            exp012d_summary = None

    summary = {
        "experiment": "EXP-012E",
        "status": "DONE",
        "split": args.split,
        "sample_requested": args.sample,
        "seed": args.seed,
        "workers": args.workers,
        "n_sample": int(len(sample)),
        "n_fraud": int(sample["is_fraud"].sum()),
        "n_normal": int((sample["is_fraud"] == 0).sum()),
        "shadow_pred_col": shadow_pred_col,
        "runtime_executed": not args.skip_runtime,
        "runtime_error": runtime_error,
        "metrics": metrics_df.to_dict(orient="records"),
        "deltas": delta_df.to_dict(orient="records"),
        "exp012d_reference": {
            "objective_status": exp012d_summary.get("objective_status") if isinstance(exp012d_summary, dict) else None,
            "champion_candidate_id": exp012d_summary.get("champion_candidate_id") if isinstance(exp012d_summary, dict) else None,
            "champion_family": exp012d_summary.get("champion_family") if isinstance(exp012d_summary, dict) else None,
            "champion_holdout_label_safe": exp012d_summary.get("champion_holdout_label_safe") if isinstance(exp012d_summary, dict) else None,
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    recommendation = make_recommendation(summary, metrics_df, delta_df)
    (output_dir / "09_recommendation.md").write_text(recommendation, encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXP-012E CONCLUÍDO")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nArtefatos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_runtime_predictions.csv",
        output_dir / "02_policy_metrics.csv",
        output_dir / "03_delta_shadow_vs_runtime.csv",
        output_dir / "04_comparison_by_transaction.csv",
        output_dir / "09_recommendation.md",
    ]:
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
