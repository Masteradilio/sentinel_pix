"""
EXP-006F — Quick-E2E C1 Near-Threshold

Objetivo:
  Consolidar o resultado positivo do EXP-006E antes do EXP-007A.

Este experimento:
  - Não roda grid.
  - Não testa múltiplas regras.
  - Não altera scoring_config.json.
  - Não altera decision_engine.py.
  - Usa baseline já salvo, se disponível.
  - Opcionalmente roda baseline real com PipelineOrquestrador.
  - Aplica C1 como patch temporário/overlay pós-engine.
  - Salva métricas e decisão de promoção.

C1:
  decisao == APROVAR
  first_receiver_flag == 1
  pix_key_random_flag == 0
  qt_tempo_relacionamento_mes <= 12
  100 <= vl_pix < 500
  0.06 <= lgbm_raw < 0.10
  60 <= score_final < 62
  se_score <= 0
  beh_score <= 0

Uso recomendado:
  python experimentos\\exp_006f_quick_e2e_c1\\run_exp_006f_quick_e2e_c1.py --from-cache

Opcional, se quiser reprocessar baseline real:
  python experimentos\\exp_006f_quick_e2e_c1\\run_exp_006f_quick_e2e_c1.py --run-baseline --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

EXP_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "experimentos").exists():
            return p
    return start.parent.parent


PROJECT_ROOT = find_project_root(EXP_DIR)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

INPUT_CACHE_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006F-C1"

DECISOES_POSITIVAS = {"CONFIRMAR", "BLOQUEAR"}


# Só importamos utils se necessário.
def lazy_import_utils():
    from experimentos.utils_experimentos import (
        compute_metrics as utils_compute_metrics,
        get_logger,
        load_dataset,
        process_dataframe_via_orquestrador,
        safe_json_dump,
        stratified_sample,
    )

    return {
        "compute_metrics": utils_compute_metrics,
        "get_logger": get_logger,
        "load_dataset": load_dataset,
        "process_dataframe_via_orquestrador": process_dataframe_via_orquestrador,
        "safe_json_dump": safe_json_dump,
        "stratified_sample": stratified_sample,
    }


# =========================================================
# HELPERS
# =========================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [safe_json(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(DECISOES_POSITIVAS)


def compute_metrics_local(df: pd.DataFrame, label: str) -> dict[str, Any]:
    y = df["is_fraud"].astype(int)
    pred = flagged(df).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)

    return {
        "label": label,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
        "FPR": round(fpr, 8),
    }


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "is_fraud",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["is_fraud"] = out["is_fraud"].astype(int)
    out["decisao"] = out["decisao"].astype(str)

    return out


def ensure_columns(df: pd.DataFrame, source: Path) -> None:
    required = [
        "is_fraud",
        "decisao",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "se_score",
        "beh_score",
        "score_final",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Arquivo {source} sem colunas obrigatórias: {missing}")


# =========================================================
# C1 RULE
# =========================================================

RULE_ID = "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER"

RULE_SPEC = {
    "rule_id": RULE_ID,
    "status": "candidate_runtime_overlay",
    "source": "EXP-006E",
    "action": "APROVAR_TO_CONFIRMAR",
    "conditions": {
        "decisao": "APROVAR",
        "first_receiver_flag": 1,
        "pix_key_random_flag": 0,
        "qt_tempo_relacionamento_mes_max": 12,
        "vl_pix_min_inclusive": 100.0,
        "vl_pix_max_exclusive": 500.0,
        "lgbm_raw_min_inclusive": 0.06,
        "lgbm_raw_max_exclusive": 0.10,
        "score_final_min_inclusive": 60.0,
        "score_final_max_exclusive": 62.0,
        "se_score_max": 0.0,
        "beh_score_max": 0.0,
    },
}


def c1_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["decisao"].astype(str).eq("APROVAR")
        & df["first_receiver_flag"].astype(int).eq(1)
        & df["pix_key_random_flag"].astype(int).eq(0)
        & df["qt_tempo_relacionamento_mes"].le(12)
        & df["vl_pix"].ge(100.0)
        & df["vl_pix"].lt(500.0)
        & df["lgbm_raw"].ge(0.06)
        & df["lgbm_raw"].lt(0.10)
        & df["score_final"].ge(60.0)
        & df["score_final"].lt(62.0)
        & df["se_score"].le(0.0)
        & df["beh_score"].le(0.0)
    )


def apply_c1_overlay(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = c1_mask(out)

    out["exp006f_c1_hit"] = mask
    out["exp006f_original_decisao"] = ""
    out["exp006f_original_score_final"] = np.nan
    out["exp006f_reason"] = ""

    idx = out.index[mask]

    out.loc[idx, "exp006f_original_decisao"] = out.loc[idx, "decisao"]
    out.loc[idx, "exp006f_original_score_final"] = out.loc[idx, "score_final"]

    out.loc[idx, "decisao"] = "CONFIRMAR"
    out.loc[idx, "score_final"] = out.loc[idx, "score_final"].apply(lambda x: max(safe_float(x), 62.0))
    out.loc[idx, "exp006f_reason"] = (
        "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER: "
        "APROVAR->CONFIRMAR | rel<=12, first_receiver=1, pix_random=0, "
        "100<=vl<500, 0.06<=lgbm<0.10, 60<=score<62, SE=0, BEH=0"
    )

    return out


def compare(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    b = flagged(baseline)
    c = flagged(candidate)
    y = baseline["is_fraud"].astype(int)

    recovered_fn = y.eq(1) & (~b) & c
    added_fp = y.eq(0) & (~b) & c
    lost_tp = y.eq(1) & b & (~c)
    removed_fp = y.eq(0) & b & (~c)

    cols = [
        "seed",
        "transaction_id",
        "customer_id",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "lgbm_mapped",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
        "decisao",
        "exp006f_c1_hit",
        "exp006f_original_decisao",
        "exp006f_original_score_final",
        "exp006f_reason",
    ]
    cols = [c for c in cols if c in candidate.columns]

    return {
        "fns_recuperados": int(recovered_fn.sum()),
        "fps_adicionados": int(added_fp.sum()),
        "tps_perdidos": int(lost_tp.sum()),
        "fps_removidos": int(removed_fp.sum()),
        "rule_hits": int(candidate["exp006f_c1_hit"].sum()),
        "top_fns_recuperados": candidate.loc[recovered_fn, cols].to_dict(orient="records"),
        "top_fps_adicionados": candidate.loc[added_fp, cols].to_dict(orient="records"),
        "top_tps_perdidos": candidate.loc[lost_tp, cols].to_dict(orient="records"),
        "top_fps_removidos": candidate.loc[removed_fp, cols].to_dict(orient="records"),
    }


# =========================================================
# BASELINE LOADING / RUNNING
# =========================================================

def load_baseline_from_cache(seed: int) -> pd.DataFrame:
    path = INPUT_CACHE_DIR / f"baseline_predictions_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Baseline cache não encontrado: {path}")

    df = pd.read_csv(path)
    ensure_columns(df, path)
    df = normalize_numeric(df)
    df["seed"] = seed
    df["source_file"] = str(path)

    return df


def run_baseline_real(seed: int, workers: int, sample_size: int = 6000) -> pd.DataFrame:
    utils = lazy_import_utils()
    logger = utils["get_logger"]("EXP-006F-C1")

    df_full = utils["load_dataset"]()
    sample = utils["stratified_sample"](df_full, n=sample_size, seed=seed, logger=logger)

    preds = utils["process_dataframe_via_orquestrador"](
        sample,
        workers=workers,
        logger=logger,
        engine_config_overrides=None,
    )

    preds = normalize_numeric(preds)
    preds["seed"] = seed
    preds["source_file"] = "runtime_baseline"

    return preds


def get_baseline(seed: int, mode: str, workers: int) -> pd.DataFrame:
    if mode == "from_cache":
        return load_baseline_from_cache(seed)

    if mode == "run_baseline":
        return run_baseline_real(seed=seed, workers=workers)

    raise ValueError(f"Modo inválido: {mode}")


# =========================================================
# DECISION
# =========================================================

def decide(metrics_rows: list[dict[str, Any]], deltas: dict[str, Any]) -> dict[str, Any]:
    metrics = pd.DataFrame(metrics_rows)
    seeds = sorted(deltas.keys())

    per_seed_pass = {}

    for seed in seeds:
        d = deltas[seed]
        b = metrics[(metrics["seed"].astype(str) == str(seed)) & (metrics["config"] == "BASELINE")]
        c = metrics[(metrics["seed"].astype(str) == str(seed)) & (metrics["config"] == RULE_ID)]

        f1_ok = False
        if not b.empty and not c.empty:
            f1_ok = float(c.iloc[0]["F1"]) >= float(b.iloc[0]["F1"])

        per_seed_pass[seed] = (
            d["fns_recuperados"] >= 1
            and d["fps_adicionados"] == 0
            and d["tps_perdidos"] == 0
            and f1_ok
        )

    all_pass = all(per_seed_pass.values())

    if all_pass:
        status = "APROVADO_PARA_PATCH_PERMANENTE"
        next_action = (
            "Gerar patch permanente no DecisionEngine/scoring_config para C1, "
            "mantendo flag configurável e desligável."
        )
    else:
        status = "REJEITAR_C1_SEM_PATCH"
        next_action = (
            "Não promover C1. Prosseguir para EXP-007A Meta-Learner Shadow."
        )

    return {
        "status": status,
        "per_seed_pass": per_seed_pass,
        "all_pass": all_pass,
        "next_action": next_action,
    }


def write_recommendation(path: Path, decision: dict[str, Any], metrics_rows: list[dict[str, Any]], deltas: dict[str, Any]) -> None:
    metrics = pd.DataFrame(metrics_rows)

    lines = [
        "# EXP-006F — Quick-E2E C1 Near-Threshold",
        "",
        f"- Status: `{decision['status']}`",
        "",
        "## Métricas",
        "",
        "| Seed | Config | TP | FP | FN | Precision | Recall | F1 | FPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in metrics.sort_values(["seed", "config"]).iterrows():
        lines.append(
            f"| {int(r['seed'])} | `{r['config']}` | {int(r['TP'])} | {int(r['FP'])} | {int(r['FN'])} | "
            f"{float(r['Precision']):.4%} | {float(r['Recall']):.4%} | "
            f"{float(r['F1']):.4f} | {float(r['FPR']):.4%} |"
        )

    lines.extend([
        "",
        "## Delta por seed",
        "",
        "| Seed | FNs recuperados | FPs adicionados | TPs perdidos | FPs removidos | Rule hits |",
        "|---:|---:|---:|---:|---:|---:|",
    ])

    for seed, d in sorted(deltas.items()):
        lines.append(
            f"| {seed} | {d['fns_recuperados']} | {d['fps_adicionados']} | "
            f"{d['tps_perdidos']} | {d['fps_removidos']} | {d['rule_hits']} |"
        )

    lines.extend([
        "",
        "## Decisão",
        "",
        decision["next_action"],
        "",
    ])

    if decision["all_pass"]:
        lines.append("C1 passou no quick-E2E/cached-runtime e pode ser transformado em patch permanente configurável.")
    else:
        lines.append("C1 não passou. Não aplicar patch permanente.")

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-006F Quick-E2E C1")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-cache", action="store_true", help="Usa baseline_predictions já salvos.")
    mode.add_argument("--run-baseline", action="store_true", help="Roda baseline real via PipelineOrquestrador.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_mode = "from_cache" if args.from_cache else "run_baseline"

    print("=" * 72)
    print("EXP-006F — Quick-E2E C1 Near-Threshold")
    print("=" * 72)
    print(f"[INFO] Modo: {run_mode}")

    write_json(
        OUTPUT_DIR / "00_rule_spec.json",
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": run_mode,
            "rule_spec": RULE_SPEC,
        },
    )

    metrics_rows = []
    deltas = {}

    for seed in [42, 123]:
        print()
        print("=" * 72)
        print(f"Seed {seed}")
        print("=" * 72)

        baseline = get_baseline(seed=seed, mode=run_mode, workers=args.workers)

        baseline_path = OUTPUT_DIR / f"01_baseline_seed_{seed}.csv"
        baseline.to_csv(baseline_path, index=False, encoding="utf-8-sig")

        candidate = apply_c1_overlay(baseline)

        candidate_path = OUTPUT_DIR / f"02_candidate_c1_seed_{seed}.csv"
        candidate.to_csv(candidate_path, index=False, encoding="utf-8-sig")

        hits = candidate[candidate["exp006f_c1_hit"]].copy()
        hits_path = OUTPUT_DIR / f"03_rule_hits_seed_{seed}.csv"
        hits.to_csv(hits_path, index=False, encoding="utf-8-sig")

        b_metrics = compute_metrics_local(baseline, f"BASELINE_seed_{seed}")
        c_metrics = compute_metrics_local(candidate, f"{RULE_ID}_seed_{seed}")

        b_metrics["seed"] = seed
        b_metrics["config"] = "BASELINE"

        c_metrics["seed"] = seed
        c_metrics["config"] = RULE_ID

        metrics_rows.append(b_metrics)
        metrics_rows.append(c_metrics)

        delta = compare(baseline, candidate)
        deltas[str(seed)] = delta

        write_json(OUTPUT_DIR / f"04_delta_seed_{seed}.json", delta)

        print(
            f"[OK] Baseline: TP={b_metrics['TP']} FP={b_metrics['FP']} FN={b_metrics['FN']} F1={b_metrics['F1']}"
        )
        print(
            f"[OK] C1:       TP={c_metrics['TP']} FP={c_metrics['FP']} FN={c_metrics['FN']} F1={c_metrics['F1']}"
        )
        print(
            f"[OK] Delta: FN_rec={delta['fns_recuperados']} FP_add={delta['fps_adicionados']} "
            f"TP_lost={delta['tps_perdidos']} hits={delta['rule_hits']}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUT_DIR / "05_metrics_comparison.csv", index=False, encoding="utf-8-sig")

    write_json(OUTPUT_DIR / "06_delta_by_seed.json", deltas)

    decision = decide(metrics_rows, deltas)
    write_json(OUTPUT_DIR / "07_decision.json", decision)

    write_recommendation(
        OUTPUT_DIR / "08_recommendation.md",
        decision=decision,
        metrics_rows=metrics_rows,
        deltas=deltas,
    )

    print()
    print("=" * 72)
    print("[OK] EXP-006F concluído")
    print(f"[OK] Decisão: {decision['status']}")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print(f"[OK] Tempo total: {time.perf_counter() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()