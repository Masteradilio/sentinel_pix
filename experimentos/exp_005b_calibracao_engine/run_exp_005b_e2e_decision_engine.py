"""
experimentos/exp_005b_e2e_decision_engine/run_exp_005b_e2e_decision_engine.py

EXP-005B-E2E — DecisionEngine real com LGBM v6.2 candidato.

Objetivo:
  Avaliar o LGBM_C_SPW_2_0X do EXP-005A dentro do PipelineOrquestrador/
  DecisionEngine real, usando swap temporário de artefatos.

Não promove nada permanentemente:
  - Faz backup dos artefatos LGBM e scoring_config.
  - Copia temporariamente modelo/features candidatos.
  - Aplica temporariamente thresholds por config.
  - Restaura tudo ao final de cada execução.

Candidatos principais:
  - 0.20: candidato operacional principal
  - 0.15: intermediário forte
  - 0.30: conservador
  - 0.07: recall-oriented comparativo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime
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

BACKEND_DIR = PROJECT_ROOT / "backend"
ARTEFATOS_DIR = BACKEND_DIR / "artefatos"
CANDIDATE_DIR = BACKEND_DIR / "artefatos_candidatos" / "exp_005a_lgbm_v6_2_recall"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-005B-E2E"

SCORING_PATH = ARTEFATOS_DIR / "scoring_config.json"
CANDIDATE_MODEL = CANDIDATE_DIR / "lgbm_v6_2_recall_candidate.joblib"
CANDIDATE_FEATURES = CANDIDATE_DIR / "lgbm_features_v6_2.json"


from experimentos.utils_experimentos import (  # noqa: E402
    compute_metrics,
    get_logger,
    load_dataset,
    process_dataframe_via_orquestrador,
    safe_json_dump,
    stratified_sample,
)


# =========================================================
# LOGGING
# =========================================================

EXP_ID = "EXP-005B-E2E"
logger = get_logger(EXP_ID)


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =========================================================
# BASELINE
# =========================================================

BASELINE_FASE2_REF = {
    "TP": 346,
    "FP": 15,
    "FN": 9,
    "Precision": 0.958449,
    "Recall": 0.974648,
    "F1": 0.9665,
    "FPR": 0.002657,
}


# =========================================================
# JSON HELPERS
# =========================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [safe_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [safe_json(x) for x in obj.tolist()]
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
    return obj


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_json(path: Path) -> dict[str, Any]:
    try:
        obj = read_json(path)
        return {"path": str(path), "valid_json": True, "type": type(obj).__name__}
    except Exception as exc:
        return {"path": str(path), "valid_json": False, "error": str(exc)}


# =========================================================
# SCORING PATCH TEMPORARIO
# =========================================================

def set_thresholds_in_faixas(config: dict[str, Any], threshold_confirmar: float, threshold_bloquear: float) -> None:
    faixas = config.get("faixas_decisao")

    if isinstance(faixas, dict):
        for key, value in faixas.items():
            if not isinstance(value, dict):
                continue

            k = str(key).lower()

            if "aprovar" in k:
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in value:
                        value[field] = threshold_confirmar

            if "confirmar" in k:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in value:
                        value[field] = threshold_confirmar
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in value:
                        value[field] = threshold_bloquear

            if "bloquear" in k:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in value:
                        value[field] = threshold_bloquear

    elif isinstance(faixas, list):
        for item in faixas:
            if not isinstance(item, dict):
                continue

            label = " ".join(
                str(item.get(k, ""))
                for k in ["decisao", "nome", "label", "categoria", "tipo"]
            ).lower()

            if "aprovar" in label:
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in item:
                        item[field] = threshold_confirmar

            if "confirmar" in label:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in item:
                        item[field] = threshold_confirmar
                for field in ["max", "max_score", "score_max", "threshold_max"]:
                    if field in item:
                        item[field] = threshold_bloquear

            if "bloquear" in label:
                for field in ["min", "min_score", "score_min", "threshold", "threshold_min"]:
                    if field in item:
                        item[field] = threshold_bloquear


def patch_scoring_config_for_candidate(config: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    threshold_confirmar = float(cfg["threshold_confirmar"])
    threshold_bloquear = float(cfg["threshold_bloquear"])
    lgbm_effective_threshold = float(cfg["lgbm_effective_threshold"])
    lgbm_guard_threshold = float(cfg["lgbm_guard_threshold"])

    config["threshold_confirmar"] = threshold_confirmar
    config["threshold_bloquear"] = threshold_bloquear
    set_thresholds_in_faixas(config, threshold_confirmar, threshold_bloquear)

    config["lgbm_effective_threshold"] = lgbm_effective_threshold
    config["lgbm_guard_enabled"] = True
    config["lgbm_guard_threshold"] = lgbm_guard_threshold

    config["se_pattern_residual_enabled"] = False
    config["exp003_residual_confirm_enabled"] = False

    # FASE 1 permanente; só terá efeito se o DecisionEngine já tiver esse patch.
    config["guard_exception_alto_valor_se_beh_enabled"] = True
    config["guard_exception_alto_valor_min"] = 15000.0
    config["guard_exception_alto_valor_rel_max"] = 12.0
    config["guard_exception_alto_valor_if_min"] = 0.985
    config["guard_exception_alto_valor_lgbm_min"] = 0.01
    config["guard_exception_alto_valor_age_min"] = 18
    config["guard_exception_alto_valor_age_max"] = 90
    config["guard_exception_alto_valor_require_first_receiver"] = True
    config["guard_exception_alto_valor_require_pf"] = True

    config["_metadata_exp005b_e2e_temp"] = {
        "patched_at": datetime.now().isoformat(timespec="seconds"),
        "config_id": cfg["id"],
        "note": "Patch temporario para EXP-005B-E2E; restaurado automaticamente apos execucao.",
    }

    return config


# =========================================================
# ARTIFACT SWAP
# =========================================================

def find_current_lgbm_model_file() -> Path:
    candidates = []
    for pattern in ["*lgbm*.joblib", "*lightgbm*.joblib", "*model*.joblib"]:
        candidates.extend(ARTEFATOS_DIR.glob(pattern))

    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        raise FileNotFoundError(f"Nenhum modelo LGBM encontrado em {ARTEFATOS_DIR}")

    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if "lgbm" in p.name.lower() else 1,
            len(p.name),
            p.name.lower(),
        ),
    )
    return candidates[0]


@contextmanager
def temporary_candidate_runtime(cfg: dict[str, Any]):
    """
    Troca temporariamente:
      - modelo LGBM atual pelo candidato
      - lgbm_features.json atual pelo candidato
      - scoring_config.json com thresholds da configuração

    Restaura tudo no finally.
    """
    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(f"Modelo candidato não encontrado: {CANDIDATE_MODEL}")

    if not CANDIDATE_FEATURES.exists():
        raise FileNotFoundError(f"Features candidatas não encontradas: {CANDIDATE_FEATURES}")

    current_model = find_current_lgbm_model_file()
    current_features = ARTEFATOS_DIR / "lgbm_features.json"

    if not current_features.exists():
        raise FileNotFoundError(f"lgbm_features.json atual não encontrado: {current_features}")

    backup_dir = ARTEFATOS_DIR.parent / f"_backup_exp005b_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_model = backup_dir / current_model.name
    backup_features = backup_dir / "lgbm_features.json"
    backup_scoring = backup_dir / "scoring_config.json"

    try:
        shutil.copy2(current_model, backup_model)
        shutil.copy2(current_features, backup_features)
        shutil.copy2(SCORING_PATH, backup_scoring)

        scoring = read_json(SCORING_PATH)
        scoring = patch_scoring_config_for_candidate(scoring, cfg)
        write_json(SCORING_PATH, scoring)

        shutil.copy2(CANDIDATE_MODEL, current_model)
        shutil.copy2(CANDIDATE_FEATURES, current_features)

        yield

    finally:
        try:
            shutil.copy2(backup_model, current_model)
            shutil.copy2(backup_features, current_features)
            shutil.copy2(backup_scoring, SCORING_PATH)
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)


# =========================================================
# ENGINE OVERRIDES
# =========================================================

def filter_engine_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Remove chaves não suportadas pelo EngineConfig atual.
    """
    try:
        from core.decision_engine import EngineConfig

        valid_fields = set(getattr(EngineConfig, "__dataclass_fields__", {}).keys())
        if not valid_fields:
            return overrides

        removed = sorted([k for k in overrides if k not in valid_fields])
        if removed:
            logger.warning("Overrides não suportados removidos: %s", removed)

        return {k: v for k, v in overrides.items() if k in valid_fields}

    except Exception as exc:
        logger.warning("Não foi possível inspecionar EngineConfig: %s", exc)
        return overrides


def engine_overrides_for_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = {
        "threshold_confirmar": float(cfg["threshold_confirmar"]),
        "threshold_bloquear": float(cfg["threshold_bloquear"]),
        "lgbm_guard_enabled": True,
        "lgbm_guard_threshold": float(cfg["lgbm_guard_threshold"]),
        "se_pattern_residual_enabled": False,
        "exp003_residual_confirm_enabled": False,
        # Só será mantido se existir no EngineConfig.
        "lgbm_effective_threshold": float(cfg["lgbm_effective_threshold"]),
        "guard_exception_alto_valor_se_beh_enabled": True,
    }
    return filter_engine_overrides(raw)


# =========================================================
# METRICS
# =========================================================

def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(["CONFIRMAR", "BLOQUEAR"])


def evaluate_preds(preds: pd.DataFrame, label: str, cfg_id: str, seed: int) -> dict[str, Any]:
    y_true = preds["is_fraud"].astype(int).values
    y_pred = flagged(preds).astype(int).values
    m = compute_metrics(y_true, y_pred, label).to_dict()

    return {
        "config_id": cfg_id,
        "seed": seed,
        **m,
    }


def compare_to_baseline(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    b_flag = flagged(baseline)
    c_flag = flagged(candidate)
    y = baseline["is_fraud"].astype(int)

    recovered_fn = y.eq(1) & (~b_flag) & c_flag
    added_fp = y.eq(0) & (~b_flag) & c_flag
    lost_tp = y.eq(1) & b_flag & (~c_flag)
    removed_fp = y.eq(0) & b_flag & (~c_flag)

    cols = [
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
        "veto_reason",
        "veto_suppressed_reason",
    ]
    cols = [c for c in cols if c in candidate.columns]

    return {
        "fns_recuperados": int(recovered_fn.sum()),
        "fps_adicionados": int(added_fp.sum()),
        "tps_perdidos": int(lost_tp.sum()),
        "fps_removidos": int(removed_fp.sum()),
        "top_fns_recuperados": candidate.loc[recovered_fn, cols].head(30).to_dict(orient="records"),
        "top_fps_adicionados": candidate.loc[added_fp, cols].head(30).to_dict(orient="records"),
        "top_tps_perdidos": candidate.loc[lost_tp, cols].head(30).to_dict(orient="records"),
        "top_fps_removidos": candidate.loc[removed_fp, cols].head(30).to_dict(orient="records"),
    }


def select_best(results_df: pd.DataFrame) -> str:
    """
    Seleciona pela validação conjunta dos seeds.
    """
    candidates = results_df[results_df["config_id"] != "BASELINE"].copy()

    grouped = []

    for cfg_id, g in candidates.groupby("config_id"):
        if len(g) < 2:
            continue

        row = {
            "config_id": cfg_id,
            "worst_FN": int(g["FN"].max()),
            "worst_FP": int(g["FP"].max()),
            "min_precision": float(g["Precision"].min()),
            "min_recall": float(g["Recall"].min()),
            "min_f1": float(g["F1"].min()),
            "avg_f1": float(g["F1"].mean()),
            "avg_precision": float(g["Precision"].mean()),
            "avg_recall": float(g["Recall"].mean()),
        }
        grouped.append(row)

    summary = pd.DataFrame(grouped)

    if summary.empty:
        return "BASELINE"

    eligible = summary[
        (summary["worst_FN"] <= 5)
        & (summary["worst_FP"] <= 25)
        & (summary["min_precision"] >= 0.935)
        & (summary["min_f1"] >= BASELINE_FASE2_REF["F1"])
    ].copy()

    if eligible.empty:
        eligible = summary[
            (summary["worst_FN"] <= 7)
            & (summary["worst_FP"] <= 25)
            & (summary["min_precision"] >= 0.93)
        ].copy()

    if eligible.empty:
        eligible = summary.copy()

    eligible = eligible.sort_values(
        ["worst_FN", "worst_FP", "avg_f1", "avg_precision"],
        ascending=[True, True, False, False],
    )

    return str(eligible.iloc[0]["config_id"])


# =========================================================
# CONFIGS
# =========================================================

def build_configs(full_grid: bool = False) -> list[dict[str, Any]]:
    if full_grid:
        configs = []
        for lgbm_th in [0.07, 0.10, 0.15, 0.20, 0.30]:
            for final_th in [60.0, 62.0, 65.0]:
                for guard_th in [0.10, 0.20, 0.30]:
                    configs.append({
                        "id": f"CAND_LGBM{str(lgbm_th).replace('.', '')}_FINAL{int(final_th)}_GUARD{str(guard_th).replace('.', '')}",
                        "threshold_confirmar": final_th,
                        "threshold_bloquear": 95.0,
                        "lgbm_effective_threshold": lgbm_th,
                        "lgbm_guard_threshold": guard_th,
                    })
        return configs

    return [
        {
            "id": "CAND_020_MAIN",
            "threshold_confirmar": 62.0,
            "threshold_bloquear": 95.0,
            "lgbm_effective_threshold": 0.20,
            "lgbm_guard_threshold": 0.20,
        },
        {
            "id": "CAND_015_BALANCED",
            "threshold_confirmar": 62.0,
            "threshold_bloquear": 95.0,
            "lgbm_effective_threshold": 0.15,
            "lgbm_guard_threshold": 0.20,
        },
        {
            "id": "CAND_030_CONSERVATIVE",
            "threshold_confirmar": 62.0,
            "threshold_bloquear": 95.0,
            "lgbm_effective_threshold": 0.30,
            "lgbm_guard_threshold": 0.20,
        },
        {
            "id": "CAND_007_RECALL",
            "threshold_confirmar": 62.0,
            "threshold_bloquear": 95.0,
            "lgbm_effective_threshold": 0.07,
            "lgbm_guard_threshold": 0.20,
        },
    ]


# =========================================================
# REPORT
# =========================================================

def write_conclusion(path: Path, results_df: pd.DataFrame, best_id: str, preflight: dict[str, Any]) -> None:
    lines = [
        "# EXP-005B-E2E — DecisionEngine real",
        "",
        f"- Vencedor: `{best_id}`",
        f"- scoring_config válido: `{preflight['scoring_config']['valid_json']}`",
        "",
        "## Resultados",
        "",
        "| Config | Seed | TP | FP | FN | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in results_df.sort_values(["config_id", "seed"]).iterrows():
        lines.append(
            f"| `{r['config_id']}` | {int(r['seed'])} | {int(r['TP'])} | {int(r['FP'])} | {int(r['FN'])} | "
            f"{float(r['Precision']):.4%} | {float(r['Recall']):.4%} | {float(r['F1']):.4f} | {float(r['FPR']):.4%} |"
        )

    lines.extend([
        "",
        "## Interpretação",
        "",
        "Este experimento usa o `PipelineOrquestrador` e `DecisionEngine` reais com swap temporário do LGBM v6.2.",
        "Nada é promovido automaticamente. A promoção depende da análise dos deltas, FNs recuperados, FPs adicionados e estabilidade entre seeds.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-005B-E2E DecisionEngine real")
    parser.add_argument("--sample", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=123)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()

    print_section("EXP-005B-E2E — DecisionEngine real")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preflight = {
        "scoring_config": validate_json(SCORING_PATH),
        "candidate_model_exists": CANDIDATE_MODEL.exists(),
        "candidate_features_exists": CANDIDATE_FEATURES.exists(),
    }
    safe_json_dump(preflight, OUTPUT_DIR / "00_preflight_runtime.json")

    if not preflight["scoring_config"]["valid_json"]:
        raise RuntimeError(
            f"scoring_config.json inválido. Rode primeiro scripts/patch_scoring_config_fase2.py. "
            f"Erro: {preflight['scoring_config'].get('error')}"
        )

    if not preflight["candidate_model_exists"] or not preflight["candidate_features_exists"]:
        raise RuntimeError("Artefatos candidatos do EXP-005A não encontrados.")

    print_section("1. Carregar dataset e samples")

    df_full = load_dataset()
    sample_main = stratified_sample(df_full, n=args.sample, seed=args.seed, logger=logger)

    seeds = [(args.seed, sample_main)]

    if not args.skip_validation:
        sample_val = stratified_sample(df_full, n=args.sample, seed=args.validation_seed, logger=logger)
        seeds.append((args.validation_seed, sample_val))

    print_section("2. Rodar baseline atual")

    rows = []
    baseline_predictions: dict[int, pd.DataFrame] = {}

    baseline_overrides = filter_engine_overrides({
        "threshold_confirmar": 62.0,
        "threshold_bloquear": 95.0,
        "lgbm_guard_enabled": True,
        "lgbm_guard_threshold": 0.30,
        "se_pattern_residual_enabled": False,
        "exp003_residual_confirm_enabled": False,
    })

    for seed, sample in seeds:
        logger.info("Baseline seed=%s", seed)
        preds = process_dataframe_via_orquestrador(
            sample,
            workers=args.workers,
            logger=logger,
            engine_config_overrides=baseline_overrides,
        )
        baseline_predictions[seed] = preds
        rows.append(evaluate_preds(preds, "BASELINE", "BASELINE", seed))

    print_section("3. Rodar configs candidatas com LGBM v6.2")

    configs = build_configs(full_grid=args.full_grid)
    comparisons: dict[str, Any] = {}

    for cfg in configs:
        cfg_id = cfg["id"]
        logger.info("Config candidata: %s", cfg_id)

        with temporary_candidate_runtime(cfg):
            overrides = engine_overrides_for_cfg(cfg)

            for seed, sample in seeds:
                logger.info("Rodando %s seed=%s", cfg_id, seed)

                preds = process_dataframe_via_orquestrador(
                    sample,
                    workers=args.workers,
                    logger=logger,
                    engine_config_overrides=overrides,
                )

                rows.append(evaluate_preds(preds, cfg_id, cfg_id, seed))
                comparisons[f"{cfg_id}__seed_{seed}"] = compare_to_baseline(
                    baseline_predictions[seed],
                    preds,
                )

    results_df = pd.DataFrame(rows)
    results_path = OUTPUT_DIR / "01_e2e_grid_decision_engine.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    print_section("4. Selecionar melhor config")

    best_id = select_best(results_df)
    logger.info("Melhor config E2E: %s", best_id)

    safe_json_dump(
        {
            "best_config_id": best_id,
            "results": results_df.to_dict(orient="records"),
            "comparisons": comparisons,
        },
        OUTPUT_DIR / "02_best_config_e2e.json",
    )

    safe_json_dump(
        {
            "best_config_id": best_id,
            "comparisons": {
                k: v for k, v in comparisons.items()
                if k.startswith(f"{best_id}__")
            },
        },
        OUTPUT_DIR / "03_delta_fp_fn_best_e2e.json",
    )

    validation_rows = results_df[results_df["seed"].eq(args.validation_seed)].to_dict(orient="records")
    safe_json_dump(
        {
            "validation_seed": args.validation_seed,
            "best_config_id": best_id,
            "rows": validation_rows,
        },
        OUTPUT_DIR / "04_validacao_cruzada_e2e.json",
    )

    write_conclusion(
        OUTPUT_DIR / "05_conclusao_executiva.md",
        results_df=results_df,
        best_id=best_id,
        preflight=preflight,
    )

    logger.info("============================================================")
    logger.info("EXP-005B-E2E concluído")
    logger.info("Melhor config: %s", best_id)
    logger.info("Artefatos em: %s", OUTPUT_DIR)
    logger.info("Tempo total: %.1fs", time.perf_counter() - t0)
    logger.info("============================================================")


if __name__ == "__main__":
    main()