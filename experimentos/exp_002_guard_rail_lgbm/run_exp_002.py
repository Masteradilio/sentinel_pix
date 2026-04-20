from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimentos.utils_experimentos import (
    compute_metrics,
    get_experiment_output_dir,
    get_logger,
    load_dataset,
    print_section,
    process_dataframe_via_orquestrador,
    safe_json_dump,
    stratified_sample,
)

EXP_ID = "EXP-002"
CONFIG_PATH = EXP_DIR / "config_variantes.json"
logger = get_logger("EXP-002")


def _flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(["CONFIRMAR", "BLOQUEAR"])


def _evaluate_predictions(preds: pd.DataFrame, label: str, variant_id: str) -> dict[str, Any]:
    y_true = preds["is_fraud"].astype(int).values
    y_pred = _flagged(preds).astype(int).values
    metrics = compute_metrics(y_true, y_pred, label)
    suppressed = preds["veto_suppressed_reason"].fillna("").astype(str) if "veto_suppressed_reason" in preds else pd.Series("", index=preds.index)
    return {
        "variante_id": variant_id,
        "label": label,
        **metrics.to_dict(),
        "suppressed_count": int((suppressed != "").sum()),
        "suppressed_rate": round(float((suppressed != "").mean()), 6),
    }


def _compare_to_baseline(baseline: pd.DataFrame, variant: pd.DataFrame) -> dict[str, Any]:
    baseline_flagged = _flagged(baseline)
    variant_flagged = _flagged(variant)
    y_true = baseline["is_fraud"].astype(int)

    removed_fp_mask = (y_true == 0) & baseline_flagged & (~variant_flagged)
    lost_tp_mask = (y_true == 1) & baseline_flagged & (~variant_flagged)
    recovered_fn_mask = (y_true == 1) & (~baseline_flagged) & variant_flagged
    added_fp_mask = (y_true == 0) & (~baseline_flagged) & variant_flagged

    cols = [
        "transaction_id",
        "customer_id",
        "vl_pix",
        "nr_idade",
        "lgbm_raw",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
        "veto_reason",
        "veto_suppressed_reason",
    ]
    cols = [c for c in cols if c in baseline.columns or c in variant.columns]

    def _top(df: pd.DataFrame, sort_col: str, n: int = 10) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return df.sort_values(sort_col, ascending=False)[cols].head(n).to_dict(orient="records")

    return {
        "fps_removidos": {
            "total": int(removed_fp_mask.sum()),
            "top_por_valor": _top(variant[removed_fp_mask].copy(), "vl_pix"),
        },
        "tps_perdidos": {
            "total": int(lost_tp_mask.sum()),
            "top_por_valor": _top(variant[lost_tp_mask].copy(), "vl_pix"),
        },
        "fns_recuperados": {
            "total": int(recovered_fn_mask.sum()),
            "top_por_valor": _top(variant[recovered_fn_mask].copy(), "vl_pix"),
        },
        "fps_novos": {
            "total": int(added_fp_mask.sum()),
            "top_por_valor": _top(variant[added_fp_mask].copy(), "vl_pix"),
        },
    }


def _suppression_analysis(preds: pd.DataFrame) -> dict[str, Any]:
    suppressed = preds[preds["veto_suppressed_reason"].fillna("").astype(str) != ""].copy()
    if suppressed.empty:
        return {"total": 0, "por_reason": {}, "top_por_valor": []}
    return {
        "total": int(len(suppressed)),
        "por_reason": suppressed["veto_suppressed_reason"].value_counts().to_dict(),
        "top_por_valor": suppressed.sort_values("vl_pix", ascending=False).head(20).to_dict(orient="records"),
    }


def _pick_winner(results_df: pd.DataFrame) -> str:
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    eligible = results_df[
        (results_df["variante_id"] != "BASELINE")
        & (results_df["FP"] <= baseline["FP"] - 3)
        & (results_df["TP"] >= baseline["TP"] - 2)
        & (results_df["F1"] >= baseline["F1"] - 0.005)
        & (results_df["Recall"] >= 0.92)
    ].copy()
    if eligible.empty:
        return results_df.sort_values(["F1", "Recall", "FP"], ascending=[False, False, True]).iloc[0]["variante_id"]
    return eligible.sort_values(["F1", "Recall", "FP"], ascending=[False, False, True]).iloc[0]["variante_id"]


def _write_conclusion(path: Path, results_df: pd.DataFrame, winner_id: str, validation: dict[str, Any]) -> None:
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    winner = results_df.loc[results_df["variante_id"] == winner_id].iloc[0]
    lines = [
        f"# {EXP_ID} - Conclusao Executiva",
        "",
        f"- Vencedor: `{winner_id}`",
        f"- Baseline: TP={int(baseline['TP'])}, FP={int(baseline['FP'])}, FN={int(baseline['FN'])}, F1={baseline['F1']:.4f}, Recall={baseline['Recall']:.4%}, Precision={baseline['Precision']:.4%}",
        f"- Vencedor: TP={int(winner['TP'])}, FP={int(winner['FP'])}, FN={int(winner['FN'])}, F1={winner['F1']:.4f}, Recall={winner['Recall']:.4%}, Precision={winner['Precision']:.4%}",
        f"- Delta: TP={int(winner['TP'] - baseline['TP']):+d}, FP={int(winner['FP'] - baseline['FP']):+d}, FN={int(winner['FN'] - baseline['FN']):+d}, F1={winner['F1'] - baseline['F1']:+.4f}",
        f"- Validacao seed=123: TP={validation['TP']}, FP={validation['FP']}, FN={validation['FN']}, F1={validation['F1']:.4f}, Recall={validation['Recall']:.4%}, Precision={validation['Precision']:.4%}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=123)
    args = parser.parse_args()

    t0 = time.perf_counter()
    output_dir = get_experiment_output_dir(EXP_ID)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    print_section(f"{EXP_ID} - Guard Rail LGBM")
    df = load_dataset()
    sample_df = stratified_sample(df, n=args.sample, seed=args.seed, logger=logger)

    predictions: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []

    baseline_cfg = cfg["baseline"]
    baseline_preds = process_dataframe_via_orquestrador(
        sample_df,
        workers=args.workers,
        logger=logger,
        engine_config_overrides=baseline_cfg["engine_config_overrides"],
    )
    predictions["BASELINE"] = baseline_preds
    rows.append(_evaluate_predictions(baseline_preds, baseline_cfg["label"], "BASELINE"))

    suppression_summary: dict[str, Any] = {}
    comparison_summary: dict[str, Any] = {}

    for variant_cfg in cfg["variantes"]:
        variant_id = variant_cfg["id"]
        preds = process_dataframe_via_orquestrador(
            sample_df,
            workers=args.workers,
            logger=logger,
            engine_config_overrides=variant_cfg["engine_config_overrides"],
        )
        predictions[variant_id] = preds
        rows.append(_evaluate_predictions(preds, variant_cfg["label"], variant_id))
        suppression_summary[variant_id] = _suppression_analysis(preds)
        comparison_summary[variant_id] = _compare_to_baseline(baseline_preds, preds)

    results_df = pd.DataFrame(rows)
    baseline_row = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    results_df["delta_TP"] = results_df["TP"] - baseline_row["TP"]
    results_df["delta_FP"] = results_df["FP"] - baseline_row["FP"]
    results_df["delta_FN"] = results_df["FN"] - baseline_row["FN"]
    results_df["delta_F1"] = (results_df["F1"] - baseline_row["F1"]).round(6)
    results_df["delta_Recall"] = (results_df["Recall"] - baseline_row["Recall"]).round(6)
    results_df["delta_Precision"] = (results_df["Precision"] - baseline_row["Precision"]).round(6)
    results_df.to_csv(output_dir / "01_tabela_comparativa.csv", index=False)

    safe_json_dump(suppression_summary, output_dir / "02_analise_supressoes.json")
    safe_json_dump(comparison_summary, output_dir / "03_analise_fp_fn.json")

    winner_id = _pick_winner(results_df)
    validation_sample = stratified_sample(df, n=args.sample, seed=args.validation_seed, logger=logger)
    winner_cfg = baseline_cfg if winner_id == "BASELINE" else next(v for v in cfg["variantes"] if v["id"] == winner_id)
    validation_preds = process_dataframe_via_orquestrador(
        validation_sample,
        workers=args.workers,
        logger=logger,
        engine_config_overrides=winner_cfg["engine_config_overrides"],
    )
    validation_metrics = _evaluate_predictions(validation_preds, winner_cfg["label"], winner_id)
    validation_payload = {
        "winner_id": winner_id,
        "seed": args.validation_seed,
        "sample": args.sample,
        "metrics": validation_metrics,
        "suppression": _suppression_analysis(validation_preds),
    }
    safe_json_dump(validation_payload, output_dir / "04_validacao_cruzada.json")
    _write_conclusion(output_dir / "05_conclusao_executiva.md", results_df, winner_id, validation_metrics)

    logger.info("Resultados salvos em %s", output_dir)
    logger.info("Tempo total: %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
