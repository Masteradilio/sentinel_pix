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

EXP_ID = "EXP-003"
CONFIG_PATH = EXP_DIR / "config_variantes.json"
logger = get_logger("EXP-003")
RESIDUAL_CUSTOMER_ID = "32339437172"
PATTERN_NAME = "IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL"


def _flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(["CONFIRMAR", "BLOQUEAR"])


def _metrics(preds: pd.DataFrame, label: str, variant_id: str) -> dict[str, Any]:
    y_true = preds["is_fraud"].astype(int).values
    y_pred = _flagged(preds).astype(int).values
    m = compute_metrics(y_true, y_pred, label)
    pattern_hits = preds["se_worst_pattern"].fillna("").astype(str).eq(PATTERN_NAME).sum() if "se_worst_pattern" in preds else 0
    return {
        "variante_id": variant_id,
        "label": label,
        **m.to_dict(),
        "pattern_hits": int(pattern_hits),
    }


def _residual_analysis(preds: pd.DataFrame) -> dict[str, Any]:
    residual = preds[preds["customer_id"].astype(str) == RESIDUAL_CUSTOMER_ID].copy()
    if residual.empty:
        return {"residual_found": False}
    row = residual.iloc[0]
    return {
        "residual_found": True,
        "customer_id": RESIDUAL_CUSTOMER_ID,
        "decisao": row.get("decisao"),
        "score_final": row.get("score_final"),
        "se_score": row.get("se_score"),
        "beh_score": row.get("beh_score"),
        "if_percentile": row.get("if_percentile"),
        "se_worst_pattern": row.get("se_worst_pattern"),
    }


def _pattern_analysis(preds: pd.DataFrame) -> dict[str, Any]:
    pattern_rows = preds[preds["se_worst_pattern"].fillna("").astype(str) == PATTERN_NAME].copy()
    total = len(pattern_rows)
    fraud = int(pattern_rows["is_fraud"].sum()) if total else 0
    return {
        "total": total,
        "fraud": fraud,
        "precision": round(fraud / total, 6) if total else 0.0,
        "top_por_valor": pattern_rows.sort_values("vl_pix", ascending=False).head(15).to_dict(orient="records") if total else [],
    }


def _winner(df: pd.DataFrame) -> str:
    baseline = df.loc[df["variante_id"] == "BASELINE"].iloc[0]
    eligible = df[
        (df["variante_id"] != "BASELINE")
        & (df["TP"] >= baseline["TP"] + 1)
        & (df["FP"] <= baseline["FP"] + 3)
        & (df["F1"] >= baseline["F1"])
    ].copy()
    if eligible.empty:
        return baseline["variante_id"]
    return eligible.sort_values(["TP", "F1", "FP"], ascending=[False, False, True]).iloc[0]["variante_id"]


def _write_conclusion(path: Path, results_df: pd.DataFrame, residuals: dict[str, Any], winner_id: str, validation: dict[str, Any]) -> None:
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    winner = results_df.loc[results_df["variante_id"] == winner_id].iloc[0]
    residual = residuals.get(winner_id, {})
    lines = [
        f"# {EXP_ID} - Conclusao Executiva",
        "",
        f"- Vencedor: `{winner_id}`",
        f"- Baseline: TP={int(baseline['TP'])}, FP={int(baseline['FP'])}, FN={int(baseline['FN'])}, F1={baseline['F1']:.4f}",
        f"- Vencedor: TP={int(winner['TP'])}, FP={int(winner['FP'])}, FN={int(winner['FN'])}, F1={winner['F1']:.4f}",
        f"- Delta: TP={int(winner['TP'] - baseline['TP']):+d}, FP={int(winner['FP'] - baseline['FP']):+d}, FN={int(winner['FN'] - baseline['FN']):+d}, F1={winner['F1'] - baseline['F1']:+.4f}",
        f"- Caso residual: decisao={residual.get('decisao')} score={residual.get('score_final')}",
        f"- Validacao seed=123: TP={validation['TP']}, FP={validation['FP']}, FN={validation['FN']}, F1={validation['F1']:.4f}",
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
    print_section(f"{EXP_ID} - Residual Pattern")
    output_dir = get_experiment_output_dir(EXP_ID)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    df = load_dataset()
    sample_df = stratified_sample(df, n=args.sample, seed=args.seed, logger=logger)

    rows: list[dict[str, Any]] = []
    residuals: dict[str, Any] = {}
    patterns: dict[str, Any] = {}

    all_variants = [cfg["baseline"], *cfg["variantes"]]
    for variant_cfg in all_variants:
        preds = process_dataframe_via_orquestrador(
            sample_df,
            workers=args.workers,
            logger=logger,
            engine_config_overrides=variant_cfg["engine_config_overrides"],
        )
        rows.append(_metrics(preds, variant_cfg["label"], variant_cfg["id"]))
        residuals[variant_cfg["id"]] = _residual_analysis(preds)
        patterns[variant_cfg["id"]] = _pattern_analysis(preds)

    results_df = pd.DataFrame(rows)
    baseline = results_df.loc[results_df["variante_id"] == "BASELINE"].iloc[0]
    results_df["delta_TP"] = results_df["TP"] - baseline["TP"]
    results_df["delta_FP"] = results_df["FP"] - baseline["FP"]
    results_df["delta_FN"] = results_df["FN"] - baseline["FN"]
    results_df["delta_F1"] = (results_df["F1"] - baseline["F1"]).round(6)
    results_df.to_csv(output_dir / "01_tabela_comparativa.csv", index=False)
    safe_json_dump(residuals, output_dir / "02_analise_residual.json")
    safe_json_dump(patterns, output_dir / "03_analise_padroes.json")

    winner_id = _winner(results_df)
    winner_cfg = next(v for v in all_variants if v["id"] == winner_id)
    val_df = stratified_sample(df, n=args.sample, seed=args.validation_seed, logger=logger)
    val_preds = process_dataframe_via_orquestrador(
        val_df,
        workers=args.workers,
        logger=logger,
        engine_config_overrides=winner_cfg["engine_config_overrides"],
    )
    val_metrics = _metrics(val_preds, winner_cfg["label"], winner_id)
    val_payload = {
        "winner_id": winner_id,
        "seed": args.validation_seed,
        "sample": args.sample,
        "metrics": val_metrics,
        "residual": _residual_analysis(val_preds),
        "pattern": _pattern_analysis(val_preds),
    }
    safe_json_dump(val_payload, output_dir / "04_validacao_cruzada.json")
    _write_conclusion(output_dir / "05_conclusao_executiva.md", results_df, residuals, winner_id, val_metrics)
    logger.info("Resultados salvos em %s", output_dir)
    logger.info("Tempo total: %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
