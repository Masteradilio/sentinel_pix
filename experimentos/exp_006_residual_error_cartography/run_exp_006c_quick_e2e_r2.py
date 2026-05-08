"""
EXP-006C — Quick-E2E R2_LOW_VALUE_GRAY_FIRST_RECEIVER

Objetivo:
  Validar rapidamente a regra contrafactual R2 descoberta no EXP-006B:

    first_receiver_flag == 1
    vl_pix < 500
    candidate_lgbm_v6_2_raw >= 0.08
    candidate_lgbm_v6_2_raw < 0.20
    se_score <= 0
    beh_score <= 0
    decisão baseline == APROVAR

  A ação contrafactual é promover APROVAR -> CONFIRMAR.

Este script:
  - Roda baseline com PipelineOrquestrador real.
  - NÃO troca artefatos de produção.
  - NÃO altera scoring_config.json.
  - Carrega o LGBM v6.2 candidato apenas em shadow.
  - Aplica overlay R2 sobre as decisões baseline.
  - Salva artefatos incrementalmente.
  - Para em modo quick se não houver ganho.

Uso:
  python experimentos\\exp_006c_quick_e2e_r2\\run_exp_006c_quick_e2e_r2.py --quick
  python experimentos\\exp_006c_quick_e2e_r2\\run_exp_006c_quick_e2e_r2.py --final
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import joblib
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

CANDIDATE_DIR = PROJECT_ROOT / "backend" / "artefatos_candidatos" / "exp_005a_lgbm_v6_2_recall"
CANDIDATE_MODEL = CANDIDATE_DIR / "lgbm_v6_2_recall_candidate.joblib"
CANDIDATE_FEATURES = CANDIDATE_DIR / "lgbm_features_v6_2.json"

OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006C-R2"


from experimentos.utils_experimentos import (  # noqa: E402
    compute_metrics,
    get_logger,
    load_dataset,
    process_dataframe_via_orquestrador,
    safe_json_dump,
    stratified_sample,
)


EXP_ID = "EXP-006C-R2"
logger = get_logger(EXP_ID)


# =========================================================
# HELPERS
# =========================================================

def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_features() -> list[str]:
    if not CANDIDATE_FEATURES.exists():
        raise FileNotFoundError(f"Features candidatas não encontradas: {CANDIDATE_FEATURES}")

    obj = read_json(CANDIDATE_FEATURES)

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        for key in ["features", "feature_names", "lgbm_features"]:
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]

    raise ValueError(f"Formato inesperado em {CANDIDATE_FEATURES}")


def ensure_numeric_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)

    for col in features:
        if col in df.columns:
            x[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            x[col] = np.nan

    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.fillna(med).fillna(0.0)

    return x.astype(float)


def add_exp005a_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Recria features do EXP-005A, caso alguma esteja na lista do candidato.
    O vencedor atual usa baseline_features, mas este helper mantém robustez.
    """
    df = df_in.copy()

    def num(col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    vl = num("vl_pix", 0.0).clip(lower=0)
    rel = num("qt_tempo_relacionamento_mes", 999.0).clip(lower=0)
    idade = num("nr_idade", 0.0).clip(lower=0)
    first_receiver = num("first_receiver_flag", 0.0).clip(0, 1)
    pix_random = num("pix_key_random_flag", 0.0).clip(0, 1)
    burst = num("burst_30m_flag", 0.0).clip(0, 1)
    tx30 = num("tx_count_prev_30m", 0.0).clip(lower=0)
    distinct_receivers = num("distinct_receivers_so_far", 0.0).clip(lower=0)

    if "log_vl_pix" not in df.columns:
        df["log_vl_pix"] = np.log1p(vl)

    log_vl = num("log_vl_pix", 0.0)

    df["exp005_conta_nova_valor_alto_flag"] = (
        (rel <= 12)
        & (vl >= 5000)
        & (first_receiver == 1)
    ).astype(int)

    df["exp005_interaction_rel_valor"] = log_vl / (rel + 1.0)
    df["exp005_pix_random_x_first_receiver"] = pix_random * first_receiver
    df["exp005_valor_x_first_receiver"] = log_vl * first_receiver
    df["exp005_idade_x_valor_alto"] = idade * (vl >= 5000).astype(int)
    df["exp005_burst_x_valor"] = burst * log_vl
    df["exp005_tx30_x_valor"] = tx30 * log_vl
    df["exp005_first_receiver_x_rel_curto"] = first_receiver * (rel <= 12).astype(int)
    df["exp005_receiver_diversity_x_burst"] = distinct_receivers * burst

    if "valor_x_first_recv" not in df.columns:
        df["valor_x_first_recv"] = log_vl * first_receiver

    if "idade_x_first_recv" not in df.columns:
        df["idade_x_first_recv"] = idade * first_receiver

    if "valor_x_burst" not in df.columns:
        df["valor_x_burst"] = log_vl * burst

    if "burst_x_distinct_recv" not in df.columns:
        df["burst_x_distinct_recv"] = burst * distinct_receivers

    if "pix_random_x_first_receiver" not in df.columns:
        df["pix_random_x_first_receiver"] = pix_random * first_receiver

    return df


def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(["CONFIRMAR", "BLOQUEAR"])


def eval_predictions(df: pd.DataFrame, label: str) -> dict[str, Any]:
    y_true = df["is_fraud"].astype(int).values
    y_pred = flagged(df).astype(int).values
    return compute_metrics(y_true, y_pred, label).to_dict()


def compute_candidate_lgbm_scores(sample: pd.DataFrame) -> pd.Series:
    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(f"Modelo candidato não encontrado: {CANDIDATE_MODEL}")

    model = joblib.load(CANDIDATE_MODEL)
    features = load_candidate_features()

    df_feat = add_exp005a_features(sample)
    x = ensure_numeric_matrix(df_feat, features)

    scores = model.predict_proba(x)[:, 1]
    return pd.Series(scores, index=sample.index, name="candidate_lgbm_v6_2_raw")


def apply_r2_overlay(baseline_preds: pd.DataFrame, candidate_scores: pd.Series) -> pd.DataFrame:
    out = baseline_preds.copy()
    out["candidate_lgbm_v6_2_raw"] = candidate_scores.reindex(out.index).values

    # Garantir colunas numéricas.
    for col in ["vl_pix", "first_receiver_flag", "se_score", "beh_score"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["exp006c_r2_hit"] = (
        out["decisao"].astype(str).eq("APROVAR")
        & out["first_receiver_flag"].astype(int).eq(1)
        & out["vl_pix"].lt(500)
        & out["candidate_lgbm_v6_2_raw"].ge(0.08)
        & out["candidate_lgbm_v6_2_raw"].lt(0.20)
        & out["se_score"].le(0)
        & out["beh_score"].le(0)
    )

    hit_mask = out["exp006c_r2_hit"]

    out.loc[hit_mask, "decisao_original_exp006c"] = out.loc[hit_mask, "decisao"]
    out.loc[hit_mask, "score_final_original_exp006c"] = out.loc[hit_mask, "score_final"] if "score_final" in out.columns else np.nan

    out.loc[hit_mask, "decisao"] = "CONFIRMAR"

    if "score_final" in out.columns:
        out.loc[hit_mask, "score_final"] = out.loc[hit_mask, "score_final"].apply(lambda x: max(safe_float(x), 62.0))

    out.loc[hit_mask, "exp006c_reason"] = (
        "R2_LOW_VALUE_GRAY_FIRST_RECEIVER: "
        "APROVAR->CONFIRMAR | first_receiver=1, vl_pix<500, "
        "candidate_lgbm_v6_2 in [0.08,0.20), SE=0, BEH=0"
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
        "transaction_id",
        "customer_id",
        "vl_pix",
        "nr_idade",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "candidate_lgbm_v6_2_raw",
        "lgbm_raw",
        "if_percentile",
        "se_score",
        "beh_score",
        "score_final",
        "decisao",
        "exp006c_r2_hit",
        "exp006c_reason",
    ]
    cols = [c for c in cols if c in candidate.columns]

    return {
        "fns_recuperados": int(recovered_fn.sum()),
        "fps_adicionados": int(added_fp.sum()),
        "tps_perdidos": int(lost_tp.sum()),
        "fps_removidos": int(removed_fp.sum()),
        "rule_hits": int(candidate.get("exp006c_r2_hit", pd.Series(False, index=candidate.index)).sum()),
        "top_fns_recuperados": candidate.loc[recovered_fn, cols].head(50).to_dict(orient="records"),
        "top_fps_adicionados": candidate.loc[added_fp, cols].head(50).to_dict(orient="records"),
        "top_tps_perdidos": candidate.loc[lost_tp, cols].head(50).to_dict(orient="records"),
        "top_fps_removidos": candidate.loc[removed_fp, cols].head(50).to_dict(orient="records"),
    }


def write_decision(path: Path, baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any], delta: dict[str, Any]) -> None:
    fn_base = int(baseline_metrics["FN"])
    fp_base = int(baseline_metrics["FP"])
    f1_base = float(baseline_metrics["F1"])

    fn_cand = int(candidate_metrics["FN"])
    fp_cand = int(candidate_metrics["FP"])
    f1_cand = float(candidate_metrics["F1"])

    passes = (
        fn_cand < fn_base
        and fp_cand <= fp_base + 3
        and f1_cand >= f1_base
        and int(delta["tps_perdidos"]) == 0
    )

    status = "APROVADO_PARA_FINAL_E2E" if passes else "REJEITAR_NO_QUICK_E2E"

    lines = [
        "# EXP-006C — Quick-E2E R2",
        "",
        f"- Status: `{status}`",
        "",
        "## Baseline",
        "",
        f"- TP={baseline_metrics['TP']}, FP={baseline_metrics['FP']}, FN={baseline_metrics['FN']}",
        f"- Precision={baseline_metrics['Precision']}, Recall={baseline_metrics['Recall']}, F1={baseline_metrics['F1']}",
        "",
        "## Candidato R2",
        "",
        f"- TP={candidate_metrics['TP']}, FP={candidate_metrics['FP']}, FN={candidate_metrics['FN']}",
        f"- Precision={candidate_metrics['Precision']}, Recall={candidate_metrics['Recall']}, F1={candidate_metrics['F1']}",
        "",
        "## Delta",
        "",
        f"- FNs recuperados: `{delta['fns_recuperados']}`",
        f"- FPs adicionados: `{delta['fps_adicionados']}`",
        f"- TPs perdidos: `{delta['tps_perdidos']}`",
        f"- FPs removidos: `{delta['fps_removidos']}`",
        f"- Rule hits: `{delta['rule_hits']}`",
        "",
        "## Decisão",
        "",
    ]

    if passes:
        lines.extend([
            "A regra R2 passou no quick-E2E.",
            "Próximo passo: rodar final-E2E somente para R2, com sample 6000 e seeds 42/123.",
        ])
    else:
        lines.extend([
            "A regra R2 não passou no quick-E2E.",
            "Não promover e não rodar E2E completo.",
            "Próximo passo: EXP-006C/006D de censo completo dos 9 FNs residuais.",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(seed: int, sample_size: int, workers: int) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section(f"Seed {seed} | sample {sample_size}")

    df_full = load_dataset()
    sample = stratified_sample(df_full, n=sample_size, seed=seed, logger=logger)

    print_section("1. Baseline E2E real")
    baseline_preds = process_dataframe_via_orquestrador(
        sample,
        workers=workers,
        logger=logger,
        engine_config_overrides=None,
    )

    baseline_metrics = eval_predictions(baseline_preds, f"BASELINE_seed_{seed}")

    baseline_path = OUTPUT_DIR / f"baseline_predictions_seed_{seed}.csv"
    baseline_preds.to_csv(baseline_path, index=False, encoding="utf-8-sig")

    safe_json_dump(
        baseline_metrics,
        OUTPUT_DIR / f"00_baseline_metrics_seed_{seed}.json",
    )

    print(f"[OK] Baseline salvo: {baseline_path}")
    print(f"[OK] Baseline metrics: TP={baseline_metrics['TP']} FP={baseline_metrics['FP']} FN={baseline_metrics['FN']} F1={baseline_metrics['F1']}")

    print_section("2. LGBM v6.2 shadow score")
    candidate_scores = compute_candidate_lgbm_scores(sample)

    print_section("3. Aplicar overlay R2")
    candidate_preds = apply_r2_overlay(baseline_preds, candidate_scores)
    candidate_metrics = eval_predictions(candidate_preds, f"R2_seed_{seed}")
    delta = compare(baseline_preds, candidate_preds)

    candidate_path = OUTPUT_DIR / f"candidate_r2_predictions_seed_{seed}.csv"
    candidate_preds.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    rule_hits_path = OUTPUT_DIR / f"rule_hits_r2_seed_{seed}.csv"
    candidate_preds[candidate_preds["exp006c_r2_hit"]].to_csv(rule_hits_path, index=False, encoding="utf-8-sig")

    safe_json_dump(
        candidate_metrics,
        OUTPUT_DIR / f"01_candidate_r2_metrics_seed_{seed}.json",
    )

    safe_json_dump(
        delta,
        OUTPUT_DIR / f"02_delta_r2_seed_{seed}.json",
    )

    write_decision(
        OUTPUT_DIR / f"03_decision_seed_{seed}.md",
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        delta=delta,
    )

    print(f"[OK] Candidato salvo: {candidate_path}")
    print(f"[OK] Rule hits salvo: {rule_hits_path}")
    print(f"[OK] Candidate metrics: TP={candidate_metrics['TP']} FP={candidate_metrics['FP']} FN={candidate_metrics['FN']} F1={candidate_metrics['F1']}")
    print(f"[OK] Delta: FN_rec={delta['fns_recuperados']} FP_add={delta['fps_adicionados']} TP_lost={delta['tps_perdidos']}")

    return {
        "seed": seed,
        "sample_size": sample_size,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
    }


def aggregate_final(results: list[dict[str, Any]]) -> None:
    rows = []

    for r in results:
        seed = r["seed"]

        b = dict(r["baseline_metrics"])
        b["seed"] = seed
        b["config"] = "BASELINE"
        rows.append(b)

        c = dict(r["candidate_metrics"])
        c["seed"] = seed
        c["config"] = "R2_LOW_VALUE_GRAY_FIRST_RECEIVER"
        rows.append(c)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "04_summary_metrics.csv", index=False, encoding="utf-8-sig")

    safe_json_dump(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": results,
        },
        OUTPUT_DIR / "05_summary_results.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-006C Quick-E2E R2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="sample=1000, seed=42")
    group.add_argument("--final", action="store_true", help="sample=6000, seeds=42/123")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.quick:
        results = [run_one(seed=42, sample_size=1000, workers=args.workers)]
    else:
        results = [
            run_one(seed=42, sample_size=6000, workers=args.workers),
            run_one(seed=123, sample_size=6000, workers=args.workers),
        ]

    aggregate_final(results)

    print()
    print("=" * 72)
    print("[OK] EXP-006C-R2 concluído")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print(f"[OK] Tempo total: {time.perf_counter() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()