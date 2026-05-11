"""
EXP-006E — Residual FN Counterfactual Designer

Objetivo:
  Testar, em modo artifact-only, uma única hipótese cirúrgica para recuperar
  o FN residual NEAR_THRESHOLD identificado no EXP-006D.

Este experimento:
  - Não chama PipelineOrquestrador.
  - Não treina modelo.
  - Não troca artefatos.
  - Não altera scoring_config.json.
  - Usa baseline_predictions_seed_42.csv e baseline_predictions_seed_123.csv.
  - Aplica uma regra overlay APROVAR -> CONFIRMAR apenas em shadow.
  - Calcula delta real em FN, FP, TP perdido e F1.
  - Só recomenda quick-E2E se houver ganho líquido nos dois seeds.

Regra C1:
  decisao == APROVAR
  first_receiver_flag == 1
  pix_key_random_flag == 0
  qt_tempo_relacionamento_mes <= 12
  100 <= vl_pix < 500
  0.06 <= lgbm_raw < 0.10
  60 <= score_final < 62
  se_score <= 0
  beh_score <= 0

Saídas:
  resultados/experimentos/EXP-006E/
    00_input_summary.json
    01_candidate_rule_hits.csv
    02_metrics_comparison.csv
    03_delta_by_seed.json
    04_risk_audit.csv
    05_recommendation.md
    06_candidate_rule_spec.json
    07_next_experiment_spec.md
"""

from __future__ import annotations

import json
import math
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
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent.parent


PROJECT_ROOT = find_project_root(EXP_DIR)

BASELINE_INPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
CENSUS_INPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006D-FN-CENSUS"
OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-006E"


DECISOES_POSITIVAS = {"CONFIRMAR", "BLOQUEAR"}


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


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        if pd.isna(x):
            return default
        return int(float(x))
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


def compute_metrics(df: pd.DataFrame, label: str) -> dict[str, Any]:
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


def ensure_required_columns(df: pd.DataFrame, path: Path) -> None:
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
        raise ValueError(f"Arquivo {path} não contém colunas obrigatórias: {missing}")


def normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def load_baseline_predictions() -> pd.DataFrame:
    paths = {
        42: BASELINE_INPUT_DIR / "baseline_predictions_seed_42.csv",
        123: BASELINE_INPUT_DIR / "baseline_predictions_seed_123.csv",
    }

    missing = [p for p in paths.values() if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Arquivos baseline do EXP-006C-R2 não encontrados:\n"
            + "\n".join(str(p) for p in missing)
        )

    frames = []

    for seed, path in paths.items():
        df = pd.read_csv(path)
        ensure_required_columns(df, path)
        df = normalize_numeric_columns(df)
        df["seed"] = seed
        df["source_file"] = str(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_optional_census_summary() -> dict[str, Any]:
    assessment_path = CENSUS_INPUT_DIR / "05_recoverability_assessment.json"
    if not assessment_path.exists():
        return {}

    try:
        return json.loads(assessment_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# =========================================================
# REGRA CANDIDATA
# =========================================================

RULE_SPEC = {
    "rule_id": "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER",
    "description": (
        "Exceção cirúrgica para transação APROVAR near-threshold, "
        "primeiro recebedor, relacionamento curto, baixo/médio valor e LGBM em zona cinza."
    ),
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
    "promotion_reason": (
        "EXP-006D encontrou 1 FN NEAR_THRESHOLD recuperável e maioria dos demais FNs como data-limited."
    ),
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

    out["exp006e_c1_hit"] = mask
    out["exp006e_original_decisao"] = ""
    out["exp006e_original_score_final"] = np.nan
    out["exp006e_reason"] = ""

    hit_idx = out.index[mask]

    out.loc[hit_idx, "exp006e_original_decisao"] = out.loc[hit_idx, "decisao"]
    out.loc[hit_idx, "exp006e_original_score_final"] = out.loc[hit_idx, "score_final"]

    out.loc[hit_idx, "decisao"] = "CONFIRMAR"
    out.loc[hit_idx, "score_final"] = out.loc[hit_idx, "score_final"].apply(lambda x: max(safe_float(x), 62.0))
    out.loc[hit_idx, "exp006e_reason"] = (
        "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER: "
        "APROVAR->CONFIRMAR | rel<=12, first_receiver=1, pix_random=0, "
        "100<=vl<500, 0.06<=lgbm<0.10, 60<=score<62, SE=0, BEH=0"
    )

    return out


def compare_baseline_candidate(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
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
        "exp006e_c1_hit",
        "exp006e_original_decisao",
        "exp006e_original_score_final",
        "exp006e_reason",
    ]
    cols = [c for c in cols if c in candidate.columns]

    return {
        "fns_recuperados": int(recovered_fn.sum()),
        "fps_adicionados": int(added_fp.sum()),
        "tps_perdidos": int(lost_tp.sum()),
        "fps_removidos": int(removed_fp.sum()),
        "rule_hits": int(candidate["exp006e_c1_hit"].sum()),
        "top_fns_recuperados": candidate.loc[recovered_fn, cols].to_dict(orient="records"),
        "top_fps_adicionados": candidate.loc[added_fp, cols].to_dict(orient="records"),
        "top_tps_perdidos": candidate.loc[lost_tp, cols].to_dict(orient="records"),
        "top_fps_removidos": candidate.loc[removed_fp, cols].to_dict(orient="records"),
    }


# =========================================================
# RELATÓRIOS
# =========================================================

def decide_promotion(delta_by_seed: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted(delta_by_seed.keys())

    per_seed_pass = {}

    for seed in seeds:
        d = delta_by_seed[seed]
        per_seed_pass[seed] = (
            d["fns_recuperados"] >= 1
            and d["fps_adicionados"] == 0
            and d["tps_perdidos"] == 0
        )

    # Verificar F1 não piora.
    metrics = pd.DataFrame(metrics_rows)
    f1_ok = True

    for seed in seeds:
        b = metrics[(metrics["seed"].astype(str) == str(seed)) & (metrics["config"] == "BASELINE")]
        c = metrics[(metrics["seed"].astype(str) == str(seed)) & (metrics["config"] == RULE_SPEC["rule_id"])]

        if not b.empty and not c.empty:
            if float(c.iloc[0]["F1"]) < float(b.iloc[0]["F1"]):
                f1_ok = False

    all_pass = all(per_seed_pass.values()) and f1_ok

    if all_pass:
        status = "APROVADO_PARA_QUICK_E2E_PATCH_TEMPORARIO"
        next_action = (
            "Criar EXP-006F quick-E2E com patch temporário no DecisionEngine implementando C1. "
            "Rodar baseline + C1, sample 1000 ou 6000, sem grid."
        )
    else:
        status = "REJEITAR_SEM_E2E"
        next_action = (
            "Não rodar E2E. Seguir para EXP-007A Meta-Learner Shadow ou novas fontes de dados."
        )

    return {
        "status": status,
        "per_seed_pass": per_seed_pass,
        "f1_ok": f1_ok,
        "all_pass": all_pass,
        "next_action": next_action,
    }


def write_recommendation(
    path: Path,
    decision: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    delta_by_seed: dict[str, Any],
) -> None:
    metrics = pd.DataFrame(metrics_rows)

    lines = [
        "# EXP-006E — Residual FN Counterfactual Designer",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        f"- Status: `{decision['status']}`",
        "",
        "## Regra candidata",
        "",
        f"`{RULE_SPEC['rule_id']}`",
        "",
        RULE_SPEC["description"],
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

    for seed, d in sorted(delta_by_seed.items()):
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
        lines.extend([
            "A regra passou no artifact-only porque recuperou FN nos dois seeds, sem adicionar FP e sem perder TP.",
            "Ainda assim, não deve ser promovida diretamente: primeiro precisa virar patch temporário e passar em quick-E2E.",
        ])
    else:
        lines.extend([
            "A regra não passou no artifact-only. Não vale gastar tempo com E2E.",
            "Nesse caso, a FASE 2 deve avançar para meta-learner shadow ou para classificação dos FNs como dependentes de novas fontes de dados.",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment(path: Path, decision: dict[str, Any]) -> None:
    if decision["all_pass"]:
        title = "EXP-006F — Quick-E2E C1 Near-Threshold"
        objective = (
            "Implementar C1 como patch temporário no DecisionEngine e validar com baseline + 1 candidato, "
            "sem grid, com salvamento incremental."
        )
        command = (
            "Criar script de patch temporário e rodar somente C1. "
            "Não testar outros thresholds ou variantes."
        )
    else:
        title = "EXP-007A — Meta-Learner Shadow"
        objective = (
            "Verificar se há separabilidade não-linear nos sinais atuais. "
            "Se o meta-learner também não capturar os FNs sem FP, encerrar FASE 2 como limitada pelos dados atuais."
        )
        command = "Treinar/avaliar shadow-only, sem alterar DecisionEngine."

    lines = [
        "# Próximo experimento recomendado",
        "",
        f"## {title}",
        "",
        "## Objetivo",
        "",
        objective,
        "",
        "## Restrições de produtividade",
        "",
        "1. Não rodar grid E2E.",
        "2. Não testar múltiplos candidatos.",
        "3. Salvar baseline e candidato incrementalmente.",
        "4. Interromper se FN não cair, FP subir ou F1 piorar.",
        "",
        "## Ação",
        "",
        command,
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-006E — Residual FN Counterfactual Designer")
    print("=" * 72)

    print("[1/6] Carregando predições baseline...")
    all_preds = load_baseline_predictions()
    census_summary = load_optional_census_summary()

    print(f"[OK] Linhas carregadas: {len(all_preds)}")
    print(f"[OK] Seeds: {sorted(all_preds['seed'].unique().tolist())}")

    write_json(
        OUTPUT_DIR / "00_input_summary.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "baseline_input_dir": str(BASELINE_INPUT_DIR),
            "census_input_dir": str(CENSUS_INPUT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "rows_loaded": int(len(all_preds)),
            "seeds": sorted([int(x) for x in all_preds["seed"].unique().tolist()]),
            "census_summary": census_summary,
        },
    )

    print("[2/6] Aplicando regra C1 em shadow...")
    candidate_all = apply_c1_overlay(all_preds)

    hits = candidate_all[candidate_all["exp006e_c1_hit"]].copy()
    hits_path = OUTPUT_DIR / "01_candidate_rule_hits.csv"
    hits.to_csv(hits_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Rule hits: {len(hits)} -> {hits_path}")

    print("[3/6] Calculando métricas e deltas...")
    metrics_rows = []
    delta_by_seed = {}

    for seed, baseline_seed in all_preds.groupby("seed"):
        candidate_seed = candidate_all[candidate_all["seed"].eq(seed)].copy()

        baseline_metrics = compute_metrics(baseline_seed, f"BASELINE_seed_{seed}")
        candidate_metrics = compute_metrics(candidate_seed, f"{RULE_SPEC['rule_id']}_seed_{seed}")

        baseline_metrics["seed"] = int(seed)
        baseline_metrics["config"] = "BASELINE"

        candidate_metrics["seed"] = int(seed)
        candidate_metrics["config"] = RULE_SPEC["rule_id"]

        metrics_rows.append(baseline_metrics)
        metrics_rows.append(candidate_metrics)

        delta_by_seed[str(seed)] = compare_baseline_candidate(baseline_seed, candidate_seed)

        print(
            f"[OK] seed={seed}: "
            f"baseline FN={baseline_metrics['FN']} FP={baseline_metrics['FP']} F1={baseline_metrics['F1']} | "
            f"candidate FN={candidate_metrics['FN']} FP={candidate_metrics['FP']} F1={candidate_metrics['F1']} | "
            f"FN_rec={delta_by_seed[str(seed)]['fns_recuperados']} "
            f"FP_add={delta_by_seed[str(seed)]['fps_adicionados']} "
            f"TP_lost={delta_by_seed[str(seed)]['tps_perdidos']}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUT_DIR / "02_metrics_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    write_json(OUTPUT_DIR / "03_delta_by_seed.json", delta_by_seed)

    print(f"[OK] Métricas salvas: {metrics_path}")

    print("[4/6] Gerando auditoria de risco...")
    risk_rows = []

    for seed, d in delta_by_seed.items():
        for group_name in ["top_fns_recuperados", "top_fps_adicionados", "top_tps_perdidos", "top_fps_removidos"]:
            for row in d.get(group_name, []):
                r = dict(row)
                r["audit_group"] = group_name
                r["seed"] = int(seed)
                risk_rows.append(r)

    risk_df = pd.DataFrame(risk_rows)
    risk_path = OUTPUT_DIR / "04_risk_audit.csv"
    risk_df.to_csv(risk_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Risk audit salvo: {risk_path}")

    print("[5/6] Decidindo próximo passo...")
    decision = decide_promotion(delta_by_seed, metrics_rows)

    write_json(
        OUTPUT_DIR / "06_candidate_rule_spec.json",
        {
            "rule_spec": RULE_SPEC,
            "decision": decision,
        },
    )

    print(f"[OK] Decisão: {decision['status']}")

    print("[6/6] Escrevendo relatórios...")
    write_recommendation(
        OUTPUT_DIR / "05_recommendation.md",
        decision=decision,
        metrics_rows=metrics_rows,
        delta_by_seed=delta_by_seed,
    )

    write_next_experiment(
        OUTPUT_DIR / "07_next_experiment_spec.md",
        decision=decision,
    )

    print()
    print("[OK] EXP-006E concluído sem E2E.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '01_candidate_rule_hits.csv'}")
    print(f"  {OUTPUT_DIR / '02_metrics_comparison.csv'}")
    print(f"  {OUTPUT_DIR / '03_delta_by_seed.json'}")
    print(f"  {OUTPUT_DIR / '05_recommendation.md'}")
    print(f"  {OUTPUT_DIR / '07_next_experiment_spec.md'}")


if __name__ == "__main__":
    main()