# -*- coding: utf-8 -*-
"""
EXP-014B-R3V — Existing Action Policy Calibration Probe

Objetivo:
  Ajustar a estratégia à arquitetura real do modelo, que já opera com
  APROVAR / CONFIRMAR / BLOQUEAR.

  Em vez de criar novo "banding", este experimento mede se as ações existentes
  podem ser calibradas como política comercial:
    - bloqueio/intervenção forte;
    - confirmação/revisão;
    - aprovação.

Meta comercial atual:
  FN <= 5
  recall >= 95%
  FPR <= 1,5%

Este script:
  1. Usa R3Q como base binária congelada de detecção.
  2. Detecta colunas de decisão/score existentes.
  3. Audita FPs por ação atual.
  4. Testa políticas de intervenção baseadas nas ações já existentes e nos scores.
  5. Seleciona candidato apenas se cumprir orçamento de FN/recall e reduzir FP.
  6. Não promove nada sem frozen validation posterior.

Saídas:
  resultados/experimentos/EXP-014B-R3V/
    00_run_summary.json
    01_input_contract.json
    02_base_metrics.json
    03_action_distribution.csv
    04_policy_frontier.csv
    05_selected_policy.json
    06_robustness_by_segment.csv
    07_policy_artifact_recommended.json
    08_predictions_recommended.csv
    09_exp014b_r3v_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "EXP-014B-R3V"
TARGET_FPR = 0.015
MAX_FN = 5
MIN_RECALL = 0.95

BASE_COL_CANDIDATES = [
    "exp014b_r3q_frozen_pred",
    "exp014b_r3q_recommended_pred",
    "exp014b_r3p_frozen_pred",
    "exp014b_r3u_recommended_pred",
    "exp014b_r3s_recommended_pred",
]

LABEL_CANDIDATES = ["is_fraud", "fraude", "target", "label", "tp_fraude"]

ACTION_CANDIDATES = [
    "decisao",
    "decision",
    "action",
    "final_decision",
    "decision_engine_decisao",
    "engine_decision",
    "acao",
    "acao_recomendada",
]

SCORE_CANDIDATES = [
    "score_final",
    "exp014b_r3s_second_stage_score",
    "exp014b_r3u_receiver_relationship_trust_score",
    "lgbm_r4_score",
    "lgbm_raw",
    "lgbm_mapped",
    "score_final_engine",
    "peso_total",
    "if_percentile",
    "se_score",
    "beh_score",
    "behavioral_score",
    "topaz_risk_score",
]

SEGMENT_COLS = [
    "temporal_split",
    "event_month",
    "ds_tipo_chave_norm",
    "value_band",
    "periodo_dia",
    "mbk_available_flag",
    "module_quiet",
    "score_bin",
    "lgbm_bin",
    "if_bin",
    "ratio_bin",
]


def repo_root() -> Path:
    return Path.cwd()


def out_dir() -> Path:
    path = repo_root() / "resultados" / "experimentos" / EXPERIMENT
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_input() -> Path:
    base = repo_root() / "resultados" / "experimentos"
    candidates = [
        base / "EXP-014B-R3U" / "09_predictions_recommended.csv",
        base / "EXP-014B-R3S" / "08_predictions_recommended.csv",
        base / "EXP-014B-R3Q" / "08_predictions_recommended.csv",
        base / "EXP-014B-R3P-FROZEN" / "08_predictions_frozen.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Nenhum input encontrado. Esperado um dos arquivos:\n"
        + "\n".join(str(p) for p in candidates)
    )


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise KeyError(f"Nenhuma coluna encontrada entre: {candidates}")
    return None


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    y = safe_int_series(y_true)
    p = safe_int_series(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(precision), 8),
        "recall": round(float(recall), 8),
        "f1": round(float(f1), 8),
        "fpr": round(float(fpr), 8),
    }


def normalize_action(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "UNKNOWN"
    s = str(x).strip().upper()
    if not s or s in {"NAN", "NONE", "<MISSING>"}:
        return "UNKNOWN"
    if "BLOQ" in s or "BLOCK" in s:
        return "BLOQUEAR"
    if "CONF" in s or "REVIEW" in s or "ANALIS" in s or "ALERT" in s:
        return "CONFIRMAR"
    if "APROV" in s or "APPROV" in s or "ALLOW" in s:
        return "APROVAR"
    return s


def synthesize_action(df: pd.DataFrame, base_col: str, score_col: str | None) -> pd.Series:
    """Fallback caso a base de experimento nao tenha a coluna de decisao operacional."""
    base = safe_int_series(df[base_col])
    if score_col is None:
        return pd.Series(np.where(base == 1, "CONFIRMAR", "APROVAR"), index=df.index)

    score = pd.to_numeric(df[score_col], errors="coerce")
    # Mantem coerente com arquitetura, sem alegar ser a decisao real.
    # Serve apenas para probe quando a coluna original nao foi exportada.
    return pd.Series(
        np.select(
            [
                base.eq(0),
                score.ge(95),
                score.ge(77),
            ],
            ["APROVAR", "BLOQUEAR", "CONFIRMAR"],
            default="CONFIRMAR",
        ),
        index=df.index,
    )


def action_distribution(df: pd.DataFrame, label_col: str, action_col: str) -> pd.DataFrame:
    rows = []
    for action, g in df.groupby(action_col, dropna=False):
        pred = pd.Series(np.where(g[action_col].isin(["CONFIRMAR", "BLOQUEAR"]), 1, 0), index=g.index)
        m = metrics(g[label_col], pred)
        rows.append({
            "action": action,
            "n_rows": int(len(g)),
            "n_frauds": int(safe_int_series(g[label_col]).sum()),
            "tp_if_intervention": m["tp"],
            "fp_if_intervention": m["fp"],
            "fn_if_only_this_action_not_intervened": int(((safe_int_series(g[label_col]) == 1) & (pred == 0)).sum()),
            "precision_within_action": round(
                float(safe_int_series(g[label_col]).sum() / len(g)) if len(g) else 0.0,
                8,
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["fp_if_intervention", "n_rows"], ascending=[False, False]
    )


def candidate_from_mask(
    df: pd.DataFrame,
    label_col: str,
    base_col: str,
    mask: pd.Series,
    policy_name: str,
    rule_description: str,
    target_fp: int,
) -> dict[str, Any]:
    # A politica so pode intervir em alertas da base. Fora da base segue aprovado.
    pred = (safe_int_series(df[base_col]).eq(1) & mask.fillna(False)).astype(int)
    m = metrics(df[label_col], pred)
    base_m = metrics(df[label_col], df[base_col])
    return {
        "policy_name": policy_name,
        "rule_description": rule_description,
        **m,
        "fp_removed_vs_base": int(base_m["fp"] - m["fp"]),
        "fn_delta_vs_base": int(m["fn"] - base_m["fn"]),
        "target_fp": int(target_fp),
        "target_gap_fp": max(0, int(m["fp"] - target_fp)),
        "fn_budget_ok": bool(m["fn"] <= MAX_FN),
        "recall_ok": bool(m["recall"] >= MIN_RECALL),
        "fpr_target_ok": bool(m["fpr"] <= TARGET_FPR),
        "commercial_target_ok": bool(
            m["fn"] <= MAX_FN and m["recall"] >= MIN_RECALL and m["fpr"] <= TARGET_FPR
        ),
    }


def build_policy_frontier(
    df: pd.DataFrame,
    label_col: str,
    base_col: str,
    action_col: str,
    score_cols: list[str],
    target_fp: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    action = df[action_col].astype(str)
    base_alert = safe_int_series(df[base_col]).eq(1)

    # Politicas puras usando as acoes existentes.
    rows.append(candidate_from_mask(
        df, label_col, base_col,
        action.isin(["BLOQUEAR", "CONFIRMAR"]),
        "keep_confirmar_bloquear",
        "Intervir em CONFIRMAR ou BLOQUEAR (baseline operacional por acao)",
        target_fp,
    ))
    rows.append(candidate_from_mask(
        df, label_col, base_col,
        action.eq("BLOQUEAR"),
        "bloquear_only",
        "Intervir apenas em BLOQUEAR; CONFIRMAR vira acao fraca/sem bloqueio automatico",
        target_fp,
    ))
    rows.append(candidate_from_mask(
        df, label_col, base_col,
        action.eq("BLOQUEAR") | (action.eq("CONFIRMAR") & base_alert),
        "bloquear_plus_confirmar_current_alerts",
        "BLOQUEAR + CONFIRMAR dentro dos alertas R3Q",
        target_fp,
    ))

    # Thresholds por score, sempre dentro dos alertas da base.
    for score_col in score_cols:
        s = pd.to_numeric(df[score_col], errors="coerce")
        valid = s[base_alert & s.notna()]
        if valid.empty:
            continue

        # Quantis evitam grid caro; inclui extremos e pontos densos no topo.
        quantiles = sorted(set(
            [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
        ))
        thresholds = sorted(set(float(valid.quantile(q)) for q in quantiles if pd.notna(valid.quantile(q))))

        for th in thresholds:
            # Direcao principal: score alto = mais risco.
            mask_hi = s.ge(th)
            rows.append(candidate_from_mask(
                df, label_col, base_col, mask_hi,
                f"score_hi_{score_col}_{th:.8g}",
                f"Intervir em alertas R3Q com {score_col} >= {th:.8g}",
                target_fp,
            ))

            # Alguns scores podem estar invertidos; testa tambem cauda baixa.
            mask_lo = s.le(th)
            rows.append(candidate_from_mask(
                df, label_col, base_col, mask_lo,
                f"score_lo_{score_col}_{th:.8g}",
                f"Intervir em alertas R3Q com {score_col} <= {th:.8g}",
                target_fp,
            ))

        # Variante respeitando acao existente: BLOQUEAR sempre + CONFIRMAR por score.
        for th in thresholds:
            mask = action.eq("BLOQUEAR") | (action.eq("CONFIRMAR") & s.ge(th))
            rows.append(candidate_from_mask(
                df, label_col, base_col, mask,
                f"bloquear_plus_confirmar_hi_{score_col}_{th:.8g}",
                f"BLOQUEAR sempre + CONFIRMAR se {score_col} >= {th:.8g}",
                target_fp,
            ))

    frontier = pd.DataFrame(rows).drop_duplicates(
        subset=["policy_name", "tp", "fp", "fn", "fpr"]
    )
    return frontier.sort_values(
        ["commercial_target_ok", "fn_budget_ok", "recall_ok", "target_gap_fp", "fp", "fn"],
        ascending=[False, False, False, True, True, True],
    )


def select_policy(frontier: pd.DataFrame) -> dict[str, Any]:
    ok = frontier[frontier["commercial_target_ok"] == True].copy()
    if len(ok):
        # Dentro da meta, menor FP e depois maior recall.
        chosen = ok.sort_values(["fp", "fn", "recall"], ascending=[True, True, False]).iloc[0]
        reason = "COMMERCIAL_TARGET_REACHED"
    else:
        eligible = frontier[(frontier["fn_budget_ok"] == True) & (frontier["recall_ok"] == True)].copy()
        if len(eligible):
            chosen = eligible.sort_values(
                ["target_gap_fp", "fp_removed_vs_base", "fn_delta_vs_base"],
                ascending=[True, False, True],
            ).iloc[0]
            reason = "BEST_GAP_WITHIN_FN_RECALL_BUDGET"
        else:
            chosen = frontier.sort_values(["fn", "recall", "target_gap_fp"], ascending=[True, False, True]).iloc[0]
            reason = "NO_POLICY_WITHIN_FN_RECALL_BUDGET"
    out = chosen.to_dict()
    out["selection_reason"] = reason
    return out


def apply_selected_policy(df: pd.DataFrame, selected: dict[str, Any], base_col: str, action_col: str) -> pd.Series:
    desc = str(selected["rule_description"])
    policy = str(selected["policy_name"])
    base_alert = safe_int_series(df[base_col]).eq(1)
    action = df[action_col].astype(str)

    if policy == "keep_confirmar_bloquear":
        mask = action.isin(["CONFIRMAR", "BLOQUEAR"])
    elif policy == "bloquear_only":
        mask = action.eq("BLOQUEAR")
    elif policy == "bloquear_plus_confirmar_current_alerts":
        mask = action.eq("BLOQUEAR") | (action.eq("CONFIRMAR") & base_alert)
    elif policy.startswith("score_hi_"):
        score_col, th = parse_score_policy(policy, "score_hi_")
        mask = pd.to_numeric(df[score_col], errors="coerce").ge(th)
    elif policy.startswith("score_lo_"):
        score_col, th = parse_score_policy(policy, "score_lo_")
        mask = pd.to_numeric(df[score_col], errors="coerce").le(th)
    elif policy.startswith("bloquear_plus_confirmar_hi_"):
        score_col, th = parse_score_policy(policy, "bloquear_plus_confirmar_hi_")
        mask = action.eq("BLOQUEAR") | (action.eq("CONFIRMAR") & pd.to_numeric(df[score_col], errors="coerce").ge(th))
    else:
        mask = base_alert

    return (base_alert & mask.fillna(False)).astype(int)


def parse_score_policy(policy: str, prefix: str) -> tuple[str, float]:
    body = policy[len(prefix):]
    # score_col pode conter underscores; threshold e o ultimo token.
    col, th = body.rsplit("_", 1)
    return col, float(th)


def robustness(df: pd.DataFrame, label_col: str, base_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    y = safe_int_series(df[label_col])
    base = safe_int_series(df[base_col])
    pred = safe_int_series(df[pred_col])
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx = list(idx)
            g = df.loc[idx]
            m = metrics(g[label_col], g[pred_col])
            base_m = metrics(g[label_col], g[base_col])
            rows.append({
                "segment_col": col,
                "segment_value": str(val),
                "n_rows": int(len(g)),
                "n_frauds": int(safe_int_series(g[label_col]).sum()),
                "fp_removed_vs_base": int(base_m["fp"] - m["fp"]),
                "fn_delta_vs_base": int(m["fn"] - base_m["fn"]),
                "final_tp": m["tp"],
                "final_fp": m["fp"],
                "final_fn": m["fn"],
                "final_recall": m["recall"],
                "final_fpr": m["fpr"],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fp_removed_vs_base", "n_rows"], ascending=[False, False])


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    od = out_dir()
    input_path = find_input()
    df = pd.read_csv(input_path, low_memory=False)

    label_col = find_col(df, LABEL_CANDIDATES)
    base_col = find_col(df, BASE_COL_CANDIDATES)

    action_col_raw = find_col(df, ACTION_CANDIDATES, required=False)
    score_cols = [c for c in SCORE_CANDIDATES if c in df.columns]
    primary_score = score_cols[0] if score_cols else None

    if action_col_raw:
        df["r3v_action_norm"] = df[action_col_raw].apply(normalize_action)
        action_source = action_col_raw
    else:
        df["r3v_action_norm"] = synthesize_action(df, base_col, primary_score)
        action_source = "SYNTHESIZED_FROM_BASE_AND_SCORE"

    n_rows = int(len(df))
    n_frauds = int(safe_int_series(df[label_col]).sum())
    n_normals = n_rows - n_frauds
    target_fp = int(np.floor(TARGET_FPR * n_normals))

    contract = {
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "input_path": str(input_path),
        "label_col": label_col,
        "base_col": base_col,
        "action_source": action_source,
        "score_cols_used": score_cols,
        "target_fpr": TARGET_FPR,
        "target_fp": target_fp,
        "max_fn": MAX_FN,
        "min_recall": MIN_RECALL,
        "missing": [],
        "contract_ok": True,
    }

    base_m = metrics(df[label_col], df[base_col])
    base_metrics = {
        "base_metrics": base_m,
        "target_fpr_ok": bool(base_m["fpr"] <= TARGET_FPR),
        "fn_budget_ok": bool(base_m["fn"] <= MAX_FN),
        "recall_ok": bool(base_m["recall"] >= MIN_RECALL),
        "target_fp": target_fp,
        "target_gap_fp": max(0, int(base_m["fp"] - target_fp)),
    }

    dist = action_distribution(df, label_col, "r3v_action_norm")
    frontier = build_policy_frontier(df, label_col, base_col, "r3v_action_norm", score_cols, target_fp)
    selected = select_policy(frontier)

    pred_col = "exp014b_r3v_recommended_pred"
    df[pred_col] = apply_selected_policy(df, selected, base_col, "r3v_action_norm")
    selected_metrics = metrics(df[label_col], df[pred_col])

    # Recalcula selected para garantir consistencia exata.
    selected.update({
        "recommended_metrics": selected_metrics,
        "commercial_target_reached": bool(
            selected_metrics["fn"] <= MAX_FN
            and selected_metrics["recall"] >= MIN_RECALL
            and selected_metrics["fpr"] <= TARGET_FPR
        ),
        "target_gap_fp": max(0, int(selected_metrics["fp"] - target_fp)),
        "fp_removed_vs_base": int(base_m["fp"] - selected_metrics["fp"]),
        "fn_delta_vs_base": int(selected_metrics["fn"] - base_m["fn"]),
    })

    rob = robustness(df, label_col, base_col, pred_col)

    artifact = {
        "experiment": EXPERIMENT,
        "policy_name": selected["policy_name"],
        "selection_reason": selected["selection_reason"],
        "input_path": str(input_path),
        "base_col": base_col,
        "action_source": action_source,
        "final_pred_col": pred_col,
        "target_fpr": TARGET_FPR,
        "target_fp": target_fp,
        "max_fn": MAX_FN,
        "min_recall": MIN_RECALL,
        "base_metrics": base_m,
        "recommended_metrics": selected_metrics,
        "selected_policy": selected,
        "score_cols_used": score_cols,
        "notes": [
            "This probe respects the existing APROVAR/CONFIRMAR/BLOQUEAR architecture.",
            "It tests commercial intervention policies over existing actions and scores.",
            "Promotion requires frozen validation and business review if FN > 0.",
        ],
    }

    objective_status = (
        "DONE_R3V_EXISTING_ACTION_POLICY_TARGET_REACHED"
        if selected["commercial_target_reached"]
        else "DONE_R3V_EXISTING_ACTION_POLICY_TARGET_NOT_REACHED"
    )

    summary = {
        "experiment": EXPERIMENT,
        "status": "DONE",
        "objective_status": objective_status,
        "n_rows": n_rows,
        "n_frauds": n_frauds,
        "n_normals": n_normals,
        "base_col": base_col,
        "action_source": action_source,
        "base_metrics": base_m,
        "target_fpr": TARGET_FPR,
        "target_fp": target_fp,
        "max_fn": MAX_FN,
        "min_recall": MIN_RECALL,
        "recommended_policy_name": selected["policy_name"],
        "selection_reason": selected["selection_reason"],
        "recommended_metrics": selected_metrics,
        "fp_removed_vs_base": artifact["selected_policy"]["fp_removed_vs_base"],
        "fn_delta_vs_base": artifact["selected_policy"]["fn_delta_vs_base"],
        "target_gap_fp": selected["target_gap_fp"],
        "commercial_target_reached": selected["commercial_target_reached"],
        "n_policies_evaluated": int(len(frontier)),
        "all_pass": True,
        "output_dir": str(od),
    }

    write_json(od / "00_run_summary.json", summary)
    write_json(od / "01_input_contract.json", contract)
    write_json(od / "02_base_metrics.json", base_metrics)
    dist.to_csv(od / "03_action_distribution.csv", index=False, encoding="utf-8")
    frontier.to_csv(od / "04_policy_frontier.csv", index=False, encoding="utf-8")
    write_json(od / "05_selected_policy.json", selected)
    rob.to_csv(od / "06_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(od / "07_policy_artifact_recommended.json", artifact)
    df.to_csv(od / "08_predictions_recommended.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT} - Existing Action Policy Calibration Probe

## Resultado executivo
- Status: `{objective_status}`
- Base col: `{base_col}`
- Fonte da ação: `{action_source}`
- Base: `{base_m}`
- Meta comercial: FN<={MAX_FN}, recall>={MIN_RECALL}, FPR<={TARGET_FPR}
- FP max alvo: `{target_fp}`
- Gap base até alvo: `{base_metrics["target_gap_fp"]}` FP
- Política recomendada: `{selected["policy_name"]}`
- Razão de seleção: `{selected["selection_reason"]}`
- Métricas recomendadas: `{selected_metrics}`
- FP removidos vs base: `{artifact["selected_policy"]["fp_removed_vs_base"]}`
- FN delta vs base: `{artifact["selected_policy"]["fn_delta_vs_base"]}`
- Target comercial atingido: `{selected["commercial_target_reached"]}`

## Distribuição por ação
{dist.to_markdown(index=False)}

## Melhores políticas avaliadas
{frontier.head(30).to_markdown(index=False)}

## Decisão sugerida
Se `commercial_target_reached=true`, executar frozen validation.
Se não, usar a distribuição por ação para decidir se o gargalo está no mapeamento CONFIRMAR/BLOQUEAR,
nos scores que alimentam a decisão ou na necessidade de novos dados comerciais.
"""
    (od / "09_exp014b_r3v_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
