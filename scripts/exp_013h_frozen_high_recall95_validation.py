#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013H — Frozen High-Recall95 Micro Policy Validation

Objetivo:
  Congelar a política vencedora do EXP-013G e validá-la sem nova busca.

Política congelada:
  high_recall95_micro_refined_policy

Benchmark esperado no dataset atual:
  TP=118
  FP=414
  FN=6
  recall=95.16%
  precision=22.18%

O que este script faz:
  1. Carrega um arquivo de predições/comparação.
  2. Usa a coluna exp013g_micro_pred se ela já existir; OU
     aplica as microações congeladas do EXP-013G sobre pred_HIGH_RECALL_95.
  3. Calcula métricas globais.
  4. Calcula métricas por blocos temporais.
  5. Calcula bootstrap de recall/FP.
  6. Calcula métricas por segmentos críticos.
  7. Gera arquivos de FN/FP para revisão.
  8. Aplica um gate de promoção shadow:
       - global recall >= 95%
       - TP >= min_tp_required
       - FP <= baseline/reference FP, quando informado
       - sem reotimização

Entradas default:
  resultados/experimentos/EXP-013G/06_selected_predictions.csv
  resultados/experimentos/EXP-013G/12_policy_artifact.json

Uso no dataset atual:
  python scripts/exp_013h_frozen_high_recall95_validation.py

Uso em nova janela/validação externa:
  python scripts/exp_013h_frozen_high_recall95_validation.py --input caminho\\novo_arquivo.csv

Observação:
  Para validação externa, o arquivo deve conter:
    - is_fraud
    - pred_HIGH_RECALL_95
    - features usadas nas microações do EXP-013G
  Se já contiver exp013g_micro_pred, o script usa essa coluna diretamente.

Saídas:
  resultados/experimentos/EXP-013H/
    00_run_summary.json
    01_global_metrics.csv
    02_time_block_metrics.csv
    03_bootstrap_confidence_intervals.csv
    04_segment_metrics.csv
    05_frozen_predictions.csv
    06_false_negatives.csv
    07_false_positives.csv
    08_gate_report.md
    09_policy_used.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
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

DEFAULT_INPUT = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013G" / "06_selected_predictions.csv"
DEFAULT_POLICY = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013G" / "12_policy_artifact.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "experimentos" / "EXP-013H"

FLAGGED_DECISIONS = {"CONFIRMAR", "BLOQUEAR"}


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

    if "is_fraud" not in df.columns:
        raise RuntimeError("Coluna is_fraud ausente.")

    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    for pred_col in ["exp013g_micro_pred", "pred_HIGH_RECALL_95", "exp013g_high_recall_base_pred", "exp013e_refined_pred"]:
        if pred_col in df.columns:
            df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)

    if "runtime_flagged" not in df.columns:
        if "decisao" in df.columns:
            df["runtime_flagged"] = df["decisao"].astype(str).str.upper().isin(FLAGGED_DECISIONS).astype(int)
        else:
            df["runtime_flagged"] = 0
    df["runtime_flagged"] = pd.to_numeric(df["runtime_flagged"], errors="coerce").fillna(0).astype(int)

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype("string").str.strip()

    for c in ["event_datetime", "data_pix"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

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


def text(df: pd.DataFrame, names: str | list[str], default: str = "<MISSING>") -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).astype(str)


def boolish(df: pd.DataFrame, names: str | list[str], default: bool = False) -> pd.Series:
    col = pick_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    return s.astype(str).str.upper().isin({"1", "1.0", "TRUE", "T", "SIM", "YES", "Y"})


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


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy artifact não encontrado: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selected_actions" not in obj:
        raise RuntimeError("Policy artifact não contém selected_actions.")
    return obj


def get_lgbm_score(df: pd.DataFrame) -> pd.Series:
    return num(df, ["lgbm_r4_score", "r4_score", "lgbm_mapped", "lgbm_raw"], 0.0)


def parse_segment_description(desc: str) -> tuple[list[str], list[str], float | None]:
    """
    Parse descriptions like:
      veto current positive segment value_band=F_10000_PLUS AND ds_tipo_chave_norm=EMAIL
      veto current positive segment periodo_dia=noite AND value_band=D_1000_5000 AND lgbm<0.02
    """
    desc = str(desc)
    if "segment " in desc:
        segment_part = desc.split("segment ", 1)[1]
    else:
        segment_part = desc

    parts = [p.strip() for p in segment_part.split(" AND ")]
    cols = []
    vals = []
    lgbm_lt = None

    for p in parts:
        if p.startswith("lgbm<"):
            try:
                lgbm_lt = float(p.replace("lgbm<", "").strip())
            except Exception:
                lgbm_lt = None
        elif "=" in p:
            c, v = p.split("=", 1)
            cols.append(c.strip())
            vals.append(v.strip())

    return cols, vals, lgbm_lt


def apply_action_mask(df: pd.DataFrame, action: dict[str, Any], base_pred: np.ndarray) -> np.ndarray:
    desc = str(action.get("description", ""))
    family = str(action.get("family", ""))

    if family not in {"segment_veto", "segment_lgbm_veto"}:
        # EXP-013G selected only these two families. Keep strict to avoid silent mistakes.
        raise RuntimeError(f"Família de microação não suportada neste validador: {family}")

    cols, vals, lgbm_lt = parse_segment_description(desc)
    if not cols:
        raise RuntimeError(f"Não consegui parsear segmento da ação: {desc}")

    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(cols, vals):
        if c not in df.columns:
            raise RuntimeError(f"Coluna necessária para ação congelada ausente: {c}")
        mask = mask & (text(df, c) == str(v)).to_numpy(dtype=bool)

    if lgbm_lt is not None:
        mask = mask & (get_lgbm_score(df) < lgbm_lt).to_numpy(dtype=bool)

    return mask & (base_pred == 1)


def apply_frozen_policy(df: pd.DataFrame, policy: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    if "exp013g_micro_pred" in df.columns:
        pred = df["exp013g_micro_pred"].to_numpy(dtype=int)
        return pred, pd.DataFrame([{
            "mode": "used_existing_exp013g_micro_pred",
            "action_id": None,
            "family": None,
            "description": "Input already contained exp013g_micro_pred; no reapplication needed.",
            "tp_delta": None,
            "fp_delta": None,
            "n_affected": None,
        }])

    if "pred_HIGH_RECALL_95" not in df.columns:
        raise RuntimeError("Input precisa conter exp013g_micro_pred ou pred_HIGH_RECALL_95.")

    y = df["is_fraud"].to_numpy(dtype=int)
    pred = df["pred_HIGH_RECALL_95"].to_numpy(dtype=int).copy()

    rows = []
    for idx, action in enumerate(policy.get("selected_actions", [])):
        action_type = action.get("action_type")
        if action_type != "veto":
            raise RuntimeError(f"EXP-013H só reaplica vetos do EXP-013G; ação inválida: {action_type}")

        mask = apply_action_mask(df, action, pred)
        tp_delta = -int(((y == 1) & mask).sum())
        fp_delta = -int(((y == 0) & mask).sum())

        pred[mask] = 0

        rows.append({
            "mode": "applied_micro_action",
            "action_id": action.get("action_id", f"action_{idx}"),
            "family": action.get("family"),
            "description": action.get("description"),
            "tp_delta": tp_delta,
            "fp_delta": fp_delta,
            "n_affected": int(mask.sum()),
        })

    return pred, pd.DataFrame(rows)


def make_time_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    if "data_pix" in df.columns and df["data_pix"].notna().any():
        dates = pd.to_datetime(df["data_pix"], errors="coerce")
    elif "event_datetime" in df.columns and df["event_datetime"].notna().any():
        dates = pd.to_datetime(df["event_datetime"], errors="coerce")
    else:
        return pd.qcut(np.arange(len(df)), q=min(n_blocks, len(df)), labels=False, duplicates="drop").astype(int)

    tmp = pd.DataFrame({"date": dates, "_idx": np.arange(len(df))}).sort_values(["date", "_idx"])
    tmp["block"] = pd.qcut(np.arange(len(tmp)), q=min(n_blocks, len(tmp)), labels=False, duplicates="drop")
    out = pd.Series(index=tmp["_idx"].values, data=tmp["block"].values).sort_index()
    return out.astype(int)


def block_metrics(df: pd.DataFrame, pred: np.ndarray, n_blocks: int, policy_name: str) -> pd.DataFrame:
    blocks = make_time_blocks(df, n_blocks)
    rows = []
    for b in sorted(blocks.dropna().unique()):
        part = df.loc[blocks == b].copy()
        pred_b = pred[blocks == b]
        m = compute_metrics(part["is_fraud"].to_numpy(dtype=int), pred_b)
        m.update({
            "policy_name": policy_name,
            "block": int(b),
            "n_rows": int(len(part)),
            "n_frauds": int(part["is_fraud"].sum()),
            "dt_min": str(part["data_pix"].min().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
            "dt_max": str(part["data_pix"].max().date()) if "data_pix" in part.columns and part["data_pix"].notna().any() else None,
        })
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_eval(df: pd.DataFrame, pred_col: str, iters: int, seed: int, target_recall: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    rows = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        y = df.iloc[idx]["is_fraud"].to_numpy(dtype=int)
        pred = df.iloc[idx][pred_col].to_numpy(dtype=int)
        rows.append(compute_metrics(y, pred))

    boot = pd.DataFrame(rows)
    out = []
    for metric in ["tp", "fp", "fn", "precision", "recall", "f1", "fpr"]:
        vals = boot[metric].astype(float)
        out.append({
            "metric": metric,
            "mean": float(vals.mean()),
            "p025": float(vals.quantile(0.025)),
            "p050": float(vals.quantile(0.50)),
            "p975": float(vals.quantile(0.975)),
            "target_recall": target_recall if metric == "recall" else None,
            "p_below_target_recall": float((boot["recall"] < target_recall).mean()) if metric == "recall" else None,
        })
    return pd.DataFrame(out)


def segment_metrics(df: pd.DataFrame, pred: np.ndarray, segment_cols: list[str]) -> pd.DataFrame:
    y = df["is_fraud"].to_numpy(dtype=int)
    rows = []

    for col in segment_cols:
        if col not in df.columns:
            continue

        for level, idxs in df.groupby(text(df, col), dropna=False).indices.items():
            idxs = np.array(list(idxs), dtype=int)
            if len(idxs) < 20:
                continue
            m = compute_metrics(y[idxs], pred[idxs])
            m.update({
                "segment_col": col,
                "segment_value": str(level),
                "n_rows": int(len(idxs)),
                "n_frauds": int(y[idxs].sum()),
            })
            rows.append(m)

    combos = [
        ["value_band", "ds_tipo_chave_norm"],
        ["periodo_dia", "value_band"],
        ["first_receiver_flag_real", "value_band"],
    ]

    for cols in combos:
        if any(c not in df.columns for c in cols):
            continue

        tmp = pd.DataFrame({c: text(df, c) for c in cols})
        grouped = tmp.groupby(cols, dropna=False).indices

        for key, idxs in grouped.items():
            idxs = np.array(list(idxs), dtype=int)
            if len(idxs) < 20:
                continue
            m = compute_metrics(y[idxs], pred[idxs])
            vals = key if isinstance(key, tuple) else (key,)
            m.update({
                "segment_col": "|".join(cols),
                "segment_value": "|".join(str(v) for v in vals),
                "n_rows": int(len(idxs)),
                "n_frauds": int(y[idxs].sum()),
            })
            rows.append(m)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["fn", "fp", "n_frauds"], ascending=[False, False, False]).reset_index(drop=True)


def gate_decision(metrics: dict[str, Any], blocks: pd.DataFrame, boot: pd.DataFrame, target_recall: float, min_tp_required: int, reference_fp: int | None, strict_blocks: bool) -> dict[str, Any]:
    risks = []
    warnings = []

    if metrics["recall"] < target_recall or metrics["tp"] < min_tp_required:
        risks.append("GLOBAL_RECALL_OR_TP_BELOW_TARGET")

    if reference_fp is not None and metrics["fp"] > reference_fp:
        risks.append("FP_ABOVE_REFERENCE")

    if not blocks.empty:
        min_block_recall = float(blocks["recall"].min())
        if min_block_recall < target_recall:
            if strict_blocks:
                risks.append("TIME_BLOCK_RECALL_BELOW_TARGET")
            else:
                warnings.append("TIME_BLOCK_RECALL_BELOW_TARGET")
    else:
        min_block_recall = None

    recall_ci = boot[boot["metric"] == "recall"] if not boot.empty else pd.DataFrame()
    if not recall_ci.empty:
        recall_p025 = float(recall_ci["p025"].iloc[0])
        prob_below = float(recall_ci["p_below_target_recall"].iloc[0])
        if recall_p025 < target_recall:
            warnings.append("BOOTSTRAP_RECALL_P025_BELOW_TARGET")
        if prob_below > 0.35:
            warnings.append("BOOTSTRAP_TARGET_FAILURE_PROB_HIGH")
    else:
        recall_p025 = None
        prob_below = None

    if risks:
        gate_status = "FAIL"
    elif warnings:
        gate_status = "PASS_WITH_WARNINGS"
    else:
        gate_status = "PASS"

    return {
        "gate_status": gate_status,
        "risks": risks,
        "warnings": warnings,
        "min_block_recall": min_block_recall,
        "bootstrap_recall_p025": recall_p025,
        "bootstrap_prob_recall_below_target": prob_below,
    }


def make_report(summary: dict[str, Any], global_df: pd.DataFrame, blocks: pd.DataFrame, boot: pd.DataFrame, actions: pd.DataFrame) -> str:
    lines = []
    lines.append("# EXP-013H — Frozen High-Recall95 Micro Policy Validation")
    lines.append("")
    lines.append("## Gate")
    lines.append(f"- Gate status: `{summary['gate']['gate_status']}`")
    lines.append(f"- Risks: `{', '.join(summary['gate']['risks']) or 'none'}`")
    lines.append(f"- Warnings: `{', '.join(summary['gate']['warnings']) or 'none'}`")
    lines.append("")
    lines.append("## Métricas globais")
    lines.append(global_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Ações congeladas")
    if actions.empty:
        lines.append("Nenhuma ação registrada.")
    else:
        lines.append(actions.to_markdown(index=False))
    lines.append("")
    lines.append("## Blocos temporais")
    if blocks.empty:
        lines.append("Sem blocos temporais.")
    else:
        lines.append(blocks.to_markdown(index=False))
    lines.append("")
    lines.append("## Bootstrap recall")
    if boot.empty:
        lines.append("Sem bootstrap.")
    else:
        lines.append(boot[boot["metric"] == "recall"].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretação")
    if summary["gate"]["gate_status"] == "PASS":
        lines.append("A política congelada passou no gate definido. Próximo passo: patch shadow configurável no DecisionEngine/PipelineOrquestrador.")
    elif summary["gate"]["gate_status"] == "PASS_WITH_WARNINGS":
        lines.append("A política congelada manteve o alvo global, mas ainda apresenta alertas de estabilidade. Próximo passo recomendado: validação E2E shadow configurável com monitoramento dos FNs por janela.")
    else:
        lines.append("A política congelada falhou no gate. Não promover; revisar política ou usar benchmark anterior.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--policy-artifact", default=str(DEFAULT_POLICY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--reference-fp", type=int, default=414, help="FP máximo esperado para passar gate. Use -1 para ignorar.")
    parser.add_argument("--time-blocks", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-blocks", action="store_true", help="Transforma recall por bloco abaixo do alvo em FAIL em vez de warning.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_path = Path(args.input)
    policy_path = Path(args.policy_artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("EXP-013H — Frozen High-Recall95 Micro Policy Validation")
    log("=" * 80)
    log(f"Input: {input_path}")
    log(f"Policy: {policy_path}")
    log(f"Output: {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    policy = load_policy(policy_path)
    dump_json(policy, output_dir / "09_policy_used.json")

    y = df["is_fraud"].to_numpy(dtype=int)
    total_frauds = int(y.sum())
    min_tp_required = int(math.ceil(args.target_recall * total_frauds))

    pred, action_impacts = apply_frozen_policy(df, policy)
    metrics = compute_metrics(y, pred)

    global_rows = [{"policy_name": "FROZEN_EXP013G_MICRO_REFINED", **metrics}]
    if "pred_HIGH_RECALL_95" in df.columns:
        high_metrics = compute_metrics(y, df["pred_HIGH_RECALL_95"].to_numpy(dtype=int))
        global_rows.insert(0, {"policy_name": "BASE_HIGH_RECALL_95", **high_metrics})
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(output_dir / "01_global_metrics.csv", index=False)

    predictions = df.copy()
    predictions["exp013h_frozen_pred"] = pred
    predictions.to_csv(output_dir / "05_frozen_predictions.csv", index=False)
    predictions[(predictions["is_fraud"] == 1) & (predictions["exp013h_frozen_pred"] == 0)].to_csv(output_dir / "06_false_negatives.csv", index=False)
    predictions[(predictions["is_fraud"] == 0) & (predictions["exp013h_frozen_pred"] == 1)].to_csv(output_dir / "07_false_positives.csv", index=False)

    action_impacts.to_csv(output_dir / "action_impacts.csv", index=False)

    blocks = block_metrics(df, pred, args.time_blocks, "FROZEN_EXP013G_MICRO_REFINED")
    blocks.to_csv(output_dir / "02_time_block_metrics.csv", index=False)

    boot_input = predictions.copy()
    boot = bootstrap_eval(boot_input, "exp013h_frozen_pred", args.bootstrap_iters, args.seed, args.target_recall)
    boot.to_csv(output_dir / "03_bootstrap_confidence_intervals.csv", index=False)

    seg_cols = ["value_band", "ds_tipo_chave_norm", "periodo_dia", "first_receiver_flag_real", "mbk_available_flag"]
    seg = segment_metrics(df, pred, seg_cols)
    seg.to_csv(output_dir / "04_segment_metrics.csv", index=False)

    ref_fp = None if args.reference_fp is None or args.reference_fp < 0 else args.reference_fp
    gate = gate_decision(metrics, blocks, boot, args.target_recall, min_tp_required, ref_fp, args.strict_blocks)

    objective_status = f"GATE_{gate['gate_status']}"
    objective_status += "_TARGET_RECALL_MET" if metrics["recall"] >= args.target_recall else "_TARGET_RECALL_NOT_MET"
    objective_status += "_REFERENCE_FP_MET" if ref_fp is None or metrics["fp"] <= ref_fp else "_REFERENCE_FP_NOT_MET"

    summary = {
        "experiment": "EXP-013H",
        "status": "DONE",
        "objective_status": objective_status,
        "input_path": str(input_path),
        "policy_path": str(policy_path),
        "n_rows": int(len(df)),
        "total_frauds": total_frauds,
        "target_recall": args.target_recall,
        "min_tp_required": min_tp_required,
        "reference_fp": ref_fp,
        "metrics": metrics,
        "gate": gate,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_dir": str(output_dir),
    }
    dump_json(summary, output_dir / "00_run_summary.json")

    report = make_report(summary, global_df, blocks, boot, action_impacts)
    (output_dir / "08_gate_report.md").write_text(report, encoding="utf-8")

    log("")
    log("=" * 80)
    log("EXP-013H CONCLUÍDO")
    log("=" * 80)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("Arquivos principais:")
    for p in [
        output_dir / "00_run_summary.json",
        output_dir / "01_global_metrics.csv",
        output_dir / "02_time_block_metrics.csv",
        output_dir / "03_bootstrap_confidence_intervals.csv",
        output_dir / "04_segment_metrics.csv",
        output_dir / "08_gate_report.md",
        output_dir / "09_policy_used.json",
    ]:
        log(f"  {p}")


if __name__ == "__main__":
    main()
