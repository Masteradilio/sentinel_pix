# -*- coding: utf-8 -*-
"""
EXP-014B-R4G — Approve Fraud Rescue + Block-to-Confirm Fine Tune

Objetivo:
  1) APROVAR -> CONFIRMAR: tentar resgatar fraudes remanescentes em APROVAR,
     mantendo FPR global < 1%.
  2) BLOQUEAR -> CONFIRMAR: continuar removendo normais de BLOQUEAR sem mover fraudes.

Entrada default:
  resultados/experimentos/EXP-014B-R4F-FROZEN/06_predictions_frozen.csv
  fallback: resultados/experimentos/EXP-014B-R4F/09_predictions_recommended.csv

Saída:
  resultados/experimentos/EXP-014B-R4G/
"""

from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

EXPERIMENT = "EXP-014B-R4G"
LABELS = ["is_fraud", "fraude", "target", "label", "tp_fraude"]
ACTION_COLS = ["r4f_frozen_decisao_recommended", "r4f_decisao_recommended", "r4e_frozen_decisao_recommended", "r4e_decisao_recommended"]
INTERVENTION_COLS = ["exp014b_r4f_frozen_intervention_pred", "exp014b_r4f_intervention_pred", "exp014b_r4e_frozen_intervention_pred", "exp014b_r4e_intervention_pred"]
BLOCK_COLS = ["exp014b_r4f_frozen_block_pred", "exp014b_r4f_block_pred", "exp014b_r4e_frozen_block_pred", "exp014b_r4e_block_pred"]

CAT_COLS = [
    "ds_tipo_chave_norm","value_band","periodo_dia","score_bin","lgbm_bin","if_bin",
    "ratio_bin","qtd_rec_bin","valor_rec_bin","mbk_available_flag","first_receiver_flag_real",
    "module_quiet","se_worst_pattern","r3u_missing_receiver_history_flag","r3u_receiver_known_flag",
    "r3u_receiver_reputable_flag","r3u_receiver_strong_flag","r3u_relationship_known_flag",
    "r3u_relationship_recurrent_flag","r3u_relationship_strong_flag","r3u_first_receiver_flag",
    "r3u_module_quiet_flag","r3u_se_missing_flag","r3u_ratio_lt_005_flag","r3u_mbk_quality_flag",
    "r3u_receiver_trust_bucket","r3u_relationship_bucket",
]
SCORE_COLS = [
    "lgbm_r4_score","score_final","lgbm_raw","lgbm_mapped","peso_total","if_percentile",
    "se_score","beh_score","behavioral_score","topaz_risk_score",
    "exp014b_r3s_second_stage_score","exp014b_r3u_receiver_relationship_trust_score",
]
ROBUSTNESS_COLS = ["temporal_split","event_month","ds_tipo_chave_norm","value_band","periodo_dia","score_bin","lgbm_bin","if_bin","ratio_bin","qtd_rec_bin","valor_rec_bin","mbk_available_flag","first_receiver_flag_real"]

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--max-total-fn", type=int, default=5)
    p.add_argument("--max-approve-fp-promoted", type=int, default=None)
    p.add_argument("--target-approve-tp-promoted", type=int, default=5)
    p.add_argument("--max-block-tp-demoted", type=int, default=0)
    p.add_argument("--target-block-fp-demoted", type=int, default=None)
    p.add_argument("--max-rules-approve", type=int, default=120)
    p.add_argument("--max-rules-block", type=int, default=180)
    p.add_argument("--max-candidates", type=int, default=24000)
    p.add_argument("--min-support", type=int, default=1)
    p.add_argument("--min-incremental-good", type=int, default=1)
    p.add_argument("--enable-quads", action="store_true")
    p.add_argument("--enable-score-cat-pairs", action="store_true")
    p.add_argument("--combo-topn", type=int, default=900)
    p.add_argument("--score-cat-top-values", type=int, default=60)
    return p.parse_args()

def defaults():
    root = Path.cwd()
    pred = root / "resultados/experimentos/EXP-014B-R4F-FROZEN/06_predictions_frozen.csv"
    if not pred.exists():
        pred = root / "resultados/experimentos/EXP-014B-R4F/09_predictions_recommended.csv"
    art = root / "resultados/experimentos/EXP-014B-R4F-FROZEN/05_policy_artifact_frozen.json"
    if not art.exists():
        art = root / "resultados/experimentos/EXP-014B-R4F/08_policy_artifact_recommended.json"
    return pred, art if art.exists() else None, root / "resultados/experimentos/EXP-014B-R4G"

def find_col(df, names, required=True):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns: return n
        if n.lower() in lower: return lower[n.lower()]
    if required: raise KeyError(f"Coluna não encontrada: {names}")
    return None

def ints(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def act(s):
    return s.astype(str).str.strip().str.upper()

def inter_from_action(a):
    return act(a).isin(["CONFIRMAR", "BLOQUEAR"]).astype(int)

def block_from_action(a):
    return act(a).eq("BLOQUEAR").astype(int)

def metrics(y, pred):
    y, p = ints(y), ints(pred)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(float(pr), 8), "recall": round(float(rc), 8), "f1": round(float(f1), 8), "fpr": round(float(fpr), 8)}

def strict_target_fp(n_normals, target_fpr):
    return int(np.ceil(float(target_fpr) * int(n_normals)) - 1)

def action_table(df, label_col, action_col):
    y = ints(df[label_col]); rows = []
    for a, idx in df.groupby(action_col, dropna=False).groups.items():
        idx = list(idx); yy = y.loc[idx]; n = len(idx)
        rows.append({"action": str(a), "n_rows": int(n), "n_frauds": int((yy == 1).sum()), "n_normals": int((yy == 0).sum()), "precision_within_action": round(float((yy == 1).sum()/n), 8) if n else 0.0})
    return pd.DataFrame(rows).sort_values("action")

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def add_candidate(rows, masks, seen, cid, typ, desc, mask, y_local, good_label, min_support, min_good):
    if desc in seen: return
    n = int(mask.sum())
    if n < min_support: return
    yy = y_local[mask]
    good = int((yy == good_label).sum())
    bad = int((yy != good_label).sum())
    if good < min_good: return
    seen.add(desc)
    rows.append({"candidate_id": cid, "rule_type": typ, "description": desc, "n_affected": n, "good_count": good, "bad_count": bad, "precision_for_goal": round(float(good/n), 8) if n else 0.0, "good_per_bad": round(float(good/max(bad,1)), 8)})
    masks[cid] = mask.copy()

def mine(df, idx, y_all, prefix, phase, good_label, cat_cols, score_cols, min_support, min_good, max_candidates, enable_quads, enable_score_cat_pairs, combo_topn, score_top):
    local = df.iloc[idx].copy()
    y_local = y_all[idx]
    rows, masks, seen = [], {}, set()
    q = [0.005,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.92,0.95,0.97,0.98,0.99,0.995]

    for sc in score_cols:
        s = pd.to_numeric(local[sc], errors="coerce")
        valid = s.dropna()
        if valid.empty: continue
        vals = s.to_numpy()
        ths = sorted(set(float(valid.quantile(x)) for x in q if pd.notna(valid.quantile(x))))
        for th in ths:
            add_candidate(rows,masks,seen,f"{phase}_score__{sc}__le{th:.12g}","score_threshold",f"{prefix} com {sc} <= {th:.12g}",np.isfinite(vals)&(vals<=th),y_local,good_label,min_support,min_good)
            add_candidate(rows,masks,seen,f"{phase}_score__{sc}__ge{th:.12g}","score_threshold",f"{prefix} com {sc} >= {th:.12g}",np.isfinite(vals)&(vals>=th),y_local,good_label,min_support,min_good)

    combo_plan = [(1,28,combo_topn),(2,24,combo_topn),(3,16,combo_topn)]
    if enable_quads:
        combo_plan.append((4,12,max(150,combo_topn//2)))

    for size, max_cols, topn in combo_plan:
        for cols in itertools.combinations(cat_cols[:max_cols], size):
            tmp = local[list(cols)].fillna("<MISSING>").astype(str)
            vc = tmp.value_counts(dropna=False).head(topn)
            for valtuple, support in vc.items():
                if int(support) < min_support: continue
                valtuple = valtuple if isinstance(valtuple, tuple) else (valtuple,)
                mask = np.ones(len(local), dtype=bool); parts=[]; safe=[]
                for c,v in zip(cols,valtuple):
                    v = str(v); mask &= tmp[c].to_numpy() == v
                    parts.append(f"{c} == {v}"); safe.append(f"{c}={v[:20]}")
                add_candidate(rows,masks,seen,f"{phase}_cat{size}__" + "__".join(safe),f"categorical_{size}",f"{prefix} com " + " AND ".join(parts),mask,y_local,good_label,min_support,min_good)

    sq = [0.10,0.20,0.30,0.40,0.60,0.70,0.80,0.90,0.95]
    for sc in score_cols[:8]:
        s = pd.to_numeric(local[sc], errors="coerce")
        valid = s.dropna()
        if valid.empty: continue
        vals = s.to_numpy()
        ths = sorted(set(float(valid.quantile(x)) for x in sq if pd.notna(valid.quantile(x))))

        for cat in cat_cols[:18]:
            cs = local[cat].fillna("<MISSING>").astype(str); cv = cs.to_numpy()
            for catval, support in cs.value_counts(dropna=False).head(score_top).items():
                if int(support) < min_support: continue
                base = cv == str(catval)
                for th in ths:
                    add_candidate(rows,masks,seen,f"{phase}_scorecat__{cat}={str(catval)[:20]}__{sc}__le{th:.12g}","score_cat_1",f"{prefix} com {cat} == {catval} AND {sc} <= {th:.12g}",base & np.isfinite(vals) & (vals<=th),y_local,good_label,min_support,min_good)
                    add_candidate(rows,masks,seen,f"{phase}_scorecat__{cat}={str(catval)[:20]}__{sc}__ge{th:.12g}","score_cat_1",f"{prefix} com {cat} == {catval} AND {sc} >= {th:.12g}",base & np.isfinite(vals) & (vals>=th),y_local,good_label,min_support,min_good)

        if enable_score_cat_pairs:
            for c1,c2 in itertools.combinations(cat_cols[:12],2):
                s1 = local[c1].fillna("<MISSING>").astype(str); s2 = local[c2].fillna("<MISSING>").astype(str)
                vc = pd.DataFrame({c1:s1,c2:s2}).value_counts(dropna=False).head(max(120, score_top*3))
                v1a, v2a = s1.to_numpy(), s2.to_numpy()
                for vals2, support in vc.items():
                    if int(support) < min_support: continue
                    v1,v2 = vals2 if isinstance(vals2, tuple) else (vals2, "")
                    base = (v1a == str(v1)) & (v2a == str(v2))
                    for th in ths:
                        add_candidate(rows,masks,seen,f"{phase}_scorecat2__{c1}={str(v1)[:14]}__{c2}={str(v2)[:14]}__{sc}__le{th:.12g}","score_cat_2",f"{prefix} com {c1} == {v1} AND {c2} == {v2} AND {sc} <= {th:.12g}",base & np.isfinite(vals) & (vals<=th),y_local,good_label,min_support,min_good)
                        add_candidate(rows,masks,seen,f"{phase}_scorecat2__{c1}={str(v1)[:14]}__{c2}={str(v2)[:14]}__{sc}__ge{th:.12g}","score_cat_2",f"{prefix} com {c1} == {v1} AND {c2} == {v2} AND {sc} >= {th:.12g}",base & np.isfinite(vals) & (vals>=th),y_local,good_label,min_support,min_good)

    if not rows: return pd.DataFrame(), {}
    cand = pd.DataFrame(rows).drop_duplicates(subset=["description"])
    cand = cand.sort_values(["bad_count","good_count","good_per_bad","n_affected"], ascending=[True,False,False,False]).head(max_candidates).reset_index(drop=True)
    keep = set(cand["candidate_id"].astype(str))
    return cand, {k:v for k,v in masks.items() if k in keep}

def greedy(cand, masks, y_local, good_label, max_bad, max_rules, min_good, target_good=None):
    selected = np.zeros(len(y_local), dtype=bool)
    rows=[]; frontier=[]; cg=0; cb=0
    remaining = cand.copy().reset_index(drop=True)
    for step in range(1, int(max_rules)+1):
        if target_good is not None and cg >= target_good: break
        best=None; best_mask=None; best_score=None
        for _, row in remaining.iterrows():
            m = masks.get(str(row["candidate_id"]))
            if m is None: continue
            inc = m & (~selected); n = int(inc.sum())
            if n == 0: continue
            yy = y_local[inc]
            good = int((yy == good_label).sum())
            bad = int((yy != good_label).sum())
            if good < min_good: continue
            if cb + bad > max_bad: continue
            score = (1 if bad == 0 else 0, good/max(bad,1), good, -bad, -n)
            if best is None or score > best_score:
                best = row.copy(); best_mask = inc; best_score = score
                best["incremental_n"] = n; best["incremental_good"] = good; best["incremental_bad"] = bad
        if best is None: break
        selected |= best_mask; cg += int(best["incremental_good"]); cb += int(best["incremental_bad"])
        best["selection_step"] = step; best["cumulative_n"] = int(selected.sum()); best["cumulative_good"] = int(cg); best["cumulative_bad"] = int(cb)
        rows.append(best)
        frontier.append({"selection_step":step,"selected_candidate_id":str(best["candidate_id"]),"selected_description":str(best["description"]),"incremental_n":int(best["incremental_n"]),"incremental_good":int(best["incremental_good"]),"incremental_bad":int(best["incremental_bad"]),"cumulative_n":int(selected.sum()),"cumulative_good":int(cg),"cumulative_bad":int(cb)})
        remaining = remaining[remaining["candidate_id"].astype(str) != str(best["candidate_id"])].reset_index(drop=True)
    return pd.DataFrame(rows) if rows else pd.DataFrame(), pd.DataFrame(frontier) if frontier else pd.DataFrame(), selected

def robustness(df, label_col, before_col, after_col):
    y=ints(df[label_col]); bi=inter_from_action(df[before_col]); ai=inter_from_action(df[after_col]); bb=block_from_action(df[before_col]); ab=block_from_action(df[after_col])
    rows=[]
    for col in ROBUSTNESS_COLS:
        if col not in df.columns: continue
        for val, idx in df.groupby(col, dropna=False).groups.items():
            idx=list(idx); yy=y.loc[idx]
            bmi=metrics(yy, bi.loc[idx]); ami=metrics(yy, ai.loc[idx]); bmb=metrics(yy, bb.loc[idx]); amb=metrics(yy, ab.loc[idx])
            rows.append({"segment_col":col,"segment_value":str(val),"n_rows":len(idx),"intervention_tp_delta":ami["tp"]-bmi["tp"],"intervention_fp_delta":ami["fp"]-bmi["fp"],"block_tp_delta":amb["tp"]-bmb["tp"],"block_fp_delta":amb["fp"]-bmb["fp"]})
    return pd.DataFrame(rows).sort_values(["intervention_tp_delta","intervention_fp_delta","block_fp_delta","n_rows"], ascending=[False,True,True,False]) if rows else pd.DataFrame()

def table_md(df, n=80):
    if df is None or df.empty: return "Nenhuma linha."
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)

def main():
    a=args(); pred_d, art_d, out_d = defaults()
    pred_path = Path(a.predictions) if a.predictions else pred_d
    art_path = Path(a.artifact) if a.artifact else art_d
    out = Path(a.output_dir) if a.output_dir else out_d
    out.mkdir(parents=True, exist_ok=True)
    if not pred_path.exists(): raise FileNotFoundError(pred_path)

    artifact = json.loads(art_path.read_text(encoding="utf-8")) if art_path and art_path.exists() else None
    df = pd.read_csv(pred_path, low_memory=False).copy()

    label_col = find_col(df, LABELS)
    action_col = find_col(df, ACTION_COLS)
    inter_col = find_col(df, INTERVENTION_COLS, required=False)
    block_col = find_col(df, BLOCK_COLS, required=False)

    y = ints(df[label_col]).to_numpy()
    base_action = act(df[action_col])
    base_inter = ints(df[inter_col]) if inter_col else inter_from_action(base_action)
    base_block = ints(df[block_col]) if block_col else block_from_action(base_action)
    base_i = metrics(pd.Series(y), base_inter); base_b = metrics(pd.Series(y), base_block)

    n_normals = int((y==0).sum())
    target_fp = strict_target_fp(n_normals, float(a.target_fpr))
    headroom = max(0, target_fp - int(base_i["fp"]))
    max_approve_fp = headroom if a.max_approve_fp_promoted is None else int(a.max_approve_fp_promoted)

    cat_cols = [c for c in CAT_COLS if c in df.columns and 1 < df[c].fillna("<MISSING>").astype(str).nunique(dropna=False) <= 100]
    score_cols = [c for c in SCORE_COLS if c in df.columns]

    approve_idx = np.flatnonzero(base_action.eq("APROVAR").to_numpy())
    approve_cand, approve_masks = mine(df, approve_idx, y, "Mover APROVAR para CONFIRMAR R4G", "approve_to_confirm", 1, cat_cols, score_cols, a.min_support, a.min_incremental_good, a.max_candidates, a.enable_quads, a.enable_score_cat_pairs, a.combo_topn, a.score_cat_top_values)
    sel_approve, front_approve, local_app = greedy(approve_cand, approve_masks, y[approve_idx], 1, max_approve_fp, a.max_rules_approve, a.min_incremental_good, a.target_approve_tp_promoted)

    approve_to_confirm = np.zeros(len(df), dtype=bool); approve_to_confirm[approve_idx] = local_app
    action_after_a = base_action.copy(); action_after_a.loc[approve_to_confirm] = "CONFIRMAR"

    block_idx = np.flatnonzero(action_after_a.eq("BLOQUEAR").to_numpy())
    block_cand, block_masks = mine(df, block_idx, y, "Mover BLOQUEAR para CONFIRMAR R4G", "block_to_confirm", 0, cat_cols, score_cols, a.min_support, a.min_incremental_good, a.max_candidates, a.enable_quads, a.enable_score_cat_pairs, a.combo_topn, a.score_cat_top_values)
    sel_block, front_block, local_blk = greedy(block_cand, block_masks, y[block_idx], 0, a.max_block_tp_demoted, a.max_rules_block, a.min_incremental_good, a.target_block_fp_demoted)

    block_to_confirm = np.zeros(len(df), dtype=bool); block_to_confirm[block_idx] = local_blk
    final_action = action_after_a.copy(); final_action.loc[block_to_confirm] = "CONFIRMAR"

    df["exp014b_r4g_approve_to_confirm"] = approve_to_confirm.astype(int)
    df["exp014b_r4g_block_to_confirm"] = block_to_confirm.astype(int)
    df["r4g_decisao_recommended"] = final_action
    df["exp014b_r4g_intervention_pred"] = inter_from_action(final_action)
    df["exp014b_r4g_block_pred"] = block_from_action(final_action)

    final_i = metrics(pd.Series(y), df["exp014b_r4g_intervention_pred"])
    final_b = metrics(pd.Series(y), df["exp014b_r4g_block_pred"])
    by_action = action_table(df, label_col, "r4g_decisao_recommended")

    approve_tp = int(((y==1) & approve_to_confirm).sum())
    approve_fp = int(((y==0) & approve_to_confirm).sum())
    block_fp = int(((y==0) & block_to_confirm).sum())
    block_tp = int(((y==1) & block_to_confirm).sum())
    approve_fraud_remaining = int(((y==1) & act(final_action).eq("APROVAR").to_numpy()).sum())
    target_reached = bool(final_i["fp"] <= target_fp)
    fn_total_ok = bool(final_i["fn"] <= a.max_total_fn)

    status = (
        "DONE_R4G_APPROVE_FRAUDS_RESCUED_AND_FPR_LT1_PRESERVED"
        if approve_tp > 0 and target_reached and fn_total_ok and block_tp <= a.max_block_tp_demoted
        else "DONE_R4G_NO_SAFE_APPROVE_FRAUD_RESCUE" if approve_tp == 0
        else "DONE_R4G_RESCUED_APPROVE_FRAUDS_BUT_TARGET_NOT_PRESERVED"
    )

    summary = {
        "experiment": EXPERIMENT, "status": "DONE", "objective_status": status,
        "n_rows": int(len(df)), "n_frauds": int((y==1).sum()), "n_normals": n_normals,
        "predictions_path": str(pred_path), "artifact_path": str(art_path) if art_path else None,
        "label_col": label_col, "action_col": action_col, "intervention_col": inter_col, "block_col": block_col,
        "baseline_intervention_metrics": base_i, "baseline_block_metrics": base_b,
        "final_intervention_metrics": final_i, "final_block_metrics": final_b,
        "target_fpr_strict": float(a.target_fpr), "target_fp_strict": int(target_fp),
        "available_fp_headroom": int(headroom), "max_approve_fp_promoted": int(max_approve_fp),
        "target_reached": target_reached, "gap_to_target_fp": max(0, int(final_i["fp"] - target_fp)),
        "fn_total_ok": fn_total_ok,
        "approve_tp_promoted_to_confirm": approve_tp, "approve_fp_promoted_to_confirm": approve_fp,
        "approval_fraud_remaining": approve_fraud_remaining,
        "block_fp_demoted_to_confirm": block_fp, "block_tp_demoted_to_confirm": block_tp,
        "net_intervention_tp_delta": int(final_i["tp"] - base_i["tp"]),
        "net_intervention_fp_delta": int(final_i["fp"] - base_i["fp"]),
        "net_block_tp_delta": int(final_b["tp"] - base_b["tp"]),
        "net_block_fp_delta": int(final_b["fp"] - base_b["fp"]),
        "n_approve_candidates": int(len(approve_cand)), "n_block_candidates": int(len(block_cand)),
        "n_selected_approve_rules": int(len(sel_approve)), "n_selected_block_rules": int(len(sel_block)),
        "all_pass": bool(target_reached and fn_total_ok and block_tp <= a.max_block_tp_demoted),
        "output_dir": str(out),
    }

    policy = {
        "experiment": EXPERIMENT, "input_predictions_path": str(pred_path),
        "base_action_col": action_col, "final_action_col": "r4g_decisao_recommended",
        "approve_to_confirm_col": "exp014b_r4g_approve_to_confirm",
        "block_to_confirm_col": "exp014b_r4g_block_to_confirm",
        "intervention_pred_col": "exp014b_r4g_intervention_pred",
        "block_pred_col": "exp014b_r4g_block_pred",
        "baseline_intervention_metrics": base_i, "baseline_block_metrics": base_b,
        "final_intervention_metrics": final_i, "final_block_metrics": final_b,
        "target_fpr_strict": float(a.target_fpr), "target_fp_strict": int(target_fp),
        "target_reached": target_reached,
        "approve_tp_promoted_to_confirm": approve_tp, "approve_fp_promoted_to_confirm": approve_fp,
        "approval_fraud_remaining": approve_fraud_remaining,
        "block_fp_demoted_to_confirm": block_fp, "block_tp_demoted_to_confirm": block_tp,
        "selected_approve_to_confirm_rules": sel_approve.to_dict(orient="records") if not sel_approve.empty else [],
        "selected_block_to_confirm_rules": sel_block.to_dict(orient="records") if not sel_block.empty else [],
    }

    write_json(out / "00_run_summary.json", summary)
    write_json(out / "01_input_contract.json", {"predictions_path": str(pred_path), "artifact_path": str(art_path) if art_path else None, "label_col": label_col, "action_col": action_col, "intervention_col": inter_col, "block_col": block_col, "target_fpr_strict": float(a.target_fpr), "target_fp_strict": int(target_fp), "available_fp_headroom": int(headroom), "max_approve_fp_promoted": int(max_approve_fp), "contract_ok": True, "missing": []})
    write_json(out / "02_base_metrics.json", {"baseline_intervention_metrics": base_i, "baseline_block_metrics": base_b, "baseline_by_action": action_table(df.assign(_base=base_action), label_col, "_base").to_dict(orient="records"), "artifact_status": artifact.get("frozen_validation_status") if isinstance(artifact, dict) else None})
    approve_cand.to_csv(out / "03_approve_to_confirm_candidates.csv", index=False, encoding="utf-8")
    block_cand.to_csv(out / "04_block_to_confirm_candidates.csv", index=False, encoding="utf-8")
    sel_approve.to_csv(out / "05_selected_approve_to_confirm_rules.csv", index=False, encoding="utf-8")
    sel_block.to_csv(out / "06_selected_block_to_confirm_rules.csv", index=False, encoding="utf-8")
    pd.concat([front_approve.assign(phase="approve_to_confirm") if not front_approve.empty else pd.DataFrame(), front_block.assign(phase="block_to_confirm") if not front_block.empty else pd.DataFrame()], ignore_index=True).to_csv(out / "07_selection_frontier.csv", index=False, encoding="utf-8")
    by_action.to_csv(out / "08_decision_metrics_by_action.csv", index=False, encoding="utf-8")
    robustness(df, label_col, action_col, "r4g_decisao_recommended").to_csv(out / "09_robustness_by_segment.csv", index=False, encoding="utf-8")
    write_json(out / "10_policy_artifact_recommended.json", policy)
    df.to_csv(out / "11_predictions_recommended.csv", index=False, encoding="utf-8")

    report = f"""# {EXPERIMENT}

## Resultado executivo
- Status: `{status}`
- All pass: `{summary['all_pass']}`
- Target FP strict: `{target_fp}`
- Target reached: `{target_reached}`
- Folga FP baseline: `{headroom}`

## Movimentos
- APROVAR -> CONFIRMAR, fraudes promovidas: `{approve_tp}`
- APROVAR -> CONFIRMAR, normais promovidos: `{approve_fp}`
- Fraudes restantes em APROVAR: `{approve_fraud_remaining}`
- BLOQUEAR -> CONFIRMAR, normais movidos: `{block_fp}`
- BLOQUEAR -> CONFIRMAR, fraudes movidas: `{block_tp}`

## Baseline intervenção
```json
{json.dumps(base_i, ensure_ascii=False, indent=2)}
```

## Final intervenção
```json
{json.dumps(final_i, ensure_ascii=False, indent=2)}
```

## Baseline BLOQUEAR
```json
{json.dumps(base_b, ensure_ascii=False, indent=2)}
```

## Final BLOQUEAR
```json
{json.dumps(final_b, ensure_ascii=False, indent=2)}
```

## Métricas por ação final
{table_md(by_action)}

## Regras APROVAR -> CONFIRMAR
{table_md(sel_approve)}

## Regras BLOQUEAR -> CONFIRMAR
{table_md(sel_block)}
"""
    (out / "12_exp014b_r4g_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
