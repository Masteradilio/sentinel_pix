#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013A - Statistical FP/TP Diagnostics

Objetivo:
  Executar testes estatisticos classicos para descobrir diferencas robustas entre
  fraudes capturadas e falsos positivos na cascata LGBM R4 + IF/BEH/SE.

Entrada default:
  resultados/experimentos/EXP-012E/04_comparison_by_transaction.csv

Saidas:
  resultados/experimentos/EXP-013A/
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / 'backend').exists() else Path.cwd()
DEFAULT_INPUT = PROJECT_ROOT / 'resultados' / 'experimentos' / 'EXP-012E' / '04_comparison_by_transaction.csv'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'resultados' / 'experimentos' / 'EXP-013A'

ID_LIKE_SUBSTRINGS = [
    'transaction_id', 'cd_pix', 'cpf', 'cnpj', 'customer_id', 'counterparty_id',
    'session_id', 'ip_address', 'hash', 'chave_pix'
]

DERIVED_LABEL_COLS = {
    'is_fraud', 'runtime_flagged', 'shadow_exp012d_flagged', 'union_flagged',
    'intersection_flagged', 'shadow_recovers_runtime_fn', 'shadow_adds_fp_vs_runtime',
    'shadow_loses_runtime_tp', 'shadow_removes_runtime_fp', 'exp012d_pred', 'r4_pred',
    'lgbm_r4_pred', 'exp012e_pred'
}

PREFERRED_NUMERIC = [
    'lgbm_r4_score', 'r4_score', 'if_percentile', 'if_raw', 'se_score',
    'se_pattern_count', 'se_has_critico', 'se_max_pattern_score', 'behavioral_score',
    'behavioral_risk_factor_count', 'behavioral_has_velocity_factor',
    'behavioral_has_dormancy_factor', 'behavioral_has_age_value_factor',
    'behavioral_max_precision', 'vl_pix', 'hour', 'qtd_pix_pagador_7d',
    'qtd_pix_pagador_30d', 'qtd_pix_pagador_90d', 'qtd_pix_pagador_180d',
    'valor_total_pagador_7d', 'valor_total_pagador_30d', 'valor_total_pagador_90d',
    'valor_total_pagador_180d', 'max_qtd_pix_dia_pagador_7d',
    'max_qtd_pix_dia_pagador_30d', 'max_qtd_pix_dia_pagador_90d',
    'qtd_pix_mesmo_recebedor_30d', 'qtd_pix_mesmo_recebedor_90d',
    'qtd_pix_mesmo_recebedor_180d', 'valor_total_para_recebedor_30d',
    'valor_total_para_recebedor_90d', 'valor_total_para_recebedor_180d',
    'qtd_pix_recebidos_30d', 'qtd_pix_recebidos_90d', 'qtd_pix_recebidos_180d',
    'valor_total_recebido_30d', 'valor_total_recebido_90d', 'valor_total_recebido_180d',
    'soma_pagadores_distintos_dia_recebedor_180d', 'ratio_valor_media_pagador_90d',
    'ratio_valor_maximo_pagador_180d', 'mbk_available_flag', 'first_receiver_flag_real',
    'primeiro_envio_para_recebedor_180d', 'burst_daily_7d_flag', 'topaz_risk_score',
    'topaz_transacao_rejeitada', 'mbk_completeness_score'
]

PREFERRED_CATEGORICAL = [
    'ds_tipo_chave_norm', 'ds_tipo_chave', 'value_band', 'periodo_dia',
    'mbk_available_flag', 'first_receiver_flag_real', 'primeiro_envio_para_recebedor_180d',
    'burst_daily_7d_flag', 'metodo_autenticacao', 'device_name', 'app_version',
    'topaz_transacao_rejeitada', 'is_agendamento_recorrente', 'dataset_role',
    'sample_strategy', 'source_dataset', 'decisao', 'motivo', 'rule_name'
]

SEGMENT_SETS = [
    ['ds_tipo_chave_norm'], ['value_band'], ['periodo_dia'], ['mbk_available_flag'],
    ['first_receiver_flag_real'], ['mbk_available_flag', 'ds_tipo_chave_norm'],
    ['value_band', 'ds_tipo_chave_norm'], ['periodo_dia', 'value_band'],
    ['first_receiver_flag_real', 'value_band'], ['first_receiver_flag_real', 'ds_tipo_chave_norm'],
    ['mbk_available_flag', 'first_receiver_flag_real'],
    ['mbk_available_flag', 'ds_tipo_chave_norm', 'value_band']
]


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().split('.')[-1] for c in df.columns]
    if 'transaction_id' not in df.columns and 'cd_pix' in df.columns:
        df['transaction_id'] = df['cd_pix']
    if 'event_datetime' not in df.columns and 'dt_pix' in df.columns:
        df['event_datetime'] = df['dt_pix']
    if 'is_fraud' not in df.columns:
        raise RuntimeError('Coluna is_fraud ausente.')
    df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)
    for c in ['shadow_exp012d_flagged', 'runtime_flagged', 'exp012d_pred']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    if 'shadow_exp012d_flagged' not in df.columns:
        if 'exp012d_pred' in df.columns:
            df['shadow_exp012d_flagged'] = df['exp012d_pred'].astype(int)
        elif 'r4_pred' in df.columns:
            df['shadow_exp012d_flagged'] = pd.to_numeric(df['r4_pred'], errors='coerce').fillna(0).astype(int)
        else:
            raise RuntimeError('Nao encontrei shadow_exp012d_flagged, exp012d_pred ou r4_pred.')
    if 'runtime_flagged' not in df.columns:
        if 'decisao' in df.columns:
            df['runtime_flagged'] = df['decisao'].astype(str).str.upper().isin({'CONFIRMAR', 'BLOQUEAR'}).astype(int)
        else:
            df['runtime_flagged'] = 0
    if 'transaction_id' in df.columns:
        df['transaction_id'] = df['transaction_id'].astype('string').str.strip()
    return df


def bh_fdr(pvalues):
    p = np.asarray([1.0 if pd.isna(x) else float(x) for x in pvalues], dtype=float)
    if len(p) == 0:
        return []
    if multipletests is not None:
        return multipletests(p, method='fdr_bh')[1].tolist()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        prev = min(prev, ranked[i] * n / rank)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


def cliff_delta_from_u(u, nx, ny):
    if nx == 0 or ny == 0:
        return np.nan
    return float((2.0 * u / (nx * ny)) - 1.0)


def cliffs_magnitude(delta):
    if pd.isna(delta):
        return 'NA'
    ad = abs(delta)
    if ad < 0.147:
        return 'negligible'
    if ad < 0.33:
        return 'small'
    if ad < 0.474:
        return 'medium'
    return 'large'


def cramers_v(table):
    if table.size == 0 or table.sum() == 0:
        return np.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    denom = n * max(min(k - 1, r - 1), 1)
    return float(math.sqrt(chi2 / denom)) if denom > 0 else np.nan


def odds_ratio_ci(a, b, c, d):
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orv = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(orv) - 1.96 * se)
    hi = math.exp(math.log(orv) + 1.96 * se)
    return float(orv), float(lo), float(hi)


def is_id_like(col):
    low = col.lower()
    return any(s in low for s in ID_LIKE_SUBSTRINGS)


def infer_numeric_columns(df, max_cols=None):
    cols = []
    for c in PREFERRED_NUMERIC:
        if c in df.columns and c not in DERIVED_LABEL_COLS and not is_id_like(c):
            cols.append(c)
    for c in df.columns:
        if c in cols or c in DERIVED_LABEL_COLS or is_id_like(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1:
            cols.append(c)
    out = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out[:max_cols] if max_cols else out


def infer_categorical_columns(df, max_cardinality=50):
    cols = []
    for c in PREFERRED_CATEGORICAL:
        if c in df.columns and c not in DERIVED_LABEL_COLS and not is_id_like(c):
            cols.append(c)
    for c in df.columns:
        if c in cols or c in DERIVED_LABEL_COLS or is_id_like(c):
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            nunique = df[c].astype('string').nunique(dropna=True)
            if 1 < nunique <= max_cardinality:
                cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= 12:
                cols.append(c)
    out = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def prepare_group(df, mode):
    work = df.copy()
    if mode == 'shadow_tp_vs_fp':
        sub = work[work['shadow_exp012d_flagged'] == 1].copy()
        sub['group_y'] = sub['is_fraud'].astype(int)
        return sub, 'TP_shadow', 'FP_shadow'
    if mode == 'recovered_fn_vs_added_fp':
        if 'shadow_recovers_runtime_fn' not in work.columns:
            work['shadow_recovers_runtime_fn'] = ((work['is_fraud'] == 1) & (work['runtime_flagged'] == 0) & (work['shadow_exp012d_flagged'] == 1)).astype(int)
        if 'shadow_adds_fp_vs_runtime' not in work.columns:
            work['shadow_adds_fp_vs_runtime'] = ((work['is_fraud'] == 0) & (work['runtime_flagged'] == 0) & (work['shadow_exp012d_flagged'] == 1)).astype(int)
        pos = work[work['shadow_recovers_runtime_fn'] == 1].copy()
        neg = work[work['shadow_adds_fp_vs_runtime'] == 1].copy()
        pos['group_y'] = 1
        neg['group_y'] = 0
        return pd.concat([pos, neg], ignore_index=True), 'recovered_runtime_FN', 'added_FP_vs_runtime'
    raise ValueError(mode)


def run_numeric_tests(group_df, numeric_cols, pos_name, neg_name):
    rows = []
    y = group_df['group_y'].astype(int)
    for col in numeric_cols:
        if col not in group_df.columns:
            continue
        x1 = pd.to_numeric(group_df.loc[y == 1, col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        x0 = pd.to_numeric(group_df.loc[y == 0, col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(x1) < 5 or len(x0) < 5 or (x1.nunique() <= 1 and x0.nunique() <= 1):
            continue
        try:
            mw = stats.mannwhitneyu(x1, x0, alternative='two-sided')
            mw_p, u = float(mw.pvalue), float(mw.statistic)
            delta = cliff_delta_from_u(u, len(x1), len(x0))
        except Exception:
            mw_p, delta = np.nan, np.nan
        try:
            ks = stats.ks_2samp(x1, x0, alternative='two-sided', mode='auto')
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        auc_raw, auc_oriented, direction = np.nan, np.nan, 'higher_in_tp'
        try:
            valid = group_df[['group_y', col]].copy()
            valid[col] = pd.to_numeric(valid[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            valid = valid.dropna()
            if valid['group_y'].nunique() == 2 and valid[col].nunique() > 1:
                auc_raw = float(roc_auc_score(valid['group_y'].astype(int), valid[col].astype(float)))
                if auc_raw >= 0.5:
                    auc_oriented, direction = auc_raw, 'higher_in_tp'
                else:
                    auc_oriented, direction = 1.0 - auc_raw, 'lower_in_tp'
        except Exception:
            pass
        rows.append({
            'feature': col, 'group_positive': pos_name, 'group_negative': neg_name,
            'n_pos_nonnull': int(len(x1)), 'n_neg_nonnull': int(len(x0)),
            'missing_pos_rate': round(float(1 - len(x1) / max((y == 1).sum(), 1)), 6),
            'missing_neg_rate': round(float(1 - len(x0) / max((y == 0).sum(), 1)), 6),
            'mean_pos': float(x1.mean()), 'mean_neg': float(x0.mean()),
            'median_pos': float(x1.median()), 'median_neg': float(x0.median()),
            'p10_pos': float(x1.quantile(0.10)), 'p10_neg': float(x0.quantile(0.10)),
            'p25_pos': float(x1.quantile(0.25)), 'p25_neg': float(x0.quantile(0.25)),
            'p75_pos': float(x1.quantile(0.75)), 'p75_neg': float(x0.quantile(0.75)),
            'p90_pos': float(x1.quantile(0.90)), 'p90_neg': float(x0.quantile(0.90)),
            'mannwhitney_p': mw_p, 'ks_stat': ks_stat, 'ks_p': ks_p,
            'cliffs_delta': delta, 'cliffs_magnitude': cliffs_magnitude(delta),
            'auc_raw': auc_raw, 'auc_oriented': auc_oriented, 'direction': direction,
            'abs_median_diff': float(abs(x1.median() - x0.median())),
            'relative_median_ratio_pos_over_neg': float((x1.median() + 1e-9) / (x0.median() + 1e-9)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['mannwhitney_q_bh'] = bh_fdr(out['mannwhitney_p'].tolist())
    out['ks_q_bh'] = bh_fdr(out['ks_p'].tolist())
    out['robust_signal_score'] = ((out['auc_oriented'].fillna(0.5) - 0.5) * 2.0 + out['cliffs_delta'].abs().fillna(0.0) + out['ks_stat'].fillna(0.0))
    out['is_robust_signal'] = ((out['mannwhitney_q_bh'] <= 0.05) & (out['ks_q_bh'] <= 0.05) & (out['auc_oriented'].fillna(0.5) >= 0.60) & (out['cliffs_delta'].abs().fillna(0.0) >= 0.147)).astype(int)
    return out.sort_values(['is_robust_signal', 'robust_signal_score'], ascending=[False, False]).reset_index(drop=True)


def run_categorical_tests(group_df, cat_cols, pos_name, neg_name):
    test_rows, level_rows = [], []
    y = group_df['group_y'].astype(int)
    for col in cat_cols:
        if col not in group_df.columns:
            continue
        s = group_df[col].astype('string').fillna('<MISSING>')
        counts = s.value_counts(dropna=False)
        keep = set(counts[counts >= 5].index.tolist())
        s2 = s.where(s.isin(keep), other='<RARE>')
        if s2.nunique(dropna=False) < 2:
            continue
        table_df = pd.crosstab(y, s2)
        if table_df.shape[0] != 2 or table_df.shape[1] < 2:
            continue
        table = table_df.values
        try:
            chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
            chi_p, chi2_stat, min_expected = float(p), float(chi2), float(np.min(expected))
        except Exception:
            chi_p, chi2_stat, min_expected = np.nan, np.nan, np.nan
        cv = cramers_v(table)
        test_rows.append({'feature': col, 'group_positive': pos_name, 'group_negative': neg_name, 'n': int(len(s2)), 'n_levels': int(s2.nunique(dropna=False)), 'chi2': chi2_stat, 'chi2_p': chi_p, 'cramers_v': cv, 'min_expected': min_expected})
        total_tp, total_fp = int((y == 1).sum()), int((y == 0).sum())
        base_precision = total_tp / max(total_tp + total_fp, 1)
        for level in s2.unique():
            present = s2 == level
            a = int(((y == 1) & present).sum())
            b = int(((y == 0) & present).sum())
            c = int(((y == 1) & (~present)).sum())
            d = int(((y == 0) & (~present)).sum())
            if a + b < 5:
                continue
            orv, lo, hi = odds_ratio_ci(a, b, c, d)
            try:
                fisher_p = float(stats.fisher_exact([[a, b], [c, d]])[1]) if min(a, b, c, d) < 5 else np.nan
            except Exception:
                fisher_p = np.nan
            prec = a / max(a + b, 1)
            level_rows.append({'feature': col, 'level': str(level), 'tp_in_level': a, 'fp_in_level': b, 'tp_out_level': c, 'fp_out_level': d, 'n_level': a + b, 'precision_in_level': prec, 'base_precision': base_precision, 'fraud_lift_vs_base': prec / max(base_precision, 1e-9), 'tp_contribution': a / max(total_tp, 1), 'fp_contribution': b / max(total_fp, 1), 'odds_ratio': orv, 'or_ci_low': lo, 'or_ci_high': hi, 'fisher_p': fisher_p})
    tests, levels = pd.DataFrame(test_rows), pd.DataFrame(level_rows)
    if not tests.empty:
        tests['chi2_q_bh'] = bh_fdr(tests['chi2_p'].tolist())
        tests['is_robust_association'] = ((tests['chi2_q_bh'] <= 0.05) & (tests['cramers_v'].fillna(0.0) >= 0.10)).astype(int)
        tests = tests.sort_values(['is_robust_association', 'cramers_v'], ascending=[False, False]).reset_index(drop=True)
    if not levels.empty:
        levels['fisher_q_bh'] = bh_fdr(levels['fisher_p'].fillna(1.0).tolist())
        levels['absolute_fp_minus_tp'] = levels['fp_in_level'] - levels['tp_in_level']
        levels = levels.sort_values(['fraud_lift_vs_base', 'n_level'], ascending=[False, False]).reset_index(drop=True)
    return tests, levels


def build_segments(group_df, segment_sets, min_segment_n, max_tp_loss):
    work = group_df.copy()
    total_tp = int(work['group_y'].sum())
    total_fp = int((work['group_y'] == 0).sum())
    base_precision = total_tp / max(total_tp + total_fp, 1)
    segment_rows, veto_rows = [], []
    for cols in segment_sets:
        if any(c not in work.columns for c in cols):
            continue
        seg = work[cols].copy()
        for c in cols:
            seg[c] = seg[c].astype('string').fillna('<MISSING>')
        seg_key = seg.apply(lambda r: ' | '.join([f'{c}={r[c]}' for c in cols]), axis=1)
        tmp = pd.DataFrame({'segment_key': seg_key, 'group_y': work['group_y'].astype(int).values, 'n': 1})
        agg = tmp.groupby('segment_key', dropna=False).agg(n=('n', 'sum'), tp=('group_y', 'sum')).reset_index()
        agg['fp'] = agg['n'] - agg['tp']
        for _, r in agg.iterrows():
            n, tp, fp = int(r['n']), int(r['tp']), int(r['fp'])
            if n < min_segment_n:
                continue
            precision = tp / max(n, 1)
            row = {'segment_cols': '|'.join(cols), 'segment_key': r['segment_key'], 'n': n, 'tp': tp, 'fp': fp, 'precision': precision, 'fraud_lift_vs_base': precision / max(base_precision, 1e-9), 'tp_contribution': tp / max(total_tp, 1), 'fp_contribution': fp / max(total_fp, 1), 'fp_minus_tp': fp - tp}
            segment_rows.append(row)
            if tp <= max_tp_loss and fp >= max(20, min_segment_n // 2):
                veto = row.copy()
                veto['estimated_recall_after_veto'] = (total_tp - tp) / max(total_tp, 1)
                veto['estimated_fp_after_veto'] = total_fp - fp
                veto['estimated_fp_reduction'] = fp
                veto['estimated_tp_loss'] = tp
                veto_rows.append(veto)
    seg_df, veto_df = pd.DataFrame(segment_rows), pd.DataFrame(veto_rows)
    if not seg_df.empty:
        seg_df = seg_df.sort_values(['fp_minus_tp', 'fp'], ascending=[False, False]).reset_index(drop=True)
    if not veto_df.empty:
        veto_df = veto_df.sort_values(['estimated_tp_loss', 'estimated_fp_reduction'], ascending=[True, False]).reset_index(drop=True)
    return seg_df, veto_df


def numeric_threshold_hypotheses(group_df, numeric_tests, max_features=25):
    if numeric_tests.empty:
        return pd.DataFrame()
    work = group_df.copy()
    y = work['group_y'].astype(int)
    total_tp, total_fp = int(y.sum()), int((y == 0).sum())
    rows = []
    for feat in numeric_tests.head(max_features)['feature'].tolist():
        vals = pd.to_numeric(work[feat], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() < 20 or vals.nunique(dropna=True) < 2:
            continue
        direction = numeric_tests.loc[numeric_tests['feature'] == feat, 'direction'].iloc[0]
        qs = vals.dropna().quantile([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]).drop_duplicates().tolist()
        for th in qs:
            if direction == 'higher_in_tp':
                veto = vals < th
                rule = f'{feat} < {th:.8g}'
            else:
                veto = vals > th
                rule = f'{feat} > {th:.8g}'
            tp_loss = int(((y == 1) & veto.fillna(False)).sum())
            fp_removed = int(((y == 0) & veto.fillna(False)).sum())
            if fp_removed <= 0:
                continue
            rows.append({'feature': feat, 'rule': rule, 'direction': direction, 'threshold': float(th), 'tp_loss_if_veto': tp_loss, 'fp_removed_if_veto': fp_removed, 'recall_after_veto': (total_tp - tp_loss) / max(total_tp, 1), 'fp_after_veto': total_fp - fp_removed, 'fp_removed_per_tp_lost': fp_removed / max(tp_loss, 1), 'source_auc_oriented': float(numeric_tests.loc[numeric_tests['feature'] == feat, 'auc_oriented'].iloc[0]), 'source_cliffs_delta': float(numeric_tests.loc[numeric_tests['feature'] == feat, 'cliffs_delta'].iloc[0]), 'source_mannwhitney_q_bh': float(numeric_tests.loc[numeric_tests['feature'] == feat, 'mannwhitney_q_bh'].iloc[0])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['tp_loss_if_veto', 'fp_removed_if_veto'], ascending=[True, False]).reset_index(drop=True)
    return out


def make_report(summary, numeric, cats, levels, veto, hypotheses):
    lines = ['# EXP-013A - Statistical FP/TP Diagnostics', '', '## Amostra principal']
    for k in ['n_rows_input', 'n_shadow_positive', 'n_shadow_tp', 'n_shadow_fp', 'n_recovered_runtime_fn', 'n_added_fp_vs_runtime']:
        lines.append(f'- {k}: {summary.get(k)}')
    lines.append('')
    lines.append('## Top variaveis numericas robustas')
    if numeric.empty:
        lines.append('Nenhuma variavel numerica testavel encontrada.')
    else:
        cols = ['feature', 'median_pos', 'median_neg', 'mannwhitney_q_bh', 'ks_q_bh', 'cliffs_delta', 'cliffs_magnitude', 'auc_oriented', 'direction', 'is_robust_signal']
        lines.append(numeric[cols].head(20).to_markdown(index=False))
    lines.append('')
    lines.append('## Top associacoes categoricas')
    if cats.empty:
        lines.append('Nenhuma variavel categorica testavel encontrada.')
    else:
        cols = ['feature', 'n_levels', 'chi2_q_bh', 'cramers_v', 'is_robust_association']
        lines.append(cats[cols].head(20).to_markdown(index=False))
    lines.append('')
    lines.append('## Niveis/segmentos com maior lift de fraude')
    if levels.empty:
        lines.append('Nenhum nivel categorico relevante encontrado.')
    else:
        cols = ['feature', 'level', 'tp_in_level', 'fp_in_level', 'precision_in_level', 'fraud_lift_vs_base', 'odds_ratio']
        lines.append(levels[cols].head(20).to_markdown(index=False))
    lines.append('')
    lines.append('## Segmentos candidatos a veto')
    if veto.empty:
        lines.append('Nenhum segmento simples com baixo TP loss e alto FP removivel foi encontrado pelos criterios atuais.')
    else:
        cols = ['segment_cols', 'segment_key', 'tp', 'fp', 'precision', 'estimated_recall_after_veto', 'estimated_fp_reduction']
        lines.append(veto[cols].head(20).to_markdown(index=False))
    lines.append('')
    lines.append('## Hipoteses de filtros por threshold')
    if hypotheses.empty:
        lines.append('Nenhuma hipotese numerica gerada.')
    else:
        cols = ['feature', 'rule', 'tp_loss_if_veto', 'fp_removed_if_veto', 'recall_after_veto', 'fp_removed_per_tp_lost']
        lines.append(hypotheses[cols].head(30).to_markdown(index=False))
    lines.append('')
    lines.append('## Proximo passo')
    lines.append('Usar os achados no EXP-013B - Statistical Policy Search, testando filtros guiados por evidencias para reduzir FP sem derrubar recall.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default=str(DEFAULT_INPUT))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--min-segment-n', type=int, default=20)
    parser.add_argument('--max-tp-loss-segment', type=int, default=1)
    parser.add_argument('--max-numeric-cols', type=int, default=None)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f'Arquivo de entrada nao encontrado: {input_path}')
    df = normalize_columns(pd.read_csv(input_path, low_memory=False))
    g_shadow, pos_name, neg_name = prepare_group(df, 'shadow_tp_vs_fp')
    numeric_cols = infer_numeric_columns(g_shadow, args.max_numeric_cols)
    cat_cols = infer_categorical_columns(g_shadow)
    numeric = run_numeric_tests(g_shadow, numeric_cols, pos_name, neg_name)
    cats, levels = run_categorical_tests(g_shadow, cat_cols, pos_name, neg_name)
    segments, veto = build_segments(g_shadow, SEGMENT_SETS, args.min_segment_n, args.max_tp_loss_segment)
    hypotheses = numeric_threshold_hypotheses(g_shadow, numeric, 25)
    g_rec, pos2, neg2 = prepare_group(df, 'recovered_fn_vs_added_fp')
    numeric2 = run_numeric_tests(g_rec, numeric_cols, pos2, neg2)
    cats2, _ = run_categorical_tests(g_rec, cat_cols, pos2, neg2)
    summary = {
        'experiment': 'EXP-013A', 'status': 'DONE', 'input_path': str(input_path),
        'n_rows_input': int(len(df)), 'n_shadow_positive': int(len(g_shadow)),
        'n_shadow_tp': int((g_shadow['group_y'] == 1).sum()),
        'n_shadow_fp': int((g_shadow['group_y'] == 0).sum()),
        'n_recovered_runtime_fn': int((g_rec['group_y'] == 1).sum()) if not g_rec.empty else 0,
        'n_added_fp_vs_runtime': int((g_rec['group_y'] == 0).sum()) if not g_rec.empty else 0,
        'n_numeric_features_tested': int(len(numeric)), 'n_categorical_features_tested': int(len(cats)),
        'n_segment_rows': int(len(segments)), 'n_veto_candidate_segments': int(len(veto)),
        'n_threshold_hypotheses': int(len(hypotheses)),
        'top_numeric_robust_signals': numeric.head(10).to_dict(orient='records') if not numeric.empty else [],
        'top_categorical_associations': cats.head(10).to_dict(orient='records') if not cats.empty else [],
    }
    numeric.to_csv(output_dir / '01_numeric_tests_shadow_tp_vs_fp.csv', index=False)
    cats.to_csv(output_dir / '02_categorical_tests_shadow_tp_vs_fp.csv', index=False)
    levels.to_csv(output_dir / '03_categorical_level_odds_shadow_tp_vs_fp.csv', index=False)
    segments.to_csv(output_dir / '04_segment_analysis_shadow_positives.csv', index=False)
    veto.to_csv(output_dir / '05_veto_candidate_segments.csv', index=False)
    hypotheses.to_csv(output_dir / '06_feature_threshold_hypotheses.csv', index=False)
    numeric2.to_csv(output_dir / '07_numeric_tests_recovered_fn_vs_added_fp.csv', index=False)
    cats2.to_csv(output_dir / '08_categorical_tests_recovered_fn_vs_added_fp.csv', index=False)
    dump_json(summary, output_dir / '00_run_summary.json')
    dump_json({'numeric_columns': numeric_cols, 'categorical_columns': cat_cols, 'segment_sets': SEGMENT_SETS, 'derived_label_cols_excluded': sorted(DERIVED_LABEL_COLS), 'id_like_substrings_excluded': ID_LIKE_SUBSTRINGS, 'tests': {'numeric': ['Mann-Whitney U', 'Kolmogorov-Smirnov', "Cliff's Delta", 'univariate ROC-AUC', 'Benjamini-Hochberg FDR'], 'categorical': ['Chi-square', "Cramer's V", 'Odds Ratio', 'Fisher exact for sparse 2x2 when applicable', 'Benjamini-Hochberg FDR'], 'segments': ['precision/lift/contribution', 'veto candidate simulation']}}, output_dir / '10_data_dictionary_used.json')
    (output_dir / '09_distribution_quality_report.md').write_text(make_report(summary, numeric, cats, levels, veto, hypotheses), encoding='utf-8')
    print('=' * 80)
    print('EXP-013A concluido')
    print('=' * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nArquivos principais:')
    for p in [output_dir / '00_run_summary.json', output_dir / '01_numeric_tests_shadow_tp_vs_fp.csv', output_dir / '02_categorical_tests_shadow_tp_vs_fp.csv', output_dir / '03_categorical_level_odds_shadow_tp_vs_fp.csv', output_dir / '04_segment_analysis_shadow_positives.csv', output_dir / '05_veto_candidate_segments.csv', output_dir / '06_feature_threshold_hypotheses.csv', output_dir / '09_distribution_quality_report.md']:
        print(f'  {p}')


if __name__ == '__main__':
    main()
