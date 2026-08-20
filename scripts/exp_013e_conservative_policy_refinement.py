
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-013E - Conservative Policy Refinement

Starts from EXP-013D selected policy CONSERVATIVE_NO_RECEIVER_VALUE.
Searches only local, auditable refinements:
  1) restore exceptions to recover true positives vetoed by the conservative policy;
  2) extra low-risk vetoes to remove remaining false positives.

No model training and no broad retuning.
"""
from __future__ import annotations
import argparse, json, math, re, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent / 'backend').exists() else Path.cwd()
DEFAULT_INPUT = PROJECT_ROOT / 'resultados' / 'experimentos' / 'EXP-012E' / '04_comparison_by_transaction.csv'
DEFAULT_POLICY = PROJECT_ROOT / 'resultados' / 'experimentos' / 'EXP-013D' / '09_selected_policy_artifact.json'
DEFAULT_OUTPUT = PROJECT_ROOT / 'resultados' / 'experimentos' / 'EXP-013E'
FLAGGED_DECISIONS = {'CONFIRMAR','BLOQUEAR'}

@dataclass
class Action:
    action_id: str
    action_type: str
    family: str
    description: str
    mask: np.ndarray
    tp_delta: int
    fp_delta: int
    params: dict[str, Any]

@dataclass
class State:
    pred: np.ndarray
    restore_ids: tuple[int, ...]
    veto_ids: tuple[int, ...]
    metrics: dict[str, Any]

def log(s: str):
    print(s, flush=True)

def dump_json(obj: Any, path: Path):
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
        raise RuntimeError('Missing is_fraud column.')
    df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)
    if 'shadow_exp012d_flagged' not in df.columns:
        for c in ['exp012d_pred','r4_pred','lgbm_r4_pred']:
            if c in df.columns:
                df['shadow_exp012d_flagged'] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
                break
    if 'shadow_exp012d_flagged' not in df.columns:
        raise RuntimeError('Missing shadow_exp012d_flagged/exp012d_pred/r4_pred/lgbm_r4_pred.')
    df['shadow_exp012d_flagged'] = pd.to_numeric(df['shadow_exp012d_flagged'], errors='coerce').fillna(0).astype(int)
    if 'runtime_flagged' not in df.columns:
        if 'decisao' in df.columns:
            df['runtime_flagged'] = df['decisao'].astype(str).str.upper().isin(FLAGGED_DECISIONS).astype(int)
        else:
            df['runtime_flagged'] = 0
    df['runtime_flagged'] = pd.to_numeric(df['runtime_flagged'], errors='coerce').fillna(0).astype(int)
    if 'transaction_id' in df.columns:
        df['transaction_id'] = df['transaction_id'].astype('string').str.strip()
    for c in ['event_datetime','data_pix']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df.reset_index(drop=True)

def pick_col(df: pd.DataFrame, names):
    if isinstance(names, str): names=[names]
    for n in names:
        if n in df.columns: return n
    return None

def num(df: pd.DataFrame, names, default=0.0) -> pd.Series:
    c=pick_col(df,names)
    if c is None: return pd.Series([default]*len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors='coerce').replace([np.inf,-np.inf], np.nan).fillna(default).astype(float)

def text(df: pd.DataFrame, names, default='<MISSING>') -> pd.Series:
    c=pick_col(df,names)
    if c is None: return pd.Series([default]*len(df), index=df.index, dtype='string')
    return df[c].astype('string').fillna(default).astype(str)

def boolish(df: pd.DataFrame, names, default=False) -> pd.Series:
    c=pick_col(df,names)
    if c is None: return pd.Series([default]*len(df), index=df.index)
    s=df[c]
    if s.dtype == bool: return s.fillna(default)
    return s.astype(str).str.upper().isin({'1','1.0','TRUE','T','SIM','YES','Y'})

def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true=np.asarray(y_true).astype(int); y_pred=np.asarray(y_pred).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),
            'precision':round(float(precision_score(y_true,y_pred,zero_division=0)),8),
            'recall':round(float(recall_score(y_true,y_pred,zero_division=0)),8),
            'f1':round(float(f1_score(y_true,y_pred,zero_division=0)),8),
            'fpr':round(float(fp/max(fp+tn,1)),8)}

def strong_preserve_mask(df: pd.DataFrame) -> np.ndarray:
    se=num(df,['se_score_x','se_score_y','se_score'],0.0)
    sec=num(df,['se_patterns_count','se_pattern_count'],0.0)
    beh=num(df,['beh_score','behavioral_score'],0.0)
    behc=num(df,['beh_factors_count','behavioral_risk_factor_count'],0.0)
    runtime=num(df,'runtime_flagged',0.0)
    cascade=boolish(df,'cascade_triggered',False)
    decisao=text(df,'decisao','').str.upper()
    return ((se>=65)|(sec>=2)|(beh>=45)|(behc>=2)|(runtime>=1)|decisao.isin(FLAGGED_DECISIONS)|cascade).to_numpy(dtype=bool)

def parse_params(rule):
    raw=rule.get('params_json','{}')
    if isinstance(raw, dict): return raw
    try: return json.loads(raw)
    except Exception: return {}

def get_lgbm_score(df):
    return num(df,['lgbm_r4_score','r4_score','lgbm_mapped','lgbm_raw'],0.0)

def get_if_percentile(df):
    return num(df,['if_percentile_x','if_percentile_y','if_percentile'],0.0)

def apply_rule_mask(df: pd.DataFrame, rule: dict[str,Any], threshold_multiplier=1.0) -> np.ndarray:
    fam=str(rule.get('family','')); p=parse_params(rule); preserve=strong_preserve_mask(df)
    if fam=='numeric':
        vals=num(df,p.get('feature'),np.nan); th=float(p.get('threshold'))*threshold_multiplier; op=p.get('op')
        mask=(vals<th).to_numpy(dtype=bool) if op=='lt' else (vals>th).to_numpy(dtype=bool)
        if bool(p.get('preserve',False)): mask &= ~preserve
        return mask
    if fam=='segment':
        mask=np.ones(len(df),dtype=bool)
        for c,v in zip(p.get('segment_cols',[]), p.get('segment_values',[])):
            mask &= (text(df,c)==str(v)).to_numpy(dtype=bool)
        if bool(p.get('preserve',False)): mask &= ~preserve
        return mask
    if fam=='segment_lgbm':
        mask=np.ones(len(df),dtype=bool)
        for c,v in zip(p.get('segment_cols',[]), p.get('segment_values',[])):
            mask &= (text(df,c)==str(v)).to_numpy(dtype=bool)
        mask &= (get_lgbm_score(df) < float(p.get('lgbm_lt'))*threshold_multiplier).to_numpy(dtype=bool)
        if bool(p.get('preserve',False)): mask &= ~preserve
        return mask
    if fam=='receiver_value_established':
        mask=((get_lgbm_score(df) < float(p.get('lgbm_lt'))*threshold_multiplier) &
              (num(df,'valor_total_recebido_180d',0.0) > float(p.get('receiver_value_gt'))*threshold_multiplier)).to_numpy(dtype=bool)
        return mask & (~preserve)
    if fam=='quiet_veto':
        mask=((get_lgbm_score(df) < float(p.get('lgbm_lt'))*threshold_multiplier) &
              (get_if_percentile(df) < float(p.get('if_lt'))*threshold_multiplier) &
              (num(df,['se_score_x','se_score_y','se_score'],0.0) <= 20) &
              (num(df,['se_patterns_count','se_pattern_count'],0.0) < 2) &
              (num(df,['beh_score','behavioral_score'],0.0) <= 25) &
              (num(df,['beh_factors_count','behavioral_risk_factor_count'],0.0) < 2)).to_numpy(dtype=bool)
        return mask & (~preserve)
    raise ValueError(f'Rule family not implemented: {fam}')

def load_selected_policy(path: Path) -> dict[str,Any]:
    obj=json.loads(path.read_text(encoding='utf-8'))
    if 'base_policy' not in obj or 'selected_variant' not in obj:
        raise RuntimeError('EXP-013D artifact missing base_policy/selected_variant.')
    return obj

def apply_base_selected_policy(df, artifact):
    base=artifact['base_policy']; var=artifact['selected_variant']; shadow=df['shadow_exp012d_flagged'].to_numpy(dtype=int).astype(bool)
    excluded=set(var.get('excluded_rule_ids',[])); mult=float(var.get('threshold_multiplier',1.0)); veto=np.zeros(len(df),dtype=bool); rows=[]; y=df['is_fraud'].to_numpy(dtype=int)
    for i,rule in enumerate(base['rules']):
        rid=str(rule.get('rule_id',f'rule_{i}'))
        if rid in excluded:
            rows.append({'rule_id':rid,'family':rule.get('family'),'description':rule.get('description'),'excluded':True,'tp_loss':None,'fp_removed':None}); continue
        mask=apply_rule_mask(df,rule,mult)&shadow; veto|=mask
        rows.append({'rule_id':rid,'family':rule.get('family'),'description':rule.get('description'),'excluded':False,'tp_loss':int(((y==1)&mask).sum()),'fp_removed':int(((y==0)&mask).sum())})
    pred=shadow.astype(int); pred[veto]=0
    return pred, pd.DataFrame(rows)

def sanitize_id(s, max_len=120):
    t=re.sub(r'[^A-Za-z0-9_]+','_',str(s)); t=re.sub(r'_+','_',t).strip('_')
    return (t or 'action')[:max_len]

def add_action(actions, action_type, family, desc, mask, base_pred, y, min_fp_effect, max_fp_restore_per_tp):
    mask=np.asarray(mask,dtype=bool)
    if action_type=='restore':
        eff=mask&(base_pred==0); tp=int(((y==1)&eff).sum()); fp=int(((y==0)&eff).sum())
        if tp<=0 or fp/max(tp,1)>max_fp_restore_per_tp: return
        tp_delta=tp; fp_delta=fp
    else:
        eff=mask&(base_pred==1); tp=int(((y==1)&eff).sum()); fp=int(((y==0)&eff).sum())
        if fp<min_fp_effect: return
        tp_delta=-tp; fp_delta=-fp
    if not eff.any(): return
    actions.append(Action(sanitize_id(f'{action_type}_{family}_{len(actions):04d}_{desc}'),action_type,family,desc,eff,tp_delta,fp_delta,{}))

def dedupe_actions(actions):
    best={}
    for a in actions:
        key=np.packbits(a.mask).tobytes()+a.action_type.encode()
        old=best.get(key)
        if old is None:
            best[key]=a
        else:
            nk=(a.tp_delta, -a.fp_delta if a.action_type=='restore' else abs(a.fp_delta), -len(a.description))
            ok=(old.tp_delta, -old.fp_delta if old.action_type=='restore' else abs(old.fp_delta), -len(old.description))
            if nk>ok: best[key]=a
    return list(best.values())

def action_dataframe(actions):
    return pd.DataFrame([{'action_index':i,'action_id':a.action_id,'action_type':a.action_type,'family':a.family,'description':a.description,'tp_delta':a.tp_delta,'fp_delta':a.fp_delta,'n_affected':int(a.mask.sum())} for i,a in enumerate(actions)])

def generate_restore_candidates(df, base_pred, y, max_fp_restore_per_tp):
    actions=[]; shadow=df['shadow_exp012d_flagged'].to_numpy(dtype=int).astype(bool); vetoed=shadow&(base_pred==0); fn=vetoed&(y==1)
    if not fn.any(): return actions
    specs=[('lgbm_r4_score',get_lgbm_score(df),[0.005,0.01,0.02,0.05,0.10,0.20]),('if_percentile',get_if_percentile(df),[0.70,0.80,0.90,0.95,0.98]),('vl_pix',num(df,'vl_pix',0.0),[500,1000,1500,2000,5000,10000]),('se_score',num(df,['se_score_x','se_score_y','se_score'],0.0),[20,40,65]),('behavioral_score',num(df,['beh_score','behavioral_score'],0.0),[15,25,45])]
    for feat,vals,ths in specs:
        for th in ths:
            add_action(actions,'restore','numeric_preserve',f'restore vetoed where {feat}>={th}',vetoed&(vals>=th).to_numpy(dtype=bool),base_pred,y,0,max_fp_restore_per_tp)
    segs=[['value_band'],['ds_tipo_chave_norm'],['periodo_dia'],['first_receiver_flag_real'],['mbk_available_flag'],['value_band','ds_tipo_chave_norm'],['first_receiver_flag_real','value_band'],['first_receiver_flag_real','ds_tipo_chave_norm'],['periodo_dia','value_band'],['mbk_available_flag','ds_tipo_chave_norm']]
    for cols in segs:
        if any(c not in df.columns for c in cols): continue
        kf=pd.DataFrame(index=df.index)
        for c in cols: kf[c]=text(df,c)
        for _,row in kf.loc[fn].drop_duplicates().iterrows():
            mask=vetoed.copy(); parts=[]
            for c in cols:
                v=str(row[c]); parts.append(f'{c}={v}'); mask &= (kf[c]==v).to_numpy(dtype=bool)
            add_action(actions,'restore','segment_preserve','restore vetoed segment '+' AND '.join(parts),mask,base_pred,y,0,max_fp_restore_per_tp)
    return dedupe_actions(actions)

def generate_veto_candidates(df, base_pred, y, min_fp_removed, max_tp_loss_per_veto):
    actions=[]; preserve=strong_preserve_mask(df); lgbm=get_lgbm_score(df); ifp=get_if_percentile(df)
    vl=num(df,'vl_pix',0.0); ratio=num(df,'ratio_valor_media_pagador_90d',0.0); qtd=num(df,'qtd_pix_recebidos_180d',0.0); valrec=num(df,'valor_total_recebido_180d',0.0)
    se=num(df,['se_score_x','se_score_y','se_score'],0.0); beh=num(df,['beh_score','behavioral_score'],0.0)
    def maybe(desc,fam,mask):
        mask=np.asarray(mask,dtype=bool); tp=int(((y==1)&mask&(base_pred==1)).sum())
        if tp>max_tp_loss_per_veto: return
        add_action(actions,'veto',fam,desc,mask,base_pred,y,min_fp_removed,9999)
    for th in [0.00076308066,0.001,0.0019429789,0.003,0.005,0.01]:
        maybe(f'veto positives where lgbm_r4_score<{th} and not strong_preserve','numeric_veto',(lgbm<th).to_numpy(dtype=bool)&(~preserve))
    for th in [0.320032,0.50,0.70]:
        maybe(f'veto positives where if<{th} and lgbm<0.02','if_lgbm_veto',((ifp<th)&(lgbm<0.02)).to_numpy(dtype=bool)&(~preserve))
    for th in [10,20,50,100]:
        maybe(f'veto positives where vl_pix<{th} and lgbm<0.02','low_value_veto',((vl<th)&(lgbm<0.02)).to_numpy(dtype=bool)&(~preserve))
    for th in [0.068208507,0.10726481,0.19765786]:
        maybe(f'veto positives where ratio<{th} and lgbm<0.02','low_ratio_veto',((ratio<th)&(lgbm<0.02)).to_numpy(dtype=bool)&(~preserve))
    for th in [5,10,20,50]:
        maybe(f'veto positives where receiver_qtd_180d>{th} and lgbm<0.02','receiver_history_veto',((qtd>th)&(lgbm<0.02)).to_numpy(dtype=bool)&(~preserve))
    for th in [500,1000,2000,5000,10000]:
        maybe(f'veto positives where receiver_value_180d>{th} and lgbm<0.01','receiver_value_light',((valrec>th)&(lgbm<0.01)).to_numpy(dtype=bool)&(~preserve))
    maybe('veto quiet positives lgbm<0.02 if<0.7 se<=20 beh<=25','quiet_veto_light',((lgbm<0.02)&(ifp<0.7)&(se<=20)&(beh<=25)).to_numpy(dtype=bool)&(~preserve))
    segs=[['value_band'],['ds_tipo_chave_norm'],['periodo_dia'],['first_receiver_flag_real'],['value_band','ds_tipo_chave_norm'],['first_receiver_flag_real','value_band'],['first_receiver_flag_real','ds_tipo_chave_norm'],['periodo_dia','value_band'],['mbk_available_flag','ds_tipo_chave_norm']]
    current=base_pred==1
    for cols in segs:
        if any(c not in df.columns for c in cols): continue
        kf=pd.DataFrame(index=df.index)
        for c in cols: kf[c]=text(df,c)
        grouped=kf[current].groupby(cols,dropna=False).indices
        for key,idxs_rel in grouped.items():
            idxs=kf[current].iloc[list(idxs_rel)].index.to_numpy(dtype=int)
            if len(idxs)<min_fp_removed: continue
            mask=np.zeros(len(df),dtype=bool); mask[idxs]=True
            fp=int(((y==0)&mask).sum()); tp=int(((y==1)&mask).sum())
            if fp<min_fp_removed or tp>max_tp_loss_per_veto: continue
            kt=key if isinstance(key,tuple) else (key,)
            maybe('veto current positive segment '+' AND '.join([f'{c}={v}' for c,v in zip(cols,kt)]),'segment_veto',mask&(~preserve))
    return dedupe_actions(actions)

def state_score(m, base_tp, base_fp, target_recall, min_tp_required):
    return (int(m['recall']>=target_recall and m['tp']>=min_tp_required), int(m['tp']>=base_tp), int(m['fp']<=base_fp), -m['fp'], m['tp'], m['precision'], m['f1'])

def beam_refine(df, base_pred, restore_actions, veto_actions, target_recall, min_tp_required, base_tp, base_fp, restore_beam, veto_beam, max_restore_rules, max_veto_rules):
    y=df['is_fraud'].to_numpy(dtype=int); all_actions=restore_actions+veto_actions; veto_offset=len(restore_actions)
    restore_states=[State(base_pred.copy(),tuple(),tuple(),compute_metrics(y,base_pred))]; best_restore=list(restore_states)
    for depth in range(1,max_restore_rules+1):
        nxt={}
        for st in restore_states:
            last=st.restore_ids[-1] if st.restore_ids else -1
            for i in range(last+1,len(restore_actions)):
                pred=st.pred.copy(); pred[restore_actions[i].mask]=1; m=compute_metrics(y,pred)
                if m['fp']>base_fp+200: continue
                key=np.packbits(pred.astype(bool)).tobytes(); ns=State(pred,st.restore_ids+(i,),tuple(),m)
                if key not in nxt or state_score(m,base_tp,base_fp,target_recall,min_tp_required)>state_score(nxt[key].metrics,base_tp,base_fp,target_recall,min_tp_required): nxt[key]=ns
        if not nxt: break
        restore_states=sorted(nxt.values(),key=lambda s:(s.metrics['tp'],-s.metrics['fp'],s.metrics['recall']),reverse=True)[:restore_beam]
        best_restore.extend(restore_states); log(f'  restore depth={depth}: states={len(restore_states)}, best_tp={restore_states[0].metrics["tp"]}, best_fp={restore_states[0].metrics["fp"]}')
    best_restore=sorted(best_restore,key=lambda s:(s.metrics['tp'],-s.metrics['fp'],len(s.restore_ids)),reverse=True)[:restore_beam]
    frontier=[]; global_best=best_restore[0]
    for rs_idx,rs in enumerate(best_restore):
        states=[rs]
        for depth in range(1,max_veto_rules+1):
            nxt={}
            for st in states:
                last=(st.veto_ids[-1]-veto_offset) if st.veto_ids else -1
                for rel in range(last+1,len(veto_actions)):
                    idx=veto_offset+rel; pred=st.pred.copy(); pred[veto_actions[rel].mask]=0; m=compute_metrics(y,pred)
                    if m['tp']<base_tp or m['recall']<target_recall or m['fp']>base_fp: continue
                    key=np.packbits(pred.astype(bool)).tobytes(); ns=State(pred,st.restore_ids,st.veto_ids+(idx,),m)
                    if key not in nxt or state_score(m,base_tp,base_fp,target_recall,min_tp_required)>state_score(nxt[key].metrics,base_tp,base_fp,target_recall,min_tp_required): nxt[key]=ns
            if not nxt: break
            states=sorted(nxt.values(),key=lambda s:state_score(s.metrics,base_tp,base_fp,target_recall,min_tp_required),reverse=True)[:veto_beam]
            for s in states[:50]:
                frontier.append({'restore_state_idx':rs_idx,'depth':depth,**s.metrics,'n_restore_rules':len(s.restore_ids),'n_veto_rules':len(s.veto_ids),'restore_action_ids':'|'.join(all_actions[i].action_id for i in s.restore_ids),'veto_action_ids':'|'.join(all_actions[i].action_id for i in s.veto_ids)})
            if state_score(states[0].metrics,base_tp,base_fp,target_recall,min_tp_required)>state_score(global_best.metrics,base_tp,base_fp,target_recall,min_tp_required): global_best=states[0]
    if not frontier:
        frontier=[{'restore_state_idx':0,'depth':0,**global_best.metrics,'n_restore_rules':len(global_best.restore_ids),'n_veto_rules':len(global_best.veto_ids),'restore_action_ids':'|'.join(all_actions[i].action_id for i in global_best.restore_ids),'veto_action_ids':'|'.join(all_actions[i].action_id for i in global_best.veto_ids)}]
    return pd.DataFrame(frontier).sort_values(['fp','tp'],ascending=[True,False]).reset_index(drop=True), global_best

def make_blocks(df,pred,n_blocks,policy_name):
    if 'data_pix' in df.columns and df['data_pix'].notna().any(): dates=pd.to_datetime(df['data_pix'],errors='coerce')
    elif 'event_datetime' in df.columns and df['event_datetime'].notna().any(): dates=pd.to_datetime(df['event_datetime'],errors='coerce')
    else: dates=pd.Series(np.arange(len(df)),index=df.index)
    tmp=pd.DataFrame({'date':dates,'idx':np.arange(len(df))}).sort_values(['date','idx']); tmp['block']=pd.qcut(np.arange(len(tmp)),q=min(n_blocks,len(tmp)),labels=False,duplicates='drop')
    blocks=pd.Series(index=tmp['idx'].values,data=tmp['block'].values).sort_index().astype(int)
    rows=[]
    for b in sorted(blocks.unique()):
        part=df.loc[blocks==b]; pb=pred[blocks==b]; m=compute_metrics(part['is_fraud'].to_numpy(dtype=int),pb)
        m.update({'policy_name':policy_name,'block':int(b),'n_rows':int(len(part)),'n_frauds':int(part['is_fraud'].sum()),'dt_min':str(part['data_pix'].min().date()) if 'data_pix' in part.columns and part['data_pix'].notna().any() else None,'dt_max':str(part['data_pix'].max().date()) if 'data_pix' in part.columns and part['data_pix'].notna().any() else None})
        rows.append(m)
    return pd.DataFrame(rows)

def bootstrap(predictions,iters,seed,target):
    rng=np.random.default_rng(seed); n=len(predictions); rows=[]
    for _ in range(iters):
        idx=rng.integers(0,n,size=n); rows.append(compute_metrics(predictions.iloc[idx]['is_fraud'].to_numpy(dtype=int),predictions.iloc[idx]['exp013e_refined_pred'].to_numpy(dtype=int)))
    boot=pd.DataFrame(rows); out=[]
    for metric in ['tp','fp','fn','precision','recall','f1','fpr']:
        vals=boot[metric].astype(float); out.append({'metric':metric,'mean':float(vals.mean()),'p025':float(vals.quantile(.025)),'p050':float(vals.quantile(.5)),'p975':float(vals.quantile(.975)),'target_recall':target if metric=='recall' else None,'p_below_target_recall':float((boot['recall']<target).mean()) if metric=='recall' else None})
    return pd.DataFrame(out)

def make_report(summary,frontier,selected_rules,blocks,boot):
    lines=['# EXP-013E - Conservative Policy Refinement','', '## Resultado', f"- Status: `{summary['objective_status']}`", f"- Base conservadora: TP={summary['base_conservative_metrics']['tp']}, FP={summary['base_conservative_metrics']['fp']}, FN={summary['base_conservative_metrics']['fn']}, recall={summary['base_conservative_metrics']['recall']}", f"- Refinada: TP={summary['selected_metrics']['tp']}, FP={summary['selected_metrics']['fp']}, FN={summary['selected_metrics']['fn']}, recall={summary['selected_metrics']['recall']}, precision={summary['selected_metrics']['precision']}", f"- FP delta vs conservadora: {summary['fp_delta_vs_conservative']}", f"- TP delta vs conservadora: {summary['tp_delta_vs_conservative']}", '', '## Acoes selecionadas']
    if selected_rules.empty: lines.append('Nenhuma acao selecionada; politica conservadora permanece vencedora.')
    else: lines.append(selected_rules[['action_type','family','description','tp_delta','fp_delta']].to_markdown(index=False))
    lines += ['', '## Fronteira']
    lines.append(frontier[['tp','fp','fn','precision','recall','f1','n_restore_rules','n_veto_rules']].head(20).to_markdown(index=False))
    lines += ['', '## Blocos temporais', blocks.to_markdown(index=False), '', '## Bootstrap recall', boot[boot['metric']=='recall'].to_markdown(index=False), '', '## Interpretacao']
    if summary['selected_metrics']['tp']>summary['base_conservative_metrics']['tp']: lines.append('O refinamento recuperou TP em relacao a politica conservadora. Avaliar se o FP devolvido e aceitavel.')
    elif summary['selected_metrics']['fp']<summary['base_conservative_metrics']['fp']: lines.append('O refinamento reduziu FP sem piorar TP em relacao a politica conservadora.')
    else: lines.append('A politica conservadora permanece como melhor equilibrio local; nao houve refinamento superior sob as restricoes atuais.')
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input',default=str(DEFAULT_INPUT)); ap.add_argument('--selected-policy',default=str(DEFAULT_POLICY)); ap.add_argument('--output-dir',default=str(DEFAULT_OUTPUT))
    ap.add_argument('--target-recall',type=float,default=0.95); ap.add_argument('--bootstrap-iters',type=int,default=500); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--time-blocks',type=int,default=5)
    ap.add_argument('--max-fp-restore-per-tp',type=float,default=60.0); ap.add_argument('--min-fp-removed',type=int,default=20); ap.add_argument('--max-tp-loss-per-veto',type=int,default=1)
    ap.add_argument('--restore-beam',type=int,default=50); ap.add_argument('--veto-beam',type=int,default=80); ap.add_argument('--max-restore-rules',type=int,default=3); ap.add_argument('--max-veto-rules',type=int,default=5)
    args=ap.parse_args(); t0=time.perf_counter(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    log('='*80); log('EXP-013E - Conservative Policy Refinement'); log('='*80); log(f'Input: {args.input}'); log(f'Selected policy: {args.selected_policy}')
    df=normalize_columns(pd.read_csv(args.input,low_memory=False)); art=load_selected_policy(Path(args.selected_policy)); y=df['is_fraud'].to_numpy(dtype=int)
    shadow=df['shadow_exp012d_flagged'].to_numpy(dtype=int); cons, rule_imp=apply_base_selected_policy(df,art)
    sh_m=compute_metrics(y,shadow); cons_m=compute_metrics(y,cons); min_tp=int(math.ceil(args.target_recall*int(y.sum())))
    log(f"Shadow: TP={sh_m['tp']} FP={sh_m['fp']} FN={sh_m['fn']} recall={sh_m['recall']}"); log(f"Conservative: TP={cons_m['tp']} FP={cons_m['fp']} FN={cons_m['fn']} recall={cons_m['recall']}")
    pd.DataFrame([{'policy_name':'BASELINE_SHADOW_EXP012D',**sh_m},{'policy_name':'CONSERVATIVE_NO_RECEIVER_VALUE',**cons_m}]).to_csv(out/'01_base_metrics.csv',index=False)
    rule_imp.to_csv(out/'base_selected_policy_rule_impacts.csv',index=False)
    log('[1/4] Restore candidates...'); restore=generate_restore_candidates(df,cons,y,args.max_fp_restore_per_tp); rdf=action_dataframe(restore).sort_values(['tp_delta','fp_delta'],ascending=[False,True]) if restore else pd.DataFrame(); rdf.to_csv(out/'02_restore_candidates.csv',index=False); log(f'  restore={len(restore)}')
    log('[2/4] Veto candidates...'); veto=generate_veto_candidates(df,cons,y,args.min_fp_removed,args.max_tp_loss_per_veto); vdf=action_dataframe(veto).sort_values(['fp_delta','tp_delta'],ascending=[True,False]) if veto else pd.DataFrame(); vdf.to_csv(out/'03_veto_candidates.csv',index=False); log(f'  veto={len(veto)}')
    log('[3/4] Local beam search...'); frontier,best=beam_refine(df,cons,restore,veto,args.target_recall,min_tp,cons_m['tp'],cons_m['fp'],args.restore_beam,args.veto_beam,args.max_restore_rules,args.max_veto_rules); frontier.to_csv(out/'04_refinement_frontier.csv',index=False)
    actions=restore+veto; sel_idx=list(best.restore_ids)+list(best.veto_ids); sel_rules=action_dataframe([actions[i] for i in sel_idx]) if sel_idx else pd.DataFrame(); sel_rules.to_csv(out/'05_selected_refinement_rules.csv',index=False)
    pred=best.pred; ref_m=compute_metrics(y,pred); predictions=df.copy(); predictions['exp013e_refined_pred']=pred; predictions['exp013e_conservative_pred']=cons; predictions['exp013e_changed_vs_conservative']=(pred!=cons).astype(int); predictions.to_csv(out/'06_selected_predictions.csv',index=False)
    predictions[(predictions['is_fraud']==1)&(predictions['exp013e_refined_pred']==0)].to_csv(out/'07_selected_false_negatives.csv',index=False); predictions[(predictions['is_fraud']==0)&(predictions['exp013e_refined_pred']==1)].to_csv(out/'08_selected_false_positives.csv',index=False)
    log('[4/4] Blocks and bootstrap...'); blocks=pd.concat([make_blocks(df,cons,args.time_blocks,'CONSERVATIVE_NO_RECEIVER_VALUE'),make_blocks(df,pred,args.time_blocks,'EXP013E_REFINED')],ignore_index=True); blocks.to_csv(out/'09_time_block_metrics.csv',index=False); boot=bootstrap(predictions,args.bootstrap_iters,args.seed,args.target_recall); boot.to_csv(out/'10_bootstrap_confidence_intervals.csv',index=False)
    obj='TARGET_RECALL_MET' if ref_m['recall']>=args.target_recall else 'TARGET_RECALL_NOT_MET'; obj += '_TP_FLOOR_MET' if ref_m['tp']>=cons_m['tp'] else '_TP_FLOOR_NOT_MET'; obj += '_FP_NOT_WORSE' if ref_m['fp']<=cons_m['fp'] else '_FP_WORSE';
    if ref_m['tp']>cons_m['tp']: obj+='_TP_RECOVERED'
    if ref_m['fp']<cons_m['fp']: obj+='_FP_REDUCED'
    artifact={'experiment':'EXP-013E','policy_name':'conservative_refined_local_policy','objective_status':obj,'base_selected_policy_artifact':str(args.selected_policy),'base_conservative_metrics':cons_m,'selected_metrics':ref_m,'selected_refinement_actions':sel_rules.to_dict(orient='records') if not sel_rules.empty else [],'notes':['Starts from EXP-013D selected conservative policy.','If no action is selected, keep EXP-013D conservative policy.','Validate externally before production.']}; dump_json(artifact,out/'12_refined_policy_artifact.json')
    summary={'experiment':'EXP-013E','status':'DONE','objective_status':obj,'input_path':str(args.input),'selected_policy_path':str(args.selected_policy),'n_rows':int(len(df)),'total_frauds':int(y.sum()),'target_recall':args.target_recall,'min_tp_required':min_tp,'shadow_metrics':sh_m,'base_conservative_metrics':cons_m,'selected_metrics':ref_m,'tp_delta_vs_conservative':int(ref_m['tp']-cons_m['tp']),'fp_delta_vs_conservative':int(ref_m['fp']-cons_m['fp']),'fn_delta_vs_conservative':int(ref_m['fn']-cons_m['fn']),'n_restore_candidates':int(len(restore)),'n_veto_candidates':int(len(veto)),'n_selected_restore_actions':int(len(best.restore_ids)),'n_selected_veto_actions':int(len(best.veto_ids)),'min_block_recall_refined':float(blocks[blocks['policy_name']=='EXP013E_REFINED']['recall'].min()),'bootstrap_recall':boot[boot['metric']=='recall'].iloc[0].to_dict() if not boot.empty else {},'elapsed_seconds':round(time.perf_counter()-t0,2),'output_dir':str(out)}; dump_json(summary,out/'00_run_summary.json')
    (out/'11_refinement_report.md').write_text(make_report(summary,frontier,sel_rules,blocks,boot),encoding='utf-8')
    log('\n'+'='*80); log('EXP-013E CONCLUIDO'); log('='*80); log(json.dumps(summary,ensure_ascii=False,indent=2)); log('\nArquivos principais:')
    for p in ['00_run_summary.json','04_refinement_frontier.csv','05_selected_refinement_rules.csv','06_selected_predictions.csv','09_time_block_metrics.csv','10_bootstrap_confidence_intervals.csv','11_refinement_report.md','12_refined_policy_artifact.json']:
        log(f'  {out/p}')

if __name__=='__main__': main()
