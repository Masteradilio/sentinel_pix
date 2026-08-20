#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3O - Consolidate R3N-FROZEN + Zero-FN FP-only Reducer

Parte A: reaplica o artifact recomendado do R3N e valida:
  TP=1465, FP=4769, FN=0

Parte B: a partir do R3N congelado, reduz somente FP:
  - sem novos rescues;
  - sem troca recall x FP;
  - somente vetos em alertas atuais;
  - toda nova regra precisa ter TP_loss=0;
  - FN precisa permanecer 0.

Uso recomendado:
  python scripts/exp_014b_r3o_consolidate_r3n_and_fp_only_reducer.py --max-rules 20 --min-fp-removed 10 --max-combo-size 4 --max-seconds 240
"""
from __future__ import annotations

import argparse, itertools, json, sys, time
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

SCRIPT_PATH=Path(__file__).resolve()
PROJECT_ROOT=SCRIPT_PATH.parent.parent if (SCRIPT_PATH.parent.parent/'dados').exists() else Path.cwd()
DEFAULT_INPUT=PROJECT_ROOT/'resultados'/'experimentos'/'EXP-014B-R3N'/'15_predictions_recommended.csv'
DEFAULT_ARTIFACT=PROJECT_ROOT/'resultados'/'experimentos'/'EXP-014B-R3N'/'14_policy_artifact_recommended.json'
DEFAULT_RESCUES=PROJECT_ROOT/'resultados'/'experimentos'/'EXP-014B-R3N'/'09_r3n_rescue_candidates.csv'
DEFAULT_OUT=PROJECT_ROOT/'resultados'/'experimentos'/'EXP-014B-R3O'

R3M_FROZEN_COL='exp014b_r3m_frozen_pred'
R3M_FALLBACK_COL='exp014b_r3m_recommended_pred'
R3N_FROZEN_COL='exp014b_r3n_frozen_pred'
R3O_FINAL_COL='exp014b_r3o_recommended_pred'
EXPECTED={'tp':1465,'fp':4769,'fn':0,'headroom_fp_removed':141,'rescue_fn_recovered':21,'rescue_fp_added':157,'retightening_fp_removed':63,'retightening_tp_loss':0,'wilson_low_min':0.99}
FEATURE_COLS=['ds_tipo_chave_norm','value_band','mbk_available_flag','first_receiver_flag_real','periodo_dia','module_quiet','lgbm_bin','if_bin','score_bin','ratio_bin','qtd_rec_bin','vl_bin']
NUMERIC_COLS=['lgbm_r4_score','lgbm_mapped','lgbm_raw','score_final','if_percentile','if_percentile_x','if_percentile_y','vl_pix','ratio_valor_media_pagador_90d','qtd_pix_recebidos_180d','valor_total_recebido_180d']

@dataclass
class Candidate:
    rule_id:str
    family:str
    description:str
    mask:np.ndarray
    tp_loss:int
    fp_removed:int
    n_removed:int
    params:dict[str,Any]

def log(s): print(s, flush=True)
def dump_json(obj,path): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(np.asarray(y).astype(int),np.asarray(p).astype(int),labels=[0,1]).ravel()
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),'precision':round(float(precision_score(y,p,zero_division=0)),8),'recall':round(float(recall_score(y,p,zero_division=0)),8),'f1':round(float(f1_score(y,p,zero_division=0)),8),'fpr':round(float(fp/max(fp+tn,1)),8)}

def wilson(successes,n,z=1.959963984540054):
    if n<=0: return float('nan'),float('nan')
    phat=successes/n; denom=1+z*z/n
    center=(phat+z*z/(2*n))/denom
    margin=z*((phat*(1-phat)/n)+(z*z/(4*n*n)))**0.5/denom
    return max(0.0,center-margin),min(1.0,center+margin)

def normalize(df):
    df=df.copy(); df.columns=[str(c).strip().split('.')[-1] for c in df.columns]
    if 'transaction_id' not in df.columns and 'cd_pix' in df.columns: df['transaction_id']=df['cd_pix']
    if 'event_datetime' not in df.columns and 'dt_pix' in df.columns: df['event_datetime']=df['dt_pix']
    if 'is_fraud' not in df.columns: raise RuntimeError('Coluna obrigatoria ausente: is_fraud')
    df['is_fraud']=pd.to_numeric(df['is_fraud'],errors='coerce').fillna(0).astype(int)
    for c in [R3M_FROZEN_COL,R3M_FALLBACK_COL,R3N_FROZEN_COL,R3O_FINAL_COL,'exp014b_r3n_recommended_pred']:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0).astype(int)
    return df.reset_index(drop=True)

def pick(df,names):
    if isinstance(names,str): names=[names]
    for n in names:
        if n in df.columns: return n
    return None

def num(df,names,default=0.0):
    c=pick(df,names)
    if c is None: return pd.Series([default]*len(df),index=df.index,dtype=float)
    return pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(default).astype(float)

def qbin(s,name,bins):
    vals=pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)
    edges=[-np.inf]+bins+[np.inf]; labels=[]
    for a,b in zip(edges[:-1],edges[1:]):
        if np.isneginf(a): labels.append(f'{name}_LT_{b:g}')
        elif np.isposinf(b): labels.append(f'{name}_GE_{a:g}')
        else: labels.append(f'{name}_{a:g}_{b:g}')
    return pd.cut(vals,bins=edges,labels=labels,include_lowest=True).astype('string').fillna(f'{name}_MISSING').astype(str)

def add_bins(df):
    df=df.copy()
    if 'lgbm_bin' not in df.columns and pick(df,['lgbm_r4_score','lgbm_mapped','lgbm_raw']): df['lgbm_bin']=qbin(num(df,['lgbm_r4_score','lgbm_mapped','lgbm_raw']),'lgbm',[0.001,0.003,0.005,0.01,0.02,0.05,0.1])
    if 'if_bin' not in df.columns and pick(df,['if_percentile','if_percentile_x','if_percentile_y']): df['if_bin']=qbin(num(df,['if_percentile','if_percentile_x','if_percentile_y']),'if',[0.32,0.5,0.7,0.85,0.95])
    if 'score_bin' not in df.columns and 'score_final' in df.columns: df['score_bin']=qbin(num(df,'score_final'),'score',[0.5,1,2,3,5,10])
    if 'ratio_bin' not in df.columns and 'ratio_valor_media_pagador_90d' in df.columns: df['ratio_bin']=qbin(num(df,'ratio_valor_media_pagador_90d'),'ratio',[0.05,0.1,0.2,0.5,1,2,5])
    if 'qtd_rec_bin' not in df.columns and 'qtd_pix_recebidos_180d' in df.columns: df['qtd_rec_bin']=qbin(num(df,'qtd_pix_recebidos_180d'),'qtdrec',[0,1,2,5,10,20,50,100])
    if 'vl_bin' not in df.columns and 'vl_pix' in df.columns: df['vl_bin']=qbin(num(df,'vl_pix'),'vl',[20,50,100,250,500,1000,5000,10000])
    if 'module_quiet' not in df.columns:
        se=num(df,['se_score_x','se_score_y','se_score']); sec=num(df,['se_patterns_count','se_pattern_count']); beh=num(df,['beh_score','behavioral_score']); behc=num(df,['beh_factors_count','behavioral_risk_factor_count']); runtime=num(df,'runtime_flagged')
        strong=(se>=40)|(sec>=2)|(beh>=25)|(behc>=2)|(runtime>=1)
        df['module_quiet']=np.where(strong,'module_strong','module_quiet')
    return df

def parse_params(raw):
    if isinstance(raw,dict): return raw
    return json.loads(str(raw).replace('Infinity','1e999'))

def rule_mask(df,current_pred,params,mode,scope_mask=None):
    typ=params.get('type')
    if typ in ['numeric_headroom','numeric_threshold_rescue','numeric_retighten']:
        c=params.get('col')
        if c not in df.columns: return np.zeros(len(df),dtype=bool)
        vals=num(df,c).to_numpy(dtype=float); cut=float(params.get('cut')); direction=params.get('direction')
        mask=(vals>=cut) if direction=='ge' else (vals<=cut)
    elif typ in ['combo_headroom','combo_rescue','combo_retighten']:
        mask=np.ones(len(df),dtype=bool)
        for c,v in zip(params.get('combo_cols',[]),params.get('combo_values',[])):
            if c not in df.columns: return np.zeros(len(df),dtype=bool)
            mask &= (df[c].astype('string').fillna('<MISSING>').astype(str).to_numpy()==str(v))
    else:
        return np.zeros(len(df),dtype=bool)
    if mode in ['headroom','retighten']: mask &= (current_pred.astype(int)==1)
    elif mode=='rescue': mask &= (current_pred.astype(int)==0)
    else: raise RuntimeError('mode invalido')
    if scope_mask is not None: mask &= scope_mask
    return mask

def apply_rules(df,pred,rules,mode,y,scope_mask=None):
    cur=pred.copy().astype(int); rows=[]
    for i,rule in enumerate(rules):
        params=parse_params(rule.get('params_json') or rule.get('params') or '{}')
        mask=rule_mask(df,cur,params,mode,scope_mask)
        tp_loss=int(((y==1)&mask).sum()) if mode in ['headroom','retighten'] else 0
        fp_removed=int(((y==0)&mask).sum()) if mode in ['headroom','retighten'] else 0
        fn_recovered=int(((y==1)&mask).sum()) if mode=='rescue' else 0
        fp_added=int(((y==0)&mask).sum()) if mode=='rescue' else 0
        if mode in ['headroom','retighten']: cur[mask]=0
        else: cur[mask]=1
        rows.append({'phase':mode,'rule_index':i,'rule_id':rule.get('rule_id') or rule.get('candidate_id'),'family':rule.get('family'),'description':rule.get('description'),'tp_loss':tp_loss,'fp_removed':fp_removed,'fn_recovered':fn_recovered,'fp_added':fp_added,'n_effect':int(mask.sum()),'params_json':json.dumps(params,ensure_ascii=False)})
    return cur,pd.DataFrame(rows)

def apply_rescue_ids(df,pred,rescue_df,ids,y):
    sel=rescue_df[rescue_df['candidate_id'].astype(str).isin(set(ids))].copy()
    order={cid:i for i,cid in enumerate(ids)}
    sel['_order']=sel['candidate_id'].astype(str).map(order); sel=sel.sort_values('_order')
    return apply_rules(df,pred,sel.to_dict(orient='records'),'rescue',y)

def validate_r3n(df,artifact,rescues,outdir,base_col):
    y=df['is_fraud'].to_numpy(dtype=int)
    scenario_name=artifact.get('recommended_scenario') or 'r3n_cap_5000'
    scenario=artifact['scenario_artifacts'][scenario_name]
    base=df[base_col].to_numpy(dtype=int); base_m=metrics(y,base)
    ph,impact_h=apply_rules(df,base,scenario.get('selected_headroom_rules',[]),'headroom',y); head_m=metrics(y,ph)
    ids=[str(x) for x in scenario.get('selected_rescue_candidate_ids',[])]
    pr,impact_r=apply_rescue_ids(df,ph,rescues,ids,y); rescue_m=metrics(y,pr)
    added=(pr.astype(int)==1)&(ph.astype(int)==0)
    pf,impact_t=apply_rules(df,pr,scenario.get('selected_retighten_rules',[]),'retighten',y,scope_mask=added); final_m=metrics(y,pf)
    pd.concat([impact_h,impact_r,impact_t],ignore_index=True).to_csv(outdir/'04_r3n_frozen_rule_impact.csv',index=False)
    pd.DataFrame([{'policy_name':'R3M_FROZEN_BASE',**base_m},{'policy_name':'R3N_AFTER_HEADROOM',**head_m},{'policy_name':'R3N_AFTER_RESCUE_BEFORE_RETIGHTEN',**rescue_m},{'policy_name':'EXP014B_R3N_FROZEN_FINAL',**final_m}]).to_csv(outdir/'03_r3n_frozen_metrics.csv',index=False)
    headroom_fp_removed=base_m['fp']-head_m['fp']; rescue_fn=head_m['fn']-rescue_m['fn']; rescue_fp=rescue_m['fp']-head_m['fp']; ret_fp=rescue_m['fp']-final_m['fp']; ret_tp=rescue_m['tp']-final_m['tp']; wl,wh=wilson(final_m['tp'],int(y.sum()))
    exp_m=(final_m['tp']==EXPECTED['tp'] and final_m['fp']==EXPECTED['fp'] and final_m['fn']==EXPECTED['fn'])
    exp_p=(headroom_fp_removed==EXPECTED['headroom_fp_removed'] and rescue_fn==EXPECTED['rescue_fn_recovered'] and rescue_fp==EXPECTED['rescue_fp_added'] and ret_fp==EXPECTED['retightening_fp_removed'] and ret_tp==EXPECTED['retightening_tp_loss'])
    val={'scenario_name':scenario_name,'base_col':base_col,'base_metrics':base_m,'headroom_metrics':head_m,'rescue_metrics_before_retightening':rescue_m,'final_metrics':final_m,'headroom_fp_removed':int(headroom_fp_removed),'rescue_fn_recovered':int(rescue_fn),'rescue_fp_added':int(rescue_fp),'retightening_fp_removed':int(ret_fp),'retightening_tp_loss':int(ret_tp),'wilson_low':wl,'wilson_high':wh,'expected_metrics_match':bool(exp_m),'expected_phases_match':bool(exp_p),'wilson_pass':bool(wl>=EXPECTED['wilson_low_min']),'all_pass':bool(exp_m and exp_p and wl>=EXPECTED['wilson_low_min'])}
    val['status']='PASS_R3N_FROZEN_VALIDATED' if val['all_pass'] else 'FAIL_R3N_FROZEN_DIVERGENCE'
    dump_json(val,outdir/'02_r3n_frozen_validation.json')
    return pf,val

def feat_frame(df):
    out=pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns: out[c]=df[c].astype('string').fillna('<MISSING>').astype(str)
    return out

def add_cand(out,family,desc,mask,y,min_fp,params):
    if not mask.any(): return
    tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
    if tp!=0 or fp<min_fp: return
    out.append(Candidate(f'fp_{len(out):05d}',family,desc,mask,tp,fp,int(mask.sum()),params))

def mine_fp_candidates(df,pred,min_fp,max_combo,top_groups):
    y=df['is_fraud'].to_numpy(dtype=int); alerted=pred.astype(int)==1; out=[]
    for c in NUMERIC_COLS:
        if c not in df.columns: continue
        vals=num(df,c).to_numpy(dtype=float); active=vals[alerted]
        if len(active)==0: continue
        try: cuts=sorted(set(float(x) for x in np.quantile(active,[.01,.03,.05,.1,.2,.3,.5,.7,.9,.95,.97,.99]) if np.isfinite(x)))
        except Exception: cuts=[]
        for cut in cuts:
            for d in ['le','ge']:
                mask=alerted & ((vals<=cut) if d=='le' else (vals>=cut))
                desc=f'alert AND {c}<={cut:g}' if d=='le' else f'alert AND {c}>={cut:g}'
                add_cand(out,'fp_only_numeric_tp0',desc,mask,y,min_fp,{'type':'numeric_headroom','col':c,'direction':d,'cut':cut})
    feat=feat_frame(df); cols=list(feat.columns); bins=[c for c in cols if c.endswith('_bin') or c=='module_quiet']; important=['ds_tipo_chave_norm','value_band','mbk_available_flag','first_receiver_flag_real','periodo_dia']; idx=np.where(alerted)[0]
    for r in range(1,max_combo+1):
        for combo in itertools.combinations(cols,r):
            combo=list(combo)
            if r==1 and combo[0] not in bins+['ds_tipo_chave_norm','value_band']: continue
            if r>=2 and not any(c in combo for c in important+bins): continue
            sub=feat.iloc[idx][combo]
            if sub.empty: continue
            groups=[]
            for key,rel in sub.groupby(combo,dropna=False).indices.items():
                rows=sub.iloc[list(rel)].index.to_numpy(dtype=int)
                if len(rows)<min_fp: continue
                mask=np.zeros(len(df),dtype=bool); mask[rows]=True; mask &= alerted
                tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
                if tp==0 and fp>=min_fp: groups.append((-fp,key,mask))
            groups.sort()
            for _,key,mask in groups[:top_groups]:
                vals=key if isinstance(key,tuple) else (key,); vals=[str(v) for v in vals]
                desc='alert AND '+' AND '.join([f'{c}={v}' for c,v in zip(combo,vals)])
                add_cand(out,'fp_only_combo_tp0',desc,mask,y,min_fp,{'type':'combo_headroom','combo_cols':combo,'combo_values':vals})
    best={}
    for c in out:
        k=np.packbits(c.mask).tobytes(); old=best.get(k)
        if old is None or (c.fp_removed,-len(c.description))>(old.fp_removed,-len(old.description)): best[k]=c
    out=list(best.values()); out.sort(key=lambda c:(-c.fp_removed,len(c.description)))
    for i,c in enumerate(out): c.rule_id=f'r3o_fp_{i:05d}'
    return out

def cand_dicts(cands):
    return [{'candidate_index':i,'rule_id':c.rule_id,'family':c.family,'description':c.description,'tp_loss':c.tp_loss,'fp_removed':c.fp_removed,'n_removed':c.n_removed,'params_json':json.dumps(c.params,ensure_ascii=False)} for i,c in enumerate(cands)]

def select_fp(cands,pred,y,max_rules,max_seconds):
    t0=time.perf_counter(); cur=pred.copy().astype(int); selected=[]; used=set(); rows=[]; stop='completed'
    for depth in range(1,max_rules+1):
        if time.perf_counter()-t0>=max_seconds:
            stop=f'max_seconds_before_rule_{depth}'; break
        best=None; alerted=cur.astype(int)==1
        for i,c in enumerate(cands[:2000]):
            if i in used: continue
            mask=c.mask & alerted
            tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
            if tp!=0 or fp<=0: continue
            rank=(fp,-int(mask.sum()),-len(c.description))
            if best is None or rank>best[0]: best=(rank,i,c,mask,fp)
        if best is None:
            stop=f'no_more_tp0_fp_rules_at_depth_{depth}'; break
        _,i,c,mask,fp=best; cur[mask]=0; used.add(i)
        chosen=Candidate(c.rule_id,c.family,c.description,mask.copy(),0,int(fp),int(mask.sum()),c.params)
        selected.append(chosen); m=metrics(y,cur)
        rows.append({'depth':depth,'rule_id':chosen.rule_id,'family':chosen.family,'description':chosen.description,'marginal_fp_removed':int(fp),'cumulative_fp_removed':int(sum(x.fp_removed for x in selected)),'tp_loss':0,'n_selected_rules':len(selected),**m})
    if not rows: rows=[{'depth':0,'rule_id':'','family':'','description':'','marginal_fp_removed':0,'cumulative_fp_removed':0,'tp_loss':0,'n_selected_rules':0,**metrics(y,pred)}]
    return cur,selected,pd.DataFrame(rows),stop

def make_report(summary,val,cand_df,frontier,selected_df):
    lines=['# EXP-014B-R3O — R3N-FROZEN + FP-only reducer','', '## Parte A — Consolidação R3N', f"- Status R3N frozen: `{val['status']}`", f"- Métricas R3N frozen: `{val['final_metrics']}`", '', '## Parte B — Redução de FP com FN=0 preservado', f"- Status: `{summary['objective_status']}`", f"- Métricas finais: `{summary['recommended_metrics']}`", f"- FP removidos vs R3N: `{summary['fp_removed_vs_r3n']}`", f"- TP loss vs R3N: `{summary['tp_loss_vs_r3n']}`", f"- FN final: `{summary['recommended_metrics']['fn']}`", '']
    lines.append('## Top candidatos FP-only')
    if cand_df.empty: lines.append('Nenhum candidato TP0 encontrado.')
    else: lines.append(cand_df[[c for c in ['rule_id','family','description','fp_removed','tp_loss','n_removed'] if c in cand_df.columns]].head(30).to_markdown(index=False))
    lines += ['', '## Fronteira selecionada']
    if frontier.empty: lines.append('Fronteira vazia.')
    else: lines.append(frontier[[c for c in ['depth','marginal_fp_removed','cumulative_fp_removed','tp','fp','fn','precision','recall','fpr','description'] if c in frontier.columns]].to_markdown(index=False))
    lines += ['', '## Regras selecionadas']
    if selected_df.empty: lines.append('Nenhuma regra selecionada.')
    else: lines.append(selected_df[[c for c in ['rule_id','family','description','fp_removed','tp_loss'] if c in selected_df.columns]].to_markdown(index=False))
    lines += ['', '## Decisão sugerida']
    if summary['all_pass'] and summary['fp_removed_vs_r3n']>0 and summary['recommended_metrics']['fn']==0:
        lines.append('O R3O gerou um candidato FP-only superior ao R3N mantendo FN=0. Próximo passo: validação congelada do R3O.')
    else:
        lines.append('Se não houver ganho seguro de FP, consolidar R3N como benchmark principal e avançar para hardening/robustez.')
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input',default=str(DEFAULT_INPUT)); ap.add_argument('--r3n-artifact',default=str(DEFAULT_ARTIFACT)); ap.add_argument('--r3n-rescues',default=str(DEFAULT_RESCUES)); ap.add_argument('--output-dir',default=str(DEFAULT_OUT))
    ap.add_argument('--min-fp-removed',type=int,default=10); ap.add_argument('--max-combo-size',type=int,default=4); ap.add_argument('--top-groups-per-combo',type=int,default=80); ap.add_argument('--max-rules',type=int,default=20); ap.add_argument('--max-seconds',type=int,default=240); ap.add_argument('--no-write-predictions',action='store_true')
    args=ap.parse_args(); t0=time.perf_counter(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    log('='*80); log('EXP-014B-R3O — Consolidate R3N-FROZEN + Zero-FN FP-only Reducer'); log('='*80)
    inp=Path(args.input); art=Path(args.r3n_artifact); res=Path(args.r3n_rescues)
    for p,name in [(inp,'input'),(art,'r3n_artifact'),(res,'r3n_rescues')]:
        if not p.exists(): raise FileNotFoundError(f'{name} nao encontrado: {p}')
    df=add_bins(normalize(pd.read_csv(inp,low_memory=False))); artifact=load_json(art); rescues=pd.read_csv(res)
    base_col=R3M_FROZEN_COL if R3M_FROZEN_COL in df.columns else R3M_FALLBACK_COL
    missing=[]
    if 'is_fraud' not in df.columns: missing.append('is_fraud')
    if base_col not in df.columns: missing.append(R3M_FROZEN_COL)
    if 'params_json' not in rescues.columns: missing.append('r3n_rescues.params_json')
    if artifact.get('recommended_scenario') not in artifact.get('scenario_artifacts',{}): missing.append('recommended_scenario_in_artifact')
    contract={'n_rows':int(len(df)),'n_frauds':int(df['is_fraud'].sum()),'base_col_used_for_r3n_replay':base_col,'r3n_artifact_path':str(art),'r3n_rescues_rows':int(len(rescues)),'missing':missing,'contract_ok':not missing}
    dump_json(contract,out/'01_input_contract.json')
    if missing: raise RuntimeError(f'Contrato falhou: {missing}')
    y=df['is_fraud'].to_numpy(dtype=int)
    log('[A] Validando R3N congelado...')
    r3n_pred,val=validate_r3n(df,artifact,rescues,out,base_col); df[R3N_FROZEN_COL]=r3n_pred.astype(int); r3n_m=metrics(y,r3n_pred)
    log(f'R3N frozen metrics: {r3n_m}')
    log('[B] Minerando regras FP-only TP_loss=0...')
    cands=mine_fp_candidates(df,r3n_pred,args.min_fp_removed,args.max_combo_size,args.top_groups_per_combo)
    cand_df=pd.DataFrame(cand_dicts(cands)); cand_df.to_csv(out/'05_fp_reduction_candidates.csv',index=False)
    log(f'Candidatos FP-only: {len(cands)}')
    final_pred,selected,frontier,stop=select_fp(cands,r3n_pred,y,args.max_rules,args.max_seconds)
    frontier.to_csv(out/'06_fp_reduction_frontier.csv',index=False)
    selected_df=pd.DataFrame(cand_dicts(selected)); selected_df.to_csv(out/'07_selected_fp_rules.csv',index=False)
    final_m=metrics(y,final_pred); fp_removed=r3n_m['fp']-final_m['fp']; tp_loss=r3n_m['tp']-final_m['tp']; fn_delta=final_m['fn']-r3n_m['fn']; wl,wh=wilson(final_m['tp'],int(y.sum()))
    df[R3O_FINAL_COL]=final_pred.astype(int)
    status='DONE_R3N_FROZEN_CONSOLIDATED'; status += '_FP_ONLY_REDUCED' if fp_removed>0 else '_FP_ONLY_NO_GAIN'; status += '_FN_ZERO_PRESERVED' if final_m['fn']==0 and tp_loss==0 else '_FN_ZERO_BROKEN'; status += '_R3N_FROZEN_PASS' if val['all_pass'] else '_R3N_FROZEN_NOT_PASS'
    all_pass=bool(val['all_pass'] and final_m['fn']==0 and tp_loss==0)
    artifact_out={'experiment':'EXP-014B-R3O','policy_name':'r3n_consolidated_zero_fn_fp_only_reducer','objective_status':status,'r3n_frozen_validation':val,'base_r3n_frozen_metrics':r3n_m,'recommended_metrics':final_m,'fp_removed_vs_r3n':int(fp_removed),'tp_loss_vs_r3n':int(tp_loss),'fn_delta_vs_r3n':int(fn_delta),'wilson_low':wl,'wilson_high':wh,'selected_fp_rules':selected_df.to_dict(orient='records') if not selected_df.empty else [],'stop_reason':stop,'constraints':{'min_fp_removed':args.min_fp_removed,'max_combo_size':args.max_combo_size,'top_groups_per_combo':args.top_groups_per_combo,'max_rules':args.max_rules,'max_seconds':args.max_seconds,'strict_fn_zero':True,'strict_tp_loss_zero':True}}
    dump_json(artifact_out,out/'08_policy_artifact_recommended.json')
    if not args.no_write_predictions: df.to_csv(out/'09_predictions_recommended.csv',index=False)
    summary={'experiment':'EXP-014B-R3O','status':'DONE','objective_status':status,'input_path':str(inp),'r3n_artifact_path':str(art),'r3n_rescues_path':str(res),'n_rows':int(len(df)),'n_frauds':int(df['is_fraud'].sum()),'r3n_frozen_validation_status':val['status'],'r3n_frozen_metrics':r3n_m,'recommended_metrics':final_m,'fp_removed_vs_r3n':int(fp_removed),'tp_loss_vs_r3n':int(tp_loss),'fn_delta_vs_r3n':int(fn_delta),'n_fp_candidates':int(len(cands)),'n_selected_fp_rules':int(len(selected)),'stop_reason':stop,'recommended_wilson_low':wl,'recommended_wilson_high':wh,'r3n_frozen_all_pass':bool(val['all_pass']),'all_pass':all_pass,'elapsed_seconds':round(time.perf_counter()-t0,2),'output_dir':str(out)}
    dump_json(summary,out/'00_run_summary.json')
    (out/'10_exp014b_r3o_report.md').write_text(make_report(summary,val,cand_df,frontier,selected_df),encoding='utf-8')
    log(json.dumps(summary,ensure_ascii=False,indent=2)); log(''); log('Arquivos principais:')
    for p in [out/'00_run_summary.json',out/'01_input_contract.json',out/'02_r3n_frozen_validation.json',out/'03_r3n_frozen_metrics.csv',out/'04_r3n_frozen_rule_impact.csv',out/'05_fp_reduction_candidates.csv',out/'06_fp_reduction_frontier.csv',out/'07_selected_fp_rules.csv',out/'08_policy_artifact_recommended.json',out/'10_exp014b_r3o_report.md']:
        log(f'  {p}')

if __name__=='__main__': main()
