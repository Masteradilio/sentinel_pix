
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXP-014B-R3L - Headroom + Residual FN Optimizer

Continua FN First / FP Second após o R3K.
Base esperada: EXP014B_R3K_RESCUE100 com TP=1425 FP=4966 FN=40.

Estratégia:
1. Carrega EXP-014B-R3K/10_predictions_recommended.csv.
2. Usa exp014b_r3k_recommended_pred como base.
3. Cria headroom com vetos TP0 em alertas atuais.
4. Reusa R3I/07_rescue_candidates.csv e gera rescues novos dos FNs residuais.
5. Testa caps de FP absolutos, por padrão 5000,5050,5100.
6. Faz re-tightening TP0 só nos alertas adicionados.
7. Recomenda o menor FN dentro do preferred cap, se houver.
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

ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.parent/'dados').exists() else Path.cwd()
DEFAULT_INPUT = ROOT/'resultados'/'experimentos'/'EXP-014B-R3K'/'10_predictions_recommended.csv'
DEFAULT_RESCUE = ROOT/'resultados'/'experimentos'/'EXP-014B-R3I'/'07_rescue_candidates.csv'
DEFAULT_OUT = ROOT/'resultados'/'experimentos'/'EXP-014B-R3L'
BASE_COL='exp014b_r3k_recommended_pred'
FINAL_COL='exp014b_r3l_recommended_pred'
FEATURE_COLS=['ds_tipo_chave_norm','value_band','mbk_available_flag','first_receiver_flag_real','periodo_dia','module_quiet','lgbm_bin','if_bin','score_bin','ratio_bin','qtd_rec_bin','vl_bin']
NUMERIC_COLS=['lgbm_r4_score','lgbm_mapped','lgbm_raw','score_final','if_percentile','if_percentile_x','if_percentile_y','vl_pix','ratio_valor_media_pagador_90d','qtd_pix_recebidos_180d','valor_total_recebido_180d']

@dataclass
class Cand:
    rule_id: str
    family: str
    description: str
    mask: np.ndarray
    fp_effect: int
    tp_loss: int = 0
    params: dict[str,Any] | None = None

def log(x): print(x, flush=True)
def dump(obj,path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def parse(raw):
    if isinstance(raw,dict): return raw
    return json.loads(str(raw).replace('Infinity','1e999'))

def normalize(df):
    df=df.copy(); df.columns=[str(c).strip().split('.')[-1] for c in df.columns]
    if 'transaction_id' not in df.columns and 'cd_pix' in df.columns: df['transaction_id']=df['cd_pix']
    if 'event_datetime' not in df.columns and 'dt_pix' in df.columns: df['event_datetime']=df['dt_pix']
    if 'is_fraud' not in df.columns: raise RuntimeError('Coluna obrigatoria ausente: is_fraud')
    df['is_fraud']=pd.to_numeric(df['is_fraud'],errors='coerce').fillna(0).astype(int)
    for c in [BASE_COL,'exp014b_r3j_frozen_pred','exp014b_r3k_recommended_pred']:
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
    for i in range(len(edges)-1):
        a,b=edges[i],edges[i+1]
        if np.isneginf(a): labels.append(f'{name}_LT_{b:g}')
        elif np.isposinf(b): labels.append(f'{name}_GE_{a:g}')
        else: labels.append(f'{name}_{a:g}_{b:g}')
    return pd.cut(vals,bins=edges,labels=labels,include_lowest=True).astype('string').fillna(f'{name}_MISSING').astype(str)

def add_bins(df):
    df=df.copy()
    if 'lgbm_bin' not in df.columns and pick(df,['lgbm_r4_score','lgbm_mapped','lgbm_raw']): df['lgbm_bin']=qbin(num(df,['lgbm_r4_score','lgbm_mapped','lgbm_raw']),'lgbm',[.001,.003,.005,.01,.02,.05,.1])
    if 'if_bin' not in df.columns and pick(df,['if_percentile','if_percentile_x','if_percentile_y']): df['if_bin']=qbin(num(df,['if_percentile','if_percentile_x','if_percentile_y']),'if',[.32,.5,.7,.85,.95])
    if 'score_bin' not in df.columns and 'score_final' in df.columns: df['score_bin']=qbin(num(df,'score_final'),'score',[.5,1,2,3,5,10])
    if 'ratio_bin' not in df.columns and 'ratio_valor_media_pagador_90d' in df.columns: df['ratio_bin']=qbin(num(df,'ratio_valor_media_pagador_90d'),'ratio',[.05,.1,.2,.5,1,2,5])
    if 'qtd_rec_bin' not in df.columns and 'qtd_pix_recebidos_180d' in df.columns: df['qtd_rec_bin']=qbin(num(df,'qtd_pix_recebidos_180d'),'qtdrec',[0,1,2,5,10,20,50,100])
    if 'vl_bin' not in df.columns and 'vl_pix' in df.columns: df['vl_bin']=qbin(num(df,'vl_pix'),'vl',[20,50,100,250,500,1000,5000,10000])
    if 'module_quiet' not in df.columns:
        se=num(df,['se_score_x','se_score_y','se_score']); sec=num(df,['se_patterns_count','se_pattern_count']); be=num(df,['beh_score','behavioral_score']); bec=num(df,['beh_factors_count','behavioral_risk_factor_count']); rt=num(df,'runtime_flagged')
        strong=(se>=40)|(sec>=2)|(be>=25)|(bec>=2)|(rt>=1)
        df['module_quiet']=np.where(strong,'module_strong','module_quiet')
    return df

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(np.asarray(y).astype(int),np.asarray(p).astype(int),labels=[0,1]).ravel()
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),'precision':round(float(precision_score(y,p,zero_division=0)),8),'recall':round(float(recall_score(y,p,zero_division=0)),8),'f1':round(float(f1_score(y,p,zero_division=0)),8),'fpr':round(float(fp/max(fp+tn,1)),8)}

def wilson(tp,n,z=1.959963984540054):
    if n<=0: return (float('nan'),float('nan'))
    ph=tp/n; den=1+z*z/n; cen=(ph+z*z/(2*n))/den; mar=z*((ph*(1-ph)/n)+(z*z/(4*n*n)))**0.5/den
    return max(0,cen-mar),min(1,cen+mar)

def feat_frame(df):
    out=pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        if c in df.columns: out[c]=df[c].astype('string').fillna('<MISSING>').astype(str)
    return out

def cands_df(cands,scenario=''):
    return pd.DataFrame([{'scenario':scenario,'rule_id':c.rule_id,'family':c.family,'description':c.description,'fp_effect':c.fp_effect,'tp_loss':c.tp_loss,'params_json':json.dumps(c.params or {},ensure_ascii=False)} for c in cands])

def add_head(out,df,y,mask,fam,desc,min_fp,params):
    if not mask.any(): return
    tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
    if tp==0 and fp>=min_fp: out.append(Cand(f'head_{len(out):05d}',fam,desc,mask.copy(),fp,0,params))

def mine_headroom(df,pred,min_fp=10,max_combo=3,top_groups=40):
    y=df['is_fraud'].to_numpy(int); alerted=pred.astype(int)==1; out=[]
    for c in NUMERIC_COLS:
        if c not in df.columns: continue
        vals=num(df,c).to_numpy(float); active=vals[alerted]
        if len(active)==0: continue
        try: cuts=sorted(set(float(x) for x in np.quantile(active,[.03,.05,.1,.2,.3,.5,.7,.9]) if np.isfinite(x)))
        except Exception: cuts=[]
        for cut in cuts:
            for d in ['le','ge']:
                mask=alerted & ((vals<=cut) if d=='le' else (vals>=cut))
                add_head(out,df,y,mask,'headroom_numeric_tp0',f'base_alert AND {c}{"<=" if d=="le" else ">="}{cut:g}',min_fp,{'type':'numeric_headroom','col':c,'direction':d,'cut':cut})
    feat=feat_frame(df); cols=list(feat.columns); idx=np.where(alerted)[0]
    bins=[c for c in cols if c.endswith('_bin') or c=='module_quiet']; important=['ds_tipo_chave_norm','value_band','mbk_available_flag','first_receiver_flag_real','periodo_dia']
    for r in range(1,max_combo+1):
        for combo in itertools.combinations(cols,r):
            combo=list(combo)
            if r==1 and combo[0] not in bins+['ds_tipo_chave_norm','value_band']: continue
            if r>=2 and not any(c in combo for c in important+bins): continue
            sub=feat.iloc[idx][combo]
            if sub.empty: continue
            groups=[]
            for key,rel in sub.groupby(combo,dropna=False).indices.items():
                ids=sub.iloc[list(rel)].index.to_numpy(int)
                if len(ids)<min_fp: continue
                mask=np.zeros(len(df),bool); mask[ids]=True; mask &= alerted
                tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
                if tp==0 and fp>=min_fp: groups.append((-fp,key,mask,fp))
            groups.sort()
            for _,key,mask,fp in groups[:top_groups]:
                vals=key if isinstance(key,tuple) else (key,); vals=[str(v) for v in vals]
                desc='base_alert AND '+' AND '.join(f'{c}={v}' for c,v in zip(combo,vals))
                add_head(out,df,y,mask,'headroom_combo_tp0',desc,min_fp,{'type':'combo_headroom','combo_cols':combo,'combo_values':vals})
    best={}
    for c in out:
        k=np.packbits(c.mask).tobytes(); old=best.get(k)
        if old is None or c.fp_effect>old.fp_effect: best[k]=c
    out=list(best.values()); out.sort(key=lambda c:(-c.fp_effect,len(c.description)))
    return out

def greedy_veto(cands,pred,y,max_rules,max_seconds):
    t0=time.perf_counter(); cur=pred.copy(); selected=[]; used=set()
    for _ in range(max_rules):
        if time.perf_counter()-t0>=max_seconds: break
        best=None; alerted=cur.astype(int)==1
        for i,c in enumerate(cands[:1000]):
            if i in used: continue
            mask=c.mask & alerted; tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
            if tp==0 and fp>0:
                rank=(fp,-int(mask.sum()))
                if best is None or rank>best[0]: best=(rank,i,c,mask,fp)
        if best is None: break
        _,i,c,mask,fp=best; used.add(i); cur[mask]=0; selected.append(Cand(c.rule_id,c.family,c.description,mask.copy(),fp,0,c.params))
    return cur,selected

def rescue_mask(df,pred,params):
    not_alerted=pred.astype(int)==0
    if params.get('type')=='numeric_threshold_rescue':
        c=params.get('col')
        if c not in df.columns: return np.zeros(len(df),bool)
        vals=num(df,c).to_numpy(float); return not_alerted & ((vals>=float(params['cut'])) if params.get('direction')=='ge' else (vals<=float(params['cut'])))
    if params.get('type')=='combo_rescue':
        mask=not_alerted.copy()
        for c,v in zip(params.get('combo_cols',[]),params.get('combo_values',[])):
            if c not in df.columns: return np.zeros(len(df),bool)
            mask &= (df[c].astype('string').fillna('<MISSING>').astype(str).to_numpy()==str(v))
        return mask
    return np.zeros(len(df),bool)

def add_rescue(rows,df,pred,mask,fam,desc,params,min_fn,max_fp):
    y=df['is_fraud'].to_numpy(int); mask=mask & (pred.astype(int)==0)
    fn=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
    if fn>=min_fn and fp<=max_fp: rows.append({'candidate_id':f'r3l_rescue_{len(rows):05d}','family':fam,'description':desc,'fn_recovered':fn,'fp_added':fp,'n_added':int(mask.sum()),'fp_per_fn':fp/max(fn,1),'params_json':json.dumps(params,ensure_ascii=False)})

def build_rescues(df,pred,lib,min_fn,max_fp,max_combo,top_groups):
    rows=[]; y=df['is_fraud'].to_numpy(int); not_alerted=pred.astype(int)==0
    if lib is not None and not lib.empty and 'params_json' in lib.columns:
        for _,r in lib.iterrows():
            params=parse(r['params_json']); mask=rescue_mask(df,pred,params)
            add_rescue(rows,df,pred,mask,'reused_r3i_rescue',str(r.get('description')),params,min_fn,max_fp)
    for c in NUMERIC_COLS:
        if c not in df.columns: continue
        vals=num(df,c).to_numpy(float); fn_vals=vals[(y==1)&not_alerted]
        if len(fn_vals)==0: continue
        try: cuts=sorted(set(float(x) for x in np.quantile(fn_vals,[0,.05,.1,.25,.5,.75,.9]) if np.isfinite(x)))
        except Exception: cuts=[]
        for cut in cuts:
            for d in ['ge','le']:
                mask=not_alerted & ((vals>=cut) if d=='ge' else (vals<=cut))
                add_rescue(rows,df,pred,mask,'new_numeric_residual_fn_rescue',f'{c}{">=" if d=="ge" else "<="}{cut:g}',{'type':'numeric_threshold_rescue','col':c,'direction':d,'cut':cut},min_fn,max_fp)
    feat=feat_frame(df); cols=list(feat.columns); idx=np.where(not_alerted)[0]
    for r in range(1,max_combo+1):
        for combo in itertools.combinations(cols,r):
            combo=list(combo)
            if r==1 and combo[0] not in ['lgbm_bin','if_bin','score_bin','ds_tipo_chave_norm','value_band','ratio_bin']: continue
            sub=feat.iloc[idx][combo]
            if sub.empty: continue
            groups=[]
            for key,rel in sub.groupby(combo,dropna=False).indices.items():
                ids=sub.iloc[list(rel)].index.to_numpy(int); mask=np.zeros(len(df),bool); mask[ids]=True
                fn=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
                if fn>=min_fn and fp<=max_fp: groups.append((fp/max(fn,1),-fn,key,mask))
            groups.sort()
            for _,_,key,mask in groups[:top_groups]:
                vals=key if isinstance(key,tuple) else (key,); vals=[str(v) for v in vals]
                desc=' AND '.join(f'{c}={v}' for c,v in zip(combo,vals))
                add_rescue(rows,df,pred,mask,'new_combo_residual_fn_rescue',desc,{'type':'combo_rescue','combo_cols':combo,'combo_values':vals},min_fn,max_fp)
    out=pd.DataFrame(rows)
    if out.empty: return out
    out=out.drop_duplicates(subset=['params_json']).sort_values(['fp_per_fn','fp_added','fn_recovered'],ascending=[True,True,False]).reset_index(drop=True)
    out['candidate_id']=[f'r3l_rescue_{i:05d}' for i in range(len(out))]
    return out

def apply_rescue(df,pred,row):
    params=parse(row['params_json']); mask=rescue_mask(df,pred,params); y=df['is_fraud'].to_numpy(int)
    gain={'fn_recovered':int(((y==1)&mask).sum()),'fp_added':int(((y==0)&mask).sum()),'n_added':int(mask.sum())}
    new=pred.copy(); new[mask]=1; return new,gain

def greedy_rescue_caps(df,pred,cands,caps):
    y=df['is_fraud'].to_numpy(int); base=metrics(y,pred); rows=[]; preds={}; cand_rows=cands.head(1500).to_dict('records') if not cands.empty else []
    for cap in caps:
        cur=pred.copy(); selected=[]
        while True:
            curm=metrics(y,cur); best=None
            for cand in cand_rows:
                if cand['candidate_id'] in selected: continue
                new,gain=apply_rescue(df,cur,cand)
                if gain['fn_recovered']<=0: continue
                if curm['fp']+gain['fp_added']>cap: continue
                rank=(gain['fn_recovered']/max(gain['fp_added'],1),gain['fn_recovered'],-gain['fp_added'])
                if best is None or rank>best[0]: best=(rank,cand,new,gain)
            if best is None: break
            _,cand,cur,gain=best; selected.append(cand['candidate_id'])
        m=metrics(y,cur); scen=f'r3l_cap_{cap}'
        rows.append({'scenario':scen,'fp_cap':cap,'n_selected_rescues':len(selected),'selected_rescue_candidate_ids':'|'.join(selected),'fn_recovered_vs_headroom':base['fn']-m['fn'],'fp_added_vs_headroom':m['fp']-base['fp'],**m})
        preds[scen]=cur
    out=pd.DataFrame(rows)
    if not out.empty: out=out.sort_values(['fn','fp'],ascending=[True,True]).reset_index(drop=True)
    return out,preds

def mine_retighten(df,head_pred,scenario_pred,min_fp,max_combo,top_groups):
    y=df['is_fraud'].to_numpy(int); added=(scenario_pred.astype(int)==1)&(head_pred.astype(int)==0); out=[]
    def add(mask,fam,desc,params):
        if not mask.any(): return
        tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
        if tp==0 and fp>=min_fp: out.append(Cand(f'ret_{len(out):05d}',fam,desc,mask.copy(),fp,0,params))
    for c in NUMERIC_COLS:
        if c not in df.columns: continue
        vals=num(df,c).to_numpy(float); active=vals[added]
        if len(active)==0: continue
        try: cuts=sorted(set(float(x) for x in np.quantile(active,[.05,.1,.2,.3,.5,.7,.9]) if np.isfinite(x)))
        except Exception: cuts=[]
        for cut in cuts:
            for d in ['le','ge']:
                add(added & ((vals<=cut) if d=='le' else (vals>=cut)),'retighten_numeric_added_only',f'added_only AND {c}{"<=" if d=="le" else ">="}{cut:g}',{'type':'numeric_retighten','scope':'rescue_added_only','col':c,'direction':d,'cut':cut})
    feat=feat_frame(df); cols=list(feat.columns); idx=np.where(added)[0]; bins=[c for c in cols if c.endswith('_bin') or c=='module_quiet']; important=['ds_tipo_chave_norm','value_band','mbk_available_flag','first_receiver_flag_real','periodo_dia']
    for r in range(1,max_combo+1):
        for combo in itertools.combinations(cols,r):
            combo=list(combo)
            if r==1 and combo[0] not in bins+['ds_tipo_chave_norm','value_band']: continue
            if r>=2 and not any(c in combo for c in important+bins): continue
            sub=feat.iloc[idx][combo]
            if sub.empty: continue
            groups=[]
            for key,rel in sub.groupby(combo,dropna=False).indices.items():
                ids=sub.iloc[list(rel)].index.to_numpy(int)
                if len(ids)<min_fp: continue
                mask=np.zeros(len(df),bool); mask[ids]=True; mask&=added
                tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
                if tp==0 and fp>=min_fp: groups.append((-fp,key,mask,fp))
            groups.sort()
            for _,key,mask,fp in groups[:top_groups]:
                vals=key if isinstance(key,tuple) else (key,); vals=[str(v) for v in vals]
                desc='added_only AND '+' AND '.join(f'{c}={v}' for c,v in zip(combo,vals))
                add(mask,'retighten_combo_added_only',desc,{'type':'combo_retighten','scope':'rescue_added_only','combo_cols':combo,'combo_values':vals})
    best={}
    for c in out:
        k=np.packbits(c.mask).tobytes(); old=best.get(k)
        if old is None or c.fp_effect>old.fp_effect: best[k]=c
    out=list(best.values()); out.sort(key=lambda c:(-c.fp_effect,len(c.description)))
    return out

def select(after,preferred_cap):
    under=after[after['fp']<=preferred_cap]
    if not under.empty: return str(under.sort_values(['fn','fp'],ascending=[True,True]).iloc[0]['scenario'])
    return str(after.sort_values(['fn','fp'],ascending=[True,True]).iloc[0]['scenario'])

def make_report(summary,base,head,scen,after,head_rules,sel_rules):
    lines=['# EXP-014B-R3L — Headroom + Residual FN Optimizer','', '## Resultado', f"- Status: `{summary['objective_status']}`", f"- Cenário recomendado: `{summary['recommended_scenario']}`", f"- Métricas recomendadas: `{summary['recommended_metrics']}`", '', '## Base e headroom', f'- Base R3K: `{base}`', f'- Após headroom TP0: `{head}`','']
    lines += ['## Cenários antes do re-tightening', scen[[c for c in ['scenario','fp_cap','fn_recovered_vs_headroom','fp_added_vs_headroom','tp','fp','fn','precision','recall','fpr'] if c in scen.columns]].to_markdown(index=False),'']
    lines += ['## Cenários após re-tightening', after[[c for c in ['scenario','net_fn_recovered_vs_base','net_fp_delta_vs_base','headroom_fp_removed','retightening_fp_removed','tp','fp','fn','precision','recall','fpr'] if c in after.columns]].to_markdown(index=False),'']
    lines += ['## Headroom selecionado', head_rules[[c for c in ['rule_id','family','description','fp_effect','tp_loss'] if c in head_rules.columns]].to_markdown(index=False) if not head_rules.empty else 'Nenhuma regra de headroom selecionada.','']
    lines += ['## Retightening selecionado', sel_rules[[c for c in ['scenario','rule_id','family','description','fp_effect','tp_loss'] if c in sel_rules.columns]].to_markdown(index=False) if not sel_rules.empty else 'Nenhuma regra de retightening selecionada.','']
    lines.append('## Próximo passo')
    lines.append('Se o cenário recomendado reduzir FN mantendo FP dentro do cap, executar validação congelada do R3L. Caso contrário, consolidar R3K e migrar para hard-negative mining/segundo estágio.')
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input',default=str(DEFAULT_INPUT)); ap.add_argument('--rescue-library',default=str(DEFAULT_RESCUE)); ap.add_argument('--output-dir',default=str(DEFAULT_OUT))
    ap.add_argument('--fp-caps',default='5000,5050,5100'); ap.add_argument('--preferred-fp-cap',type=int,default=5000)
    ap.add_argument('--min-fp-removed-headroom',type=int,default=10); ap.add_argument('--max-headroom-rules',type=int,default=5); ap.add_argument('--max-seconds-headroom',type=int,default=120); ap.add_argument('--max-headroom-combo-size',type=int,default=3)
    ap.add_argument('--max-fp-added-candidate',type=int,default=500); ap.add_argument('--min-fn-recovered',type=int,default=1); ap.add_argument('--max-new-combo-size',type=int,default=3); ap.add_argument('--top-groups-per-combo',type=int,default=50)
    ap.add_argument('--min-fp-removed-retighten',type=int,default=5); ap.add_argument('--max-retighten-rules',type=int,default=5); ap.add_argument('--max-seconds-retighten',type=int,default=90)
    ap.add_argument('--no-write-predictions',action='store_true')
    args=ap.parse_args(); t0=time.perf_counter(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    inp=Path(args.input); libp=Path(args.rescue_library)
    log('='*80); log('EXP-014B-R3L — Headroom + Residual FN Optimizer'); log('='*80)
    if not inp.exists(): raise FileNotFoundError(f'Input não encontrado: {inp}')
    if not libp.exists(): raise FileNotFoundError(f'Biblioteca de resgate não encontrada: {libp}')
    df=add_bins(normalize(pd.read_csv(inp,low_memory=False))); lib=pd.read_csv(libp)
    missing=[]
    if BASE_COL not in df.columns: missing.append(BASE_COL)
    if 'params_json' not in lib.columns: missing.append('rescue_library.params_json')
    contract={'n_rows':int(len(df)),'n_frauds':int(df['is_fraud'].sum()),'base_col':BASE_COL,'rescue_library_rows':int(len(lib)),'missing':missing,'contract_ok':not missing}
    dump(contract,out/'01_input_contract.json')
    if missing: raise RuntimeError(f'Contrato falhou: {missing}')
    y=df['is_fraud'].to_numpy(int); base_pred=df[BASE_COL].to_numpy(int); base=metrics(y,base_pred)
    pd.DataFrame([{'policy_name':'R3K_BASE',**base}]).to_csv(out/'02_base_metrics.csv',index=False)
    log(f'Base metrics: {base}')
    head_cands=mine_headroom(df,base_pred,args.min_fp_removed_headroom,args.max_headroom_combo_size,args.top_groups_per_combo)
    cands_df(head_cands).to_csv(out/'03_headroom_candidates.csv',index=False)
    head_pred,head_sel=greedy_veto(head_cands,base_pred,y,args.max_headroom_rules,args.max_seconds_headroom)
    head_rules=cands_df(head_sel); head_rules.to_csv(out/'04_headroom_selected_rules.csv',index=False)
    head=metrics(y,head_pred); pd.DataFrame([{'policy_name':'R3L_HEADROOM_BASE',**head}]).to_csv(out/'05_headroom_metrics.csv',index=False)
    headroom_fp=base['fp']-head['fp']; log(f'Headroom metrics: {head}')
    rescues=build_rescues(df,head_pred,lib,args.min_fn_recovered,args.max_fp_added_candidate,args.max_new_combo_size,args.top_groups_per_combo)
    rescues.to_csv(out/'06_rescue_candidates.csv',index=False); log(f'Rescue candidates: {len(rescues)}')
    caps=[int(x.strip()) for x in str(args.fp_caps).split(',') if x.strip()]
    scen, scen_preds=greedy_rescue_caps(df,head_pred,rescues,caps); scen.to_csv(out/'07_rescue_scenarios_before_retightening.csv',index=False)
    after_rows=[]; summaries=[]; selected_dfs=[]; artifacts={}; final_preds={}
    for _,row in scen.iterrows():
        scenario=str(row['scenario']); scen_pred=scen_preds[scenario]; rescue_m=metrics(y,scen_pred)
        ret_cands=mine_retighten(df,head_pred,scen_pred,args.min_fpr if False else args.min_fp_removed_retighten,args.max_new_combo_size,args.top_groups_per_combo)
        final_pred,ret_sel=greedy_veto(ret_cands,scen_pred,y,args.max_retighten_rules,args.max_seconds_retighten)
        final=metrics(y,final_pred); final_preds[scenario]=final_pred
        ret_fp=rescue_m['fp']-final['fp']; ret_tp=rescue_m['tp']-final['tp']
        summaries.append({'scenario':scenario,'n_retightening_candidates':len(ret_cands),'n_selected_retighten_rules':len(ret_sel),'retightening_fp_removed':ret_fp,'retightening_tp_loss':ret_tp})
        sel_df=cands_df(ret_sel,scenario); selected_dfs.append(sel_df)
        after_rows.append({'scenario':scenario,'fp_cap':int(row['fp_cap']),'headroom_fp_removed':int(headroom_fp),'fn_recovered_before_retighten':int(row['fn_recovered_vs_headroom']),'fp_added_before_retighten':int(row['fp_added_vs_headroom']),'retightening_fp_removed':int(ret_fp),'retightening_tp_loss':int(ret_tp),'net_fn_recovered_vs_base':int(base['fn']-final['fn']),'net_fp_delta_vs_base':int(final['fp']-base['fp']),**final,'selected_rescue_candidate_ids':row['selected_rescue_candidate_ids'],'selected_retighten_rule_ids':'|'.join(c.rule_id for c in ret_sel)})
        artifacts[scenario]={'scenario':scenario,'fp_cap':int(row['fp_cap']),'selected_headroom_rules':head_rules.to_dict('records') if not head_rules.empty else [],'selected_rescue_candidate_ids':str(row['selected_rescue_candidate_ids']).split('|') if pd.notna(row['selected_rescue_candidate_ids']) and str(row['selected_rescue_candidate_ids']) else [],'selected_retighten_rules':sel_df.to_dict('records') if not sel_df.empty else [],'metrics_after_headroom':head,'metrics_before_retightening':rescue_m,'final_metrics':final}
        log(f'  {scenario}: final={final}')
    pd.DataFrame(summaries).to_csv(out/'08_retightening_candidate_summary.csv',index=False)
    after=pd.DataFrame(after_rows); after.to_csv(out/'09_scenario_metrics_after_retightening.csv',index=False)
    selected=pd.concat(selected_dfs,ignore_index=True) if selected_dfs else pd.DataFrame(); selected.to_csv(out/'10_selected_rules_by_scenario.csv',index=False)
    rec=select(after,args.preferred_fp_cap); rec_pred=final_preds[rec]; rec_m=metrics(y,rec_pred); wl,wh=wilson(rec_m['tp'],int(df['is_fraud'].sum()))
    df[FINAL_COL]=rec_pred.astype(int); df['exp014b_r3l_recommended_scenario']=rec
    status='DONE_HEADROOM_RESCUE_RETIGHTEN'; status += '_FN_IMPROVED_VS_R3K' if rec_m['fn']<base['fn'] else '_FN_NOT_IMPROVED_VS_R3K'; status += '_FP_WITHIN_PREFERRED_CAP' if rec_m['fp']<=args.preferred_fp_cap else '_FP_ABOVE_PREFERRED_CAP'
    artifact={'experiment':'EXP-014B-R3L','policy_name':'headroom_residual_fn_optimizer','objective_status':status,'base_r3k_metrics':base,'headroom_metrics':head,'recommended_scenario':rec,'recommended_metrics':rec_m,'wilson_low':wl,'wilson_high':wh,'scenario_artifacts':artifacts,'constraints':{'fp_caps':caps,'preferred_fp_cap':args.preferred_fp_cap,'max_headroom_rules':args.max_headroom_rules,'max_retighten_rules':args.max_retighten_rules,'max_fp_added_candidate':args.max_fp_added_candidate},'notes':['Short experiment: creates TP0 FP headroom, then applies residual FN rescues.','Needs frozen validation before promotion.']}
    dump(artifact,out/'11_policy_artifact_recommended.json')
    if not args.no_write_predictions: df.to_csv(out/'12_predictions_recommended.csv',index=False)
    summary={'experiment':'EXP-014B-R3L','status':'DONE','objective_status':status,'input_path':str(inp),'rescue_library_path':str(libp),'n_rows':int(len(df)),'n_frauds':int(df['is_fraud'].sum()),'base_metrics':base,'headroom_metrics':head,'headroom_fp_removed':int(headroom_fp),'n_headroom_candidates':int(len(head_cands)),'n_headroom_selected':int(len(head_sel)),'n_rescue_candidates':int(len(rescues)),'fp_caps':caps,'preferred_fp_cap':args.preferred_fp_cap,'recommended_scenario':rec,'recommended_metrics':rec_m,'recommended_wilson_low':wl,'recommended_wilson_high':wh,'elapsed_seconds':round(time.perf_counter()-t0,2),'output_dir':str(out)}
    dump(summary,out/'00_run_summary.json')
    (out/'13_exp014b_r3l_report.md').write_text(make_report(summary,base,head,scen,after,head_rules,selected),encoding='utf-8')
    log(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
