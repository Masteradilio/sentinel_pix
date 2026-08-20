# -*- coding: utf-8 -*-
"""EXP-014B-R4A-FROZEN - replay do artifact campeão R4A."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import pandas as pd

EXPERIMENT='EXP-014B-R4A-FROZEN'
LABELS=['is_fraud','fraude','target','label','tp_fraude']

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--base-predictions'); p.add_argument('--artifact'); p.add_argument('--r4a-predictions'); p.add_argument('--output-dir'); return p.parse_args()

def defaults():
    r=Path.cwd(); return (r/'resultados/experimentos/EXP-014B-R3Z-FROZEN/06_predictions_frozen.csv', r/'resultados/experimentos/EXP-014B-R4A/08_policy_artifact_recommended.json', r/'resultados/experimentos/EXP-014B-R4A/09_predictions_recommended.csv', r/'resultados/experimentos'/EXPERIMENT)

def find_col(df,cands):
    low={c.lower():c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in low: return low[c.lower()]
    raise KeyError(f'coluna ausente: {cands}')

def I(s): return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
def A(s): return s.astype(str).str.strip().str.upper()
def inter(a): return A(a).isin(['CONFIRMAR','BLOQUEAR']).astype(int)
def blk(a): return A(a).eq('BLOQUEAR').astype(int)

def m(y,p):
    y=I(y); p=I(p); tp=int(((y==1)&(p==1)).sum()); fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum()); tn=int(((y==0)&(p==0)).sum())
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0; f1=2*pr*rc/(pr+rc) if pr+rc else 0.0; fpr=fp/(fp+tn) if fp+tn else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':round(pr,8),'recall':round(rc,8),'f1':round(f1,8),'fpr':round(fpr,8)}

def mask_desc(df,desc):
    pref='Demover CONFIRMAR R4A com '
    if not desc.startswith(pref): raise ValueError('descricao nao suportada: '+desc)
    mask=pd.Series(True,index=df.index)
    for part in desc[len(pref):].split(' AND '):
        if ' == ' in part:
            c,v=part.split(' == ',1); mask &= df[c].fillna('<MISSING>').astype(str).eq(str(v))
        elif ' <= ' in part:
            c,v=part.split(' <= ',1); mask &= pd.to_numeric(df[c], errors='coerce').le(float(v))
        elif ' >= ' in part:
            c,v=part.split(' >= ',1); mask &= pd.to_numeric(df[c], errors='coerce').ge(float(v))
        else: raise ValueError('parte nao suportada: '+part)
    return mask

def by_action(df,label,act):
    y=I(df[label]); rows=[]
    for a,idx in df.groupby(act, dropna=False).groups.items():
        yy=y.loc[list(idx)]; n=int(len(yy)); f=int((yy==1).sum()); normal=int((yy==0).sum())
        rows.append({'action':str(a),'n_rows':n,'n_frauds':f,'n_normals':normal,'precision_within_action':round(f/n,8) if n else 0.0})
    return pd.DataFrame(rows).sort_values('action')

def wj(p,o): p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    a=parse_args(); db,da,dr,do=defaults(); artp=Path(a.artifact) if a.artifact else da
    if not artp.exists(): raise FileNotFoundError(f'artifact nao encontrado: {artp}')
    art=json.loads(artp.read_text(encoding='utf-8'))
    base=Path(a.base_predictions) if a.base_predictions else Path(art.get('input_predictions_path') or db)
    if not base.exists(): base=db
    if not base.exists(): raise FileNotFoundError(f'base predictions nao encontrado: {base}')
    ref=Path(a.r4a_predictions) if a.r4a_predictions else dr; out=Path(a.output_dir) if a.output_dir else do; out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(base, low_memory=False); label=find_col(df,LABELS); base_action_col=art['base_action_col']
    action=A(df[base_action_col]); demote=pd.Series(False,index=df.index); confirm=action.eq('CONFIRMAR')
    for rule in art.get('selected_demotions',[]): demote |= confirm & mask_desc(df,str(rule['description']))
    final=action.copy(); final[demote]='APROVAR'
    df['exp014b_r4a_frozen_demote_confirm_to_approve']=demote.astype(int); df['r4a_frozen_decisao_recommended']=final
    df['exp014b_r4a_frozen_intervention_pred']=inter(final); df['exp014b_r4a_frozen_block_pred']=blk(final)
    fm=m(df[label],df['exp014b_r4a_frozen_intervention_pred']); bm=m(df[label],df['exp014b_r4a_frozen_block_pred'])
    nmis=0; misdf=pd.DataFrame()
    if ref.exists():
        r=pd.read_csv(ref, low_memory=False); mismatch=pd.Series(False,index=df.index)
        ac=art.get('final_action_col','r4a_decisao_recommended'); ic=art.get('intervention_pred_col','exp014b_r4a_intervention_pred'); bc=art.get('block_pred_col','exp014b_r4a_block_pred')
        if ac in r.columns: mismatch |= A(r[ac]).ne(A(df['r4a_frozen_decisao_recommended']))
        if ic in r.columns: mismatch |= I(r[ic]).ne(I(df['exp014b_r4a_frozen_intervention_pred']))
        if bc in r.columns: mismatch |= I(r[bc]).ne(I(df['exp014b_r4a_frozen_block_pred']))
        nmis=int(mismatch.sum()); cols=[c for c in [label,base_action_col,'r4a_frozen_decisao_recommended','score_final','lgbm_r4_score'] if c in df.columns]; misdf=df.loc[mismatch,cols]
    ok=(fm==art['final_intervention_metrics'] and bm==art['final_block_metrics'] and nmis==0); status='PASS_R4A_FROZEN_VALIDATED_REPLAY_OK' if ok else 'FAIL_R4A_FROZEN_VALIDATION_MISMATCH'
    ba=by_action(df,label,'r4a_frozen_decisao_recommended'); y=I(df[label])
    val={'status':status,'base_predictions_path':str(base),'artifact_path':str(artp),'reference_predictions_path':str(ref) if ref.exists() else None,'base_action_col':base_action_col,'frozen_action_col':'r4a_frozen_decisao_recommended','frozen_intervention_col':'exp014b_r4a_frozen_intervention_pred','frozen_block_col':'exp014b_r4a_frozen_block_pred','expected_intervention_metrics':art['final_intervention_metrics'],'frozen_intervention_metrics':fm,'expected_block_metrics':art['final_block_metrics'],'frozen_block_metrics':bm,'intervention_match_expected':fm==art['final_intervention_metrics'],'block_match_expected':bm==art['final_block_metrics'],'n_any_mismatches':nmis,'all_pass':ok}
    frozen={**art,'experiment':EXPERIMENT,'frozen_validation_status':status,'frozen_action_col':'r4a_frozen_decisao_recommended','frozen_demote_col':'exp014b_r4a_frozen_demote_confirm_to_approve','frozen_intervention_pred_col':'exp014b_r4a_frozen_intervention_pred','frozen_block_pred_col':'exp014b_r4a_frozen_block_pred','frozen_intervention_metrics':fm,'frozen_block_metrics':bm,'validation':val}
    summ={'experiment':EXPERIMENT,'status':'DONE','objective_status':status,'n_rows':int(len(df)),'n_frauds':int((y==1).sum()),'n_normals':int((y==0).sum()),'frozen_intervention_metrics':fm,'frozen_block_metrics':bm,'n_selected_demotions':int(len(art.get('selected_demotions',[]))),'n_any_mismatches':nmis,'all_pass':ok,'output_dir':str(out)}
    contract={'base_predictions_path':str(base),'artifact_path':str(artp),'label_col':label,'base_action_col':base_action_col,'n_selected_demotions':int(len(art.get('selected_demotions',[]))),'contract_ok':True,'missing':[]}
    wj(out/'00_run_summary.json',summ); wj(out/'01_input_contract.json',contract); wj(out/'02_frozen_validation.json',val); ba.to_csv(out/'03_decision_metrics_by_action.csv',index=False,encoding='utf-8'); misdf.to_csv(out/'04_prediction_mismatches.csv',index=False,encoding='utf-8'); wj(out/'05_policy_artifact_frozen.json',frozen); df.to_csv(out/'06_predictions_frozen.csv',index=False,encoding='utf-8')
    (out/'07_exp014b_r4a_frozen_report.md').write_text(f"""# {EXPERIMENT}\n\nStatus: `{status}`\n\n## Intervenção congelada\n```json\n{json.dumps(fm, ensure_ascii=False, indent=2)}\n```\n\n## BLOQUEAR congelado\n```json\n{json.dumps(bm, ensure_ascii=False, indent=2)}\n```\n\n## Ações\n{ba.to_markdown(index=False)}\n""",encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
