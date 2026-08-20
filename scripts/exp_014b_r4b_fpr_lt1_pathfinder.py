# -*- coding: utf-8 -*-
"""EXP-014B-R4B — meta final FPR < 1%, FN total <= 5.

Parte do R4A-FROZEN, se existir; fallback para R4A recomendado.
Ponto crítico: com 112379 normais, FPR<1% exige FP<=1123. Como o BLOQUEAR
atual tem 1200 FP, CONFIRMAR-only é matematicamente inviável. Por isso o R4B
pode demover qualquer intervenção (CONFIRMAR ou BLOQUEAR) para APROVAR, dentro
de FN total <=5, e reporta claramente impacto no BLOQUEAR.
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
from typing import Any
import pandas as pd

EXPERIMENT='EXP-014B-R4B'
LABELS=['is_fraud','fraude','target','label','tp_fraude']
ACTIONS=['r4a_frozen_decisao_recommended','r4a_decisao_recommended','r3z_frozen_decisao_recommended','r3z_decisao_recommended']
BLOCKS=['exp014b_r4a_frozen_block_pred','exp014b_r4a_block_pred','exp014b_r3z_frozen_block_pred','exp014b_r3z_block_pred']
INTERS=['exp014b_r4a_frozen_intervention_pred','exp014b_r4a_intervention_pred','exp014b_r3z_frozen_intervention_pred','exp014b_r3z_intervention_pred']
SCORES=['lgbm_r4_score','score_final','lgbm_raw','lgbm_mapped','peso_total','if_percentile','se_score','beh_score','topaz_risk_score','exp014b_r3s_second_stage_score','exp014b_r3u_receiver_relationship_trust_score']
CATS=['_r4b_base_action','ds_tipo_chave_norm','value_band','periodo_dia','score_bin','lgbm_bin','if_bin','ratio_bin','qtd_rec_bin','valor_rec_bin','module_quiet','se_worst_pattern','mbk_available_flag','first_receiver_flag_real','r3u_missing_receiver_history_flag','r3u_receiver_known_flag','r3u_receiver_reputable_flag','r3u_receiver_strong_flag','r3u_relationship_known_flag','r3u_relationship_recurrent_flag','r3u_relationship_strong_flag','r3u_first_receiver_flag','r3u_module_quiet_flag','r3u_se_missing_flag','r3u_ratio_lt_005_flag','r3u_mbk_quality_flag','r3u_receiver_trust_bucket','r3u_relationship_bucket']
SEGS=['_r4b_base_action','ds_tipo_chave_norm','value_band','periodo_dia','score_bin','lgbm_bin','if_bin','ratio_bin','qtd_rec_bin','valor_rec_bin','module_quiet','se_worst_pattern','mbk_available_flag','first_receiver_flag_real']

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--predictions'); p.add_argument('--artifact'); p.add_argument('--output-dir'); p.add_argument('--target-fpr',type=float,default=0.01); p.add_argument('--max-total-fn',type=int,default=5); p.add_argument('--min-incremental-fp',type=int,default=1); p.add_argument('--max-rules',type=int,default=160); p.add_argument('--max-candidates',type=int,default=10000); p.add_argument('--enable-quads',action='store_true'); p.add_argument('--enable-quints',action='store_true'); p.add_argument('--continue-after-target',action='store_true'); return p.parse_args()

def defaults():
    r=Path.cwd(); frozen=r/'resultados/experimentos/EXP-014B-R4A-FROZEN/06_predictions_frozen.csv'; rec=r/'resultados/experimentos/EXP-014B-R4A/09_predictions_recommended.csv'; pred=frozen if frozen.exists() else rec; art=r/'resultados/experimentos/EXP-014B-R4A-FROZEN/05_policy_artifact_frozen.json';
    if not art.exists(): art=r/'resultados/experimentos/EXP-014B-R4A/08_policy_artifact_recommended.json'
    return pred, art if art.exists() else None, r/'resultados/experimentos'/EXPERIMENT

def find_col(df,cands,required=True):
    low={c.lower():c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in low: return low[c.lower()]
    if required: raise KeyError(f'coluna ausente: {cands}')
    return None

def I(s): return pd.to_numeric(s,errors='coerce').fillna(0).astype(int)
def A(s): return s.astype(str).str.strip().str.upper()
def inter(a): return A(a).isin(['CONFIRMAR','BLOQUEAR']).astype(int)
def blk(a): return A(a).eq('BLOQUEAR').astype(int)

def metric(y,p):
    y=I(y); p=I(p); tp=int(((y==1)&(p==1)).sum()); fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum()); tn=int(((y==0)&(p==0)).sum()); pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0; f1=2*pr*rc/(pr+rc) if pr+rc else 0.0; fpr=fp/(fp+tn) if fp+tn else 0.0; return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':round(pr,8),'recall':round(rc,8),'f1':round(f1,8),'fpr':round(fpr,8)}

def target_fp_strict(fpr,n_normals): return max(0,int(math.ceil(fpr*n_normals)-1))

def cand_row(df,label,eligible,cond,cid,typ,desc):
    y=I(df[label]); mask=eligible & cond.fillna(False); n=int(mask.sum())
    if n==0: return None
    fp=int(((y==0)&mask).sum()); tp_loss=int(((y==1)&mask).sum()); ba=A(df['_r4b_base_action']); block_rows=int((ba.eq('BLOQUEAR')&mask).sum()); confirm_rows=int((ba.eq('CONFIRMAR')&mask).sum())
    return {'candidate_id':cid,'rule_type':typ,'description':desc,'n_demoted':n,'fp_removed':fp,'tp_loss':tp_loss,'block_rows_demoted':block_rows,'confirm_rows_demoted':confirm_rows,'precision_demoted':round(tp_loss/n,8) if n else 0.0,'fp_per_tp_loss':round(fp/max(tp_loss,1),8)}

def mask_desc(df,desc):
    pref='Demover INTERVENCAO R4B com '
    if not desc.startswith(pref): raise ValueError('descricao nao suportada: '+desc)
    mask=pd.Series(True,index=df.index)
    for part in desc[len(pref):].split(' AND '):
        if ' == ' in part:
            c,v=part.split(' == ',1); mask &= df[c].fillna('<MISSING>').astype(str).eq(str(v))
        elif ' <= ' in part:
            c,v=part.split(' <= ',1); mask &= pd.to_numeric(df[c],errors='coerce').le(float(v))
        elif ' >= ' in part:
            c,v=part.split(' >= ',1); mask &= pd.to_numeric(df[c],errors='coerce').ge(float(v))
        else: raise ValueError('parte nao suportada: '+part)
    return mask

def build_candidates(df,label,action_col,min_fp,max_candidates,quads,quints):
    action=A(df[action_col]); eligible=action.isin(['CONFIRMAR','BLOQUEAR']); rows=[]; scores=[c for c in SCORES if c in df.columns]; cats=[c for c in CATS if c in df.columns]; useful=[c for c in cats if 1 < df.loc[eligible,c].fillna('<MISSING>').astype(str).nunique() <= 80]
    qs=[.005,.01,.02,.03,.05,.08,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.85,.9,.92,.95,.97,.98,.99,.995]
    for col in scores:
        s=pd.to_numeric(df[col],errors='coerce'); valid=s[eligible&s.notna()]
        if valid.empty: continue
        ths=sorted(set(float(valid.quantile(q)) for q in qs if pd.notna(valid.quantile(q))))
        for th in ths:
            for d,cond,op in [('lo',s.le(th),'<='),('hi',s.ge(th),'>=')]:
                r=cand_row(df,label,eligible,cond,f'score_{d}__{col}__{th:.12g}','score_threshold',f'Demover INTERVENCAO R4B com {col} {op} {th:.12g}')
                if r: rows.append(r)
    specs=[(1,useful[:30],600),(2,useful[:26],600),(3,useful[:20],500)]
    if quads: specs.append((4,useful[:15],350))
    if quints: specs.append((5,useful[:11],220))
    for size, cols_list, topn in specs:
        for cols in itertools.combinations(cols_list,size):
            tmp=df.loc[eligible,list(cols)].fillna('<MISSING>').astype(str)
            for vals,cnt in tmp.value_counts(dropna=False).head(topn).items():
                vals = vals if isinstance(vals,tuple) else (vals,)
                if int(cnt)<min_fp: continue
                cond=pd.Series(True,index=df.index); parts=[]
                for c,v in zip(cols,vals): cond &= df[c].fillna('<MISSING>').astype(str).eq(str(v)); parts.append(f'{c} == {v}')
                r=cand_row(df,label,eligible,cond,'cat%d__'%size+'__'.join(f'{c}={str(v)[:16]}' for c,v in zip(cols,vals)),f'categorical_{size}','Demover INTERVENCAO R4B com '+' AND '.join(parts))
                if r: rows.append(r)
    sq=[.05,.1,.2,.3,.7,.8,.9,.95]
    for sc in scores[:8]:
        s=pd.to_numeric(df[sc],errors='coerce'); valid=s[eligible&s.notna()]
        if valid.empty: continue
        ths=sorted(set(float(valid.quantile(q)) for q in sq if pd.notna(valid.quantile(q))))
        for size, cols_list, topn in [(1,useful[:18],160),(2,useful[:14],180),(3,useful[:10],120)]:
            for cols in itertools.combinations(cols_list,size):
                tmp=df.loc[eligible,list(cols)].fillna('<MISSING>').astype(str)
                for vals,cnt in tmp.value_counts(dropna=False).head(topn).items():
                    vals = vals if isinstance(vals,tuple) else (vals,)
                    if int(cnt)<min_fp: continue
                    base=pd.Series(True,index=df.index); parts=[]
                    for c,v in zip(cols,vals): base &= df[c].fillna('<MISSING>').astype(str).eq(str(v)); parts.append(f'{c} == {v}')
                    for th in ths:
                        for d,scond,op in [('lo',s.le(th),'<='),('hi',s.ge(th),'>=')]:
                            cond=base&scond; desc='Demover INTERVENCAO R4B com '+' AND '.join(parts+[f'{sc} {op} {th:.12g}'])
                            r=cand_row(df,label,eligible,cond,f'scorecat_{d}__{sc}__{th:.12g}__'+'__'.join(str(v)[:14] for v in vals),f'score_cat_{size}',desc)
                            if r: rows.append(r)
    if not rows: return pd.DataFrame()
    cand=pd.DataFrame(rows).drop_duplicates(subset=['description']); cand=cand[cand.fp_removed>=int(min_fp)].copy()
    if cand.empty: return cand
    cand=cand.sort_values(['tp_loss','fp_removed','fp_per_tp_loss','n_demoted'],ascending=[True,False,False,False]).head(max_candidates)
    return cand.reset_index(drop=True)

def apply_demote(df,action_col,demote):
    action=A(df[action_col]).copy(); out=action.copy(); out[action.isin(['CONFIRMAR','BLOQUEAR']) & demote.fillna(False)]='APROVAR'; return out

def select(df,label,action_col,candidates,base_fn,max_fn,target_fp,max_rules,min_fp,cont):
    y=I(df[label]); action=A(df[action_col]); eligible=action.isin(['CONFIRMAR','BLOQUEAR']); cum=pd.Series(False,index=df.index); rows=[]; front=[]; cfp=0; cfn=0; remaining=candidates.copy().reset_index(drop=True)
    for step in range(1,int(max_rules)+1):
        cur=metric(y,inter(apply_demote(df,action_col,cum)))
        if cur['fp']<=target_fp and not cont: break
        best=None; bestmask=None; bestscore=None
        for _,row in remaining.iterrows():
            try: mask=eligible & mask_desc(df,str(row.description)) & (~cum)
            except Exception: continue
            n=int(mask.sum())
            if n==0: continue
            fp=int(((y==0)&mask).sum()); fn=int(((y==1)&mask).sum())
            if fp<int(min_fp): continue
            if base_fn+cfn+fn>int(max_fn): continue
            block_rows=int((A(df[action_col]).eq('BLOQUEAR')&mask).sum())
            score=(1 if fn==0 else 0, fp/max(fn,1), fp, -fn, -block_rows, -n)
            if best is None or score>bestscore:
                best=row.copy(); bestmask=mask; bestscore=score; best['incremental_n_demoted']=n; best['incremental_fp_removed']=fp; best['incremental_tp_loss']=fn; best['incremental_block_rows_demoted']=block_rows; best['incremental_confirm_rows_demoted']=int((A(df[action_col]).eq('CONFIRMAR')&mask).sum())
        if best is None: break
        cum |= bestmask; cfp+=int(best.incremental_fp_removed); cfn+=int(best.incremental_tp_loss); sm=metric(y,inter(apply_demote(df,action_col,cum)))
        best['selection_step']=step; best['cumulative_fp_removed']=cfp; best['cumulative_fn_added']=cfn; best['result_fp']=sm['fp']; best['result_fn']=sm['fn']; best['result_fpr']=sm['fpr']; rows.append(best)
        front.append({'selection_step':step,'selected_candidate_id':best.candidate_id,'selected_description':best.description,'incremental_n_demoted':int(best.incremental_n_demoted),'incremental_fp_removed':int(best.incremental_fp_removed),'incremental_tp_loss':int(best.incremental_tp_loss),'incremental_block_rows_demoted':int(best.incremental_block_rows_demoted),'incremental_confirm_rows_demoted':int(best.incremental_confirm_rows_demoted),'cumulative_n_demoted':int(cum.sum()),'cumulative_fp_removed':int(cfp),'cumulative_fn_added':int(cfn),'result_fp':sm['fp'],'result_fn':sm['fn'],'result_fpr':sm['fpr'],'target_reached':bool(sm['fp']<=target_fp)})
        remaining=remaining[remaining.description!=best.description].reset_index(drop=True)
    return pd.DataFrame(rows),pd.DataFrame(front),cum

def by_action(df,label,act):
    y=I(df[label]); rows=[]
    for a,idx in df.groupby(act,dropna=False).groups.items():
        yy=y.loc[list(idx)]; n=int(len(yy)); f=int((yy==1).sum()); normal=int((yy==0).sum()); rows.append({'action':str(a),'n_rows':n,'n_frauds':f,'n_normals':normal,'precision_within_action':round(f/n,8) if n else 0.0})
    return pd.DataFrame(rows).sort_values('action')

def robustness(df,label,before,after):
    y=I(df[label]); bp=inter(df[before]); ap=inter(df[after]); rows=[]
    for col in SEGS:
        if col not in df.columns: continue
        for val,idx in df.groupby(col,dropna=False).groups.items():
            idx=list(idx); bm=metric(y.loc[idx],bp.loc[idx]); am=metric(y.loc[idx],ap.loc[idx]); rows.append({'segment_col':col,'segment_value':str(val),'n_rows':int(len(idx)),'n_frauds':int((y.loc[idx]==1).sum()),'fp_removed':int(bm['fp']-am['fp']),'tp_loss':int(bm['tp']-am['tp']),'before_fp':bm['fp'],'after_fp':am['fp'],'before_tp':bm['tp'],'after_tp':am['tp'],'after_fn':am['fn']})
    return pd.DataFrame(rows).sort_values(['fp_removed','n_rows'],ascending=[False,False]) if rows else pd.DataFrame()

def wj(p,o): p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    a=parse_args(); dp,da,do=defaults(); pred=Path(a.predictions) if a.predictions else dp; artp=Path(a.artifact) if a.artifact else da; out=Path(a.output_dir) if a.output_dir else do; out.mkdir(parents=True,exist_ok=True)
    if not pred.exists(): raise FileNotFoundError(f'predictions nao encontrado: {pred}')
    df=pd.read_csv(pred,low_memory=False); art=json.loads(Path(artp).read_text(encoding='utf-8')) if artp and Path(artp).exists() else None
    label=find_col(df,LABELS); action_col=find_col(df,ACTIONS); block_col=find_col(df,BLOCKS,False); inter_col=find_col(df,INTERS,False); df['_r4b_base_action']=A(df[action_col])
    y=I(df[label]); action=A(df[action_col]); base_pred=I(df[inter_col]) if inter_col else inter(action); base_block=I(df[block_col]) if block_col else blk(action)
    bm=metric(y,base_pred); bbm=metric(y,base_block); n_normals=int((y==0).sum()); target_fp=target_fp_strict(float(a.target_fpr),n_normals); target_baseline=bool(bm['fp']<=target_fp); block_fp=int(bbm['fp']); confirm_fp=int(bm['fp']-block_fp); confirm_only_impossible=bool(block_fp>target_fp); fp_need=max(0,int(bm['fp']-target_fp))
    cand=build_candidates(df,label,action_col,int(a.min_incremental_fp),int(a.max_candidates),bool(a.enable_quads),bool(a.enable_quints))
    if cand.empty or target_baseline: sel=pd.DataFrame(); front=pd.DataFrame(); demote=pd.Series(False,index=df.index)
    else: sel,front,demote=select(df,label,action_col,cand,int(bm['fn']),int(a.max_total_fn),target_fp,int(a.max_rules),int(a.min_incremental_fp),bool(a.continue_after_target))
    df['exp014b_r4b_demote_intervention_to_approve']=(action.isin(['CONFIRMAR','BLOQUEAR']) & demote.fillna(False)).astype(int); df['r4b_decisao_recommended']=apply_demote(df,action_col,demote); df['exp014b_r4b_intervention_pred']=inter(df['r4b_decisao_recommended']); df['exp014b_r4b_block_pred']=blk(df['r4b_decisao_recommended'])
    fm=metric(y,df['exp014b_r4b_intervention_pred']); fbm=metric(y,df['exp014b_r4b_block_pred']); target_reached=bool(fm['fp']<=target_fp and fm['fpr']<float(a.target_fpr)); fn_ok=bool(fm['fn']<=int(a.max_total_fn)); fp_removed=int(bm['fp']-fm['fp']); fn_added=int(fm['fn']-bm['fn']); block_fp_removed=int(bbm['fp']-fbm['fp']); block_tp_loss=int(bbm['tp']-fbm['tp']); ba=by_action(df.assign(_action=action),label,'_action'); fa=by_action(df,label,'r4b_decisao_recommended'); rob=robustness(df,label,action_col,'r4b_decisao_recommended')
    def av(t,act,col):
        x=t[t.action.eq(act)]; return int(x.iloc[0][col]) if not x.empty else 0
    status='DONE_R4B_FPR_LT1_TARGET_REACHED_WITHIN_FN_BUDGET' if target_reached and fn_ok else 'DONE_R4B_FPR_LT1_TARGET_NOT_REACHED_BUT_IMPROVED' if fp_removed>0 and fn_ok else 'DONE_R4B_NO_SAFE_IMPROVEMENT'
    summary={'experiment':EXPERIMENT,'status':'DONE','objective_status':status,'n_rows':int(len(df)),'n_frauds':int((y==1).sum()),'n_normals':n_normals,'predictions_path':str(pred),'artifact_path':str(artp) if artp else None,'action_col':action_col,'block_col':block_col,'intervention_col':inter_col,'baseline_intervention_metrics':bm,'baseline_block_metrics':bbm,'final_intervention_metrics':fm,'final_block_metrics':fbm,'target_fpr_strict_lt':float(a.target_fpr),'target_fp_strict_lt':target_fp,'target_reached':target_reached,'target_reached_at_baseline':target_baseline,'gap_to_target_fp':max(0,int(fm['fp']-target_fp)),'fp_to_remove_needed_from_baseline':fp_need,'confirm_only_impossible':confirm_only_impossible,'confirm_only_min_possible_fp':block_fp,'baseline_block_fp':block_fp,'baseline_confirm_fp':confirm_fp,'fp_removed_total':fp_removed,'fn_added_total':fn_added,'block_fp_removed':block_fp_removed,'block_tp_loss':block_tp_loss,'max_total_fn':int(a.max_total_fn),'confirm_before_n':av(ba,'CONFIRMAR','n_rows'),'confirm_before_frauds':av(ba,'CONFIRMAR','n_frauds'),'confirm_before_normals':av(ba,'CONFIRMAR','n_normals'),'confirm_after_n':av(fa,'CONFIRMAR','n_rows'),'confirm_after_frauds':av(fa,'CONFIRMAR','n_frauds'),'confirm_after_normals':av(fa,'CONFIRMAR','n_normals'),'block_before_n':av(ba,'BLOQUEAR','n_rows'),'block_after_n':av(fa,'BLOQUEAR','n_rows'),'n_candidates_evaluated':int(len(cand)),'n_selected_rules':int(len(sel)),'min_incremental_fp':int(a.min_incremental_fp),'max_rules':int(a.max_rules),'enable_quads':bool(a.enable_quads),'enable_quints':bool(a.enable_quints),'all_pass':bool(fn_ok),'output_dir':str(out)}
    contract={'predictions_path':str(pred),'artifact_path':str(artp) if artp else None,'label_col':label,'action_col':action_col,'block_col':block_col,'intervention_col':inter_col,'target_fpr_strict_lt':float(a.target_fpr),'target_fp_strict_lt':target_fp,'max_total_fn':int(a.max_total_fn),'min_incremental_fp':int(a.min_incremental_fp),'max_rules':int(a.max_rules),'enable_quads':bool(a.enable_quads),'enable_quints':bool(a.enable_quints),'contract_ok':True,'missing':[]}
    base_obj={'baseline_intervention_metrics':bm,'baseline_block_metrics':bbm,'baseline_by_action':ba.to_dict(orient='records'),'artifact_status':art.get('frozen_validation_status') if isinstance(art,dict) else None}
    rec={'experiment':EXPERIMENT,'input_predictions_path':str(pred),'base_action_col':action_col,'final_action_col':'r4b_decisao_recommended','demote_col':'exp014b_r4b_demote_intervention_to_approve','intervention_pred_col':'exp014b_r4b_intervention_pred','block_pred_col':'exp014b_r4b_block_pred','baseline_intervention_metrics':bm,'baseline_block_metrics':bbm,'final_intervention_metrics':fm,'final_block_metrics':fbm,'target_fpr_strict_lt':float(a.target_fpr),'target_fp_strict_lt':target_fp,'target_reached':target_reached,'gap_to_target_fp':summary['gap_to_target_fp'],'confirm_only_impossible':confirm_only_impossible,'confirm_only_min_possible_fp':block_fp,'fp_removed_total':fp_removed,'fn_added_total':fn_added,'block_fp_removed':block_fp_removed,'block_tp_loss':block_tp_loss,'selected_demotions':sel.to_dict(orient='records') if not sel.empty else [],'notes':['FPR<1% é impossível com CONFIRMAR-only porque BLOQUEAR FP sozinho excede o alvo.','R4B pode demover BLOQUEAR ou CONFIRMAR para APROVAR dentro de FN<=5.','Promoção exige R4B-FROZEN e revisão de negócio porque BLOQUEAR pode mudar.']}
    wj(out/'00_run_summary.json',summary); wj(out/'01_input_contract.json',contract); wj(out/'02_base_metrics.json',base_obj); cand.to_csv(out/'03_candidates.csv',index=False,encoding='utf-8'); front.to_csv(out/'04_selection_frontier.csv',index=False,encoding='utf-8'); sel.to_csv(out/'05_selected_demotions.csv',index=False,encoding='utf-8'); fa.to_csv(out/'06_decision_metrics_by_action.csv',index=False,encoding='utf-8'); rob.to_csv(out/'07_robustness_by_segment.csv',index=False,encoding='utf-8'); wj(out/'08_policy_artifact_recommended.json',rec); df.to_csv(out/'09_predictions_recommended.csv',index=False,encoding='utf-8')
    mdsel=sel.to_markdown(index=False) if not sel.empty else 'Nenhuma regra selecionada.'; mdfront=front.to_markdown(index=False) if not front.empty else 'Nenhuma seleção possível.'
    (out/'10_exp014b_r4b_report.md').write_text(f"""# {EXPERIMENT} - Final FPR < 1%\n\n## Resultado executivo\n- Status: `{status}`\n- Target FPR strict < `{a.target_fpr}`\n- Target FP strict: `{target_fp}`\n- Target reached: `{target_reached}`\n- Gap FP: `{summary['gap_to_target_fp']}`\n- Confirm-only impossible: `{confirm_only_impossible}`\n- FP removidos: `{fp_removed}`\n- FN adicionados: `{fn_added}`\n- Block FP removidos: `{block_fp_removed}`\n- Block TP loss: `{block_tp_loss}`\n\n## Baseline intervenção\n```json\n{json.dumps(bm, ensure_ascii=False, indent=2)}\n```\n\n## Final intervenção\n```json\n{json.dumps(fm, ensure_ascii=False, indent=2)}\n```\n\n## Baseline BLOQUEAR\n```json\n{json.dumps(bbm, ensure_ascii=False, indent=2)}\n```\n\n## Final BLOQUEAR\n```json\n{json.dumps(fbm, ensure_ascii=False, indent=2)}\n```\n\n## Métricas por ação final\n{fa.to_markdown(index=False)}\n\n## Regras selecionadas\n{mdsel}\n\n## Frontier\n{mdfront}\n""",encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
