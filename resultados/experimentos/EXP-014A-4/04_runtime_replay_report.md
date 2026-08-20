# EXP-014A-4 — Official Runtime Replay Builder

## Resultado
- Status: `DONE_CONTRACT_OK_BUILT`
- Built: `True`
- Target: `C:\Users\u857755\OneDrive - BRB - Banco de Brasilia SA\Documentos\Projetos\squad_IA\PIX\rebuild_pix\dados\exp014a_expanded_scored_input.csv`

## Contrato
- Contract OK: `True`
- Missing: `[]`
- Final pred cols: `['exp014a_frozen_pred', 'exp013k_residual_fp_pred']`
- Has score_final: `True`
- Has if_percentile: `True`

## Preview de métricas runtime
- tp: `59`
- fp: `31`
- fn: `1406`
- tn: `112348`
- precision: `0.6555555555555556`
- recall: `0.040273037542662114`
- fpr: `0.00027585224997552925`

## Próximo passo
Rodar:
```powershell
python scripts\exp_014a_expanded_frozen_validation.py --allow-final-direct
```