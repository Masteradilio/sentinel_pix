# EXP-002

- Vencedor canônico: `V1` (`lgbm_guard_enabled=true`, `lgbm_guard_threshold=0.30`)
- Sample principal `seed=42`: `TP=345`, `FP=15`, `FN=10`, `Precision=95.83%`, `Recall=97.18%`, `F1=0.9650`
- Baseline pós-`EXP-001`: `TP=346`, `FP=18`, `FN=9`, `Precision=95.05%`, `Recall=97.46%`, `F1=0.9624`
- Delta do vencedor: `-3 FP`, `-1 TP`, `+1 FN`, `+0.0026 F1`
- Validação cruzada `seed=123`: `TP=345`, `FP=12`, `FN=10`, `Precision=96.64%`, `Recall=97.18%`, `F1=0.9691`

Artefatos completos:
- `resultados/experimentos/EXP-002/01_tabela_comparativa.csv`
- `resultados/experimentos/EXP-002/02_analise_supressoes.json`
- `resultados/experimentos/EXP-002/03_analise_fp_fn.json`
- `resultados/experimentos/EXP-002/04_validacao_cruzada.json`
- `resultados/experimentos/EXP-002/05_conclusao_executiva.md`
