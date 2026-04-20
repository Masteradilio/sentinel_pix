`EXP-002` testa um guard rail de veto: quando `lgbm_raw` fica abaixo de um limiar, vetos secundários são suprimidos.

Arquivos:
- `config_variantes.json`: baseline + variantes `0.20`, `0.30`, `0.40`
- `run_exp_002.py`: runner E2E com o `PipelineOrquestrador` real

Saídas em `resultados/experimentos/EXP-002/`:
- `01_tabela_comparativa.csv`
- `02_analise_supressoes.json`
- `03_analise_fp_fn.json`
- `04_validacao_cruzada.json`
- `05_conclusao_executiva.md`
