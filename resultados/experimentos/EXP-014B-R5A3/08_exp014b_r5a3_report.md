# EXP-014B-R5A3 - Dataset and Feature Contract Reconciliation

## Relatório de Conciliação e Linhagem de Dados

Este experimento realiza a auditoria e conciliação estrutural e estatística entre a base legada de 355 fraudes e a nova base de treino expandida com dados do MAF (1465 fraudes), mapeando as features reais, HQLs, scripts de treino e modelos de produção.

### 1. Conclusões Principais da Linhagem
- **Treinamento no Dataset Legado:** A auditoria física e lógica confirmou que os modelos serializados em `backend/artefatos` (`model_lightgbm.joblib` e `model_isolation_forest.joblib`) foram treinados na base antiga (`base_treino_final.csv` de 100.355 linhas e 355 fraudes).
- **Ensemble Campeão Operacional:** O baseline campeão `R4G-FAST` (Recall de 99.86%, FPR < 1% no dataset expandido) foi calibrado e resolvido **exclusivamente** via política pós-modelo e regras em cascata aplicadas sobre a base expandida. O LightGBM e o Isolation Forest não sofreram retreino.
- **Falta de features no Isolation Forest:** Confirmamos que 4 features declaradas pelo IF no treino do baseline estão ausentes dos novos datasets canônicos, devendo o contrato de features ser limpo no próximo ciclo.

### 2. Inventário de Datasets Principais
Abaixo, o inventário resumido das bases de dados analisadas fisicamente:

| Dataset | Finalidade | N_Rows | N_Frauds | Temporal Split / Range | Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `hmo_ml_tb_pix_dataset_v3_features_180d_v1_TRAIN.csv` | Treino Canônico | 78.681 | 1.025 | 2025-11-28 -> 2026-03-27 | **ATIVO** |
| `hmo_ml_tb_pix_dataset_v3_features_180d_v1_VALIDATION.csv` | Validação Canônica | 18.067 | 240 | 2026-03-28 -> 2026-04-26 | **ATIVO** |
| `hmo_ml_tb_pix_dataset_v3_features_180d_v1_HOLDOUT.csv` | Holdout Canônico | 17.096 | 200 | 2026-04-27 -> 2026-05-26 | **ATIVO** |
| `exp014a_expanded_scored_input.csv` | Replay Scored | 113.844 | 1.465 | 2025-11-28 -> 2026-05-26 | **ATIVO** |
| `base_treino_final.csv` (archive) | Legado Treino | 100.355 | 355 | Histórico (MVP) | **ARQUIVADO** |

### 3. Ações Corretivas Recomendadas
1. **Novo Script de Treino LightGBM:** Criar `backend/modelos/train_lgbm_canonical.py` para apontar diretamente para os arquivos `_TRAIN.csv` e `_VALIDATION.csv` do dataset v3.
2. **Novo Script de Treino Isolation Forest:** Criar `backend/modelos/train_isolation_forest_canonical.py` adaptado para rodar na base de treino expandida.
3. **Limpeza de Arquivos Antigos:** Remover os scripts depreciados `train_lgbm_v2.py`, `train_lgbm_v3.py` e `train_isolation_forest_v2.py`.
4. **Remoção de Arquivos de Dados Legados:** Todos os CSVs de dados legados foram movidos para a pasta `dados/archive/` para manter a raiz limpa.

