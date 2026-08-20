# Rebuild PIX - Motor antifraude PIX

Motor hibrido para decisao antifraude em transacoes PIX, com classificacao operacional em tres acoes:

- `APROVAR`: liberar a transacao automaticamente.
- `CONFIRMAR`: reter para step-up, biometria, 2FA ou validacao adicional.
- `BLOQUEAR`: impedir a transacao por alto risco.

O estado atual do projeto e o baseline oficial **R5B22**, consolidado sobre as novas bases MAF, contrato congelado R5B16/R5B18 e politica final de restricao APROVAR/CONFIRMAR/BLOQUEAR.

## Estado atual

| Item | Valor |
|---|---|
| Baseline oficial | `R5B22_OFFICIAL_CONSTRAINED_BASELINE` |
| Versao de scoring | `1.5.0-r5b22` |
| Base de validacao | 113.844 transacoes |
| Fraudes confirmadas | 1.465 |
| Normais | 112.379 |
| FPR global | 0,957474% |
| Fraudes em APROVAR | 2 |
| Fraudes em CONFIRMAR | 10 |
| Fraudes em BLOQUEAR | 1.453 |

O objetivo operacional atual e reduzir falsos bloqueios sem abrir excesso de risco:

- manter FPR global abaixo de 1%;
- manter no maximo 5 fraudes em APROVAR;
- manter no maximo 10 fraudes em CONFIRMAR;
- maximizar a precisao de BLOQUEAR.

## Metricas oficiais R5B22

Metricas globais considerando intervencao como `CONFIRMAR` ou `BLOQUEAR`:

| Metrica | Valor |
|---|---:|
| TP | 1.463 |
| FP | 1.076 |
| FN | 2 |
| TN | 111.303 |
| Precision | 57,621111% |
| Recall | 99,863481% |
| F1 | 0,73076923 |
| FPR | 0,957474% |

Metricas especificas de `BLOQUEAR`:

| Metrica | Valor |
|---|---:|
| TP | 1.453 |
| FP | 760 |
| FN fora de BLOQUEAR | 12 |
| TN | 111.619 |
| Precision | 65,657479% |
| Recall | 99,180887% |
| F1 | 0,79010332 |
| FPR | 0,676283% |

Distribuicao por decisao:

| Decisao | Transacoes | Fraudes | Normais |
|---|---:|---:|---:|
| APROVAR | 111.305 | 2 | 111.303 |
| CONFIRMAR | 326 | 10 | 316 |
| BLOQUEAR | 2.213 | 1.453 | 760 |

## Arquitetura operacional

O pipeline atual combina modelo, contrato congelado e politicas operacionais:

1. `PipelineOrquestrador` recebe a transacao, prepara features e executa o motor.
2. `PixDecisionEngine` calcula score e decisao runtime.
3. O contrato R5B16 usa `r4g_fast_frozen_decisao_recommended` como decisao-base congelada.
4. A politica R5B14 aplica restricoes de baixo falso negativo.
5. A politica R5B22 aplica democoes controladas para reduzir normais em `BLOQUEAR`, respeitando os tetos de fraude em `APROVAR` e `CONFIRMAR`.

Configuracoes oficiais em `backend/artefatos/scoring_config.json`:

```json
{
  "versao": "1.5.0-r5b22",
  "r5b14_operational_zero_fn_enabled": true,
  "r5b16_frozen_contract_enabled": true,
  "r5b22_official_baseline_enabled": true,
  "official_baseline_policy": "R5B22_OFFICIAL_CONSTRAINED_BASELINE"
}
```

## Artefatos oficiais

```text
backend/artefatos/r5b22_official_baseline_policy.json
backend/artefatos/r5b22_official_baseline_summary.json
backend/artefatos/model_lgbm_distilled_r5b22_intervention.joblib
backend/artefatos/model_lgbm_distilled_r5b22_block.joblib
backend/artefatos/model_lgbm_distilled_r5b22_metadata.json
```

O LGBM aluno e uma distilacao do contrato R5B16/R5B18. Ele usa sinais do professor, incluindo `r4g_fast_frozen_decisao_recommended`, `r5b14_rule_applied` e `r5b14_layer_applied`; portanto, nao deve ser descrito como LGBM puro treinado apenas com features brutas.

## Estrutura principal

```text
backend/
  artefatos/                 Modelos, politicas e configs oficiais
  core/
    decision_engine.py       Motor de decisao
    pipeline_orquestrador.py Orquestrador E2E runtime
    r5b14_operational_policy.py
    social_engineering.py
    behavioral_analytics.py
  scripts/
    simular_pipeline_e2e_v2.py
dados/
  hmo_ml_tb_pix_dataset_v3_features_180d_v1.csv
docs/
  JOURNAL_4.md
  PIX_FRAUD_MODEL_FINAL_IMPROVEMENT_PLAN.md
  apresentacao_mvp_v2.md
resultados/
  experimentos/
tests/
```

## Como executar

Criar ambiente:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Validacao curta:

```bash
python -m py_compile backend/core/decision_engine.py backend/core/pipeline_orquestrador.py backend/scripts/simular_pipeline_e2e_v2.py
```

Simulacao E2E global com o baseline oficial:

```bash
python backend/scripts/simular_pipeline_e2e_v2.py --full --workers 4
```

Simulacao amostral:

```bash
python backend/scripts/simular_pipeline_e2e_v2.py --sample 2000
```

API local:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints principais:

| Endpoint | Metodo | Uso |
|---|---|---|
| `/api/v1/analyze` | POST | Inferencia de uma transacao |
| `/api/v1/batch` | POST | Inferencia em lote |
| `/api/v1/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## Documentacao relevante

| Documento | Conteudo |
|---|---|
| `docs/JOURNAL_4.md` | Historico dos experimentos R4/R5 e consolidacao R5B22 |
| `docs/PIX_FRAUD_MODEL_FINAL_IMPROVEMENT_PLAN.md` | Plano de melhoria e experimentos do baseline final |
| `docs/apresentacao_mvp_v2.md` | Apresentacao executiva atualizada com metricas e regras ativas |

## Roadmap tecnico

- Executar homologacao E2E full sempre que a politica oficial mudar.
- Monitorar drift das novas bases MAF e degradacao dos tetos de fraude por decisao.
- Evoluir o LGBM aluno para reduzir dependencia dos sinais do professor sem perder os gates operacionais.
- Integrar step-up real para a faixa `CONFIRMAR`.
- Persistir historico e perfis em feature store/cache operacional para inferencia em tempo real.

