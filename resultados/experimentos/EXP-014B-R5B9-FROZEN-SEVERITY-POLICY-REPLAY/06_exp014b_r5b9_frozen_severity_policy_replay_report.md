# EXP-014B-R5B9-FROZEN-SEVERITY-POLICY-REPLAY — Replay congelado da política de severidade

## Resultado executivo
- Status: `PASS_R5B9_FROZEN_REPLAY_MATCHED_R5B8`
- Linhas avaliadas: `113844`
- Divergências de decisão vs R5B8: `0`
- Divergências de aplicação vs R5B8: `0`
- Normais movidos de BLOQUEAR para CONFIRMAR: `469`
- Fraudes movidas de BLOQUEAR para CONFIRMAR: `0`
- Normais restantes em BLOQUEAR: `2355`
- Fraudes restantes em BLOQUEAR: `279`
- Fraudes restantes em APROVAR: `682`

## Métricas finais de BLOQUEAR
```json
{
  "tp": 279,
  "fp": 2355,
  "fn": 1186,
  "tn": 110024,
  "precision": 0.10592255,
  "recall": 0.19044369,
  "f1": 0.13613076,
  "fpr": 0.02095587
}
```

## Contagem por regra
```json
{
  "R5B8_01_RELATIONSHIP_AGE_GTE_35D": 357,
  "R5B8_02_LOW_RECEIVER_REP_LOW_PAYER_COUNT": 58,
  "R5B8_03_LOW_RECEIVER_VALUE_LOW_PAYER_VALUE": 54
}
```

## Decisão técnica
Este replay valida que a implementação explícita em `backend.core.severity_policy`
reproduz o artefato R5B8. Ele ainda não ativa a política automaticamente no
runtime produtivo; serve como gate antes de conectar a política ao orquestrador
ou a um arquivo de configuração versionado.
