# EXP-014B-R5B10-CONSOLIDATED-SEVERITY-POLICY — Política consolidada candidata

## Resultado executivo
- Status: `PASS_R5B10_CONSOLIDATED_CANDIDATE_WRITTEN`
- Artefato candidato: `backend\artefatos_candidatos\exp014b_r5b10_severity_policy\severity_policy_candidate.json`
- Camadas: `3`
- Regras totais: `65`
- Normais removidos de BLOQUEAR desde R5B2: `11865`
- Normais restantes em BLOQUEAR: `2355`
- Fraudes demovidas para CONFIRMAR: `0`
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

## Decisão técnica
O artefato consolidado é candidato e não está ativo em produção. Ele preserva a
proveniência das camadas R5B4, R5B5 e R5B8/R5B9 e serve como contrato de entrada
para o próximo passo: integração configurável ao orquestrador e replay E2E.
