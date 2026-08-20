# Próximo script recomendado

## EXP-010C — Build MAF Curated Fraud Tables

## Objetivo

Criar as tabelas definitivas derivadas da fonte textual MAF.

## Tabelas sugeridas

```text
hmo_ml.tb_pix_fraude_labels_maf_curated_v1
hmo_ml.tb_pix_fraudes_maf_hidratadas_v1
```

## Regras sugeridas

- A tabela de labels deve conter transaction_id, label_status, label_confidence, fraud_type e bank_direction.
- A tabela final de fraudes deve conter apenas casos hidratados com features compatíveis com o pipeline atual.
- Textos pós-evento devem ficar apenas para auditoria, não como feature.
- Casos BRB_CREDITADO_RECEBEDOR devem ficar segregados até decisão de escopo.
- Casos de triangulação devem ficar segregados ou tipados explicitamente.
- CSVs finais para treino devem passar pelo `preprocessing.py`.
