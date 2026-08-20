# EXP-010C — Build MAF Curated Fraud Tables

Gerado em: `2026-05-12T21:16:47`

## Resultado

- Label table: `hmo_ml.tb_pix_fraude_labels_maf_curated_v1`
- Hydrated fraud table: `hmo_ml.tb_pix_fraudes_maf_hidratadas_v1`
- ENABLE_MOBILE: `False`
- Labels curados: `134599`
- POSITIVE_FOR_CURRENT_MODEL: `15564`
- Hidratados finais: `13558`

## Decisão esperada

Se a tabela hidratada tiver volume coerente e cobertura suficiente de cliente/relacionamento, o próximo passo é copiar/exportar `dados_pix_fraudes_maf_hidratadas_v1.csv` para o ambiente local do projeto e rodar o `preprocessing.py` junto com a base de normais.

## Observações

- Textos pós-evento permanecem fora das features.
- Casos de triangulação permanecem segregados.
- Casos BRB_CREDITADO_RECEBEDOR permanecem segregados.
- Casos com conflito de label/direção não entram na tabela hidratada final.
