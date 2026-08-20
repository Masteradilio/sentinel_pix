# EXP-010B — MAF Fraud Label Acquisition Audit

Gerado em: `2026-05-12T20:30:53`

## Objetivo

Auditar a nova tabela textual de fraudes antes de criar tabela intermediária ou tabela final de fraudes hidratadas.

## Resultado executivo

- Linhas brutas auditadas: `135262`
- Transações deduplicadas por E2E ID: `134599`
- Candidatos fortes para o modelo atual (`BRB_DEBITADO_PAGADOR` + `CONFIRMED_FRAUD_CANDIDATE`): `35913`
- Casos para revisão: `19008`
- Conflitos de label: `882`
- Match PIX: `28531`
- Match mobile: `None`
- Match cliente: `NOT_RUN`

## Decisão preliminar

A nova fonte deve ser tratada como fonte de labels pós-evento. Os textos de relato e conclusão não devem entrar como features do modelo.

Casos recomendados para próxima etapa:

```text
bank_direction = BRB_DEBITADO_PAGADOR
label_status = CONFIRMED_FRAUD_CANDIDATE
transaction_id_valid_flag = true
triangulation_flag = false, salvo se decidirmos modelar triangulação separadamente
```

## Próximo passo

Após avaliar os artefatos, gerar o script definitivo para:

1. criar tabela intermediária curada de labels MAF;
2. criar tabela final de fraudes hidratadas via join com PIX/mobile/cliente;
3. produzir CSV compatível com o `preprocessing.py` e com o contrato v1.1.
