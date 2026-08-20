# REVALUATION_HARNESS_SPEC — Antifraude PIX

Gerado em: `2026-05-12T18:28:23`

## Versão

- `schema_version`: `1.1`
- Baseline alvo: `post_fase2_c1`

## Objetivo

Definir o harness de reavaliação do baseline em novas janelas de dados extraídas do Big Data.

## Modos de execução

### 1. New Data Dry Run sem labels

Usado quando há transações novas, mas ainda não há confirmação madura de fraude.

Fluxo:

1. Validar schema pelo `DATA_INTAKE_CONTRACT`.
2. Aplicar baseline atual sem alterar modelo.
3. Gerar decision logs no padrão EXP-009A.
4. Rodar drift monitor do EXP-009B.
5. Gerar fila de revisão humana no padrão EXP-009C.
6. Atualizar painel operacional no padrão EXP-009D.
7. Não calcular métricas supervisionadas.

### 2. Reavaliação supervisionada com labels

Usado quando há labels confirmados e uma janela de observação madura.

Fluxo:

1. Validar transações.
2. Validar labels.
3. Fazer join por `transaction_id`.
4. Aplicar baseline atual.
5. Gerar métricas TP, FP, FN, TN, Precision, Recall, F1 e FPR.
6. Gerar decision logs, drift, fila e dashboard.
7. Rodar EXP-009E antes de qualquer promoção.

## Guardas contra leakage

- Labels confirmados após a transação não podem virar feature.
- Tabelas de contestação, MED, chargeback ou análise manual só podem ser usadas como label/target, não como feature preditiva no mesmo instante.
- Features devem ser calculadas com dados disponíveis até o momento da transação.
- Janelas agregadas devem respeitar corte temporal.
- O conjunto de teste futuro deve ficar separado de qualquer retreinamento.

## Critérios para retreinamento futuro

Retreinamento só deve ser considerado se houver:

- volume suficiente de novas fraudes confirmadas;
- aumento relevante de FN no baseline atual;
- drift material nas features críticas;
- evidência de novos padrões não cobertos por V1/C1;
- validação offline mostrando ganho em FN sem aumento inseguro de FP.

## Critérios para nova regra futura

Uma nova regra só deve avançar se:

- recuperar FN novo ou residual;
- adicionar 0 FP ou FP operacionalmente aceitável;
- não perder TP;
- ser configurável/desligável;
- passar no EXP-009E.
