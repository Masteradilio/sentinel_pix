# Perfeito, Adilio — vamos começar pelo **MVP de features**

A melhor forma de montar esse MVP é separar as features em camadas:

1. **Features brutas disponíveis**
2. **Features derivadas de comportamento**
3. **Features derivadas de risco / regras**
4. **Features de device/app**
5. **Features finais priorizadas para treino do MVP**

Vou focar no que **existe nas suas bases** e no que pode ser **derivado com segurança** a partir delas.

---

# 1. Princípios para priorização do MVP

Para o MVP, as features precisam ser:

- **fortes para fraude**
- **fáceis de calcular**
- **disponíveis em produção em tempo real**
- **explicáveis para analista**
- **com baixo risco de leakage**

Então, inicialmente, eu priorizaria features que usam:

- valor da transação
- histórico do cliente
- ritmo/frequência
- perfil do cliente
- device/mobile
- topaz
- regras de negócio

---

# 2. Fontes disponíveis

## 2.1 Base PIX normal / fraude
Campos relevantes identificados:
- `cd_pix`
- `dt_pix`
- `cd_cpf_pagador`
- `vl_pix`
- `qt_total_pix_trimestre`
- `vl_mediana_pix_trimestre`
- `vl_desvio_padrao_pix_trimestre`
- `qt_intervalo_transacao_minuto`
- `qt_intervalo_mediana_trimestre`
- `qt_intervalo_desvio_padrao_trimestre`
- `qt_pix_dia_maximo_trimestre`
- `latencia_rede_ms`
- `vl_latencia_rede_media_trimestre`
- `tempo_interacao_ms`
- `vl_tempo_interacao_medio_trimestre`
- `qt_aparelhos_distintos_trimestre`
- `nr_idade`
- `qt_tempo_relacionamento_mes`
- label de fraude na base de fraude (`tp_fraude`)

## 2.2 Base mobile
Campos relevantes:
- `end_to_end_id`
- `data_hora_inicio`
- `nr_conta`
- `valor_transacao`
- `cd_tipo_transacao`
- `cd_retorno`
- `device_name`
- `app_version`
- `ip_address`
- `latencia_rede_ms`
- `tempo_interacao_ms`
- `tempo_processamento_host_ms`
- `metodo_autenticacao`
- `session_id`
- `topaz_risk_score`
- `topaz_transacao_rejeitada`
- `topaz_transacao_habilitada`
- `is_agendamento_recorrente`

---

# 3. Estratégia de feature set do MVP

Eu sugiro montar o MVP com **5 grupos de features**:

- **A. Identificação e label**
- **B. Perfil histórico da transação**
- **C. Desvio comportamental**
- **D. Risco mobile/device**
- **E. Regras / score especialista**

---

# 4. Lista priorizada de features do MVP

---

## A. Features de identificação e label
Essas não entram no modelo como preditoras, mas são essenciais para pipeline.

### Chaves
- `transaction_id`  
  Derivada de:
  - `cd_pix` nas bases PIX
  - `end_to_end_id` na base mobile

- `customer_id`
  - `cd_cpf_pagador`

- `event_datetime`
  - `dt_pix`

- `is_fraud`
  - `tp_fraude` na base de fraude
  - `0` na base normal

---

## B. Features históricas diretas do cliente
Essas devem estar entre as **mais importantes do MVP**.

### B1. Valor e distribuição histórica
1. `vl_pix`
2. `qt_total_pix_trimestre`
3. `vl_mediana_pix_trimestre`
4. `vl_desvio_padrao_pix_trimestre`
5. `qt_pix_dia_maximo_trimestre`

### B2. Intervalos / frequência
6. `qt_intervalo_transacao_minuto`
7. `qt_intervalo_mediana_trimestre`
8. `qt_intervalo_desvio_padrao_trimestre`

### B3. Perfil do cliente
9. `nr_idade`
10. `qt_tempo_relacionamento_mes`
11. `qt_aparelhos_distintos_trimestre`

### B4. Uso do app/rede
12. `latencia_rede_ms`
13. `vl_latencia_rede_media_trimestre`
14. `tempo_interacao_ms`
15. `vl_tempo_interacao_medio_trimestre`

---

## C. Features derivadas de desvio comportamental
Essas são as **mais importantes para behavioral analytics**.

## C1. Desvio de valor
16. `ratio_valor_mediana`
```text
vl_pix / vl_mediana_pix_trimestre
```

17. `diff_valor_mediana`
```text
vl_pix - vl_mediana_pix_trimestre
```

18. `ratio_valor_desvio_padrao`
```text
vl_pix / vl_desvio_padrao_pix_trimestre
```

19. `zscore_valor_aprox`
```text
(vl_pix - vl_mediana_pix_trimestre) / vl_desvio_padrao_pix_trimestre
```

20. `log_vl_pix`
```text
log1p(vl_pix)
```

---

## C2. Desvio de frequência / velocidade
21. `ratio_intervalo_vs_mediana`
```text
qt_intervalo_transacao_minuto / qt_intervalo_mediana_trimestre
```

22. `diff_intervalo_vs_mediana`
```text
qt_intervalo_transacao_minuto - qt_intervalo_mediana_trimestre
```

23. `zscore_intervalo_aprox`
```text
(qt_intervalo_transacao_minuto - qt_intervalo_mediana_trimestre) / qt_intervalo_desvio_padrao_trimestre
```

24. `burst_30m_flag`
- 1 se o comportamento indicar burst/alta concentração temporal
- heurística inicial:
  - `qt_intervalo_transacao_minuto <= 30` ou
  - regra derivada de velocity

25. `pix_freq_high_flag`
- flag para frequência acima do padrão

---

## C3. Desvio de rede e interação
26. `ratio_latencia_cliente`
```text
latencia_rede_ms / vl_latencia_rede_media_trimestre
```

27. `diff_latencia_cliente`
```text
latencia_rede_ms - vl_latencia_rede_media_trimestre
```

28. `ratio_tempo_interacao_cliente`
```text
tempo_interacao_ms / vl_tempo_interacao_medio_trimestre
```

29. `diff_tempo_interacao_cliente`
```text
tempo_interacao_ms - vl_tempo_interacao_medio_trimestre
```

30. `tempo_interacao_baixo_flag`
- 1 se interação muito abaixo do histórico

31. `tempo_interacao_alto_flag`
- 1 se interação muito acima do histórico

---

## D. Features temporais
Essas são simples e muito úteis.

32. `hour`
- hora da transação

33. `day_of_week`
- dia da semana

34. `is_night`
- 1 se horário noturno  
Sugestão: 22h–6h

35. `is_weekend`
- sábado/domingo

36. `is_business_hours`
- 1 se entre 8h e 18h

37. `period_of_day`
- madrugada / manhã / tarde / noite

---

## E. Features mobile/device
Essas são críticas para account takeover e risco operacional.

### E1. Topaz
38. `topaz_risk_score`
39. `topaz_score_filled`
- mesma feature com imputação para missing
40. `topaz_transacao_rejeitada`
41. `topaz_transacao_habilitada`

### E2. Sessão / autenticação
42. `metodo_autenticacao`
43. `session_id`  
- não usar bruto no modelo, mas usar para derivar flags/frequência se depois quiser

44. `tempo_processamento_host_ms`

### E3. Device / app
45. `device_name`
46. `app_version`
47. `ip_address`
48. `cd_retorno`
49. `is_agendamento_recorrente`

---

## F. Features derivadas de device/app para o MVP
Essas são muito úteis e fáceis.

50. `device_missing_flag`
- 1 se `device_name` ausente

51. `ip_missing_flag`
- 1 se `ip_address` ausente

52. `app_version_missing_flag`

53. `auth_method_missing_flag`

54. `topaz_missing_flag`

55. `host_time_missing_flag`

56. `device_name_normalized`
- texto padronizado

57. `app_version_major`
- extrair major version, ex.: 7 do 7.30.0

58. `app_version_minor`
- extrair minor version, ex.: 30

59. `latencia_host_ratio`
```text
latencia_rede_ms / tempo_processamento_host_ms
```
se ambos existirem

60. `processamento_host_alto_flag`

---

## G. Features de regras de negócio
Agora vamos transformar suas regras em variáveis.

### G1. PIX em < 30 min
61. `rule_pix_30m_score`
- 0 = nenhum caso
- 1 = 1 transação em <30 min
- 2 = 2+ transações em <30 min

**No MVP**, como não temos todas as janelas prontas por CPF na mesma base histórica detalhada de eventos antigos além das transações listadas, vamos derivar isso pela ordenação temporal por cliente.

---

### G2. Razão PIX / limite
Você citou limite, mas **essa informação não aparece nas bases anexadas**.

Então:
- **essa feature não está disponível no momento**
- podemos deixar prevista no pipeline:
  62. `rule_ratio_pix_limite_score`
- por ora ficará nula/ausente até termos base de limite transacional

---

### G3. Idade
63. `rule_age_score`
- 60–65 = 1
- 66–75 = 2
- 76+ = 3

64. `is_elderly_flag`
- 1 se >= 60

---

### G4. Relacionamento
65. `rule_relationship_score`
- 61–90 dias = 1
- 31–60 dias = 2
- 0–30 dias = 3

66. `is_new_customer_flag`
- 1 se relacionamento <= 30 dias

---

### G5. Conta laranja
Não existe um indicador explícito na base fornecida.

Então, no MVP:
67. `rule_mule_account_score`
- **não disponível ainda**
- deixar placeholder

---

### G6. Chave aleatória
Não vejo um campo explícito de tipo de chave nas bases fornecidas.

Então:
68. `rule_random_key_score`
- **informação não disponível nas amostras fornecidas**

---

### G7. Horário noturno
69. `rule_night_score`
- 3 se noturno, senão 0

---

### G8. Velocity
70. `rule_velocity_score`
- de 0 a 4 com base em burst temporal e repetição

Exemplo inicial:
- 0 = normal
- 2 = 1 transação muito próxima do histórico
- 3 = múltiplas curtas
- 4 = burst intenso

---

### G9. Topaz
71. `rule_topaz_score`
Sugestão MVP:
- missing/0/1 = 0
- 2 = 2
- 3 = 3
- 4 = 4
- 5+ = 5

---

### G10. Atenuante
72. `autorizacao_previa_flag`
- **informação não disponível nas bases anexadas**
- deixar placeholder

73. `rule_pre_authorization_discount`
- por ora 0

---

## H. Score agregado de regras
74. `rule_score_raw`
- soma das regras disponíveis

75. `rule_score_normalized`
- normalização 0–1

---

# 5. Lista final de features realmente priorizadas para o primeiro treino

Se eu tivesse que escolher o **núcleo mínimo viável**, começaria com estas:

## Núcleo 1 — valor/comportamento
1. `vl_pix`
2. `log_vl_pix`
3. `vl_mediana_pix_trimestre`
4. `vl_desvio_padrao_pix_trimestre`
5. `ratio_valor_mediana`
6. `diff_valor_mediana`
7. `zscore_valor_aprox`

## Núcleo 2 — frequência/velocity
8. `qt_total_pix_trimestre`
9. `qt_intervalo_transacao_minuto`
10. `qt_intervalo_mediana_trimestre`
11. `qt_intervalo_desvio_padrao_trimestre`
12. `ratio_intervalo_vs_mediana`
13. `zscore_intervalo_aprox`
14. `qt_pix_dia_maximo_trimestre`

## Núcleo 3 — perfil do cliente
15. `nr_idade`
16. `qt_tempo_relacionamento_mes`
17. `qt_aparelhos_distintos_trimestre`
18. `rule_age_score`
19. `rule_relationship_score`

## Núcleo 4 — comportamento digital
20. `latencia_rede_ms`
21. `vl_latencia_rede_media_trimestre`
22. `ratio_latencia_cliente`
23. `tempo_interacao_ms`
24. `vl_tempo_interacao_medio_trimestre`
25. `ratio_tempo_interacao_cliente`

## Núcleo 5 — mobile risk
26. `topaz_risk_score`
27. `rule_topaz_score`
28. `device_missing_flag`
29. `ip_missing_flag`
30. `app_version_major`

## Núcleo 6 — tempo
31. `hour`
32. `day_of_week`
33. `is_night`
34. `is_weekend`

## Núcleo 7 — regras agregadas
35. `rule_pix_30m_score`
36. `rule_velocity_score`
37. `rule_night_score`
38. `rule_score_raw`
39. `rule_score_normalized`

---

# 6. Features que eu **não colocaria ainda** no primeiro treino
Para manter o MVP limpo:

- `session_id` bruto
- `ip_address` bruto
- `device_name` bruto sem tratamento
- `cd_pix` bruto
- `customer_id` bruto
- placeholders sem dado real como:
  - limite transacional
  - conta laranja
  - chave aleatória
  - autorização prévia

Essas podem entrar depois.

---

# 7. O que está indisponível nas bases anexadas
Pelo conteúdo fornecido, **não encontrei informação explícita** para:

- limite da transação / limite do cliente
- identificação de conta laranja
- tipo da chave PIX (aleatória/celular/cpf/email)
- autorização prévia do cliente

Então essas regras devem ficar como:
- **placeholders no código**
- ou colunas nulas para futura integração

---

# 8. Estrutura da base final do MVP

A base consolidada para treino deve ter:

## Identificação
- `transaction_id`
- `customer_id`
- `event_datetime`
- `is_fraud`

## Features do modelo
- todas as features priorizadas acima

## Metadados opcionais
- `source_dataset` (`normal` ou `fraud`)
- `join_status_mobile`

---

# 9. Ordem correta de implementação

Agora eu sugiro esta sequência:

## Etapa 1
Criar um script que:

- leia os 3 CSVs
- padronize nomes de colunas
- marque `is_fraud`
- una normal + fraude
- faça join com mobile por `cd_pix` ↔ `end_to_end_id`
- gere as features derivadas
- salve um CSV final pronto para modelagem

## Etapa 2
Treinar o primeiro modelo supervisionado

## Etapa 3
Criar score de regras e score de anomalia

---

# 10. Minha recomendação prática para o script de features
No script, vamos implementar:

- limpeza e padronização
- parsing de datas
- casting numérico
- criação de flags de missing
- derivação de razões e z-scores
- derivação temporal
- regras
- saída final em CSV

---

# 11. Resumo do que vamos codificar no próximo passo

No próximo script, eu vou montar para você:

## Pipeline de engenharia de features
1. carregar `dados_pix_normais.csv`
2. carregar `dados_fraudes_pix.csv`
3. carregar `dados_features_mobile.csv`
4. padronizar colunas
5. criar `is_fraud`
6. concatenar normal + fraude
7. juntar com mobile
8. criar todas as features derivadas do MVP
9. gerar:
   - `base_mvp_features.csv`

---

