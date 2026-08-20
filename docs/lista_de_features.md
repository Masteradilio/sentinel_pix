# Lista de Features - Baseline R5B22

O modelo preditivo oficial (Baseline R5B22) baseia-se em um conjunto destilado de **78 features**, construídas a partir das variáveis transacionais em tempo real (runtime) combinadas com históricos gerados no Feature Store (HBase) e regras contratuais consolidadas pela política "professora".

A antiga estrutura com 52 features focada nos experimentos iniciais do MVP foi depreciada em favor de um modelo mais robusto e aderente à produção, que incorpora não apenas dados brutos, mas os sinais emitidos por pipelines anteriores (contrato congelado) e modulações sistêmicas (regras R5B14).

A seguir, a catalogação oficial destas features de acordo com o metadado `model_lgbm_distilled_r5b22_metadata.json`.

---

## A. Features de Identificação e Controle (Não-preditoras)

Estas colunas são imprescindíveis para a orquestração do pipeline, roteamento e auditoria, porém, **não são utilizadas como entrada de predição no modelo** de Machine Learning.

*   `transaction_id`: Identificador unívoco da transação atual.
*   `customer_id` / `cd_cpf_pagador`: Chave unívoca do pagador na instituição.
*   `event_datetime` / `dt_pix`: Carimbo de tempo do evento PIX (ISO 8601).
*   `is_fraud`: Rótulo de fraude, usado exclusivamente para avaliação/treino, inexistente em runtime real.
*   **Splits / Auditoria:** `dataset_role`, `temporal_split`, etc. (utilizados apenas offline).

---

## B. Transação Atual (Runtime)

Dados contextuais extraídos diretamente do payload transacional no exato momento da avaliação.

1.  `vl_pix`: Valor original da transferência.
2.  `ds_tipo_chave_norm` (Categórica): Tipo da chave formatada. Ex.: *DOCUMENTO_TELEFONE*, *CHAVE_ALEATORIA*, etc.
3.  `hour`: Hora isolada do carimbo de tempo.
4.  `periodo_dia` (Categórica): Agrupamento do período (madrugada, manha, tarde, noite).
5.  `value_band` (Categórica): Discretização / bucketização do `vl_pix`.
6.  `autcodret`: Código de autorização/retorno transacional (caso disponível).

---

## C. Dados de Telemetria, Sessão e Mobile (Enriquecimento Online/Host)

Variáveis relativas ao dispositivo, segurança do app móvel e performance técnica na sessão do usuário.

7.  `latencia_rede_ms`: Tempo de latência originado do client (aparelho móvel/rede).
8.  `tempo_processamento_host_ms`: Tempo total despendido nos servidores internos da instituição/BACEN.
9.  `topaz_risk_score`: Score de risco devolvido por parceiro antifraude especializado em endpoint.
10. `mbk_completeness_score`: Avaliação do grau de completude das informações enviadas pelo Mobile Banking.
11. `mbk_available_flag`: Flag se o Mobile Banking / telemetria estava, de fato, disponível e atrelado à transação.

---

## D. Histórico do Pagador (Feature Store / HBase)

Atributos pré-computados (ou atualizados por janelas curtas) que trazem o comportamento passado do detentor da conta de origem em várias janelas temporais.

12. `qtd_pix_pagador_7d`: Frequência transacional em 7 dias.
13. `qtd_pix_pagador_30d`: Frequência transacional em 30 dias.
14. `qtd_pix_pagador_90d`: Frequência transacional em 90 dias.
15. `qtd_pix_pagador_180d`: Frequência transacional histórica em 180 dias.
16. `valor_total_pagador_7d`: Volume financeiro total no período.
17. `valor_total_pagador_30d`: Volume financeiro total no período.
18. `valor_total_pagador_90d`: Volume financeiro total no período.
19. `valor_total_pagador_180d`: Volume financeiro total no período.
20. `max_qtd_pix_dia_pagador_7d`: Máximo de PIXs efetuados em um único dia (dentro dos últimos 7d).
21. `max_qtd_pix_dia_pagador_30d`: Máximo de PIXs efetuados em um único dia (dentro dos últimos 30d).
22. `valor_maximo_pix_pagador_180d`: Pico transacional do cliente ao longo do semestre.
23. `soma_recebedores_distintos_dia_180d`: Métricas de "fan-out" (amplitude de envio para novas contas).

---

## E. Relação Pagador-Recebedor (Feature Store / HBase)

Avaliação histórica exclusiva para a aresta (tupla: `customer_id` -> `counterparty_id`), útil para detectar contas laranjas recém-adicionadas vs contas de relacionamento estreito e seguro.

24. `qtd_pix_mesmo_recebedor_7d`
25. `qtd_pix_mesmo_recebedor_30d`
26. `qtd_pix_mesmo_recebedor_90d`
27. `qtd_pix_mesmo_recebedor_180d`
28. `valor_total_para_recebedor_30d`
29. `valor_total_para_recebedor_90d`
30. `valor_total_para_recebedor_180d`
31. `valor_medio_para_recebedor_180d`
32. `primeiro_envio_para_recebedor_180d`: Data do primeiro relacionamento mútuo.
33. `dias_desde_primeiro_envio_recebedor`: Idade do vínculo de transferência.
34. `dias_desde_ultima_transacao_recebedor`: Frequência de interação/recorrência.
35. `is_recebedor_recorrente_180d`: Rótulo de relação saudável vs recebedor efêmero.
36. `ratio_valor_pix_vs_max_recebedor_180d`: Discrepância perante o volume médio pago a esta pessoa em específico.

---

## F. Histórico do Recebedor (Feature Store / HBase)

Comportamento isolado da contraparte (que costuma concentrar o lado da engenharia social ou das contas alugadas).

37. `qtd_pix_recebidos_30d`
38. `qtd_pix_recebidos_90d`
39. `qtd_pix_recebidos_180d`
40. `valor_total_recebido_30d`
41. `valor_total_recebido_90d`
42. `valor_total_recebido_180d`
43. `soma_pagadores_distintos_dia_recebedor_180d`: Avaliação de "fan-in" (diversas contas enviando para a mesma).
44. `max_qtd_pix_recebidos_dia_180d`: Pico direcional (Mule Accounts pattern).
45. `first_receiver_flag_real`: O CPF/CNPJ de origem e o CPF/CNPJ de destino interagem pela primeiríssima vez.

---

## G. Ratios de Anomalias Transacionais e Comportamentais (Runtime)

Variáveis calculadas em memória com base em dados de evento dividido pelos históricos extraídos.

46. `burst_daily_7d_flag`: Pico/rajada atípica em contraste ao comportamento dos últimos sete dias.
47. `ratio_valor_media_pagador_90d`: Diferencial do `vl_pix` contra a média padrão trimestral do cliente.
48. `ratio_valor_maximo_pagador_180d`: Distanciamento para o pico semestral do usuário.

---

## H. Scores de Componentes Consultivos

Valores gerados no runtime do motor pela suite de algoritmos especialistas, cujo output abastece a engrenagem (Cascade, regras, e SHAP) final do R5B22.

49. `module_quiet` (Categórica): Indicação qualitativa sobre o silêncio suspeito de componentes/dados.
50. `se_worst_pattern` (Categórica): Extração categórica do módulo de Engenharia Social (`SocialEngineeringDetector`), informando a principal tática nociva encontrada (ex.: *BURST_ESVAZIAMENTO_CONTA*).
51. `lgbm_raw`: O score base puro em si (do LightGBM legado/cascade v3).
52. `lgbm_r4_score`: Score modificado em camadas iterativas R4 do pipeline (versão calibrada para falso positivo).
53. `score_final`: A consolidação orquestrada.
54. `lgbm_mapped`: Score normalizado no plano [0, 100].
55. `if_percentile`: Saída consultiva do *Isolation Forest* indicativo de anomalia isolada de perfil (Fator estatístico multivariado).
56. `se_score`: Métrica preditiva gerada internamente pela matriz analítica de Engenharia Social (SE).
57. `beh_score`: Métrica de Análise Comportamental.

---

## I. Sinais / Restrições do Professor (Contrato Congelado R5B16/R5B18)

Estas métricas representam os metadados fixos de decisões retroativas (e flags operacionais da API) que, através do método Distillation Training, informam ao LightGBM R5B22 como simular os gates estritos, regras heurísticas complexas e mitigações de falsos positivos adotadas nos experimentos de redução.

58. `r4g_fast_frozen_decisao_recommended` (Categórica): Qual foi a deliberação base original na R4G.
59. `r5b14_rule_applied` (Categórica): Que regra de contenção FN exata de segurança (R5B14) foi acionada, se houve.
60. `r5b14_layer_applied` (Categórica): Escalonamento forçado pela R5B14 (ex: `APPROVE_TO_BLOCK`).
61. `ds_tipo_chave_norm_frozen` (Categórica)
62. `value_band_frozen` (Categórica)
63. `periodo_dia_frozen` (Categórica)
64. `score_bin` (Categórica)
65. `lgbm_bin` (Categórica)
66. `if_bin` (Categórica)
67. `ratio_bin` (Categórica)
68. `qtd_rec_bin` (Categórica)
69. `valor_rec_bin` (Categórica)
70. `mbk_available_flag_frozen`
71. `first_receiver_flag_real_frozen`
72. `ratio_valor_maximo_pagador_180d_frozen`
73. `ratio_valor_media_pagador_90d_frozen`
74. `vl_pix_frozen`
75. `qtd_pix_pagador_180d_frozen`
76. `valor_total_pagador_180d_frozen`
77. `qtd_pix_mesmo_recebedor_180d_frozen`
78. `valor_total_para_recebedor_180d_frozen`

---

> **Observação:** As colunas com os sufixos `_frozen` ou sufixos baseadas em classificações (ex: `_bin`) são reflexos da base mestra original consolidada do Baseline, fornecendo um roteiro estável (sem drift transacional) ao classificador aluno (LGBM) da fase R5B22 de simular com perfeição (Distillation Model) as lógicas contratuais restritivas de intervenção e bloqueio, resultando numa precisão superior sem a quebra do "teto" de falso positivo.
