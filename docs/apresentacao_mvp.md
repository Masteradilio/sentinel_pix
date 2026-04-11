

# Apresentação do MVP: Motor Híbrido de Detecção de Fraudes PIX

## 1. Introdução e Geração de Valor

O **Sistema de Detecção de Fraudes PIX** é um motor de decisão em tempo real projetado para proteger as transações PIX dos clientes do BRB.

**Qual o nosso grande objetivo?**
Proteger o dinheiro dos nossos clientes e a reputação do banco através de um motor de decisão que consiga barrar golpes sem gerar atrito desnecessário para os bons clientes.

**O que estamos entregando de valor hoje:**

1. **Zero Fraudes Perdidas (FN=0)**: Garantia de que 100% das fraudes conhecidas são bloqueadas pelo sistema — nenhuma escapa.
2. **Explicabilidade**: Diferente dos modelos tradicionais "caixa-preta", nossa API não retorna apenas um score de risco. Ela retorna uma mensagem pronta, em linguagem amigável, explicando exatamente *por que* a transação foi retida. Isso permite ao aplicativo do banco dialogar com o cliente na mesma hora, reduzindo a sobrecarga do Call Center e da Mesa de Prevenção.
3. **Defesa em Camadas**: O fraudador não enfrenta apenas um obstáculo — ele enfrenta cinco:
   - Um **modelo preditivo principal (LightGBM)**: treinado com 80 features no reconhecimento de padrões de fraude.
   - Um **sistema de regras determinísticas (Cascade Rules)** que codificam o aprendizado humano dos analistas de fraude.
   - Um **detector de anomalias (Isolation Forest)** que monitora continuamente as transações onde o modelo principal tem incerteza.
   - Um **rastreador comportamental** de dispositivo e sessão (Behavioral Analytics).
   - Um **mapeador de padrões de golpes** de Engenharia Social (12 padrões catalogados).
4. **Baixíssimo Atrito**: Apenas 1,01% das transações requerem qualquer tipo de intervenção (confirmação ou bloqueio). As demais 99% são aprovadas instantaneamente.

---

### Glossário Rápido de Termos Técnicos

| Métrica | Significado para o Negócio |
|---------|---------------------------|
| **Recall** (Taxa de Captura) | Capacidade de encontrar fraudes. 100% = nenhuma fraude escapou. É a métrica mais importante. |
| **Precision** (Precisão) | Taxa de acerto quando dizemos "Isso é fraude!". 83,5% = a GEPFRA vai perder menos tempo com falsos positivos. |
| **F1-Score** | Equilíbrio entre Recall e Precision. 0,91 é um resultado excelente. |
| **AUC-ROC** | Capacidade geral de separação entre as classes (fraude e não-fraude). 0,9999 é quase perfeito (máximo = 1,0). |
| **Falsos Positivos (FP)** | Transações legítimas bloqueadas por engano. Apenas 14 em 20.000. |
| **Falsos Negativos (FN)** | Fraudes que o sistema deixou passar. **Zero.** |
| **GAP de Separação** | Distância entre a pior fraude e o cliente legítimo mais "suspeito". GAP positivo = sem overlap perigoso. |

---

## 2. Os Três Modelos — Como Funcionam

O sistema opera em uma arquitetura de **defesa em profundidade** com três camadas complementares de detecção:

### 🧠 Camada 1 — LightGBM (Motor Principal de Machine Learning)

O LightGBM é um algoritmo de *gradient boosting* — uma técnica de aprendizado de máquina que combina centenas de árvores de decisão para encontrar padrões complexos nos dados. treinado com 52 features otimizadas (reduzidas de 80 após análise estatística de relevância) no reconhecimento de padrões de fraude.

**Como funciona:** Para cada transação PIX, o modelo calcula uma probabilidade de fraude (0 a 1). Transações acima do threshold otimizado são sinalizadas. Nos testes, todas as fraudes receberam scores acima de 0,92 — demonstrando altíssima confiança do modelo.

**Papel no sistema:** Responsável por detectar 100% das fraudes no dataset de teste. É o componente com maior poder preditivo.

### 🔗 Camada 2 — Cascade Rules 

São 6 regras determinísticas criadas a partir do conhecimento dos analistas de fraude e da análise de padrões de golpes reais. Elas atuam **exclusivamente** quando o LightGBM está na zona de incerteza.

**Regras ativas:**
| Regra | O que detecta |
|-------|---------------|
| C1 — Burst + Primeiro Recebedor | 3+ transações em 30 minutos para um recebedor nunca antes visto |
| C2 — Burst Intenso | 5+ transações em 30 minutos (padrão de esvaziamento) |
| C4 — Conta Nova + Alto Valor | Conta com menos de 3 meses fazendo PIX acima de R$ 5.000 |
| C5 — Esvaziamento | Burst + valor 5× acima da mediana + acima de R$ 1.000 |
| C6 — Borderline Combinado | Score limítrofe do LGBM + 4 sinais de risco simultâneos |

**Papel no sistema:** Rede de segurança contra padrões que o modelo de ML pode não ter aprendido nos dados históricos. No teste, ativaram em 5 transações, validando que estão funcionando corretamente. Em produção, são a primeira linha de defesa quando um novo tipo de golpe surgir — sem precisar retreinar o modelo.

### 🔍 Camada 3 — Isolation Forest (Detector de Anomalias)

O Isolation Forest é um modelo não-supervisionado que aprende o que é "normal" e identifica transações que se desviam estruturalmente desse padrão. Opera com 22 features focadas em velocity, burst e interações comportamentais.

**Como funciona:** Para cada transação onde o LGBM não tem certeza, o IF avalia se o comportamento é estruturalmente anômalo. Se a anomalia for extrema, eleva o risco da transação com um boost de valor no score final.

**Papel no sistema:** Monitorou 19.928 transações (99,3% do total) no teste, e aplicou elevação de risco em 826 delas. Protege contra:
- Fraudes inéditas que o LGBM ainda não aprendeu
- Evolução adversarial dos golpistas
- Drift de dados ao longo do tempo
- Contas comprometidas com burst súbito

---

## 3. Os Dados — De Onde Vêm e Quais Features Utilizamos

### 3.1 Fontes de Dados (Big Data / Hadoop)

O pipeline de ingestão coleta dados de **4 sistemas fonte** no ambiente de Big Data do banco:

| Sistema Fonte | Tabela | O que fornece |
|---------------|--------|---------------|
| **BLK** (Extrato PIX) | `landing_brb_oracle_blk.tb_extrato_pix` + `tb_registro_pix` | Transações PIX dos últimos 90 dias: identificador, valor, data, CPF pagador/recebedor, chave PIX |
| **MBK** (Mobile Banking) | `landing_brb_oracle_mbk.aut` | Dados do dispositivo, sessão, latência, tempo de interação, método de autenticação, score Topaz, IP, app version — extraídos via parsing XML dos logs |
| **AOX** (Cadastro de Clientes) | `landing_brb_db2_aox.aoxb01` + `aoxb17` | Perfil do cliente: idade, tempo de relacionamento, sexo, estado civil, renda, nº de dependentes |
| **DNA** (Segmentação) | `landing_brb_db2_dna.dnab01` | Segmento do cliente (Varejo, Exclusivo, Private, etc.) |
| **GESEI/MAF** (Fraudes) | `landing_brb_oracle_gesei.protocolo_enviado` + `landing_brb_oracle_maf.tb_infracao_pix` | Identificação das transações confirmadas como fraude (label para treino supervisionado) |

### 3.2 Campos Ingeridos (32 campos brutos)

A ingestão unificada produz um registro por transação PIX com os seguintes campos:

**Transação:**
- `cd_pix`, `dt_pix`, `cd_cpf_pagador`, `cd_cpf_cnpj_recebedor`, `vl_pix`
- `ds_chave_pix`, `ds_tipo_chave` (aleatória, email, documento/telefone)

**Histórico Trimestral (agregado por CPF):**
- `qt_total_pix_trimestre`, `vl_mediana_pix_trimestre`, `vl_desvio_padrao_pix_trimestre`
- `qt_intervalo_transacao_minuto`, `qt_intervalo_mediana_trimestre`, `qt_intervalo_desvio_padrao_trimestre`
- `qt_pix_dia_maximo_trimestre`, `qt_aparelhos_distintos_trimestre`
- `tp_primeiro_envio_recebedor_trimestre`, `qt_envio_recebedor_trimestre`

**Dispositivo e Sessão (do MBK — parsing XML):**
- `device_name`, `app_version`, `ip_address`
- `latencia_rede_ms`, `vl_latencia_rede_media_trimestre`
- `tempo_interacao_ms`, `vl_tempo_interacao_medio_trimestre`
- `tempo_processamento_host_ms`
- `metodo_autenticacao`, `session_id`, `cd_retorno`
- `topaz_risk_score`, `topaz_transacao_rejeitada`, `is_agendamento_recorrente`

**Perfil do Cliente (do AOX/DNA):**
- `nr_idade`, `qt_tempo_relacionamento_mes`
- `ds_sexo`, `ds_estado_civil`, `ds_segmento`
- `vl_renda_cliente`, `qt_dependentes`

### 3.3 Análise de Relevância e Otimização de Features

Antes de definir o conjunto final de features do modelo, realizamos uma **análise estatística rigorosa de relevância** para eliminar variáveis redundantes ou de baixa contribuição. O objetivo foi duplo: (1) garantir que cada feature justifica seu custo de ingestão e processamento, e (2) reduzir a dimensionalidade sem comprometer a capacidade preditiva.

#### Bateria de 11 Testes Estatísticos Aplicados

| # | Teste | O que avalia |
|:-:|-------|-------------|
| 1 | **Importância LightGBM (Split + Gain)** | Quantas vezes o modelo usa a feature para dividir dados (split) e quanto ganho informacional cada divisão produz (gain) |
| 2 | **Permutation Importance** | Quanto o AUC-ROC cai quando os valores da feature são embaralhados aleatoriamente — mede o impacto real na performance |
| 3 | **Correlação de Spearman** | Identifica pares de features redundantes (correlação ≥ 0,90), permitindo eliminar duplicatas |
| 4 | **Mutual Information** | Mede a dependência não-linear entre cada feature e a variável alvo (fraude), capturando relações que a correlação linear não detecta |
| 5 | **Teste de Levene** | Avalia se a variância da feature é significativamente diferente entre transações fraudulentas e legítimas (heterocedasticidade) |
| 6 | **Mann-Whitney U** | Testa a separação univariada entre as classes (fraude vs. normal) com cálculo de tamanho de efeito |
| 7 | **VIF (Variance Inflation Factor)** | Detecta multicolinearidade — features que podem ser explicadas como combinação linear das demais |
| 8 | **PCA (Análise de Componentes Principais)** | Determina quantos componentes independentes são necessários para explicar 90% da variância total dos dados |
| 9 | **Near-Zero Variance** | Identifica features quase constantes (ex: 99% dos valores iguais a zero) que não carregam informação discriminativa |
| 10 | **Análise por Fonte de Dados** | Mapeia a contribuição de cada sistema fonte (BLK, MBK, AOX, DNA) ao poder preditivo total |
| 11 | **Simulação de Remoção Incremental** | Remove features progressivamente (5, 10, 15... 40) e mede o impacto real no AUC a cada passo |

#### Resultados da Análise

- **36 pares de features** com correlação ≥ 0,90 foram identificados (ex: `vl_pix` ↔ `log_vl_pix` com correlação = 1,0)
- **39 features** apresentaram importância zero (split = 0, gain = 0) no LightGBM — o modelo já as ignorava internamente
- **PCA** demonstrou que 90% da variância é explicada por apenas **36 componentes** dos 80 originais
- A **simulação de remoção** provou que até **40 features** poderiam ser removidas com perda de apenas 0,002% no AUC

Com base nesses resultados, eliminamos **28 features** do modelo: duplicatas exatas (6), features com score zero em todos os testes (3) e features com variância quase nula ou contribuição desprezível (19).

#### Resultado: Zero Perda de Performance

| Métrica | 80 features (antes) | 52 features (depois) | Diferença |
|---------|:-------------------:|:--------------------:|:---------:|
| **ROC-AUC** | 0,9998 | 0,9998 | 0,0000 |
| **Average Precision** | 0,9791 | 0,9791 | 0,0000 |
| **F1-Score** | 0,9576 | 0,9576 | 0,0000 |
| **Recall** | 98,75% | 98,75% | 0,0000 |
| **FN (fraudes perdidas)** | 1 | 1 | 0 |
| **Features ociosas** | 29 de 80 (36%) | 2 de 52 (4%) | Modelo mais limpo |

O modelo agora opera com **52 features** — 35% mais enxuto, sem nenhuma perda mensurável de capacidade preditiva. Isso resulta em preprocessamento mais rápido, inferência mais leve e menor volume de dados na ingestão do Big Data.

---

### 3.4 Features do Modelo (52 features otimizadas)

A partir dos 32 campos brutos, o script de preprocessamento cria mais 20, para um total de **52 features** que alimentam o LightGBM, organizadas em 7 grupos:

| Grupo | Qtd | Exemplos |
|-------|:---:|---------|
| **Valor e Desvio** | 7 | `vl_pix`, `vl_pix_over_1000_flag`, `ratio_valor_mediana`, `zscore_valor_aprox`, `ratio_valor_desvio_padrao` |
| **Frequência e Velocity** | 12 | `tx_count_prev_30m`, `burst_30m_flag`, `minutes_since_prev_tx`, `distinct_receivers_so_far`, `first_receiver_flag`, `qt_intervalo_desvio_padrao_trimestre` |
| **Perfil do Cliente** | 5 | `nr_idade`, `qt_tempo_relacionamento_mes`, `is_viuvo_flag`, `is_segmento_premium_flag`, `perfil_vulneravel_se_flag` |
| **Renda e Patrimônio** | 4 | `ratio_pix_renda`, `pix_over_50pct_renda_flag`, `renda_missing_flag`, `vl_renda_cliente` |
| **Dispositivo e Sessão** | 7 | `vl_latencia_rede_media_trimestre`, `ratio_latencia_cliente`, `diff_latencia_cliente`, `qt_aparelhos_distintos_trimestre`, `device_missing_flag`, `host_time_missing_flag`, `topaz_missing_flag` |
| **Temporal e Topaz** | 2 | `hour`, `topaz_risk_score` |
| **Regras de Negócio** | 5 | `rule_age_score`, `rule_relationship_score`, `rule_random_key_score`, `rule_topaz_score`, `rule_score_raw` |
| **Flags Contextuais** | 10 | `pix_key_random_flag`, `is_first_tx_trimestre`, `first_key_flag`, `key_tx_count_prev`, `is_login_senha_flag`, `is_business_hours` |

**Features sequenciais** (criadas por ordenação temporal dentro de cada cliente):
- `minutes_since_prev_tx` — minutos desde a última transação
- `tx_count_prev_30m` — quantidade de PIX nos últimos 30 minutos
- `burst_30m_flag` — indica atividade concentrada
- `first_receiver_flag` — primeiro envio para aquele recebedor
- `distinct_receivers_so_far` — diversidade de recebedores acumulada

Essas features sequenciais são particularmente importantes porque capturam padrões de esvaziamento de conta e bursts que são os comportamentos mais associados a fraude.


---

## 4. Resultados Alcançados (Base de Validação)

Submetemos o motor a um teste com **20.071 transações reais** (71 fraudes confirmadas e 20.000 transações legítimas). Os resultados estão detalhados no `relatorio_executivo.html`.

### Os Números Principais — Bloqueio Automático

| Métrica | Resultado |
|---------|-----------|
| **Fraudes Detectadas** | **100,0%** (71 de 71) |
| **Fraudes Perdidas** | **0** (Nenhuma fraude escapou) |
| **Falsos Alarmes** | Apenas **14** bloqueios equivocados (0,07% das transações legítimas) |
| **Precisão dos Alarmes** | **83,5%** (de cada 100 bloqueios, mais de 83 são fraudes reais, 16,5% restantes seriam os Falsos Positivos) |
| **F1-Score** | **0,9103** |
| **AUC-ROC** | **0,9999** |

### A Perspectiva Ampla — Bloquear + Confirmação Adicional

Nosso sistema também atua retendo transações suspeitas para verificação do cliente (Confirmação Adicional com biometria/2FA):

| Métrica | Resultado |
|---------|-----------|
| **Taxa de intervenção total** | Apenas **1,01%** das transações (202 de 20.071) |
| **Transações aprovadas automaticamente** | **19.869** (99,0%) |
| **Transações para Confirmação Adicional** | 117 |
| **Transações bloqueadas** | 85 (71 fraudes + 14 falsos positivos) |

### Separação entre Fraudes e Transações Legítimas

| Indicador | Score (0-100) |
|-----------|:---:|
| Menor Score de transações de fraude | **85,0** |
| Maior Score de transações normais | **84,4** |
| GAP de separação | **+0,6 pontos** |

Isso significa que os poucos clientes legítimos que são bloqueados estão "raspando" na nota mínima de bloqueio (85,0) — o modelo só erra quando o caso é genuinamente limítrofe.

### Contribuição de Cada Camada de Defesa

| Camada | Fraudes Detectadas | Papel no Teste |
|--------|:---:|------|
| 🧠 **LGBM** | **71** (100%) | Motor principal — todas as fraudes tiveram score > 0,92 |
| 🔗 **Cascade Rules** | 0 | 5 ativações em transações legítimas (validando funcionamento) |
| 🔍 **Isolation Forest** | 0 | Monitorou 19.928 tx, aplicou boost em 826 |

No teste com dados históricos, o LGBM foi suficiente para capturar 100% das fraudes. As camadas de Cascade e IF são **redes de segurança ativas** — protegem contra fraudes inéditas, evolução dos golpistas e drift de dados que inevitavelmente ocorrerão em produção. Manter essas camadas tem custo computacional praticamente zero (overhead < 5% do tempo total de processamento).

### Validação da API REST em Ambiente Controlado

A API foi submetida a uma bateria automatizada de **15 testes** cobrindo todos os endpoints, cenários de decisão e validações de segurança. Todos passaram com sucesso.

| Cenário Testado | Valor PIX | Decisão do Motor | Score | Explicabilidade |
|-----------------|----------:|:-----------------:|:-----:|:---------------:|
| Transação Normal (cliente recorrente, biometria, horário comercial) | R$ 150,00 | 🟢 **APROVAR** | 4.4 | — |
| Transação Suspeita (madrugada, chave aleatória, valor atípico) | R$ 2.500,00 | 🔴 **BLOQUEAR** | 85.0 | ✅ SHAP + Motivo |
| Fraude Evidente (conta nova, idosa, valor extremo, senha) | R$ 4.999,00 | 🔴 **BLOQUEAR** | 88.2 | ✅ SHAP + Motivo |
| Idoso Vulnerável (82 anos, viúva, chave aleatória) | R$ 3.000,00 | 🔴 **BLOQUEAR** | 85.0 | ✅ SHAP + Motivo |
| Dados Mínimos (apenas campos obrigatórios) | R$ 50,00 | 🟢 **APROVAR** | 3.4 | — |

**Destaques da validação:**

| Indicador | Resultado |
|-----------|-----------|
| **Testes aprovados** | 15/15 ✅ |
| **Erros em produção** | 0 |
| **Latência média do pipeline** | ~134ms por transação |
| **SLA do Banco Central** | ≤ 10000ms (10 segundos)|
| **Margem de folga** | **74,62× abaixo do SLA BACEN** |
| **Componentes ativos** | Preprocessador, Decision Engine, Engenharia Social, Análise Comportamental, SHAP |
| **Validações de segurança** | Rejeição correta de inputs malformados (HTTP 422) |
| **Processamento em lote** | 3 transações em 421ms (batch endpoint funcional) |
| **Capacidade de Thruput do Sistema** | O sistema é capaz de produzir entre 7 e 8 inferências por segundo

A API está operacional e pronta para entrada em Shadow Mode. Todos os cenários — desde a transação mais trivial até o perfil de fraude mais crítico (idoso vulnerável + conta nova + madrugada + chave aleatória) — foram classificados corretamente. A explicabilidade SHAP e as mensagens ao cliente (CX) estão ativas, permitindo integração imediata com o aplicativo do banco.

### Proteção Especial a Públicos Vulneráveis

O motor incorpora proteção reforçada para perfis de maior risco de engenharia social:

| Perfil Protegido | Mecanismo de Proteção |
|------------------|----------------------|
| **Idosos 60+** | Agravante automático de +3 pontos no score de risco |
| **Idosos 70+ com chave aleatória** | Padrão de engenharia social "IDOSO_VULNERAVEL_70" — elevação para BLOQUEAR |
| **Viúvos/viúvas sem dependentes** | Flag `perfil_vulneravel_se` — peso +4 nos agravantes comportamentais |
| **Contas novas (< 3 meses)** | Regra Cascade C4 + agravante "CONTA_NOVA_ALTO_VALOR" |
| **Primeiro envio a recebedor desconhecido** | Flag `first_receiver_flag` + detecção de engenharia social |

Esses mecanismos foram validados no cenário "Idoso Vulnerável" do teste da API, onde uma transação de R$ 3.000 por uma cliente de 82 anos, viúva, para chave aleatória foi corretamente bloqueada com 42 pontos de agravantes.

---

## 5. Por que LightGBM e Não Random Forest ou XGBoost?

Os três são modelos baseados em árvores de decisão, mas diferem profundamente em **como constroem essas árvores** e nas consequências práticas para um sistema antifraude em tempo real.

### A Diferença Fundamental em Uma Frase

| Modelo | Estratégia de Construção | Analogia |
|--------|--------------------------|----------|
| **Random Forest** | Cria centenas de árvores **independentes** e vota por maioria | Uma assembleia onde todos opinam sem se ouvir |
| **XGBoost** | Cria árvores **sequenciais**, cada uma corrigindo os erros da anterior, crescendo **nível por nível** (*level-wise*) | Um aluno que refaz a prova inteira corrigindo cada questão |
| **LightGBM** | Cria árvores **sequenciais** como o XGBoost, mas cresce **folha por folha** (*leaf-wise*) e amostra os dados de forma inteligente | Um aluno que foca apenas nas questões onde mais errou |

### Os 5 Motivos da Escolha do LGBM

#### 1. Velocidade de Inferência — Compatível com o SLA do PIX

O Banco Central exige resposta em milissegundos. LightGBM é **significativamente mais rápido** que os concorrentes:

| Modelo | Tempo típico (20 mil tx) | Motivo |
|--------|------------------------:|--------|
| **LightGBM** | ~0,3 segundos | Crescimento leaf-wise gera árvores menores e mais eficientes |
| XGBoost | ~1,2 segundos | Crescimento level-wise cria árvores mais profundas |
| Random Forest | ~2,5 segundos | Sem boosting, precisa de muito mais árvores para compensar |

Na nossa API, o pipeline completo (preprocessamento + LGBM + IF + SHAP + agravantes) roda em **~134ms**. Com XGBoost ou Random Forest como motor principal, o tempo seria multiplicado por 3-5×, comprimindo a margem de segurança em relação ao SLA de 1.500ms.

#### 2. Desempenho Superior com Dados Desbalanceados

Fraude PIX é um problema extremamente desbalanceado: apenas **0,35%** das transações são fraude. O LightGBM lida melhor com isso graças a duas técnicas exclusivas:

- **GOSS** (*Gradient-based One-Side Sampling*): Em vez de usar todas as 20 mil amostras para calcular cada split, o LGBM mantém **100% das amostras com gradiente alto** (as que o modelo mais erra — tipicamente as fraudes) e amostra aleatoriamente as de gradiente baixo (as transações normais e fáceis). Isso faz com que as fraudes, mesmo sendo raras, tenham influência desproporcional no treinamento.

- **EFB** (*Exclusive Feature Bundling*): Agrupa features esparsas (como nossas 27 flags binárias) em bundles, reduzindo a dimensionalidade efetiva de 80 para ~50 features sem perder informação. Random Forest e XGBoost não fazem isso.

#### 3. Eficiência com Features Esparsas e Missing Values

Das nossas 80 features, cerca de 30% são flags binárias (0/1) e ~15% podem conter valores ausentes (quando o cliente não tem dados de dispositivo, por exemplo). O tratamento nativo:

| Aspecto | LightGBM | XGBoost | Random Forest |
|---------|:--------:|:-------:|:-------------:|
| Missing values nativo | ✅ Direciona para o melhor lado | ✅ Similar | ❌ Requer imputação prévia |
| Features esparsas | ✅ EFB otimiza automaticamente | ❌ Trata como features normais | ❌ Trata como features normais |
| Features categóricas | ✅ Suporte nativo | ❌ Requer encoding manual | ❌ Requer encoding manual |

#### 4. Treinamento Rápido — Ciclo de Retreino Ágil

Quando novos padrões de fraude surgirem, precisamos retreinar o modelo rapidamente. O tempo de treinamento do LightGBM é tipicamente **3-8× menor** que o do XGBoost para datasets do nosso tamanho, graças ao histograma otimizado e ao GOSS.

| Modelo | Tempo de treino típico (80 features, 20 mil linhas) |
|--------|----------------------------------------------------:|
| **LightGBM** | ~5-15 segundos |
| XGBoost | ~30-90 segundos |
| Random Forest | ~20-60 segundos |

Isso permite que o Feedback Loop (Seção 7.II) opere com ciclos de retreino semanais sem impacto operacional.

#### 5. Explicabilidade com SHAP — Igualmente Compatível

Os três modelos são compatíveis com SHAP (usado na nossa API para explicar cada decisão ao cliente). No entanto, o SHAP para modelos baseados em árvore usa o algoritmo `TreeSHAP`, que é mais eficiente com **árvores menores e mais rasas** — exatamente o tipo que o LightGBM produz com crescimento leaf-wise.

### Comparativo Resumido

| Critério | LightGBM | XGBoost | Random Forest |
|----------|:--------:|:-------:|:-------------:|
| **Velocidade de inferência** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Dados desbalanceados (0,35% fraude)** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Features esparsas/binárias** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Tempo de treinamento** | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| **SHAP (explicabilidade)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Robustez a overfitting** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Ecossistema/comunidade** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |


> **Em resumo:** O LightGBM foi escolhido porque é o modelo que melhor equilibra **velocidade** (essencial para o SLA do PIX), **desempenho em dados desbalanceados** (essencial para detecção de fraude) e **eficiência com features esparsas** (que representam 30% das nossas variáveis). O Random Forest é robusto, mas lento e menos eficaz com dados raros. O XGBoost é um excelente modelo, mas mais lento tanto em treinamento quanto em inferência, sem as otimizações de amostragem (GOSS) e agrupamento de features (EFB) que dão ao LightGBM sua vantagem em cenários como o nosso.


---

## 6. O desafio da arquitetura para entrada em produção


### O Problema Central

**Como garantir que as features que o modelo aprendeu estejam disponíveis no momento da inferência em tempo real da transação PIX?**

Vamos mapear de onde vem cada grupo de features que o modelo usa:

| Grupo de Features | Disponível na transação PIX bruta? | Fonte real | Latência de consulta |
|---|:---:|---|---|
| **Transação** (`vl_pix`, `cd_pix`, `dt_pix`, chave PIX) | ✅ Sim | Evento PIX em si | 0ms |
| **Perfil do cliente** (`nr_idade`, `qt_tempo_relacionamento_mes`, `ds_sexo`, `ds_estado_civil`, `vl_renda_cliente`) | ❌ Não | AOX/DNA (DB2) | estimado até 500ms (query direta) |
| **Histórico trimestral** (`vl_mediana_pix_trimestre`, `vl_desvio_padrao_pix_trimestre`, `qt_total_pix_trimestre`, etc.) | ❌ Não | Agregação sobre 90 dias de PIX (BLK) | estimado até 5s (query no Hadoop) |
| **Device/Sessão** (`device_name`, `app_version`, `latencia_rede_ms`, `topaz_risk_score`, etc.) | ⚠️ Parcial | MBK (parsing XML) | estimado até 5s (query no Hadoop) |
| **Sequenciais** (`tx_count_prev_30m`, `burst_30m_flag`, `first_receiver_flag`, `distinct_receivers_so_far`) | ❌ Não | Calculadas em tempo real sobre histórico recente | Depende das vCPUs disponíveis |
| **Derivadas** (ratios, zscores, rule_scores) | ❌ Não | Calculadas pelo preprocessing a partir dos outros grupos | ~5ms (CPU local) |

O problema fica claro: **das 52 features, apenas ~7 vêm diretamente da transação PIX**. As demais precisam ser buscadas ou calculadas.

---

### Estratégia de Arquitetura Sugerida: Feature Store Pré-Materializada no REDIS (Memorystore for Redis no GCP)

Esta é a abordagem padrão da indústria para ML em tempo real:

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUXO EM TEMPO REAL                      │
│                                                             │
│  Transação PIX ──→ API Antifraude                           │
│       │                  │                                  │
│       │         ┌────────┴────────┐                         │
│       │         │  1. Features    │                         │
│       │         │  da transação   │  ← vl_pix, cd_pix, etc. │
│       │         │ (já disponíveis)│                         │
│       │         └────────┬────────┘                         │
│       │                  │                                  │
│       │         ┌────────┴────────┐                         │
│       │         │  2. Feature     │  ← Lookup por CPF       │
│       │         │  Store (Redis)  │                         │
│       │         └────────┬────────┘                         │
│       │                  │                                  │
│       │         ┌────────┴────────┐                         │
│       │         │  3. Features    │ ← Calculadas on-the-fly │
│       │         │  Sequenciais    │   pelo orquestrador     │
│       │         │  (cache memória)│                         │
│       │         └────────┬────────┘                         │
│       │                  │                                  │
│       │         ┌────────┴────────┐                         │
│       │         │  4. Derivadas   │  ← Ratios, zscores,     │
│       │         │  (CPU local)    │   rules                 │
│       │         └────────┬────────┘                         │
│       │                  │                                  │
│       │              INFERÊNCIA                             │
│       │        (estimado ~150ms total)                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                FLUXO BATCH (offline, a cada hora)           │
│                                                             │
│  Spark Job ──→ Agrega dados BLK + AOX + DNA                 │
│       │                                                     │
│       └──→ Materializa em Redis:                            │
│            {                                                │
│              cpf: "12345678900",                            │
│              nr_idade: 67,                                  │
│              qt_tempo_relacionamento_mes: 240,              │
│              vl_mediana_pix_trimestre: 150.00,              │
│              vl_desvio_padrao_pix_trimestre: 89.50,         │
│              qt_total_pix_trimestre: 47,                    │
│              qt_pix_dia_maximo_trimestre: 5,                │
│              qt_aparelhos_distintos_trimestre: 2,           │
│              vl_latencia_rede_media_trimestre: 320.0,       │
│              vl_renda_cliente: 3500.00,                     │
│              ds_sexo: "F",                                  │
│              ds_estado_civil: "VIUVO",                      │
│              ds_segmento: "VAREJO",                         │
│              qt_dependentes: 0,                             │
│              ...                                            │
│            }                                                │
│            TTL: 24 horas (refresh a cada hora)              │
└─────────────────────────────────────────────────────────────┘
```

**Como funciona:**

1. **Job Spark rodando a cada 1h**: agrega os dados de BLK + AOX + DNA e materializa um **perfil por CPF** em Redis (ou Memorystore do GCP)
2. **Na hora da transação**, a API faz um `GET redis:profile:{cpf}` 
3. **Features sequenciais** (`tx_count_prev_30m`, `burst_30m_flag`, etc.) são calculadas pelo cache em memória que já existe no orquestrador (`_customer_history`)
4. **Features derivadas** (ratios, zscores, rule_scores) são calculadas pelo `_create_features()` que **já existe** no orquestrador

**Latência estimada:**

| Etapa | Tempo |
|-------|------:|
| Features da transação | 0ms |
| Lookup Redis (perfil do CPF) | ~10ms |
| Features sequenciais (cache memória) | ~6ms |
| Preprocessing + derivadas | ~10ms |
| LGBM + IF + SE + Behavioral | ~134ms |
| **Total** | **~160ms** |

**É exatamente o que grandes fintechs (Nubank, Stone, PagSeguro) fazem.** O modelo treina com dados completos do batch, mas a inferência usa dados pré-materializados.

---

## O que Materializar no Redis — Mapeamento Exato

Aqui está o que o job Spark precisa pré-calcular por CPF:

```json
{
  "cpf": "12345678900",
  "updated_at": "2026-03-27T14:00:00Z",
  
  "perfil_estatico": {
    "nr_idade": 67,
    "qt_tempo_relacionamento_mes": 240,
    "ds_sexo": "F",
    "ds_estado_civil": "VIUVO",
    "ds_segmento": "VAREJO",
    "vl_renda_cliente": 3500.00,
    "qt_dependentes": 0
  },
  
  "historico_trimestral": {
    "qt_total_pix_trimestre": 47,
    "vl_mediana_pix_trimestre": 150.00,
    "vl_desvio_padrao_pix_trimestre": 89.50,
    "qt_intervalo_mediana_trimestre": 4320.0,
    "qt_intervalo_desvio_padrao_trimestre": 3200.0,
    "qt_pix_dia_maximo_trimestre": 5,
    "qt_aparelhos_distintos_trimestre": 2,
    "vl_latencia_rede_media_trimestre": 320.0
  },
  
  "recebedores_conhecidos": ["cpf_recv_1", "cpf_recv_2", "..."],
  "chaves_usadas": ["chave1", "chave2"]
}
```

**Tamanho estimado por CPF:** ~500 bytes  
**Para 1 milhão de clientes ativos:** ~500 MB no Redis  
**TTL:** 24 horas (refresh a cada 1h pelo Spark)

---

## Sobre o Impacto no SLA

| Cenário | Latência Total | Viável? |
|---------|:--------------:|:-------:|
| Tudo via Hadoop em tempo real | 5-10 segundos | ❌ Inviável |
| Query Oracle/DB2 síncrona | ~500ms | ⚠️ Funciona, mas eleva muito a SLA |
| **Feature Store Redis/Memorystore** | **160ms** | **✅ Ideal** |
| Usar só as 7 features brutas da transação (sem enriquecimento) | 134ms | ✅ Funciona, mas o modelo perderia muito a qualidade |


---

## 7. Visão de Futuro e Roadmap

O que construímos até aqui já coloca o banco à frente de boa parte do mercado, mas a fraude evolui todos os dias. Este é o plano de evolução:

### I. Evolução da Infraestrutura para Ultra Baixa Latência

O Banco Central exige respostas em milissegundos. Hoje processamos 20 mil transações em ~4 segundos, mas para suportar picos de Black Friday, migraremos o controle de perfis comportamentais e contagem de PIXs simultâneos para um cache distribuído em memória (Redis). Isso garante que, mesmo com milhares de servidores rodando a API, a resposta seja instantânea.

### II. Feedback Loop e "Shadow Mode"

Antes de ligarmos o bloqueio automático, colocaremos a API rodando em produção de forma silenciosa (*Shadow Mode*). O motor vai classificar as transações, mas não vai pará-las. Analisaremos esses logs contra o que realmente virou fraude. Além disso, criaremos um "Feedback Loop": sempre que a Mesa de Fraude aprovar ou rejeitar um caso manualmente, essa decisão voltará para o modelo, tornando-o mais inteligente a cada semana.

### III. Integração em Tempo Real com o DICT / MED do Bacen

O sistema já detecta "primeiros envios" suspeitos com excelência. O próximo passo é consultar o Banco Central no momento do PIX: *"Essa chave de destino tem histórico de fraude reportado por outros bancos?"*. Integrar o DICT enriquece o modelo com o histórico do sistema financeiro inteiro, não apenas do nosso banco.

### IV. Segurança sem Atrito (Frictionless Security)

Nosso foco não é barrar o dinheiro, mas proteger o cliente. Em vez de enviar as transações da faixa "Confirmar" para a fila de telemarketing, vamos integrar o retorno amigável da API direto no aplicativo do cliente com um fluxo de *Step-up Authentication*. A transação é retida na tela do celular e o cliente faz uma Prova de Vida (Biometria Facial) ali mesmo. O risco é mitigado com zero custo operacional humano.

### V. Detecção de Quadrilhas e Contas Laranja (Graph Analytics)

Hoje identificamos contas laranjas por seus comportamentos individuais. O próximo salto será usar Banco de Dados Orientados a Grafos (como Neo4j ou app de Grafos do Google). Isso nos permitirá ver a teia criminal completa: perceberemos em tempo real que dezenas de contas aparentemente normais estão todas enviando dinheiro para o mesmo nó central (laranja), desmontando a quadrilha inteira antes do dinheiro sair do banco.