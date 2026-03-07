# Arquitetura prática para detecção **em tempo real** de fraude/anomalia em PIX

Adilio, dá para estruturar isso de forma **muito boa** como um **motor híbrido**:

1. **Regras especialistas**  
2. **Modelos supervisionados** para probabilidade de fraude  
3. **Modelos não supervisionados / anomalia** para “fora do padrão”  
4. **Behavioral analytics**  
5. **Camada de decisão e priorização humana**  

Para o seu caso, **eu não recomendaria um modelo único**.  
O melhor desenho tende a ser um **ensemble híbrido**, porque:

- fraude muda rápido;
- há fraude “parecida com histórico já conhecido”;
- há fraude nova, sem padrão antigo;
- há transação legítima, mas **muito anômala**;
- há fraude por **engenharia social**, em que o cliente “faz tudo certo” tecnicamente, mas o comportamento foge do normal.

---

# 1. O que as bases anexadas mostram sobre a estrutura dos dados

Pelas amostras fornecidas, você já tem uma base muito rica.

## 1.1 Base de PIX normais
Tem variáveis úteis como:

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
- `dt_carga`

Isso é excelente para:
- desvio do valor da transação vs histórico;
- desvio de frequência;
- burst de transações;
- mudança de dispositivo;
- perfil de cliente;
- comportamento temporal.

## 1.2 Base de fraudes PIX
Tem estrutura semelhante e claramente contém casos rotulados com fraude.  
Pelos campos, parece existir uma coluna como `tp_fraude` com valor `1` nas amostras. Isso funciona como label de fraude.

Campos importantes:
- `cd_cpf_pagador`
- `tp_fraude`
- `nr_idade`
- `latencia_rede_ms`
- `vl_mediana_pix_trimestre`
- `qt_tempo_relacionamento_mes`
- `dt_pix`
- `cd_pix`
- `vl_pix`
- `qt_aparelhos_distintos_trimestre`
- `tempo_interacao_ms`
- `qt_total_pix_trimestre`
- `qt_intervalo_transacao_minuto`
- `vl_desvio_padrao_pix_trimestre`
- `qt_pix_dia_maximo_trimestre`
- `vl_tempo_interacao_medio_trimestre`

## 1.3 Base mobile / app
Tem campos muito úteis para risco transacional:

- `nsu_transacao`
- `nr_conta`
- `end_to_end_id`
- `data_hora_inicio`
- `cd_tipo_transacao`
- `valor_transacao`
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

Essa base é crucial para:
- device fingerprint;
- session risk;
- risco do aparelho;
- sinais de automação/coerção;
- sinais de takeover/account compromise.

---

# 2. Objetivo real do sistema

Você descreveu dois problemas diferentes, que devem virar **duas cabeças do mesmo motor**:

## 2.1 Fraude clássica / transação suspeita
Casos em que a transação parece fraude por:
- valor;
- velocity;
- device;
- perfil de conta;
- histórico semelhante ao de fraudes passadas.

## 2.2 Engenharia social / behavioral anomaly
Casos em que:
- o cliente autenticou corretamente;
- o dispositivo pode até ser o usual;
- mas o comportamento transacional está **muito fora do normal**;
- há sinais como horário incomum, valor incomum, pix para novo favorecido, sequência anormal, interação muito curta/estranha, pressa, múltiplos pix, mudança de padrão.

Então eu montaria um sistema com **3 scores**:

- **Score A – Fraude supervisionada**
- **Score B – Anomalia comportamental**
- **Score C – Regras especialistas**

E depois um **meta-score final**.

---

# 3. Arquitetura recomendada

## 3.1 Camada 1 — Feature store em tempo real
Antes do modelo, você precisa de uma camada que calcule features online.

### Features online por transação
No momento da autorização do PIX, calcule:

- valor atual;
- hora do dia;
- dia da semana;
- valor / mediana do cliente;
- valor / p95 histórico do cliente;
- quantidade de PIX na última:
  - 5 min
  - 30 min
  - 1h
  - 24h
- soma dos valores nas últimas:
  - 30 min
  - 24h
- tempo desde último PIX;
- primeiro PIX do dia?;
- novo dispositivo?;
- novo IP?;
- nova versão do app?;
- novo destinatário?;
- chave aleatória?;
- conta destino suspeita?;
- score topaz;
- idade do cliente;
- tempo de relacionamento;
- tipo de autenticação;
- mudança de padrão de interação mobile;
- latência do app vs baseline do cliente.

Isso precisa estar disponível em **latência muito baixa**.

---

## 3.2 Camada 2 — Regras
As regras não substituem ML; elas entram como um **risk booster**.

## 3.3 Camada 3 — Modelo supervisionado
Classifica probabilidade de fraude com base em fraudes históricas conhecidas.

## 3.4 Camada 4 — Modelo de anomalia
Detecta quão “fora do normal” a transação está para aquele cliente/população.

## 3.5 Camada 5 — Behavioral analytics
Captura padrões ligados a coerção/engenharia social.

## 3.6 Camada 6 — Orquestrador de decisão
Produz:
- aprovar automaticamente;
- mandar para fila humana;
- reprovar automaticamente.

---

# 4. Melhor estratégia de modelagem

# 4.1 Não faça só classificação binária
Se fizer só `fraude vs não fraude`, você perde muito sinal.

O ideal é combinar:

## (a) Modelo supervisionado
Para aprender padrões de fraude conhecida.

Sugestões:
- **LightGBM**
- **XGBoost**
- **CatBoost**

Para dados tabulares bancários, esses geralmente são melhores que deep learning.

### Minha recomendação principal:
**LightGBM ou CatBoost** como modelo principal supervisionado.

Por quê?
- lida bem com não linearidade;
- capta interações;
- funciona muito bem em tabular;
- é rápido para inferência em tempo real;
- aceita missing com relativa robustez.

---

## (b) Modelo de anomalia
Para pegar fraudes novas e comportamento “não visto”.

Sugestões:
- **Isolation Forest**
- **Local Outlier Factor** (menos prático online)
- **One-Class SVM** (difícil escalar)
- **Autoencoder** (se tiver maturidade maior)
- **Robust z-score / distance-to-profile** por cliente

### Minha recomendação:
Use uma combinação de:
- **Isolation Forest** na população
- **score de desvio individual do cliente** baseado em baseline histórico

Porque uma fraude pode não ser anômala globalmente, mas pode ser **super anômala para aquele cliente**.

---

## (c) Behavioral analytics dedicado
Aqui você cria features específicas para desvio de comportamento, mais do que um modelo sofisticado.

Exemplos:
- cliente normalmente faz PIX de manhã e agora faz de madrugada;
- cliente fazia PIX de até R$ 100 e agora faz R$ 2.000;
- cliente fazia 1 PIX por dia e fez 4 em 10 min;
- cliente usava 1 aparelho e agora apareceu outro;
- cliente nunca usou chave aleatória;
- cliente demorava 40–90s na jornada e agora fez em 5s ou 1s;
- cliente fez transações “escadinha”;
- cliente começou com pequenos testes e logo subiu o valor.

Isso pode virar:
- um submodelo próprio;
- ou um bloco de features para o modelo principal.

---

# 5. Ensemble recomendado

Eu sugeriria um ensemble como este:

## Modelo 1 — `FraudClassifier`
Supervisionado com label de fraude.

**Entrada:**
- features históricas
- features de device
- features de sessão
- features de velocity
- topaz
- perfil do cliente

**Saída:**
- `p_fraude_supervisionado` entre 0 e 1

---

## Modelo 2 — `BehaviorAnomalyScore`
Anomalia comportamental por cliente.

**Entrada:**
- valor vs mediana / média / p95 do cliente
- tempo entre transações
- horário
- frequência
- destinatário novo
- aparelho novo
- padrão de interação

**Saída:**
- `score_anomalia_comportamental` entre 0 e 1

---

## Modelo 3 — `GlobalOutlierScore`
Anomalia populacional.

**Entrada:**
- valor
- relação com histórico
- velocity
- idade
- relacionamento
- latência
- tempo interação
- score topaz
- diversidade de aparelhos

**Saída:**
- `score_outlier_global` entre 0 e 1

---

## Motor de regras — `RuleRiskScore`
Aplica os agravantes/atenuantes que você definiu.

**Saída:**
- `score_regras`

---

## Meta-score final
Algo como:

```text
score_final =
0.45 * p_fraude_supervisionado +
0.25 * score_anomalia_comportamental +
0.15 * score_outlier_global +
0.15 * score_regras
```

Isso é só ponto de partida.  
Os pesos precisam ser calibrados com validação e custo de negócio.

---

# 6. Como transformar suas regras em score utilizável

Você já tem um bom conjunto inicial.

## 6.1 Agravantes

### 1. PIX em < 30 min
Sugestão:
- 1 tx extra em 30 min: +1
- 2 ou mais: +2

Melhor ainda usar:
- count últimas 30 min
- soma valores últimas 30 min
- número de destinatários distintos nas últimas 30 min

---

### 2. Razão PIX / Limite
Muito bom.

Sugestão:
- 0.40–0.59 = +1
- 0.60–0.79 = +2
- >=0.80 = +3

Também vale usar:
- razão valor atual / limite
- razão soma 24h / limite

---

### 3. Idade
Coerente como agravante operacional, desde que validado com cuidado regulatório e fairness.

- 60–65 = +1
- 66–75 = +2
- 76+ = +3

**Importante:** usar idade como variável pode trazer risco de viés.  
Eu manteria:
- como regra de priorização humana;
- ou como feature monitorada com governança forte.

---

### 4. Tempo de relacionamento
Muito útil.

- 61–90 dias = +1
- 31–60 = +2
- 0–30 = +3

---

### 5. Conta laranja
Peso 3 faz sentido, talvez até mais dependendo da confiança do sinal.

Mas “conta laranja” precisa ser uma variável robusta, derivada de:
- recebimentos concentrados e dispersão rápida;
- alta rotatividade de contrapartes;
- baixa permanência de saldo;
- padrão de funnel/transbordo;
- forte conexão com contas já marcadas.

---

### 6. Chave aleatória
Peso 2 razoável.

Mas sugiro refinar:
- chave aleatória + favorecido novo = peso maior
- chave aleatória isolada = peso moderado

---

### 7. Horário noturno
Peso 3 é forte; eu usaria com nuance:
- noturno **fora do padrão do cliente** pesa mais
- noturno por si só pesa menos

Exemplo:
- 22h–6h = +1
- 22h–6h e fora do hábito do cliente = +3

---

### 8. Velocity checks
Excelente. Aqui eu criaria uma subpontuação de 0 a 4.

Exemplos:
- 2+ transações em 5 min
- 3+ em 30 min
- aumento abrupto da soma 1h
- vários favorecidos novos em sequência
- transações em escadinha

---

### 9. Topaz
Peso 2 a 5 faz sentido.

Exemplo:
- 0–1 = +0
- 2 = +1
- 3 = +2
- 4 = +4
- 5 = +5

Também use:
- `topaz_transacao_rejeitada`
- `topaz_transacao_habilitada`

---

## 6.2 Atenuante
### Autorização prévia
“Reduz 50% do score final” é simples e bom.

Mas eu tomaria cuidado:
- só aplicar se a autorização prévia for forte e recente;
- talvez atenuar menos se houver Topaz alto + velocity alta.

Exemplo:
```text
se autorizacao_previa = true e score_regras < limiar_medio:
    score_final *= 0.5
senão se risco extremo:
    score_final *= 0.8
```

---

# 7. Behavioral analytics: o que criar de verdade

Esse é o ponto mais importante para engenharia social.

## 7.1 Features de desvio individual
Para cada cliente:

### Valor
- `vl_pix / mediana_30d`
- `vl_pix / media_30d`
- `vl_pix / p95_30d`
- z-score robusto do valor

### Frequência
- `qt_pix_30min`
- `qt_pix_1h`
- `qt_pix_24h`
- `qt_destinatarios_novos_24h`

### Tempo
- horário incomum para o cliente
- dia da semana incomum
- diferença para última transação
- burst após longos períodos de inatividade

### Dispositivo / canal
- novo device
- novo IP
- nova sessão
- mudança de app version
- método de autenticação diferente
- score topaz acima do baseline do cliente

### Interação
- `tempo_interacao_ms / media_cliente`
- `latencia_rede_ms / media_cliente`
- sequência muito rápida pode indicar script/pressa;
- sequência muito lenta pode indicar confusão/coação.

---

## 7.2 Padrões específicos de engenharia social
Você pode criar regras/features para sinais típicos:

- primeiro PIX alto para favorecido nunca usado;
- série de PIX crescentes;
- cliente idoso + horário incomum + destinatário novo;
- cliente com pouco relacionamento + transação alta;
- transação muito acima do perfil seguida de outras próximas;
- redução abrupta no tempo de interação;
- múltiplas tentativas/rejeições antes da efetivação;
- forte discrepância entre device risk e histórico do cliente;
- cliente que raramente faz PIX agora faz vários.

---

## 7.3 Sequência temporal
Fraude de engenharia social muitas vezes aparece como **sequência**, não como evento isolado.

Você pode criar:
- features de janela deslizante;
- score por sessão;
- score de cadeia transacional.

Se tiver maturidade futura:
- modelo de sequência (LSTM/Transformer) por sessão.  
Mas para começar, **não precisa**.  
Janela temporal + gradient boosting já entrega muito.

---

# 8. Features que eu criaria imediatamente

## 8.1 Features de valor
- `ratio_valor_mediana = vl_pix / max(vl_mediana_pix_trimestre, eps)`
- `ratio_valor_desvio = vl_pix / max(vl_desvio_padrao_pix_trimestre, eps)`
- `delta_valor = vl_pix - vl_mediana_pix_trimestre`
- `valor_log = log1p(vl_pix)`

## 8.2 Features de frequência
- `qt_total_pix_trimestre`
- `qt_intervalo_transacao_minuto`
- `qt_intervalo_mediana_trimestre`
- `qt_intervalo_desvio_padrao_trimestre`
- `qt_pix_dia_maximo_trimestre`

## 8.3 Features de idade/relacionamento
- `nr_idade`
- `bucket_idade`
- `qt_tempo_relacionamento_mes`
- `bucket_relacionamento`

## 8.4 Features mobile/device
- `latencia_rede_ms`
- `tempo_interacao_ms`
- `tempo_processamento_host_ms`
- `metodo_autenticacao`
- `topaz_risk_score`
- `device_name`
- `app_version`
- `ip_address` (idealmente transformado, não em bruto)
- `session_id`
- `novo_device_flag`
- `novo_ip_flag`

## 8.5 Features de consistência
- `latencia_vs_media = latencia_rede_ms / vl_latencia_rede_media_trimestre`
- `interacao_vs_media = tempo_interacao_ms / vl_tempo_interacao_medio_trimestre`
- `aparelhos_distintos = qt_aparelhos_distintos_trimestre`

## 8.6 Features temporais
- hora
- madrugada/noite
- fim de semana
- feriado
- horário fora do padrão do cliente

---

# 9. Como juntar as 3 bases

Você vai precisar de uma chave de junção operacional.

## Mais prováveis:
- `cd_pix` / `end_to_end_id`
- conta / CPF / timestamp aproximado
- `nsu_transacao`

Idealmente a tabela final por transação deve ficar assim:

- identificador da transação
- label fraude (0/1 quando houver histórico)
- features históricas do cliente
- features do evento PIX
- features do mobile/app
- features de regras
- features de behavioral analytics

---

# 10. Tratamento de classes desbalanceadas

Fraude será minoria forte.

## Recomendações:
- usar **class weights**
- ou `scale_pos_weight` em XGBoost/LightGBM
- calibrar threshold pelo custo de negócio
- avaliar com:
  - **PR-AUC**
  - Recall no topo do ranking
  - Precision@K
  - Capture rate
  - FPR operacional

**ROC-AUC sozinho não basta**.

---

# 11. Métricas certas para banco

Como o objetivo é revisar manualmente antes da fraude, pense em ranking.

## Métricas principais:
- **Recall@TopK**
- **Precision@TopK**
- **PR-AUC**
- **Taxa de captura de fraude**
- **False Positive Rate**
- **Alert rate**
- **Custo evitado**
- **Tempo médio de resposta**

## Exemplo de meta operacional:
- mandar só 0,5% ou 1% das transações para análise manual;
- dentro desse 1%, capturar 60%+ das fraudes.

---

# 12. Estratégia de decisão em produção

## Faixas de decisão
Exemplo:

- `score_final < 0.35` → aprova automático
- `0.35 <= score_final < 0.70` → revisão humana
- `score_final >= 0.70` → bloqueio/retenção temporária

Melhor ainda com política adaptativa:
- cliente alta renda/comportamento normal → threshold diferente
- cliente novo + topaz alto → threshold mais conservador

---

# 13. Explicabilidade

Em banco isso é fundamental.

## Recomendo:
- modelo principal em árvore boosted
- explicabilidade por **SHAP**
- guardar:
  - top 5 fatores do score
  - regras disparadas
  - score do topaz
  - score de anomalia
  - comparação com baseline do cliente

Exemplo de explicação para analista:
- valor 8,2x acima da mediana do cliente
- 3 PIX em 18 minutos
- destinatário novo
- horário noturno
- topaz = 4
- cliente 78 anos

Isso aumenta muito a utilidade operacional.

---

# 14. Cuidados metodológicos importantes

## 14.1 Evitar leakage temporal
Muito importante:
- treino com passado
- validação com período futuro
- teste com período ainda mais futuro

Nunca embaralhar aleatoriamente sem respeitar o tempo.

---

## 14.2 Validação por tempo
Faça algo como:

- treino: meses 1–6
- validação: mês 7
- teste: mês 8

Ou rolling window.

---

## 14.3 Concept drift
Fraude muda.

Monitore:
- distribuição das features
- queda de recall
- mudança de precision
- mudança no score médio
- novas regras emergentes

Retreine com frequência.

---

## 14.4 Fairness / governança
Idade e tempo de relacionamento são sensíveis.  
Pode usar, mas com governança:

- justificar tecnicamente;
- monitorar impacto;
- garantir que o uso é para proteção e não discriminação indevida;
- manter humano no loop.

---

# 15. Minha proposta concreta de solução

## Solução MVP forte
### Motor híbrido com 4 componentes:

### **(1) Regras especialistas**
Seu conjunto de agravantes/atenuantes.

### **(2) Classificador supervisionado**
- LightGBM ou CatBoost
- alvo: fraude conhecida

### **(3) Score de anomalia do cliente**
- distance-to-profile
- z-score robusto por cliente
- features de valor/frequência/horário/dispositivo

### **(4) Score de anomalia global**
- Isolation Forest

E depois:

### **(5) Meta-score final**
Combina os 4.

---

# 16. Fórmula inicial de score final

Exemplo simples:

```text
score_ml = 0.60 * p_fraude_supervisionado
         + 0.25 * score_anomalia_cliente
         + 0.15 * score_anomalia_global
```

Depois normalize `score_regras` entre 0 e 1 e combine:

```text
score_final = 0.75 * score_ml + 0.25 * score_regras_normalizado
```

Se `autorizacao_previa = 1`:
```text
score_final = score_final * 0.5
```

Se `topaz >= 4` e `velocity alta` e `destinatario_novo`:
```text
score_final += booster
```

---

# 17. Como modelar “conta laranja”

Você citou isso como regra, e faz muito sentido.  
Mas precisa virar uma feature objetiva.

## Sinais de conta laranja:
- recebe de muitos CPFs distintos;
- repassa rapidamente;
- saldo médio baixo;
- alta rotatividade;
- muitos créditos seguidos de débito imediato;
- conexões com contas já suspeitas;
- horário incomum;
- padrão de funil.

Você pode criar um **score de mule account** para a conta destino e usar como feature/regras.

---

# 18. Como detectar engenharia social sem depender só do valor

Algumas fraudes não são absurdas em valor.  
Então eu criaria sinais como:

- novo destinatário + horário incomum;
- valor moderado mas muito acima do ticket usual;
- primeira transação relevante do dia e para conta nova;
- sequência “teste pequeno + maior depois”;
- cliente idoso + relacionamento curto + topaz alto;
- mudança súbita na forma de autenticação;
- aumento de frequência sem histórico correspondente.

---

# 19. Roadmap de implementação

## Fase 1 — MVP
- consolidar tabela única por transação;
- criar regras;
- treinar LightGBM;
- criar score de desvio do cliente;
- combinar tudo;
- colocar SHAP;
- rodar em shadow mode.

## Fase 2 — Produção assistida
- fila humana;
- thresholds calibrados;
- monitoramento;
- feedback dos analistas para retreino.

## Fase 3 — Evolução
- graph features de rede pagador-destinatário;
- score de conta laranja;
- modelos de sequência;
- active learning;
- champion/challenger.

---

# 20. Recomendação final objetiva

Se eu estivesse desenhando isso no seu lugar, faria assim:

## Stack analítica recomendada
- **Modelo principal:** LightGBM
- **Anomalia global:** Isolation Forest
- **Anomalia individual:** score estatístico por cliente
- **Regras:** motor de decisão parametrizado
- **Explicabilidade:** SHAP + trilha de regras
- **Decisão:** score final com thresholds em 3 níveis

## Motivo
Porque isso entrega:
- boa performance;
- rapidez para inferência em tempo real;
- interpretabilidade;
- flexibilidade para fraude nova;
- aderência ao contexto bancário.

---

# 21. Exemplo de desenho final do score

```text
Inputs:
- p_fraude_supervisionado
- score_anomalia_cliente
- score_anomalia_global
- score_regras
- atenuante_autorizacao_previa

Combinação:
base = 0.50*p_fraude_supervisionado
     + 0.25*score_anomalia_cliente
     + 0.10*score_anomalia_global
     + 0.15*score_regras

Se autorizacao_previa:
    score_final = 0.5 * base
Senão:
    score_final = base
```

---

# 22. O que eu acho mais promissor nas suas bases

Pelos campos disponíveis, eu apostaria muito em:

- `vl_pix`
- `vl_mediana_pix_trimestre`
- `vl_desvio_padrao_pix_trimestre`
- `qt_intervalo_transacao_minuto`
- `qt_intervalo_mediana_trimestre`
- `qt_intervalo_desvio_padrao_trimestre`
- `qt_pix_dia_maximo_trimestre`
- `qt_tempo_relacionamento_mes`
- `nr_idade`
- `qt_aparelhos_distintos_trimestre`
- `latencia_rede_ms`
- `vl_latencia_rede_media_trimestre`
- `tempo_interacao_ms`
- `vl_tempo_interacao_medio_trimestre`
- `topaz_risk_score`
- `device_name`
- `ip_address`
- `metodo_autenticacao`
- `session_id`

Esses campos já permitem um motor forte sem inventar demais.

---

Se você quiser, eu posso fazer o próximo passo e te entregar uma destas opções:

1. **uma arquitetura técnica end-to-end em diagrama textual**,  
2. **uma lista de features priorizadas para o MVP**,  
3. **um pseudo-código do cálculo do score final**,  
4. **um pipeline em Python/sklearn/LightGBM**,  
5. **uma proposta de apresentação executiva para seu banco**.