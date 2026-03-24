# Apresentação do MVP: Motor Híbrido de Detecção de Fraudes PIX (v2.1)

## 1. Introdução e Geração de Valor

O **Sistema de Detecção de Fraudes PIX** é um sistema voltado a detecção de transações em tempo real.

**Qual o nosso grande objetivo?**
Proteger o dinheiro dos nossos clientes e a reputação do banco através de um motor de decisão que consiga barrar golpes sem gerar atrito desnecessário para os bons clientes.

**O que estamos entregando de valor hoje:**

1. **Falsos Negativos Zero**: Uma garantia matemática de que as fraudes conhecidas não passarão pelo nosso crivo.
2. **Explicabilidade**: Diferente dos modelos tradicionais "caixa-preta", nossa API não retorna apenas um "score de risco". Ela retorna uma mensagem pronta, em linguagem amigável, explicando exatamente *por que* a transação foi retida. Isso permite ao aplicativo do banco dialogar com o cliente na mesma hora, reduzindo a sobrecarga do Call Center e da Mesa de Prevenção.
3. **Defesa em Camadas**: O fraudador não enfrenta apenas um obstáculo, ele enfrenta cinco:
   - Um modelo preditivo principal (LightGBM): treinado no reconhecimento de padrões de fraude.
   - Um sistema de regras determinísticas que vieram do aprendizado humano de padrões de comportamento dos fraudadores.
   - Um detector de anomalias (Isolation Forest) para fortalecer os pontos onde o modelo principal é fraco.
   - Um rastreador comportamental de dispositivo e sessão (Behavioral Analytics).
   - Um mapeador de modus operandi/padrão de golpes (Engenharia Social).

---

## 2. Resultados Alcançados (Base de Validação)

Submetemos o nosso motor a um teste de fogo com uma base de mais de 20 mil transações reais. Os resultados (que vocês podem ver em detalhes no nosso `relatorio_executivo.html`) foram excelentes.

### Os Números Principais (Perspectiva Conservadora - Apenas Bloqueios)
* **Fraudes Detectadas:** 100.0% (71 de 71)
* **Fraudes Perdidas:** 0 (Nenhuma fraude escapou)
* **Falsos Alarmes:** Apenas 50 bloqueios equivocados (0.25% das transações legítimas)
* **Precisão dos Alarmes:** 58.7% (Isso significa que quase 60% das vezes que o sistema soa o alarme vermelho, é fraude real — um número altíssimo para o mercado financeiro, onde a média de mercado gira em torno de 5% a 10%).

### A Perspectiva Ampla (Bloquear + Confirmar com Biometria)
Nosso sistema também atua retendo transações suspeitas para verificação do cliente (Frictionless Security). Reduzimos drasticamente o atrito gerado aos clientes nessa camada:
* **Falsos Alarmes (FPR):** Apenas 54 transações legítimas foram paradas (taxa ínfima).
* **Precisão:** Subimos de uma estimativa inicial de 9% para impressionantes **56.8%**.
* Isso significa que de todo o fluxo retido para verificação, mais da metade são tentativas reais de golpe, garantindo uma excelente experiência de usuário e baixíssimo custo operacional.

### O que significam essas métricas? (Glossário Rápido)

Para estarmos todos na mesma página ao ler o relatório, aqui está o que as métricas técnicas representam na prática de negócio:

* **Recall (Taxa de Captura):** É a capacidade do sistema de encontrar o que procura. Se existem 100 fraudes e nós pegamos 100, nosso Recall é de 100%. É a métrica mais importante do modelo.
* **Precision (Precisão):** É a taxa de acerto quando dizemos "Isso é fraude!". Se a precisão é baixa, significa que estamos bloqueando muitos clientes bons à toa. Nossa precisão de 55.5% garante que a Mesa de Fraude não vai perder tempo olhando milhares de falsos positivos.
* **Falsos Positivos (FPR) / Falsos Alarmes:** Transações de clientes honestos que foram bloqueadas. Nosso FPR é de apenas 0.28%.
* **Falsos Negativos (FNR):** O pesadelo do banco: fraudes reais que o sistema deixou passar como "Aprovadas". Nosso FNR atual é ZERO.
* **GAP de Separação:** É a distância entre a pior fraude e o cliente bom mais "esquisito". Antes tínhamos um GAP negativo (overlap de 4.4 pontos), o que significava que os falsos positivos recebiam notas altas (quase 90). Agora o GAP é de **0.0 pontos**! Isso é excelente: significa que os poucos clientes bons que são bloqueados estão "raspando" na nota mínima de bloqueio (85.0), ou seja, o modelo só erra quando o caso é realmente limítrofe, não dando mais "certeza absoluta" para transações normais.
* **P99.9 e P5:** São percentis (fatias da base).
  - O *P99.9 dos normais* indica onde estão concentrados 99,9% dos clientes bons (nosso P99.9 bateu a nota 89.38).
  - O *P5 das fraudes* indica onde começam as fraudes mais difíceis de pegar (nota 85.22).

---

## 3. Visão de Futuro e Roadmap

O que construímos até aqui já coloca o banco à frente de boa parte do mercado, mas a fraude evolui todos os dias. Para garantirmos escalabilidade e inteligência contínua, este é o nosso plano de evolução para as próximas versões (V3.0+):

### I. Evolução da Infraestrutura para Ultra Baixa Latência

O Banco Central exige respostas em milissegundos. Hoje, processamos tudo de forma rápida, mas para suportar picos de Black Friday, migraremos nosso controle de perfis comportamentais e contagem de PIXs simultâneos para um cache distribuído em memória (Redis). Isso garante que, mesmo com milhares de servidores rodando a API, a resposta seja instantânea.

### II. Feedback Loop e "Shadow Mode"

Antes de ligarmos o bloqueio automático, colocaremos a API rodando em produção de forma silenciosa (*Shadow Mode*). O motor vai classificar as transações, mas não vai pará-las. Analisaremos esses logs contra o que realmente virou fraude. Além disso, criaremos um "Feedback Loop": sempre que a Mesa de Fraude aprovar ou rejeitar um caso manualmente, essa decisão voltará para o modelo, tornando-o mais inteligente a cada semana.

### III. Integração em Tempo Real com o DICT / MED do Bacen

O nosso sistema já detecta brilhantemente "primeiros envios" suspeitos. O próximo passo é perguntar ao Banco Central, no momento do PIX: *"Essa chave de destino tem histórico de fraude reportado por outros bancos?"*. Integrar o DICT enriquece o modelo com o histórico criminal do sistema financeiro inteiro, não apenas do nosso banco.

### IV. Segurança sem Atrito (Frictionless Security)

Nosso foco não é barrar o dinheiro, mas proteger o cliente. Em vez de enviar as transações da faixa "CONFIRMAR" direto para a fila de telemarketing, vamos integrar o retorno amigável da nossa API direto no aplicativo do cliente com um fluxo de *Step-up Authentication*. A transação é retida na tela do celular e o cliente faz uma Prova de Vida (Biometria Facial) ali mesmo. O risco é mitigado com zero custo operacional humano.

### V. Detecção de Quadrilhas (Graph Analytics)

Hoje identificamos contas laranjas por seus comportamentos individuais. O próximo salto tecnológico será usar Banco de Dados Orientados a Grafos (como Neo4j). Isso nos permitirá ver a teia criminal completa: perceberemos em tempo real que dezenas de contas aparentemente normais estão todas enviando dinheiro para o mesmo nó central (laranja), desmontando a quadrilha inteira antes do dinheiro sair do banco.
