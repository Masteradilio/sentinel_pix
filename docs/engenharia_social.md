# Detector de Padrões de Engenharia Social

## 1. Visão Geral

O **Detector de Padrões de Engenharia Social** é um módulo crítico do sistema Anomalia PIX que identifica possíveis vítimas de golpes financeiros através da análise comportamental de transações PIX. O sistema combina múltiplos indicadores que, individualmente, podem parecer normais, mas juntos formam o perfil característico de fraudes de engenharia social.

### Objetivo

Proteger clientes vulneráveis detectando padrões de fraude em tempo real, permitindo intervenção antes que o dano seja consumado.

### Principais Características

- **Detecção em tempo real**: Análise durante a transação
- **Score progressivo**: Quanto mais indicadores combinados, maior o risco
- **Foco em vulneráveis**: Atenção especial a idosos, mulheres idosas e clientes de alto patrimônio
- **Padrões baseados em casos reais**: Golpes documentados no mercado brasileiro

## 2. Indicadores de Risco

Os indicadores são sinais individuais que, quando combinados, revelam padrões de fraude. O sistema possui 40+ indicadores organizados em 6 categorias:

### 2.1 Indicadores de Perfil

Identificam características demográficas e de relacionamento com o banco que tornam o cliente mais vulnerável a golpes.

| Indicador | Descrição | Critério | Justificativa |
|-----------|-----------|----------|---------------|
| `idade_60_plus` | Cliente com 60+ anos | `nr_idade >= 60` | Maior vulnerabilidade a manipulação psicológica |
| `idade_70_plus` | Cliente com 70+ anos | `nr_idade >= 70` | Vulnerabilidade significativamente maior |
| `idade_80_plus` | Cliente com 80+ anos | `nr_idade >= 80` | Grupo de altíssimo risco |
| `mulher_idosa` | Mulher com 60+ anos | `nr_idade >= 60 AND ds_sexo IN ('F', 'FEMININO', 'FEMALE')` | Estatisticamente 2.3x mais vítimas que homens idosos |
| `viuvo_viuva` | Estado civil viúvo | `'VIUV' IN ds_estado_civil` | Alvo preferencial de romance scams |
| `segmento_alto_patrimonio` | Cliente premium | `ds_segmento IN ('EXCLUSIVO', 'PRIVATE', 'MILLENIUM', 'PREMIUM', 'VIP')` | Valores mais altos em risco |
| `cliente_novo` | Conta com até 6 meses | `qt_tempo_relacionamento_mes <= 6` | Possível conta laranja |
| `cliente_muito_novo` | Conta com até 3 meses | `qt_tempo_relacionamento_mes <= 3` | Alta suspeita de conta laranja |
| `conta_recem_aberta` | Conta com até 1 mês | `qt_tempo_relacionamento_mes <= 1` | Muito suspeito para volumes altos |

**Estatísticas:**
- 64% das vítimas de golpes têm 60+ anos (Febraban, 2023)
- Mulheres idosas representam 68% dos casos de romance scam
- Clientes premium são alvos 3x mais frequentes

### 2.2 Indicadores de Horário

Certos golpes ocorrem em horários específicos que fazem parte do modus operandi.

| Indicador | Descrição | Critério | Golpe Associado |
|-----------|-----------|----------|-----------------|
| `horario_noturno` | Noite (22h-6h) | `hour >= 22 OR hour < 6` | Sequestros, coação |
| `horario_madrugada` | Madrugada (0h-5h) | `0 <= hour < 5` | Falso sequestro, desespero |
| `horario_comercial` | Expediente bancário (8h-18h, seg-sex) | `8 <= hour < 18 AND weekday < 5` | **Falso funcionário do banco** |
| `horario_almoco` | Almoço (11h-14h) | `11 <= hour < 14` | Momento de menor atenção |
| `fim_de_semana` | Sábado ou domingo | `weekday >= 5` | Menos suporte disponível |

**Insight Crítico:** 
O golpe do falso funcionário ocorre durante **horário comercial** (não noturno!) porque o criminoso precisa fingir credibilidade de estar "no banco". Ligam dizendo "Aqui é do setor de segurança do banco", exatamente quando o banco está aberto.

### 2.3 Indicadores de Recebedor

Características do destino da transferência.

| Indicador | Descrição | Critério | Risco |
|-----------|-----------|----------|-------|
| `primeiro_envio` | Primeiro PIX para este recebedor | `tp_primeiro_envio_recebedor_trimestre == 1` | ALTO - Destino desconhecido |
| `recebedor_nunca_visto` | Zero envios históricos | `qt_envio_recebedor_trimestre == 0` | MUITO ALTO |
| `recebedor_pj` | Recebedor pessoa jurídica | `len(cd_cpf_cnpj_recebedor) >= 14` | Usado em golpes de investimento |
| `conta_laranja` | Recebedor marcado como laranja | `tp_recebedor_conta_laranja == 1` | CRÍTICO - Conta criminosa |

### 2.4 Indicadores de Tipo de Chave PIX

O tipo de chave revela o relacionamento entre pagador e recebedor.

| Indicador | Descrição | Critério | Interpretação |
|-----------|-----------|----------|---------------|
| `chave_aleatoria` | Chave aleatória (UUID) | `'ALEATORIA' IN ds_tipo_chave` | **Conta controlada por golpista** - não conhece CPF/celular da vítima |
| `chave_celular` | Chave de celular | `'CELULAR' IN ds_tipo_chave` | Comum em golpes com contato telefônico |
| `chave_email` | Chave de email | `'EMAIL' IN ds_tipo_chave` | Comum em romance scams e investimentos |

**Por que chave aleatória é crítica?**
Quando um idoso envia para chave aleatória, significa que ele não conhece nem o CPF nem o celular do recebedor - apenas a chave aleatória que o golpista passou por telefone. Isto é o padrão exato do golpe do falso funcionário.

### 2.5 Indicadores de Valor

Análise do valor da transação em múltiplas dimensões.

| Indicador | Descrição | Critério | Score |
|-----------|-----------|----------|-------|
| `valor_alto` | 30%+ do limite | `vl_razao_pix_limite >= 0.3` | Médio |
| `valor_muito_alto` | 50%+ do limite | `vl_razao_pix_limite >= 0.5` | Alto |
| `valor_critico` | 80%+ do limite | `vl_razao_pix_limite >= 0.8` | Crítico |
| `valor_redondo` | Múltiplo de 100/500 | `vl_pix % 100 == 0` | Típico de golpes (ex: R$ 1.000, R$ 5.000) |
| `valor_absoluto_alto` | Acima de R$ 5.000 | `vl_pix >= 5000` | Alto impacto |
| `valor_absoluto_muito_alto` | Acima de R$ 10.000 | `vl_pix >= 10000` | Impacto crítico |
| `pix_acima_mediana` | 3x a mediana histórica | `vl_pix > (mediana * 3)` | Atípico para o cliente |
| `pix_acima_mediana_3x` | 3x a mediana | `vl_pix > (mediana * 3)` | Mesmo que anterior |
| `pix_muito_acima_mediana` | 5x a mediana histórica | `vl_pix > (mediana * 5)` | Muito atípico |
| `pix_acima_maximo_historico` | Maior que máximo trimestral | `vl_pix > vl_maximo_pix_trimestre` | Recorde pessoal |


### 2.6 Indicadores de Velocity e Frequência

Padrões temporais de transações que indicam urgência, desespero ou automação.

| Indicador | Descrição | Critério | Padrão |
|-----------|-----------|----------|--------|
| `intervalo_curto` | Menos de 30 min desde último PIX | `qt_intervalo_transacao_minuto <= 30` | Velocidade aumentada |
| `intervalo_muito_curto` | Menos de 5 min desde último PIX | `qt_intervalo_transacao_minuto <= 5` | Urgência extrema |
| `intervalo_suspeitissimo` | Menos de 5 min | `qt_intervalo_transacao_minuto <= 5` | Coação física possível |
| `multiplos_pix_rapidos` | 3+ PIX/dia + intervalo < 10min | `qt_pix_dia_maximo >= 3 AND intervalo <= 10` | Esvaziamento de conta |
| `alta_frequencia_diaria` | 5+ PIX em um dia | `qt_pix_dia_maximo_trimestre >= 5` | Atividade atípica |
| `frequencia_anormal` | Máximo > 3x mediana diária | `qt_pix_dia_maximo > (mediana_dia * 3)` | Pico anormal |

### 2.7 Indicadores Compostos

Combinações lógicas de outros indicadores que formam padrões específicos.

| Indicador | Descrição | Lógica | Detecta |
|-----------|-----------|--------|---------|
| `escalada_valores` | Valores crescentes | `vl_pix > vl_anterior > mediana` | Golpe de investimento, romance scam |
| `aproximando_limite` | Perto do limite com velocity | `vl_razao >= 0.7 AND intervalo <= 60` | Esvaziamento planejado |

## 3. Padrões de Fraude Detectados

O sistema detecta 11 padrões distintos de golpes de engenharia social. Cada padrão possui:
- **Indicadores obrigatórios (required)**: Devem TODOS estar presentes
- **Indicadores opcionais (optional)**: Cada um adiciona pontos ao score
- **Score mínimo**: Threshold para ativação do padrão
- **Severidade**: CRÍTICO, ALTO ou MÉDIO

### 3.1 Golpe do Falso Funcionário do Banco

**Nome Técnico:** `FALSO_FUNCIONARIO_BANCO`

#### Modus Operandi

1. **Ligação telefônica**: Criminoso liga para a vítima (geralmente idosa)
2. **Falsa identidade**: "Aqui é do setor de segurança do Banco X"
3. **Criação de urgência**: "Detectamos uma tentativa de fraude na sua conta"
4. **Solução proposta**: "Para proteger seu dinheiro, transfira para uma conta temporária de segurança"
5. **Pressão psicológica**: "Rápido, senão você vai perder tudo!"
6. **Execução**: Vítima faz PIX para chave aleatória controlada pelo golpista

#### Perfil Típico da Vítima

- **Idade**: 60+ anos (64% dos casos)
- **Gênero**: Mulheres idosas são 2.3x mais vítimas
- **Segmento**: Alto patrimônio (EXCLUSIVO, PREMIUM) - valores maiores
- **Tecnologia**: Pouca familiaridade com PIX
- **Psicológico**: Confiança em autoridade bancária

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `chave_aleatoria` - **ESSENCIAL**: A vítima não conhece nem CPF nem celular do golpista

**Opcionais (cada um adiciona 1 ponto):**
- `idade_60_plus` - Alvo preferencial
- `mulher_idosa` - Alvo muito preferencial  
- `horario_comercial` - Fingem ser do banco DURANTE o expediente (8h-18h)
- `valor_alto` - Valores significativos
- `valor_redondo` - Típico: R$ 1.000, R$ 2.000, R$ 5.000
- `primeiro_envio` - Destino desconhecido
- `segmento_alto_patrimonio` - Alvos lucrativos
- `pix_acima_mediana` - Valor atípico para o cliente

#### Exemplo de Cenário Real

```
Cliente: Maria, 68 anos, segmento EXCLUSIVO
Horário: 14:30 (quarta-feira) - HORÁRIO COMERCIAL
Ligação: "Sra. Maria, aqui é Rodrigo do setor de segurança. 
          Detectamos uma compra suspeita de R$ 8.000..."
Ação: PIX de R$ 5.000 para chave aleatória (UUID)
Histórico: Mediana de PIX = R$ 500

Indicadores detectados:
✓ chave_aleatoria (OBRIGATÓRIO)
✓ mulher_idosa (+1)
✓ horario_comercial (+1)
✓ valor_redondo (+1) - R$ 5.000
✓ primeiro_envio (+1)
✓ segmento_alto_patrimonio (+1)
✓ pix_acima_mediana (+1) - 10x a mediana

Score: 2 (required) + 6 (optional) = 8 pontos
Min_score: 5 → PADRÃO DETECTADO
Severidade: CRÍTICO
```

#### Estatísticas

- **Prevalência**: Golpe #1 em volume no Brasil (2023)
- **Valor médio**: R$ 4.200
- **Taxa de recuperação**: Apenas 12%
- **Horário predominante**: 8h-18h (87% dos casos)

#### Como o Banco Deve Responder

1. **Bloqueio imediato** com mensagem: "Atenção! Funcionários do banco NUNCA pedem transferência para 'proteger' seu dinheiro"
2. **Ligação do banco** para o cliente
3. **Educação**: Orientar sobre o modus operandi
4. **Monitoramento**: Cliente em watchlist por 48h

---

### 3.2 Golpe do Falso Sequestro

**Nome Técnico:** `FALSO_SEQUESTRO`

#### Modus Operandi

1. **Ligação noturna/madrugada**: Criminoso liga em horário de pânico
2. **Simulação de desespero**: Barulhos, gritos ao fundo
3. **Ameaça**: "Estou com seu filho/filha, se não pagar agora..."
4. **Urgência extrema**: Múltiplos PIX em sequência rápida
5. **Valores crescentes**: Começam com um valor, pedem mais

#### Perfil Típico da Vítima

- **Idade**: Qualquer idade (pais de adolescentes/jovens)
- **Estado emocional**: Pânico, desespero
- **Horário**: Madrugada (2h-5h) - momento de maior vulnerabilidade
- **Comportamento**: Múltiplos PIX sem parar para pensar

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `horario_noturno` - Golpe ocorre à noite (22h-6h)
- ✅ `valor_muito_alto` - Valores significativos (50%+ do limite)

**Opcionais:**
- `horario_madrugada` - Pior momento (0h-5h)
- `multiplos_pix_rapidos` - Sequência em pânico
- `intervalo_muito_curto` - Menos de 5 minutos entre PIX
- `chave_aleatoria` - Não conhece o recebedor
- `chave_celular` - Recebeu por telefone
- `primeiro_envio` - Destino desconhecido
- `aproximando_limite` - Esvaziando a conta

#### Exemplo de Cenário Real

```
Cliente: João, 45 anos
Horário: 03:20 (madrugada)
Sequência:
  03:20 - PIX R$ 2.000 (chave celular)
  03:23 - PIX R$ 3.000 (chave celular)
  03:26 - PIX R$ 2.500 (chave aleatória)
Total: R$ 7.500 em 6 minutos

Indicadores detectados:
✓ horario_noturno (OBRIGATÓRIO)
✓ valor_muito_alto (OBRIGATÓRIO) - 75% do limite
✓ horario_madrugada (+1)
✓ multiplos_pix_rapidos (+1)
✓ intervalo_muito_curto (+1)
✓ chave_aleatoria (+1)
✓ primeiro_envio (+1)
✓ aproximando_limite (+1)

Score: 4 (required) + 6 (optional) = 10 pontos
Min_score: 6 → PADRÃO DETECTADO
Severidade: CRÍTICO
```

#### Estatísticas

- **Horário predominante**: 85% entre 23h-5h
- **Duração média**: 15 minutos de desespero
- **Valor médio**: R$ 6.800
- **Modus**: Geralmente são 2-4 PIX seguidos

#### Como o Banco Deve Responder

1. **Bloqueio após 2º PIX** com mensagem forte
2. **SMS imediato**: "Se alguém está pedindo dinheiro alegando sequestro, LIGUE 190 PRIMEIRO"
3. **Delay forçado**: 5 minutos entre transações
4. **Verificação biométrica** adicional

---

### 3.3 Esvaziamento de Conta (Account Takeover ou Coação)

**Nome Técnico:** `ESVAZIAMENTO_CONTA`

#### Modus Operandi

Duas variantes principais:

**A) Account Takeover:**
1. Criminoso obtém acesso à conta (phishing, malware)
2. Realiza múltiplos PIX rápidos para esvaziar
3. Destinos: Contas laranjas controladas pelo criminoso

**B) Coação Física:**
1. Vítima é abordada fisicamente (sequestro relâmpago)
2. Forçada a fazer PIX sob ameaça
3. Múltiplos PIX até próximo do limite

#### Perfil Típico

- **Account Takeover**: Cliente distraído, clicou em link falso
- **Coação**: Qualquer cliente (crime oportunista)
- **Padrão**: Múltiplos PIX em sequência muito rápida
- **Destinos**: Geralmente múltiplos destinos diferentes

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `multiplos_pix_rapidos` - 3+ PIX com intervalo < 10min
- ✅ `aproximando_limite` - 70%+ do limite em < 60min

**Opcionais:**
- `intervalo_suspeitissimo` - Menos de 5 minutos
- `valor_critico` - 80%+ do limite
- `primeiro_envio` - Destinos desconhecidos
- `chave_aleatoria` - Contas laranjas
- `horario_noturno` - Coação mais provável
- `escalada_valores` - Valores crescentes

#### Exemplo de Cenário Real

```
Cliente: Carlos, 52 anos, limite R$ 10.000
Horário: 22:45
Sequência:
  22:45 - PIX R$ 1.500 (chave aleatória A)
  22:48 - PIX R$ 2.000 (chave aleatória B)
  22:51 - PIX R$ 2.500 (chave aleatória C)
  22:53 - PIX R$ 2.000 (chave aleatória D)
Total: R$ 8.000 em 8 minutos (80% do limite)

Indicadores detectados:
✓ multiplos_pix_rapidos (OBRIGATÓRIO)
✓ aproximando_limite (OBRIGATÓRIO) - 80% em 8 min
✓ intervalo_suspeitissimo (+1) - PIX a cada 2-3 min
✓ valor_critico (+1) - 80% do limite
✓ primeiro_envio (+1) - 4 destinos novos
✓ chave_aleatoria (+1) - Todas as 4 chaves
✓ horario_noturno (+1)
✓ escalada_valores (+1)

Score: 4 (required) + 6 (optional) = 10 pontos
Min_score: 6 → PADRÃO DETECTADO
Severidade: CRÍTICO
```

#### Estatísticas

- **Duração média**: 5-15 minutos
- **Número de PIX**: 3-6 transações
- **Recuperação**: 8% dos valores (difícil rastrear)

#### Como o Banco Deve Responder

1. **Bloqueio imediato** após 3º PIX rápido
2. **Verificação por vídeo chamada** (confirmar que não está sob coação)
3. **Delay progressivo**: Cada PIX adicional requer mais tempo
4. **Alerta imediato** à equipe de fraude

---


### 3.4 Golpe do PIX Errado

**Nome Técnico:** `GOLPE_PIX_ERRADO`

#### Modus Operandi

1. **Primeiro ato**: Criminoso envia PIX de baixo valor para a vítima
2. **Contato**: Liga ou manda mensagem: "Oi, fiz um PIX errado para você por engano"
3. **Pedido de devolução**: "Pode devolver? Mas minha chave mudou, usa essa aqui..."
4. **Nova chave**: Fornece chave aleatória diferente (a conta do golpista)
5. **Valor aumentado**: Vítima pode devolver valor maior que recebeu

#### Perfil Típico da Vítima

- **Perfil**: Pessoas solidárias, querem ajudar
- **Idade**: Qualquer idade (mas idosos mais vulneráveis)
- **Comportamento**: Boa fé, não questiona

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `primeiro_envio` - Primeiro contato com este recebedor
- ✅ `chave_aleatoria` - Chave "nova" fornecida

**Opcionais:**
- `valor_redondo` - Valor suspeito (R$ 100, R$ 200)
- `intervalo_curto` - Logo após receber o PIX
- `horario_comercial` - Golpe durante o dia
- `pix_acima_mediana_3x` - Devolve mais que recebeu

#### Exemplo de Cenário Real

```
Vítima recebe: R$ 50 (10h da manhã)
Mensagem: "Oi, fiz PIX errado pra vc. Pode devolver?"
Vítima: "Claro! Qual sua chave?"
Golpista: "Mudei de banco, agora é essa chave aleatória..."

10h15 - PIX R$ 100 para chave aleatória
(Vítima devolveu o dobro!)

Indicadores detectados:
✓ primeiro_envio (OBRIGATÓRIO)
✓ chave_aleatoria (OBRIGATÓRIO)
✓ valor_redondo (+1) - R$ 100
✓ intervalo_curto (+1) - 15 minutos depois
✓ horario_comercial (+1)

Score: 4 (required) + 3 (optional) = 7 pontos
Min_score: 4 → PADRÃO DETECTADO
Severidade: ALTO
```

#### Estatísticas

- **Valor típico solicitado**: R$ 50-200
- **Valor devolvido médio**: R$ 150-500 (sempre mais)
- **Taxa de sucesso do golpe**: 35%

#### Como o Banco Deve Responder

1. **Alerta educacional**: "Cuidado! Devoluções devem usar a mesma chave do remetente"
2. **Confirmar**: "Você recebeu um PIX desta pessoa recentemente?"
3. **Sugerir**: "Use a mesma chave que enviou para você"

---

### 3.5 Romance Scam / Golpe do Amor

**Nome Técnico:** `ROMANCE_SCAM`

#### Modus Operandi

1. **Conhecimento online**: Redes sociais, apps de relacionamento
2. **Construção de relacionamento**: Semanas/meses de conversas
3. **Criação de vínculo emocional**: Declarações de amor
4. **Emergência fabricada**: "Preciso de dinheiro urgente para..."
   - Passagem de avião para se encontrarem
   - Tratamento médico
   - Problema no trabalho
5. **Pedido de dinheiro**: Primeiro pequeno, depois valores crescentes
6. **Escalada**: Pedidos cada vez maiores

#### Perfil Típico da Vítima

- **Idade**: 60+ anos
- **Estado civil**: Viúvo(a), divorciado(a), solteiro(a)
- **Gênero**: 68% são mulheres idosas
- **Solidão**: Isolamento social
- **Esperança**: Busca por companhe

iro

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `primeiro_envio` - Primeiro PIX para esta "pessoa amada"
- ✅ `valor_alto` - Valores significativos

**Opcionais:**
- `idade_60_plus` - Alvo preferencial
- `viuvo_viuva` - Grupo de altíssimo risco
- `mulher_idosa` - Perfil #1 de vítimas
- `chave_celular` - Contato por WhatsApp
- `chave_email` - Contato por email/redes sociais
- `pix_muito_acima_mediana` - Valor atípico
- `fim_de_semana` - Pedidos em momentos de solidão

#### Exemplo de Cenário Real

```
Cliente: Helena, 67 anos, viúva
Histórico: 2 meses de conversas por WhatsApp com "Roberto"
Roberto: "Diz que é empresário, trabalha em plataforma de petróleo"
Emergência: "Amor, tive um acidente, preciso pagar hospital urgente"

Sábado 15h - PIX R$ 3.000 (chave celular)
Mediana histórica: R$ 250

Indicadores detectados:
✓ primeiro_envio (OBRIGATÓRIO) - Nunca enviou para ele
✓ valor_alto (OBRIGATÓRIO) - 60% do limite
✓ mulher_idosa (+1)
✓ viuvo_viuva (+1) - Alto risco
✓ idade_60_plus (+1)
✓ chave_celular (+1)
✓ pix_muito_acima_mediana (+1) - 12x a mediana
✓ fim_de_semana (+1)

Score: 4 (required) + 6 (optional) = 10 pontos
Min_score: 5 → PADRÃO DETECTADO
Severidade: ALTO
```

#### Estatísticas

- **Duração média do golpe**: 45 dias (construção de relacionamento)
- **Valor médio total**: R$ 8.400 (múltiplas transferências)
- **Perfil predominante**: Mulheres viúvas 60-75 anos (68%)
- **Taxa de recuperação**: < 5% (geralmente fora do Brasil)

#### Como o Banco Deve Responder

1. **Alerta educacional forte**: "Cuidado com relacionamentos online que pedem dinheiro"
2. **Perguntas**: "Você conhece esta pessoa pessoalmente?"
3. **Delay**: 24h para valores altos em primeiro envio
4. **Material educativo**: Enviar cartilha sobre romance scam

---

### 3.6 Idoso Vulnerável (70+)

**Nome Técnico:** `IDOSO_VULNERAVEL_70`

#### Modus Operandi

Não é um golpe específico, mas um **padrão de vulnerabilidade geral**. Clientes 70+ fazendo PIX para destinos desconhecidos têm alta probabilidade de estarem sendo vítimas de algum tipo de golpe.

#### Perfil Típico

- **Idade**: 70-79 anos
- **Vulnerabilidade**: Cognitiva, tecnológica, social
- **Comportamento**: Confia em autoridade, dificuldade em dizer não

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `idade_70_plus` - Cliente com 70+ anos
- ✅ `primeiro_envio` - Destino desconhecido

**Opcionais:**
- `valor_alto` - Valor significativo
- `chave_aleatoria` - Não conhece o recebedor
- `horario_comercial` - Golpe do falso funcionário
- `valor_redondo` - Valores típicos de golpes
- `mulher_idosa` - Vulnerabilidade adicional
- `segmento_alto_patrimonio` - Alvo lucrativo

#### Severidade e Score

- **Score mínimo**: 5 pontos
- **Severidade**: CRÍTICO
- **Ação**: Sempre revisar manualmente

---

### 3.7 Idoso Vulnerável (80+)

**Nome Técnico:** `IDOSO_VULNERAVEL_80`

#### Modus Operandi

Clientes 80+ são considerados de **risco crítico automático**. Qualquer PIX com sinais adicionais de risco merece atenção imediata.

#### Perfil Típico

- **Idade**: 80+ anos
- **Vulnerabilidade**: MUITO ALTA
- **Estatística**: 78% dos clientes 80+ vítimas de golpes não recuperam o valor

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `idade_80_plus` - Cliente com 80+ anos (suficiente sozinho!)

**Opcionais:**
- `valor_alto` - Qualquer valor alto
- `primeiro_envio` - Destino desconhecido
- `chave_aleatoria` - Aumenta ainda mais o risco

#### Severidade e Score

- **Score mínimo**: 3 pontos (mais baixo!)
- **Severidade**: CRÍTICO
- **Ação**: Intervir sempre, sem exceção

---

### 3.8 Conta Laranja (Money Mule)

**Nome Técnico:** `CONTA_LARANJA_SAIDA`

#### Modus Operandi

Criminosos aliciam pessoas vulneráveis (desempregados, estudantes) para:
1. Abrir conta bancária nova
2. Receber PIX de vítimas de golpes
3. Sacar o dinheiro ou transferir para outros
4. Recebem "comissão" de 10-20%

A conta nova movimenta valores altíssimos de forma atípica.

#### Perfil Típico do "Laranja"

- **Situação**: Desempregado, necessidade financeira
- **Conta**: Recém aberta (1-3 meses)
- **Proposta**: "Trabalho fácil, só emprestar sua conta"
- **Crime**: Participação em organização criminosa (não sabem a gravidade)

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `conta_recem_aberta` - Conta com até 1 mês
- ✅ `valor_muito_alto` - Volume incompatível com conta nova

**Opcionais:**
- `multiplos_pix_rapidos` - Sequência rápida
- `alta_frequencia_diaria` - 5+ PIX por dia
- `chave_aleatoria` - Múltiplos destinos
- `primeiro_envio` - Destinos sempre novos
- `frequencia_anormal` - Muito acima do esperado

#### Exemplo de Cenário Real

```
Cliente: Ricardo, 22 anos
Conta: Aberta há 15 dias
Histórico: Sem movimentação até ontem
Hoje:
  10h - PIX saída R$ 2.000
  10h05 - PIX saída R$ 3.500
  10h12 - PIX saída R$ 1.800
  14h - PIX saída R$ 2.700
Total dia: R$ 10.000 em conta de 15 dias

Indicadores detectados:
✓ conta_recem_aberta (OBRIGATÓRIO)
✓ valor_muito_alto (OBRIGATÓRIO)
✓ multiplos_pix_rapidos (+1)
✓ alta_frequencia_diaria (+1) - 4 PIX
✓ chave_aleatoria (+1) - Múltiplos destinos
✓ primeiro_envio (+1)
✓ frequencia_anormal (+1)

Score: 4 (required) + 5 (optional) = 9 pontos
Min_score: 5 → PADRÃO DETECTADO
Severidade: CRÍTICO
```

#### Estatísticas

- **Idade média do laranja**: 18-28 anos
- **Comissão típica**: 15% do valor
- **Pena criminal**: 3-8 anos de prisão
- **Recuperação para vítimas**: < 10%

#### Como o Banco Deve Responder

1. **Bloqueio imediato**
2. **Investigação**: Analisar últimas 48h
3. **Notificação**: BC e autoridades
4. **Educação**: Explicar ao titular a gravidade (podem ser vítimas também)

---

### 3.9 Golpe de Investimento Falso

**Nome Técnico:** `GOLPE_INVESTIMENTO`

#### Modus Operandi

1. **Contato inicial**: Anúncio em rede social, WhatsApp
2. **Proposta**: "Investimento em criptomoedas/forex com retorno garantido"
3. **Prova social falsa**: Prints de lucros, testemunhos
4. **Primeiro investimento**: Valor pequeno
5. **Retorno simulado**: Mostram "lucro" para ganhar confiança
6. **Escalada**: Pedem valores cada vez maiores
7. **Desaparecimento**: Quando valor total é alto, somem

#### Perfil Típico da Vítima

- **Idade**: 40-65 anos
- **Perfil**: Quer multiplicar patrimônio
- **Conhecimento financeiro**: Baixo/médio
- **Comportamento**: Ganância supera cautela

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `escalada_valores` - Valores crescentes ao longo do tempo
- ✅ `valor_alto` - Investimento significativo

**Opcionais:**
- `primeiro_envio` - Novo "corretor"
- `chave_aleatoria` - Conta empresarial fake
- `recebedor_pj` - Empresa de fachada
- `pix_acima_maximo_historico` - Recorde pessoal

#### Exemplo de Cenário Real

```
Cliente: Paulo, 52 anos
Histórico de PIX para "Invest Pro LTDA":
  Semana 1: R$ 500 (teste)
  Semana 2: R$ 1.000 ("viu lucro" de 20%)
  Semana 3: R$ 2.500 (confiança aumentou)
  Hoje: R$ 5.000 (chave PJ)

Mediana histórica: R$ 800

Indicadores detectados:
✓ escalada_valores (OBRIGATÓRIO) - 500 → 1000 → 2500 → 5000
✓ valor_alto (OBRIGATÓRIO)
✓ primeiro_envio (+1) - Empresa nova
✓ recebedor_pj (+1)
✓ pix_acima_maximo_historico (+1) - R$ 5.000 é recorde
✓ chave_aleatoria (+1)

Score: 4 (required) + 4 (optional) = 8 pontos
Min_score: 5 → PADRÃO DETECTADO
Severidade: ALTO
```

#### Estatísticas

- **Duração média**: 3-6 semanas
- **Valor total médio**: R$ 12.000
- **Taxa de recuperação**: < 3%
- **Crescimento**: +340% em 2023

#### Como o Banco Deve Responder

1. **Alerta**: "Investimento com retorno garantido não existe"
2. **Verificar**: "Esta empresa está registrada na CVM?"
3. **Educar**: Material sobre pirâmides e esquemas Ponzi
4. **Delay**: 48h para valores em escalada

---

### 3.10 Coação Física (Sequestro Relâmpago)

**Nome Técnico:** `COACAO_FISICA`

#### Modus Operandi

1. **Abordagem física**: Vítima é sequestrada (geralmente ao sair do banco/caixa)
2. **Ameaça de violência**: Arma, agressão
3. **Exigência**: Fazer múltiplos PIX até o limite
4. **Urgência extrema**: Menos de 5 minutos entre PIX
5. **Liberação**: Após esvaziar a conta

#### Perfil Típico da Vítima

- **Qualquer perfil**: Crime oportunista
- **Local**: Próximo a bancos, caixas eletrônicos
- **Horário**: Noite (mais comum)

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `intervalo_suspeitissimo` - Menos de 5 minutos entre PIX
- ✅ `valor_critico` - 80%+ do limite

**Opcionais:**
- `horario_noturno` - Mais provável
- `horario_madrugada` - Muito provável
- `multiplos_pix_rapidos` - Sequência sob pressão
- `aproximando_limite` - Esvaziando tudo
- `chave_aleatoria` - Contas dos criminosos

#### Exemplo de Cenário Real

```
Cliente: Ana, 38 anos, limite R$ 5.000
Horário: 21:45 (saindo do supermercado)

Sequência:
  21:47 - PIX R$ 2.000 (chave aleatória)
  21:50 - PIX R$ 1.500 (chave aleatória)
  21:53 - PIX R$ 1.000 (chave aleatória)
Total: R$ 4.500 em 6 minutos (90% do limite)

Indicadores detectados:
✓ intervalo_suspeitissimo (OBRIGATÓRIO) - 3 minutos
✓ valor_critico (OBRIGATÓRIO) - 90% do limite
✓ horario_noturno (+1)
✓ multiplos_pix_rapidos (+1)
✓ aproximando_limite (+1)
✓ chave_aleatoria (+1)

Score: 4 (required) + 4 (optional) = 8 pontos
Min_score: 6 → PADRÃO DETECTADO
Severidade: CRÍTICO
```

#### Estatísticas

- **Duração média**: 5-10 minutos
- **Valor médio**: 85% do limite disponível
- **Horário predominante**: 70% entre 19h-23h

#### Como o Banco Deve Responder

1. **AÇÃO IMEDIATA**: Ligar para 190 automaticamente
2. **Bloqueio**: Após 2º PIX com intervalo < 5min
3. **Protocolo de segurança**: Perguntas que só a vítima sabe (sem coação)
4. **Localização**: Usar GPS do app

---

### 3.11 Transação Atípica Genérica

**Nome Técnico:** `TRANSACAO_ATIPICA`

#### Modus Operandi

Este é um padrão **catch-all** (pega-tudo) para transações que não se encaixam perfeitamente nos padrões específicos, mas ainda assim são suspeitas.

#### Indicadores Utilizados

**Obrigatórios:**
- ✅ `pix_acima_maximo_historico` - Valor maior que o histórico máximo
- ✅ `primeiro_envio` - Destino desconhecido

**Opcionais:**
- `chave_aleatoria` - Não conhece o recebedor
- `horario_noturno` - Horário suspeito
- `valor_muito_alto` - Valor significativo

#### Severidade e Score

- **Score mínimo**: 4 pontos
- **Severidade**: MÉDIO
- **Ação**: Alerta simples

---

## 4. Fluxo de Detecção

### 4.1 Processamento de Transação

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. RECEBE TRANSAÇÃO PIX                                         │
│    features = {nr_idade, vl_pix, ds_tipo_chave, ...}            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. AVALIA TODOS OS INDICADORES (40+)                            │
│    active_indicators = {                                        │
│      "idade_60_plus": True,                                     │
│      "chave_aleatoria": True,                                   │
│      "valor_alto": True,                                        │
│      ...                                                        │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. VERIFICA CADA PADRÃO (11 padrões)                            │
│    Para cada padrão:                                            │
│    a) Verificar REQUIRED (todos devem estar presentes)          │
│    b) Contar OPTIONAL (cada um adiciona pontos)                 │
│    c) Se score >= min_score → PADRÃO DETECTADO                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. ORDENA PADRÕES POR SEVERIDADE                                │
│    Ordem: CRITICO > ALTO > MEDIO                                │
│    Dentro da mesma severidade: maior score primeiro             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. RETORNA LISTA DE PADRÕES DETECTADOS                          │
│    [                                                            │
│      PatternMatch(                                              │
│        pattern_name="FALSO_FUNCIONARIO_BANCO",                  │
│        severity="CRITICO",                                      │
│        score=8,                                                 │
│        matched_indicators=[...],                                │
│        description="..."                                        │
│      ),                                                         │
│      ...                                                        │
│    ]                                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Sistema de Pontuação

#### Indicadores Obrigatórios (Required)
- **Peso**: 2 pontos cada
- **Lógica**: TODOS devem estar presentes
- **Se algum faltar**: Padrão não é detectado

#### Indicadores Opcionais (Optional)
- **Peso**: 1 ponto cada
- **Lógica**: Somados após os required
- **Finalidade**: Aumentar confiança na detecção

#### Score Mínimo (min_score)
- Threshold para ativar o padrão
- Varia de 3 (idoso 80+) a 6 (coação física)

#### Exemplo de Cálculo

```python
Padrão: FALSO_FUNCIONARIO_BANCO
Required: ["chave_aleatoria"]
Optional: ["idade_60_plus", "mulher_idosa", "horario_comercial", 
           "valor_alto", "valor_redondo"]
Min_score: 5

Transação:
✓ chave_aleatoria → +2 (required)
✓ idade_60_plus → +1 (optional)
✓ mulher_idosa → +1 (optional)
✓ horario_comercial → +1 (optional)
✓ valor_redondo → +1 (optional)
✗ valor_alto → +0 (não ativo)

Score total: 2 + 4 = 6
6 >= 5 → PADRÃO DETECTADO ✓
```

### 4.3 Integração com Decision Engine

O `SocialEngineeringDetector` é chamado pelo `PixDecisionEngine`:

```python
# No Decision Engine
se_detector = SocialEngineeringDetector()
patterns = se_detector.detect_patterns(features)

if patterns:
    # Adicionar agravantes
    for pattern in patterns:
        if pattern.severity == "CRITICO":
            score += 40  # Agravante forte
        elif pattern.severity == "ALTO":
            score += 25
        # ...
```

---

## 5. Estatísticas e Referências

### 5.1 Dados do Mercado Brasileiro

#### Volume de Fraudes PIX (2023)

| Tipo de Golpe | % do Total | Valor Médio | Taxa Recuperação |
|---------------|------------|-------------|------------------|
| Falso Funcionário | 32% | R$ 4.200 | 12% |
| Falso Sequestro | 18% | R$ 6.800 | 8% |
| Romance Scam | 15% | R$ 8.400 | 5% |
| PIX Errado | 12% | R$ 280 | 45% |
| Investimento Falso | 11% | R$ 12.000 | 3% |
| Outros | 12% | R$ 3.500 | 15% |

**Fonte**: Febraban, Banco Central, relatórios de instituições financeiras

#### Perfil das Vítimas

| Característica | Estatística |
|----------------|-------------|
| Idade 60+ | 64% das vítimas |
| Mulheres 60+ | 68% dos romance scams |
| Homens 40-60 | 72% dos investimentos falsos |
| Horário 8h-18h | 61% dos golpes (exceto sequestro) |
| Horário 22h-6h | 85% dos sequestros |
| Primeiro envio | 89% das fraudes |
| Chave aleatória | 76% das fraudes |

### 5.2 Impacto Financeiro

#### Brasil (2023)
- **Volume total**: R$ 2,1 bilhões em fraudes PIX
- **Crescimento**: +156% vs 2022
- **Vítimas**: 4,3 milhões de pessoas
- **Taxa média de recuperação**: 18%

#### Por Faixa Etária
- **18-39 anos**: R$ 1.200 médio (maior volume, menor valor)
- **40-59 anos**: R$ 4.500 médio
- **60-79 anos**: R$ 6.200 médio
- **80+ anos**: R$ 8.100 médio

### 5.3 Referências Técnicas

1. **Febraban** (2023). "Relatório Anual de Fraudes Bancárias"
2. **Banco Central do Brasil** (2023). "Estatísticas do Sistema de Pagamentos Brasileiro"
3. **Polícia Federal** (2023). "Operações de Combate a Fraudes Bancárias"
4. **Artigos Acadêmicos**:
   - Silva et al. (2023). "Social Engineering Detection in Brazilian Financial System"
   - Costa & Oliveira (2023). "Elderly Vulnerability to Financial Fraud"

---

## 6. Limitações Conhecidas

### 6.1 Limitações Técnicas

#### Falsos Positivos

**Cenários legítimos que podem acionar alertas:**

1. **Compra de veículo/imóvel**
   - Valor alto + primeiro envio + chave aleatória
   - **Mitigação**: Contextualizar com histórico de pesquisas, conversas com vendedor

2. **Presente para familiar distante**
   - Idoso enviando para destino novo
   - **Mitigação**: Perguntar "É para um familiar?"

3. **Pagamento de serviço novo**
   - Primeiro envio + valor alto
   - **Mitigação**: Verificar se há contrato, nota fiscal

#### Falsos Negativos

**Golpes que podem não ser detectados:**

1. **Relacionamento pré-existente**
   - Golpista que já recebeu PIX antes (não é primeiro_envio)
   - **Exemplo**: Esquema Ponzi com histórico

2. **Valores baixos**
   - Golpes de baixo valor não acionam indicadores de valor_alto
   - **Exemplo**: PIX errado de R$ 50

3. **Perfil atípico**
   - Jovens também podem ser vítimas
   - **Exemplo**: Jovem vítima de falso funcionário

### 6.2 Limitações de Dados

#### Features Não Disponíveis (mas desejáveis)

| Feature Desejada | Uso Potencial | Status |
|------------------|---------------|--------|
| Histórico de ligações | Correlacionar com golpe telefônico | ❌ Não disponível |
| Localização GPS | Detectar coação física (local incomum) | ❌ Não disponível |
| Padrão de digitação | Detectar se outra pessoa está usando o app | ❌ Não disponível |
| Análise de mensagens | Detectar padrões de golpe em conversas | ❌ Privacidade |
| Biometria comportamental | Detectar uso sob coação | 🔄 Em desenvolvimento |

#### Features Subutilizadas

Algumas features disponíveis ainda não estão totalmente exploradas:
- `ds_tipo_servico` - Tipo de serviço da conta
- `qt_intervalo_minimo_trimestre` - Padrão de intervalos mínimos
- Correlação com transações de entrada (recebimentos antes de envios)

### 6.3 Desafios Operacionais

1. **Fricção vs Segurança**
   - Alertas demais → clientes frustrados
   - Alertas de menos → fraudes não detectadas
   - **Solução**: Calibrar thresholds com dados reais

2. **Evolução dos Golpes**
   - Criminosos adaptam técnicas
   - Padrões precisam ser atualizados constantemente
   - **Solução**: Revisão trimestral dos padrões

3. **Educação do Cliente**
   - Alertas só são eficazes se o cliente entende
   - Muitos ignoram avisos
   - **Solução**: Mensagens claras e educacionais

---

## 7. Exemplos de Uso

### 7.1 Uso Básico

```python
from app.core.social_engineering import SocialEngineeringDetector

# Inicializar detector
detector = SocialEngineeringDetector()

# Features da transação
features = {
    "nr_idade": 68,
    "ds_sexo": "F",
    "vl_pix": 5000,
    "ds_tipo_chave": "ALEATORIA",
    "dt_pix": "2024-01-15T14:30:00",
    "tp_primeiro_envio_recebedor_trimestre": 1,
    "vl_razao_pix_limite": 0.6,
    "vl_mediana_pix_trimestre": 500,
    # ... outras features
}

# Detectar padrões
patterns = detector.detect_patterns(features)

# Analisar resultados
for pattern in patterns:
    print(f"Padrão: {pattern.pattern_name}")
    print(f"Severidade: {pattern.severity}")
    print(f"Score: {pattern.score}")
    print(f"Descrição: {pattern.description}")
    print(f"Indicadores: {pattern.matched_indicators}")
    print("-" * 50)
```

**Output:**
```
Padrão: FALSO_FUNCIONARIO_BANCO
Severidade: CRITICO
Score: 8
Descrição: Padrão de golpe do falso funcionário do banco...
Indicadores: ['chave_aleatoria', 'mulher_idosa', 'horario_comercial', 
              'valor_alto', 'primeiro_envio', 'pix_acima_mediana']
--------------------------------------------------
```

### 7.2 Obter Padrão Mais Grave

```python
# Apenas o padrão mais grave
worst_pattern = detector.get_worst_pattern(features)

if worst_pattern:
    if worst_pattern.severity == "CRITICO":
        # Bloquear transação
        return {
            "decisao": "BLOQUEADA",
            "motivo": worst_pattern.description,
            "pattern": worst_pattern.pattern_name
        }
```

### 7.3 Calcular Score de Engenharia Social

```python
# Score 0-100 para integração com outros scores
se_score, patterns = detector.calculate_social_engineering_score(features)

print(f"Score Engenharia Social: {se_score}%")
# Output: Score Engenharia Social: 40%

# Integrar com score geral
score_final = (
    score_ml * 0.4 +
    se_score * 0.3 +
    score_regras * 0.3
)
```

### 7.4 Integração com Decision Engine

```python
# No PixDecisionEngine
from app.core.social_engineering import SocialEngineeringDetector

class PixDecisionEngine:
    def __init__(self):
        self.se_detector = SocialEngineeringDetector()
        # ... outros componentes
    
    def evaluate(self, features):
        # ... outros scores
        
        # Detectar padrões de engenharia social
        patterns = self.se_detector.detect_patterns(features)
        
        # Adicionar como agravantes
        agravantes = []
        for pattern in patterns:
            peso = 0
            if pattern.severity == "CRITICO":
                peso = 40
            elif pattern.severity == "ALTO":
                peso = 25
            elif pattern.severity == "MEDIO":
                peso = 15
            
            agravantes.append({
                "codigo": f"SOCIAL_ENG_{pattern.pattern_name}",
                "descricao": pattern.description,
                "peso": peso,
                "matched_indicators": pattern.matched_indicators
            })
        
        return {
            "social_engineering_patterns": patterns,
            "agravantes": agravantes,
            # ... resto da decisão
        }
```

### 7.5 Teste Unitário Exemplo

```python
import pytest
from datetime import datetime
from app.core.social_engineering import SocialEngineeringDetector

def test_falso_funcionario_banco():
    """Testa detecção do golpe do falso funcionário."""
    detector = SocialEngineeringDetector()
    
    features = {
        "nr_idade": 72,
        "ds_sexo": "F",
        "vl_pix": 3000,
        "ds_tipo_chave": "ALEATORIA",
        "dt_pix": datetime(2024, 1, 15, 14, 30),  # Segunda 14:30
        "tp_primeiro_envio_recebedor_trimestre": 1,
        "vl_razao_pix_limite": 0.5,
        "vl_mediana_pix_trimestre": 300,
        "ds_segmento": "EXCLUSIVO"
    }
    
    patterns = detector.detect_patterns(features)
    
    # Deve detectar o padrão
    assert len(patterns) > 0
    assert patterns[0].pattern_name == "FALSO_FUNCIONARIO_BANCO"
    assert patterns[0].severity == "CRITICO"
    assert patterns[0].score >= 5

def test_transacao_legitima():
    """Testa que transação legítima não aciona falsos positivos."""
    detector = SocialEngineeringDetector()
    
    features = {
        "nr_idade": 35,
        "ds_sexo": "M",
        "vl_pix": 150,
        "ds_tipo_chave": "CPF",
        "dt_pix": datetime(2024, 1, 15, 14, 30),
        "tp_primeiro_envio_recebedor_trimestre": 0,  # Recebedor conhecido
        "vl_razao_pix_limite": 0.05,
        "vl_mediana_pix_trimestre": 200,
    }
    
    patterns = detector.detect_patterns(features)
    
    # Não deve detectar nenhum padrão crítico
    critical_patterns = [p for p in patterns if p.severity == "CRITICO"]
    assert len(critical_patterns) == 0
```

---

## 8. Manutenção e Evolução

### 8.1 Quando Adicionar Novo Padrão

**Critérios para adicionar novo padrão:**

1. **Volume significativo**: Pelo menos 50 casos/mês
2. **Padrão claro**: Indicadores consistentes em 70%+ dos casos
3. **Diferencial**: Não coberto pelos padrões existentes
4. **Valor médio**: Prejuízo médio > R$ 500

**Processo:**
1. Coletar casos reais
2. Identificar indicadores comuns
3. Definir required e optional
4. Calibrar min_score (começar conservador)
5. Testar com dados históricos
6. Monitorar por 30 dias
7. Ajustar thresholds

### 8.2 Quando Atualizar Padrão Existente

**Sinais de que um padrão precisa atualização:**

- Taxa de falsos positivos > 30%
- Taxa de falsos negativos > 20%
- Mudança no modus operandi reportada pela Polícia
- Feedback de analistas de fraude

### 8.3 Monitoramento Contínuo

**Métricas a acompanhar:**

| Métrica | Meta | Ação se Fora da Meta |
|---------|------|----------------------|
| Taxa de detecção | > 75% | Revisar indicadores required |
| Falsos positivos | < 25% | Aumentar min_score |
| Tempo de resposta | < 500ms | Otimizar código |
| Cobertura de casos | > 80% | Adicionar novos padrões |

---

## 9. Glossário

| Termo | Definição |
|-------|-----------|
| **Chave Aleatória** | Chave PIX do tipo UUID, indica que pagador não conhece CPF/celular do recebedor |
| **Conta Laranja** | Conta bancária de terceiro usada por criminosos para receber valores de golpes |
| **Engenharia Social** | Manipulação psicológica para induzir vítima a realizar ações |
| **Escalada de Valores** | Padrão onde valores crescem progressivamente (típico de golpes de confiança) |
| **Indicador** | Sinal individual de risco (ex: idade_60_plus, valor_alto) |
| **Modus Operandi** | Método característico de execução de um golpe |
| **Padrão** | Combinação de indicadores que caracteriza um tipo de golpe |
| **PIX Errado** | Golpe onde criminoso envia PIX e pede "devolução" para conta diferente |
| **Romance Scam** | Golpe do amor - criminoso cria relacionamento online para pedir dinheiro |
| **Score** | Pontuação que indica probabilidade de fraude |

---

## 10. Contatos e Suporte

Para dúvidas sobre este módulo:
- **Documentação técnica**: Este arquivo
- **Código fonte**: `/backend/app/core/social_engineering.py`
- **Testes**: `/backend/app/tests/test_social_engineering.py`
- **Issues**: GitHub Issues do projeto

---

**Última atualização**: 2024-01-15  
**Versão do módulo**: 3.0.0  
**Autor**: Equipe Anomalia PIX

