<!-- decision_id: EXP014B_R3I_FN_RECOVERY_FRONTIER_FOUND -->

## EXP-014B-R3I — Auditoria dos 56 FNs encontrou fronteira de resgate

**Decisão:** aprovar o EXP-014B-R3I como auditoria diagnóstica e consolidar que os 56 FNs residuais do R3H não são irredutíveis.

O patch do R3H-FROZEN foi aprovado funcionalmente:

```text
patch_status=PASS_R3H_FROZEN_VALIDATED_METRICS_MATCH_RULE_OVERLAP_OK_WILSON_PASS
patched_all_pass=true

Benchmark validado R3H:

TP=1409
FP=4935
FN=56
recall=96,177%
precision=22,210%
FPR=4,391%

A auditoria encontrou 2.851 candidatos de resgate e confirmou recuperação possível de FN.

Fronteira diagnóstica principal:

+100 FP  -> +13 TP, FN=43, FP=5025, recall=97,06%
+250 FP  -> +19 TP, FN=37, FP=5181, recall=97,47%
+500 FP  -> +27 TP, FN=29, FP=5418, recall=98,02%
+1000 FP -> +38 TP, FN=18, FP=5908, recall=98,77%
+2000 FP -> +51 TP, FN=5,  FP=6895, recall=99,66%

Conclusão: o melhor candidato operacional de curto prazo é o cenário R3I_RESCUE_100, pois recupera 13 FNs com apenas 90 FPs adicionados e preserva precision próxima do R3H.

Próximo passo: executar EXP-014B-R3J — Frozen Rescue Frontier + FP Re-tightening, validando cenários fixos de resgate e tentando reduzir novamente os FPs adicionados sem perder os FNs recuperados.

---

<!-- decision_id: EXP014B_R3J_FROZEN_RESCUE_FRONTIER_FP_RETIGHTENING -->

## EXP-014B-R3J — Fronteira congelada de resgate + FP re-tightening

**Decisão:** iniciar o EXP-014B-R3J para transformar a auditoria diagnóstica do R3I em cenários congelados e testáveis, sem nova mineração longa de FN.

O R3H congelado permanece como benchmark principal:

```text
TP=1409
FP=4935
FN=56
recall=96,177%
precision=22,210%
FPR=4,391%
```

O R3I mostrou que os 56 FNs não são irredutíveis. A fronteira de resgate trouxe:

```text
+100 FP  -> +13 TP, FN=43, FP=5025
+250 FP  -> +19 TP, FN=37, FP=5181
+500 FP  -> +27 TP, FN=29, FP=5418
+1000 FP -> +38 TP, FN=18, FP=5908
+2000 FP -> +51 TP, FN=5,  FP=6895
```

O R3J deve:

```text
1. reaplicar os candidate_ids congelados de cada cenário do R3I;
2. confirmar as métricas reproduzidas;
3. executar FP re-tightening curto somente nos alertas adicionados pelo rescue;
4. preservar os FNs recuperados por padrão;
5. selecionar um candidato recomendado, priorizando o rescue_budget_100.
```

Critério principal de sucesso:

```text
recuperar pelo menos 13 FNs
manter FP próximo ou abaixo de 5000
não perder FNs recuperados no re-tightening
```

Qualquer candidato do R3J ainda precisará de validação congelada posterior antes de promoção.

---

<!-- decision_id: EXP014B_R3J_RESCUE100_CANDIDATE -->

## EXP-014B-R3J — Candidato Rescue100 com FP abaixo de 5000

**Decisão:** aprovar o EXP-014B-R3J como candidato pós-R3H e selecionar `rescue_budget_100` para validação congelada.

O R3H congelado era:

```text
TP=1409
FP=4935
FN=56
recall=96,177%
precision=22,210%
FPR=4,391%

O R3J reaplicou cenários congelados do R3I e executou FP re-tightening curto somente sobre alertas adicionados pelo rescue.

Resultado recomendado:

EXP014B_R3J_RESCUE100
TP=1422
FP=4965
FN=43
recall=97,065%
precision=22,264%
FPR=4,418%
Wilson low=96,070%

Ganho líquido contra R3H:

+13 TP
-13 FN
+30 FP
FP ainda abaixo de 5000
TP_loss no re-tightening=0

Conclusão: o rescue_budget_100 é o melhor candidato operacional atual porque reduz FN de 56 para 43 mantendo FP abaixo de 5000. Próximo passo: EXP-014B-R3J-FROZEN, para validar o artifact recomendado sem nova mineração.

---

<!-- decision_id: EXP014B_R3J_FROZEN_VALIDATION -->

## EXP-014B-R3J-FROZEN — Validação congelada do candidato Rescue100

**Decisão:** iniciar a validação congelada do candidato `EXP014B_R3J_RESCUE100`, mantendo a estratégia FN First / FP Second.

O R3H congelado era:

```text
TP=1409
FP=4935
FN=56
recall=96,177%
precision=22,210%
FPR=4,391%
```

O EXP-014B-R3J selecionou `rescue_budget_100` como candidato recomendado:

```text
TP=1422
FP=4965
FN=43
recall=97,065%
precision=22,264%
FPR=4,418%
Wilson low=96,070%
```

Ganho contra R3H:

```text
+13 TP
-13 FN
+30 FP líquido
precision levemente maior
FP continua abaixo de 5000
```

A validação congelada deve:

```text
1. carregar R3H-FROZEN/10_predictions.csv;
2. carregar R3J/08_policy_artifact_recommended.json;
3. carregar R3I/07_rescue_candidates.csv;
4. reaplicar apenas os candidate_ids congelados do rescue_budget_100;
5. reaplicar apenas as regras congeladas de re-tightening;
6. confirmar exatamente TP=1422, FP=4965, FN=43.
```

Critério de PASS:

```text
TP=1422
FP=4965
FN=43
rescue_fn_recovered=13
rescue_fp_added=90
retightening_fp_removed=60
retightening_tp_loss=0
Wilson low >= 95%
schema OK
```

Se passar, o R3J_RESCUE100 se torna o novo candidato congelado principal e pode substituir o R3H como benchmark expandido operacional.

---

<!-- decision_id: EXP014B_R3K_REUSE_RESCUE_LIBRARY_MICROEVOLUTION -->

## EXP-014B-R3K — Reuso da biblioteca de resgate sobre R3J-FROZEN

**Decisão:** iniciar uma microevolução curta sobre o R3J-FROZEN, mantendo a estratégia FN First / FP Second sem repetir mineração longa.

O R3J-FROZEN foi validado com:

```text
TP=1422
FP=4965
FN=43
recall=97,065%
precision=22,264%
FPR=4,418%
Wilson low=96,070%
```

Este ponto substitui o R3H como melhor candidato congelado atual. Ainda restam 43 FNs, mas agora o orçamento de FP é curto porque o FP já está em 4965, próximo do cap simbólico de 5000.

O R3K deve:

```text
1. carregar R3J-FROZEN/10_predictions.csv;
2. usar exp014b_r3j_frozen_pred como base;
3. reaproveitar R3I/07_rescue_candidates.csv;
4. reavaliar a biblioteca de resgate sobre os 43 FNs residuais;
5. montar cenários pequenos de orçamento FP: 25, 50, 100, 250;
6. aplicar re-tightening curto apenas nos alertas adicionados;
7. selecionar candidato se FN cair mantendo FP preferencialmente <=5000.
```

Critério de sucesso:

```text
reduzir FN abaixo de 43
manter FP <=5000, se possível
não executar mineração longa
gerar artifact recomendado para validação congelada
```

Se o R3K não melhorar dentro do cap, o R3J-FROZEN deve ser consolidado como benchmark principal e a próxima etapa deve migrar para preparação de produção, hard-negative mining ou segundo estágio.

---

<!-- decision_id: EXP014B_R3K_RESCUE100_CANDIDATE -->

## EXP-014B-R3K — Microevolução Rescue100 sobre R3J-FROZEN

**Decisão:** aprovar o EXP-014B-R3K como microevolução candidata e selecionar `r3k_rescue_budget_100` para validação congelada.

Base R3J-FROZEN:

```text
TP=1422
FP=4965
FN=43
recall=97,065%
precision=22,264%
FPR=4,418%

Resultado recomendado R3K:

TP=1425
FP=4966
FN=40
recall=97,270%
precision=22,297%
FPR=4,419%
Wilson low=96,303%

Ganho líquido contra R3J-FROZEN:

+3 TP
-3 FN
+1 FP
TP_loss no re-tightening=0
FP ainda abaixo de 5000

Conclusão: o R3K mantém a estratégia FN First / FP Second e melhora o candidato congelado anterior praticamente sem custo de FP. Próximo passo: EXP-014B-R3K-FROZEN, para validar o artifact recomendado sem nova mineração.

---

<!-- decision_id: EXP014B_R3L_HEADROOM_RESIDUAL_FN_OPTIMIZER -->

## EXP-014B-R3L — Headroom + otimizador dos FNs residuais

**Decisão:** iniciar o EXP-014B-R3L como próxima estratégia de redução de FN após o R3K.

O R3K gerou novo candidato recomendado:

```text
TP=1425
FP=4966
FN=40
recall=97,270%
precision=22,297%
FPR=4,419%
Wilson low=96,303%
```

A margem para manter FP <=5000 é pequena, apenas ~34 FPs. Por isso, a próxima estratégia deve primeiro criar `headroom` de FP com vetos TP0 sobre alertas já existentes, sem aumentar FN. Em seguida, esse headroom será usado para comprar novos rescues nos 40 FNs residuais.

Critério de sucesso:

```text
reduzir FN abaixo de 40
manter FP <=5000, se possível
não perder TP em headroom/re-tightening
gerar artifact recomendado para validação congelada
```

---

<!-- decision_id: EXP014B_R3L_CAP5100_CANDIDATE -->

## EXP-014B-R3L — Headroom + rescue reduz FN e FP ao mesmo tempo

**Decisão:** aprovar o EXP-014B-R3L como candidato superior ao R3K e selecionar `r3l_cap_5100` para validação congelada.

Base R3K:

```text
TP=1425
FP=4966
FN=40
recall=97,270%
precision=22,297%
FPR=4,419%

Resultado recomendado R3L:

TP=1436
FP=4921
FN=29
recall=98,020%
precision=22,589%
FPR=4,379%
Wilson low=97,172%

Ganho líquido contra R3K:

+11 TP
-11 FN
-45 FP
TP_loss no headroom=0
TP_loss no re-tightening=0
FP final abaixo de 5000

Conclusão: a estratégia de criar headroom TP0 antes de novos rescues funcionou muito bem. O R3L é o melhor candidato expandido até agora. Próximo passo: EXP-014B-R3L-FROZEN, para validar o artifact recomendado sem nova mineração.

---

<!-- decision_id: EXP014B_R3M_CONSOLIDATE_R3L_AND_RESIDUAL_FN_V2 -->

## EXP-014B-R3M — Consolidação R3L e novo avanço sobre FNs residuais

**Decisão:** consolidar o excelente resultado do EXP-014B-R3L e, na mesma execução, testar uma microevolução adicional para reduzir falsos negativos.

O R3L produziu o melhor candidato expandido até agora:

```text
TP=1436
FP=4921
FN=29
recall=98,020%
precision=22,589%
FPR=4,379%
Wilson low=97,172%
```

Ganho contra R3K:

```text
+11 TP
-11 FN
-45 FP
TP_loss no headroom=0
TP_loss no re-tightening=0
FP final abaixo de 5000
```

O EXP-014B-R3M deve executar duas etapas:

```text
A. R3L-FROZEN
   1. carregar EXP-014B-R3K/10_predictions_recommended.csv;
   2. carregar EXP-014B-R3L/11_policy_artifact_recommended.json;
   3. carregar EXP-014B-R3L/06_rescue_candidates.csv;
   4. reaplicar headroom, rescues e re-tightening congelados;
   5. confirmar TP=1436, FP=4921, FN=29.

B. Residual FN v2
   1. usar o R3L congelado como nova base;
   2. criar novo headroom TP0, se possível;
   3. reusar bibliotecas R3L/R3I de rescues;
   4. gerar candidatos novos sobre os 29 FNs residuais;
   5. aplicar re-tightening curto apenas nos alertas adicionados;
   6. recomendar candidato somente se reduzir FN mantendo FP preferencialmente <=5000.
```

Critério de sucesso:

```text
R3L frozen validado
FN reduzido abaixo de 29
FP preferencialmente <=5000
sem perda de TP no headroom/re-tightening
```

Se o R3M não melhorar materialmente o R3L, o R3L deve ser consolidado como benchmark principal e a próxima fase deve migrar para hard-negative mining, segundo estágio ou preparação de produção.

---

<!-- decision_id: EXP014B_R3M_CAP5100_CANDIDATE -->

## EXP-014B-R3M — R3L consolidado e nova redução forte de FN/FP

**Decisão:** aprovar o EXP-014B-R3M como melhor candidato expandido até agora e selecionar `r3m_cap_5100` para validação congelada.

O R3M primeiro validou o R3L congelado:

```text
R3L-FROZEN=PASS_R3L_FROZEN_VALIDATED
TP=1436
FP=4921
FN=29
recall=98,020%
precision=22,589%
FPR=4,379%

Em seguida, executou nova microevolução FN First / FP Second e recomendou:

R3M_CAP5100
TP=1444
FP=4816
FN=21
recall=98,567%
precision=23,067%
FPR=4,285%
Wilson low=97,819%

Ganho líquido contra R3L:

+8 TP
-8 FN
-105 FP
TP_loss no headroom v2=0
TP_loss no re-tightening=0
FP final abaixo de 5000

Conclusão: o R3M melhorou FN e FP simultaneamente e é o novo melhor candidato. Próximo passo: EXP-014B-R3M-FROZEN, para validar o artifact recomendado sem nova mineração.

---

<!-- decision_id: EXP014B_R3N_CONSOLIDATE_R3M_AND_IRREDUCIBLE_FN_SEARCH -->

## EXP-014B-R3N — Consolidação R3M e busca de FNs irredutíveis

**Decisão:** continuar a estratégia FN First / FP Second com uma rodada que primeiro consolida o R3M e depois tenta avançar sobre os 21 FNs residuais.

O EXP-014B-R3M produziu o melhor candidato até agora:

```text
TP=1444
FP=4816
FN=21
recall=98,567%
precision=23,067%
FPR=4,285%
Wilson low=97,819%
```

O R3N deve executar duas etapas:

```text
A. R3M-FROZEN
   1. carregar EXP-014B-R3M/14_predictions_recommended.csv;
   2. carregar EXP-014B-R3M/13_policy_artifact_recommended.json;
   3. carregar EXP-014B-R3M/09_r3m_rescue_candidates.csv;
   4. reaplicar headroom, rescues e re-tightening congelados;
   5. confirmar TP=1444, FP=4816, FN=21.

B. Irreducible FN Search
   1. usar R3M congelado como nova base;
   2. criar headroom TP0 adicional;
   3. reusar bibliotecas R3M/R3L/R3I de rescues;
   4. gerar novos candidatos diretamente dos 21 FNs residuais;
   5. aplicar re-tightening TP0 nos alertas adicionados;
   6. recomendar candidato apenas se FN cair mantendo FP preferencialmente dentro do cap.
```

Critério de sucesso:

```text
R3M frozen validado
FN reduzido abaixo de 21
FP preferencialmente <=5000
sem perda de TP no headroom/re-tightening
```

Se a redução adicional de FN ficar marginal ou cara em FP, iniciar auditoria dos FNs residuais e considerar esses casos como limite prático até hard-negative mining/segundo estágio.

---

<!-- decision_id: EXP014B_R3N_CAP5000_ZERO_FN_CANDIDATE -->

## EXP-014B-R3N — FN zerado no dataset expandido

**Decisão:** aprovar o EXP-014B-R3N como melhor candidato expandido até agora e selecionar `r3n_cap_5000` para validação congelada.

O R3N primeiro validou o R3M congelado:

```text
R3M-FROZEN=PASS_R3M_FROZEN_VALIDATED
TP=1444
FP=4816
FN=21
recall=98,567%
precision=23,067%
FPR=4,285%

Em seguida, executou nova busca FN First / FP Second e recomendou:

R3N_CAP5000
TP=1465
FP=4769
FN=0
recall=100,000%
precision=23,500%
FPR=4,244%
Wilson low=99,738%

Ganho líquido contra R3M:

+21 TP
-21 FN
-47 FP
TP_loss no headroom=0
TP_loss no re-tightening=0
FP final abaixo de 5000

Conclusão: no dataset expandido atual, os FNs residuais foram zerados. Próximo passo obrigatório: EXP-014B-R3N-FROZEN, seguido de hardening/robustez para avaliar sobreajuste e segurança antes de promoção.

---

<!-- decision_id: EXP014B_R3O_CONSOLIDATE_R3N_FP_ONLY -->

## EXP-014B-R3O — Consolidar R3N e iniciar fase FP-only

**Decisão:** consolidar a configuração campeã `R3N_CAP5000` e, daqui em diante, trabalhar apenas na redução de falsos positivos, preservando `FN=0`.

O EXP-014B-R3N atingiu o melhor resultado do projeto no dataset expandido:

```text
TP=1465
FP=4769
FN=0
recall=100,000%
precision=23,500%
FPR=4,244%
Wilson low=99,738%
```

Ganho contra o R3M:

```text
+21 TP
-21 FN
-47 FP
TP_loss no headroom=0
TP_loss no re-tightening=0
FP final abaixo de 5000
```

O R3O deve executar duas etapas:

```text
A. R3N-FROZEN
   1. carregar EXP-014B-R3N/15_predictions_recommended.csv;
   2. carregar EXP-014B-R3N/14_policy_artifact_recommended.json;
   3. carregar EXP-014B-R3N/09_r3n_rescue_candidates.csv;
   4. reaplicar headroom, rescues e re-tightening congelados;
   5. confirmar TP=1465, FP=4769, FN=0.

B. FP-only reducer
   1. usar R3N congelado como nova base;
   2. minerar apenas vetos sobre alertas atuais;
   3. aceitar somente regras com TP_loss=0;
   4. não aplicar novos rescues;
   5. não trocar recall por FP;
   6. recomendar candidato somente se FP cair mantendo FN=0.
```

Critério de sucesso:

```text
R3N frozen validado
FP reduzido abaixo de 4769
FN preservado em 0
TP_loss=0 nas novas regras
```

Se houver ganho, o próximo passo será `EXP-014B-R3O-FROZEN`. Caso contrário, o R3N fica consolidado como benchmark campeão e a próxima etapa deve ser hardening/robustez.

---

<!-- decision_id: EXP014B_R3P_RESIDUAL_FP_ONLY_CANDIDATE -->

## EXP-014B-R3P — Redução residual conservadora de falsos positivos

**Decisão:** aprovar o EXP-014B-R3P como candidato FP-only conservador superior ao R3O-FROZEN, mantendo a exigência de validação congelada antes de substituir o benchmark oficial.

Base R3O-FROZEN:

```text
TP=1465
FP=4252
FN=0
recall=100,000%
precision=25,625%
FPR=3,784%
```

Resultado recomendado R3P:

```text
TP=1465
FP=4221
FN=0
recall=100,000%
precision=25,765%
FPR=3,756%
```

Ganho líquido contra R3O-FROZEN:

```text
FP removidos=31
TP_loss=0
FN_delta=0
```

O experimento usou restrições conservadoras: sem rescues, sem mudança de threshold, máximo de 2 condições por regra, suporte em múltiplos splits temporais, suporte em múltiplos meses e suporte fora de TRAIN.

Foram selecionadas 2 regras TP0 estáveis. A busca parou com `no_more_stable_tp0_fp_rules_at_depth_3`, sugerindo que, sob as restrições atuais, o ganho residual seguro já está próximo do limite imediato.

**Status:** R3P aprovado como candidato.

**Próximo passo:** executar `EXP-014B-R3P-FROZEN`, sem nova mineração, para validar replay exato do artifact recomendado antes de substituir o R3O-FROZEN como benchmark principal.

---

<!-- decision_id: EXP014B_R3P_FROZEN_VALIDATED -->

## EXP-014B-R3P-FROZEN — R3P validado como novo benchmark congelado

**Decisão:** aprovar o EXP-014B-R3P-FROZEN e substituir o R3O-FROZEN como benchmark congelado principal.

A validação congelada reaplicou exclusivamente as 2 regras FP-only do R3P, sem nova mineração, sem rescues e sem alteração de threshold.

Resultado:

```text
R3O-FROZEN:
TP=1465
FP=4252
FN=0
recall=100,000%
precision=25,625%
FPR=3,784%

R3P-FROZEN:
TP=1465
FP=4221
FN=0
recall=100,000%
precision=25,765%
FPR=3,756%

Ganho consolidado:

FP removidos vs R3O-FROZEN = 31
TP_loss = 0
FN_delta = 0
prediction_mismatches_vs_existing = 0

As duas regras foram classificadas como estáveis, com suporte em múltiplos splits temporais, múltiplos meses e suporte fora de TRAIN.

Status: PASS_R3P_FROZEN_VALIDATED_FN_ZERO_PRESERVED_FP_REDUCED.

Nova configuração campeã congelada: EXP-014B-R3P-FROZEN.

Próximo passo: executar EXP-014B-R3Q — Residual FP-only Boundary Probe para verificar se ainda existe ganho FP-only seguro relevante ou se o R3P-FROZEN representa a fronteira prática atual antes do hardening final.

---

<!-- decision_id: EXP014B_COMMERCIAL_TARGET_SHIFT_R3T_R3U -->

## Mudanca de meta comercial + diagnostico de FPs R3T

**Decisao:** manter o R3Q como referencia tecnica atual e mudar a direcao da otimizacao para custo-beneficio comercial, aceitando pequeno orcamento de FN para tentar reduzir fortemente FP.

Referencia atual analisada:

```text
EXP-014B-R3Q / exp014b_r3q_frozen_pred
TP=1465
FP=4074
FN=0
recall=100,000%
precision=26,449%
FPR=3,625%
```

Nova meta comercial:

```text
FN <= 5
recall >= 95%
FPR <= 1,5%
```

Com 112.379 normais, o alvo FPR<=1,5% implica aproximadamente FP<=1685. Gap atual: 2389 FPs acima do alvo.

### Resultado do R3S

O segundo estagio/ranker nao atingiu a meta. Melhor politica:

```text
TP=1460
FP=4045
FN=5
recall=99,659%
FPR=3,599%
FP removidos=29
gap restante=2360 FP
```

Conclusao: nao promover R3S. O ranker consumiu todo o orcamento de FN para ganho pequeno de FP.

### Achados do R3T

O R3T foi diagnostico, sem promocao de politica. Principais concentradores de FP:

```text
qtd_rec_bin=qtdrec_LT_0: FP=3457, 84,9% dos FPs
valor_rec_bin=valrec_LT_0: FP=3457, 84,9% dos FPs
module_quiet=module_quiet: FP=4068, 99,9% dos FPs
se_worst_pattern=<MISSING>: FP=4068, 99,9% dos FPs
first_receiver_flag_real=1: FP=3932, 96,5% dos FPs
ratio_bin=ratio_LT_0.05: FP=767, precision=17,7%
```

Interpretacao: o excesso de FP parece concentrado em recebedores novos/sem historico/reputacao suficiente e em alertas sem reforco dos modulos SE/BEH/runtime. O problema principal deixou de parecer microveto e passou a parecer falta de features comerciais de confianca do recebedor e do relacionamento pagador-recebedor.

**Proxima rodada:** executar `EXP-014B-R3U — Receiver/Relationship Trust Feature Probe`, testando features derivadas de historico do recebedor, recorrencia pagador-recebedor, reputacao de recebimento, MBK/completude e sinais de recebedor confiavel.

---

<!-- decision_id: EXP014B_R3V_EXISTING_ACTION_POLICY_CALIBRATION -->

## EXP-014B-R3V — Ajuste de direção para a arquitetura real APROVAR/CONFIRMAR/BLOQUEAR

**Decisão:** corrigir a direção após revisão dos módulos principais. O modelo já opera com ações `APROVAR`, `CONFIRMAR` e `BLOQUEAR`; portanto, a próxima estratégia não é criar novo banding, mas calibrar a política comercial sobre as ações existentes.

Meta comercial vigente:

```text
FN <= 5
recall >= 95%
FPR <= 1,5%
```

Base operacional preservada para diagnóstico:

```text
EXP-014B-R3Q
TP=1465
FP=4074
FN=0
recall=100,000%
FPR=3,625%
gap até FPR<=1,5% = 2389 FP
```

Achados recentes:

```text
R3S: segundo estágio simples não atingiu a meta; removeu só 29 FP consumindo FN=5.
R3T: FPs concentrados em ausência/histórico fraco de recebedor e module_quiet.
R3U: features derivadas de confiança não trouxeram ganho seguro; 0 regras selecionadas.
```

Novo experimento:

```text
EXP-014B-R3V — Existing Action Policy Calibration Probe
```

Objetivo:

```text
auditar FP por APROVAR/CONFIRMAR/BLOQUEAR;
testar políticas comerciais usando as ações existentes;
ver se BLOQUEAR/CONFIRMAR podem ser recalibrados sem violar FN<=5 e recall>=95%;
não promover nada sem validação congelada.
```

---

<!-- decision_id: EXP014B_R3W_ACTION_ALIGNMENT_AUDIT -->

## EXP-014B-R3W — Auditoria de alinhamento entre R3Q e decisão operacional

**Decisão:** executar auditoria antes de novas calibrações, pois o R3V mostrou desalinhamento entre `exp014b_r3q_frozen_pred` e `decisao`.

Motivo:

```text
R3Q: TP=1465, FP=4074, FN=0
decisao exportada: só 90 intervenções (73 CONFIRMAR, 17 BLOQUEAR)
```

O R3W deve cruzar:

```text
R3Q alert + CONFIRMAR/BLOQUEAR
R3Q alert + APROVAR
R3Q no alert + CONFIRMAR/BLOQUEAR
R3Q no alert + APROVAR
```

Objetivo:

```text
descobrir se a coluna decisao vem do DecisionEngine vanilla,
de uma política anterior, ou de exportação antes do artifact R3Q;
quantificar quantas fraudes R3Q aparecem como APROVAR;
impedir calibração comercial sobre coluna desalinhada.
```

Critério de decisão:

```text
Se houver muitas fraudes em R3Q alert + APROVAR:
    alinhar exportação/pipeline antes de otimizar FP.
Se a decisão estiver alinhada:
    seguir para calibração comercial de CONFIRMAR/BLOQUEAR.
```

---

<!-- decision_id: EXP014B_R3X_DECISION_POLICY_RECONSTRUCTION -->

## EXP-014B-R3X — Reconstrução da decisão operacional alinhada ao R3Q

**Decisão:** alinhar a ação operacional ao benchmark experimental antes de qualquer nova otimização de FP.

O R3W confirmou desalinhamento:

```text
R3Q: TP=1465, FP=4074, FN=0
decisao original: TP=59, FP=31, FN=1406
1406 fraudes R3Q estavam como APROVAR
```

Nova regra de reconstrução:

```text
exp014b_r3q_frozen_pred=0 -> APROVAR
exp014b_r3q_frozen_pred=1 -> CONFIRMAR ou BLOQUEAR
```

O R3X deve:

```text
1. criar r3x_decisao_pos_policy;
2. garantir que todo alerta R3Q vire pelo menos CONFIRMAR;
3. escolher subconjunto de maior risco para BLOQUEAR;
4. medir separadamente FPR total de detecção e FPR de BLOQUEAR;
5. gerar artifact operacional alinhado para validação congelada.
```

Critério de sucesso:

```text
decisao_pos_policy como intervenção deve reproduzir exatamente R3Q:
TP=1465, FP=4074, FN=0.
```

A calibragem comercial posterior deve ser feita sobre a separação CONFIRMAR/BLOQUEAR, não sobre uma `decisao` desalinhada.

---

<!-- decision_id: EXP014B_R3X_OPERATIONAL_BASELINE_CONSOLIDATED -->

## EXP-014B-R3X - Baseline operacional alinhado consolidado

**Decisão:** consolidar o R3X como novo marco operacional do projeto.

O R3X corrigiu o desalinhamento entre `exp014b_r3q_frozen_pred` e `decisao`, criando `r3x_decisao_pos_policy`.

Resultado consolidado:

```text
APROVAR:
108305 linhas, 0 fraudes

CONFIRMAR:
3046 linhas, 172 fraudes, 2874 normais, precision=5,65%

BLOQUEAR:
2493 linhas, 1293 fraudes, 1200 normais, precision=51,87%, FPR=1,068%
```

A intervenção total `CONFIRMAR+BLOQUEAR` reproduz o R3Q:

```text
TP=1465
FP=4074
FN=0
recall=100,000%
FPR=3,625%
```

A meta de `FPR<=1,5%` foi atingida para `BLOQUEAR`, mas não para a intervenção total.

Próximo passo obrigatório:

```text
EXP-014B-R3X-FROZEN
```

Depois do frozen, a melhoria deve focar exclusivamente a fila `CONFIRMAR`, preservando:

```text
APROVAR com 0 fraudes conhecidas
BLOQUEAR com FPR<=1,5%
alinhamento operacional com R3Q
```

---

<!-- decision_id: EXP014B_R3Y_CONFIRM_QUEUE_REDUCTION -->

## EXP-014B-R3Y — Redução da fila CONFIRMAR

**Objetivo:** melhorar a fila `CONFIRMAR` consolidada no R3X-FROZEN, mantendo `BLOQUEAR` intocado.

Baseline:

```text
APROVAR: 108305 linhas, 0 fraudes
BLOQUEAR: 2493 linhas, 1293 fraudes, 1200 normais, FPR=1,068%
CONFIRMAR: 3046 linhas, 172 fraudes, 2874 normais, precision=5,65%
```

Regra de segurança:

```text
Somente CONFIRMAR pode ser rebaixado para APROVAR.
BLOQUEAR não pode mudar.
FN adicional máximo: 5.
```

Meta inicial:

```text
remover 500 a 1000 normais de CONFIRMAR, se possível;
manter BLOQUEAR congelado;
não perder mais de 5 fraudes conhecidas.
```

Se o ganho for relevante, executar R3Y-FROZEN.
Se não houver ganho, o próximo passo deve ser enriquecimento de features para distinguir CONFIRMAR benigno de fraude residual.

---

<!-- decision_id: EXP014B_R3Y_PROMOTION_AND_R3Z_NEXT -->

## Promocao R3Y e proximo experimento R3Z

**Decisao:** promover R3Y para validacao congelada e iniciar R3Z para reducao residual de falsos positivos.

R3Y recomendado:

```text
TP=1465
FP=3489
FN=0
FPR=3,105%
FP removidos vs R3X-FROZEN = 585
BLOQUEAR unchanged = true
```

R3Y-FROZEN deve validar replay exato das 25 demotions selecionadas.

R3Z deve partir do R3Y-FROZEN e tentar nova reducao somente no `CONFIRMAR` residual:

```text
CONFIRMAR apos R3Y:
2461 linhas
172 fraudes
2289 normais
```

Regras de seguranca do R3Z:

```text
BLOQUEAR intocado
somente CONFIRMAR residual pode virar APROVAR
FN adicional default <= 5
demotions R3Y congeladas
```

Se R3Z trouxer ganho relevante, executar R3Z-FROZEN. Se nao, consolidar R3Y-FROZEN e partir para novas features.

---

<!-- decision_id: EXP014B_R3Z_CONSOLIDATION_AND_R4A_FPR15_PATHFINDER -->

## Consolidação R3Z e próximo avanço para FPR 1,5%

**Decisão:** promover R3Z para frozen validation e iniciar R4A como busca direcionada ao alvo FPR<=1,5%.

R3Z recomendado:

```text
TP=1460
FP=2833
FN=5
recall=99,659%
FPR=2,521%
BLOQUEAR unchanged=true
```

O R3Z reduziu mais 656 falsos positivos contra o R3Y-FROZEN, mas consumiu todo o orçamento comercial de FN.

Próximo passo obrigatório:

```text
EXP-014B-R3Z-FROZEN
```

Depois, executar R4A:

```text
EXP-014B-R4A — FPR 1.5% Pathfinder
```

R4A testa duas trilhas:

```text
A) residual_zero_fn_from_r3z:
   partir do R3Z-FROZEN e exigir FN adicional=0.

B) reoptimize_from_r3y:
   partir do R3Y-FROZEN e reotimizar com FN total<=5.
```

Critério:

```text
Manter BLOQUEAR intocado.
Buscar redução adicional de FP.
Não ultrapassar FN=5.
Se target FPR<=1,5% não for atingido, avaliar necessidade de novas features/sinais.
```

<!-- decision_id: EXP014B_R4A_CHAMPION_AND_R4B_FINAL_FPR_LT1 -->

## R4A campeão consolidado e nova meta final R4B

**Resultado campeão atual:** EXP-014B-R4A ampliado.

```text
TP=1465
FP=1677
FN=0
Recall=100,000%
Precision=46,626%
FPR=1,492%
BLOQUEAR intacto
```

A meta anterior `FPR<=1,5%` foi atingida com folga pequena de 8 FP contra o alvo de 1685.

Configuração operacional campeã:

```text
APROVAR:   110702 normais, 0 fraudes
BLOQUEAR:  2493 linhas, 1293 fraudes, 1200 normais
CONFIRMAR: 649 linhas, 172 fraudes, 477 normais
```

Próximo passo obrigatório:

```text
EXP-014B-R4A-FROZEN
```

Nova meta final:

```text
EXP-014B-R4B
FPR < 1,0%
FN total <= 5
```

Observação crítica: com 112379 normais, `FPR<1%` exige `FP<=1123`. Como o BLOQUEAR atual sozinho tem `FP=1200`, a meta final não pode ser alcançada mexendo apenas no CONFIRMAR. O R4B deve permitir reotimização de intervenções `CONFIRMAR` e `BLOQUEAR` para APROVAR dentro do orçamento `FN<=5`, com revisão semântica posterior.

---