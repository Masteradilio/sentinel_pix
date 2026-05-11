# Validation Report Pós-FASE 2 — Pipeline Antifraude PIX

**Gerado em:** `2026-05-09T13:38:10`

## 1. Decisão oficial

A FASE 2 foi encerrada com sucesso mínimo validado.

A versão oficial do pipeline passa a ser o **baseline pós-C1**, mantendo o modelo LGBM de produção anterior, o guard rail LGBM, a exceção contextual de alto valor da FASE 1 e a regra cirúrgica `C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER`.

Decisão final:

```text
Promovido: V1_GUARD_CONTEXTUAL
Promovido: C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER
Rejeitado: LGBM v6.2 para runtime
Rejeitado: R2_LOW_VALUE_GRAY_FIRST_RECEIVER
Rejeitado: meta-learner shadow como componente de decisão
Próxima etapa: FASE 3 — consolidação operacional, testes, documentação e observabilidade
```

## 2. Dataset e fonte das métricas

- Fonte dos artefatos baseline: `E:\Projetos\rebuild_pix\resultados\experimentos\EXP-006C-R2`
- Seeds oficiais: `42` e `123`
- C1 reaplicada no relatório com `min_score=58.0`, conforme validação runtime pós-FASE 2.

## 3. Métricas oficiais

| Seed/Conjunto | TP | FP | FN | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 42 | 347 | 14 | 8 | 96,1219% | 97,7465% | 0,9693 | 0,2480% |
| seed 123 | 347 | 12 | 8 | 96,6574% | 97,7465% | 0,9720 | 0,2126% |
| unique union | 347 | 25 | 8 | 93,2796% | 97,7465% | 0,9546 | 0,2278% |

> Observação: a linha `unique union` é usada apenas como visão deduplicada de inventário de casos únicos entre os seeds. As métricas oficiais de performance operacional da versão pós-FASE 2 são as métricas por seed, especialmente seed 42 e seed 123. O `unique union` não deve ser usado como métrica principal de comparação entre versões.

## 4. Delta da C1

| Seed | FNs recuperados | FPs adicionados | TPs perdidos | FPs removidos | Rule hits |
|---:|---:|---:|---:|---:|---:|
| 42 | 1 | 0 | 0 | 0 | 1 |
| 123 | 1 | 0 | 0 | 0 | 1 |

Conclusão: a C1 recupera 1 FN nos dois seeds, adiciona 0 FP e não perde TP.

## 5. Caso recuperado pela C1

| Campo | Valor |
|---|---:|
| transaction_id | `E0000020820260205003505340630525` |
| customer_id | `4321433355` |
| is_fraud | `1` |
| vl_pix | R$ 381.00 |
| relacionamento meses | 6 |
| first_receiver_flag | 1 |
| pix_key_random_flag | 0 |
| lgbm_raw | 0.06847418 |
| score_final pós-C1 | 62.00 |

## 6. Configuração oficial

```json
{
  "threshold_confirmar": 62.0,
  "threshold_bloquear": 95.0,
  "lgbm_guard_enabled": true,
  "lgbm_guard_threshold": 0.3,
  "guard_exception_alto_valor_se_beh_enabled": true,
  "exp006f_c1_enabled": true,
  "exp006f_c1_min_score": 58.0,
  "exp006f_c1_max_score": 62.0,
  "exp006f_c1_min_valor": 100.0,
  "exp006f_c1_max_valor": 500.0,
  "exp006f_c1_max_rel_meses": 12.0,
  "exp006f_c1_min_lgbm_raw": 0.06,
  "exp006f_c1_max_lgbm_raw": 0.1,
  "exp006f_c1_require_first_receiver": true,
  "exp006f_c1_require_not_pix_random": true,
  "exp006f_c1_max_se_score": 0.0,
  "exp006f_c1_max_beh_score": 0.0,
  "se_pattern_residual_enabled": false,
  "exp003_residual_confirm_enabled": false
}
```

## 7. Experimentos promovidos e rejeitados

| Experimento | Decisão | Motivo |
|---|---|---|
| EXP-004-FINAL / V1_GUARD_CONTEXTUAL | Promovido | Recuperou FN de alto valor sem adicionar FP |
| EXP-005A / LGBM v6.2 recall-oriented | Não promovido | Promissor model-only, mas exigia validação no engine |
| EXP-005B / calibração pós-LGBM v6.2 | Rejeitado | Não reduziu FN líquido no engine real e aumentou FP |
| EXP-006C / R2_LOW_VALUE_GRAY_FIRST_RECEIVER | Rejeitado | Recuperou 0 FN e adicionou FP |
| EXP-006E/006F / C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER | Promovido | Recuperou 1 FN nos dois seeds, 0 FP adicionado |
| EXP-007A / Meta-Learner Shadow | Não promovido | Sem candidato seguro para overlay adicional |

## 8. Resultado do EXP-007A

- Status: `SEM_CANDIDATO_SEGURO`
- Próxima ação indicada: Não rodar EXP-007B. Os sinais atuais não geraram overlay seguro. Considerar novas fontes de dados ou encerrar FASE 2 como próxima do limite atual.

Interpretação: com os sinais atuais, o meta-learner não encontrou threshold seguro para recuperar FN residual sem custo operacional em FP. Portanto, não há justificativa para rodar EXP-007B com os mesmos sinais.

## 9. FNs residuais pós-C1

Quantidade de FNs residuais únicos: `8`

| transaction_id | valor | rel. meses | first_receiver | pix_random | LGBM | IF | SE | BEH | score | decisão |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E0000020820260103172401005155525` | 142.00 | 216 | 1 | 0 | 0.04782 | 0.47706 | 0.00 | 0.00 | 53.20 | APROVAR |
| `E0000020820260107140624591248525` | 390.00 | 58 | 1 | 1 | 0.00371 | 0.36004 | 0.00 | 0.00 | 17.44 | APROVAR |
| `E0000020820260123162214303665525` | 50.00 | 456 | 1 | 0 | 0.02232 | 0.76187 | 0.00 | 0.00 | 41.87 | APROVAR |
| `E0000020820260213145228155991525` | 29.90 | 267 | 1 | 1 | 0.00005 | 0.70873 | 0.00 | 0.00 | 4.06 | APROVAR |
| `E0000020820260214233339434522525` | 57.88 | 163 | 1 | 1 | 0.00418 | 0.37974 | 0.00 | 0.00 | 21.34 | APROVAR |
| `E0000020820260224213046993254525` | 188.82 | 41 | 1 | 1 | 0.00003 | 0.27674 | 0.00 | 0.00 | 1.75 | APROVAR |
| `E0000020820260227174607667379525` | 498.96 | 317 | 1 | 1 | 0.02834 | 0.76376 | 0.00 | 0.00 | 44.88 | APROVAR |
| `E0000020820260316231246610306525` | 300.00 | 243 | 1 | 0 | 0.02850 | 0.95701 | 0.00 | 0.00 | 46.39 | APROVAR |

Interpretação: os FNs remanescentes devem ser tratados como próximos do limite dos sinais atuais. A maior parte deles não possui convergência suficiente entre LGBM, IF, SE, BEH e score final para justificar nova regra manual segura.

## 10. Procedimento de regressão obrigatório

Antes de qualquer mudança futura no engine, scoring_config, simulação E2E ou artefatos, executar:

```powershell
python -m pytest tests\test_regression_post_fase2.py -q
python -m pytest tests\test_regression_post_fase2.py -q -m slow
```

### Resultado da regressão no momento da geração

- Executado em: `2026-05-09T13:38:10`
- Tudo passou: `True`

#### `E:\Projetos\rebuild_pix\venv\Scripts\python.exe -m pytest tests/test_regression_post_fase2.py -q`

```text
......                                                                   [100%]
6 passed in 2.56s
```

#### `E:\Projetos\rebuild_pix\venv\Scripts\python.exe -m pytest tests/test_regression_post_fase2.py -q -m slow`

```text
.                                                                        [100%]
1 passed, 5 deselected in 2.23s
```

## 11. Critérios de aceite da versão pós-FASE 2

A versão pós-FASE 2 é considerada válida enquanto:

- seed 42 mantiver `TP=347`, `FP=14`, `FN=8`;
- seed 123 mantiver `TP=347`, `FP=12`, `FN=8`;
- C1 recuperar exatamente 1 FN nos dois seeds;
- C1 adicionar 0 FP;
- C1 perder 0 TP;
- teste runtime da transação alvo retornar `CONFIRMAR` com `exp006f_c1_applied=True`;
- `EXP-007A` permanecer sem candidato seguro adicional usando os sinais atuais.

## 12. Próxima etapa

Com o relatório oficial gerado, a próxima etapa da FASE 3 deve ser:

```text
EXP-008C — Rules Catalog e Decision Trace
```

Objetivo: documentar todas as regras ativas, regras rejeitadas, guard rails, thresholds e motivos de decisão do engine.
