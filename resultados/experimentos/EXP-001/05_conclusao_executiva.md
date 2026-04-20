# EXP-001 — Conclusão Executiva

> **Experimento:** Ajuste do Threshold Final (77 -> 62)
> **Status:** ✅ APROVADO
> **Validação cruzada:** ✅ CONFIRMADA
> **Tempo total:** 548.3s (9.1min)

---

## 1. Resumo do Resultado

| Métrica | Baseline (t=77) | Vencedor (V1 (threshold=62, F1 ótimo)) | Delta |
|---|---:|---:|---:|
| **TP** | 332 | **346** | **+14** |
| **FP** | 8 | **17** | **+9** |
| **FN** | 23 | **9** | **-14** |
| **Precision** | 97.65% | **95.32%** | -0.0233 |
| **Recall** | 93.52% | **97.46%** | +0.0394 |
| **F1** | 0.9554 | **0.9638** | +0.0084 |
| **FPR** | 0.1417% | **0.3012%** | — |

---

## 2. Comparativo de Variantes

| variante_id   |   threshold_confirmar |   TP |   FP |   FN |   Precision |   Recall |     F1 |
|:--------------|----------------------:|-----:|-----:|-----:|------------:|---------:|-------:|
| BASELINE      |                    77 |  332 |    8 |   23 |      0.9765 |   0.9352 | 0.9554 |
| V1            |                    62 |  346 |   17 |    9 |      0.9532 |   0.9746 | 0.9638 |
| V2            |                    65 |  346 |   17 |    9 |      0.9532 |   0.9746 | 0.9638 |
| V3            |                    70 |  338 |   12 |   17 |      0.9657 |   0.9521 | 0.9589 |

---

## 3. Threshold Sweep Fino

**Melhor F1 no sweep:** F1=0.9638 @ threshold=62 (TP=346, FP=17, FN=9)

|   threshold |       TP |      FP |      FN |   Precision |   Recall |     F1 |
|------------:|---------:|--------:|--------:|------------:|---------:|-------:|
|     55.0000 | 347.0000 | 23.0000 |  8.0000 |      0.9378 |   0.9775 | 0.9572 |
|     56.0000 | 347.0000 | 23.0000 |  8.0000 |      0.9378 |   0.9775 | 0.9572 |
|     57.0000 | 347.0000 | 22.0000 |  8.0000 |      0.9404 |   0.9775 | 0.9586 |
|     58.0000 | 347.0000 | 22.0000 |  8.0000 |      0.9404 |   0.9775 | 0.9586 |
|     59.0000 | 347.0000 | 22.0000 |  8.0000 |      0.9404 |   0.9775 | 0.9586 |
|     60.0000 | 347.0000 | 21.0000 |  8.0000 |      0.9429 |   0.9775 | 0.9599 |
|     61.0000 | 346.0000 | 21.0000 |  9.0000 |      0.9428 |   0.9746 | 0.9584 |
|     62.0000 | 346.0000 | 17.0000 |  9.0000 |      0.9532 |   0.9746 | 0.9638 |
|     63.0000 | 346.0000 | 17.0000 |  9.0000 |      0.9532 |   0.9746 | 0.9638 |
|     64.0000 | 346.0000 | 17.0000 |  9.0000 |      0.9532 |   0.9746 | 0.9638 |
|     65.0000 | 346.0000 | 17.0000 |  9.0000 |      0.9532 |   0.9746 | 0.9638 |
|     66.0000 | 343.0000 | 17.0000 | 12.0000 |      0.9528 |   0.9662 | 0.9594 |
|     67.0000 | 342.0000 | 16.0000 | 13.0000 |      0.9553 |   0.9634 | 0.9593 |
|     68.0000 | 342.0000 | 16.0000 | 13.0000 |      0.9553 |   0.9634 | 0.9593 |
|     69.0000 | 338.0000 | 13.0000 | 17.0000 |      0.9630 |   0.9521 | 0.9575 |
|     70.0000 | 338.0000 | 12.0000 | 17.0000 |      0.9657 |   0.9521 | 0.9589 |
|     71.0000 | 336.0000 | 12.0000 | 19.0000 |      0.9655 |   0.9465 | 0.9559 |
|     72.0000 | 335.0000 | 11.0000 | 20.0000 |      0.9682 |   0.9437 | 0.9558 |
|     73.0000 | 334.0000 | 10.0000 | 21.0000 |      0.9709 |   0.9408 | 0.9557 |
|     74.0000 | 333.0000 | 10.0000 | 22.0000 |      0.9708 |   0.9380 | 0.9542 |
|     75.0000 | 332.0000 |  9.0000 | 23.0000 |      0.9736 |   0.9352 | 0.9540 |
|     76.0000 | 332.0000 |  8.0000 | 23.0000 |      0.9765 |   0.9352 | 0.9554 |
|     77.0000 | 332.0000 |  8.0000 | 23.0000 |      0.9765 |   0.9352 | 0.9554 |
|     78.0000 | 329.0000 |  8.0000 | 26.0000 |      0.9763 |   0.9268 | 0.9509 |
|     79.0000 | 328.0000 |  8.0000 | 27.0000 |      0.9762 |   0.9239 | 0.9493 |
|     80.0000 | 328.0000 |  8.0000 | 27.0000 |      0.9762 |   0.9239 | 0.9493 |

---

## 4. Análise dos FN Recuperados

**Total recuperado:** 14 fraudes

**Perfil das fraudes recuperadas:**
- Valor mediano: R$ 995.00
- Valor máximo: R$ 10,000.00
- Idade mediana: 44 anos
- Idosos (60+): 4
- Jovens (<25): 3
- Score final mediano: 68.9
- Com first_receiver_flag: 12
- Com perfil vulneravel: 1

---

## 5. Análise dos FP Novos

**Total novos FP:** 9

**Perfil dos novos FP:**
- Valor mediano: R$ 1,173.74
- Valor máximo: R$ 7,350.00
- Idade mediana: 49 anos
- Idosos (60+): 3
- Score final mediano: 69.1

**Alertas de segurança:**
- ⚠️ INFO: 2 novos FP com valor >= R$5.000. Revisar manualmente antes de deploy.

---

## 6. Validação Cruzada

- **Sample:** 6,000 tx (seed=123)
- **Fraudes no sample:** 355
- **F1 baseline:** 0.9623
- **F1 vencedor:** 0.9692
- **Delta F1:** +0.0069
- **Delta Recall:** +0.0394
- **Interpretação:** VALIDADO: F1 do vencedor > F1 do baseline no sample independente.

---

## 7. Critérios de Aceitação

| Critério | Valor obtido | Threshold | Status |
|---|---:|---:|:---:|
| delta_F1 >= 0.005 | 0.0084 | 0.0050 | ✅ |
| Recall >= 95% | 0.9746 | 0.9500 | ✅ |
| FPR <= 0.50% | 0.0030 | 0.0050 | ✅ |
| Precision >= 90% | 0.9532 | 0.9000 | ✅ |

---

## 8. Recomendação Final

### ✅ APROVAR — Deploy da variante **V1** (threshold=62)

**Próximos passos:**

1. Atualizar `backend/artefatos/scoring_config.json`:
   ```json
   {
     "score_final_threshold_confirmar": 62
   }
   ```
2. Incrementar `engine_version`: 3.0.5 → 3.0.6
3. Criar PR com link para este relatório
4. Monitorar métricas por 48h pós-deploy
5. Seguir para o próximo experimento (EXP-004 — Rate Limiting)

---

## 9. Metadata de Execução

- **Experimento:** EXP-001
- **Sample size:** 6,000 tx
- **Seed principal:** 42
- **Seed validação:** 123
- **Tempo total:** 548.3s (9.1min)

---

*Gerado automaticamente por `experimentos/exp_001_threshold_final/run_exp_001.py`*
