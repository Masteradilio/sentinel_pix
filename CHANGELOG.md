
---

## CHANGELOG.md

```markdown
# CHANGELOG

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

---

## [3.1.0] - 2026-04-20
### Adicionado
- **EXP-003 (Novo Padrão SE IDOSO_JOVEM_VALOR_MODERADO_RESIDUAL)**: Implementado novo padrão no módulo `SocialEngineeringDetector` focado em detectar fraudes residuais em perfis vulneráveis (jovens ≤25 ou idosos ≥60) com contas recentes (<24m) realizando transferências atípicas moderadas (R$ 1.500 - R$ 15.000) confirmadas por alta anomalia (`if_percentile` ≥ 0.90).
- **EXP-002 (Guard Rail LGBM)**: Inclusão de trava de segurança no `PixDecisionEngine` para vetos IF-based. O sistema agora suprime vetos originados do Isolation Forest quando o score do `LGBM` é considerado de baixa predição de fraude. 

### Alterado
- **EXP-001 (Threshold Confirmar)**: Threshold global ajustado no `scoring_config.json` de `77.0` para `62.0` após validação que demonstrou a viabilidade de recuperar fraudes na zona cinza (com incremento em Recall de ~4pp e ganho substancial de F1 Score).
- **Pipeline Orquestrador (v1.4)**: `pipeline_orquestrador.py` modificado para pré-computar e inserir o `if_percentile` no dicionário de features de forma independente do boost do motor antes da avaliação do `SocialEngineeringDetector`.

---

## [3.0.5] - 2026-04-12
### Adicionado
- **Graph Feature Engineering (GFE)**: 13 features de grafo temporal incremental no `preprocessing.py` v4.1 — cada transação só vê o grafo de transações anteriores (sem leakage). Features incluem sender/receiver degree, pair history, HHI de concentração e z-score de valor.
- **LightGBM v6.1** (`train_lgbm_v3.py`): Script de treino preparado para usar as 13 graph features quando houver ≥6 meses de dados com cobertura adequada.
- **Fast-Approve Override**: Mecanismo de desescalada no Decision Engine — quando LGBM < 0,25 + SE = 0 + BEH = 0, suprime vetos IF-based. Eliminou ~20 FP sem perder nenhum TP.
- **Cascade v3 com LGBM guard**: Regra C3 agora exige LGBM ≥ 0,35 além de IF ≥ 99,5% + burst. Precision C3 subiu de 61% → 95% (-56 FP).
- **Simulação E2E leakage-free** (`simular_pipeline_e2e_lf.py`): Script de validação end-to-end com todas as correções de leakage aplicadas.

### Alterado
- **Decision Engine v3.0.5**: Precision subiu de 62,97% → 68,87% (+5,9pp). FP total de 207 → 159 (-48). Recall mantido em 99,15%. F1 de 0,770 → 0,813.
- **Preprocessing v4.1**: Pipeline completo com 6 fases (load → features → leakage fix → graph features → seleção → PixPreprocessor). Output: `base_treino_final.csv` com 116 colunas.
- **Reorganização do projeto**: Limpeza de 69 artefatos de desenvolvimento (scripts exploratórios, simulações, relatórios intermediários) movidos para `_archive/`. Estrutura de produção enxuta.

### Documentação
- **Motor de Decisão** (`motor_decisao_modelo.md`): Relatório completo da arquitetura v3.0.5 com análise de FN/FP, contribuição marginal dos componentes e métricas operacionais.
- **Treino de Modelos** (`relatorio_tecnico_treino_modelos_v2.md`): Documentação do LGBM v5.1 + IF v3 com validação cruzada temporal, análise de overfitting e benchmark com a indústria.
- **Módulo SE** (`modulo_engenharia_social.md`): Documentação completa do SE v3.3 — 9 padrões, 31 indicadores, metodologia de 6 frentes de calibração.
- **Módulo BEH** (`modulo_comportamental.md`): Documentação completa do BEH v3.0 — 7 fatores, evolução v2.1→v3.0, 19 fraudes exclusivas capturadas.

---

## [3.0.0] - 2026-04-11
### Adicionado
- **Behavioral Analytics v3.0** (`behavioral_analytics.py`): Redesign completo — de 15 fatores (78k FP) para 7 fatores validados (1.797 FP). Três novos fatores dormancy descobertos na Frente B2: `CONTA_DORMANTE_VALOR_ALTO` (Precision 65%, Recall 39,7%), `CONTA_DORMANTE_IDOSO` (Precision 85,6%), `PRIMEIRA_TX_VALOR_ALTO` (Precision 72,4%). **19 fraudes exclusivas** não detectadas por nenhum outro módulo.
- **Social Engineering v3.3** (`social_engineering.py`): 9 padrões calibrados via 6 frentes de análise. Destaques: `BURST_INTENSO_RAPIDO` (100% precision, 0 FP), `BURST_VALOR_ALTO` (78,8% precision, gate R$500). Cobertura de 262/355 fraudes (73,8%).

### Alterado
- **Correção de leakage temporal**: 14 features trimestrais recalculadas com rolling window estritamente causal (90 dias, apenas transações anteriores por CPF). AUC LGBM: 0,9998 → 0,9996 (degradação mínima — modelo aprendeu padrões reais).
- **LightGBM v5.1** (`train_lgbm_v2.py`): Retreino com features leakage-free. 52 features (45 core + 7 extras). Holdout: F1 0,9112, Recall 96,25%, Precision 86,52%.
- **Isolation Forest v3** (`train_isolation_forest_v2.py`): Features reduzidas de 22 → 13 (remoção de 10 com importance negativa). Treino segmentado (apenas normais regulares). AUC: 0,8919 → 0,9625 (+7,1pp). Gap de separação fraude-normal dobrou.

### Removido
- **11 fatores BEH com performance negativa**: DEVICE_NOVO (Lift 0,57x, 78k FP), DEVICE_NOVO_PREMIUM, 7 fatores com 0 ativações (features 100% missing), RENDA_INCOMPATIVEL, VALOR_CONCENTRADO_TRIMESTRE.
- **9 indicadores SE anti-indicadores**: `valor_alto_vs_historico` (Lift 0,25x), `escalada_valores` (Lift 0,21x), `horario_noturno` (0 fraudes), entre outros.
- **`rule_engine.py`**: Módulo obsoleto substituído pelo Cascade v3 integrado ao Decision Engine.

---

## [2.1.1] - 2026-03-22
### Alterado
- **Otimização de Precisão**: Regra C6 do Cascade ajustada (exige 4 sinais). Precision do bloqueio elevada para 58,7% (50 FP). FN mantido em zero.
- **Calibragem de âncoras** (`scoring_config.json`): Transações em CONFIRMAR reduzidas de ~700 para 4. Precision da camada CONFIRMAR de 9% → 56,8%.

---

## [2.1.0] - 2026-03-22
### Adicionado
- **API com Explicabilidade**: Bloco `explicabilidade` no payload de resposta com mensagem CX-Friendly para exibição ao cliente.
- **Decision Engine v2.1**: Integração LGBM + Cascade Rules + Isolation Forest (boost condicional).
- **Módulos SE v1.0 e BEH v1.0**: Implementação inicial de Social Engineering (12 padrões) e Behavioral Analytics (15 fatores).

### Alterado
- **Falsos Negativos Zero** (no dataset de teste): Âncoras mapeando `lgbm_threshold` 0,08 para faixa de bloqueio 85+.
- **FPR reduzido para 0,28%**: IF Boost thresholds: `if_high=0.99`, `if_very_high=0.9994`.
- **Cascade refatorada**: C3 desativada, C6 ajustada (gatilho 0,015).

---

## [0.0.1] - 2026-03-06
### Adicionado
- Documento inicial de requisitos e arquitetura (`docs/PRD.md`).
- Lista de features para MVP (`docs/lista_de_features.md`).
- README, CHANGELOG e `requirements.txt`.
- Scripts de backend e ingestão de dados em `/dados/scripts_origem`.
- Ambiente virtual e estrutura de diretórios.

---

[3.0.5]: https://github.com/adilio/rebuild_pix/compare/v3.0.0...v3.0.5
[3.0.0]: https://github.com/adilio/rebuild_pix/compare/v2.1.1...v3.0.0
[2.1.1]: https://github.com/adilio/rebuild_pix/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/adilio/rebuild_pix/compare/v0.0.1...v2.1.0
[0.0.1]: https://github.com/adilio/rebuild_pix/releases/tag/v0.0.1
