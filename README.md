# Rebuild PIX — Motor Híbrido de Detecção de Fraudes PIX

Sistema **ensemble multi-camada** para detecção de fraudes em transações PIX em tempo real, com foco em **Recall ≥ 99%**, **alta explicabilidade** e **taxa de falso positivo < 0,2%**.

Desenvolvido para o BRB (Banco de Brasília), calibrado com **100.355 transações reais** (355 fraudes confirmadas pela GEPFRA).

## Resultados (v3.0.5)

| Métrica | Valor |
|---|---|
| **Recall** | **99,15%** (352/355 fraudes detectadas) |
| **Precision** | **68,87%** |
| **F1-Score** | **0,8129** |
| **FPR** | **0,159%** (159 FP em 100k normais) |
| **Fraudes perdidas** | 3 (valor total: R$ 3.166,76) |
| **Precision BLOQUEAR** | **85,89%** |
| **Recuperação financeira** | **99,86%** (~R$ 2,2M interceptados) |
| **Carga de revisão manual** | 0,107% das transações (~11 por 10k) |

## Arquitetura do Motor
Transação PIX → Feature Engineering │ ┌───────────────┼───────────────┐ ▼ ▼ ▼ LightGBM Isolation SE v3.3 v5.1 Forest v3 BEH v3.0 (superv.) (não-superv.) (regras) │ │ │ └───────────────┼───────────────┘ ▼ Decision Engine v3.0.5 ├── Cascade v3 (C1 + C3) ├── Fast-Approve Override ├── 9 Vetos hierárquicos └── Score 0-100 → APROVAR / CONFIRMAR / BLOQUEAR




### Componentes

| Componente | Tipo | Papel | Performance-chave |
|---|---|---|---|
| **LightGBM v5.1** | ML supervisionado | Backbone — prob. de fraude (0→1) | AUC 0.9996, 52 features |
| **Isolation Forest v3** | ML não-supervisionado | Detector de anomalias complementar | AUC 0.9625, recupera 2/3 FN do LGBM |
| **Cascade v3** | Regras determinísticas | Alta precisão: burst ≥3 e IF+LGBM guard | C1: 100% precision, C3: 95% precision |
| **SE v3.3** | Sistema especialista | 9 padrões de golpes (eng. social) | 262/355 fraudes, Precision 43,5% |
| **BEH v3.0** | Análise comportamental | 7 fatores (velocity + dormancy + profile) | 228/355 fraudes, 19 exclusivas do BEH |
| **Fast-Approve** | Override de segurança | Suprime vetos IF quando LGBM discorda | -48 FP, 0 FN perdidos |

## Estrutura do Projeto
rebuild_pix/ ├── backend/ │ ├── api.py # API REST FastAPI (inferência tempo real) │ ├── artefatos/ # Modelos serializados + configs de produção │ │ ├── model_lightgbm.joblib │ │ ├── model_isolation_forest.joblib │ │ ├── scaler_isolation_forest.joblib │ │ ├── if_ref_raw_train.npy │ │ ├── preprocessing.joblib │ │ ├── scoring_config.json │ │ ├── thresholds_config.json │ │ ├── isolation_forest_config.json │ │ ├── lgbm_features.json │ │ ├── if_features.json │ │ └── diagnostico_features.csv │ ├── core/ # Lógica de negócio (produção) │ │ ├── preprocessing.py # Feature engineering + graph features │ │ ├── decision_engine.py # Motor de decisão ensemble │ │ ├── pipeline_orquestrador.py # Orquestrador do fluxo │ │ ├── social_engineering.py # Detector SE (9 padrões de golpes) │ │ └── behavioral_analytics.py # Análise comportamental (7 fatores) │ ├── modelos/ # Scripts de treino + resultados │ │ ├── train_lgbm_v2.py # LGBM v5.1 (produção) │ │ ├── train_lgbm_v3.py # LGBM v6.1 (graph features — futuro) │ │ ├── train_isolation_forest_v2.py # IF v3 (produção) │ │ ├── resultado_treino_lgbm/ # Métricas e relatórios LGBM v5.1 │ │ ├── resultado_treino_lgbm_v3/ # Métricas e relatórios LGBM v6.1 │ │ └── resultado_treino_if/ # Métricas e relatórios IF v3 │ └── scripts/ │ └── simular_pipeline_e2e_lf.py # Simulação end-to-end leakage-free ├── dados/ │ ├── dados_pix_normais_optimized.csv # Transações normais (100k) │ ├── dados_pix_fraudes_optimized.csv # Transações fraudulentas (355) │ ├── base_treino_final.csv # Dataset processado (model-ready) │ └── scripts_origem/ # Scripts originais de extração Big Data ├── docs/ # Documentação de arquitetura e módulos ├── tests/ # Testes automatizados ├── .gitignore ├── CHANGELOG.md ├── README.md └── requirements.txt




## Como Executar

### Pré-requisitos

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
1. Preprocessamento (gerar dataset de treino)
bash


python backend/core/preprocessing.py
# Output: dados/base_treino_final.csv (100.355 tx, 116 features)
2. Treino dos modelos
bash


# LightGBM v5.1 (produção)
python backend/modelos/train_lgbm_v2.py

# Isolation Forest v3 (produção)
python backend/modelos/train_isolation_forest_v2.py

# LightGBM v6.1 com graph features (experimental — requer ≥6 meses de dados)
python backend/modelos/train_lgbm_v3.py
3. Simulação end-to-end
bash


python backend/scripts/simular_pipeline_e2e_lf.py
4. API REST
bash


uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload



Endpoint	Método	Descrição
/api/v1/analyze	POST	Inferência em tempo real (única transação)
/api/v1/batch	POST	Inferência em lote
/api/v1/health	GET	Health check dos componentes
/docs	GET	Swagger UI
Exemplo de resposta da API
json


{
  "decisao": "BLOQUEAR",
  "score_final": 97.2,
  "confianca": "ALTA",
  "explicabilidade": {
    "motivo_principal": "Múltiplas transferências rápidas para recebedor desconhecido",
    "componentes": {
      "lgbm_score": 0.9985,
      "if_percentile": 0.9991,
      "se_score": 65,
      "se_patterns": ["BURST_VALOR_ALTO", "ESVAZIAMENTO_CONTA"],
      "beh_score": 45,
      "beh_factors": ["FREQUENCIA_BURST", "CONTA_DORMANTE_VALOR_ALTO"]
    }
  }
}
Dataset



Dimensão	Valor
Total de transações	100.355
Fraudes confirmadas	355 (0,35%)
Período	20/dez/2025 → 19/mar/2026
Fonte fraudes	GEPFRA (BRB)
Fonte normais	Extrato PIX BLK (BRB)
Split temporal	90% Dev + 10% Holdout
Leakage	Corrigido — rolling window causal de 90 dias
Documentação Técnica



Documento	Conteúdo
docs/motor_decisao_modelo.md	Arquitetura completa do Engine v3.0.5
docs/relatorio_tecnico_treino_modelos_v2.md	Treino, validação e métricas LGBM + IF
docs/modulo_engenharia_social.md	SE v3.3 — 9 padrões, 31 indicadores
docs/modulo_comportamental.md	BEH v3.0 — 7 fatores, calibração empírica
Stack Técnico



Componente	Tecnologia
ML	LightGBM, scikit-learn (Isolation Forest)
API	FastAPI, Uvicorn
Dados	pandas, NumPy
Serialização	joblib
Python	3.12+
Roadmap
 Graph Feature Engineering — 13 features de grafo temporal já implementadas no preprocessing; aguardando ≥6 meses de dados para cobertura adequada
 Device fingerprinting — 6 features (device, app, auth, session, IP, latência) 100% missing no dataset atual; infraestrutura pronta no BEH
 Persistência de profiles — Migrar Profile Manager de memória para Redis/DynamoDB
 Feedback loop — Decisões da GEPFRA retroalimentando retreino semanal
 Monitoramento de drift — Tracking de distribuição de scores em produção
Motor de Decisão Antifraude PIX v3.0.5 — BRB/GEPFRA — Abril 2026