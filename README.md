# Rebuild PIX Fraud Detection (v2.1)

Este repositório contém os scripts, a API e os artefatos necessários para construir e operar um Motor Híbrido de Detecção de Fraudes em transações PIX, focado em atingir Falsos Negativos Zero com Alta Explicabilidade.

## Estrutura do projeto

```
backend/               
  api.py                # API REST FastAPI para inferência em tempo real
  pipeline_orquestrador.py # Orquestrador principal do fluxo de decisão
  core/                 # Lógica de negócio e detecção
    behavioral_analytics.py # Análise de perfil, dispositivo e sessão
    social_engineering.py   # Detecção de padrões de golpes e vulnerabilidade
    decision_engine.py      # Motor de regras, ensemble e cálculo de score final
    preprocessing.py        # Limpeza e normalização de features
  modelos/              # Scripts de treinamento
    train_lgbm.py
  artefatos/            # Modelos salvos (LightGBM, IF) e JSONs de configuração
dados/                  # Arquivos CSV de entrada e saída
docs/                   # Documentação projetual e de arquitetura
  PRD.md
  behavioral_analytics_v2.md
  engenharia_social.md
relatorio/              # Relatórios de performance e dashboards gerados
requirements.txt        # Dependências Python
README.md               # Este arquivo
CHANGELOG.md            # Histórico de versões
```

## Visão Geral do Sistema (v2.1)

- **Objetivo**: Proteger transações PIX em tempo real através de um motor híbrido de decisão que garanta a detecção de 100% das fraudes conhecidas (FNR = 0) mantendo a taxa de Falsos Positivos sob controle estrito (< 0,3%).
- **Componentes do Motor Híbrido**:
  1. **LightGBM**: Modelo preditivo principal (Supervisionado).
  2. **Cascade Rules**: Regras determinísticas de fallback para capturar falsos negativos residuais do modelo.
  3. **Isolation Forest (IF Boost)**: Modelo não-supervisionado de anomalias com atuação condicional (boost).
  4. **Engenharia Social (SE)**: Detector de 12 padrões de golpes mapeados no Brasil.
  5. **Behavioral Analytics**: Detector de 15 fatores de risco de dispositivo, rede e sessão (integrado com Risk Score do Topaz).
- **Explicabilidade**: A API fornece uma saída estruturada e amigável (CX-Friendly) detalhando exatamente o motivo de bloqueio ou confirmação adicional de cada transação.

## Como Executar a API

Ative o ambiente virtual e inicie o servidor:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Iniciar a API REST
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints disponíveis (Swagger UI disponível em `/docs`):
- `POST /api/v1/analyze`: Inferência em tempo real de uma única transação
- `POST /api/v1/batch`: Inferência em lote
- `GET /api/v1/health`: Monitoramento dos componentes

## Como Avaliar o Pipeline

Para gerar o dashboard de métricas e testar toda a lógica contra o dataset de testes:

```powershell
python backend\teste_pipeline_relatorio.py
```
Outputs serão salvos em `/relatorio/`.

## Documentação Adicional

- Para entender a estratégia do Ensemble, veja `docs/PRD.md`.
- Para os detalhes dos padrões de golpes detectados, veja `docs/engenharia_social.md`.
- Para detalhes dos parâmetros de dispositivo e sessão, veja `docs/behavioral_analytics_v2.md`.

## Notas

- Caminhos de dados e artefatos são construídos dinamicamente usando `DADOS_DIR` e `ARTEFACTOS_DIR`.
- Utilize cópias locais dos CSVs em `/dados` para testar.

---

Este README será atualizado conforme o projeto evolui.
