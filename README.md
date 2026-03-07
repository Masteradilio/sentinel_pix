# Rebuild PIX Fraud Detection

Este repositório contém os scripts e artefatos necessários para construir um MVP de detecção de fraude/anomalia em transações PIX.

## Estrutura do projeto

```
backend/               # código de engenharia de features e treino
  feature_engineering.py
  preprocessing.py
  artefatos/            # modelos, métricas e artefatos salvos
  modelos/
    train_lgbm.py
dados/                  # arquivos CSV de entrada e saída
  dados_pix_normais.csv
  dados_fraudes_pix.csv
  dados_features_mobile.csv
  base_mvp_features.csv
  base_mvp_model_ready.csv
docs/                   # documentação projetual
    lista_de_features.md
    PRD.md
requirements.txt        # dependências Python
venv/                   # ambiente virtual (não comitado)
README.md               # este arquivo
CHANGELOG.md            # histórico de versões
```

## Visão geral

- **Objetivo**: montar um motor híbrido de detecção de fraude em tempo real, combinando regras, modelos supervisionados, anomalia e behavioral analytics.
- **Fontes**: bases de PIX normais, fraudes e dados mobile/app.
- **Pipeline**:
  1. Carregar e padronizar CSVs brutos.
  2. Marcar `is_fraud` e unir datasets.
  3. Engenharia de features conforme lista do MVP (veja `docs/lista_de_features.md`).
  4. Salvar bases consolidadas em `/dados`.
  5. Treinar modelo (LightGBM) e gerar artefatos.

## Como executar

Ative o ambiente virtual e instale dependências:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Execute os scripts em ordem:

```powershell
python backend\feature_engineering.py
python backend\preprocessing.py
python backend\modelos\train_lgbm.py
```

Outputs:
- `/dados/base_mvp_features.csv` (features brutas)
- `/dados/base_mvp_model_ready.csv` (dados prontos para modelo)
- artefatos em `/backend/artefatos/`

## Features

A lista completa e a priorização estão documentadas em `docs/lista_de_features.md`.

## Visão de arquitetura

Consulte `docs/PRD.md` para entender a estratégia geral, ensemble híbrido e recomendações de produção.

## Notas

- Caminhos de dados e artefatos são construídos dinamicamente usando `DADOS_DIR` e `ARTEFACTOS_DIR`.
- Utilize cópias locais dos CSVs em `/dados` para testar.

---

Este README será atualizado conforme o projeto evolui.
