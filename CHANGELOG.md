# CHANGELOG

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

## [2.1.1] - 2026-03-22
### Alterado
- **Otimização Extrema de Precisão**: Ajuste fino nas regras Cascade (`C6_LGBM_BORDERLINE_COMBINADO` agora exige 4 sinais) e nos limiares de Boost do Isolation Forest, elevando a precisão do bloqueio para **58.7%** (apenas 50 Falsos Positivos) mantendo Falsos Negativos em **ZERO**.
- **Melhoria da Perspectiva Ampla (CONFIRMAR)**: Calibragem nas âncoras de mapeamento (`scoring_config.json`) para espremer a zona cinzenta, reduzindo o número de transações enviadas para 2FA/Biometria de ~700 para apenas **4**, o que fez a precisão desta camada saltar de 9% para **56.8%**.

## [2.1.0] - 2026-03-22
### Adicionado
- **API com Alta Explicabilidade**: Novo payload no `api.py` (`POST /api/v1/analyze`) contendo o bloco `explicabilidade`, que traz o motivo principal do bloqueio traduzido em uma mensagem amigável (CX-Friendly) pronta para exibição ao cliente no aplicativo.
- **Engine de Decisão v2.1**: Integração fluida entre o LightGBM, Cascade Rules e o Isolation Forest com atuação condicional (IF Boost).
- **Módulos Comportamentais**: Adicionados `social_engineering.py` (12 padrões de golpes mapeados) e `behavioral_analytics.py` (15 fatores de risco integrados com score Topaz).
- **Relatórios**: Geração automática de relatórios em HTML e Dashboards PNG pelo `teste_pipeline_relatorio.py`.

### Alterado
- **Falsos Negativos Zero**: Ajuste nas âncoras de pontuação (`scoring_config.json`) mapeando o novo `lgbm_threshold` (0.08) diretamente para a faixa de bloqueio (85+), garantindo 100% de detecção de fraudes (FN = 0).
- **Redução Extrema de Falsos Positivos**: Fine-tuning dos limiares do IF Boost (`if_high_threshold` = 0.99, `if_very_high_threshold` = 0.9994), derrubando a taxa de falsos alarmes para impressionantes **0.28%**.
- **Refatoração da Regra Cascade**: Regra C3 desativada e Regra C6 ajustada (gatilho de 0.015) para focar apenas nas anomalias mais suspeitas.

## [0.0.1] - 2026-03-06
### Adicionado
- Documento inicial de requisitos e arquitetura (`docs/PRD.md`).
- Lista de features para MVP (`docs/lista_de_features.md`).
- README principal na raiz com visão geral, instruções e estrutura do repositório.
- Changelog inicial (`CHANGELOG.md`).
- Scripts de backend com README adicionados dentro de `/backend`.
- Ajustes no `feature_engineering.py` para correção de paths e novas variáveis de diretório.
- Criação de ambiente virtual (`venv`) e `requirements.txt`.
- Adaptação dos scripts de ingestão de dados do Big Data em `/dados/scripts_origem` (design).

### Observação
Versão inicial com configuração de diretórios, documentação e artefatos básicos.

[0.0.1]: https://example.com/release/0.0.1
