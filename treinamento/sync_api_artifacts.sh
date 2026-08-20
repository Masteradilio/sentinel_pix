#!/bin/bash
# sync_api_artifacts.sh
# 
# Script executado via Cron na máquina onde roda o backend (API).
# Objetivo: Baixar os artefatos gerados pelo workflow do Oozie, substituindo os antigos.
# Deve rodar diariamente (ex: 04:00 AM) para garantir a integridade da inferência.

API_ARTIFACTS_DIR="/caminho/para/projeto/backend/artefatos"
HDFS_SOURCE="hdfs:///modelos_ml/nudan/nudan_hmo/anomalia_pix/artefatos/*"

echo "=================================================="
echo "Sincronização Diária de Artefatos ML PIX - $(date)"
echo "=================================================="

# 1. Puxa os dados do HDFS para a pasta local da API
echo "Baixando artefatos do HDFS..."
hdfs dfs -get -f ${HDFS_SOURCE} ${API_ARTIFACTS_DIR}

if [ $? -eq 0 ]; then
    echo "✅ Arquivos transferidos com sucesso."
    
    # 2. Reinicia o servidor ASGI (Uvicorn/Gunicorn) via systemd (se aplicável)
    # Isso engatilha o método app.on_event("startup") do FastAPI e recarrega os joblibs na RAM.
    # Exemplo:
    # sudo systemctl restart api-pix-fraude
    echo "INFO: O processo da API precisa ser reiniciado para espelhar a memória."
else
    echo "❌ Falha ao tentar sincronizar do HDFS. A API permanecerá usando a versão em cache."
    exit 1
fi
