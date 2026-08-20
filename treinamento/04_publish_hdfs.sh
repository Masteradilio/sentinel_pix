#!/bin/bash
# 04_publish_hdfs.sh
# Publicação de Modelo Aprovado (Caixa 4 do Oozie)
# 
# Este script é ativado apenas se a Caixa 3 (Gatekeeper) devolver Exit Code 0.
# Sua função é sobrescrever os binários antigos de produção no ambiente Hadoop.

echo "=========================================================================="
echo " [Oozie Action] HDFS Model Publishing"
echo "=========================================================================="

HDFS_TARGET="hdfs:///modelos_ml/nudan/nudan_hmo/anomalia_pix/artefatos/"
LOCAL_ARTIFACTS_DIR="treinamento/resultado_treino_r5b22"

echo "INFO: Checando diretório alvo no HDFS..."
hdfs dfs -mkdir -p ${HDFS_TARGET}

echo "INFO: Fazendo upload dos novos artefatos (.joblib e .json)..."

# Em caso de falha de I/O de disco, encerra script
set -e 

hdfs dfs -put -f ${LOCAL_ARTIFACTS_DIR}/*.joblib ${HDFS_TARGET}
hdfs dfs -put -f ${LOCAL_ARTIFACTS_DIR}/*.json ${HDFS_TARGET}
hdfs dfs -put -f ${LOCAL_ARTIFACTS_DIR}/*.csv ${HDFS_TARGET}

echo "✅ SUCESSO: Arquivos promovidos para ${HDFS_TARGET}!"
echo "A API de inferência importará automaticamente estes novos arquivos na sua rotina diária."
exit 0