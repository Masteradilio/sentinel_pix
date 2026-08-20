#!/bin/bash
set -e

echo "[INFO] Iniciando execucao HQL"
echo "[INFO] HQL_FILE=${HQL_FILE}"
echo "[INFO] DB_WORK=${DB_WORK}"
echo "[INFO] DT_INI=${DT_INI}"
echo "[INFO] DT_FIM=${DT_FIM}"

if [ -z "${HQL_FILE}" ]; then
  echo "[ERRO] Variavel HQL_FILE nao informada"
  exit 10
fi

if [ ! -f "${HQL_FILE}" ]; then
  echo "[ERRO] Arquivo HQL nao encontrado no container: ${HQL_FILE}"
  ls -la
  exit 11
fi

# Remove CRLF caso o arquivo tenha sido salvo no Windows.
sed -i 's/\r$//' "${HQL_FILE}"

hive \
  --hivevar DB_WORK="${DB_WORK}" \
  --hivevar DT_INI="${DT_INI}" \
  --hivevar DT_FIM="${DT_FIM}" \
  -f "${HQL_FILE}"

RC=$?

echo "[INFO] Hive terminou com RC=${RC}"
exit ${RC}