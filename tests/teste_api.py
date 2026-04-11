"""
teste_api.py — Teste completo de todos os endpoints da API Antifraude PIX
=========================================================================

Uso:
  1. Inicie a API:  uvicorn api:app --host 0.0.0.0 --port 8000
  2. Em outro terminal:  python teste_api.py

Testa:
  - GET  /                    → Info
  - GET  /api/v1/health       → Health check
  - GET  /api/v1/status       → Status detalhado
  - GET  /api/v1/metrics      → Métricas
  - POST /api/v1/analyze      → 5 cenários (aprovar, confirmar, bloquear, idoso, mínimo)
  - POST /api/v1/batch        → Lote com 3 transações
  - POST /api/v1/cache/reset  → Reset cache
  - POST /api/v1/analyze      → Validação (vl_pix negativo → 422)
"""

import json
import sys
import time
import requests

BASE_URL = "http://localhost:8001"
PASS = 0
FAIL = 0


def _print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _print_result(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    icon = "✅" if passed else "❌"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {icon} {name}{f' — {detail}' if detail else ''}")


def _check_response(name: str, resp, expected_status: int = 200, required_fields=None):
    ok = resp.status_code == expected_status
    detail = f"HTTP {resp.status_code}"

    if ok and required_fields:
        data = resp.json()
        missing = [f for f in required_fields if f not in data]
        if missing:
            ok = False
            detail += f" | campos faltando: {missing}"
        else:
            detail += " | campos OK"

    _print_result(name, ok, detail)
    return ok


# =========================================================
# 1. INFO
# =========================================================
def test_root():
    _print_header("1. GET / — Info da API")
    resp = requests.get(f"{BASE_URL}/")
    _check_response("Root", resp, 200, ["name", "version", "endpoints"])
    data = resp.json()
    print(f"     API: {data.get('name')} v{data.get('version')}")
    print(f"     Pipeline: {data.get('pipeline', 'N/A')}")


# =========================================================
# 2. HEALTH
# =========================================================
def test_health():
    _print_header("2. GET /api/v1/health — Health Check")
    resp = requests.get(f"{BASE_URL}/api/v1/health")
    _check_response("Health", resp, 200, ["status", "components"])
    data = resp.json()
    print(f"     Status: {data.get('status')}")
    components = data.get("components", {})
    for comp, status in components.items():
        icon = "✅" if status else "⚠️"
        print(f"     {icon} {comp}: {status}")


# =========================================================
# 3. STATUS
# =========================================================
def test_status():
    _print_header("3. GET /api/v1/status — Status Detalhado")
    resp = requests.get(f"{BASE_URL}/api/v1/status")
    _check_response("Status", resp, 200, ["pipeline", "config", "metrics"])
    data = resp.json()
    config = data.get("config", {})
    print(f"     Threshold CONFIRMAR: {config.get('threshold_confirmar')}")
    print(f"     Threshold BLOQUEAR:  {config.get('threshold_bloquear')}")
    print(f"     SHAP ativo:          {config.get('shap_enabled')}")
    print(f"     Cascade ativo:       {config.get('cascade_enabled')}")


# =========================================================
# 4. METRICS
# =========================================================
def test_metrics():
    _print_header("4. GET /api/v1/metrics — Métricas")
    resp = requests.get(f"{BASE_URL}/api/v1/metrics")
    _check_response("Metrics", resp, 200, ["total_requests", "decisions"])
    data = resp.json()
    print(f"     Total requests: {data.get('total_requests')}")
    print(f"     Uptime: {data.get('uptime_seconds')}s")


# =========================================================
# 5. ANALYZE — 5 cenários
# =========================================================
def test_analyze():
    _print_header("5. POST /api/v1/analyze — Cenários de Transação")

    scenarios = {
        "Normal (deve APROVAR)": {
            "cd_pix": "E00000208202603261400TEST000001",
            "dt_pix": "2026-03-26 14:30:00",
            "cd_cpf_pagador": "12345678901",
            "cd_cpf_cnpj_recebedor": "98765432100",
            "ds_chave_pix": "98765432100",
            "ds_tipo_chave": "CPF",
            "vl_pix": 150.00,
            "qt_total_pix_trimestre": 45,
            "vl_mediana_pix_trimestre": 200.0,
            "vl_desvio_padrao_pix_trimestre": 80.0,
            "qt_intervalo_transacao_minuto": 1440.0,
            "qt_intervalo_mediana_trimestre": 1200.0,
            "qt_intervalo_desvio_padrao_trimestre": 300.0,
            "qt_pix_dia_maximo_trimestre": 3,
            "device_name": "iPhone 15",
            "app_version": "7.12.0",
            "ip_address": "192.168.1.50",
            "latencia_rede_ms": 35.0,
            "vl_latencia_rede_media_trimestre": 38.0,
            "tempo_interacao_ms": 8000.0,
            "vl_tempo_interacao_medio_trimestre": 7500.0,
            "tempo_processamento_host_ms": 100.0,
            "metodo_autenticacao": "biometria",
            "session_id": "sess_normal_001",
            "cd_retorno": "00",
            "topaz_risk_score": 1.0,
            "topaz_transacao_rejeitada": 0,
            "is_agendamento_recorrente": "false",
            "qt_aparelhos_distintos_trimestre": 1,
            "nr_idade": 35,
            "qt_tempo_relacionamento_mes": 84,
            "vl_renda_cliente": 8000.0,
            "ds_sexo": "M",
            "ds_estado_civil": "CASADO",
            "ds_segmento": "VAREJO",
            "qt_dependentes": 2,
        },
        "Suspeito (deve CONFIRMAR)": {
            "cd_pix": "E00000208202603261400TEST000002",
            "dt_pix": "2026-03-26 02:30:00",
            "cd_cpf_pagador": "11122233344",
            "cd_cpf_cnpj_recebedor": "55566677788",
            "ds_chave_pix": "abc123-random-key",
            "ds_tipo_chave": "CHAVE ALEATORIA",
            "vl_pix": 2500.00,
            "qt_total_pix_trimestre": 5,
            "vl_mediana_pix_trimestre": 300.0,
            "vl_desvio_padrao_pix_trimestre": 150.0,
            "qt_intervalo_transacao_minuto": 15.0,
            "qt_intervalo_mediana_trimestre": 500.0,
            "qt_intervalo_desvio_padrao_trimestre": 200.0,
            "qt_pix_dia_maximo_trimestre": 3,
            "device_name": "Samsung Galaxy S23",
            "app_version": "7.10.0",
            "metodo_autenticacao": "senha",
            "topaz_risk_score": 3.0,
            "nr_idade": 62,
            "qt_tempo_relacionamento_mes": 36,
            "vl_renda_cliente": 4000.0,
            "ds_sexo": "F",
            "ds_estado_civil": "CASADO",
            "ds_segmento": "VAREJO",
            "qt_dependentes": 1,
        },
        "Fraude (deve BLOQUEAR)": {
            "cd_pix": "E00000208202603261400TEST000003",
            "dt_pix": "2026-03-26 03:15:00",
            "cd_cpf_pagador": "99887766554",
            "cd_cpf_cnpj_recebedor": "11223344556",
            "ds_chave_pix": "xyz789-random-fraud",
            "ds_tipo_chave": "CHAVE ALEATORIA",
            "vl_pix": 4999.00,
            "qt_total_pix_trimestre": 1,
            "vl_mediana_pix_trimestre": 0.0,
            "vl_desvio_padrao_pix_trimestre": 0.0,
            "qt_intervalo_transacao_minuto": 0.0,
            "qt_intervalo_mediana_trimestre": 0.0,
            "qt_intervalo_desvio_padrao_trimestre": 0.0,
            "qt_pix_dia_maximo_trimestre": 1,
            "metodo_autenticacao": "senha",
            "topaz_risk_score": 4.0,
            "topaz_transacao_rejeitada": 0,
            "nr_idade": 78,
            "qt_tempo_relacionamento_mes": 2,
            "vl_renda_cliente": 3200.0,
            "ds_sexo": "F",
            "ds_estado_civil": "VIUVA",
            "ds_segmento": "VAREJO",
            "qt_dependentes": 0,
        },
        "Idoso vulnerável + chave aleatória": {
            "cd_pix": "E00000208202603261400TEST000004",
            "dt_pix": "2026-03-26 10:00:00",
            "cd_cpf_pagador": "44455566677",
            "cd_cpf_cnpj_recebedor": "88899900011",
            "ds_chave_pix": "random-key-idoso",
            "ds_tipo_chave": "CHAVE ALEATORIA",
            "vl_pix": 3000.00,
            "qt_total_pix_trimestre": 2,
            "vl_mediana_pix_trimestre": 100.0,
            "vl_desvio_padrao_pix_trimestre": 50.0,
            "qt_intervalo_transacao_minuto": 5.0,
            "qt_intervalo_mediana_trimestre": 2000.0,
            "qt_intervalo_desvio_padrao_trimestre": 500.0,
            "qt_pix_dia_maximo_trimestre": 2,
            "metodo_autenticacao": "senha",
            "topaz_risk_score": 3.0,
            "nr_idade": 82,
            "qt_tempo_relacionamento_mes": 120,
            "vl_renda_cliente": 2500.0,
            "ds_sexo": "F",
            "ds_estado_civil": "VIUVA",
            "ds_segmento": "VAREJO",
            "qt_dependentes": 0,
        },
        "Dados mínimos (só obrigatórios)": {
            "cd_pix": "E00000208202603261400TEST000005",
            "dt_pix": "2026-03-26 12:00:00",
            "cd_cpf_pagador": "00011122233",
            "vl_pix": 50.00,
        },
    }

    for name, tx in scenarios.items():
        t0 = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/api/v1/analyze", json=tx)
        elapsed = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            data = resp.json()
            decisao = data.get("decisao", "?")
            score = data.get("score_final", -1)
            has_shap = "explicabilidade" in data and data["explicabilidade"] is not None
            has_cx = "cx" in data and data["cx"] is not None
            has_se = "social_engineering" in data
            has_beh = "behavioral" in data

            detail = (
                f"{decisao} | Score={score:.1f} | "
                f"SHAP={'✅' if has_shap else '—'} | "
                f"CX={'✅' if has_cx else '—'} | "
                f"SE={'✅' if has_se else '—'} | "
                f"BEH={'✅' if has_beh else '—'} | "
                f"{elapsed:.0f}ms"
            )
            _print_result(name, True, detail)

            # Mostrar CX se presente
            if has_cx:
                cx = data["cx"]
                print(f"       💬 CX: {cx.get('motivo_principal', '')[:80]}")
                print(f"       🎯 Fator: {cx.get('fator_predominante', '')}")

            # Mostrar top 3 SHAP se presente
            if has_shap:
                top = data["explicabilidade"].get("top_features", [])[:3]
                for f in top:
                    arrow = "▲" if f["direction"] == "aumenta_risco" else "▼"
                    print(
                        f"       {arrow} {f['label']}: "
                        f"SHAP={f['shap_value']:+.4f} ({f['impact_pct']}%)"
                    )
        else:
            _print_result(name, False, f"HTTP {resp.status_code}: {resp.text[:100]}")


# =========================================================
# 6. BATCH
# =========================================================
def test_batch():
    _print_header("6. POST /api/v1/batch — Lote de 3 transações")

    batch = {
        "transactions": [
            {
                "cd_pix": "E00000208202603261400BATCH00001",
                "dt_pix": "2026-03-26 14:00:00",
                "cd_cpf_pagador": "12345678901",
                "vl_pix": 100.0,
                "nr_idade": 30,
                "qt_tempo_relacionamento_mes": 60,
                "qt_total_pix_trimestre": 50,
                "vl_mediana_pix_trimestre": 150.0,
                "metodo_autenticacao": "biometria",
            },
            {
                "cd_pix": "E00000208202603261400BATCH00002",
                "dt_pix": "2026-03-26 03:00:00",
                "cd_cpf_pagador": "99887766554",
                "vl_pix": 4999.0,
                "nr_idade": 75,
                "qt_tempo_relacionamento_mes": 3,
                "ds_tipo_chave": "CHAVE ALEATORIA",
                "metodo_autenticacao": "senha",
                "ds_estado_civil": "VIUVA",
                "qt_dependentes": 0,
            },
            {
                "cd_pix": "E00000208202603261400BATCH00003",
                "dt_pix": "2026-03-26 10:30:00",
                "cd_cpf_pagador": "55566677788",
                "vl_pix": 500.0,
                "nr_idade": 45,
                "qt_total_pix_trimestre": 20,
                "vl_mediana_pix_trimestre": 400.0,
            },
        ]
    }

    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/api/v1/batch", json=batch)
    elapsed = (time.perf_counter() - t0) * 1000

    if _check_response("Batch", resp, 200, ["total", "resultados", "resumo"]):
        data = resp.json()
        print(f"     Total: {data['total']}")
        resumo = data.get("resumo", {})
        print(f"     Decisões: {resumo.get('decisoes')}")
        print(f"     Score médio: {resumo.get('score_medio')}")
        print(f"     Latência total: {data['metadata']['latency_total_ms']:.0f}ms")

        for i, r in enumerate(data["resultados"]):
            print(
                f"       TX {i+1}: {r.get('decisao')} | "
                f"Score={r.get('score_final', -1):.1f}"
            )


# =========================================================
# 7. CACHE RESET
# =========================================================
def test_cache_reset():
    _print_header("7. POST /api/v1/cache/reset — Reset Cache")
    resp = requests.post(f"{BASE_URL}/api/v1/cache/reset")
    _check_response("Cache Reset", resp, 200, ["message"])
    data = resp.json()
    print(f"     Clientes removidos: {data.get('customers_removed')}")


# =========================================================
# 8. VALIDAÇÃO — Input inválido
# =========================================================
def test_validation():
    _print_header("8. Validação de Input — Erros esperados")

    # vl_pix negativo
    resp = requests.post(
        f"{BASE_URL}/api/v1/analyze",
        json={
            "cd_pix": "INVALID001",
            "dt_pix": "2026-03-26 12:00:00",
            "cd_cpf_pagador": "12345678901",
            "vl_pix": -100.0,
        },
    )
    _print_result(
        "vl_pix negativo → 422",
        resp.status_code == 422,
        f"HTTP {resp.status_code}",
    )

    # Sem campos obrigatórios
    resp = requests.post(
        f"{BASE_URL}/api/v1/analyze",
        json={"vl_pix": 100.0},
    )
    _print_result(
        "Sem cd_pix/dt_pix/cd_cpf → 422",
        resp.status_code == 422,
        f"HTTP {resp.status_code}",
    )

    # Batch vazio
    resp = requests.post(
        f"{BASE_URL}/api/v1/batch",
        json={"transactions": []},
    )
    _print_result(
        "Batch vazio → 422",
        resp.status_code == 422,
        f"HTTP {resp.status_code}",
    )


# =========================================================
# 9. MÉTRICAS FINAIS
# =========================================================
def test_metrics_final():
    _print_header("9. Métricas Finais (após todos os testes)")
    resp = requests.get(f"{BASE_URL}/api/v1/metrics")
    if resp.status_code == 200:
        data = resp.json()
        print(f"     Total requests:      {data.get('total_requests')}")
        print(f"     Total transactions:  {data.get('total_transactions')}")
        print(f"     Total erros:         {data.get('total_errors')}")
        print(f"     Decisões:            {data.get('decisions')}")
        print(f"     Latência média:      {data.get('latency_avg_ms')}ms")
        print(f"     Latência máxima:     {data.get('latency_max_ms')}ms")
        _print_result("Métricas finais", True)
    else:
        _print_result("Métricas finais", False, f"HTTP {resp.status_code}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 70)
    print("  TESTE COMPLETO DA API ANTIFRAUDE PIX v1.1")
    print(f"  Base URL: {BASE_URL}")
    print("=" * 70)

    # Verificar se a API está rodando
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
    except requests.ConnectionError:
        print(f"\n  ❌ API não está rodando em {BASE_URL}")
        print(f"     Inicie com: uvicorn api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    test_root()
    test_health()
    test_status()
    test_metrics()
    test_analyze()
    test_batch()
    test_cache_reset()
    test_validation()
    test_metrics_final()

    # Resumo
    total = PASS + FAIL
    print(f"\n{'='*70}")
    print(f"  RESULTADO FINAL: {PASS}/{total} testes passaram")
    if FAIL == 0:
        print(f"  ✅ TODOS OS TESTES PASSARAM!")
    else:
        print(f"  ⚠️ {FAIL} teste(s) falharam")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
