# teste_shap_rapido.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from core.pipeline_orquestrador import PipelineOrquestrador

pipeline = PipelineOrquestrador()

tx_suspeita = {
    "cd_pix": "E00000208202603190315009876543210",
    "dt_pix": "2026-03-19 03:15:00",
    "cd_cpf_pagador": "99887766554",
    "cd_cpf_cnpj_recebedor": "11223344556",
    "ds_chave_pix": "abc123-def456-ghi789",
    "ds_tipo_chave": "CHAVE ALEATORIA",
    "vl_pix": 4999.00,
    "qt_total_pix_trimestre": 1,
    "vl_mediana_pix_trimestre": 0,
    "vl_desvio_padrao_pix_trimestre": 0,
    "qt_intervalo_transacao_minuto": 0,
    "qt_intervalo_mediana_trimestre": 0,
    "qt_intervalo_desvio_padrao_trimestre": 0,
    "qt_pix_dia_maximo_trimestre": 1,
    "device_name": None,
    "app_version": "7.10.0",
    "latencia_rede_ms": None,
    "vl_latencia_rede_media_trimestre": None,
    "tempo_interacao_ms": None,
    "vl_tempo_interacao_medio_trimestre": None,
    "tempo_processamento_host_ms": None,
    "metodo_autenticacao": "senha",
    "topaz_risk_score": 4.0,
    "topaz_transacao_rejeitada": 0,
    "nr_idade": 78,
    "qt_tempo_relacionamento_mes": 2,
    "vl_renda_cliente": 3200.00,
    "ds_sexo": "F",
    "ds_estado_civil": "VIUVA",
    "ds_segmento": "VAREJO",
    "qt_dependentes": 0,
}

resultado = pipeline.analisar(tx_suspeita)

print(f"\nDecisão: {resultado['decisao']}")
print(f"Score: {resultado['score_final']}")

if "explicabilidade" in resultado:
    print(f"\n✅ SHAP ATIVO — {len(resultado['explicabilidade']['top_features'])} top features:")
    for f in resultado["explicabilidade"]["top_features"]:
        direcao = "▲" if f["direction"] == "aumenta_risco" else "▼"
        print(f"   {direcao} {f['label']:40} = {str(f['feature_value']):>10} | SHAP: {f['shap_value']:+.4f} ({f['impact_pct']:.1f}%)")
    
    print(f"\n📄 JSON completo da explicabilidade:")
    print(json.dumps(resultado["explicabilidade"], indent=2, ensure_ascii=False))
else:
    print("\n⚠️  Bloco 'explicabilidade' AUSENTE no resultado!")
    print(f"    Decisão foi: {resultado['decisao']}")
    print(f"    SHAP habilitado: {resultado['metadata'].get('shap_enabled')}")
