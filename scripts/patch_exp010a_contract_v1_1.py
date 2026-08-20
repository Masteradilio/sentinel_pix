from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent


ROOT = find_project_root(Path(__file__).resolve().parent)

EXP010A_DIR = ROOT / "resultados" / "experimentos" / "EXP-010A"
OUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-010A-R1"

DOCS_DIR = ROOT / "docs"
DOC_DATA_CONTRACT = DOCS_DIR / "DATA_INTAKE_CONTRACT.md"
DOC_REVALUATION_SPEC = DOCS_DIR / "REVALUATION_HARNESS_SPEC.md"

INPUT_SUMMARY_PATH = EXP010A_DIR / "00_input_summary.json"
OLD_TRANSACTION_CONTRACT_PATH = EXP010A_DIR / "02_transaction_schema_contract.json"
OLD_LABEL_CONTRACT_PATH = EXP010A_DIR / "03_label_schema_contract.json"
MANIFEST_PATH = ROOT / "backend" / "artefatos" / "MANIFEST_MODEL.json"


TRANSACTION_REQUIRED_BASE = [
    "transaction_id",
    "customer_id",
    "vl_pix",
    "qt_tempo_relacionamento_mes",
    "first_receiver_flag",
    "pix_key_random_flag",
]

TRANSACTION_REQUIRED_ONE_OF = [
    ["event_datetime", "dt_transacao"],
]

SUPERVISED_LABEL_REQUIRED = [
    "transaction_id",
    "is_fraud",
]

LABEL_RECOMMENDED = [
    "fraud_type",
    "label_source",
    "label_created_at",
    "label_confidence",
    "chargeback_flag",
    "confirmed_by_human",
    "contestacao_id",
    "motivo_fraude",
    "canal_confirmacao",
]

TRANSACTION_RECOMMENDED = [
    "event_datetime",
    "dt_transacao",
    "idade_cliente",
    "nr_idade",
    "tipo_pessoa",
    "canal",
    "chave_pix_tipo",
    "uf_origem",
    "uf_destino",
    "device_id",
    "ip",
    "merchant_category",
    "vl_renda_cliente",
    "topaz_risk_score",
    "qt_total_pix_trimestre",
    "qt_envio_recebedor_trimestre",
    "qt_aparelhos_distintos_trimestre",
]

BIGDATA_EXTRACTION_MINIMUM = [
    "transaction_id",
    "customer_id",
    "event_datetime ou dt_transacao",
    "vl_pix",
    "qt_tempo_relacionamento_mes",
    "first_receiver_flag",
    "pix_key_random_flag",
]

BIGDATA_LABEL_SOURCES_TO_INVESTIGATE = [
    "tabela de contestação/chargeback PIX",
    "tabela de fraude confirmada por canal operacional",
    "tabela de casos analisados por prevenção a fraudes",
    "tabela de alertas/manuais com decisão final",
    "tabela de devolução/medida especial de devolução, se disponível",
    "base de ocorrências ou protocolo interno, se ligada ao transaction_id/E2E ID",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def backup_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None

    backup_path = path.with_suffix(path.suffix + f".bak_exp010a_r1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, backup_path)
    return backup_path


def load_manifest() -> dict[str, Any]:
    return read_json(MANIFEST_PATH, {}) or {}


def load_old_contract() -> dict[str, Any]:
    return read_json(OLD_TRANSACTION_CONTRACT_PATH, {}) or {}


def load_old_label_contract() -> dict[str, Any]:
    return read_json(OLD_LABEL_CONTRACT_PATH, {}) or {}


def build_transaction_contract_v1_1(old_contract: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    old_columns = old_contract.get("columns", []) or []

    column_map = {row.get("column"): row for row in old_columns if row.get("column")}

    for col in TRANSACTION_REQUIRED_BASE:
        column_map.setdefault(
            col,
            {
                "column": col,
                "dtype_family": "unknown",
                "source": "contract_v1_1_required",
            },
        )

    for one_of_group in TRANSACTION_REQUIRED_ONE_OF:
        for col in one_of_group:
            column_map.setdefault(
                col,
                {
                    "column": col,
                    "dtype_family": "datetime_or_string",
                    "source": "contract_v1_1_required_one_of",
                },
            )

    for col in TRANSACTION_RECOMMENDED:
        column_map.setdefault(
            col,
            {
                "column": col,
                "dtype_family": "unknown",
                "source": "contract_v1_1_recommended",
            },
        )

    columns = []

    for col, row in sorted(column_map.items()):
        role = "reference_feature"

        if col in TRANSACTION_REQUIRED_BASE:
            role = "required_scoring"
        elif any(col in group for group in TRANSACTION_REQUIRED_ONE_OF):
            role = "required_one_of_temporal"
        elif col in TRANSACTION_RECOMMENDED:
            role = "recommended"

        new_row = dict(row)
        new_row["contract_role_v1_1"] = role
        new_row["required_for_scoring"] = col in TRANSACTION_REQUIRED_BASE
        new_row["required_one_of_group"] = next(
            ("event_datetime_or_dt_transacao" for group in TRANSACTION_REQUIRED_ONE_OF if col in group),
            "",
        )
        new_row["required_for_supervised_evaluation"] = col in TRANSACTION_REQUIRED_BASE or any(
            col in group for group in TRANSACTION_REQUIRED_ONE_OF
        )
        new_row["label_column"] = False

        # is_fraud deixa de ser obrigatório no arquivo de transações.
        if col == "is_fraud":
            new_row["contract_role_v1_1"] = "deprecated_in_transaction_file_use_label_contract"
            new_row["required_for_scoring"] = False
            new_row["required_for_supervised_evaluation"] = False
            new_row["label_column"] = True

        columns.append(new_row)

    return {
        "schema_version": "1.1",
        "generated_at": now_iso(),
        "previous_schema_version": old_contract.get("schema_version", "1.0"),
        "model_version": manifest.get("model_version", "post_fase2_c1"),
        "status": manifest.get("status", "ACTIVE_BASELINE"),
        "important_change": "is_fraud nao e mais obrigatorio no arquivo de transacoes; labels devem ficar em contrato separado no modo supervisionado.",
        "modes": {
            "scoring_inference": {
                "description": "Modo para pontuar novas transacoes sem labels disponiveis.",
                "required_transaction_columns": TRANSACTION_REQUIRED_BASE,
                "required_one_of": TRANSACTION_REQUIRED_ONE_OF,
                "is_fraud_required": False,
                "labels_required": False,
            },
            "supervised_evaluation": {
                "description": "Modo para avaliar performance quando ha labels confirmados.",
                "required_transaction_columns": TRANSACTION_REQUIRED_BASE,
                "required_one_of": TRANSACTION_REQUIRED_ONE_OF,
                "labels_required": True,
                "required_label_columns": SUPERVISED_LABEL_REQUIRED,
            },
        },
        "c1_required_columns": [
            "vl_pix",
            "qt_tempo_relacionamento_mes",
            "first_receiver_flag",
            "pix_key_random_flag",
        ],
        "recommended_transaction_columns": TRANSACTION_RECOMMENDED,
        "columns": columns,
    }


def build_label_contract_v1_1(old_label_contract: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "generated_at": now_iso(),
        "previous_schema_version": old_label_contract.get("schema_version", "1.0"),
        "model_version": manifest.get("model_version", "post_fase2_c1"),
        "description": "Contrato de labels para avaliacao supervisionada e reavaliacao com novos casos confirmados.",
        "required_columns": SUPERVISED_LABEL_REQUIRED,
        "recommended_columns": LABEL_RECOMMENDED,
        "allowed_is_fraud_values": [0, 1],
        "join_key": "transaction_id",
        "label_modes": {
            "confirmed_fraud_only": {
                "description": "Arquivo contem apenas transacoes fraudulentas confirmadas; ausencia de label nao implica normalidade.",
                "requires_negative_sampling": True,
            },
            "full_supervised_window": {
                "description": "Arquivo contem labels 0/1 para uma janela fechada de transacoes.",
                "requires_negative_sampling": False,
            },
        },
        "rules": [
            "transaction_id deve existir no arquivo de transacoes ou em tabela de mapeamento para E2E ID.",
            "is_fraud deve ser 0 ou 1.",
            "Fraude confirmada deve ser marcada com is_fraud=1.",
            "Normalidade deve ser marcada com is_fraud=0 apenas quando a janela de observacao ja estiver madura.",
            "Nao tratar ausencia de fraude confirmada como normalidade automaticamente sem janela de maturacao.",
            "label_created_at deve ser posterior ou igual ao evento da transacao.",
            "confirmed_by_human deve ser usado quando houver decisao manual.",
            "label_confidence deve diferenciar confirmado, provavel e suspeito quando possivel.",
            "transaction_id duplicado no arquivo de labels deve ser tratado como excecao e auditado.",
        ],
    }


def build_data_contract_md(contract: dict[str, Any], label_contract: dict[str, Any]) -> str:
    manifest = load_manifest()

    lines = [
        "# DATA_INTAKE_CONTRACT — Antifraude PIX",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Versão",
        "",
        "- `schema_version`: `1.1`",
        f"- `model_version`: `{manifest.get('model_version', 'post_fase2_c1')}`",
        f"- `status`: `{manifest.get('status', 'ACTIVE_BASELINE')}`",
        "",
        "## Objetivo",
        "",
        "Definir o contrato de entrada para novos dados de transações PIX e novos labels de fraude.",
        "",
        "Esta versão separa explicitamente dois modos de uso:",
        "",
        "1. **scoring/inferência**: novas transações podem não ter `is_fraud`.",
        "2. **avaliação supervisionada**: labels vêm em arquivo/tabela separada ligada por `transaction_id`.",
        "",
        "## Modo 1 — Scoring / inferência",
        "",
        "Usado quando o objetivo é aplicar o baseline `post_fase2_c1` a uma nova janela ainda sem labels completos.",
        "",
        "### Colunas obrigatórias de transações",
        "",
    ]

    for col in TRANSACTION_REQUIRED_BASE:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "### Coluna temporal obrigatória",
            "",
            "O arquivo deve conter pelo menos uma das colunas abaixo:",
            "",
        ]
    )

    for group in TRANSACTION_REQUIRED_ONE_OF:
        lines.append("- " + " **ou** ".join(f"`{col}`" for col in group))

    lines.extend(
        [
            "",
            "### Observação sobre `is_fraud`",
            "",
            "`is_fraud` **não é obrigatório** no arquivo de transações em modo scoring/inferência.",
            "",
            "## Modo 2 — Avaliação supervisionada",
            "",
            "Usado quando há labels confirmados e o objetivo é calcular métricas como TP, FP, FN, Precision, Recall e F1.",
            "",
            "Neste modo, as transações seguem o mesmo contrato do modo scoring, mas os labels devem vir em arquivo ou tabela separada.",
            "",
            "### Colunas obrigatórias de labels",
            "",
        ]
    )

    for col in label_contract["required_columns"]:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "### Colunas recomendadas de labels",
            "",
        ]
    )

    for col in label_contract["recommended_columns"]:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Colunas necessárias para a C1",
            "",
        ]
    )

    for col in contract["c1_required_columns"]:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Colunas recomendadas para enriquecer novas extrações",
            "",
        ]
    )

    for col in TRANSACTION_RECOMMENDED:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Regras mínimas",
            "",
            "- `transaction_id` deve ser único no arquivo de transações.",
            "- `transaction_id` deve permitir join com labels quando labels existirem.",
            "- O arquivo deve ter uma data transacional explícita: `event_datetime` ou `dt_transacao`.",
            "- Colunas obrigatórias do modo scoring não devem vir nulas.",
            "- `is_fraud` deve usar valores `0` ou `1` quando existir em labels.",
            "- Não assumir que ausência de fraude confirmada significa normalidade sem janela de maturação.",
            "- Mudanças grandes de distribuição em `vl_pix`, relacionamento ou flags devem ser tratadas como drift.",
            "- Novos dados não devem sobrescrever o baseline oficial; devem ser avaliados em diretório próprio.",
            "",
            "## Arquivos técnicos",
            "",
            "- `resultados/experimentos/EXP-010A-R1/01_transaction_schema_contract_v1_1.json`",
            "- `resultados/experimentos/EXP-010A-R1/02_label_schema_contract_v1_1.json`",
            "- `resultados/experimentos/EXP-010A-R1/03_DATA_INTAKE_CONTRACT_v1_1.md`",
            "- `resultados/experimentos/EXP-010A-R1/04_REVALUATION_HARNESS_SPEC_v1_1.md`",
            "",
        ]
    )

    return "\n".join(lines)


def build_harness_spec_md() -> str:
    lines = [
        "# REVALUATION_HARNESS_SPEC — Antifraude PIX",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Versão",
        "",
        "- `schema_version`: `1.1`",
        "- Baseline alvo: `post_fase2_c1`",
        "",
        "## Objetivo",
        "",
        "Definir o harness de reavaliação do baseline em novas janelas de dados extraídas do Big Data.",
        "",
        "## Modos de execução",
        "",
        "### 1. New Data Dry Run sem labels",
        "",
        "Usado quando há transações novas, mas ainda não há confirmação madura de fraude.",
        "",
        "Fluxo:",
        "",
        "1. Validar schema pelo `DATA_INTAKE_CONTRACT`.",
        "2. Aplicar baseline atual sem alterar modelo.",
        "3. Gerar decision logs no padrão EXP-009A.",
        "4. Rodar drift monitor do EXP-009B.",
        "5. Gerar fila de revisão humana no padrão EXP-009C.",
        "6. Atualizar painel operacional no padrão EXP-009D.",
        "7. Não calcular métricas supervisionadas.",
        "",
        "### 2. Reavaliação supervisionada com labels",
        "",
        "Usado quando há labels confirmados e uma janela de observação madura.",
        "",
        "Fluxo:",
        "",
        "1. Validar transações.",
        "2. Validar labels.",
        "3. Fazer join por `transaction_id`.",
        "4. Aplicar baseline atual.",
        "5. Gerar métricas TP, FP, FN, TN, Precision, Recall, F1 e FPR.",
        "6. Gerar decision logs, drift, fila e dashboard.",
        "7. Rodar EXP-009E antes de qualquer promoção.",
        "",
        "## Guardas contra leakage",
        "",
        "- Labels confirmados após a transação não podem virar feature.",
        "- Tabelas de contestação, MED, chargeback ou análise manual só podem ser usadas como label/target, não como feature preditiva no mesmo instante.",
        "- Features devem ser calculadas com dados disponíveis até o momento da transação.",
        "- Janelas agregadas devem respeitar corte temporal.",
        "- O conjunto de teste futuro deve ficar separado de qualquer retreinamento.",
        "",
        "## Critérios para retreinamento futuro",
        "",
        "Retreinamento só deve ser considerado se houver:",
        "",
        "- volume suficiente de novas fraudes confirmadas;",
        "- aumento relevante de FN no baseline atual;",
        "- drift material nas features críticas;",
        "- evidência de novos padrões não cobertos por V1/C1;",
        "- validação offline mostrando ganho em FN sem aumento inseguro de FP.",
        "",
        "## Critérios para nova regra futura",
        "",
        "Uma nova regra só deve avançar se:",
        "",
        "- recuperar FN novo ou residual;",
        "- adicionar 0 FP ou FP operacionalmente aceitável;",
        "- não perder TP;",
        "- ser configurável/desligável;",
        "- passar no EXP-009E.",
        "",
    ]

    return "\n".join(lines)


def build_bigdata_checklist_md() -> str:
    lines = [
        "# BIGDATA_PREP_CHECKLIST — Preparação para extração de novos dados PIX",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Objetivo",
        "",
        "Listar os pontos que precisam ser resolvidos antes de extrair novos dados de fraude diretamente do ambiente Big Data.",
        "",
        "## 1. Identificadores de join",
        "",
        "- Confirmar se `transaction_id` do projeto corresponde ao E2E ID PIX no Big Data.",
        "- Confirmar se há outra chave interna de transação que precise ser mapeada.",
        "- Confirmar se tabelas de fraude/contestação possuem o mesmo identificador.",
        "",
        "## 2. Colunas mínimas de transação",
        "",
    ]

    for item in BIGDATA_EXTRACTION_MINIMUM:
        lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## 3. Fontes candidatas de label de fraude",
            "",
        ]
    )

    for item in BIGDATA_LABEL_SOURCES_TO_INVESTIGATE:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 4. Maturação dos labels",
            "",
            "- Definir janela mínima para considerar uma transação normal.",
            "- Evitar tratar ausência de contestação imediata como `is_fraud=0`.",
            "- Separar fraude confirmada, suspeita, contestada e descartada.",
            "",
            "## 5. Janelas sugeridas",
            "",
            "- Janela de referência atual: preservar baseline `post_fase2_c1`.",
            "- Nova janela scoring: últimos N dias/semanas disponíveis.",
            "- Nova janela supervisionada: período antigo o suficiente para labels estarem maduros.",
            "",
            "## 6. Segurança contra leakage",
            "",
            "- Não usar campos de pós-evento como features.",
            "- Não usar decisão manual, contestação ou chargeback como feature no tempo da transação.",
            "- Usar essas fontes apenas como label/target.",
            "",
            "## 7. Saída esperada da extração",
            "",
            "- `new_transactions.csv` ou parquet equivalente.",
            "- `new_labels.csv` ou parquet equivalente, se labels estiverem disponíveis.",
            "- Relatório com período extraído, filtros aplicados, contagens e origem das tabelas.",
            "",
        ]
    )

    return "\n".join(lines)


def build_next_experiment_spec_md() -> str:
    return """# Próximo experimento recomendado

## EXP-010B — New Data Dry Run

## Objetivo

Executar o baseline `post_fase2_c1` em uma nova janela de dados validada pelo contrato `DATA_INTAKE_CONTRACT` v1.1.

## Pré-requisito

Ter um arquivo novo de transações extraído do Big Data com as colunas mínimas do modo scoring/inferência.

## Ações

- Validar novo dataset com o contrato v1.1.
- Rodar baseline atual sem alterar modelo.
- Gerar decision logs no padrão EXP-009A.
- Rodar drift monitor EXP-009B contra a referência.
- Gerar fila de revisão EXP-009C para a nova janela.
- Atualizar painel EXP-009D para a nova janela.
- Comparar métricas supervisionadas apenas se labels maduros existirem.

## Critério de aprovação

- Novo dataset passa no contrato de entrada.
- Baseline roda na nova janela sem erro.
- Logs estruturados são gerados com schema válido.
- Drift e métricas operacionais são calculados.
- Nenhuma mudança é promovida ainda.
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    input_summary = read_json(INPUT_SUMMARY_PATH, {})
    manifest = load_manifest()
    old_contract = load_old_contract()
    old_label_contract = load_old_label_contract()

    transaction_contract = build_transaction_contract_v1_1(old_contract, manifest)
    label_contract = build_label_contract_v1_1(old_label_contract, manifest)

    data_contract_md = build_data_contract_md(transaction_contract, label_contract)
    harness_spec_md = build_harness_spec_md()
    checklist_md = build_bigdata_checklist_md()
    next_spec_md = build_next_experiment_spec_md()

    write_json(OUT_DIR / "01_transaction_schema_contract_v1_1.json", transaction_contract)
    write_json(OUT_DIR / "02_label_schema_contract_v1_1.json", label_contract)

    (OUT_DIR / "03_DATA_INTAKE_CONTRACT_v1_1.md").write_text(data_contract_md, encoding="utf-8")
    (OUT_DIR / "04_REVALUATION_HARNESS_SPEC_v1_1.md").write_text(harness_spec_md, encoding="utf-8")
    (OUT_DIR / "05_BIGDATA_PREP_CHECKLIST.md").write_text(checklist_md, encoding="utf-8")
    (OUT_DIR / "06_next_experiment_spec.md").write_text(next_spec_md, encoding="utf-8")

    backup_data = backup_if_exists(DOC_DATA_CONTRACT)
    backup_harness = backup_if_exists(DOC_REVALUATION_SPEC)

    DOC_DATA_CONTRACT.write_text(data_contract_md, encoding="utf-8")
    DOC_REVALUATION_SPEC.write_text(harness_spec_md, encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "status": "APPLIED",
        "experiment": "EXP-010A-R1",
        "purpose": "Ajustar contrato v1.1 separando scoring/inferencia de avaliacao supervisionada.",
        "previous_exp010a_summary": input_summary,
        "model_version": manifest.get("model_version", "post_fase2_c1"),
        "manifest_status": manifest.get("status", "ACTIVE_BASELINE"),
        "changes": [
            "is_fraud deixou de ser obrigatorio no arquivo de transacoes em modo scoring/inferencia.",
            "labels supervisionados agora ficam em contrato separado.",
            "event_datetime ou dt_transacao virou requisito operacional obrigatorio.",
            "harness passou a separar dry run sem labels de reavaliacao supervisionada com labels.",
            "foi criado checklist de preparacao para extracao no Big Data.",
        ],
        "canonical_docs_updated": {
            "DATA_INTAKE_CONTRACT.md": str(DOC_DATA_CONTRACT),
            "REVALUATION_HARNESS_SPEC.md": str(DOC_REVALUATION_SPEC),
        },
        "backups": {
            "DATA_INTAKE_CONTRACT.md": str(backup_data) if backup_data else None,
            "REVALUATION_HARNESS_SPEC.md": str(backup_harness) if backup_harness else None,
        },
        "artifacts": [
            "01_transaction_schema_contract_v1_1.json",
            "02_label_schema_contract_v1_1.json",
            "03_DATA_INTAKE_CONTRACT_v1_1.md",
            "04_REVALUATION_HARNESS_SPEC_v1_1.md",
            "05_BIGDATA_PREP_CHECKLIST.md",
            "06_next_experiment_spec.md",
        ],
    }

    write_json(OUT_DIR / "00_adjustment_summary.json", summary)

    print("[OK] EXP-010A-R1 aplicado.")
    print(f"[OK] Artefatos em: {OUT_DIR}")
    print(f"[OK] Documento atualizado: {DOC_DATA_CONTRACT}")
    print(f"[OK] Documento atualizado: {DOC_REVALUATION_SPEC}")
    if backup_data:
        print(f"[OK] Backup DATA_INTAKE_CONTRACT: {backup_data}")
    if backup_harness:
        print(f"[OK] Backup REVALUATION_HARNESS_SPEC: {backup_harness}")


if __name__ == "__main__":
    main()