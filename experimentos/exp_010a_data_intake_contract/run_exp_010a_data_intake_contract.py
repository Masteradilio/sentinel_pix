"""
EXP-010A — Data Intake Contract e Harness de Reavaliacao com Novos Dados

Objetivo:
  Preparar o projeto para receber novos dados de transacoes normais e novos casos
  de fraude, sem ainda executar retreinamento nem E2E pesado.

Este experimento:
  - Nao altera modelo.
  - Nao altera scoring_config.json.
  - Nao altera DecisionEngine.
  - Nao roda E2E.
  - Gera contrato de schema para novos dados.
  - Gera contrato de labels.
  - Gera perfil da base atual de referencia.
  - Gera relatorio de compatibilidade de features.
  - Opcionalmente valida um novo arquivo de transacoes/labels se informado.

Uso basico:
  python experimentos\\exp_010a_data_intake_contract\\run_exp_010a_data_intake_contract.py

Uso futuro com novos dados:
  python experimentos\\exp_010a_data_intake_contract\\run_exp_010a_data_intake_contract.py ^
    --new-transactions caminho\\novas_transacoes.csv ^
    --new-labels caminho\\novos_labels.csv

Saidas:
  resultados/experimentos/EXP-010A/
    00_input_summary.json
    01_transaction_schema_contract.csv
    02_transaction_schema_contract.json
    03_label_schema_contract.json
    04_reference_data_profile.csv
    05_feature_compatibility_report.csv
    06_DATA_INTAKE_CONTRACT.md
    07_REVALUATION_HARNESS_SPEC.md
    08_new_data_validation.json       opcional
    09_new_data_validation_report.md  opcional
    10_next_experiment_spec.md

Tambem atualiza:
  docs/DATA_INTAKE_CONTRACT.md
  docs/REVALUATION_HARNESS_SPEC.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent.parent


ROOT = find_project_root(EXP_DIR)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "core"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

OUTPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-010A"
DOCS_DIR = ROOT / "docs"

MANIFEST_PATH = ROOT / "backend" / "artefatos" / "MANIFEST_MODEL.json"
SCORING_PATH = ROOT / "backend" / "artefatos" / "scoring_config.json"
EXP009D_FACT_PATH = ROOT / "resultados" / "experimentos" / "EXP-009D" / "07_powerbi_decision_fact.csv"

DOC_DATA_CONTRACT_PATH = DOCS_DIR / "DATA_INTAKE_CONTRACT.md"
DOC_REVALUATION_SPEC_PATH = DOCS_DIR / "REVALUATION_HARNESS_SPEC.md"


CORE_REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "is_fraud",
    "vl_pix",
    "qt_tempo_relacionamento_mes",
    "first_receiver_flag",
    "pix_key_random_flag",
]

C1_REQUIRED_COLUMNS = [
    "vl_pix",
    "qt_tempo_relacionamento_mes",
    "first_receiver_flag",
    "pix_key_random_flag",
]

OPTIONAL_BUT_RECOMMENDED_COLUMNS = [
    "dt_transacao",
    "idade_cliente",
    "tipo_pessoa",
    "canal",
    "chave_pix_tipo",
    "uf_origem",
    "uf_destino",
    "device_id",
    "ip",
    "merchant_category",
]

LABEL_REQUIRED_COLUMNS = [
    "transaction_id",
    "is_fraud",
]

LABEL_RECOMMENDED_COLUMNS = [
    "fraud_type",
    "label_source",
    "label_created_at",
    "label_confidence",
    "chargeback_flag",
    "confirmed_by_human",
]

NUMERIC_DRIFT_COLUMNS = [
    "vl_pix",
    "qt_tempo_relacionamento_mes",
    "first_receiver_flag",
    "pix_key_random_flag",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [safe_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe_json(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "records" in data:
            return pd.DataFrame(data["records"])
        raise ValueError(f"JSON nao reconhecido como lista de registros: {path}")

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    return pd.read_csv(path)


def load_reference_dataset() -> tuple[pd.DataFrame, str]:
    """
    Carrega a base de referencia do projeto.

    Prioridade:
      1. experimentos.utils_experimentos.load_dataset()
      2. maior CSV dentro de dados/
      3. fallback no fact do EXP-009D, se existir
    """
    try:
        from experimentos.utils_experimentos import load_dataset

        df = load_dataset()
        return df, "experimentos.utils_experimentos.load_dataset()"
    except Exception as exc:
        print(f"[AVISO] Falha ao usar load_dataset(): {exc}")

    dados_dir = ROOT / "dados"
    csvs = sorted(dados_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)

    if csvs:
        path = csvs[0]
        df = pd.read_csv(path)
        return df, str(path)

    if EXP009D_FACT_PATH.exists():
        df = pd.read_csv(EXP009D_FACT_PATH)
        return df, str(EXP009D_FACT_PATH)

    raise RuntimeError("Nao foi possivel carregar dataset de referencia.")


def infer_contract_role(col: str) -> str:
    if col in CORE_REQUIRED_COLUMNS:
        return "required_core"
    if col in C1_REQUIRED_COLUMNS:
        return "required_c1"
    if col in OPTIONAL_BUT_RECOMMENDED_COLUMNS:
        return "recommended"
    return "reference_feature"


def infer_dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    converted = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()

    if non_null > 0 and converted.notna().sum() / max(non_null, 1) >= 0.95:
        if (converted.dropna() % 1 == 0).all():
            return "integer_like"
        return "float_like"

    return "string"


def profile_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    dtype_family = infer_dtype_family(s)
    null_rate = float(s.isna().mean())
    n_unique = int(s.nunique(dropna=True))

    result: dict[str, Any] = {
        "column": col,
        "contract_role": infer_contract_role(col),
        "dtype_observed": str(s.dtype),
        "dtype_family": dtype_family,
        "required": infer_contract_role(col) in {"required_core", "required_c1"},
        "recommended": infer_contract_role(col) == "recommended",
        "n_rows_reference": int(len(df)),
        "null_rate_reference": null_rate,
        "n_unique_reference": n_unique,
        "example_values": json.dumps(
            [str(x) for x in s.dropna().astype(str).head(5).tolist()],
            ensure_ascii=False,
        ),
    }

    if dtype_family in {"integer", "float", "integer_like", "float_like"}:
        num = pd.to_numeric(s, errors="coerce")
        result.update(
            {
                "min_reference": float(num.min()) if num.notna().any() else None,
                "p01_reference": float(num.quantile(0.01)) if num.notna().any() else None,
                "p50_reference": float(num.quantile(0.50)) if num.notna().any() else None,
                "p99_reference": float(num.quantile(0.99)) if num.notna().any() else None,
                "max_reference": float(num.max()) if num.notna().any() else None,
                "mean_reference": float(num.mean()) if num.notna().any() else None,
            }
        )
    else:
        result.update(
            {
                "min_reference": None,
                "p01_reference": None,
                "p50_reference": None,
                "p99_reference": None,
                "max_reference": None,
                "mean_reference": None,
            }
        )

    return result


def build_transaction_contract(reference_df: pd.DataFrame) -> pd.DataFrame:
    rows = [profile_column(reference_df, col) for col in reference_df.columns]

    # Garante que colunas obrigatorias/recomendadas aparecam mesmo se ausentes na base atual.
    existing = set(reference_df.columns)

    for col in CORE_REQUIRED_COLUMNS + OPTIONAL_BUT_RECOMMENDED_COLUMNS:
        if col in existing:
            continue

        rows.append(
            {
                "column": col,
                "contract_role": infer_contract_role(col),
                "dtype_observed": "MISSING_IN_REFERENCE",
                "dtype_family": "unknown",
                "required": col in CORE_REQUIRED_COLUMNS,
                "recommended": col in OPTIONAL_BUT_RECOMMENDED_COLUMNS,
                "n_rows_reference": int(len(reference_df)),
                "null_rate_reference": None,
                "n_unique_reference": None,
                "example_values": "[]",
                "min_reference": None,
                "p01_reference": None,
                "p50_reference": None,
                "p99_reference": None,
                "max_reference": None,
                "mean_reference": None,
            }
        )

    contract = pd.DataFrame(rows)

    role_order = {
        "required_core": 0,
        "required_c1": 1,
        "recommended": 2,
        "reference_feature": 3,
    }

    contract["role_order"] = contract["contract_role"].map(role_order).fillna(99)
    contract = contract.sort_values(["role_order", "column"]).drop(columns=["role_order"])

    return contract


def build_label_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "required_columns": LABEL_REQUIRED_COLUMNS,
        "recommended_columns": LABEL_RECOMMENDED_COLUMNS,
        "allowed_is_fraud_values": [0, 1],
        "join_key": "transaction_id",
        "rules": [
            "transaction_id deve existir no arquivo de transacoes.",
            "is_fraud deve ser 0 ou 1.",
            "Novos casos confirmados de fraude devem ser marcados com is_fraud=1.",
            "Labels suspeitos ou nao confirmados devem usar label_confidence e confirmed_by_human quando disponivel.",
            "O arquivo de labels nao deve duplicar transaction_id sem justificativa.",
        ],
    }


def contract_to_json(contract_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "model_version": read_json(MANIFEST_PATH, {}).get("model_version", "post_fase2_c1"),
        "required_core_columns": CORE_REQUIRED_COLUMNS,
        "required_c1_columns": C1_REQUIRED_COLUMNS,
        "optional_but_recommended_columns": OPTIONAL_BUT_RECOMMENDED_COLUMNS,
        "columns": contract_df.to_dict(orient="records"),
    }


def psi_from_series(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if len(ref) == 0 or len(cur) == 0:
        return None

    if ref.nunique(dropna=True) <= 1:
        ref_value = float(ref.iloc[0])
        return 0.0 if float((cur == ref_value).mean()) == 1.0 else 1.0

    edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))

    if len(edges) < 3:
        min_v = min(float(ref.min()), float(cur.min()))
        max_v = max(float(ref.max()), float(cur.max()))
        if min_v == max_v:
            max_v = min_v + 1e-9
        edges = np.array([min_v, max_v])

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_bins = pd.cut(ref, bins=edges, include_lowest=True, duplicates="drop")
    cur_bins = pd.cut(cur, bins=edges, include_lowest=True, duplicates="drop")

    ref_counts = ref_bins.value_counts(sort=False)
    cur_counts = cur_bins.value_counts(sort=False)

    eps = 1e-6
    psi = 0.0

    for b in ref_counts.index:
        ref_pct = float(ref_counts.get(b, 0) / max(len(ref), 1))
        cur_pct = float(cur_counts.get(b, 0) / max(len(cur), 1))

        ref_adj = max(ref_pct, eps)
        cur_adj = max(cur_pct, eps)

        psi += (cur_adj - ref_adj) * math.log(cur_adj / ref_adj)

    return float(psi)


def validate_new_transactions(
    reference_df: pd.DataFrame,
    new_df: pd.DataFrame,
    contract_df: pd.DataFrame,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    required_cols = contract_df[contract_df["required"].astype(bool)]["column"].tolist()
    missing_required = [c for c in required_cols if c not in new_df.columns]

    checks.append(
        {
            "check_id": "missing_required_columns",
            "status": "PASS" if not missing_required else "FAIL",
            "expected": [],
            "actual": missing_required,
        }
    )

    if "transaction_id" in new_df.columns:
        duplicate_tx = int(new_df["transaction_id"].duplicated().sum())
    else:
        duplicate_tx = None

    checks.append(
        {
            "check_id": "duplicate_transaction_id",
            "status": "PASS" if duplicate_tx == 0 else "FAIL",
            "expected": 0,
            "actual": duplicate_tx,
        }
    )

    null_checks = []

    for col in required_cols:
        if col not in new_df.columns:
            continue

        null_rate = float(new_df[col].isna().mean())
        status = "PASS" if null_rate == 0.0 else "WARN"

        null_checks.append(
            {
                "column": col,
                "null_rate": null_rate,
                "status": status,
            }
        )

    drift_rows = []

    for col in NUMERIC_DRIFT_COLUMNS:
        if col not in reference_df.columns or col not in new_df.columns:
            continue

        psi = psi_from_series(reference_df[col], new_df[col])

        if psi is None:
            status = "MISSING"
        elif psi >= 0.25:
            status = "ALERT"
        elif psi >= 0.10:
            status = "WARN"
        else:
            status = "OK"

        drift_rows.append(
            {
                "column": col,
                "psi": psi,
                "status": status,
                "reference_mean": float(pd.to_numeric(reference_df[col], errors="coerce").mean()),
                "new_mean": float(pd.to_numeric(new_df[col], errors="coerce").mean()),
            }
        )

    has_fail = any(c["status"] == "FAIL" for c in checks)
    has_alert = any(r["status"] == "ALERT" for r in drift_rows)
    has_warn = any(c["status"] == "WARN" for c in null_checks) or any(r["status"] == "WARN" for r in drift_rows)

    if has_fail:
        status = "FAIL"
    elif has_alert:
        status = "ALERT"
    elif has_warn:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "n_rows_new": int(len(new_df)),
        "n_cols_new": int(len(new_df.columns)),
        "checks": checks,
        "null_checks": null_checks,
        "numeric_drift": drift_rows,
    }


def validate_new_labels(new_tx: pd.DataFrame | None, labels_df: pd.DataFrame) -> dict[str, Any]:
    missing_required = [c for c in LABEL_REQUIRED_COLUMNS if c not in labels_df.columns]

    checks = [
        {
            "check_id": "missing_required_label_columns",
            "status": "PASS" if not missing_required else "FAIL",
            "expected": [],
            "actual": missing_required,
        }
    ]

    if "transaction_id" in labels_df.columns:
        dup = int(labels_df["transaction_id"].duplicated().sum())
    else:
        dup = None

    checks.append(
        {
            "check_id": "duplicate_label_transaction_id",
            "status": "PASS" if dup == 0 else "WARN",
            "expected": 0,
            "actual": dup,
        }
    )

    invalid_labels: list[Any] = []

    if "is_fraud" in labels_df.columns:
        values = set(pd.to_numeric(labels_df["is_fraud"], errors="coerce").dropna().astype(int).unique().tolist())
        invalid_labels = sorted(values - {0, 1})

    checks.append(
        {
            "check_id": "invalid_is_fraud_values",
            "status": "PASS" if not invalid_labels else "FAIL",
            "expected": [0, 1],
            "actual": invalid_labels,
        }
    )

    join_missing = None

    if new_tx is not None and "transaction_id" in new_tx.columns and "transaction_id" in labels_df.columns:
        tx_ids = set(new_tx["transaction_id"].astype(str))
        label_ids = set(labels_df["transaction_id"].astype(str))
        join_missing = len(label_ids - tx_ids)

        checks.append(
            {
                "check_id": "labels_without_transaction",
                "status": "PASS" if join_missing == 0 else "WARN",
                "expected": 0,
                "actual": join_missing,
            }
        )

    has_fail = any(c["status"] == "FAIL" for c in checks)
    has_warn = any(c["status"] == "WARN" for c in checks)

    status = "FAIL" if has_fail else "WARN" if has_warn else "PASS"

    return {
        "status": status,
        "n_rows_labels": int(len(labels_df)),
        "checks": checks,
    }


def build_feature_compatibility_report(contract_df: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    active_config = manifest.get("active_config", {}) or {}

    rows = []

    contract_cols = set(contract_df["column"].tolist())

    for col in sorted(contract_cols):
        role = contract_df.loc[contract_df["column"] == col, "contract_role"].iloc[0]
        required = bool(contract_df.loc[contract_df["column"] == col, "required"].iloc[0])

        rows.append(
            {
                "feature": col,
                "contract_role": role,
                "required": required,
                "present_in_reference": bool(
                    contract_df.loc[contract_df["column"] == col, "dtype_observed"].iloc[0] != "MISSING_IN_REFERENCE"
                ),
                "used_by_c1": col in C1_REQUIRED_COLUMNS,
                "used_by_manifest_config": col in active_config,
                "compatibility_status": "REQUIRED" if required else "REFERENCE",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["required", "used_by_c1", "feature"],
        ascending=[False, False, True],
    )


def write_data_contract_doc(contract_df: pd.DataFrame, label_contract: dict[str, Any]) -> str:
    manifest = read_json(MANIFEST_PATH, {})
    lines = [
        "# DATA_INTAKE_CONTRACT — Antifraude PIX",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Objetivo",
        "",
        "Definir o contrato mínimo para entrada de novos dados de transações e novos labels de fraude no projeto antifraude PIX.",
        "",
        "## Baseline alvo",
        "",
        f"- `model_version`: `{manifest.get('model_version', 'post_fase2_c1')}`",
        f"- `status`: `{manifest.get('status', 'ACTIVE_BASELINE')}`",
        "",
        "## Colunas obrigatórias de transações",
        "",
    ]

    for col in CORE_REQUIRED_COLUMNS:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Colunas necessárias para validação da C1",
            "",
        ]
    )

    for col in C1_REQUIRED_COLUMNS:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Colunas recomendadas",
            "",
        ]
    )

    for col in OPTIONAL_BUT_RECOMMENDED_COLUMNS:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Contrato de labels",
            "",
            "Colunas obrigatórias:",
            "",
        ]
    )

    for col in label_contract["required_columns"]:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "Colunas recomendadas:",
            "",
        ]
    )

    for col in label_contract["recommended_columns"]:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Regras mínimas",
            "",
            "- `transaction_id` deve ser único no arquivo de transações.",
            "- `transaction_id` deve permitir join com labels quando labels existirem.",
            "- `is_fraud` deve usar valores `0` ou `1`.",
            "- Colunas obrigatórias não devem vir nulas.",
            "- Mudanças grandes de distribuição em `vl_pix`, relacionamento ou flags devem ser tratadas como drift.",
            "- Novos dados não devem sobrescrever o baseline oficial; devem ser avaliados em diretório próprio.",
            "",
            "## Arquivos técnicos gerados",
            "",
            "- `01_transaction_schema_contract.csv`",
            "- `02_transaction_schema_contract.json`",
            "- `03_label_schema_contract.json`",
            "- `05_feature_compatibility_report.csv`",
            "",
        ]
    )

    text = "\n".join(lines)

    (OUTPUT_DIR / "06_DATA_INTAKE_CONTRACT.md").write_text(text, encoding="utf-8")
    DOC_DATA_CONTRACT_PATH.write_text(text, encoding="utf-8")

    return text


def write_revaluation_harness_spec() -> str:
    lines = [
        "# REVALUATION_HARNESS_SPEC — Antifraude PIX",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Objetivo",
        "",
        "Definir o harness de reavaliação do baseline `post_fase2_c1` em novas janelas de dados.",
        "",
        "## Fluxo recomendado",
        "",
        "1. Receber novo arquivo de transações.",
        "2. Receber labels novos, se disponíveis.",
        "3. Validar schema com o contrato do EXP-010A.",
        "4. Gerar relatório de compatibilidade de features.",
        "5. Rodar baseline atual em modo controlado.",
        "6. Gerar decision logs estruturados no formato do EXP-009A.",
        "7. Rodar drift monitor do EXP-009B contra a janela de referência.",
        "8. Gerar fila de revisão humana no padrão EXP-009C.",
        "9. Atualizar painel operacional no padrão EXP-009D.",
        "10. Rodar EXP-009E antes de qualquer promoção de mudança.",
        "",
        "## Critérios para retreinamento futuro",
        "",
        "Retreinamento só deve ser considerado se houver:",
        "",
        "- volume suficiente de novas fraudes confirmadas;",
        "- aumento relevante de FN no baseline atual;",
        "- drift material nas features críticas;",
        "- evidência de novos padrões de fraude não cobertos por V1/C1;",
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

    text = "\n".join(lines)

    (OUTPUT_DIR / "07_REVALUATION_HARNESS_SPEC.md").write_text(text, encoding="utf-8")
    DOC_REVALUATION_SPEC_PATH.write_text(text, encoding="utf-8")

    return text


def write_validation_report(validation: dict[str, Any]) -> None:
    lines = [
        "# EXP-010A — Validação opcional de novos dados",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        f"**Status geral:** `{validation.get('status')}`",
        "",
        "## Transações",
        "",
    ]

    tx = validation.get("transactions")

    if tx:
        lines.extend(
            [
                f"- Status: `{tx.get('status')}`",
                f"- Linhas: `{tx.get('n_rows_new')}`",
                f"- Colunas: `{tx.get('n_cols_new')}`",
                "",
                "### Checks",
                "",
            ]
        )

        for check in tx.get("checks", []):
            lines.append(f"- `{check['check_id']}`: `{check['status']}` esperado=`{check['expected']}` atual=`{check['actual']}`")

        lines.extend(["", "### Drift numérico", ""])

        for row in tx.get("numeric_drift", []):
            lines.append(f"- `{row['column']}`: status=`{row['status']}` PSI=`{row['psi']}`")

    labels = validation.get("labels")

    lines.extend(["", "## Labels", ""])

    if labels:
        lines.extend(
            [
                f"- Status: `{labels.get('status')}`",
                f"- Linhas: `{labels.get('n_rows_labels')}`",
                "",
            ]
        )

        for check in labels.get("checks", []):
            lines.append(f"- `{check['check_id']}`: `{check['status']}` esperado=`{check['expected']}` atual=`{check['actual']}`")
    else:
        lines.append("Labels não informados nesta execução.")

    (OUTPUT_DIR / "09_new_data_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment_spec() -> None:
    lines = [
        "# Próximo experimento recomendado",
        "",
        "## EXP-010B — New Data Dry Run",
        "",
        "## Objetivo",
        "",
        "Executar o baseline `post_fase2_c1` em uma nova janela de dados validada pelo contrato do EXP-010A.",
        "",
        "## Pré-requisito",
        "",
        "Ter um arquivo novo de transações que passe no contrato de entrada.",
        "",
        "## Ações",
        "",
        "- Validar novo dataset com `DATA_INTAKE_CONTRACT`.",
        "- Rodar baseline atual sem alterar modelo.",
        "- Gerar decision logs no padrão EXP-009A.",
        "- Rodar drift monitor EXP-009B contra a referência.",
        "- Gerar fila de revisão EXP-009C para a nova janela.",
        "- Comparar métricas, se labels novos existirem.",
        "",
        "## Critério de aprovação",
        "",
        "- Baseline roda na nova janela sem erro.",
        "- Logs estruturados são gerados com schema válido.",
        "- Drift e métricas são calculados.",
        "- Nenhuma mudança é promovida ainda.",
    ]

    (OUTPUT_DIR / "10_next_experiment_spec.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-010A Data Intake Contract")
    parser.add_argument("--new-transactions", type=str, default="")
    parser.add_argument("--new-labels", type=str, default="")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-010A — Data Intake Contract e Harness de Reavaliacao")
    print("=" * 72)

    print("[1/8] Carregando dataset de referencia...")
    reference_df, reference_source = load_reference_dataset()
    print(f"[OK] Referencia: {len(reference_df)} linhas | {len(reference_df.columns)} colunas")
    print(f"[OK] Fonte: {reference_source}")

    print("[2/8] Gerando contrato de transacoes...")
    contract_df = build_transaction_contract(reference_df)
    contract_df.to_csv(OUTPUT_DIR / "01_transaction_schema_contract.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT_DIR / "02_transaction_schema_contract.json", contract_to_json(contract_df))

    print("[3/8] Gerando contrato de labels...")
    label_contract = build_label_contract()
    write_json(OUTPUT_DIR / "03_label_schema_contract.json", label_contract)

    print("[4/8] Gerando perfil da base de referencia...")
    reference_profile = pd.DataFrame([profile_column(reference_df, col) for col in reference_df.columns])
    reference_profile.to_csv(OUTPUT_DIR / "04_reference_data_profile.csv", index=False, encoding="utf-8-sig")

    print("[5/8] Gerando relatorio de compatibilidade de features...")
    manifest = read_json(MANIFEST_PATH, {})
    compatibility_df = build_feature_compatibility_report(contract_df, manifest)
    compatibility_df.to_csv(OUTPUT_DIR / "05_feature_compatibility_report.csv", index=False, encoding="utf-8-sig")

    print("[6/8] Escrevendo documentacao do contrato e harness...")
    write_data_contract_doc(contract_df, label_contract)
    write_revaluation_harness_spec()

    optional_validation = None

    if args.new_transactions:
        print("[7/8] Validando novo arquivo de transacoes/labels...")
        new_tx_path = Path(args.new_transactions).resolve()
        new_tx = read_table(new_tx_path)

        tx_validation = validate_new_transactions(reference_df, new_tx, contract_df)

        labels_validation = None

        if args.new_labels:
            new_labels_path = Path(args.new_labels).resolve()
            labels_df = read_table(new_labels_path)
            labels_validation = validate_new_labels(new_tx, labels_df)

        status_candidates = [tx_validation["status"]]

        if labels_validation:
            status_candidates.append(labels_validation["status"])

        if "FAIL" in status_candidates:
            status = "FAIL"
        elif "ALERT" in status_candidates:
            status = "ALERT"
        elif "WARN" in status_candidates:
            status = "WARN"
        else:
            status = "PASS"

        optional_validation = {
            "status": status,
            "new_transactions_path": str(new_tx_path),
            "new_labels_path": str(Path(args.new_labels).resolve()) if args.new_labels else None,
            "transactions": tx_validation,
            "labels": labels_validation,
        }

        write_json(OUTPUT_DIR / "08_new_data_validation.json", optional_validation)
        write_validation_report(optional_validation)

    else:
        print("[7/8] Nenhum novo arquivo informado; validacao opcional pulada.")

    print("[8/8] Escrevendo sumario e proximo experimento...")
    input_summary = {
        "generated_at": now_iso(),
        "reference_source": reference_source,
        "reference_rows": int(len(reference_df)),
        "reference_columns": int(len(reference_df.columns)),
        "output_dir": str(OUTPUT_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_model_version": manifest.get("model_version"),
        "manifest_status": manifest.get("status"),
        "new_data_validation_status": optional_validation.get("status") if optional_validation else "NOT_RUN",
        "artifacts": [
            "01_transaction_schema_contract.csv",
            "02_transaction_schema_contract.json",
            "03_label_schema_contract.json",
            "04_reference_data_profile.csv",
            "05_feature_compatibility_report.csv",
            "06_DATA_INTAKE_CONTRACT.md",
            "07_REVALUATION_HARNESS_SPEC.md",
            "10_next_experiment_spec.md",
        ],
    }

    write_json(OUTPUT_DIR / "00_input_summary.json", input_summary)
    write_next_experiment_spec()

    print()
    print("[OK] EXP-010A concluido.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '00_input_summary.json'}")
    print(f"  {OUTPUT_DIR / '01_transaction_schema_contract.csv'}")
    print(f"  {OUTPUT_DIR / '02_transaction_schema_contract.json'}")
    print(f"  {OUTPUT_DIR / '03_label_schema_contract.json'}")
    print(f"  {OUTPUT_DIR / '04_reference_data_profile.csv'}")
    print(f"  {OUTPUT_DIR / '05_feature_compatibility_report.csv'}")
    print(f"  {OUTPUT_DIR / '06_DATA_INTAKE_CONTRACT.md'}")
    print(f"  {OUTPUT_DIR / '07_REVALUATION_HARNESS_SPEC.md'}")
    print(f"  {OUTPUT_DIR / '10_next_experiment_spec.md'}")

    if optional_validation:
        print(f"  {OUTPUT_DIR / '08_new_data_validation.json'}")
        print(f"  {OUTPUT_DIR / '09_new_data_validation_report.md'}")


if __name__ == "__main__":
    main()