"""
EXP-009E — Smoke Test de Reprodutibilidade e Pacote de Governança

Objetivo:
  Criar um comando unico que valide a governanca minima do projeto antes de
  qualquer nova rodada experimental.

Este experimento:
  - Nao altera o modelo.
  - Nao altera scoring_config.json.
  - Nao altera DecisionEngine.
  - Nao roda E2E pesado.
  - Roda py_compile nos modulos criticos.
  - Roda regressao pos-C1 normal e slow.
  - Valida existencia dos artefatos oficiais.
  - Verifica hashes registrados no MANIFEST_MODEL.json.
  - Valida logs estruturados, drift monitor, fila de revisao e dashboard operacional.
  - Gera relatorio GOVERNANCE_SMOKE_TEST.md.

Saidas:
  resultados/experimentos/EXP-009E/
    00_smoke_summary.json
    01_command_results.json
    02_artifact_checks.csv
    03_hash_checks.csv
    04_data_quality_checks.csv
    05_GOVERNANCE_SMOKE_TEST.md
    06_next_experiment_spec.md

Uso:
  python experimentos\\exp_009e_governance_smoke_test\\run_exp_009e_governance_smoke_test.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Paths
# ============================================================

EXP_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent.parent


ROOT = find_project_root(EXP_DIR)

OUTPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-009E"

BACKEND_DIR = ROOT / "backend"
DOCS_DIR = ROOT / "docs"
RESULTADOS_DIR = ROOT / "resultados" / "experimentos"

DECISION_ENGINE_PATH = BACKEND_DIR / "core" / "decision_engine.py"
PIPELINE_PATH = BACKEND_DIR / "core" / "pipeline_orquestrador.py"
SIMULAR_PATH = BACKEND_DIR / "scripts" / "simular_pipeline_e2e_v2.py"

SCORING_PATH = BACKEND_DIR / "artefatos" / "scoring_config.json"
MANIFEST_PATH = BACKEND_DIR / "artefatos" / "MANIFEST_MODEL.json"

REGRESSION_TEST_PATH = ROOT / "tests" / "test_regression_post_fase2.py"

VALIDATION_REPORT_PATH = DOCS_DIR / "VALIDATION_REPORT_POST_FASE2.md"
RULES_CATALOG_PATH = DOCS_DIR / "RULES_CATALOG.md"
DECISION_TRACE_SPEC_PATH = DOCS_DIR / "DECISION_TRACE_SPEC.md"
DECISION_TRACE_EXAMPLE_PATH = DOCS_DIR / "DECISION_TRACE_EXAMPLE.json"
JOURNAL_PATH = DOCS_DIR / "JOURNAL.md"

EXP009A_SCHEMA_PATH = RESULTADOS_DIR / "EXP-009A" / "04_schema_validation.json"
EXP009A_DECISION_LOG_PATH = RESULTADOS_DIR / "EXP-009A" / "03_decision_log_all.jsonl"

EXP009B_ALERTS_PATH = RESULTADOS_DIR / "EXP-009B" / "05_alerts.json"
EXP009B_REPORT_PATH = RESULTADOS_DIR / "EXP-009B" / "06_drift_report.md"

EXP009C_QUEUE_PATH = RESULTADOS_DIR / "EXP-009C" / "02_review_queue_dedup.csv"
EXP009C_REPORT_PATH = RESULTADOS_DIR / "EXP-009C" / "07_recommendation.md"

EXP009D_INPUT_SUMMARY_PATH = RESULTADOS_DIR / "EXP-009D" / "00_input_summary.json"
EXP009D_KPI_PATH = RESULTADOS_DIR / "EXP-009D" / "01_kpi_overall.csv"
EXP009D_DECISION_FACT_PATH = RESULTADOS_DIR / "EXP-009D" / "07_powerbi_decision_fact.csv"
EXP009D_REVIEW_FACT_PATH = RESULTADOS_DIR / "EXP-009D" / "08_powerbi_review_queue_fact.csv"
EXP009D_DASHBOARD_README_PATH = RESULTADOS_DIR / "EXP-009D" / "09_dashboard_readme.md"


# ============================================================
# Helpers
# ============================================================

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


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def status_rank(status: str) -> int:
    order = {
        "PASS": 0,
        "WARN": 1,
        "FAIL": 2,
    }
    return order.get(status, 2)


def overall_status(rows: list[dict[str, Any]]) -> str:
    statuses = [str(r.get("status", "FAIL")) for r in rows]

    if any(s == "FAIL" for s in statuses):
        return "FAIL"

    if any(s == "WARN" for s in statuses):
        return "WARN"

    return "PASS"


def add_check(
    rows: list[dict[str, Any]],
    *,
    group: str,
    check_id: str,
    status: str,
    message: str,
    path: str = "",
    expected: Any = None,
    actual: Any = None,
    critical: bool = True,
) -> None:
    rows.append(
        {
            "group": group,
            "check_id": check_id,
            "status": status,
            "critical": bool(critical),
            "message": message,
            "path": path,
            "expected": expected,
            "actual": actual,
        }
    )


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def years_from(series: pd.Series) -> list[int]:
    years = pd.to_datetime(series, errors="coerce").dt.year.dropna().astype(int).unique()
    return sorted(years.tolist())


# ============================================================
# Command checks
# ============================================================

def run_command(command: list[str], timeout: int = 180) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )

    return {
        "command": " ".join(command),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }


def run_command_suite() -> list[dict[str, Any]]:
    import_check = (
        "import sys; "
        "sys.path.insert(0, 'backend'); "
        "sys.path.insert(0, 'backend/core'); "
        "from core.decision_engine import PixDecisionEngine, EngineConfig; "
        "print('hydrate OK', hasattr(PixDecisionEngine, '_hydrate_config_from_scoring_config'), "
        "'C1 field OK', 'exp006f_c1_enabled' in EngineConfig.__dataclass_fields__)"
    )

    commands = [
        [sys.executable, "-m", "py_compile", "backend/core/decision_engine.py"],
        [sys.executable, "-m", "py_compile", "backend/core/pipeline_orquestrador.py"],
        [sys.executable, "-m", "py_compile", "backend/scripts/simular_pipeline_e2e_v2.py"],
        [sys.executable, "-c", import_check],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q", "-m", "slow"],
    ]

    return [run_command(cmd) for cmd in commands]


# ============================================================
# Artifact checks
# ============================================================

def artifact_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    required_artifacts = [
        ("critical_code", "decision_engine_exists", DECISION_ENGINE_PATH),
        ("critical_code", "pipeline_orquestrador_exists", PIPELINE_PATH),
        ("critical_code", "simular_pipeline_exists", SIMULAR_PATH),
        ("config", "scoring_config_exists", SCORING_PATH),
        ("config", "manifest_exists", MANIFEST_PATH),
        ("tests", "regression_test_exists", REGRESSION_TEST_PATH),
        ("docs", "validation_report_exists", VALIDATION_REPORT_PATH),
        ("docs", "rules_catalog_exists", RULES_CATALOG_PATH),
        ("docs", "decision_trace_spec_exists", DECISION_TRACE_SPEC_PATH),
        ("docs", "decision_trace_example_exists", DECISION_TRACE_EXAMPLE_PATH),
        ("docs", "journal_exists", JOURNAL_PATH),
        ("exp009a", "decision_log_exists", EXP009A_DECISION_LOG_PATH),
        ("exp009a", "schema_validation_exists", EXP009A_SCHEMA_PATH),
        ("exp009b", "drift_alerts_exists", EXP009B_ALERTS_PATH),
        ("exp009b", "drift_report_exists", EXP009B_REPORT_PATH),
        ("exp009c", "review_queue_exists", EXP009C_QUEUE_PATH),
        ("exp009c", "review_report_exists", EXP009C_REPORT_PATH),
        ("exp009d", "dashboard_input_summary_exists", EXP009D_INPUT_SUMMARY_PATH),
        ("exp009d", "kpi_overall_exists", EXP009D_KPI_PATH),
        ("exp009d", "powerbi_decision_fact_exists", EXP009D_DECISION_FACT_PATH),
        ("exp009d", "powerbi_review_fact_exists", EXP009D_REVIEW_FACT_PATH),
        ("exp009d", "dashboard_readme_exists", EXP009D_DASHBOARD_README_PATH),
    ]

    for group, check_id, path in required_artifacts:
        exists = path.exists()
        add_check(
            rows,
            group=group,
            check_id=check_id,
            status="PASS" if exists else "FAIL",
            message="arquivo encontrado" if exists else "arquivo ausente",
            path=str(path),
            expected=True,
            actual=exists,
            critical=True,
        )

    return rows


# ============================================================
# Hash checks
# ============================================================

def hash_checks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    artifact_hashes = manifest.get("artifact_hashes", {}) or {}

    hash_key_to_path = {
        "scoring_config_sha256": SCORING_PATH,
        "validation_report_sha256": VALIDATION_REPORT_PATH,
        "rules_catalog_sha256": RULES_CATALOG_PATH,
        "decision_trace_spec_sha256": DECISION_TRACE_SPEC_PATH,
        "decision_trace_example_sha256": DECISION_TRACE_EXAMPLE_PATH,
        "regression_test_sha256": REGRESSION_TEST_PATH,
    }

    for key, path in hash_key_to_path.items():
        expected = artifact_hashes.get(key)
        actual = sha256_file(path)

        if expected is None:
            status = "WARN"
            message = "hash nao registrado no manifest"
        elif actual is None:
            status = "FAIL"
            message = "arquivo ausente para hash"
        elif actual == expected:
            status = "PASS"
            message = "hash confere"
        else:
            status = "WARN"
            message = "hash diferente do manifest; verificar se o artefato foi editado apos o EXP-008E"

        add_check(
            rows,
            group="hash",
            check_id=key,
            status=status,
            message=message,
            path=str(path),
            expected=expected,
            actual=actual,
            critical=(status == "FAIL"),
        )

    return rows


# ============================================================
# Data quality checks
# ============================================================

def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    expected_values = {
        "model_version": "post_fase2_c1",
        "status": "ACTIVE_BASELINE",
    }

    for key, expected in expected_values.items():
        actual = manifest.get(key)
        add_check(
            rows,
            group="manifest",
            check_id=f"manifest_{key}",
            status="PASS" if actual == expected else "FAIL",
            message="valor esperado" if actual == expected else "valor inesperado",
            path=str(MANIFEST_PATH),
            expected=expected,
            actual=actual,
        )

    active_rules = set(manifest.get("active_rules", []) or [])
    expected_rules = {
        "V1_GUARD_CONTEXTUAL",
        "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER",
    }

    add_check(
        rows,
        group="manifest",
        check_id="manifest_active_rules",
        status="PASS" if expected_rules.issubset(active_rules) else "FAIL",
        message="regras ativas esperadas presentes" if expected_rules.issubset(active_rules) else "regras ativas esperadas ausentes",
        path=str(MANIFEST_PATH),
        expected=sorted(expected_rules),
        actual=sorted(active_rules),
    )

    metrics = manifest.get("official_metrics", {}) or {}

    expected_metrics = {
        "seed_42": {"TP": 347, "FP": 14, "FN": 8},
        "seed_123": {"TP": 347, "FP": 12, "FN": 8},
    }

    for seed, expected in expected_metrics.items():
        actual = metrics.get(seed, {})
        ok = all(int(actual.get(k, -999)) == v for k, v in expected.items())

        add_check(
            rows,
            group="manifest",
            check_id=f"official_metrics_{seed}",
            status="PASS" if ok else "FAIL",
            message="metricas oficiais conferem" if ok else "metricas oficiais divergentes",
            path=str(MANIFEST_PATH),
            expected=expected,
            actual=actual,
        )

    return rows


def validate_scoring_config() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config = read_json(SCORING_PATH, {})

    required_keys = [
        "exp006f_c1_enabled",
        "exp006f_c1_min_score",
        "exp006f_c1_max_score",
        "exp006f_c1_min_valor",
        "exp006f_c1_max_valor",
        "exp006f_c1_max_rel_meses",
        "exp006f_c1_min_lgbm_raw",
        "exp006f_c1_max_lgbm_raw",
        "exp006f_c1_require_first_receiver",
        "exp006f_c1_require_not_pix_random",
    ]

    missing = [k for k in required_keys if k not in config]

    add_check(
        rows,
        group="scoring_config",
        check_id="scoring_config_c1_keys",
        status="PASS" if not missing else "FAIL",
        message="chaves C1 presentes" if not missing else "chaves C1 ausentes",
        path=str(SCORING_PATH),
        expected=[],
        actual=missing,
    )

    add_check(
        rows,
        group="scoring_config",
        check_id="scoring_config_c1_enabled",
        status="PASS" if bool(config.get("exp006f_c1_enabled")) else "FAIL",
        message="C1 habilitada" if bool(config.get("exp006f_c1_enabled")) else "C1 desabilitada",
        path=str(SCORING_PATH),
        expected=True,
        actual=config.get("exp006f_c1_enabled"),
    )

    add_check(
        rows,
        group="scoring_config",
        check_id="scoring_config_c1_min_score",
        status="PASS" if float(config.get("exp006f_c1_min_score", -1)) == 58.0 else "FAIL",
        message="min_score C1 confere" if float(config.get("exp006f_c1_min_score", -1)) == 58.0 else "min_score C1 divergente",
        path=str(SCORING_PATH),
        expected=58.0,
        actual=config.get("exp006f_c1_min_score"),
    )

    return rows


def validate_exp009a() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schema = read_json(EXP009A_SCHEMA_PATH, {})

    add_check(
        rows,
        group="exp009a",
        check_id="schema_ok",
        status="PASS" if schema.get("ok") is True else "FAIL",
        message="schema EXP-009A ok" if schema.get("ok") is True else "schema EXP-009A falhou",
        path=str(EXP009A_SCHEMA_PATH),
        expected=True,
        actual=schema.get("ok"),
    )

    add_check(
        rows,
        group="exp009a",
        check_id="decision_log_rows",
        status="PASS" if int(schema.get("n_rows", 0)) == 12000 else "FAIL",
        message="decision log com 12000 linhas" if int(schema.get("n_rows", 0)) == 12000 else "quantidade de linhas divergente",
        path=str(EXP009A_SCHEMA_PATH),
        expected=12000,
        actual=schema.get("n_rows"),
    )

    return rows


def validate_exp009b() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alerts = read_json(EXP009B_ALERTS_PATH, {})

    status = alerts.get("status")

    add_check(
        rows,
        group="exp009b",
        check_id="drift_status_ok",
        status="PASS" if status == "OK" else "WARN",
        message="drift monitor OK" if status == "OK" else "drift monitor com status diferente de OK",
        path=str(EXP009B_ALERTS_PATH),
        expected="OK",
        actual=status,
        critical=False,
    )

    add_check(
        rows,
        group="exp009b",
        check_id="drift_alerts_zero",
        status="PASS" if int(alerts.get("n_alerts", -1)) == 0 else "WARN",
        message="sem alertas de drift" if int(alerts.get("n_alerts", -1)) == 0 else "ha alertas de drift",
        path=str(EXP009B_ALERTS_PATH),
        expected=0,
        actual=alerts.get("n_alerts"),
        critical=False,
    )

    return rows


def validate_exp009c() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not EXP009C_QUEUE_PATH.exists():
        add_check(
            rows,
            group="exp009c",
            check_id="review_queue_readable",
            status="FAIL",
            message="fila de revisao ausente",
            path=str(EXP009C_QUEUE_PATH),
            expected=True,
            actual=False,
        )
        return rows

    queue = read_csv_safe(EXP009C_QUEUE_PATH)

    add_check(
        rows,
        group="exp009c",
        check_id="review_queue_rows",
        status="PASS" if len(queue) > 0 else "FAIL",
        message="fila de revisao possui itens" if len(queue) > 0 else "fila de revisao vazia",
        path=str(EXP009C_QUEUE_PATH),
        expected=">0",
        actual=len(queue),
    )

    add_check(
        rows,
        group="exp009c",
        check_id="review_queue_expected_rows",
        status="PASS" if len(queue) == 730 else "WARN",
        message="fila com 730 itens deduplicados" if len(queue) == 730 else "quantidade diferente de 730; verificar se criterios mudaram",
        path=str(EXP009C_QUEUE_PATH),
        expected=730,
        actual=len(queue),
        critical=False,
    )

    return rows


def validate_exp009d() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not EXP009D_DECISION_FACT_PATH.exists():
        add_check(
            rows,
            group="exp009d",
            check_id="decision_fact_exists",
            status="FAIL",
            message="powerbi decision fact ausente",
            path=str(EXP009D_DECISION_FACT_PATH),
            expected=True,
            actual=False,
        )
        return rows

    fact = read_csv_safe(EXP009D_DECISION_FACT_PATH)
    kpi = read_csv_safe(EXP009D_KPI_PATH)
    review_fact = read_csv_safe(EXP009D_REVIEW_FACT_PATH)

    overall = kpi[(kpi["scope"] == "overall") & (kpi["label"] == "all")]

    if overall.empty:
        add_check(
            rows,
            group="exp009d",
            check_id="kpi_overall_row",
            status="FAIL",
            message="linha overall/all ausente no KPI",
            path=str(EXP009D_KPI_PATH),
            expected=True,
            actual=False,
        )
    else:
        row = overall.iloc[0].to_dict()

        expected = {
            "n_decisions": 12000,
            "n_aprovar": 11280,
            "n_confirmar": 131,
            "n_bloquear": 589,
            "n_c1_applied": 2,
        }

        for key, exp in expected.items():
            actual = int(row.get(key, -999))
            add_check(
                rows,
                group="exp009d",
                check_id=f"kpi_{key}",
                status="PASS" if actual == exp else "FAIL",
                message=f"{key} confere" if actual == exp else f"{key} divergente",
                path=str(EXP009D_KPI_PATH),
                expected=exp,
                actual=actual,
            )

    date_empty = int(fact["transaction_date"].isna().sum()) + int(
        fact["transaction_date"].fillna("").astype(str).str.strip().eq("").sum()
    )
    fact_years = years_from(fact["transaction_date"])

    allowed_years = {2025, 2026}

    add_check(
        rows,
        group="exp009d",
        check_id="decision_fact_date_not_empty",
        status="PASS" if date_empty == 0 else "FAIL",
        message="transaction_date preenchida" if date_empty == 0 else "transaction_date vazia",
        path=str(EXP009D_DECISION_FACT_PATH),
        expected=0,
        actual=date_empty,
    )

    add_check(
        rows,
        group="exp009d",
        check_id="decision_fact_years_valid",
        status="PASS" if set(fact_years).issubset(allowed_years) else "FAIL",
        message="anos do decision_fact validos" if set(fact_years).issubset(allowed_years) else "anos invalidos no decision_fact",
        path=str(EXP009D_DECISION_FACT_PATH),
        expected=sorted(allowed_years),
        actual=fact_years,
    )

    review_empty = int(review_fact["transaction_date"].isna().sum()) + int(
        review_fact["transaction_date"].fillna("").astype(str).str.strip().eq("").sum()
    )
    review_years = years_from(review_fact["transaction_date"])

    add_check(
        rows,
        group="exp009d",
        check_id="review_fact_date_not_empty",
        status="PASS" if review_empty == 0 else "FAIL",
        message="transaction_date da fila preenchida" if review_empty == 0 else "transaction_date vazia na fila",
        path=str(EXP009D_REVIEW_FACT_PATH),
        expected=0,
        actual=review_empty,
    )

    add_check(
        rows,
        group="exp009d",
        check_id="review_fact_years_valid",
        status="PASS" if set(review_years).issubset(allowed_years) else "FAIL",
        message="anos da fila validos" if set(review_years).issubset(allowed_years) else "anos invalidos na fila",
        path=str(EXP009D_REVIEW_FACT_PATH),
        expected=sorted(allowed_years),
        actual=review_years,
    )

    return rows


def validate_journal() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not JOURNAL_PATH.exists():
        add_check(
            rows,
            group="journal",
            check_id="journal_exists",
            status="FAIL",
            message="JOURNAL.md ausente",
            path=str(JOURNAL_PATH),
            expected=True,
            actual=False,
        )
        return rows

    text = JOURNAL_PATH.read_text(encoding="utf-8")

    expected_entries = [
        "FASE2_BASELINE_POS_C1",
        "EXP008A_REGRESSION_SUITE_POS_C1",
        "EXP008B_VALIDATION_REPORT",
        "EXP008C_RULES_TRACE",
        "EXP008D_CLEANUP_RESTRICAO",
        "EXP008E_JOURNAL_CRIADO",
        "EXP009A_DECISION_LOGGING_APROVADO",
        "EXP009B_DRIFT_MONITOR_APROVADO",
        "EXP009D_OPERATIONAL_DASHBOARD_APROVADO",
    ]

    missing = [entry for entry in expected_entries if entry not in text]

    add_check(
        rows,
        group="journal",
        check_id="journal_expected_entries",
        status="PASS" if not missing else "WARN",
        message="entradas esperadas presentes" if not missing else "algumas entradas esperadas nao foram encontradas",
        path=str(JOURNAL_PATH),
        expected=[],
        actual=missing,
        critical=False,
    )

    return rows


def data_quality_checks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.extend(validate_manifest(manifest))
    rows.extend(validate_scoring_config())
    rows.extend(validate_exp009a())
    rows.extend(validate_exp009b())
    rows.extend(validate_exp009c())
    rows.extend(validate_exp009d())
    rows.extend(validate_journal())

    return rows


# ============================================================
# Reports
# ============================================================

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")


def write_report(
    command_results: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    hash_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines: list[str] = [
        "# EXP-009E — Governance Smoke Test",
        "",
        f"Gerado em: `{summary['generated_at']}`",
        "",
        "## Objetivo",
        "",
        "Validar a governança mínima do baseline `post_fase2_c1` antes de qualquer nova rodada experimental.",
        "",
        "## Status geral",
        "",
        f"**Status:** `{summary['overall_status']}`",
        "",
        "| Grupo | Status | Total | PASS | WARN | FAIL |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for group_summary in summary["group_summaries"]:
        lines.append(
            f"| `{group_summary['group']}` | `{group_summary['status']}` | "
            f"{group_summary['total']} | {group_summary['pass']} | "
            f"{group_summary['warn']} | {group_summary['fail']} |"
        )

    lines.extend(
        [
            "",
            "## Comandos executados",
            "",
            "| Comando | Status | Return code |",
            "|---|---|---:|",
        ]
    )

    for r in command_results:
        lines.append(f"| `{r['command']}` | `{r['status']}` | {r['returncode']} |")

    fail_rows = [
        r for r in artifact_rows + hash_rows + data_rows
        if r.get("status") == "FAIL"
    ]
    warn_rows = [
        r for r in artifact_rows + hash_rows + data_rows
        if r.get("status") == "WARN"
    ]

    lines.extend(
        [
            "",
            "## Falhas",
            "",
        ]
    )

    if fail_rows:
        for r in fail_rows:
            lines.append(f"- `{r['group']}/{r['check_id']}`: {r['message']} | esperado=`{r.get('expected')}` atual=`{r.get('actual')}`")
    else:
        lines.append("- Nenhuma falha.")

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )

    if warn_rows:
        for r in warn_rows:
            lines.append(f"- `{r['group']}/{r['check_id']}`: {r['message']} | esperado=`{r.get('expected')}` atual=`{r.get('actual')}`")
    else:
        lines.append("- Nenhum warning.")

    lines.extend(
        [
            "",
            "## Decisão",
            "",
        ]
    )

    if summary["overall_status"] == "PASS":
        lines.append("EXP-009E aprovado: o pacote de governança passou sem falhas ou warnings.")
    elif summary["overall_status"] == "WARN":
        lines.append("EXP-009E aprovado com observações: não há falhas críticas, mas existem warnings que devem ser revisados.")
    else:
        lines.append("EXP-009E reprovado: há falhas críticas que devem ser corrigidas antes de prosseguir.")

    lines.extend(
        [
            "",
            "## Próximo passo",
            "",
            "Se aprovado, seguir para `EXP-010A — Data Intake Contract e Harness de Reavaliação com Novos Dados`.",
            "",
        ]
    )

    (OUTPUT_DIR / "05_GOVERNANCE_SMOKE_TEST.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment_spec() -> None:
    lines = [
        "# Próximo experimento recomendado",
        "",
        "## EXP-010A — Data Intake Contract e Harness de Reavaliação com Novos Dados",
        "",
        "## Objetivo",
        "",
        "Preparar o projeto para a fase de reavaliação completa com novos dados de transações normais e novos casos de fraude.",
        "",
        "## Por que este passo",
        "",
        "Os experimentos atuais indicam que os FNs residuais estão próximos do limite dos sinais disponíveis. Antes de novo treinamento ou novas regras, o projeto precisa de um contrato claro de entrada de dados e de um harness reproduzível para comparar baseline atual contra novas janelas.",
        "",
        "## Ações",
        "",
        "- Definir schema esperado para novos dados de transações.",
        "- Definir schema esperado para novos labels de fraude.",
        "- Criar validador de dataset novo.",
        "- Criar relatório de compatibilidade de features.",
        "- Criar harness para aplicar o baseline `post_fase2_c1` em nova janela.",
        "- Gerar comparação baseline antigo vs nova janela.",
        "- Preparar trilha futura de retreinamento, se houver novos casos suficientes.",
        "",
        "## Critério de aprovação",
        "",
        "- Novo dataset passa no contrato de schema.",
        "- Todas as features críticas existem ou possuem fallback documentado.",
        "- Baseline atual roda na nova janela sem erro.",
        "- Métricas e drift da nova janela são calculados sem intervenção manual.",
    ]

    (OUTPUT_DIR / "06_next_experiment_spec.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_groups(
    command_results: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    hash_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []

    for r in command_results:
        all_rows.append(
            {
                "group": "commands",
                "status": r["status"],
            }
        )

    all_rows.extend(artifact_rows)
    all_rows.extend(hash_rows)
    all_rows.extend(data_rows)

    groups = sorted(set(str(r.get("group", "unknown")) for r in all_rows))

    group_summaries = []

    for group in groups:
        g = [r for r in all_rows if str(r.get("group", "unknown")) == group]
        statuses = [str(r.get("status", "FAIL")) for r in g]
        group_summaries.append(
            {
                "group": group,
                "status": overall_status(g),
                "total": len(g),
                "pass": statuses.count("PASS"),
                "warn": statuses.count("WARN"),
                "fail": statuses.count("FAIL"),
            }
        )

    overall = overall_status(all_rows)

    return {
        "generated_at": now_iso(),
        "overall_status": overall,
        "group_summaries": group_summaries,
        "n_total_checks": len(all_rows),
        "n_pass": sum(1 for r in all_rows if r.get("status") == "PASS"),
        "n_warn": sum(1 for r in all_rows if r.get("status") == "WARN"),
        "n_fail": sum(1 for r in all_rows if r.get("status") == "FAIL"),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-009E Governance Smoke Test")
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Retorna exit code 0 mesmo se houver WARN, desde que nao haja FAIL.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-009E — Governance Smoke Test")
    print("=" * 72)

    print("[1/6] Rodando comandos criticos...")
    command_results = run_command_suite()
    write_json(OUTPUT_DIR / "01_command_results.json", command_results)

    print("[2/6] Verificando artefatos...")
    artifact_rows = artifact_checks()
    write_csv(OUTPUT_DIR / "02_artifact_checks.csv", artifact_rows)

    print("[3/6] Carregando manifest...")
    manifest = read_json(MANIFEST_PATH, {})

    print("[4/6] Verificando hashes registrados...")
    hash_rows = hash_checks(manifest)
    write_csv(OUTPUT_DIR / "03_hash_checks.csv", hash_rows)

    print("[5/6] Verificando qualidade dos dados e governanca...")
    data_rows = data_quality_checks(manifest)
    write_csv(OUTPUT_DIR / "04_data_quality_checks.csv", data_rows)

    print("[6/6] Gerando relatorios...")
    summary = summarize_groups(
        command_results=command_results,
        artifact_rows=artifact_rows,
        hash_rows=hash_rows,
        data_rows=data_rows,
    )
    write_json(OUTPUT_DIR / "00_smoke_summary.json", summary)

    write_report(
        command_results=command_results,
        artifact_rows=artifact_rows,
        hash_rows=hash_rows,
        data_rows=data_rows,
        summary=summary,
    )

    write_next_experiment_spec()

    print()
    print("[OK] EXP-009E concluido.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print(f"[OK] Status geral: {summary['overall_status']}")
    print(f"[OK] Checks: total={summary['n_total_checks']} pass={summary['n_pass']} warn={summary['n_warn']} fail={summary['n_fail']}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '00_smoke_summary.json'}")
    print(f"  {OUTPUT_DIR / '01_command_results.json'}")
    print(f"  {OUTPUT_DIR / '02_artifact_checks.csv'}")
    print(f"  {OUTPUT_DIR / '03_hash_checks.csv'}")
    print(f"  {OUTPUT_DIR / '04_data_quality_checks.csv'}")
    print(f"  {OUTPUT_DIR / '05_GOVERNANCE_SMOKE_TEST.md'}")
    print(f"  {OUTPUT_DIR / '06_next_experiment_spec.md'}")

    if summary["overall_status"] == "FAIL":
        raise SystemExit(1)

    if summary["overall_status"] == "WARN" and not args.allow_warnings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()