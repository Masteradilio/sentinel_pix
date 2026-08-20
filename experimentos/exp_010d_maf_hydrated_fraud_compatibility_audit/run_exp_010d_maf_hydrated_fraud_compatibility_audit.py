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
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent.parent


ROOT = find_project_root(EXP_DIR)

OUTPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-010D"

DEFAULT_INPUT_CANDIDATES = [
    ROOT / "dados" / "dados_pix_fraudes_maf_hidratadas_v1.csv",
    ROOT / "dados" / "raw" / "dados_pix_fraudes_maf_hidratadas_v1.csv",
    ROOT / "dados" / "external" / "dados_pix_fraudes_maf_hidratadas_v1.csv",
    ROOT / "resultados" / "experimentos" / "EXP-010C" / "dados_pix_fraudes_maf_hidratadas_v1.csv",
    ROOT / "Artefatos" / "EXP-010C" / "dados_pix_fraudes_maf_hidratadas_v1.csv",
]

OLD_FRAUD_CANDIDATES = [
    ROOT / "dados" / "dados_pix_fraudes.csv",
    ROOT / "dados" / "raw" / "dados_pix_fraudes.csv",
    ROOT / "dados_pix_fraudes.csv",
]

CRITICAL_COLUMNS = [
    "cd_pix",
    "dt_pix",
    "cd_cpf_pagador",
    "cd_cpf_cnpj_recebedor",
    "ds_chave_pix",
    "ds_tipo_chave",
    "vl_pix",
    "is_fraud",
]

IMPORTANT_MODEL_COLUMNS = [
    "qt_total_pix_trimestre",
    "vl_mediana_pix_trimestre",
    "vl_desvio_padrao_pix_trimestre",
    "qt_intervalo_transacao_minuto",
    "qt_intervalo_mediana_trimestre",
    "qt_intervalo_desvio_padrao_trimestre",
    "qt_pix_dia_maximo_trimestre",
    "nr_idade",
    "qt_tempo_relacionamento_mes",
    "ds_sexo",
    "ds_estado_civil",
    "ds_segmento",
    "vl_renda_cliente",
    "qt_dependentes",
    "tp_primeiro_envio_recebedor_trimestre",
    "qt_envio_recebedor_trimestre",
]

MBK_COLUMNS = [
    "device_name",
    "app_version",
    "ip_address",
    "latencia_rede_ms",
    "tempo_interacao_ms",
    "tempo_processamento_host_ms",
    "metodo_autenticacao",
    "session_id",
    "cd_retorno",
    "topaz_risk_score",
    "topaz_transacao_rejeitada",
    "is_agendamento_recorrente",
]

LABEL_AUDIT_COLUMNS = [
    "model_scope_status",
    "label_status",
    "label_confidence",
    "fraud_type",
    "bank_direction",
    "triangulation_flag",
    "duplicate_conflict_flag",
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
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
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


def read_csv_any(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None

    for encoding in ["utf-8-sig", "utf-8", "utf-16", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Falha ao ler CSV {path}: {last_error}")


def find_default_input() -> Path:
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path

    matches = list(ROOT.rglob("dados_pix_fraudes_maf_hidratadas_v1.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Não encontrei dados_pix_fraudes_maf_hidratadas_v1.csv. "
        "Informe o caminho com --maf-csv."
    )


def find_old_fraud_file() -> Path | None:
    for path in OLD_FRAUD_CANDIDATES:
        if path.exists():
            return path

    matches = [
        p for p in ROOT.rglob("dados_pix_fraudes*.csv")
        if "maf_hidratadas" not in p.name.lower()
    ]

    if matches:
        matches = sorted(matches, key=lambda p: p.stat().st_size, reverse=True)
        return matches[0]

    return None


def normalize_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "1.0": True,
                "sim": True,
                "s": True,
                "false": False,
                "0": False,
                "0.0": False,
                "nao": False,
                "não": False,
                "n": False,
            }
        )
    )


def non_empty_rate(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0

    x = s.copy()

    valid = (
        x.notna()
        & ~x.astype(str).str.strip().str.lower().isin(
            ["", "nan", "none", "null", "<na>", "informação ausente", "informacao ausente"]
        )
    )

    return float(valid.mean())


def add_check(
    rows: list[dict[str, Any]],
    *,
    group: str,
    check_id: str,
    status: str,
    message: str,
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
            "expected": expected,
            "actual": actual,
        }
    )


def compute_overall_status(checks: list[dict[str, Any]]) -> str:
    critical_fails = [
        c for c in checks
        if c["status"] == "FAIL" and bool(c.get("critical", True))
    ]

    if critical_fails:
        return "FAIL"

    if any(c["status"] in {"FAIL", "WARN"} for c in checks):
        return "WARN"

    return "PASS"


def build_missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        empty_strings = int(s.fillna("").astype(str).str.strip().eq("").sum())
        non_empty = non_empty_rate(s)

        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_rows": len(df),
                "n_nulls": nulls,
                "n_empty_strings": empty_strings,
                "non_empty_rate": non_empty,
                "missing_like_rate": 1.0 - non_empty,
                "n_unique": int(s.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_like_rate", "column"],
        ascending=[False, True],
    )


def build_schema_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    expected_groups = {
        "critical": CRITICAL_COLUMNS,
        "important_model": IMPORTANT_MODEL_COLUMNS,
        "mbk": MBK_COLUMNS,
        "label_audit": LABEL_AUDIT_COLUMNS,
    }

    for group, columns in expected_groups.items():
        for col in columns:
            rows.append(
                {
                    "group": group,
                    "column": col,
                    "present": col in df.columns,
                    "dtype": str(df[col].dtype) if col in df.columns else None,
                    "non_empty_rate": non_empty_rate(df[col]) if col in df.columns else None,
                    "n_unique": int(df[col].nunique(dropna=True)) if col in df.columns else None,
                }
            )

    return pd.DataFrame(rows)


def build_temporal_report(df: pd.DataFrame) -> pd.DataFrame:
    if "dt_pix" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["dt_pix_parsed"] = pd.to_datetime(tmp["dt_pix"], errors="coerce")
    tmp = tmp[tmp["dt_pix_parsed"].notna()].copy()

    if tmp.empty:
        return pd.DataFrame()

    tmp["year"] = tmp["dt_pix_parsed"].dt.year
    tmp["month"] = tmp["dt_pix_parsed"].dt.to_period("M").astype(str)

    by_month = (
        tmp.groupby(["year", "month"], dropna=False)
        .agg(
            n_rows=("cd_pix", "count"),
            n_unique_cd_pix=("cd_pix", "nunique"),
            vl_pix_sum=("vl_pix", lambda x: pd.to_numeric(x, errors="coerce").sum()),
            vl_pix_median=("vl_pix", lambda x: pd.to_numeric(x, errors="coerce").median()),
        )
        .reset_index()
        .sort_values(["month"])
    )

    return by_month


def build_value_report(df: pd.DataFrame) -> pd.DataFrame:
    if "vl_pix" not in df.columns:
        return pd.DataFrame()

    v = pd.to_numeric(df["vl_pix"], errors="coerce").dropna()

    if v.empty:
        return pd.DataFrame()

    quantiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]

    rows = [
        {
            "metric": f"p{int(q * 100):02d}",
            "value": float(v.quantile(q)),
        }
        for q in quantiles
    ]

    rows.extend(
        [
            {"metric": "mean", "value": float(v.mean())},
            {"metric": "std", "value": float(v.std())},
            {"metric": "n", "value": int(v.shape[0])},
        ]
    )

    return pd.DataFrame(rows)


def build_entity_report(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if "cd_cpf_pagador" in df.columns:
        result["n_unique_payers"] = int(df["cd_cpf_pagador"].nunique(dropna=True))
        result["top_payers"] = (
            df["cd_cpf_pagador"]
            .astype(str)
            .value_counts()
            .head(20)
            .rename_axis("cd_cpf_pagador")
            .reset_index(name="n_rows")
            .to_dict(orient="records")
        )

    if "cd_cpf_cnpj_recebedor" in df.columns:
        result["n_unique_receivers"] = int(df["cd_cpf_cnpj_recebedor"].nunique(dropna=True))
        result["top_receivers"] = (
            df["cd_cpf_cnpj_recebedor"]
            .astype(str)
            .value_counts()
            .head(20)
            .rename_axis("cd_cpf_cnpj_recebedor")
            .reset_index(name="n_rows")
            .to_dict(orient="records")
        )

    return result


def build_overlap_report(maf_df: pd.DataFrame, old_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if old_path is None:
        return pd.DataFrame(), {"old_fraud_file": None, "status": "NOT_FOUND"}

    old_df = read_csv_any(old_path)

    old_id_col = None
    for candidate in ["cd_pix", "transaction_id", "ds_id_pix"]:
        if candidate in old_df.columns:
            old_id_col = candidate
            break

    if old_id_col is None:
        return pd.DataFrame(), {
            "old_fraud_file": str(old_path),
            "status": "NO_ID_COLUMN",
            "old_columns": list(old_df.columns),
        }

    maf_ids = set(maf_df["cd_pix"].astype(str).str.strip()) if "cd_pix" in maf_df.columns else set()
    old_ids = set(old_df[old_id_col].astype(str).str.strip())

    overlap = sorted(maf_ids & old_ids)

    overlap_df = pd.DataFrame({"cd_pix": overlap})

    summary = {
        "old_fraud_file": str(old_path),
        "old_id_col": old_id_col,
        "n_maf_ids": len(maf_ids),
        "n_old_ids": len(old_ids),
        "n_overlap": len(overlap),
        "overlap_rate_vs_maf": len(overlap) / max(len(maf_ids), 1),
        "overlap_rate_vs_old": len(overlap) / max(len(old_ids), 1),
        "status": "OK",
    }

    return overlap_df, summary


def build_checks(df: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        group="shape",
        check_id="rows_positive",
        status="PASS" if len(df) > 0 else "FAIL",
        message="CSV possui linhas" if len(df) > 0 else "CSV vazio",
        expected=">0",
        actual=len(df),
    )

    missing_critical = [c for c in CRITICAL_COLUMNS if c not in df.columns]

    add_check(
        checks,
        group="schema",
        check_id="critical_columns_present",
        status="PASS" if not missing_critical else "FAIL",
        message="colunas críticas presentes" if not missing_critical else "colunas críticas ausentes",
        expected=[],
        actual=missing_critical,
    )

    missing_important = [c for c in IMPORTANT_MODEL_COLUMNS if c not in df.columns]

    add_check(
        checks,
        group="schema",
        check_id="important_model_columns_present",
        status="PASS" if not missing_important else "WARN",
        message="colunas importantes presentes" if not missing_important else "colunas importantes ausentes",
        expected=[],
        actual=missing_important,
        critical=False,
    )

    if "cd_pix" in df.columns:
        n_dup = int(df["cd_pix"].astype(str).str.strip().duplicated().sum())
        add_check(
            checks,
            group="identity",
            check_id="cd_pix_unique",
            status="PASS" if n_dup == 0 else "FAIL",
            message="cd_pix sem duplicidade" if n_dup == 0 else "cd_pix duplicado",
            expected=0,
            actual=n_dup,
        )

        valid_e2e = df["cd_pix"].astype(str).str.strip().str.match(r"^E[A-Za-z0-9]{20,}$", na=False)
        valid_rate = float(valid_e2e.mean())

        add_check(
            checks,
            group="identity",
            check_id="cd_pix_e2e_format",
            status="PASS" if valid_rate >= 0.99 else "WARN",
            message="cd_pix parece E2E ID válido" if valid_rate >= 0.99 else "muitos cd_pix fora do padrão E2E",
            expected=">=0.99",
            actual=valid_rate,
            critical=False,
        )

    if "is_fraud" in df.columns:
        fraud_values = sorted(pd.to_numeric(df["is_fraud"], errors="coerce").dropna().unique().tolist())
        all_one = fraud_values == [1] or fraud_values == [1.0]

        add_check(
            checks,
            group="label",
            check_id="is_fraud_all_one",
            status="PASS" if all_one else "FAIL",
            message="is_fraud 100% positivo" if all_one else "is_fraud possui valores diferentes de 1",
            expected=[1],
            actual=fraud_values,
        )

    if "model_scope_status" in df.columns:
        values = sorted(df["model_scope_status"].dropna().astype(str).unique().tolist())
        ok = values == ["POSITIVE_FOR_CURRENT_MODEL"]

        add_check(
            checks,
            group="label",
            check_id="model_scope_positive",
            status="PASS" if ok else "FAIL",
            message="escopo positivo forte confirmado" if ok else "escopo contém valores inesperados",
            expected=["POSITIVE_FOR_CURRENT_MODEL"],
            actual=values,
        )

    if "label_status" in df.columns:
        values = sorted(df["label_status"].dropna().astype(str).unique().tolist())
        ok = values == ["CONFIRMED_FRAUD_CANDIDATE"]

        add_check(
            checks,
            group="label",
            check_id="label_status_confirmed",
            status="PASS" if ok else "FAIL",
            message="label_status confirmado" if ok else "label_status contém valores inesperados",
            expected=["CONFIRMED_FRAUD_CANDIDATE"],
            actual=values,
        )

    if "bank_direction" in df.columns:
        values = sorted(df["bank_direction"].dropna().astype(str).unique().tolist())
        ok = values == ["BRB_DEBITADO_PAGADOR"]

        add_check(
            checks,
            group="label",
            check_id="bank_direction_payer",
            status="PASS" if ok else "FAIL",
            message="direção compatível com modelo atual" if ok else "direção contém valores inesperados",
            expected=["BRB_DEBITADO_PAGADOR"],
            actual=values,
        )

    if "triangulation_flag" in df.columns:
        tri = normalize_bool_series(df["triangulation_flag"])
        tri_true = int((tri == True).sum())

        add_check(
            checks,
            group="label",
            check_id="triangulation_excluded",
            status="PASS" if tri_true == 0 else "FAIL",
            message="triangulação excluída" if tri_true == 0 else "há triangulação na base final",
            expected=0,
            actual=tri_true,
        )

    if "duplicate_conflict_flag" in df.columns:
        conf = normalize_bool_series(df["duplicate_conflict_flag"])
        conf_true = int((conf == True).sum())

        add_check(
            checks,
            group="label",
            check_id="duplicate_conflict_excluded",
            status="PASS" if conf_true == 0 else "FAIL",
            message="conflitos excluídos" if conf_true == 0 else "há conflitos na base final",
            expected=0,
            actual=conf_true,
        )

    if "dt_pix" in df.columns:
        dt = pd.to_datetime(df["dt_pix"], errors="coerce")
        bad_dates = int(dt.isna().sum())

        add_check(
            checks,
            group="datetime",
            check_id="dt_pix_parseable",
            status="PASS" if bad_dates == 0 else "FAIL",
            message="dt_pix parseável" if bad_dates == 0 else "dt_pix possui datas inválidas",
            expected=0,
            actual=bad_dates,
        )

    if "vl_pix" in df.columns:
        v = pd.to_numeric(df["vl_pix"], errors="coerce")
        bad_values = int(v.isna().sum())
        non_positive = int((v <= 0).sum())

        add_check(
            checks,
            group="value",
            check_id="vl_pix_numeric",
            status="PASS" if bad_values == 0 else "FAIL",
            message="vl_pix numérico" if bad_values == 0 else "vl_pix possui valores não numéricos",
            expected=0,
            actual=bad_values,
        )

        add_check(
            checks,
            group="value",
            check_id="vl_pix_positive",
            status="PASS" if non_positive == 0 else "WARN",
            message="vl_pix positivo" if non_positive == 0 else "há valores <= 0",
            expected=0,
            actual=non_positive,
            critical=False,
        )

    mbk_present = [c for c in MBK_COLUMNS if c in df.columns]

    if mbk_present:
        rates = {c: non_empty_rate(df[c]) for c in mbk_present}
        best_rate = max(rates.values()) if rates else 0.0

        add_check(
            checks,
            group="mbk",
            check_id="mbk_any_coverage",
            status="PASS" if best_rate >= 0.20 else "WARN",
            message="MBK tem cobertura relevante" if best_rate >= 0.20 else "MBK ausente ou com baixa cobertura",
            expected=">=0.20 em pelo menos uma coluna MBK",
            actual=rates,
            critical=False,
        )
    else:
        add_check(
            checks,
            group="mbk",
            check_id="mbk_columns_present",
            status="WARN",
            message="colunas MBK ausentes; esperado se ENABLE_MOBILE=False no EXP-010C",
            expected=MBK_COLUMNS,
            actual=[],
            critical=False,
        )

    return checks


def write_recommendation(
    df: pd.DataFrame,
    checks: list[dict[str, Any]],
    overlap_summary: dict[str, Any],
    temporal_df: pd.DataFrame,
) -> None:
    status = compute_overall_status(checks)

    lines = [
        "# EXP-010D — MAF Hydrated Fraud Compatibility Audit",
        "",
        f"Gerado em: `{now_iso()}`",
        "",
        "## Status",
        "",
        f"**Status geral:** `{status}`",
        "",
        "## Resumo executivo",
        "",
        f"- Linhas no CSV MAF hidratado: `{len(df)}`",
        f"- `cd_pix` únicos: `{df['cd_pix'].nunique() if 'cd_pix' in df.columns else 'N/A'}`",
        f"- Colunas: `{len(df.columns)}`",
        "",
    ]

    if "dt_pix" in df.columns:
        dt = pd.to_datetime(df["dt_pix"], errors="coerce")
        lines.extend(
            [
                f"- Menor `dt_pix`: `{dt.min()}`",
                f"- Maior `dt_pix`: `{dt.max()}`",
                "",
            ]
        )

    if "vl_pix" in df.columns:
        v = pd.to_numeric(df["vl_pix"], errors="coerce")
        lines.extend(
            [
                f"- Valor mediano: `{v.median():.2f}`",
                f"- Valor p95: `{v.quantile(0.95):.2f}`",
                f"- Valor máximo: `{v.max():.2f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Checks com falha ou warning",
            "",
        ]
    )

    problems = [c for c in checks if c["status"] != "PASS"]

    if not problems:
        lines.append("- Nenhum.")
    else:
        for c in problems:
            lines.append(
                f"- `{c['group']}/{c['check_id']}`: `{c['status']}` — "
                f"{c['message']} | esperado=`{c.get('expected')}` atual=`{c.get('actual')}`"
            )

    lines.extend(
        [
            "",
            "## Overlap com fraudes antigas",
            "",
        ]
    )

    if overlap_summary.get("status") == "OK":
        lines.extend(
            [
                f"- Arquivo antigo: `{overlap_summary.get('old_fraud_file')}`",
                f"- IDs MAF: `{overlap_summary.get('n_maf_ids')}`",
                f"- IDs antigos: `{overlap_summary.get('n_old_ids')}`",
                f"- Overlap: `{overlap_summary.get('n_overlap')}`",
                "",
            ]
        )
    else:
        lines.append(f"- Status: `{overlap_summary.get('status')}`")

    lines.extend(
        [
            "",
            "## Interpretação",
            "",
            "A base MAF hidratada deve ser tratada como nova fonte positiva forte, mas ainda não deve ser usada sozinha para treino.",
            "",
            "Ela precisa ser combinada com:",
            "",
            "1. hidratação MBK por chave no EXP-010E;",
            "2. amostragem de normais em 90/180 dias no EXP-010F;",
            "3. construção de dataset unificado no EXP-010G.",
            "",
        ]
    )

    if status == "FAIL":
        lines.extend(
            [
                "## Decisão",
                "",
                "Não avançar para treino. Corrigir falhas críticas antes de seguir.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Decisão",
                "",
                "Avançar para o EXP-010E — MBK Keyed Hydration Audit, mantendo o EXP-010D como validação local da base MAF.",
                "",
            ]
        )

    (OUTPUT_DIR / "09_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-010D MAF Hydrated Fraud Compatibility Audit")
    parser.add_argument("--maf-csv", type=str, default="", help="Caminho para dados_pix_fraudes_maf_hidratadas_v1.csv")
    parser.add_argument("--old-fraud-csv", type=str, default="", help="Opcional: CSV antigo de fraudes para medir overlap")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.maf_csv).resolve() if args.maf_csv else find_default_input()
    old_path = Path(args.old_fraud_csv).resolve() if args.old_fraud_csv else find_old_fraud_file()

    print("=" * 80)
    print("EXP-010D — MAF Hydrated Fraud Compatibility Audit")
    print("=" * 80)
    print(f"[INPUT] MAF CSV: {input_path}")
    print(f"[INPUT] Old fraud CSV: {old_path if old_path else 'NOT_FOUND'}")
    print(f"[OUTPUT] {OUTPUT_DIR}")

    print("[1/9] Lendo CSV MAF...")
    df = read_csv_any(input_path)

    print(f"[OK] Linhas: {len(df)} | Colunas: {len(df.columns)}")

    print("[2/9] Gerando checks...")
    checks = build_checks(df)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUTPUT_DIR / "01_checks.csv", index=False, encoding="utf-8-sig")

    print("[3/9] Gerando schema report...")
    schema_df = build_schema_report(df)
    schema_df.to_csv(OUTPUT_DIR / "02_schema_report.csv", index=False, encoding="utf-8-sig")

    print("[4/9] Gerando missingness profile...")
    missing_df = build_missingness_profile(df)
    missing_df.to_csv(OUTPUT_DIR / "03_missingness_profile.csv", index=False, encoding="utf-8-sig")

    print("[5/9] Gerando distribuição temporal...")
    temporal_df = build_temporal_report(df)
    temporal_df.to_csv(OUTPUT_DIR / "04_temporal_distribution.csv", index=False, encoding="utf-8-sig")

    print("[6/9] Gerando distribuição de valores...")
    value_df = build_value_report(df)
    value_df.to_csv(OUTPUT_DIR / "05_value_distribution.csv", index=False, encoding="utf-8-sig")

    print("[7/9] Gerando distribuição de entidades...")
    entity_report = build_entity_report(df)
    write_json(OUTPUT_DIR / "06_entity_distribution.json", entity_report)

    print("[8/9] Medindo overlap com fraudes antigas...")
    overlap_df, overlap_summary = build_overlap_report(df, old_path)
    overlap_df.head(1000).to_csv(OUTPUT_DIR / "07_overlap_with_old_frauds_sample.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT_DIR / "08_overlap_summary.json", overlap_summary)

    print("[9/9] Escrevendo sumário e recomendação...")
    overall = compute_overall_status(checks)

    summary = {
        "generated_at": now_iso(),
        "experiment": "EXP-010D",
        "status": overall,
        "input_path": str(input_path),
        "old_fraud_path": str(old_path) if old_path else None,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_cd_pix_unique": int(df["cd_pix"].nunique()) if "cd_pix" in df.columns else None,
        "n_checks": int(len(checks)),
        "n_pass": int(sum(c["status"] == "PASS" for c in checks)),
        "n_warn": int(sum(c["status"] == "WARN" for c in checks)),
        "n_fail": int(sum(c["status"] == "FAIL" for c in checks)),
        "overlap_summary": overlap_summary,
        "artifacts": [
            "01_checks.csv",
            "02_schema_report.csv",
            "03_missingness_profile.csv",
            "04_temporal_distribution.csv",
            "05_value_distribution.csv",
            "06_entity_distribution.json",
            "07_overlap_with_old_frauds_sample.csv",
            "08_overlap_summary.json",
            "09_recommendation.md",
        ],
    }

    write_json(OUTPUT_DIR / "00_run_summary.json", summary)
    write_recommendation(df, checks, overlap_summary, temporal_df)

    print()
    print("[OK] EXP-010D concluído.")
    print(f"[OK] Status: {overall}")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    for name in summary["artifacts"]:
        print(f"  {OUTPUT_DIR / name}")
    print(f"  {OUTPUT_DIR / '00_run_summary.json'}")

    if overall == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()