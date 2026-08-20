"""
EXP-009B — Drift Monitor Offline

Objetivo:
  Criar um monitor de drift offline para o baseline post_fase2_c1.

Modo default:
  - usa os logs estruturados do EXP-009A;
  - compara seed 42 como referencia contra seed 123 como current;
  - isso funciona como self-test do monitor, nao como drift real de producao.

Uso default:
  python experimentos\\exp_009b_drift_monitor\\run_exp_009b_drift_monitor.py

Uso futuro com logs novos:
  python experimentos\\exp_009b_drift_monitor\\run_exp_009b_drift_monitor.py ^
    --reference-log resultados\\experimentos\\EXP-009A\\01_decision_log_seed_42.csv ^
    --current-log resultados\\novos_logs\\decision_log_current.csv ^
    --label-current nova_janela_YYYYMMDD

Saidas:
  resultados/experimentos/EXP-009B/
    00_input_summary.json
    01_numeric_drift.csv
    02_categorical_drift.csv
    03_decision_distribution.csv
    04_rule_distribution.csv
    05_alerts.json
    06_drift_report.md
    07_next_experiment_spec.md
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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

DEFAULT_REF_LOG = ROOT / "resultados" / "experimentos" / "EXP-009A" / "01_decision_log_seed_42.csv"
DEFAULT_CUR_LOG = ROOT / "resultados" / "experimentos" / "EXP-009A" / "02_decision_log_seed_123.csv"
OUTPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-009B"

MANIFEST_PATH = ROOT / "backend" / "artefatos" / "MANIFEST_MODEL.json"


# ============================================================
# Config
# ============================================================

REQUIRED_FIELDS = [
    "decision_id",
    "transaction_id",
    "customer_id_hash",
    "created_at",
    "model_version",
    "decision_engine_version",
    "scoring_config_version",
    "decisao",
    "score_final",
    "lgbm_raw",
    "lgbm_mapped",
    "if_percentile",
    "se_score",
    "beh_score",
    "rules_applied",
    "guardrails_applied",
    "decision_reason",
    "review_recommended",
]

NUMERIC_FEATURES = [
    "score_final",
    "score_final_original",
    "lgbm_raw",
    "lgbm_mapped",
    "if_percentile",
    "se_score",
    "beh_score",
    "vl_pix",
    "qt_tempo_relacionamento_mes",
    "first_receiver_flag",
    "pix_key_random_flag",
]

CATEGORICAL_FEATURES = [
    "decisao",
    "model_version",
    "decision_engine_version",
    "scoring_config_version",
]

PSI_WARN = 0.10
PSI_ALERT = 0.25

CAT_WARN_PP = 2.0
CAT_ALERT_PP = 5.0

DECISION_RATE_ALERT_PP = 1.0
C1_RATE_ALERT_MULTIPLIER = 3.0
C1_RATE_MIN_ALERT = 0.001  # 0.1%


# ============================================================
# IO helpers
# ============================================================

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


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log nao encontrado: {path}")

    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "records" in data:
            return pd.DataFrame(data["records"])
        raise ValueError(f"JSON nao reconhecido como lista de registros: {path}")

    return pd.read_csv(path)


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    raw = str(value).strip()

    if not raw or raw.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        # fallback: valor simples separado por |
        if "|" in raw:
            return [x.strip() for x in raw.split("|") if x.strip()]
        return [raw]


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in NUMERIC_FEATURES + ["is_fraud"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "decisao" in out.columns:
        out["decisao"] = out["decisao"].astype(str)

    if "review_recommended" in out.columns:
        out["review_recommended"] = out["review_recommended"].astype(str).str.lower().isin(
            {"true", "1", "yes", "sim"}
        )

    return out


# ============================================================
# Validation
# ============================================================

def validate_trace_schema(df: pd.DataFrame, label: str) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in df.columns]

    null_counts = {}
    empty_counts = {}

    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            continue

        null_counts[field] = int(df[field].isna().sum())

        if df[field].dtype == object:
            empty_counts[field] = int(df[field].astype(str).str.strip().eq("").sum())

    duplicate_decision_ids = None
    if "decision_id" in df.columns:
        duplicate_decision_ids = int(df["decision_id"].duplicated().sum())

    invalid_decisions = []
    if "decisao" in df.columns:
        invalid_decisions = sorted(set(df["decisao"].astype(str)) - {"APROVAR", "CONFIRMAR", "BLOQUEAR"})

    required_string_fields = [
        "decision_id",
        "transaction_id",
        "customer_id_hash",
        "created_at",
        "model_version",
        "decision_engine_version",
        "scoring_config_version",
        "decisao",
        "decision_reason",
    ]

    empty_required = {
        field: empty_counts.get(field, 0)
        for field in required_string_fields
        if field in df.columns
    }

    ok = (
        not missing
        and duplicate_decision_ids == 0
        and not invalid_decisions
        and all(v == 0 for v in empty_required.values())
    )

    return {
        "label": label,
        "ok": ok,
        "n_rows": int(len(df)),
        "missing_fields": missing,
        "duplicate_decision_ids": duplicate_decision_ids,
        "invalid_decision_values": invalid_decisions,
        "empty_required_counts": empty_required,
        "null_counts": null_counts,
    }


# ============================================================
# Drift metrics
# ============================================================

def psi_from_series(reference: pd.Series, current: pd.Series, bins: int = 10) -> dict[str, Any]:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if len(ref) == 0 or len(cur) == 0:
        return {
            "psi": None,
            "status": "MISSING",
            "reason": "sem dados numericos suficientes",
            "bins": [],
        }

    if ref.nunique(dropna=True) <= 1:
        ref_value = float(ref.iloc[0])
        cur_same_rate = float((cur == ref_value).mean())
        psi = 0.0 if cur_same_rate == 1.0 else 1.0
        return {
            "psi": psi,
            "status": classify_psi(psi),
            "reason": "referencia constante",
            "bins": [
                {
                    "bin": f"constant_{ref_value}",
                    "reference_pct": 1.0,
                    "current_pct": cur_same_rate,
                }
            ],
        }

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(ref, quantiles))

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

    all_bins = list(ref_counts.index)

    eps = 1e-6
    psi_value = 0.0
    bin_rows = []

    for b in all_bins:
        ref_pct = float(ref_counts.get(b, 0) / max(len(ref), 1))
        cur_pct = float(cur_counts.get(b, 0) / max(len(cur), 1))

        ref_adj = max(ref_pct, eps)
        cur_adj = max(cur_pct, eps)

        psi_component = (cur_adj - ref_adj) * math.log(cur_adj / ref_adj)
        psi_value += psi_component

        bin_rows.append(
            {
                "bin": str(b),
                "reference_pct": ref_pct,
                "current_pct": cur_pct,
                "psi_component": psi_component,
            }
        )

    return {
        "psi": float(psi_value),
        "status": classify_psi(psi_value),
        "reason": "",
        "bins": bin_rows,
    }


def classify_psi(value: float | None) -> str:
    if value is None:
        return "MISSING"
    if value >= PSI_ALERT:
        return "ALERT"
    if value >= PSI_WARN:
        return "WARN"
    return "OK"


def numeric_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in NUMERIC_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            rows.append(
                {
                    "feature": feature,
                    "status": "MISSING",
                    "psi": None,
                    "reference_mean": None,
                    "current_mean": None,
                    "reference_median": None,
                    "current_median": None,
                    "reference_null_rate": None,
                    "current_null_rate": None,
                    "reason": "coluna ausente",
                }
            )
            continue

        ref = pd.to_numeric(reference[feature], errors="coerce")
        cur = pd.to_numeric(current[feature], errors="coerce")

        psi_result = psi_from_series(ref, cur)

        rows.append(
            {
                "feature": feature,
                "status": psi_result["status"],
                "psi": psi_result["psi"],
                "reference_mean": float(ref.mean()) if ref.notna().any() else None,
                "current_mean": float(cur.mean()) if cur.notna().any() else None,
                "reference_median": float(ref.median()) if ref.notna().any() else None,
                "current_median": float(cur.median()) if cur.notna().any() else None,
                "reference_null_rate": float(ref.isna().mean()),
                "current_null_rate": float(cur.isna().mean()),
                "reason": psi_result.get("reason", ""),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["status", "psi"],
        ascending=[True, False],
        na_position="last",
    )


def categorical_distribution(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(dtype="float64")

    s = df[feature].fillna("<NA>").astype(str)
    return s.value_counts(normalize=True)


def categorical_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in CATEGORICAL_FEATURES:
        ref_dist = categorical_distribution(reference, feature)
        cur_dist = categorical_distribution(current, feature)

        categories = sorted(set(ref_dist.index).union(set(cur_dist.index)))

        if not categories:
            rows.append(
                {
                    "feature": feature,
                    "category": None,
                    "reference_pct": None,
                    "current_pct": None,
                    "delta_pp": None,
                    "status": "MISSING",
                }
            )
            continue

        for cat in categories:
            ref_pct = float(ref_dist.get(cat, 0.0))
            cur_pct = float(cur_dist.get(cat, 0.0))
            delta_pp = (cur_pct - ref_pct) * 100

            abs_delta = abs(delta_pp)
            if abs_delta >= CAT_ALERT_PP:
                status = "ALERT"
            elif abs_delta >= CAT_WARN_PP:
                status = "WARN"
            else:
                status = "OK"

            rows.append(
                {
                    "feature": feature,
                    "category": cat,
                    "reference_pct": ref_pct,
                    "current_pct": cur_pct,
                    "delta_pp": delta_pp,
                    "status": status,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["status", "feature", "delta_pp"],
        ascending=[True, True, False],
    )


def decision_distribution(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []

    all_decisions = ["APROVAR", "CONFIRMAR", "BLOQUEAR"]

    for decision in all_decisions:
        ref_count = int((reference["decisao"].astype(str) == decision).sum()) if "decisao" in reference.columns else 0
        cur_count = int((current["decisao"].astype(str) == decision).sum()) if "decisao" in current.columns else 0

        ref_rate = ref_count / max(len(reference), 1)
        cur_rate = cur_count / max(len(current), 1)
        delta_pp = (cur_rate - ref_rate) * 100

        status = "OK"
        if abs(delta_pp) >= DECISION_RATE_ALERT_PP:
            status = "ALERT"
        elif abs(delta_pp) >= DECISION_RATE_ALERT_PP / 2:
            status = "WARN"

        rows.append(
            {
                "decisao": decision,
                "reference_count": ref_count,
                "current_count": cur_count,
                "reference_rate": ref_rate,
                "current_rate": cur_rate,
                "delta_pp": delta_pp,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def explode_rules(df: pd.DataFrame, col: str = "rules_applied") -> pd.Series:
    values: list[str] = []

    if col not in df.columns:
        return pd.Series(dtype="object")

    for raw in df[col].tolist():
        values.extend(parse_json_list(raw))

    return pd.Series(values, dtype="object")


def rule_distribution(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    ref_rules = explode_rules(reference, "rules_applied")
    cur_rules = explode_rules(current, "rules_applied")

    ref_counts = ref_rules.value_counts()
    cur_counts = cur_rules.value_counts()

    rules = sorted(set(ref_counts.index).union(set(cur_counts.index)))

    rows = []

    for rule in rules:
        ref_count = int(ref_counts.get(rule, 0))
        cur_count = int(cur_counts.get(rule, 0))

        ref_rate = ref_count / max(len(reference), 1)
        cur_rate = cur_count / max(len(current), 1)
        delta_pp = (cur_rate - ref_rate) * 100

        status = "OK"

        if rule == "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER":
            alert_threshold = max(ref_rate * C1_RATE_ALERT_MULTIPLIER, C1_RATE_MIN_ALERT)
            if cur_rate > alert_threshold:
                status = "ALERT"
            elif cur_count == 0 and ref_count > 0:
                status = "WARN"
        else:
            if abs(delta_pp) >= CAT_ALERT_PP:
                status = "ALERT"
            elif abs(delta_pp) >= CAT_WARN_PP:
                status = "WARN"

        rows.append(
            {
                "rule": rule,
                "reference_count": ref_count,
                "current_count": cur_count,
                "reference_rate": ref_rate,
                "current_rate": cur_rate,
                "delta_pp": delta_pp,
                "status": status,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["status", "current_count"],
        ascending=[True, False],
    )


# ============================================================
# Alerts and reports
# ============================================================

def build_alerts(
    ref_schema: dict[str, Any],
    cur_schema: dict[str, Any],
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    reference_label: str,
    current_label: str,
) -> dict[str, Any]:
    alerts = []
    warnings = []

    if not ref_schema["ok"]:
        alerts.append(f"Schema invalido na referencia: {reference_label}")

    if not cur_schema["ok"]:
        alerts.append(f"Schema invalido no current: {current_label}")

    for _, r in num_df.iterrows():
        if r["status"] == "ALERT":
            alerts.append(f"PSI ALERT em {r['feature']}: {r['psi']}")
        elif r["status"] == "WARN":
            warnings.append(f"PSI WARN em {r['feature']}: {r['psi']}")

    for _, r in cat_df.iterrows():
        if r["status"] == "ALERT":
            alerts.append(f"Categorico ALERT {r['feature']}={r['category']}: delta_pp={r['delta_pp']:.4f}")
        elif r["status"] == "WARN":
            warnings.append(f"Categorico WARN {r['feature']}={r['category']}: delta_pp={r['delta_pp']:.4f}")

    for _, r in decision_df.iterrows():
        if r["status"] == "ALERT":
            alerts.append(f"Decisao ALERT {r['decisao']}: delta_pp={r['delta_pp']:.4f}")
        elif r["status"] == "WARN":
            warnings.append(f"Decisao WARN {r['decisao']}: delta_pp={r['delta_pp']:.4f}")

    for _, r in rule_df.iterrows():
        if r["status"] == "ALERT":
            alerts.append(f"Regra ALERT {r['rule']}: ref={r['reference_count']} current={r['current_count']}")
        elif r["status"] == "WARN":
            warnings.append(f"Regra WARN {r['rule']}: ref={r['reference_count']} current={r['current_count']}")

    if alerts:
        status = "ALERT"
    elif warnings:
        status = "WARN"
    else:
        status = "OK"

    return {
        "status": status,
        "reference_label": reference_label,
        "current_label": current_label,
        "alerts": alerts,
        "warnings": warnings,
        "n_alerts": len(alerts),
        "n_warnings": len(warnings),
    }


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.4f}%"


def fmt_float(value: Any, ndigits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{ndigits}f}"


def write_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    ref_schema: dict[str, Any],
    cur_schema: dict[str, Any],
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    alerts: dict[str, Any],
    reference_label: str,
    current_label: str,
    reference_path: Path,
    current_path: Path,
) -> None:
    lines: list[str] = [
        "# EXP-009B — Drift Monitor Offline",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Objetivo",
        "",
        "Monitorar drift offline do baseline `post_fase2_c1` usando logs estruturados de decisão.",
        "",
        "## Entrada",
        "",
        f"- Referência: `{reference_label}` — `{reference_path}`",
        f"- Current: `{current_label}` — `{current_path}`",
        f"- Linhas referência: `{len(reference)}`",
        f"- Linhas current: `{len(current)}`",
        "",
        "> Observação: no modo default, este experimento compara seed 42 contra seed 123 como self-test do monitor. Isso valida o harness, mas não representa drift real de produção.",
        "",
        "## Status geral",
        "",
        f"**Status:** `{alerts['status']}`",
        "",
        f"- Alertas: `{alerts['n_alerts']}`",
        f"- Warnings: `{alerts['n_warnings']}`",
        "",
        "## Schema",
        "",
        "| Dataset | OK | Linhas | Missing fields | Decision IDs duplicados |",
        "|---|---:|---:|---|---:|",
        f"| `{reference_label}` | `{ref_schema['ok']}` | {ref_schema['n_rows']} | `{ref_schema['missing_fields']}` | {ref_schema['duplicate_decision_ids']} |",
        f"| `{current_label}` | `{cur_schema['ok']}` | {cur_schema['n_rows']} | `{cur_schema['missing_fields']}` | {cur_schema['duplicate_decision_ids']} |",
        "",
        "## Drift numérico por PSI",
        "",
        "| Feature | Status | PSI | Ref mean | Cur mean | Ref median | Cur median |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for _, r in num_df.iterrows():
        lines.append(
            f"| `{r['feature']}` | `{r['status']}` | {fmt_float(r['psi'])} | "
            f"{fmt_float(r['reference_mean'], 4)} | {fmt_float(r['current_mean'], 4)} | "
            f"{fmt_float(r['reference_median'], 4)} | {fmt_float(r['current_median'], 4)} |"
        )

    lines.extend([
        "",
        "## Distribuição de decisões",
        "",
        "| Decisão | Ref count | Cur count | Ref rate | Cur rate | Delta p.p. | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])

    for _, r in decision_df.iterrows():
        lines.append(
            f"| `{r['decisao']}` | {int(r['reference_count'])} | {int(r['current_count'])} | "
            f"{fmt_pct(r['reference_rate'])} | {fmt_pct(r['current_rate'])} | "
            f"{float(r['delta_pp']):.4f} | `{r['status']}` |"
        )

    lines.extend([
        "",
        "## Distribuição de regras",
        "",
        "| Regra | Ref count | Cur count | Ref rate | Cur rate | Delta p.p. | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])

    if rule_df.empty:
        lines.append("| `(nenhuma)` | 0 | 0 | 0 | 0 | 0 | `OK` |")
    else:
        for _, r in rule_df.iterrows():
            lines.append(
                f"| `{r['rule']}` | {int(r['reference_count'])} | {int(r['current_count'])} | "
                f"{fmt_pct(r['reference_rate'])} | {fmt_pct(r['current_rate'])} | "
                f"{float(r['delta_pp']):.4f} | `{r['status']}` |"
            )

    lines.extend([
        "",
        "## Alertas",
        "",
    ])

    if alerts["alerts"]:
        for item in alerts["alerts"]:
            lines.append(f"- ALERT: {item}")
    else:
        lines.append("- Nenhum alerta.")

    lines.extend([
        "",
        "## Warnings",
        "",
    ])

    if alerts["warnings"]:
        for item in alerts["warnings"]:
            lines.append(f"- WARN: {item}")
    else:
        lines.append("- Nenhum warning.")

    lines.extend([
        "",
        "## Decisão",
        "",
    ])

    if alerts["status"] == "OK":
        lines.append("EXP-009B aprovado: o monitor offline foi criado e o self-test não encontrou drift material entre os seeds.")
        lines.append("")
        lines.append("Próximo passo recomendado: EXP-009C — Zona Cinza e Fila de Revisão Humana.")
    elif alerts["status"] == "WARN":
        lines.append("EXP-009B aprovado com observações: o monitor foi criado, mas há warnings que devem ser acompanhados.")
        lines.append("")
        lines.append("Próximo passo recomendado: revisar warnings e depois seguir para EXP-009C.")
    else:
        lines.append("EXP-009B não deve ser aprovado sem análise: há alertas de drift/schema que exigem investigação.")

    (OUTPUT_DIR / "06_drift_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_experiment(alerts: dict[str, Any]) -> None:
    if alerts["status"] == "ALERT":
        title = "EXP-009B-AUDIT — Drift Alert Audit"
        objective = "Investigar alertas antes de criar fila de revisão humana."
        actions = [
            "Inspecionar features com PSI ALERT.",
            "Verificar se o drift decorre de bug no log ou diferença real.",
            "Validar distribuicao de decisao e regras.",
            "Nao alterar modelo ate concluir a auditoria.",
        ]
    else:
        title = "EXP-009C — Zona Cinza e Fila de Revisão Humana"
        objective = "Criar uma fila offline de casos informativos para revisao humana futura."
        actions = [
            "Selecionar APROVAR em zona cinza.",
            "Priorizar score_final perto de 62.",
            "Priorizar discordancia entre LGBM, IF, SE e BEH.",
            "Incluir C1 quase acionada e V1 quase acionada.",
            "Gerar review_queue.csv com prioridade e motivo.",
        ]

    lines = [
        "# Próximo experimento recomendado",
        "",
        f"## {title}",
        "",
        "## Objetivo",
        "",
        objective,
        "",
        "## Ações",
        "",
    ]

    for action in actions:
        lines.append(f"- {action}")

    (OUTPUT_DIR / "07_next_experiment_spec.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-009B Drift Monitor Offline")
    parser.add_argument("--reference-log", type=str, default=str(DEFAULT_REF_LOG))
    parser.add_argument("--current-log", type=str, default=str(DEFAULT_CUR_LOG))
    parser.add_argument("--label-reference", type=str, default="seed_42_reference")
    parser.add_argument("--label-current", type=str, default="seed_123_current")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference_log).resolve()
    current_path = Path(args.current_log).resolve()

    print("=" * 72)
    print("EXP-009B — Drift Monitor Offline")
    print("=" * 72)

    print("[1/6] Carregando logs...")
    reference = normalize_log(read_log(reference_path))
    current = normalize_log(read_log(current_path))

    print(f"[OK] Referencia: {len(reference)} linhas")
    print(f"[OK] Current: {len(current)} linhas")

    print("[2/6] Validando schema...")
    ref_schema = validate_trace_schema(reference, args.label_reference)
    cur_schema = validate_trace_schema(current, args.label_current)

    print(f"[OK] Schema referencia: {ref_schema['ok']}")
    print(f"[OK] Schema current: {cur_schema['ok']}")

    print("[3/6] Calculando drift numerico...")
    num_df = numeric_drift(reference, current)
    num_df.to_csv(OUTPUT_DIR / "01_numeric_drift.csv", index=False, encoding="utf-8-sig")

    print("[4/6] Calculando drift categorico e distribuicoes...")
    cat_df = categorical_drift(reference, current)
    cat_df.to_csv(OUTPUT_DIR / "02_categorical_drift.csv", index=False, encoding="utf-8-sig")

    decision_df = decision_distribution(reference, current)
    decision_df.to_csv(OUTPUT_DIR / "03_decision_distribution.csv", index=False, encoding="utf-8-sig")

    rule_df = rule_distribution(reference, current)
    rule_df.to_csv(OUTPUT_DIR / "04_rule_distribution.csv", index=False, encoding="utf-8-sig")

    print("[5/6] Gerando alertas...")
    alerts = build_alerts(
        ref_schema=ref_schema,
        cur_schema=cur_schema,
        num_df=num_df,
        cat_df=cat_df,
        decision_df=decision_df,
        rule_df=rule_df,
        reference_label=args.label_reference,
        current_label=args.label_current,
    )

    input_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "default_self_test" if reference_path == DEFAULT_REF_LOG.resolve() and current_path == DEFAULT_CUR_LOG.resolve() else "custom",
        "reference_label": args.label_reference,
        "current_label": args.label_current,
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "manifest_path": str(MANIFEST_PATH),
        "manifest": read_json(MANIFEST_PATH, {}),
        "schema_reference": ref_schema,
        "schema_current": cur_schema,
    }

    write_json(OUTPUT_DIR / "00_input_summary.json", input_summary)
    write_json(OUTPUT_DIR / "05_alerts.json", alerts)

    print(f"[OK] Status geral: {alerts['status']}")

    print("[6/6] Escrevendo relatorios...")
    write_report(
        reference=reference,
        current=current,
        ref_schema=ref_schema,
        cur_schema=cur_schema,
        num_df=num_df,
        cat_df=cat_df,
        decision_df=decision_df,
        rule_df=rule_df,
        alerts=alerts,
        reference_label=args.label_reference,
        current_label=args.label_current,
        reference_path=reference_path,
        current_path=current_path,
    )

    write_next_experiment(alerts)

    print()
    print("[OK] EXP-009B concluido.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '00_input_summary.json'}")
    print(f"  {OUTPUT_DIR / '01_numeric_drift.csv'}")
    print(f"  {OUTPUT_DIR / '02_categorical_drift.csv'}")
    print(f"  {OUTPUT_DIR / '03_decision_distribution.csv'}")
    print(f"  {OUTPUT_DIR / '04_rule_distribution.csv'}")
    print(f"  {OUTPUT_DIR / '05_alerts.json'}")
    print(f"  {OUTPUT_DIR / '06_drift_report.md'}")
    print(f"  {OUTPUT_DIR / '07_next_experiment_spec.md'}")


if __name__ == "__main__":
    main()