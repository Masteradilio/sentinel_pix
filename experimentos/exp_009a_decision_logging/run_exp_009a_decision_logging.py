"""
EXP-009A — Decision Logging Estruturado

Objetivo:
  Criar logs estruturados de decisão para o baseline pós-FASE 2 / pós-C1.

Este experimento:
  - Não altera scoring_config.json.
  - Não altera DecisionEngine.
  - Não altera PipelineOrquestrador.
  - Não roda E2E pesado.
  - Usa os artefatos cacheados do EXP-006C-R2.
  - Reaplica C1 min_score=58 para representar o baseline oficial pós-C1.
  - Gera logs no formato definido por docs/DECISION_TRACE_SPEC.md.
  - Valida campos mínimos de rastreabilidade.
  - Gera sumários de decisões, regras e exemplos.

Uso:
  python experimentos\\exp_009a_decision_logging\\run_exp_009a_decision_logging.py

Opcional, para também testar 1 transação runtime real:
  python experimentos\\exp_009a_decision_logging\\run_exp_009a_decision_logging.py --runtime-target
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DECISOES_POSITIVAS = {"CONFIRMAR", "BLOQUEAR"}
TARGET_C1_TX = "E0000020820260205003505340630525"


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists() and (p / "resultados").exists():
            return p
    return start.parent.parent


EXP_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(EXP_DIR)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
sys.path.insert(0, str(ROOT / "backend" / "core"))

INPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
OUTPUT_DIR = ROOT / "resultados" / "experimentos" / "EXP-009A"

SCORING_PATH = ROOT / "backend" / "artefatos" / "scoring_config.json"
MANIFEST_PATH = ROOT / "backend" / "artefatos" / "MANIFEST_MODEL.json"
TRACE_SPEC_PATH = ROOT / "docs" / "DECISION_TRACE_SPEC.md"


REQUIRED_TRACE_FIELDS = [
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


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
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


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def stable_hash(value: Any, n: int = 16) -> str:
    raw = str(value).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:n]


def customer_hash(customer_id: Any) -> str:
    return "cust_" + stable_hash(f"antifraude_pix_v1::{customer_id}", n=24)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def as_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(DECISOES_POSITIVAS)


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    y = df["is_fraud"].astype(int)
    p = flagged(df).astype(int)

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
    }


def load_seed(seed: int) -> pd.DataFrame:
    path = INPUT_DIR / f"baseline_predictions_seed_{seed}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    df["seed"] = seed
    df["source_file"] = str(path)

    return df


def apply_c1_min58(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "decisao",
        "is_fraud",
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "score_final",
        "se_score",
        "beh_score",
    ]

    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes para aplicar C1: {missing}")

    for c in [
        "vl_pix",
        "qt_tempo_relacionamento_mes",
        "first_receiver_flag",
        "pix_key_random_flag",
        "lgbm_raw",
        "score_final",
        "se_score",
        "beh_score",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    mask = (
        out["decisao"].astype(str).eq("APROVAR")
        & out["first_receiver_flag"].astype(int).eq(1)
        & out["pix_key_random_flag"].astype(int).eq(0)
        & out["qt_tempo_relacionamento_mes"].le(12)
        & out["vl_pix"].ge(100.0)
        & out["vl_pix"].lt(500.0)
        & out["lgbm_raw"].ge(0.06)
        & out["lgbm_raw"].lt(0.10)
        & out["score_final"].ge(58.0)
        & out["score_final"].lt(62.0)
        & out["se_score"].le(0.0)
        & out["beh_score"].le(0.0)
    )

    out["exp006f_c1_applied"] = mask

    idx = out.index[mask]
    out.loc[idx, "decisao_original_exp006f_c1"] = out.loc[idx, "decisao"]
    out.loc[idx, "score_final_original_exp006f_c1"] = out.loc[idx, "score_final"]
    out.loc[idx, "decisao"] = "CONFIRMAR"
    out.loc[idx, "score_final"] = out.loc[idx, "score_final"].apply(lambda x: max(float(x), 62.0))
    out.loc[idx, "exp006f_c1_reason"] = (
        "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER: APROVAR->CONFIRMAR | "
        "rel<=12, first_receiver=1, pix_random=0, 100<=vl<500, "
        "0.06<=lgbm<0.10, 58<=score<62, SE=0, BEH=0"
    )

    return out


def infer_rules_applied(row: pd.Series, config: dict[str, Any]) -> list[str]:
    rules: list[str] = []

    decisao = as_str(row.get("decisao")).upper()
    score_final = as_float(row.get("score_final"))
    threshold_confirmar = as_float(config.get("threshold_confirmar"), 62.0)
    threshold_bloquear = as_float(config.get("threshold_bloquear"), 95.0)

    if bool(row.get("exp006f_c1_applied", False)):
        rules.append("C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER")

    if bool(row.get("v1_guard_contextual_applied", False)) or bool(row.get("guard_exception_alto_valor_applied", False)):
        rules.append("V1_GUARD_CONTEXTUAL")

    if decisao == "BLOQUEAR" and score_final >= threshold_bloquear:
        rules.append("BASE_SCORE_THRESHOLD_BLOQUEAR")
    elif decisao == "CONFIRMAR" and score_final >= threshold_confirmar:
        if "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER" not in rules:
            rules.append("BASE_SCORE_THRESHOLD_CONFIRMAR")

    if as_float(row.get("se_score")) > 0:
        rules.append("SE_RULE_SIGNAL")

    if as_float(row.get("beh_score")) > 0:
        rules.append("BEH_RULE_SIGNAL")

    # Remove duplicatas preservando ordem.
    deduped: list[str] = []
    for item in rules:
        if item not in deduped:
            deduped.append(item)

    return deduped


def infer_guardrails_applied(row: pd.Series, config: dict[str, Any]) -> list[str]:
    guardrails: list[str] = []

    if as_str(row.get("veto_reason")):
        guardrails.append("VETO_REASON_PRESENT")

    if as_str(row.get("veto_suppressed_reason")):
        guardrails.append("VETO_SUPPRESSED_REASON_PRESENT")

    if bool(row.get("lgbm_guard_applied", False)):
        guardrails.append("LGBM_GUARD_RAIL_APPLIED")

    # Heurística conservadora: não marcar guard rail apenas por lgbm_raw baixo.
    # Só registrar quando há coluna explícita de veto/guard.

    return guardrails


def decision_reason(row: pd.Series, rules: list[str], guardrails: list[str]) -> str:
    decisao = as_str(row.get("decisao"), "UNKNOWN")
    score_final = as_float(row.get("score_final"))
    lgbm_raw = as_float(row.get("lgbm_raw"))

    if bool(row.get("exp006f_c1_applied", False)):
        return as_str(
            row.get("exp006f_c1_reason"),
            (
                "C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER aplicou APROVAR->CONFIRMAR "
                "em caso near-threshold pós-FASE 2."
            ),
        )

    if rules:
        return (
            f"Decisão {decisao} sustentada por regras={rules}; "
            f"score_final={score_final:.2f}; lgbm_raw={lgbm_raw:.6f}."
        )

    if guardrails:
        return (
            f"Decisão {decisao} com guardrails={guardrails}; "
            f"score_final={score_final:.2f}; lgbm_raw={lgbm_raw:.6f}."
        )

    return (
        f"Decisão {decisao} pelo fluxo base do engine; "
        f"score_final={score_final:.2f}; lgbm_raw={lgbm_raw:.6f}."
    )


def review_recommended(row: pd.Series) -> bool:
    decisao = as_str(row.get("decisao")).upper()

    if decisao in {"CONFIRMAR", "BLOQUEAR"}:
        return True

    score_final = as_float(row.get("score_final"))
    lgbm_raw = as_float(row.get("lgbm_raw"))
    if_percentile = as_float(row.get("if_percentile"))

    # Zona cinza para futura fila de revisão.
    if 55 <= score_final < 62:
        return True

    if 0.05 <= lgbm_raw < 0.20:
        return True

    if if_percentile >= 0.95:
        return True

    return False


def build_decision_log(
    df: pd.DataFrame,
    *,
    seed: int,
    manifest: dict[str, Any],
    config: dict[str, Any],
    scoring_hash: str,
) -> pd.DataFrame:
    model_version = manifest.get("model_version", "post_fase2_c1")
    engine_version = manifest.get("decision_engine_version", "v3.0.5_post_c1_exp008d")
    scoring_version = scoring_hash[:16] if scoring_hash else "unknown"

    rows: list[dict[str, Any]] = []

    created_at = datetime.now().isoformat(timespec="seconds")

    for idx, row in df.reset_index(drop=True).iterrows():
        tx_id = as_str(row.get("transaction_id"), f"missing_tx_{seed}_{idx}")
        customer_id = as_str(row.get("customer_id"), "missing_customer")

        rules = infer_rules_applied(row, config)
        guardrails = infer_guardrails_applied(row, config)
        reason = decision_reason(row, rules, guardrails)

        decision_id_raw = f"{model_version}|{scoring_version}|seed={seed}|{tx_id}|{idx}"
        decision_id = "dec_" + stable_hash(decision_id_raw, n=24)

        trace = {
            "decision_id": decision_id,
            "transaction_id": tx_id,
            "customer_id_hash": customer_hash(customer_id),
            "created_at": created_at,
            "model_version": model_version,
            "decision_engine_version": engine_version,
            "scoring_config_version": scoring_version,
            "seed": seed,
            "decisao": as_str(row.get("decisao")),
            "score_final": as_float(row.get("score_final")),
            "score_final_original": as_float(
                row.get("score_final_original_exp006f_c1"),
                default=as_float(row.get("score_final")),
            ),
            "lgbm_raw": as_float(row.get("lgbm_raw")),
            "lgbm_mapped": as_float(row.get("lgbm_mapped")),
            "if_percentile": as_float(row.get("if_percentile")),
            "se_score": as_float(row.get("se_score")),
            "beh_score": as_float(row.get("beh_score")),
            "rules_applied": json.dumps(rules, ensure_ascii=False),
            "guardrails_applied": json.dumps(guardrails, ensure_ascii=False),
            "veto_reason": as_str(row.get("veto_reason")),
            "veto_suppressed_reason": as_str(row.get("veto_suppressed_reason")),
            "decision_reason": reason,
            "review_recommended": bool(review_recommended(row)),
            "is_fraud": int(as_float(row.get("is_fraud"))),
            "vl_pix": as_float(row.get("vl_pix")),
            "qt_tempo_relacionamento_mes": as_float(row.get("qt_tempo_relacionamento_mes")),
            "first_receiver_flag": int(as_float(row.get("first_receiver_flag"))),
            "pix_key_random_flag": int(as_float(row.get("pix_key_random_flag"))),
            "exp006f_c1_applied": bool(row.get("exp006f_c1_applied", False)),
            "exp006f_c1_reason": as_str(row.get("exp006f_c1_reason")),
            "decisao_original_exp006f_c1": as_str(row.get("decisao_original_exp006f_c1")),
            "score_final_original_exp006f_c1": as_float(row.get("score_final_original_exp006f_c1")),
        }

        rows.append(trace)

    return pd.DataFrame(rows)


def validate_schema(log_df: pd.DataFrame) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_TRACE_FIELDS if field not in log_df.columns]

    null_counts = {}
    for field in REQUIRED_TRACE_FIELDS:
        if field in log_df.columns:
            null_counts[field] = int(log_df[field].isna().sum())

    empty_string_counts = {}
    for field in REQUIRED_TRACE_FIELDS:
        if field in log_df.columns and log_df[field].dtype == object:
            empty_string_counts[field] = int(log_df[field].astype(str).str.strip().eq("").sum())

    required_empty = {
        field: empty_string_counts.get(field, 0)
        for field in [
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
    }

    invalid_decision = []
    if "decisao" in log_df.columns:
        invalid_decision = sorted(
            set(log_df["decisao"].astype(str)) - {"APROVAR", "CONFIRMAR", "BLOQUEAR"}
        )

    duplicate_decision_ids = int(log_df["decision_id"].duplicated().sum()) if "decision_id" in log_df.columns else None

    ok = (
        not missing_fields
        and all(v == 0 for v in required_empty.values())
        and not invalid_decision
        and duplicate_decision_ids == 0
    )

    return {
        "ok": ok,
        "n_rows": int(len(log_df)),
        "missing_fields": missing_fields,
        "null_counts": null_counts,
        "empty_string_counts_required": required_empty,
        "invalid_decision_values": invalid_decision,
        "duplicate_decision_ids": duplicate_decision_ids,
    }


def explode_json_list_column(df: pd.DataFrame, col: str) -> pd.Series:
    values: list[str] = []

    if col not in df.columns:
        return pd.Series(dtype="object")

    for raw in df[col].fillna("[]").astype(str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                values.extend(str(x) for x in parsed)
        except Exception:
            continue

    return pd.Series(values, dtype="object")


def write_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(safe_json(record), ensure_ascii=False) + "\n")


def run_runtime_target_validation(manifest: dict[str, Any], config: dict[str, Any], scoring_hash: str) -> dict[str, Any]:
    from experimentos.utils_experimentos import (
        get_logger,
        load_dataset,
        process_dataframe_via_orquestrador,
    )

    logger = get_logger("EXP-009A-RUNTIME-TARGET")

    df = load_dataset()

    if "transaction_id" not in df.columns:
        raise RuntimeError("Dataset não possui transaction_id.")

    sample = df[df["transaction_id"].astype(str).eq(TARGET_C1_TX)].copy()

    if sample.empty:
        raise RuntimeError(f"Transação alvo não encontrada: {TARGET_C1_TX}")

    preds = process_dataframe_via_orquestrador(
        sample,
        workers=1,
        logger=logger,
        engine_config_overrides=None,
    )

    log_df = build_decision_log(
        preds,
        seed=999,
        manifest=manifest,
        config=config,
        scoring_hash=scoring_hash,
    )

    path = OUTPUT_DIR / "09_runtime_target_decision_log.csv"
    log_df.to_csv(path, index=False, encoding="utf-8-sig")

    row = log_df.iloc[0].to_dict()

    return {
        "executed": True,
        "path": str(path),
        "transaction_id": row.get("transaction_id"),
        "decisao": row.get("decisao"),
        "exp006f_c1_applied": bool(row.get("exp006f_c1_applied", False)),
        "schema_validation": validate_schema(log_df),
    }


def write_recommendation(
    validation: dict[str, Any],
    summary_by_decision: pd.DataFrame,
    summary_by_rule: pd.DataFrame,
    runtime_result: dict[str, Any] | None,
) -> None:
    lines = [
        "# EXP-009A — Decision Logging Estruturado",
        "",
        f"Gerado em: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Objetivo",
        "",
        "Criar logs estruturados de decisão para o baseline oficial `post_fase2_c1`, seguindo a especificação mínima de `docs/DECISION_TRACE_SPEC.md`.",
        "",
        "## Validação de schema",
        "",
        f"- Status: `{validation['ok']}`",
        f"- Linhas: `{validation['n_rows']}`",
        f"- Campos ausentes: `{validation['missing_fields']}`",
        f"- Decision IDs duplicados: `{validation['duplicate_decision_ids']}`",
        f"- Valores inválidos de decisão: `{validation['invalid_decision_values']}`",
        "",
        "## Sumário por decisão",
        "",
    ]

    if not summary_by_decision.empty:
        lines.append("| Decisão | Linhas | Fraudes | Valor total | Review recommended |")
        lines.append("|---|---:|---:|---:|---:|")

        for _, r in summary_by_decision.iterrows():
            lines.append(
                f"| `{r['decisao']}` | {int(r['n_rows'])} | {int(r['n_frauds'])} | "
                f"{float(r['vl_pix_sum']):.2f} | {int(r['review_recommended_sum'])} |"
            )

    lines.extend([
        "",
        "## Sumário por regra",
        "",
    ])

    if not summary_by_rule.empty:
        lines.append("| Regra | Ocorrências |")
        lines.append("|---|---:|")

        for _, r in summary_by_rule.iterrows():
            lines.append(f"| `{r['rule']}` | {int(r['count'])} |")
    else:
        lines.append("Nenhuma regra inferida nos logs.")

    lines.extend([
        "",
        "## Validação runtime opcional",
        "",
    ])

    if runtime_result:
        lines.extend([
            f"- Executada: `{runtime_result.get('executed')}`",
            f"- Transação: `{runtime_result.get('transaction_id')}`",
            f"- Decisão: `{runtime_result.get('decisao')}`",
            f"- C1 aplicada: `{runtime_result.get('exp006f_c1_applied')}`",
            f"- Schema OK: `{runtime_result.get('schema_validation', {}).get('ok')}`",
            "",
        ])
    else:
        lines.append("Não executada. Rode com `--runtime-target` se quiser validar a transação C1 em runtime real.")

    lines.extend([
        "",
        "## Decisão",
        "",
    ])

    if validation["ok"]:
        lines.append("EXP-009A aprovado em modo offline: os logs estruturados possuem schema mínimo válido.")
        lines.append("")
        lines.append("Próximo passo recomendado: EXP-009B — Drift Monitor Offline.")
    else:
        lines.append("EXP-009A ainda não deve ser aprovado. Corrigir os problemas de schema antes de prosseguir.")

    (OUTPUT_DIR / "08_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-009A Decision Logging Estruturado")
    parser.add_argument("--runtime-target", action="store_true", help="Valida uma transação C1 em runtime real.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXP-009A — Decision Logging Estruturado")
    print("=" * 72)

    if not SCORING_PATH.exists():
        raise FileNotFoundError(f"scoring_config não encontrado: {SCORING_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"MANIFEST_MODEL não encontrado: {MANIFEST_PATH}")

    config = read_json(SCORING_PATH, {})
    manifest = read_json(MANIFEST_PATH, {})

    scoring_hash = sha256_file(SCORING_PATH) or "unknown"

    print("[1/6] Carregando seeds e aplicando C1 pós-FASE 2...")

    all_logs = []
    metrics_rows = []

    for seed in [42, 123]:
        base = load_seed(seed)
        post_c1 = apply_c1_min58(base)

        metrics = compute_metrics(post_c1)
        metrics["seed"] = seed
        metrics_rows.append(metrics)

        log_df = build_decision_log(
            post_c1,
            seed=seed,
            manifest=manifest,
            config=config,
            scoring_hash=scoring_hash,
        )

        out_path = OUTPUT_DIR / f"{1 if seed == 42 else 2:02d}_decision_log_seed_{seed}.csv"
        log_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[OK] Seed {seed}: {len(log_df)} logs -> {out_path}")

        all_logs.append(log_df)

    combined = pd.concat(all_logs, ignore_index=True)

    print("[2/6] Gravando JSONL combinado...")
    write_jsonl(OUTPUT_DIR / "03_decision_log_all.jsonl", combined)

    print("[3/6] Validando schema...")
    validation = validate_schema(combined)
    write_json(OUTPUT_DIR / "04_schema_validation.json", validation)

    print(f"[OK] Schema OK: {validation['ok']}")

    print("[4/6] Gerando sumários...")

    summary_by_decision = (
        combined
        .groupby("decisao", dropna=False)
        .agg(
            n_rows=("decision_id", "count"),
            n_frauds=("is_fraud", "sum"),
            vl_pix_sum=("vl_pix", "sum"),
            review_recommended_sum=("review_recommended", "sum"),
        )
        .reset_index()
        .sort_values("n_rows", ascending=False)
    )

    summary_by_decision.to_csv(
        OUTPUT_DIR / "05_summary_by_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rules_series = explode_json_list_column(combined, "rules_applied")

    if rules_series.empty:
        summary_by_rule = pd.DataFrame(columns=["rule", "count"])
    else:
        summary_by_rule = (
            rules_series
            .value_counts()
            .rename_axis("rule")
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    summary_by_rule.to_csv(
        OUTPUT_DIR / "06_summary_by_rule.csv",
        index=False,
        encoding="utf-8-sig",
    )

    examples = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "c1_examples": combined[combined["exp006f_c1_applied"]].head(5).to_dict(orient="records"),
        "confirmar_examples": combined[combined["decisao"].eq("CONFIRMAR")].head(5).to_dict(orient="records"),
        "bloquear_examples": combined[combined["decisao"].eq("BLOQUEAR")].head(5).to_dict(orient="records"),
        "aprovar_gray_examples": combined[
            combined["decisao"].eq("APROVAR") & combined["review_recommended"].astype(bool)
        ].head(10).to_dict(orient="records"),
        "metrics_by_seed": metrics_rows,
    }

    write_json(OUTPUT_DIR / "07_examples.json", examples)

    runtime_result = None

    if args.runtime_target:
        print("[5/6] Validando transação C1 em runtime real...")
        runtime_result = run_runtime_target_validation(manifest, config, scoring_hash)
        write_json(OUTPUT_DIR / "09_runtime_target_result.json", runtime_result)
        print(f"[OK] Runtime target: {runtime_result}")

    else:
        print("[5/6] Validação runtime não solicitada.")

    print("[6/6] Escrevendo recomendação...")
    write_recommendation(
        validation=validation,
        summary_by_decision=summary_by_decision,
        summary_by_rule=summary_by_rule,
        runtime_result=runtime_result,
    )

    print()
    print("[OK] EXP-009A concluído.")
    print(f"[OK] Artefatos em: {OUTPUT_DIR}")
    print()
    print("Arquivos principais:")
    print(f"  {OUTPUT_DIR / '01_decision_log_seed_42.csv'}")
    print(f"  {OUTPUT_DIR / '02_decision_log_seed_123.csv'}")
    print(f"  {OUTPUT_DIR / '03_decision_log_all.jsonl'}")
    print(f"  {OUTPUT_DIR / '04_schema_validation.json'}")
    print(f"  {OUTPUT_DIR / '05_summary_by_decision.csv'}")
    print(f"  {OUTPUT_DIR / '06_summary_by_rule.csv'}")
    print(f"  {OUTPUT_DIR / '07_examples.json'}")
    print(f"  {OUTPUT_DIR / '08_recommendation.md'}")


if __name__ == "__main__":
    main()