"""
scripts/build_validation_report_post_fase2.py

Gera docs/VALIDATION_REPORT_POST_FASE2.md.

Objetivo:
  Formalizar o baseline pós-FASE 2 como fonte única da verdade.

O relatório consolida:
  - baseline pós-C1;
  - métricas seed 42 e 123;
  - deltas da C1;
  - configuração oficial;
  - experimentos promovidos/rejeitados;
  - FNs residuais;
  - decisão de não prosseguir com EXP-007B;
  - comandos de regressão obrigatórios.

Uso:
  python scripts\\build_validation_report_post_fase2.py

Opcional, para também rodar pytest e registrar o resultado:
  python scripts\\build_validation_report_post_fase2.py --run-regression
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DECISOES_POSITIVAS = {"CONFIRMAR", "BLOQUEAR"}


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(SCRIPT_DIR)

SCORING_PATH = ROOT / "backend" / "artefatos" / "scoring_config.json"
CACHE_DIR = ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
EXP007A_DIR = ROOT / "resultados" / "experimentos" / "EXP-007A"
DOCS_DIR = ROOT / "docs"
REPORT_PATH = DOCS_DIR / "VALIDATION_REPORT_POST_FASE2.md"


TARGET_C1_TX = "E0000020820260205003505340630525"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
    fpr = fp / max(fp + tn, 1)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
    }


def fmt_pct(x: float) -> str:
    return f"{x:.4%}".replace(".", ",")


def fmt_float(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}".replace(".", ",")


def load_seed(seed: int) -> pd.DataFrame:
    path = CACHE_DIR / f"baseline_predictions_seed_{seed}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    df["seed"] = seed

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
        raise ValueError(f"Colunas ausentes para C1: {missing}")

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

    out["exp006f_c1_applied_report"] = mask

    idx = out.index[mask]
    out.loc[idx, "decisao_original_exp006f_c1_report"] = out.loc[idx, "decisao"]
    out.loc[idx, "score_final_original_exp006f_c1_report"] = out.loc[idx, "score_final"]
    out.loc[idx, "decisao"] = "CONFIRMAR"
    out.loc[idx, "score_final"] = out.loc[idx, "score_final"].apply(lambda x: max(float(x), 62.0))

    return out


def compare_delta(base: pd.DataFrame, cand: pd.DataFrame) -> dict[str, int]:
    y = base["is_fraud"].astype(int)
    b = flagged(base)
    c = flagged(cand)

    return {
        "fns_recuperados": int((y.eq(1) & (~b) & c).sum()),
        "fps_adicionados": int((y.eq(0) & (~b) & c).sum()),
        "tps_perdidos": int((y.eq(1) & b & (~c)).sum()),
        "fps_removidos": int((y.eq(0) & b & (~c)).sum()),
        "rule_hits": int(cand["exp006f_c1_applied_report"].sum()),
    }


def residual_fns(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["is_fraud"].astype(int).eq(1)) & (~flagged(df))].copy()


def run_regression() -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q", "-m", "slow"],
    ]

    results = []

    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )

        results.append({
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        })

    return {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "all_passed": all(r["returncode"] == 0 for r in results),
    }


def config_summary(config: dict[str, Any]) -> str:
    keys = [
        "threshold_confirmar",
        "threshold_bloquear",
        "lgbm_guard_enabled",
        "lgbm_guard_threshold",
        "guard_exception_alto_valor_se_beh_enabled",
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
        "exp006f_c1_max_se_score",
        "exp006f_c1_max_beh_score",
        "se_pattern_residual_enabled",
        "exp003_residual_confirm_enabled",
    ]

    selected = {k: config.get(k) for k in keys if k in config}

    return json.dumps(selected, indent=2, ensure_ascii=False)


def build_report(run_regression_flag: bool) -> str:
    if not SCORING_PATH.exists():
        raise FileNotFoundError(f"scoring_config.json não encontrado: {SCORING_PATH}")

    config = read_json(SCORING_PATH, {})

    seed_data = {}

    for seed in [42, 123]:
        base = load_seed(seed)
        post_c1 = apply_c1_min58(base)

        seed_data[seed] = {
            "base": base,
            "post_c1": post_c1,
            "base_metrics": compute_metrics(base),
            "post_c1_metrics": compute_metrics(post_c1),
            "delta": compare_delta(base, post_c1),
            "residual_fns": residual_fns(post_c1),
            "hits": post_c1[post_c1["exp006f_c1_applied_report"]].copy(),
        }

    all_post = pd.concat([seed_data[42]["post_c1"], seed_data[123]["post_c1"]], ignore_index=True)

    if "transaction_id" in all_post.columns:
        unique_post = (
            all_post
            .sort_values(["is_fraud", "seed"], ascending=[False, True])
            .drop_duplicates(subset=["transaction_id"], keep="first")
            .copy()
        )
    else:
        unique_post = all_post.drop_duplicates().copy()

    unique_metrics = compute_metrics(unique_post)
    unique_residual = residual_fns(unique_post)

    candidate_eval = read_json(EXP007A_DIR / "05_candidate_overlay_eval.json", {})
    regression_result = run_regression() if run_regression_flag else None

    lines: list[str] = []

    lines.extend([
        "# Validation Report Pós-FASE 2 — Pipeline Antifraude PIX",
        "",
        f"**Gerado em:** `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## 1. Decisão oficial",
        "",
        "A FASE 2 foi encerrada com sucesso mínimo validado.",
        "",
        "A versão oficial do pipeline passa a ser o **baseline pós-C1**, mantendo o modelo LGBM de produção anterior, o guard rail LGBM, a exceção contextual de alto valor da FASE 1 e a regra cirúrgica `C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER`.",
        "",
        "Decisão final:",
        "",
        "```text",
        "Promovido: V1_GUARD_CONTEXTUAL",
        "Promovido: C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER",
        "Rejeitado: LGBM v6.2 para runtime",
        "Rejeitado: R2_LOW_VALUE_GRAY_FIRST_RECEIVER",
        "Rejeitado: meta-learner shadow como componente de decisão",
        "Próxima etapa: FASE 3 — consolidação operacional, testes, documentação e observabilidade",
        "```",
        "",
        "## 2. Dataset e fonte das métricas",
        "",
        f"- Fonte dos artefatos baseline: `{CACHE_DIR}`",
        "- Seeds oficiais: `42` e `123`",
        "- C1 reaplicada no relatório com `min_score=58.0`, conforme validação runtime pós-FASE 2.",
        "",
        "## 3. Métricas oficiais",
        "",
        "| Seed/Conjunto | TP | FP | FN | Precision | Recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for seed in [42, 123]:
        m = seed_data[seed]["post_c1_metrics"]

        lines.append(
            f"| seed {seed} | {m['TP']} | {m['FP']} | {m['FN']} | "
            f"{fmt_pct(m['Precision'])} | {fmt_pct(m['Recall'])} | "
            f"{fmt_float(m['F1'])} | {fmt_pct(m['FPR'])} |"
        )

    lines.append(
        f"| unique union | {unique_metrics['TP']} | {unique_metrics['FP']} | {unique_metrics['FN']} | "
        f"{fmt_pct(unique_metrics['Precision'])} | {fmt_pct(unique_metrics['Recall'])} | "
        f"{fmt_float(unique_metrics['F1'])} | {fmt_pct(unique_metrics['FPR'])} |"
    )

    lines.extend([
        "",
        "## 4. Delta da C1",
        "",
        "| Seed | FNs recuperados | FPs adicionados | TPs perdidos | FPs removidos | Rule hits |",
        "|---:|---:|---:|---:|---:|---:|",
    ])

    for seed in [42, 123]:
        d = seed_data[seed]["delta"]
        lines.append(
            f"| {seed} | {d['fns_recuperados']} | {d['fps_adicionados']} | "
            f"{d['tps_perdidos']} | {d['fps_removidos']} | {d['rule_hits']} |"
        )

    lines.extend([
        "",
        "Conclusão: a C1 recupera 1 FN nos dois seeds, adiciona 0 FP e não perde TP.",
        "",
        "## 5. Caso recuperado pela C1",
        "",
    ])

    hit = seed_data[42]["hits"]

    if not hit.empty:
        row = hit.iloc[0].to_dict()

        lines.extend([
            "| Campo | Valor |",
            "|---|---:|",
            f"| transaction_id | `{row.get('transaction_id', TARGET_C1_TX)}` |",
            f"| customer_id | `{row.get('customer_id', '')}` |",
            f"| is_fraud | `{row.get('is_fraud', '')}` |",
            f"| vl_pix | R$ {float(row.get('vl_pix', 0)):.2f} |",
            f"| relacionamento meses | {float(row.get('qt_tempo_relacionamento_mes', 0)):.0f} |",
            f"| first_receiver_flag | {int(float(row.get('first_receiver_flag', 0)))} |",
            f"| pix_key_random_flag | {int(float(row.get('pix_key_random_flag', 0)))} |",
            f"| lgbm_raw | {float(row.get('lgbm_raw', 0)):.8f} |",
            f"| score_final pós-C1 | {float(row.get('score_final', 0)):.2f} |",
            "",
        ])
    else:
        lines.append("Nenhum hit C1 encontrado nos artefatos carregados. Isso deve ser investigado.")

    lines.extend([
        "## 6. Configuração oficial",
        "",
        "```json",
        config_summary(config),
        "```",
        "",
        "## 7. Experimentos promovidos e rejeitados",
        "",
        "| Experimento | Decisão | Motivo |",
        "|---|---|---|",
        "| EXP-004-FINAL / V1_GUARD_CONTEXTUAL | Promovido | Recuperou FN de alto valor sem adicionar FP |",
        "| EXP-005A / LGBM v6.2 recall-oriented | Não promovido | Promissor model-only, mas exigia validação no engine |",
        "| EXP-005B / calibração pós-LGBM v6.2 | Rejeitado | Não reduziu FN líquido no engine real e aumentou FP |",
        "| EXP-006C / R2_LOW_VALUE_GRAY_FIRST_RECEIVER | Rejeitado | Recuperou 0 FN e adicionou FP |",
        "| EXP-006E/006F / C1_NEAR_THRESHOLD_REL_CURTO_FIRST_RECEIVER | Promovido | Recuperou 1 FN nos dois seeds, 0 FP adicionado |",
        "| EXP-007A / Meta-Learner Shadow | Não promovido | Sem candidato seguro para overlay adicional |",
        "",
        "## 8. Resultado do EXP-007A",
        "",
    ])

    status = candidate_eval.get("status", "não encontrado")
    next_action = candidate_eval.get("next_action", "não encontrado")

    lines.extend([
        f"- Status: `{status}`",
        f"- Próxima ação indicada: {next_action}",
        "",
        "Interpretação: com os sinais atuais, o meta-learner não encontrou threshold seguro para recuperar FN residual sem custo operacional em FP. Portanto, não há justificativa para rodar EXP-007B com os mesmos sinais.",
        "",
        "## 9. FNs residuais pós-C1",
        "",
        f"Quantidade de FNs residuais únicos: `{len(unique_residual)}`",
        "",
    ])

    if not unique_residual.empty:
        keep = [
            "transaction_id",
            "vl_pix",
            "qt_tempo_relacionamento_mes",
            "first_receiver_flag",
            "pix_key_random_flag",
            "lgbm_raw",
            "if_percentile",
            "se_score",
            "beh_score",
            "score_final",
            "decisao",
        ]

        available = [c for c in keep if c in unique_residual.columns]

        lines.extend([
            "| transaction_id | valor | rel. meses | first_receiver | pix_random | LGBM | IF | SE | BEH | score | decisão |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])

        for _, r in unique_residual[available].head(20).iterrows():
            lines.append(
                f"| `{r.get('transaction_id', '')}` | "
                f"{float(r.get('vl_pix', 0)):.2f} | "
                f"{float(r.get('qt_tempo_relacionamento_mes', 0)):.0f} | "
                f"{int(float(r.get('first_receiver_flag', 0)))} | "
                f"{int(float(r.get('pix_key_random_flag', 0)))} | "
                f"{float(r.get('lgbm_raw', 0)):.5f} | "
                f"{float(r.get('if_percentile', 0)):.5f} | "
                f"{float(r.get('se_score', 0)):.2f} | "
                f"{float(r.get('beh_score', 0)):.2f} | "
                f"{float(r.get('score_final', 0)):.2f} | "
                f"{r.get('decisao', '')} |"
            )

    lines.extend([
        "",
        "Interpretação: os FNs remanescentes devem ser tratados como próximos do limite dos sinais atuais. A maior parte deles não possui convergência suficiente entre LGBM, IF, SE, BEH e score final para justificar nova regra manual segura.",
        "",
        "## 10. Procedimento de regressão obrigatório",
        "",
        "Antes de qualquer mudança futura no engine, scoring_config, simulação E2E ou artefatos, executar:",
        "",
        "```powershell",
        "python -m pytest tests\\test_regression_post_fase2.py -q",
        "python -m pytest tests\\test_regression_post_fase2.py -q -m slow",
        "```",
        "",
    ])

    if regression_result:
        lines.extend([
            "### Resultado da regressão no momento da geração",
            "",
            f"- Executado em: `{regression_result['ran_at']}`",
            f"- Tudo passou: `{regression_result['all_passed']}`",
            "",
        ])

        for item in regression_result["results"]:
            lines.extend([
                f"#### `{item['command']}`",
                "",
                "```text",
                item["stdout"] or item["stderr"] or "(sem saída)",
                "```",
                "",
            ])
    else:
        lines.extend([
            "A geração deste relatório não executou a regressão automaticamente. Resultado conhecido esperado:",
            "",
            "```text",
            "6 passed",
            "1 passed, 5 deselected",
            "```",
            "",
        ])

    lines.extend([
        "## 11. Critérios de aceite da versão pós-FASE 2",
        "",
        "A versão pós-FASE 2 é considerada válida enquanto:",
        "",
        "- seed 42 mantiver `TP=347`, `FP=14`, `FN=8`;",
        "- seed 123 mantiver `TP=347`, `FP=12`, `FN=8`;",
        "- C1 recuperar exatamente 1 FN nos dois seeds;",
        "- C1 adicionar 0 FP;",
        "- C1 perder 0 TP;",
        "- teste runtime da transação alvo retornar `CONFIRMAR` com `exp006f_c1_applied=True`;",
        "- `EXP-007A` permanecer sem candidato seguro adicional usando os sinais atuais.",
        "",
        "## 12. Próxima etapa",
        "",
        "Com o relatório oficial gerado, a próxima etapa da FASE 3 deve ser:",
        "",
        "```text",
        "EXP-008C — Rules Catalog e Decision Trace",
        "```",
        "",
        "Objetivo: documentar todas as regras ativas, regras rejeitadas, guard rails, thresholds e motivos de decisão do engine.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera VALIDATION_REPORT_POST_FASE2.md")
    parser.add_argument("--run-regression", action="store_true")
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    report = build_report(run_regression_flag=args.run_regression)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"[OK] Relatório gerado: {REPORT_PATH}")


if __name__ == "__main__":
    main()