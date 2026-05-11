"""
tests/test_regression_post_fase2.py

Suíte de regressão pós-FASE 2.

Valida o baseline oficial pós-C1:
  seed 42:  TP=347, FP=14, FN=8
  seed 123: TP=347, FP=12, FN=8

Importante:
  Este teste evita importar módulos pesados de runtime diretamente no processo
  do pytest, porque alguns módulos do pipeline mexem com logging/stdout/stderr
  e podem quebrar a captura do pytest em Windows.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


DECISOES_POSITIVAS = {"CONFIRMAR", "BLOQUEAR"}
TARGET_C1_TX = "E0000020820260205003505340630525"


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


ROOT = find_project_root(Path(__file__).resolve())

SCORING_PATH = ROOT / "backend" / "artefatos" / "scoring_config.json"
CACHE_DIR = ROOT / "resultados" / "experimentos" / "EXP-006C-R2"
SIMULAR_PATH = ROOT / "backend" / "scripts" / "simular_pipeline_e2e_v2.py"
VALIDATE_C1_SCRIPT = ROOT / "scripts" / "validate_exp006f_c1_one_tx.py"


def flagged(df: pd.DataFrame) -> pd.Series:
    return df["decisao"].astype(str).isin(DECISOES_POSITIVAS)


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    y = df["is_fraud"].astype(int)
    pred = flagged(df).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

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


def load_cached_seed(seed: int) -> pd.DataFrame:
    path = CACHE_DIR / f"baseline_predictions_seed_{seed}.csv"
    assert path.exists(), f"Arquivo cache não encontrado: {path}"

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
    assert not missing, f"Colunas obrigatórias ausentes no cache: {missing}"

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

    out["exp006f_c1_applied_test"] = mask

    idx = out.index[mask]
    out.loc[idx, "decisao_original_exp006f_c1_test"] = out.loc[idx, "decisao"]
    out.loc[idx, "score_final_original_exp006f_c1_test"] = out.loc[idx, "score_final"]
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
        "rule_hits": int(cand["exp006f_c1_applied_test"].sum()),
    }


def run_subprocess(code_or_args: list[str], *, is_code: bool = False) -> subprocess.CompletedProcess[str]:
    if is_code:
        args = [sys.executable, "-c", code_or_args[0]]
    else:
        args = [sys.executable, *code_or_args]

    return subprocess.run(
        args,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_scoring_config_post_fase2_c1() -> None:
    assert SCORING_PATH.exists(), f"scoring_config.json não encontrado: {SCORING_PATH}"

    config = json.loads(SCORING_PATH.read_text(encoding="utf-8"))

    expected = {
        "threshold_confirmar": 62.0,
        "threshold_bloquear": 95.0,
        "lgbm_guard_enabled": True,
        "lgbm_guard_threshold": 0.30,
        "guard_exception_alto_valor_se_beh_enabled": True,
        "exp006f_c1_enabled": True,
        "exp006f_c1_min_score": 58.0,
        "exp006f_c1_max_score": 62.0,
        "exp006f_c1_min_valor": 100.0,
        "exp006f_c1_max_valor": 500.0,
        "exp006f_c1_max_rel_meses": 12.0,
        "exp006f_c1_min_lgbm_raw": 0.06,
        "exp006f_c1_max_lgbm_raw": 0.10,
        "exp006f_c1_require_first_receiver": True,
        "exp006f_c1_require_not_pix_random": True,
        "exp006f_c1_max_se_score": 0.0,
        "exp006f_c1_max_beh_score": 0.0,
        "se_pattern_residual_enabled": False,
        "exp003_residual_confirm_enabled": False,
    }

    for key, value in expected.items():
        assert key in config, f"Chave ausente no scoring_config: {key}"

        if isinstance(value, float):
            assert math.isclose(float(config[key]), value, rel_tol=0, abs_tol=1e-9), key
        else:
            assert config[key] == value, key


def test_decision_engine_has_c1_fields_subprocess() -> None:
    code = r"""
import sys
from pathlib import Path
root = Path('.').resolve()
sys.path.insert(0, str(root / 'backend'))
sys.path.insert(0, str(root / 'backend' / 'core'))
from core.decision_engine import EngineConfig
fields = getattr(EngineConfig, '__dataclass_fields__', {})
required = [
    'exp006f_c1_enabled',
    'exp006f_c1_min_score',
    'exp006f_c1_max_score',
    'exp006f_c1_min_valor',
    'exp006f_c1_max_valor',
    'exp006f_c1_max_rel_meses',
    'exp006f_c1_min_lgbm_raw',
    'exp006f_c1_max_lgbm_raw',
]
missing = [x for x in required if x not in fields]
if missing:
    raise SystemExit(f'Missing fields: {missing}')
print('EngineConfig C1 fields OK')
"""
    result = run_subprocess([code], is_code=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "EngineConfig C1 fields OK" in result.stdout


def test_simular_pipeline_c1_final_wrapper_present_in_source() -> None:
    assert SIMULAR_PATH.exists(), f"Arquivo não encontrado: {SIMULAR_PATH}"

    text = SIMULAR_PATH.read_text(encoding="utf-8")

    assert "EXP006F_C1_FINAL_DF_WRAPPER_ACTIVE" in text
    assert "_exp006f_c1_apply_to_predictions_df" in text
    assert "process_batch_sequential = _exp006f_c1_wrap_batch_function" in text
    assert "process_batch_parallel = _exp006f_c1_wrap_batch_function" in text


@pytest.mark.parametrize(
    "seed,expected",
    [
        (42, {"TP": 347, "FP": 14, "FN": 8}),
        (123, {"TP": 347, "FP": 12, "FN": 8}),
    ],
)
def test_cached_post_c1_metrics(seed: int, expected: dict[str, int]) -> None:
    base = load_cached_seed(seed)
    cand = apply_c1_min58(base)

    delta = compare_delta(base, cand)
    metrics = compute_metrics(cand)

    assert delta["fns_recuperados"] == 1, delta
    assert delta["fps_adicionados"] == 0, delta
    assert delta["tps_perdidos"] == 0, delta
    assert delta["rule_hits"] == 1, delta

    for key, value in expected.items():
        assert metrics[key] == value, metrics


@pytest.mark.slow
def test_runtime_c1_target_transaction_subprocess() -> None:
    assert VALIDATE_C1_SCRIPT.exists(), f"Script não encontrado: {VALIDATE_C1_SCRIPT}"

    result = run_subprocess([str(VALIDATE_C1_SCRIPT)])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK] C1 ativa no runtime real." in result.stdout
    assert TARGET_C1_TX in result.stdout
    assert "CONFIRMAR" in result.stdout
    assert "True" in result.stdout