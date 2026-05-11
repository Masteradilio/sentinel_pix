"""
scripts/cleanup_exp008d_c1_patches.py

EXP-008D — Cleanup Técnico dos Patches C1

Objetivo:
  Remover wrappers experimentais redundantes/inefetivos criados durante a validação
  da C1, preservando o ponto efetivo validado em runtime:
    backend/scripts/simular_pipeline_e2e_v2.py

Este script:
  - Faz backup dos arquivos antes de alterar.
  - Remove o wrapper defensivo do decision_engine.py.
  - Remove o wrapper defensivo do pipeline_orquestrador.py.
  - Preserva EngineConfig + método C1 no decision_engine.py.
  - Preserva scoring_config.json.
  - Preserva o wrapper final efetivo no simular_pipeline_e2e_v2.py.
  - Opcionalmente roda py_compile e pytest.

Uso:
  python scripts\\cleanup_exp008d_c1_patches.py
  python scripts\\cleanup_exp008d_c1_patches.py --run-tests

Depois:
  python -m pytest tests\\test_regression_post_fase2.py -q
  python -m pytest tests\\test_regression_post_fase2.py -q -m slow
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(SCRIPT_DIR)

ENGINE_PATH = ROOT / "backend" / "core" / "decision_engine.py"
PIPELINE_PATH = ROOT / "backend" / "core" / "pipeline_orquestrador.py"
SIMULAR_PATH = ROOT / "backend" / "scripts" / "simular_pipeline_e2e_v2.py"
SCORING_PATH = ROOT / "backend" / "artefatos" / "scoring_config.json"

BACKUP_TAG = "exp008d_cleanup"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    backup_path = path.with_suffix(f"{path.suffix}.bak_{BACKUP_TAG}_{timestamp()}")
    shutil.copy2(path, backup_path)
    print(f"[OK] Backup criado: {backup_path}")
    return backup_path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def remove_block(text: str, start_marker: str, end_marker: str) -> tuple[str, int]:
    """
    Remove bloco começando em start_marker e terminando logo após end_marker.

    Retorna:
      texto atualizado, quantidade de blocos removidos
    """
    start = re.escape(start_marker)
    end = re.escape(end_marker)

    pattern = rf"\n?{start}.*?{end}\s*\n?"
    new_text, n = re.subn(pattern, "\n", text, flags=re.DOTALL)

    # Normaliza excesso de linhas em branco.
    new_text = re.sub(r"\n{4,}", "\n\n\n", new_text)

    return new_text, n


def cleanup_decision_engine() -> dict[str, int | bool]:
    text = read_text(ENGINE_PATH)
    original = text

    removals = 0

    # Wrapper defensivo criado após o método C1, mas que não interceptou o ponto efetivo.
    text, n = remove_block(
        text,
        "# ============================================================\n# EXP-006F-C1 PATCH — Runtime wrapper defensivo",
        "_exp006f_c1_bind_runtime_wrappers()",
    )
    removals += n

    # Bloco shim antigo, caso tenha sido inserido em alguma versão local.
    text, n = remove_block(
        text,
        "# ============================================================\n# EXP-006F-C1 PATCH — Bind defensivo da exceção near-threshold",
        "def _exp006f_c1_bound_method(self, tx, result):\n    return PixDecisionEngine._apply_exp006f_c1_near_threshold_exception(self, tx, result)",
    )
    removals += n

    changed = text != original

    if changed:
        backup(ENGINE_PATH)
        write_text(ENGINE_PATH, text)
        print(f"[OK] decision_engine.py limpo. Blocos removidos: {removals}")
    else:
        print("[OK] decision_engine.py não tinha wrappers redundantes para remover.")

    return {
        "changed": changed,
        "blocks_removed": removals,
        "has_engine_config_c1": "exp006f_c1_enabled:" in text,
        "has_c1_method": "def _apply_exp006f_c1_near_threshold_exception" in text,
        "has_runtime_wrapper": "_exp006f_c1_bind_runtime_wrappers" in text,
    }


def cleanup_pipeline_orquestrador() -> dict[str, int | bool]:
    text = read_text(PIPELINE_PATH)
    original = text

    removals = 0

    # Wrapper no PipelineOrquestrador foi carregado, mas não alterou o predictions_df final.
    text, n = remove_block(
        text,
        "# ============================================================\n# EXP-006F-C1 PATCH — PipelineOrquestrador final-result wrapper",
        "_exp006f_c1_bind_pipeline_wrappers()",
    )
    removals += n

    changed = text != original

    if changed:
        backup(PIPELINE_PATH)
        write_text(PIPELINE_PATH, text)
        print(f"[OK] pipeline_orquestrador.py limpo. Blocos removidos: {removals}")
    else:
        print("[OK] pipeline_orquestrador.py não tinha wrapper redundante para remover.")

    return {
        "changed": changed,
        "blocks_removed": removals,
        "has_pipeline_wrapper": "_exp006f_c1_bind_pipeline_wrappers" in text,
    }


def validate_simular_wrapper_present() -> dict[str, bool]:
    text = read_text(SIMULAR_PATH)

    result = {
        "simular_exists": SIMULAR_PATH.exists(),
        "has_final_df_wrapper_flag": "EXP006F_C1_FINAL_DF_WRAPPER_ACTIVE" in text,
        "has_apply_to_predictions_df": "_exp006f_c1_apply_to_predictions_df" in text,
        "wraps_process_batch_sequential": "process_batch_sequential = _exp006f_c1_wrap_batch_function" in text,
        "wraps_process_batch_parallel": "process_batch_parallel = _exp006f_c1_wrap_batch_function" in text,
    }

    if all(result.values()):
        print("[OK] Wrapper final efetivo em simular_pipeline_e2e_v2.py preservado.")
    else:
        print("[AVISO] Wrapper final em simular_pipeline_e2e_v2.py parece incompleto:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    return result


def validate_scoring_config() -> dict[str, bool]:
    import json

    cfg = json.loads(SCORING_PATH.read_text(encoding="utf-8"))

    required = [
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

    missing = [k for k in required if k not in cfg]

    if missing:
        print(f"[AVISO] scoring_config sem chaves C1: {missing}")
    else:
        print("[OK] scoring_config contém chaves C1 oficiais.")

    return {
        "ok": not missing,
        "enabled": bool(cfg.get("exp006f_c1_enabled", False)),
        "min_score_58": float(cfg.get("exp006f_c1_min_score", -1)) == 58.0,
    }


def run_command(args: list[str]) -> int:
    print()
    print("[RUN]", " ".join(args))

    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.stdout:
        print(proc.stdout.rstrip())

    if proc.stderr:
        print(proc.stderr.rstrip())

    print(f"[EXIT] {proc.returncode}")

    return proc.returncode


def run_validations() -> bool:
    commands = [
        [sys.executable, "-m", "py_compile", str(ENGINE_PATH)],
        [sys.executable, "-m", "py_compile", str(PIPELINE_PATH)],
        [sys.executable, "-m", "py_compile", str(SIMULAR_PATH)],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q", "-m", "slow"],
    ]

    codes = [run_command(cmd) for cmd in commands]

    return all(code == 0 for code in codes)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-008D cleanup técnico dos patches C1")
    parser.add_argument("--run-tests", action="store_true", help="Roda py_compile e pytest após cleanup.")
    args = parser.parse_args()

    print("=" * 72)
    print("EXP-008D — Cleanup Técnico dos Patches C1")
    print("=" * 72)

    print("[1/5] Limpando decision_engine.py...")
    engine_status = cleanup_decision_engine()

    print("[2/5] Limpando pipeline_orquestrador.py...")
    pipeline_status = cleanup_pipeline_orquestrador()

    print("[3/5] Validando wrapper final efetivo...")
    simular_status = validate_simular_wrapper_present()

    print("[4/5] Validando scoring_config...")
    scoring_status = validate_scoring_config()

    print("[5/5] Resumo")
    print()
    print("decision_engine:", engine_status)
    print("pipeline_orquestrador:", pipeline_status)
    print("simular_pipeline_e2e_v2:", simular_status)
    print("scoring_config:", scoring_status)

    if engine_status["has_runtime_wrapper"]:
        print("[AVISO] Ainda existe wrapper runtime no decision_engine.py.")

    if pipeline_status["has_pipeline_wrapper"]:
        print("[AVISO] Ainda existe wrapper pipeline no pipeline_orquestrador.py.")

    if not all(simular_status.values()):
        print("[ERRO] Wrapper final efetivo não está completo. Não prossiga.")
        raise SystemExit(2)

    if not scoring_status["ok"]:
        print("[ERRO] scoring_config não contém configuração C1 completa. Não prossiga.")
        raise SystemExit(2)

    if args.run_tests:
        print()
        print("=" * 72)
        print("Rodando validações")
        print("=" * 72)

        ok = run_validations()

        if not ok:
            print("[ERRO] Validações falharam. Verifique logs acima.")
            raise SystemExit(1)

        print("[OK] Cleanup validado por py_compile + regressão.")

    print()
    print("[OK] EXP-008D cleanup executado.")
    print("[INFO] Se não usou --run-tests, rode agora:")
    print("  python -m pytest tests\\test_regression_post_fase2.py -q")
    print("  python -m pytest tests\\test_regression_post_fase2.py -q -m slow")


if __name__ == "__main__":
    main()