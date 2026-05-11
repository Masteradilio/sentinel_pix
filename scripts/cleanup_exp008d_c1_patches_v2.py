"""
Cleanup seguro EXP-008D v2.

Remove apenas os wrappers redundantes do:
  - backend/core/decision_engine.py
  - backend/core/pipeline_orquestrador.py

Preserva:
  - campos C1 no EngineConfig;
  - scoring_config.json;
  - wrapper final efetivo em backend/scripts/simular_pipeline_e2e_v2.py.

Diferença para o cleanup anterior:
  - usa remoção por linhas;
  - valida sintaxe com compile() antes de salvar;
  - se a sintaxe ficar inválida, não grava o arquivo.
"""

from __future__ import annotations

import argparse
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


ROOT = find_project_root(Path(__file__).resolve().parent)

ENGINE_PATH = ROOT / "backend" / "core" / "decision_engine.py"
PIPELINE_PATH = ROOT / "backend" / "core" / "pipeline_orquestrador.py"
SIMULAR_PATH = ROOT / "backend" / "scripts" / "simular_pipeline_e2e_v2.py"


def backup(path: Path) -> Path:
    backup_path = path.with_suffix(
        path.suffix + f".bak_exp008d_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(path, backup_path)
    print(f"[OK] Backup criado: {backup_path}")
    return backup_path


def assert_syntax_ok(path: Path, text: str) -> None:
    try:
        compile(text, str(path), "exec")
    except SyntaxError as exc:
        raise RuntimeError(
            f"Sintaxe inválida após cleanup em {path}\n"
            f"Linha: {exc.lineno}\n"
            f"Texto: {exc.text!r}\n"
            f"Erro: {exc}"
        ) from exc


def remove_block_by_header_and_terminal(
    text: str,
    header_substring: str,
    terminal_substring: str,
) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    removed = 0

    while True:
        header_idx = None

        for i, line in enumerate(lines):
            if header_substring in line:
                header_idx = i
                break

        if header_idx is None:
            break

        # Inclui a linha de separador imediatamente anterior, se existir.
        start_idx = header_idx
        if start_idx > 0 and lines[start_idx - 1].strip().startswith("# ==="):
            start_idx -= 1

        end_idx = None

        for j in range(header_idx, len(lines)):
            if terminal_substring in lines[j]:
                end_idx = j
                break

        if end_idx is None:
            raise RuntimeError(
                f"Encontrei header {header_substring!r}, mas não encontrei terminal {terminal_substring!r}"
            )

        # Inclui linhas em branco imediatamente seguintes.
        end_exclusive = end_idx + 1
        while end_exclusive < len(lines) and lines[end_exclusive].strip() == "":
            end_exclusive += 1

        del lines[start_idx:end_exclusive]
        removed += 1

    out = "".join(lines)

    # Normalização simples: reduz excesso de linhas em branco.
    while "\n\n\n\n" in out:
        out = out.replace("\n\n\n\n", "\n\n\n")

    return out, removed


def cleanup_file(path: Path, removals: list[tuple[str, str]], required_after: list[str]) -> dict:
    original = path.read_text(encoding="utf-8")
    text = original
    total_removed = 0

    for header, terminal in removals:
        text, n = remove_block_by_header_and_terminal(text, header, terminal)
        total_removed += n

    for required in required_after:
        if required not in text:
            raise RuntimeError(f"Após cleanup, marcador obrigatório ausente em {path}: {required}")

    if text == original:
        print(f"[OK] Nada a remover em {path}")
        return {"changed": False, "removed": 0}

    assert_syntax_ok(path, text)

    backup(path)
    path.write_text(text, encoding="utf-8")

    print(f"[OK] Cleanup aplicado em {path}. Blocos removidos: {total_removed}")

    return {"changed": True, "removed": total_removed}


def validate_simular_wrapper() -> None:
    text = SIMULAR_PATH.read_text(encoding="utf-8")

    required = [
        "EXP006F_C1_FINAL_DF_WRAPPER_ACTIVE",
        "_exp006f_c1_apply_to_predictions_df",
        "process_batch_sequential = _exp006f_c1_wrap_batch_function",
        "process_batch_parallel = _exp006f_c1_wrap_batch_function",
    ]

    missing = [x for x in required if x not in text]

    if missing:
        raise RuntimeError(f"Wrapper efetivo em simular_pipeline_e2e_v2.py incompleto: {missing}")

    print("[OK] Wrapper efetivo em simular_pipeline_e2e_v2.py preservado.")


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


def run_tests() -> bool:
    commands = [
        [sys.executable, "-m", "py_compile", "backend/core/decision_engine.py"],
        [sys.executable, "-m", "py_compile", "backend/core/pipeline_orquestrador.py"],
        [sys.executable, "-m", "py_compile", "backend/scripts/simular_pipeline_e2e_v2.py"],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_regression_post_fase2.py", "-q", "-m", "slow"],
    ]

    return all(run_command(cmd) == 0 for cmd in commands)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("EXP-008D — Cleanup Técnico dos Patches C1 v2")
    print("=" * 72)

    engine_status = cleanup_file(
        ENGINE_PATH,
        removals=[
            (
                "EXP-006F-C1 PATCH — Runtime wrapper defensivo",
                "_exp006f_c1_bind_runtime_wrappers()",
            ),
            (
                "EXP-006F-C1 PATCH — Bind defensivo da exceção near-threshold",
                "return PixDecisionEngine._apply_exp006f_c1_near_threshold_exception(self, tx, result)",
            ),
        ],
        required_after=[
            "exp006f_c1_enabled",
            "def _apply_exp006f_c1_near_threshold_exception",
        ],
    )

    pipeline_status = cleanup_file(
        PIPELINE_PATH,
        removals=[
            (
                "EXP-006F-C1 PATCH — PipelineOrquestrador final-result wrapper",
                "_exp006f_c1_bind_pipeline_wrappers()",
            ),
        ],
        required_after=[
            "class PipelineOrquestrador",
        ],
    )

    validate_simular_wrapper()

    print()
    print("Resumo:")
    print("decision_engine:", engine_status)
    print("pipeline_orquestrador:", pipeline_status)

    if args.run_tests:
        ok = run_tests()

        if not ok:
            raise SystemExit("[ERRO] Cleanup v2 aplicado, mas validações falharam.")

        print()
        print("[OK] Cleanup v2 validado por py_compile + regressão.")


if __name__ == "__main__":
    main()