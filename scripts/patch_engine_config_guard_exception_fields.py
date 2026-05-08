"""
scripts/patch_engine_config_guard_exception_fields.py

Adiciona os campos da exceção contextual EXP-004-FINAL ao EngineConfig.

Corrige:
  'EngineConfig' object has no attribute 'guard_exception_alto_valor_se_beh_enabled'

Não muda regra de decisão.
Não promove LGBM v6.2.
Apenas torna os campos disponíveis no dataclass EngineConfig.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
ENGINE_PATH = PROJECT_ROOT / "backend" / "core" / "decision_engine.py"


FIELDS_BLOCK = """\
    # --- EXP-004-FINAL: excecao cirurgica ao guard rail LGBM ---
    guard_exception_alto_valor_se_beh_enabled: bool = False
    guard_exception_alto_valor_min: float = 15000.0
    guard_exception_alto_valor_rel_max: float = 12.0
    guard_exception_alto_valor_if_min: float = 0.985
    guard_exception_alto_valor_lgbm_min: float = 0.01
    guard_exception_alto_valor_age_min: int = 18
    guard_exception_alto_valor_age_max: int = 90
    guard_exception_alto_valor_require_first_receiver: bool = True
    guard_exception_alto_valor_require_pf: bool = True

"""


def find_engine_config_block(lines: list[str]) -> tuple[int, int]:
    start = None

    for i, line in enumerate(lines):
        if line.startswith("class EngineConfig"):
            start = i
            break

    if start is None:
        raise RuntimeError("Não encontrei 'class EngineConfig' em decision_engine.py")

    end = len(lines)

    for i in range(start + 1, len(lines)):
        line = lines[i]

        if not line.strip():
            continue

        # Próxima definição top-level.
        if not line.startswith((" ", "\t")) and (
            line.startswith("class ") or line.startswith("def ") or line.startswith("@")
        ):
            end = i
            break

    return start, end


def main() -> None:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ENGINE_PATH}")

    text = ENGINE_PATH.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start, end = find_engine_config_block(lines)
    block_text = "".join(lines[start:end])

    if "guard_exception_alto_valor_se_beh_enabled:" in block_text:
        print("[OK] Campos guard_exception já existem dentro de EngineConfig. Nada a aplicar.")
        return

    insert_idx = None

    # Preferir inserir depois do lgbm_guard_threshold, se existir.
    for i in range(start, end):
        if "lgbm_guard_threshold" in lines[i]:
            insert_idx = i + 1
            break

    # Fallback: depois do lgbm_guard_enabled.
    if insert_idx is None:
        for i in range(start, end):
            if "lgbm_guard_enabled" in lines[i]:
                insert_idx = i + 1
                break

    # Fallback final: antes do fim da classe.
    if insert_idx is None:
        insert_idx = end

    backup_path = ENGINE_PATH.with_suffix(
        f".py.bak_engine_config_guard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(ENGINE_PATH, backup_path)
    print(f"[OK] Backup criado: {backup_path}")

    new_lines = lines[:insert_idx] + [FIELDS_BLOCK] + lines[insert_idx:]
    ENGINE_PATH.write_text("".join(new_lines), encoding="utf-8")

    print(f"[OK] Campos adicionados ao EngineConfig em: {ENGINE_PATH}")


if __name__ == "__main__":
    main()