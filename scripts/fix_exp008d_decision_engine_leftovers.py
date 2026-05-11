"""
scripts/fix_exp008d_decision_engine_leftovers.py

Remove resíduos órfãos do wrapper runtime EXP-006F-C1 no decision_engine.py.

Preserva:
  - campos C1 no EngineConfig;
  - método _apply_exp006f_c1_near_threshold_exception;
  - scoring_config.json;
  - wrapper efetivo em simular_pipeline_e2e_v2.py.

Remove:
  - chamadas órfãs a _exp006f_c1_bind_runtime_wrappers();
  - funções auxiliares antigas do wrapper runtime;
  - blocos top-level parciais que mencionem wrappers antigos.

Uso:
  python scripts\\fix_exp008d_decision_engine_leftovers.py
  python -m py_compile backend\\core\\decision_engine.py
  python -m pytest tests\\test_regression_post_fase2.py -q
  python -m pytest tests\\test_regression_post_fase2.py -q -m slow
"""

from __future__ import annotations

import ast
import shutil
from datetime import datetime
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


ROOT = find_project_root(Path(__file__).resolve().parent)
ENGINE_PATH = ROOT / "backend" / "core" / "decision_engine.py"


REMOVE_FUNCTION_NAMES = {
    "_exp006f_c1_extract_tx_from_call",
    "_exp006f_c1_apply_to_result",
    "_exp006f_c1_wrap_method",
    "_exp006f_c1_bind_runtime_wrappers",
    "_exp006f_c1_bound_method",
}

REMOVE_MARKERS = [
    "EXP-006F-C1 PATCH — Runtime wrapper defensivo",
    "EXP-006F-C1 PATCH - Runtime wrapper defensivo",
    "EXP-006F-C1 PATCH — Bind defensivo da exceção near-threshold",
    "EXP-006F-C1 PATCH - Bind defensivo da exceção near-threshold",
    "_exp006f_c1_bind_runtime_wrappers",
    "_exp006f_c1_wrap_method",
    "_exp006f_c1_extract_tx_from_call",
    "_exp006f_c1_apply_to_result",
    "EXP-006F-C1 runtime wrappers bound",
    "Falha ao ativar wrappers EXP-006F-C1",
]


REQUIRED_AFTER = [
    "exp006f_c1_enabled",
    "def _apply_exp006f_c1_near_threshold_exception",
]


def backup(path: Path) -> Path:
    backup_path = path.with_suffix(
        path.suffix + f".bak_fix_exp008d_leftovers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(path, backup_path)
    print(f"[OK] Backup criado: {backup_path}")
    return backup_path


def node_source(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)

    if start is None or end is None:
        return ""

    return "".join(lines[start - 1:end])


def should_remove_node(node: ast.AST, lines: list[str]) -> bool:
    # Remove funções top-level antigas do wrapper runtime.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name in REMOVE_FUNCTION_NAMES:
            return True

    src = node_source(lines, node)

    # Remove blocos top-level parciais que ainda mencionem wrappers antigos.
    if any(marker in src for marker in REMOVE_MARKERS):
        # Mas nunca remover classes inteiras, para não apagar PixDecisionEngine.
        if isinstance(node, ast.ClassDef):
            return False
        return True

    return False


def remove_marked_top_level_nodes(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)

    ranges: list[tuple[int, int, str]] = []

    for node in tree.body:
        if should_remove_node(node, lines):
            start = int(getattr(node, "lineno"))
            end = int(getattr(node, "end_lineno"))
            kind = type(node).__name__
            name = getattr(node, "name", "")
            label = f"{kind}:{name}" if name else kind
            ranges.append((start, end, label))

    if not ranges:
        return text, []

    # Junta ranges sobrepostos.
    ranges = sorted(ranges, key=lambda x: x[0])
    merged: list[tuple[int, int, str]] = []

    for start, end, label in ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, label))
        else:
            old_start, old_end, old_label = merged[-1]
            merged[-1] = (old_start, max(old_end, end), old_label + "+" + label)

    remove_lines: set[int] = set()

    for start, end, _label in merged:
        # Inclui comentário separador imediatamente anterior, se houver.
        adjusted_start = start

        while adjusted_start > 1:
            prev = lines[adjusted_start - 2].strip()
            if prev == "" or prev.startswith("# ===") or prev.startswith("# ---"):
                adjusted_start -= 1
                continue
            break

        # Inclui linhas em branco posteriores.
        adjusted_end = end
        while adjusted_end < len(lines) and lines[adjusted_end].strip() == "":
            adjusted_end += 1

        for i in range(adjusted_start, adjusted_end + 1):
            remove_lines.add(i)

    new_lines = [line for i, line in enumerate(lines, start=1) if i not in remove_lines]
    out = "".join(new_lines)

    while "\n\n\n\n" in out:
        out = out.replace("\n\n\n\n", "\n\n\n")

    return out, merged


def remove_plain_dangling_calls(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    new_lines = []
    removed = 0

    for line in lines:
        stripped = line.strip()

        if stripped in {
            "_exp006f_c1_bind_runtime_wrappers()",
            "PixDecisionEngine._exp006f_c1_wrappers_bound = bound",
        }:
            removed += 1
            continue

        if stripped.startswith("logger.info(") and "EXP-006F-C1 runtime wrappers bound" in stripped:
            removed += 1
            continue

        if stripped.startswith("logger.warning(") and "Falha ao ativar wrappers EXP-006F-C1" in stripped:
            removed += 1
            continue

        new_lines.append(line)

    return "".join(new_lines), removed


def main() -> None:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ENGINE_PATH}")

    original = ENGINE_PATH.read_text(encoding="utf-8")

    # O arquivo atual compila, então o AST deve parsear.
    new_text, removed_nodes = remove_marked_top_level_nodes(original)
    new_text, removed_plain = remove_plain_dangling_calls(new_text)

    for required in REQUIRED_AFTER:
        if required not in new_text:
            raise RuntimeError(f"Marcador obrigatório ausente após correção: {required}")

    # Valida sintaxe antes de gravar.
    compile(new_text, str(ENGINE_PATH), "exec")

    if new_text == original:
        print("[OK] Nenhum resíduo órfão encontrado no decision_engine.py.")
        return

    backup(ENGINE_PATH)
    ENGINE_PATH.write_text(new_text, encoding="utf-8")

    print("[OK] Resíduos órfãos removidos do decision_engine.py.")
    print(f"[OK] Nós AST removidos: {removed_nodes}")
    print(f"[OK] Linhas soltas removidas: {removed_plain}")
    print("[OK] Sintaxe validada antes da gravação.")


if __name__ == "__main__":
    main()