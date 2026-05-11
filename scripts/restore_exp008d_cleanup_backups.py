from __future__ import annotations

import shutil
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "backend").exists() and (p / "dados").exists():
            return p
    return start.parent


ROOT = find_project_root(Path(__file__).resolve().parent)

TARGETS = [
    ROOT / "backend" / "core" / "decision_engine.py",
    ROOT / "backend" / "core" / "pipeline_orquestrador.py",
]


def latest_backup_for(path: Path) -> Path:
    candidates = sorted(
        path.parent.glob(path.name + ".bak_exp008d_cleanup_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"Nenhum backup exp008d encontrado para {path}")

    return candidates[0]


def main() -> None:
    for target in TARGETS:
        backup = latest_backup_for(target)
        restore_backup = target.with_suffix(target.suffix + ".bak_before_restore_exp008d")

        shutil.copy2(target, restore_backup)
        shutil.copy2(backup, target)

        print(f"[OK] Restaurado: {target}")
        print(f"     De backup: {backup}")
        print(f"     Estado quebrado salvo em: {restore_backup}")


if __name__ == "__main__":
    main()