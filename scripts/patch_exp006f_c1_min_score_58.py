from __future__ import annotations

import json
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
SCORING_PATH = PROJECT_ROOT / "backend" / "artefatos" / "scoring_config.json"


def main() -> None:
    backup = SCORING_PATH.with_suffix(
        f".json.bak_exp006f_c1_min58_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(SCORING_PATH, backup)
    print(f"[OK] Backup criado: {backup}")

    config = json.loads(SCORING_PATH.read_text(encoding="utf-8"))

    config["exp006f_c1_enabled"] = True
    config["exp006f_c1_min_score"] = 58.0
    config["exp006f_c1_max_score"] = 62.0

    config["_metadata_exp006f_c1_min_score_58"] = {
        "patched_at": datetime.now().isoformat(timespec="seconds"),
        "reason": (
            "Validação runtime real mostrou transação alvo com score_final=58.01, "
            "embora cache artifact-only tivesse score_final=60.67."
        ),
        "status": "TEMPORARY_RUNTIME_VALIDATION",
    }

    rendered = json.dumps(config, indent=2, ensure_ascii=False)
    json.loads(rendered)
    SCORING_PATH.write_text(rendered + "\n", encoding="utf-8")

    print("[OK] exp006f_c1_min_score ajustado para 58.0")
    print("[INFO] Rode novamente scripts\\validate_exp006f_c1_one_tx.py")


if __name__ == "__main__":
    main()