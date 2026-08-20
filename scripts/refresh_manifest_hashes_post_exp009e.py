from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(".").resolve()

MANIFEST_PATH = ROOT / "backend" / "artefatos" / "MANIFEST_MODEL.json"

HASH_PATHS = {
    "scoring_config_sha256": ROOT / "backend" / "artefatos" / "scoring_config.json",
    "validation_report_sha256": ROOT / "docs" / "VALIDATION_REPORT_POST_FASE2.md",
    "rules_catalog_sha256": ROOT / "docs" / "RULES_CATALOG.md",
    "decision_trace_spec_sha256": ROOT / "docs" / "DECISION_TRACE_SPEC.md",
    "decision_trace_example_sha256": ROOT / "docs" / "DECISION_TRACE_EXAMPLE.json",
    "regression_test_sha256": ROOT / "tests" / "test_regression_post_fase2.py",
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest não encontrado: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    manifest.setdefault("artifact_hashes", {})

    for key, path in HASH_PATHS.items():
        manifest["artifact_hashes"][key] = sha256_file(path)

    manifest["hashes_refreshed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest.setdefault("notes", [])
    manifest["notes"].append(
        "Hashes atualizados após EXP-009E, depois de ajustes documentais e validação do pacote de governança."
    )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Hashes atualizados no Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()