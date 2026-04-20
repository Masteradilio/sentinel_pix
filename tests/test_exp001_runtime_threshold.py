from __future__ import annotations

import json
from pathlib import Path
import unittest

from backend.core.decision_engine import EngineConfig, PixDecisionEngine


def _write_json(path: Path, payload: dict | list[str]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestExp001RuntimeThreshold(unittest.TestCase):
    def test_engine_uses_runtime_thresholds_from_scoring_config(self) -> None:
        tmp_path = Path(self._testMethodName)
        self.addCleanup(lambda: None)

        with TemporaryArtifactsDir(tmp_path) as artefatos_dir:
            _write_json(artefatos_dir / "lgbm_features.json", ["feature_a"])
            _write_json(
                artefatos_dir / "scoring_config.json",
                {
                    "mapeamento": {
                        "anchors_raw": [0.0, 1.0],
                        "anchors_out": [0.0, 100.0],
                    },
                    "faixas_decisao": {
                        "aprovar": {"threshold": 62.0},
                        "confirmar": {"threshold": 62.0},
                        "bloquear": {"threshold": 95.0},
                    },
                },
            )

            engine = PixDecisionEngine(EngineConfig(artefatos_dir=str(artefatos_dir)))

            self.assertEqual(engine.config.threshold_confirmar, 62.0)
            self.assertEqual(engine.config.threshold_bloquear, 95.0)

    def test_explicit_engine_config_override_still_wins(self) -> None:
        tmp_path = Path(self._testMethodName)
        self.addCleanup(lambda: None)

        with TemporaryArtifactsDir(tmp_path) as artefatos_dir:
            _write_json(artefatos_dir / "lgbm_features.json", ["feature_a"])
            _write_json(
                artefatos_dir / "scoring_config.json",
                {
                    "mapeamento": {
                        "anchors_raw": [0.0, 1.0],
                        "anchors_out": [0.0, 100.0],
                    },
                    "faixas_decisao": {
                        "aprovar": {"threshold": 62.0},
                        "confirmar": {"threshold": 62.0},
                        "bloquear": {"threshold": 95.0},
                    },
                },
            )

            engine = PixDecisionEngine(
                EngineConfig(
                    artefatos_dir=str(artefatos_dir),
                    threshold_confirmar=70.0,
                    threshold_bloquear=92.0,
                )
            )

            self.assertEqual(engine.config.threshold_confirmar, 70.0)
            self.assertEqual(engine.config.threshold_bloquear, 92.0)


class TemporaryArtifactsDir:
    def __init__(self, base_name: Path) -> None:
        self.base_name = base_name
        self.root = Path("tests") / ".tmp" / base_name

    def __enter__(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def __exit__(self, exc_type, exc, tb) -> None:
        for child in sorted(self.root.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        if self.root.exists():
            self.root.rmdir()
        parent = self.root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
