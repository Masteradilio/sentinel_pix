#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_model_artifacts.py — Testes de Fumaça de Modelos e Contratos de Features
"""

import json
from pathlib import Path
import unittest
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# Determinar paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
ARTEFATOS_DIR = PROJECT_ROOT / "backend" / "artefatos"


class TestModelArtifacts(unittest.TestCase):

    def test_load_lightgbm_model_and_contract(self) -> None:
        """Garante que o LightGBM oficial é carregável e bate com o lgbm_features.json."""
        lgb_path = ARTEFATOS_DIR / "model_lightgbm.joblib"
        json_path = ARTEFATOS_DIR / "lgbm_features.json"

        self.assertTrue(lgb_path.exists(), f"Modelo LightGBM ausente em {lgb_path}")
        self.assertTrue(json_path.exists(), f"Contrato de features do LightGBM ausente em {json_path}")

        # Carregar modelo
        model = joblib.load(lgb_path)
        self.assertIsNotNone(model, "Falha ao deserializar o modelo LightGBM")
        
        # Carregar contrato
        with open(json_path, "r", encoding="utf-8") as f:
            contract = json.load(f)

        self.assertIn("features", contract)
        self.assertIn("version", contract)
        
        # Obter features do modelo
        self.assertTrue(hasattr(model, "feature_name_"), "Modelo LightGBM não possui atributo feature_name_")
        features_in_model = list(model.feature_name_)
        features_in_json = list(contract["features"])

        self.assertEqual(
            features_in_model,
            features_in_json,
            "A lista de features do LightGBM serializado difere do lgbm_features.json!"
        )

    def test_load_isolation_forest_and_contract(self) -> None:
        """Garante que o Isolation Forest e Scaler oficiais são carregáveis e batem com if_features.json."""
        if_path = ARTEFATOS_DIR / "model_isolation_forest.joblib"
        scaler_path = ARTEFATOS_DIR / "scaler_isolation_forest.joblib"
        ref_path = ARTEFATOS_DIR / "if_ref_raw_train.npy"
        json_path = ARTEFATOS_DIR / "if_features.json"

        self.assertTrue(if_path.exists(), f"Modelo Isolation Forest ausente em {if_path}")
        self.assertTrue(scaler_path.exists(), f"RobustScaler do Isolation Forest ausente em {scaler_path}")
        self.assertTrue(ref_path.exists(), f"Array de scores de referência do IF ausente em {ref_path}")
        self.assertTrue(json_path.exists(), f"Contrato de features do IF ausente em {json_path}")

        # Carregar artefatos
        model = joblib.load(if_path)
        scaler = joblib.load(scaler_path)
        ref_scores = np.load(ref_path)

        self.assertIsInstance(model, IsolationForest, "Objeto carregado não é IsolationForest")
        self.assertIsInstance(scaler, RobustScaler, "Objeto carregado não é RobustScaler")
        self.assertTrue(len(ref_scores) > 0, "Array de referência de scores de treino está vazio")

        # Carregar contrato
        with open(json_path, "r", encoding="utf-8") as f:
            features_in_json = json.load(f)

        # Validar consistência de dimensões
        self.assertEqual(
            scaler.n_features_in_,
            len(features_in_json),
            "Número de features no RobustScaler difere das features em if_features.json!"
        )
        self.assertEqual(
            model.n_features_in_,
            len(features_in_json),
            "Número de features no IsolationForest difere das features em if_features.json!"
        )


if __name__ == "__main__":
    unittest.main()
