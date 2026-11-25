"""Chargement du modèle et inférence."""
import json
from pathlib import Path

import joblib
import numpy as np

from app.preprocessing import FEATURES, transformer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model_v1.joblib"
META_PATH = MODEL_DIR / "model_v1.json"
SEUIL_ELEVE, SEUIL_MOYEN = 0.5, 0.25


class ModelUnavailableError(Exception):
    pass


class PredictionError(Exception):
    pass


class ModelService:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        self.model_path, self.meta_path = Path(model_path), Path(meta_path)
        self.model, self.metadata = None, {}

    def load(self):
        try:
            self.model = joblib.load(self.model_path)
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception as e:  # fichier absent, corrompu, etc.
            self.model = None
            raise ModelUnavailableError(str(e)) from e

    @property
    def ready(self) -> bool:
        return self.model is not None

    def hors_distribution(self, d: dict) -> bool:
        bornes = self.metadata.get("plages_entrainement", {})
        return any(not (b[0] <= d[k] <= b[1]) for k, b in bornes.items() if k in d)

    def predict(self, d: dict) -> dict:
        if not self.ready:
            raise ModelUnavailableError("modèle non chargé")
        X = transformer(d)
        if X.shape[1] != getattr(self.model, "n_features_in_", len(FEATURES)):
            raise PredictionError("dimension des données incompatible avec le modèle")
        try:
            p = float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            raise PredictionError("échec de l'inférence") from e
        if not np.isfinite(p) or not 0.0 <= p <= 1.0:
            raise PredictionError("probabilité invalide")
        classe = "eleve" if p >= SEUIL_ELEVE else "moyen" if p >= SEUIL_MOYEN else "faible"
        return {
            "probabilite_defaut": round(p, 6),
            "classe_risque": classe,
            "alerte_hors_distribution": self.hors_distribution(d),
            "version_modele": self.metadata.get("version", "inconnue"),
        }
