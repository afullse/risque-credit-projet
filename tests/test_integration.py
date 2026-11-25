"""Tests d'intégration : validation -> prétraitement -> modèle -> API."""
import numpy as np
import pytest

from app.main import service
from app.prediction import ModelService, ModelUnavailableError
from app.preprocessing import FEATURES, PreprocessingError, transformer
from app.validation import Demandeur


def test_pipeline_bout_en_bout(client, valide):
    d = Demandeur(**valide).model_dump()                       # 2. validation
    X = transformer(d)                                          # 3. transformation
    assert X.shape == (1, len(FEATURES))
    assert X[0, 5] == pytest.approx((1_000_000 / 12) / 250_000)  # ratio mensualité/revenu
    p_direct = service.model.predict_proba(X)[0, 1]             # 4. prédiction
    r = client.post("/predict", json=valide)                    # 1. requête
    body = r.json()
    assert set(body) == {"probabilite_defaut", "classe_risque",  # 5. structure
                         "alerte_hors_distribution", "version_modele"}
    assert body["probabilite_defaut"] == pytest.approx(p_direct, abs=1e-6)
    info = client.get("/model-info").json()                     # 6. métadonnées
    assert info["version"] == body["version_modele"]
    assert info["features"] == FEATURES
    assert service.model.n_features_in_ == len(info["features"])


def test_modele_non_chargeable(tmp_path):
    s = ModelService(tmp_path / "absent.joblib", tmp_path / "absent.json")
    with pytest.raises(ModelUnavailableError):
        s.load()
    assert not s.ready


def test_modele_corrompu(tmp_path):
    f = tmp_path / "m.joblib"
    f.write_bytes(b"corrompu")
    with pytest.raises(ModelUnavailableError):
        ModelService(f, tmp_path / "m.json").load()


def test_variable_absente_au_pretraitement(valide):
    valide.pop("duree_mois")
    with pytest.raises(PreprocessingError):
        transformer(valide)


def test_erreur_pretraitement_via_api(client, valide, monkeypatch):
    def boom(_):
        raise PreprocessingError("détail interne sensible /srv/secret")
    monkeypatch.setattr("app.prediction.transformer", boom)
    r = client.post("/predict", json=valide)
    assert r.status_code == 500 and r.json()["erreur"] == "ERREUR_PREDICTION"
    assert "secret" not in r.text


def test_dimensions_incompatibles(client, valide, monkeypatch):
    monkeypatch.setattr("app.prediction.transformer", lambda d: np.zeros((1, 3)))
    r = client.post("/predict", json=valide)
    assert r.status_code == 500 and r.json()["erreur"] == "ERREUR_PREDICTION"


def test_erreur_interne_inference(client, valide, monkeypatch):
    class Casse:
        n_features_in_ = len(FEATURES)

        def predict_proba(self, X):
            raise RuntimeError("stack interne")
    monkeypatch.setattr(service, "model", Casse())
    r = client.post("/predict", json=valide)
    assert r.status_code == 500
    assert "stack" not in r.text and "Traceback" not in r.text


def test_exception_imprevue_masquee(client, valide, monkeypatch):
    monkeypatch.setattr(service, "predict", lambda d: 1 / 0)
    r = client.post("/predict", json=valide)
    assert r.status_code == 500 and r.json()["erreur"] == "ERREUR_INTERNE"
    assert "ZeroDivision" not in r.text
