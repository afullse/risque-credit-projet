"""Tests de non-régression du logiciel et du modèle (EX07).
Tolérances définies AVANT comparaison dans reports/reference_baseline.json."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, recall_score

from app.main import service
from app.preprocessing import transformer

ROOT = Path(__file__).resolve().parent.parent
REF = json.loads((ROOT / "reports" / "reference_baseline.json").read_text("utf-8"))
TOL = REF["tolerances"]


@pytest.fixture(scope="module")
def jeu_test():
    df = pd.read_csv(ROOT / "models" / "jeu_test_reference.csv")
    X = np.vstack([transformer(r) for r in df.drop(columns="defaut").to_dict("records")])
    return X, df["defaut"].to_numpy()


def test_contrat_sorties(client):
    for prof in REF["profils"]:
        res = client.post("/predict", json=prof["entree"]).json()
        assert set(res) == set(REF["cles_sortie"])
        assert 0 <= res["probabilite_defaut"] <= 1


def test_predictions_stables(client):
    for prof in REF["profils"]:
        p = client.post("/predict", json=prof["entree"]).json()["probabilite_defaut"]
        assert abs(p - prof["probabilite"]) <= TOL["proba_abs"], prof["entree"]


def test_metriques_vs_reference(jeu_test):
    X, y = jeu_test
    yhat = (service.model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    rappel, f1 = recall_score(y, yhat), f1_score(y, yhat)
    assert rappel >= REF["metriques"]["rappel"] - TOL["rappel_baisse_max"], f"rappel={rappel:.4f}"
    assert f1 >= REF["metriques"]["f1"] - TOL["f1_baisse_max"], f"f1={f1:.4f}"
    assert rappel >= TOL["rappel_min_absolu"]


def test_sauvegarde_rechargement(tmp_path, jeu_test):
    X, _ = jeu_test
    f = tmp_path / "m.joblib"
    joblib.dump(service.model, f)
    p1 = service.model.predict_proba(X)[:, 1]
    p2 = joblib.load(f).predict_proba(X)[:, 1]
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_reproductibilite(client, valide):
    ps = {client.post("/predict", json=valide).json()["probabilite_defaut"] for _ in range(5)}
    assert len(ps) == 1
