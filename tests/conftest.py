import pytest
from fastapi.testclient import TestClient

from app.main import app, service

VALIDE = {"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000,
          "duree_mois": 12, "anciennete_activite": 4}


@pytest.fixture
def valide():
    return dict(VALIDE)


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    if not service.ready:      # restaure l'état si un test a déchargé le modèle
        service.load()
