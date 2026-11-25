"""Tests fonctionnels de l'API - 10 scénarios obligatoires + endpoints annexes."""
import pytest

from app.main import service


def assert_erreur(r, status, code):
    assert r.status_code == status
    body = r.json()
    assert body["erreur"] == code
    assert "Traceback" not in r.text and "File \"" not in r.text


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["statut"] == "ok"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert {"nom", "version", "features", "metriques"} <= r.json().keys()


# S1
def test_requete_valide(client, valide):
    r = client.post("/predict", json=valide)
    assert r.status_code == 200
    res = r.json()
    assert 0 <= res["probabilite_defaut"] <= 1
    assert res["classe_risque"] in {"faible", "moyen", "eleve"}
    assert res["version_modele"] == service.metadata["version"]
    assert res["alerte_hors_distribution"] is False


# S2
def test_champ_obligatoire_absent(client, valide):
    valide.pop("revenu_mensuel")
    r = client.post("/predict", json=valide)
    assert_erreur(r, 422, "DONNEES_INVALIDES")
    assert r.json()["details"][0]["champ"] == "revenu_mensuel"


# S3
def test_age_negatif(client, valide):
    valide["age"] = -5
    assert_erreur(client.post("/predict", json=valide), 422, "DONNEES_INVALIDES")


# S4
def test_revenu_texte(client, valide):
    valide["revenu_mensuel"] = "deux cent mille"
    assert_erreur(client.post("/predict", json=valide), 422, "DONNEES_INVALIDES")


# S5
@pytest.mark.parametrize("m", [0, -1000000])
def test_montant_nul_ou_negatif(client, valide, m):
    valide["montant_credit"] = m
    assert_erreur(client.post("/predict", json=valide), 422, "DONNEES_INVALIDES")


# S6
def test_duree_non_entiere(client, valide):
    valide["duree_mois"] = 12.5
    assert_erreur(client.post("/predict", json=valide), 422, "DONNEES_INVALIDES")


# S7
def test_champ_non_autorise(client, valide):
    valide["nationalite"] = "CI"
    assert_erreur(client.post("/predict", json=valide), 422, "DONNEES_INVALIDES")


# S8
def test_valeurs_extremes(client, valide):
    valide.update(revenu_mensuel=1e15, montant_credit=1e18)
    assert_erreur(client.post("/predict", json=valide), 422, "DONNEES_INVALIDES")


# S9
def test_modele_indisponible(client, valide, monkeypatch):
    monkeypatch.setattr(service, "model", None)
    assert_erreur(client.post("/predict", json=valide), 503, "MODELE_INDISPONIBLE")
    assert_erreur(client.get("/health"), 503, "MODELE_INDISPONIBLE")
    assert_erreur(client.get("/model-info"), 503, "MODELE_INDISPONIBLE")


# S10
@pytest.mark.parametrize("corps", ['{"age": 35, "revenu_mensuel": ', "pas du json", "[1,2,3]"])
def test_requete_mal_formee(client, corps):
    r = client.post("/predict", content=corps, headers={"Content-Type": "application/json"})
    assert_erreur(r, 422, "DONNEES_INVALIDES")


def test_methode_non_autorisee(client):
    assert client.get("/predict").status_code == 405
