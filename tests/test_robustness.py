"""Tests de robustesse : entrées invalides (rejet) vs inhabituelles valides (alerte)."""
import json
import math

import pytest


def check_proba(res):
    p = res["probabilite_defaut"]
    assert math.isfinite(p) and 0.0 <= p <= 1.0


# R1 valeurs manquantes (null)
@pytest.mark.parametrize("champ", ["age", "revenu_mensuel", "duree_mois"])
def test_valeur_null(client, valide, champ):
    valide[champ] = None
    assert client.post("/predict", json=valide).status_code == 422


# R2 non numériques
@pytest.mark.parametrize("v", ["abc", "", [1], {"x": 1}, True])
def test_non_numerique(client, valide, v):
    valide["montant_credit"] = v
    assert client.post("/predict", json=valide).status_code == 422


# R3 non finies (JSON non standard : NaN / Infinity envoyés en texte brut)
@pytest.mark.parametrize("litteral", ["NaN", "Infinity", "-Infinity"])
def test_non_fini(client, valide, litteral):
    corps = json.dumps(valide).replace("250000", litteral)
    r = client.post("/predict", content=corps, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# R4 limites des intervalles autorisés (acceptées)
@pytest.mark.parametrize("maj", [
    {"age": 18, "anciennete_activite": 0}, {"age": 75}, {"duree_mois": 1},
    {"duree_mois": 120}, {"anciennete_activite": 0},
    {"revenu_mensuel": 1_000_000_000}, {"montant_credit": 5_000_000_000},
])
def test_limites_valides(client, valide, maj):
    valide.update(maj)
    r = client.post("/predict", json=valide)
    assert r.status_code == 200
    check_proba(r.json())


# R5 hors distribution d'entraînement mais valide -> accepté + alerte
@pytest.mark.parametrize("maj", [{"age": 74}, {"duree_mois": 120},
                                 {"revenu_mensuel": 50_000_000}, {"montant_credit": 2e9}])
def test_hors_distribution_alerte(client, valide, maj):
    valide.update(maj)
    r = client.post("/predict", json=valide)
    assert r.status_code == 200
    check_proba(r.json())
    assert r.json()["alerte_hors_distribution"] is True


# R6 schéma incorrect
@pytest.mark.parametrize("corps", [[], 42, "texte", {"demandeur": {"age": 35}}])
def test_schema_incorrect(client, corps):
    assert client.post("/predict", json=corps).status_code == 422


# R7 invariance : la probabilité doit être monotone croissante avec le montant
def test_monotonie_montant(client, valide):
    ps = []
    for m in [200_000, 1_000_000, 3_000_000, 6_000_000]:
        valide["montant_credit"] = m
        ps.append(client.post("/predict", json=valide).json()["probabilite_defaut"])
    assert ps == sorted(ps)
