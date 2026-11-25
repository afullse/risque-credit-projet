"""Tests unitaires du module de validation (EX01-EX03, EX06, EX08, EX09)."""
import math

import pytest
from pydantic import ValidationError

from app.validation import Demandeur


def test_valide(valide):
    assert Demandeur(**valide).age == 35


@pytest.mark.parametrize("age,ok", [(17, False), (18, True), (75, True), (76, False), (-5, False)])
def test_bornes_age(valide, age, ok):  # EX01 - analyse des valeurs limites
    valide["age"] = age
    if ok:
        Demandeur(**valide)
    else:
        with pytest.raises(ValidationError):
            Demandeur(**valide)


@pytest.mark.parametrize("v", ["250000", None, True, 0, -1, math.inf, math.nan])
def test_revenu_invalide(valide, v):  # EX02
    valide["revenu_mensuel"] = v
    with pytest.raises(ValidationError):
        Demandeur(**valide)


@pytest.mark.parametrize("v", [12.5, "12", 0, 121, 12.0])
def test_duree_invalide(valide, v):  # EX03 (12.0 refusé : type strict)
    valide["duree_mois"] = v
    with pytest.raises(ValidationError):
        Demandeur(**valide)


@pytest.mark.parametrize("champ", ["age", "revenu_mensuel", "montant_credit",
                                   "duree_mois", "anciennete_activite"])
def test_champ_manquant(valide, champ):  # EX06
    valide.pop(champ)
    with pytest.raises(ValidationError):
        Demandeur(**valide)


def test_champ_supplementaire(valide):  # EX08
    with pytest.raises(ValidationError):
        Demandeur(**valide, sexe="F")


def test_coherence_age_anciennete(valide):  # EX09
    valide.update(age=20, anciennete_activite=10)
    with pytest.raises(ValidationError):
        Demandeur(**valide)
