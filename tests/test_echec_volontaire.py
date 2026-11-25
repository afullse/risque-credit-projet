"""Preuve que le workflow échoue si un test échoue.
Activé uniquement avec FORCER_ECHEC=1 (ou en décommentant), sur une branche dédiée."""
import os

import pytest


@pytest.mark.skipif(os.getenv("FORCER_ECHEC") != "1", reason="démonstration désactivée")
def test_echec_volontaire():
    assert 0.5 <= 0.0, "Échec volontaire pour vérifier que la CI devient rouge"
