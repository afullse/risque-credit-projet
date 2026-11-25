"""Module de prétraitement : construction du vecteur de variables du modèle."""
import math

import numpy as np

FEATURES = [
    "age", "log_revenu", "log_montant", "duree_mois",
    "anciennete_activite", "ratio_mensualite_revenu", "ratio_montant_revenu_annuel",
]


class PreprocessingError(Exception):
    """Erreur contrôlée du prétraitement."""


def transformer(d: dict) -> np.ndarray:
    try:
        revenu = float(d["revenu_mensuel"])
        montant = float(d["montant_credit"])
        duree = int(d["duree_mois"])
        if revenu <= 0 or montant <= 0 or duree <= 0:
            raise PreprocessingError("valeurs non strictement positives")
        x = [
            float(d["age"]),
            math.log(revenu),
            math.log(montant),
            float(duree),
            float(d["anciennete_activite"]),
            (montant / duree) / revenu,
            montant / (12 * revenu),
        ]
    except PreprocessingError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise PreprocessingError(f"variable absente ou invalide: {e}") from e
    arr = np.asarray([x], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise PreprocessingError("valeur non finie après transformation")
    return arr
