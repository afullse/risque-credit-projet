"""Module de validation : format et cohérence des données du demandeur."""
import math

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

AGE_MIN, AGE_MAX = 18, 75
DUREE_MIN, DUREE_MAX = 1, 120
REVENU_MAX = 1_000_000_000      # 1 milliard FCFA / mois
MONTANT_MAX = 5_000_000_000     # 5 milliards FCFA
ANCIENNETE_MAX = 60
CHAMPS = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]


class Demandeur(BaseModel):
    """Contrat d'entrée. Champs supplémentaires interdits, types stricts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    age: StrictInt = Field(..., ge=AGE_MIN, le=AGE_MAX)
    revenu_mensuel: float = Field(..., gt=0, le=REVENU_MAX)
    montant_credit: float = Field(..., gt=0, le=MONTANT_MAX)
    duree_mois: StrictInt = Field(..., ge=DUREE_MIN, le=DUREE_MAX)
    anciennete_activite: float = Field(..., ge=0, le=ANCIENNETE_MAX)

    @field_validator("revenu_mensuel", "montant_credit", "anciennete_activite", mode="before")
    @classmethod
    def nombre_fini(cls, v):
        # bool est un sous-type de int en Python : on le refuse explicitement
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("doit être un nombre")
        if not math.isfinite(v):
            raise ValueError("doit être un nombre fini")
        return float(v)

    @model_validator(mode="after")
    def coherence(self):
        if self.anciennete_activite > self.age - 14:
            raise ValueError("ancienneté d'activité incohérente avec l'âge")
        return self
