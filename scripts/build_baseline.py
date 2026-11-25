"""Fige la version de référence (v1.0.0) : métriques, profils, tolérances."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.prediction import ModelService  # noqa: E402

s = ModelService()
s.load()
profils = [
    {"age": 35,
     "revenu_mensuel": 250000,
     "montant_credit": 1000000,
     "duree_mois": 12, "anciennete_activite": 4},
    {"age": 22,
     "revenu_mensuel": 60000,
     "montant_credit": 900000,
     "duree_mois": 6, "anciennete_activite": 0.5},
    {"age": 50,
     "revenu_mensuel": 800000,
     "montant_credit": 2000000,
     "duree_mois": 24, "anciennete_activite": 15},
    {"age": 41,
     "revenu_mensuel": 150000,
     "montant_credit": 3000000,
     "duree_mois": 36, "anciennete_activite": 8},
    {"age": 65,
     "revenu_mensuel": 100000,
     "montant_credit": 300000,
     "duree_mois": 12, "anciennete_activite": 30},
]
ref = {
    "version": s.metadata["version"],
    "metriques": s.metadata["metriques"],
    "tolerances": {"proba_abs": 0.02, "rappel_baisse_max": 0.03,
                   "f1_baisse_max": 0.03, "rappel_min_absolu": 0.60},
    "cles_sortie": ["probabilite_defaut", "classe_risque", "alerte_hors_distribution", "version_modele"],
    "profils": [{"entree": p, "probabilite": s.predict(p)["probabilite_defaut"]} for p in profils],
}
contenu_json = json.dumps(ref, indent=2, ensure_ascii=False)
(ROOT / "reports" / "reference_baseline.json").write_text(contenu_json, "utf-8")
print(json.dumps(ref, indent=2))
