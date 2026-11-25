"""Génère un jeu SYNTHÉTIQUE documenté et entraîne le modèle de test versionné.
ATTENTION : données synthétiques -> servent à tester le système, pas à prouver
la performance réelle d'un modèle de crédit."""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.preprocessing import FEATURES, transformer  # noqa: E402

SEED = 42


def generer(n=5000, seed=SEED):
    r = np.random.default_rng(seed)
    age = r.integers(18, 71, n)
    revenu = np.round(r.lognormal(np.log(200_000), 0.6, n), -3).clip(30_000, 5_000_000)
    duree = r.choice([3, 6, 9, 12, 18, 24, 36, 48, 60], n)
    montant = np.round(revenu * r.uniform(0.5, 12, n), -4).clip(50_000, 50_000_000)
    anc = np.minimum(r.gamma(2, 2.5, n).round(1), age - 14)
    mens = montant / duree / revenu
    z = -2.2 + 3.0 * mens - 0.12 * anc - 0.015 * (age - 40) + 0.01 * duree
    defaut = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame(dict(age=age, revenu_mensuel=revenu, montant_credit=montant,
                             duree_mois=duree, anciennete_activite=anc, defaut=defaut))


def matrice(df):
    return np.vstack([transformer(row) for row in df.to_dict("records")])


def entrainer(out_dir=ROOT / "models", version="1.0.0", C=1.0):
    df = generer()
    tr, te = train_test_split(df, test_size=0.25, random_state=SEED, stratify=df.defaut)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                               max_iter=1000, random_state=SEED))
    model.fit(matrice(tr), tr.defaut)
    p = model.predict_proba(matrice(te))[:, 1]
    yhat = (p >= 0.5).astype(int)
    metriques = dict(rappel=recall_score(te.defaut, yhat), f1=f1_score(te.defaut, yhat),
                     precision=precision_score(te.defaut, yhat), auc=roc_auc_score(te.defaut, p))
    metriques = {k: round(float(v), 4) for k, v in metriques.items()}
    cols = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]
    meta = dict(nom="credit-risk-logreg", version=version, date_entrainement=str(date.today()),
                algorithme="StandardScaler + LogisticRegression(class_weight=balanced)",
                features=FEATURES, seed=SEED, metriques=metriques,
                donnees="synthétiques (scripts/train_model.py) - non représentatives",
                plages_entrainement={c: [float(df[c].min()), float(df[c].max())] for c in cols})
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "model_v1.joblib")
    (out_dir / "model_v1.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    te.to_csv(out_dir / "jeu_test_reference.csv", index=False)
    return meta


if __name__ == "__main__":
    print(json.dumps(entrainer(), indent=2, ensure_ascii=False))
