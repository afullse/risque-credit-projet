"""Rapport de non-régression : référence v1.0.0 vs candidate v1.1.0
(candidate = même pipeline SANS class_weight='balanced', modification simulée)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.prediction import ModelService  # noqa: E402
from scripts.train_model import SEED, generer, matrice  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ref = json.loads((ROOT / "reports/reference_baseline.json").read_text("utf-8"))
tol = ref["tolerances"]
df = generer()
tr, te = train_test_split(df, test_size=0.25, random_state=SEED, stratify=df.defaut)
cand = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
cand.fit(matrice(tr), tr.defaut)
s = ModelService()
s.load()
X, y = matrice(te), te.defaut.to_numpy()


def m(model):
    p = model.predict_proba(X)[:, 1]
    yh = (p >= .5).astype(int)
    return dict(rappel=recall_score(y, yh), f1=f1_score(y, yh),
                precision=precision_score(y, yh), auc=roc_auc_score(y, p)), p


mr, pr = m(s.model)
mc, pc = m(cand)
lignes = ["| Métrique | Réf. v1.0.0 | Cand. v1.1.0 | Écart | Tolérance | Verdict |", 
          "|---|---|---|---|---|---|"]
for k in ["rappel", "f1", "precision", "auc"]:
    d = mc[k] - mr[k]
    t = tol.get(f"{k}_baisse_max")
    v = "—" if t is None else ("✅ OK" if d >= -t else "❌ RÉGRESSION")
    lignes.append(f"| {k} | {mr[k]:.4f} | {mc[k]:.4f} | {d:+.4f} | {'-'+str(t) if t else 'info'} | {v} |")
dp = np.abs(pc - pr)
lignes.append(f"\nÉcart absolu de probabilité : moyen {dp.mean():.4f}, max {dp.max():.4f}, "
              f"part > {tol['proba_abs']} : {(dp > tol['proba_abs']).mean():.1%}")
print("\n".join(lignes))
(ROOT / "reports/regression_comparison.md").write_text("\n".join(lignes), "utf-8")
