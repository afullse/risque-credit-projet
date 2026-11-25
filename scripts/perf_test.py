"""Test de performance local léger : latence d'inférence, latence API, P95, débit,
charge progressive (1 -> 32 clients concurrents). Lancer l'API avant :
  uvicorn app.main:app --port 8000 --workers 1"""
import json
import platform
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.prediction import ModelService  # noqa: E402
from app.preprocessing import transformer  # noqa: E402

URL = "http://127.0.0.1:8000/predict"
D = {"age": 35, 
     "revenu_mensuel": 250000, 
     "montant_credit": 1000000, 
     "duree_mois": 12, "anciennete_activite": 4}


def p(vals, q):
    return float(np.percentile(vals, q)) * 1000


s = ModelService()
s.load()
inf = []
for _ in range(2000):
    t = time.perf_counter()
    s.predict(D)
    inf.append(time.perf_counter() - t)
X = transformer(D)
brut = []
for _ in range(2000):
    t = time.perf_counter()
    s.model.predict_proba(X)
    brut.append(time.perf_counter() - t)

res = {"config": {"python": platform.python_version(), "os": platform.platform(),
                  "cpu": platform.processor() or platform.machine(),
                  "serveur": "uvicorn 1 worker, client httpx local (même machine)"},
       "inference": {"predict_proba_p50_ms": p(brut, 50), "predict_proba_p95_ms": p(brut, 95),
                     "pipeline_p50_ms": p(inf, 50), "pipeline_p95_ms": p(inf, 95)},
       "charge": []}

with httpx.Client() as c:
    for _ in range(50):
        c.post(URL, json=D)  # échauffement


def appel(c):
    t = time.perf_counter()
    r = c.post(URL, json=D)
    return time.perf_counter() - t, r.status_code


for users in [1, 4, 8, 16, 32]:
    n = 300
    with httpx.Client(limits=httpx.Limits(max_connections=users)) as c, ThreadPoolExecutor(users) as ex:
        t0 = time.perf_counter()
        out = list(ex.map(lambda _: appel(c), range(n)))
        dur = time.perf_counter() - t0
    lat = [o[0] for o in out]
    res["charge"].append({"clients": users, "requetes": n, "erreurs": sum(o[1] != 200 for o in out),
                          "p50_ms": p(lat, 50), "p95_ms": p(lat, 95), "max_ms": max(lat) * 1000,
                          "moyenne_ms": st.mean(lat) * 1000, "debit_rps": n / dur})

out = ROOT / "reports/performance_results.json"
out.write_text(json.dumps(res, indent=2), "utf-8")
print(json.dumps(res, indent=2))
