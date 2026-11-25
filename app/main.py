"""API FastAPI de prédiction du risque de crédit."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.prediction import ModelService, ModelUnavailableError, PredictionError
from app.preprocessing import PreprocessingError
from app.validation import Demandeur

log = logging.getLogger("creditrisk")
service = ModelService()


@asynccontextmanager
async def lifespan(_app):
    try:
        service.load()
    except ModelUnavailableError:
        log.exception("Modèle indisponible au démarrage")
    yield


app = FastAPI(title="CreditRisk API", version="1.0.0", lifespan=lifespan)


def erreur(status, code, message, details=None):
    body = {"erreur": code, "message": message}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(RequestValidationError)
async def validation_handler(_req: Request, exc: RequestValidationError):
    details = [{"champ": ".".join(str(p) for p in e["loc"][1:]) or "corps",
                "type": e["type"], "message": e["msg"]} for e in exc.errors()]
    return erreur(422, "DONNEES_INVALIDES", "Les données soumises sont invalides.", details)


@app.exception_handler(Exception)
async def generic_handler(_req: Request, _exc: Exception):
    log.exception("Erreur interne")
    return erreur(500, "ERREUR_INTERNE", "Une erreur interne est survenue.")


@app.get("/health")
def health():
    if not service.ready:
        return erreur(503, "MODELE_INDISPONIBLE", "Le modèle n'est pas chargé.")
    return {"statut": "ok", "modele_charge": True}


@app.get("/model-info")
def model_info():
    if not service.ready:
        return erreur(503, "MODELE_INDISPONIBLE", "Le modèle n'est pas chargé.")
    m = service.metadata
    return {k: m.get(k) for k in ("nom", "version", "date_entrainement", "features",
                                  "algorithme", "metriques", "donnees")}


@app.post("/predict")
def predict(demandeur: Demandeur):
    try:
        return service.predict(demandeur.model_dump())
    except ModelUnavailableError:
        log.exception("Prédiction impossible : modèle indisponible")
        return erreur(503, "MODELE_INDISPONIBLE", "Le service de prédiction est indisponible.")
    except (PreprocessingError, PredictionError):
        log.exception("Échec du pipeline")
        return erreur(500, "ERREUR_PREDICTION", "La prédiction n'a pas pu être calculée.")
