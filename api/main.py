"""FraudShield demonstration API — synthetic BAF data only."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from api.inference import FraudPredictor


class PredictRequest(BaseModel):
    features: dict = Field(description="One full raw BAF application, including month")


class BatchRequest(BaseModel):
    records: list[dict] = Field(min_length=1, max_length=100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = FraudPredictor()
    yield


app = FastAPI(title="FraudShield API", version="0.1.0", lifespan=lifespan,
              description="Educational fraud-scoring demo on synthetic BAF data. Not for banking decisions.")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "predictor")}


@app.get("/model/info")
def model_info(request: Request):
    predictor = request.app.state.predictor
    return {"version": predictor.policy["model_version"],
            "algorithm": "CatBoostClassifier",
            "raw_features": predictor.schema["raw_features"],
            "feature_count": len(predictor.schema["model_features"]),
            "threshold": predictor.policy["threshold"],
            "warning": "Demo data only; month is a dataset-relative feature."}


@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    try:
        return request.app.state.predictor.predict([payload.features])[0]
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest, request: Request):
    try:
        return {"predictions": request.app.state.predictor.predict(payload.records)}
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
