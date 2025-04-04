from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gyk_1.model_predictor import ModelPredictor
import pandas as pd

# FastAPI uygulamasını başlat
app = FastAPI(
    title="Satış Tahmin API",
    description="Northwind verisi ile ürün bazlı satış tahmin API",
    version="1.0"
)

# Model yükle
predictor = ModelPredictor(model_path="model.pkl")


# API üzerinden beklenen veri yapısı
class PredictionRequest(BaseModel):
    unit_price: float
    quantity: int
    discount: float
    year: int
    month: int
    category_id: int


# Root endpoint
@app.get("/")
def root():
    return {"message": "Satış Tahmin API çalışıyor!"}


# Predict endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = request.model_dump()
        prediction = predictor.predict(input_data)
        return {"tahmin_edilen_satis": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
