from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gyk_1.model_predictor import ModelPredictor
from gyk_1.data_loader import DataLoader
from gyk_1.featureEng import FeatureEngineer
from gyk_1.model_trainer import ModelTrainer


# maindeki loader değişkenini burada kullanıyoruz


# FastAPI uygulamasını başlat
app = FastAPI(
    title="Sales prediction API",
    description="Sales prediction API with Northwind database",
    version="1.0"
)

# Model yükle
predictor = ModelPredictor(model_path="model.pkl", scaler_path="scaler.pkl")
loader = DataLoader(user="postgres", password="new.pass3", host="localhost", db_name="mydatabase")


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
    return {"message": "API workss!! that Prediction of Sales"}


# Predict endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = request.model_dump()
        prediction = predictor.predict(input_data)
        return {"predicted_sale_value": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/products")
def get_products():
    try:
        df = loader.get_all_products()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sales_summary")
def get_sales_summary():
    try:
            df = loader.get_sales_summary()
            df["month"] = df["month"].astype(str)
            return df.to_dict(orient="records")
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain_model():
    try:
            # Veriyi yükle
        df = loader.load_data()
        # Feature engineering
        fe = FeatureEngineer(df)
        fe.process_date_features()
        fe.process_product_features()
        fe.process_customer_features()
        fe.handle_missing_values()
        final_df = fe.get_dataframe()

        # Eğitim
        trainer = ModelTrainer(final_df)
        trainer.prepare_data()
        trainer.train_and_compare_models()
        trainer.save_best_model_automatically(file_path="model.pkl", scaler_path="scaler.pkl")

        # Yeni model predictor'ı güncelle
        global predictor
        predictor = ModelPredictor(model_path="model.pkl", scaler_path="scaler.pkl")

        return {"message": "Model başarıyla yeniden eğitildi ve güncellendi."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

