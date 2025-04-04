import pandas as pd
import joblib
import os

class ModelPredictor:
    def __init__(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler_if_needed(scaler_path)

    def load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            print(f"Model başarıyla yüklendi: {model_path}")
            return model
        except FileNotFoundError:
            raise Exception(f"Model dosyası bulunamadı: {model_path}")
        
    def load_scaler_if_needed(self, scaler_path):
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                print(f"Scaler yüklendi: {scaler_path}")
                return scaler
            except Exception as e:
                raise Exception(f"Scaler yüklenemedi: {e}")
        return None


    def predict(self, input_data: dict):
        """
        input_data örneği:
        {
            "unit_price": 15.0,
            "quantity": 20,
            "discount": 0.1,
            "year": 2022,
            "month": 4,
            "category_id": 5
        }
        """

        # Dataframe'e çevir
        df = pd.DataFrame([input_data])

        # Tahmin
        # Eğer scaler varsa (örneğin KNN için), input scale edilir
        if self.scaler:
            df_scaled = self.scaler.transform(df)
            prediction = self.model.predict(df_scaled)
        else:
            prediction = self.model.predict(df)

        return prediction[0]
