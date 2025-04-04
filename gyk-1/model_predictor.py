import pandas as pd
import joblib

class ModelPredictor:
    def __init__(self, model_path="model.pkl"):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            print(f"Model başarıyla yüklendi: {model_path}")
            return model
        except FileNotFoundError:
            raise Exception(f"Model dosyası bulunamadı: {model_path}")

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
        prediction = self.model.predict(df)

        return prediction[0]
