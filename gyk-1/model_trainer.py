# model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import joblib

class ModelTrainer:
    def __init__(self, df):
        self.df = df.copy()
        self.model = RandomForestRegressor(random_state=42)

    def prepare_data(self):
        # Feature ve Target belirle
        features = ['unit_price', 'quantity', 'discount', 'year', 'month', 'category_id']
        target = 'total_sales'

        X = self.df[features]
        y = self.df[target]

        # Eğitim ve test verisi bölme
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Veri eğitim ve test setlerine ayrıldı.")

    def train(self):
        # Model eğitimi
        self.model.fit(self.X_train, self.y_train)
        print("Model başarıyla eğitildi.")

    def evaluate(self):
        # Model değerlendirme
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")


    def save_model(self, path="model.pkl"):
        joblib.dump(self.model, path)
        print(f"Model kaydedildi: {path}")
