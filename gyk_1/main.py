from data_loader import DataLoader
from featureEng import FeatureEngineer
from model_trainer import ModelTrainer
from model_predictor import ModelPredictor


# 1. Veri Yükleme
loader = DataLoader(user="postgres", password="1234", host="localhost", db_name="GYK1Northwind")
df = loader.load_data()

# 2. Feature Engineering
fe = FeatureEngineer(df)
fe.process_date_features()
fe.process_product_features()
fe.process_customer_features()
fe.handle_missing_values()

# 3. Hazır Veri
final_df = fe.get_dataframe()
print(final_df.head())

# 4. Model Eğitimi
trainer = ModelTrainer(final_df)
trainer.prepare_data()
trainer.train()
trainer.evaluate()
trainer.save_model()

#5. Model Yükleme ve Tahmin
# Model yükle
predictor = ModelPredictor()

# Örnek tahmin
sample_input = {
    "unit_price": 20.0,
    "quantity": 10,
    "discount": 0.05,
    "year": 2022,
    "month": 3,
    "category_id": 2
}

prediction = predictor.predict(sample_input)
print(f"Tahmin edilen satış: {prediction:.2f}")