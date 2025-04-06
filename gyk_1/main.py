from data_loader import DataLoader
from featureEng import FeatureEngineer
from model_trainer import ModelTrainer
from model_predictor import ModelPredictor


def main():

    # 1. Loading Data
    loader = DataLoader(user="postgres", password="new.pass3", host="localhost", db_name="mydatabase")
    df = loader.load_data()

    # 2. Feature Engineering (Preprocessing)
    fe = FeatureEngineer(df)

    # 3. Prepared of Data
    final_df = fe.get_dataframe()
    
    # final_df.to_csv("current_data.csv", inkdex=False)
    print(final_df.head(5))

    # 4. Model Learning
    trainer = ModelTrainer(final_df)
    trainer.prepare_data()
    trainer.train_and_compare_models()
    trainer.save_best_model_automatically(file_path="model.pkl")  # Automatic choose and register

    #5. Load Model and Predict
    predictor = ModelPredictor()

    # Predict Sample
    sample_input = {
        "unit_price": 20.0,
        "quantity": 10,
        "discount": 0.05,
        "year": 2022,
        "month": 3,
        "category_id": 2
    }

    prediction = predictor.predict(sample_input)
    print(f"Sales of predict: {prediction:.2f}")
    
    return 0

if __name__ == "__main__":
    main()