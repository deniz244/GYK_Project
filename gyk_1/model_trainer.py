from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

class ModelTrainer:
    def __init__(self, df):
        self.df = df.copy()
        self.models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "RandomForest": RandomForestRegressor(random_state=42)
        }
        self.trained_models = {}
        self.results = []
        self.scaler = StandardScaler()

    def prepare_data(self):
        self.features = ['unit_price', 'quantity', 'discount', 'year', 'month', 'category_id']
        target = 'total_sales'

        X = self.df[self.features]
        y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Only for KNN scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_and_compare_models(self):
        for name, model in self.models.items():
            if name == "KNN":
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

            self.trained_models[name] = model
            r2 = r2_score(self.y_test, y_pred)
            rmse = sqrt(mean_squared_error(self.y_test, y_pred))
            self.results.append((name, r2, rmse))
            print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.2f}")

        self._visualize_results()
        return self.results

    def save_best_model_automatically(self, file_path="model.pkl", scaler_path="scaler.pkl"):
        best_model_name, best_r2, _ = max(self.results, key=lambda x: x[1])
        best_model = self.trained_models[best_model_name]

        joblib.dump(best_model, file_path)
        print(f"The best model saved as ({best_model_name}) R2: {best_r2:.4f} → '{file_path}'.")

        # Save scaler only for KNN
        if best_model_name == "KNN":
            joblib.dump(self.scaler, scaler_path)
            print(f"The scaler saved to '{scaler_path}'.")

    def _visualize_results(self):
        model_names = [r[0] for r in self.results]
        r2_scores = [r[1] for r in self.results]
        rmse_scores = [r[2] for r in self.results]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_title("Model Evaluation Metrics")
        ax1.set_xlabel("Models")
        ax1.set_ylabel("R2 Score", color="tab:blue")
        ax1.bar(model_names, r2_scores, color="tab:blue", alpha=0.6, label="R2 Score")
        ax1.tick_params(axis='y', labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("RMSE", color="tab:red")
        ax2.plot(model_names, rmse_scores, color="tab:red", marker='o', label="RMSE")
        ax2.tick_params(axis='y', labelcolor="tab:red")

        fig.tight_layout()
        plt.savefig("model_evaluation.png")
        print("Model evaluation visualization saved as 'model_evaluation.png'.")
        plt.show()
