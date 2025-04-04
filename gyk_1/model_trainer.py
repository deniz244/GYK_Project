from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error
import joblib

class ModelTrainer:
    def __init__(self, df):
        self.df = df.copy()
        self.models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor()
        }
        self.trained_models = {}
        self.results = []
        self.scaler = StandardScaler()

    def prepare_data(self):
        self.features = ['unit_price', 'quantity', 'discount', 'year', 'month', 'category_id']
        target = 'total_sales'

        X = self.df[self.features]
        y = self.df[target]

        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sadece KNN için scale edilecek
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
            rmse = root_mean_squared_error(self.y_test, y_pred)
            self.results.append((name, r2, rmse))
            print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.2f}")
        return self.results

    def save_best_model_automatically(self, file_path="model.pkl", scaler_path="scaler.pkl"):
        best_model_name, best_r2, _ = max(self.results, key=lambda x: x[1])
        best_model = self.trained_models[best_model_name]

        joblib.dump(best_model, file_path)
        print(f"En iyi model ({best_model_name}) R2: {best_r2:.4f} → '{file_path}' olarak kaydedildi.")

        # Sadece KNN için scaler kaydedilir
        if best_model_name == "KNN":
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler da '{scaler_path}' dosyasına kaydedildi.")
