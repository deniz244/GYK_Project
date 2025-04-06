# Sales Forecasting API (Northwind Dataset)

This project provides a machine learning-powered REST API built with FastAPI to predict product-level sales using the Northwind database.

## Project Overview

- Train regression models (KNN, Decision Tree, Linear Regression, Random Forest)
- Compare performance using R² and RMSE
- Expose model through a REST API for real-time predictions
- Automatically document endpoints via Swagger UI

## 🛠️ Technologies Used

- Python 3.x
- FastAPI
- PostgreSQL
- SQLAlchemy
- Scikit-learn
- Pandas, NumPy
- Swagger UI (OpenAPI)
- Pydantic
- Uvicorn

## 📦 Endpoints

| Method | Endpoint         | Description                          |
| ------ | ---------------- | ------------------------------------ |
| GET    | `/`              | Check API health                     |
| POST   | `/predict`       | Get sales prediction from input data |
| POST   | `/retrain`       | Retrain model with latest data       |
| GET    | `/products`      | Get product list from DB             |
| GET    | `/sales_summary` | Get monthly sales summary            |

## 📈 Sample Input for `/predict`

```json
{
  "unit_price": 18.0,
  "quantity": 10,
  "discount": 0.1,
  "year": 1997,
  "month": 3,
  "category_id": 4
}
```

## ⚙️ Running the Project

1. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the API server:

   ```bash
   uvicorn gyk_1.api_server:app --reload
   ```

3. Open Swagger UI:
   - Navigate to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 📁 Project Structure

```
GYK_Project/
│
├── gyk_1/
│   ├── data_loader.py
│   ├── featureEng.py
│   ├── model_trainer.py
│   ├── model_predictor.py
│   └── api_server.py
├── .gitignore
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```









