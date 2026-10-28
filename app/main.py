from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Load the trained model artifact
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]


# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model",
    version="1.0.0"
)


# Define request schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    customer_data = customer.model_dump()

    # Convert input to a DataFrame with the training feature order
    input_df = pd.DataFrame(
        [customer_data],
        columns=FEATURE_NAMES
    )

    # Predict churn class and probability
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "prediction": prediction,
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4)
    }