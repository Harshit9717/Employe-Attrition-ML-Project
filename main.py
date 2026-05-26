from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.sparse import issparse

app = FastAPI()

from pydantic import BaseModel

class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    NumCompaniesWorked: int
    OverTime: int
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    TotalWorkingYears: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

# Load model + transformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "backend", "nn_model.keras"))
transform = joblib.load(os.path.join(BASE_DIR, "backend", "preprocessor.pkl"))
@app.get("/")
def home():
    return {"message": "Attrition API is running"}

@app.post("/predict")
def predict(data: EmployeeData):
    try:
        df = pd.DataFrame([data.dict()])
        df = df[transform.feature_names_in_]

        data_transformed = transform.transform(df)

        if issparse(data_transformed):
            data_transformed = data_transformed.toarray()

        prob = model.predict(data_transformed)[0][0]
        pred = int(prob > 0.5)

        return {
            "prediction": pred,
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}

print(transform.feature_names_in_)

