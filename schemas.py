from pydantic import BaseModel
from typing import Literal

class ModelInput(BaseModel):
    age: int
    ever_married: Literal["Yes", "No"]
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]
    heart_disease: Literal[0, 1]
    hypertension: Literal[0, 1]

class PredictionResponse(BaseModel):
    predicted_probability: float
    predicted_class: Literal["Stroke", "No Stroke"]