from pydantic import BaseModel

class PredictionRequest(BaseModel):
    email: str
    label: int
    prediction_confiance: float