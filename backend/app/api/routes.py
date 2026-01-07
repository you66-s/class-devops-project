from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.request import PredictionRequest
from backend.app.services.storage import Storage
import logging

# Configuration d'un logger pour voir les erreurs côté serveur
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload_email/", status_code=status.HTTP_201_CREATED)
def create_prediction(pred: PredictionRequest):
    storage = Storage()
    try:
        result = storage.store_prediction(
            email=pred.email,
            label=pred.label,
            prediction_confiance=pred.prediction_confiance
        )
        
        return {"message": "Prediction stored successfully"} 

    except ConnectionError:
        logger.error("Database connection failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is currently unavailable"
        )
        
    except Exception as e:
        logger.error(f"Error storing prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store prediction in database"
        )