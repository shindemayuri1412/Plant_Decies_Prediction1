from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --------------------------
# Load the trained model and encoders
# --------------------------
clf = joblib.load("model2_disease_predictor.pkl")
le_crop = joblib.load("le_crop.pkl")
le_soil = joblib.load("le_soil.pkl")
le_stage = joblib.load("le_stage.pkl")
le_result = joblib.load("le_result.pkl")

# --------------------------
# Initialize FastAPI
# --------------------------
app = FastAPI(
    title="Plant Disease Prediction API",
    description="Predicts disease for crops based on input features",
    version="1.0"
)

# --------------------------
# Define input schema
# --------------------------
class CropInput(BaseModel):
    crop_ID: str
    soil_type: str
    seedling_stage: str
    MOI: int
    temp: float
    humidity: float

# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict_disease(data: CropInput):
    try:
        # Encode categorical inputs using saved label encoders
        crop_enc = le_crop.transform([data.crop_ID])[0]
        soil_enc = le_soil.transform([data.soil_type])[0]
        stage_enc = le_stage.transform([data.seedling_stage])[0]

        # Prepare features for prediction
        features = np.array([[crop_enc, soil_enc, stage_enc, data.MOI, data.temp, data.humidity]])

        # Predict
        pred_enc = clf.predict(features)[0]

        # Convert back to original label
        pred_label = le_result.inverse_transform([pred_enc])[0]

        return {"predicted_disease": pred_label}

    except Exception as e:
        return {"error": str(e)}

# --------------------------
# Optional root endpoint
# --------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Plant Disease Prediction API!"}
