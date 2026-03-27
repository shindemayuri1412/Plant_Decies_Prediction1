from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model_growth.pkl"))
crop_enc = joblib.load(os.path.join(BASE_DIR, "crop_encoder.pkl"))
soil_enc = joblib.load(os.path.join(BASE_DIR, "soil_encoder.pkl"))
stage_enc = joblib.load(os.path.join(BASE_DIR, "stage_encoder.pkl"))

@app.get("/")
def home():
    return {"message": "Crop Growth API Running"}

@app.post("/predict-growth")
def predict_growth(
    crop: str = Form(...),
    soil: str = Form(...),
    stage: str = Form(...),
    moi: float = Form(...),
    temp: float = Form(...),
    humidity: float = Form(...)
):
    try:
        crop = crop.strip().lower()
        soil = soil.strip().lower()
        stage = stage.strip().lower()

        # Check if input exists in encoder classes
        if crop not in crop_enc.classes_:
            return {"prediction": f"Error: Crop '{crop}' not in trained data!"}
        if soil not in soil_enc.classes_:
            return {"prediction": f"Error: Soil '{soil}' not in trained data!"}
        if stage not in stage_enc.classes_:
            return {"prediction": f"Error: Stage '{stage}' not in trained data!"}

        # Encode categorical features
        crop_val = crop_enc.transform([crop])[0]
        soil_val = soil_enc.transform([soil])[0]
        stage_val = stage_enc.transform([stage])[0]

        # Prepare input array
        data = np.array([[crop_val, soil_val, stage_val, moi, temp, humidity]])

        # Predict
        prediction = model.predict(data)[0]
        prediction = int(prediction)  # convert numpy int to native int

        # Map to human-readable text
        growth_map = {0: "Poor Growth", 1: "Moderate Growth", 2: "Good Growth"}
        prediction_text = growth_map.get(prediction, "Unknown")

        # Return both number and text
        return {"prediction": f"{prediction} = {prediction_text}"}

    except Exception as e:
        return {"prediction": f"Error: {str(e)}"}
