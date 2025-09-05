from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Wine Quality Prediction API")

# Mount static files and templates
# This assumes you have an `index.html` file
templates = Jinja2Templates(directory=".")

# --- Model and Scaler Loading ---
MODEL_PATH = 'wine_model.joblib'
SCALER_PATH = 'scaler.joblib'

model = None
scaler = None

@app.on_event("startup")
def load_model_assets():
    """Load the model and scaler at application startup."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading model assets: {e}")
            # In a real app, you might want to prevent startup
    else:
        print("Warning: Model or scaler file not found. The /predict endpoint will not work.")
        print("Please run train.py to generate model assets.")


# --- Pydantic model for input data validation ---
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_quality(
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
    alcohol: float = Form(...)
):
    """
    Predicts the quality of wine based on input features.
    Accepts form data from the web UI.
    """
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Please train the model first."}

    try:
        # Create a pandas DataFrame from the input features
        features = pd.DataFrame([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ]], columns=[
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)

        # Convert prediction to a human-readable label
        quality = "Good" if prediction[0] == 1 else "Not Good"
        confidence = f"{np.max(probability) * 100:.2f}%"
        
        return {
            "prediction": quality,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}


# To run this app: `uvicorn main:app --reload`
