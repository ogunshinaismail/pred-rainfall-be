# from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model and PCA object
MODEL_PATH = "models/rainfall_prediction_model.h5"
PCA_PATH = "models/pca.pkl"

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)
    print(f"PCA object loaded: {pca}")

# Request schema for prediction input


class PredictionInput(BaseModel):
    features: list[float]

# Define prediction function


def predict_rainfall(input_features):
    print(f"Input features: {input_features}")  # Debugging line
    input_features_pca = pca.transform([input_features])
    print(f"Input features after PCA: {input_features_pca}")  # Debugging line
    input_reshaped = np.reshape(
        input_features_pca, (1, input_features_pca.shape[1], 1))
    prediction = model.predict(input_reshaped)
    return prediction[0][0]

# API endpoint for prediction


@app.post("/predict/")
def get_prediction(data: PredictionInput):
    try:
        # Test with hardcoded data
        test_input = [0.25, 0.5, 0.75, 1.0, 1.25]
        result = predict_rainfall(test_input)
        return {"prediction": result}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
