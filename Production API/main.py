from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import keras

# Load environment variables from .env file
load_dotenv()


# Define the request and response models
class PredictionRequest(BaseModel):
    input_data: list

class PredictionResponse(BaseModel):
    prediction: list

# Initialize FastAPI app
app = FastAPI()

# Load the TensorFlow model
model_path = os.getenv("MODEL_PATH", "model.h5")

# Load the model
try:
    model = keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the model from {model_path}: {e}")



# Define the predict endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert input data to numpy array
        input_array = np.array(request.input_data)
        # Ensure input shape matches the model's expected input
        if len(input_array.shape) == 1:
            input_array = np.expand_dims(input_array, axis=0)
        # Make prediction
        predictions = model.predict(input_array)
        # Convert predictions to list for JSON serialization
        predictions_list = predictions.tolist()
        return PredictionResponse(prediction=predictions_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))