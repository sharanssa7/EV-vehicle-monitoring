from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load models
lstm_model = load_model("models/lstm_autoencoder.keras")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")

xgb_model = joblib.load("models/fault_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

sequence_length = 30


# -------------------------
# LSTM ANOMALY DETECTION
# -------------------------

class SequenceInput(BaseModel):
    sequence: list


@app.post("/predict-anomaly")
def predict_anomaly(data: SequenceInput):
    new_sequence = np.array(data.sequence)

    scaled = scaler.transform(new_sequence)
    scaled = np.expand_dims(scaled, axis=0)

    reconstructed = lstm_model.predict(scaled)
    mse = np.mean((scaled - reconstructed) ** 2)

    if mse <= threshold:
        return {"status": "Normal", "mse": float(mse)}
    else:
        return {"status": "Abnormal", "mse": float(mse)}


# -------------------------
# XGBOOST FAULT CLASSIFIER
# -------------------------

class FaultInput(BaseModel):
    features: list

@app.post("/predict-fault")
def predict_fault(data: FaultInput):
    features = np.array(data.features).reshape(1, -1)

    prediction = xgb_model.predict(features)
    label = label_encoder.inverse_transform(prediction)

    # 👇 ADD THESE TWO LINES HERE
    print("Raw prediction:", prediction)
    print("Label:", label[0])

    return {"fault_type": label[0]}