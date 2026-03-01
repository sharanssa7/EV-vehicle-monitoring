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
    label = label_encoder.inverse_transform(prediction)[0]

    time_ms, voltage, current, temp, rpm, hall, est_soc, gt_soc, residual = data.features

    # ---------- BATTERY ----------
    battery_status = "HEALTHY"
    battery_issues = []

    if residual > 2:
        battery_status = "FAULT"
        battery_issues.append("High SOC mismatch")

    if voltage < 3.2:
        battery_status = "FAULT"
        battery_issues.append("Low voltage")

    # ---------- MOTOR ----------
    motor_status = "HEALTHY"
    motor_issues = []

    if rpm > 3000:
        motor_status = "WARNING"
        motor_issues.append("High motor speed")

    # ---------- INVERTER ----------
    inverter_status = "HEALTHY"
    inverter_issues = []

    if current > 5:
        inverter_status = "WARNING"
        inverter_issues.append("High current load")

    # ---------- THERMAL ----------
    thermal_status = "HEALTHY"
    thermal_issues = []

    if temp > 60:
        thermal_status = "FAULT"
        thermal_issues.append("Overheating detected")

    # ---------- BRAKES (Simulated logic) ----------
    brake_status = "NORMAL"
    brake_issues = []

    # example simulated condition
    if rpm > 3200:
        brake_status = "WARNING"
        brake_issues.append("Possible brake stress")

    return {
        "vehicle_id": "EV_01",
        "overall_status": label,

        "battery_system": {
            "status": battery_status,
            "issues": battery_issues if battery_issues else ["None"]
        },

        "motor_system": {
            "status": motor_status,
            "issues": motor_issues if motor_issues else ["None"]
        },

        "inverter_system": {
            "status": inverter_status,
            "issues": inverter_issues if inverter_issues else ["None"]
        },

        "thermal_system": {
            "status": thermal_status,
            "issues": thermal_issues if thermal_issues else ["None"]
        },

        "brake_system": {
            "status": brake_status,
            "issues": brake_issues if brake_issues else ["None"]
        }
    }