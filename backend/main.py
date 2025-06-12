from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

# Inisialisasi app FastAPI
app = FastAPI(title="API Prediksi Saldo 7 Hari")

# Load model dan scaler saat startup
model = tf.keras.models.load_model('model/my_lstm_model.keras')
scaler_y = joblib.load('model/scaler_y.pkl')

# Struktur input
class InputData(BaseModel):
    fitur: list[list[float]]  # input 2D: 14 hari x jumlah fitur

# Endpoint prediksi
@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array(data.fitur)
        if input_array.shape != (14, 4):
            raise ValueError("Input harus berukuran 14x4 (14 hari, 4 fitur).")

        input_array = input_array.reshape(1, 14, 4)

        # Prediksi
        pred_scaled = model.predict(input_array)
        pred_asli = scaler_y.inverse_transform(pred_scaled)

        # Return hasil sebagai list
        return {"prediksi_saldo": [int(x) for x in pred_asli[0]]}

    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))