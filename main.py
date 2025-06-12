# This file is released into the public domain (CC0 1.0 Universal).
# https://creativecommons.org/publicdomain/zero/1.0/

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np
import tensorflow as tf
import joblib
from typing import List

# Force CPU usage untuk hindari error GPU di deployment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI(
    title="API Prediksi Saldo 7 Hari",
    description="API untuk memprediksi saldo 7 hari ke depan menggunakan model LSTM"
)

# Load model dengan error handling
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model/my_lstm_model.keras')
    scaler_path = os.path.join(os.path.dirname(__file__), 'model/scaler_y.pkl')
    model = tf.keras.models.load_model(model_path)
    scaler_y = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {str(e)}")

# Validasi input ketat
class InputData(BaseModel):
    fitur: List[conlist(float, min_length=4, max_length=4)]  # 14 hari x 4 fitur

@app.post("/predict", 
          response_model=dict,
          description="Prediksi saldo 7 hari ke depan",
          responses={
              400: {"description": "Input tidak valid"},
              500: {"description": "Internal server error"}
          })
def predict(data: InputData):
    try:
        input_array = np.array(data.fitur, dtype=np.float32)
        
        if input_array.shape != (14, 4):
            raise ValueError("Input harus berukuran 14x4 (14 hari, 4 fitur).")

        # Reshape untuk LSTM (batch_size, timesteps, features)
        input_array = input_array.reshape(1, 14, 4)

        # Prediksi
        pred_scaled = model.predict(input_array)
        pred_asli = scaler_y.inverse_transform(pred_scaled)

        return {
            "prediksi_saldo": [round(float(x), 2) for x in pred_asli[0]],  # Konversi ke float untuk JSON
            "status": "sukses"
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal: {str(e)}"
        )
