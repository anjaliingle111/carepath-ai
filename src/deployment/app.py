from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("outputs/xgb_model.joblib")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
