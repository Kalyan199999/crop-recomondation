from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "saved_model", "svc.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "saved_model", "ct.pkl"))


@app.post("/predict")
def predict(data: dict):
    x = pd.DataFrame([data])
    x = preprocessor.transform(x)
    prediction = model.predict(x)
    print(prediction)
    return {"prediction": prediction[0]}
