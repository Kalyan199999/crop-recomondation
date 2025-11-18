from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load(".\saved_model\svc.pkl") 
preprocessor = joblib.load(".\saved_model\ct.pkl")

@app.post("/predict")
def predict(data: dict):
    x = pd.DataFrame([data])
    x = preprocessor.transform(x)
    prediction = model.predict(x)
    print(prediction)
    return {"prediction": prediction[0]}
