from fastapi import FastAPI
import pickle
import pandas as pd
import os

app = FastAPI()

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
model = pickle.load(open(f'models/best_churn_model_{MODEL_VERSION}.pkl', 'rb'))
preprocessor = pickle.load(
    open(f'models/preprocessor_{MODEL_VERSION}.pkl', 'rb'))
feature_names = pickle.load(
    open(f'models/feature_names_{MODEL_VERSION}.pkl', 'rb'))


@app.post('/predict')
def predict(features: dict):
    X = pd.DataFrame([features], columns=feature_names)
    X_proc = preprocessor.transform(X)
    prob = model.predict_proba(X_proc)[0, 1]
    return {'churn_probability': float(prob)}
