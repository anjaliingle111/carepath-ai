# src/pipeline.py

import pandas as pd
import joblib
import os
import mlflow
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def run_pipeline():
    mlflow.set_experiment("diabetes_readmission")

    with mlflow.start_run():
        # Load raw data
        raw_path = os.path.join("data", "raw", "diabetic_data.csv")
        df = pd.read_csv(raw_path)

        # Preprocess
        X, y = preprocess_data(df)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Balance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Train
        model = train_model(X_train_res, y_train_res)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/xgboost_model.pkl")
        mlflow.log_artifact("models/xgboost_model.pkl")

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        return metrics
