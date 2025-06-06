import shap
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import load
import pandas as pd

def shap_summary(model_path, X_path):
    model = load(model_path)
    X = pd.read_csv(X_path)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    shap_summary("outputs/xgb_model.joblib", "outputs/X_train.csv")
