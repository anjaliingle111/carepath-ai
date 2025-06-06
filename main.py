import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from joblib import dump
from src.trackers.mlflow_tracker import log_experiment
from src.monitoring.shap_interpretation import shap_summary
import optuna
import mlflow

def load_data():
    df = pd.read_csv("data/processed.csv")
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    X_train, X_test, y_train, y_test = load_data()

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds)

def main():
    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    best_params.update({"use_label_encoder": False, "eval_metric": "logloss"})

    print("Best parameters:", best_params)

    # Train final model
    X_train, X_test, y_train, y_test = load_data()
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    dump(model, "outputs/xgb_model.joblib")
    pd.DataFrame(X_train).to_csv("outputs/X_train.csv", index=False)

    # Log to MLflow
    log_experiment(model, X_train, y_train, X_test, y_test, acc, best_params)

    # SHAP Interpretability
    shap_summary("outputs/xgb_model.joblib", "outputs/X_train.csv")

if __name__ == "__main__":
    main()
