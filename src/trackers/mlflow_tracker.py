import mlflow
import mlflow.sklearn

def log_experiment(model, X_train, y_train, X_test, y_test, acc, params):
    mlflow.set_experiment("Diabetes Prediction")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
