# src/evaluate.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs metrics to MLflow."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
