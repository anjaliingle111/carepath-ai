from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", acc)
    print("Classification Report:\n", report)
