import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.utils import load_data


def train_logistic_regression(X, y, model_output="../../models/logistic_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"🔵 Train Accuracy: {train_acc:.4f}")
    print("🔵 Train Classification Report:")
    print(classification_report(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\n🟢 Test Accuracy: {test_acc:.4f}")
    print("🟢 Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    joblib.dump(model, model_output)
    print(f"✅ Mô hình đã được lưu tại: {model_output}")


if __name__ == "__main__":
    X, y = load_data()
    train_logistic_regression(X, y)
