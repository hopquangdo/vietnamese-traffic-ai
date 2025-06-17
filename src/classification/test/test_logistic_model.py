import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from src.utils import load_data


def test_model(model_path):
    # Load model và data
    model = joblib.load(model_path)
    X, y = load_data(path="../../../dataset/processed/dataset.npz")

    # Chia test cho công bằng
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Đánh giá
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {acc:.4f}")
    print("🔍 Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    model_path = "../../../models/logistic_model.pkl"
    test_model(model_path)


