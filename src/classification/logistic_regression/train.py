import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.utils import load_data


def train_logistic_regression_with_valid(
        X_train, y_train, X_valid, y_valid,
        model_output="../../../models/logistic_model.pkl"
):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Đánh giá trên tập train
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"🔵 Train Accuracy: {train_acc:.4f}")
    print("🔵 Train Classification Report:")
    print(classification_report(y_train, y_train_pred))

    # Đánh giá trên tập valid
    y_valid_pred = model.predict(X_valid)
    valid_acc = accuracy_score(y_valid, y_valid_pred)
    print(f"\n🟠 Valid Accuracy: {valid_acc:.4f}")
    print("🟠 Valid Classification Report:")
    print(classification_report(y_valid, y_valid_pred))

    X_all = np.concatenate([X_train, X_valid])
    y_all = np.concatenate([y_train, y_valid])

    final_model = LogisticRegression(max_iter=10000)
    final_model.fit(X_all, y_all)

    y_all_pred = final_model.predict(X_all)
    all_acc = accuracy_score(y_all, y_all_pred)
    print(f"\n🟠 All Accuracy: {all_acc:.4f}")
    print("🟠 All Classification Report:")
    print(classification_report(y_all, y_all_pred))

    joblib.dump(final_model, model_output)
    print(f"✅ Mô hình đã được lưu tại: {model_output}")


if __name__ == "__main__":
    X_train, y_train = load_data("../../../dataset/processed/train/dataset.npz")
    X_valid, y_valid = load_data("../../../dataset/processed/valid/dataset.npz")

    train_logistic_regression_with_valid(X_train, y_train, X_valid, y_valid)
