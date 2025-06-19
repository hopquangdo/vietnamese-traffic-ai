from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
from src.classification.utils import load_data


def train_support_vector_machine(
        train_path, valid_path, model_output="../../../models/svm_model.pkl"
):
    X_train, y_train = load_data(train_path)
    X_valid, y_valid = load_data(valid_path)

    model = SVC(C=10)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"🔵 Train Accuracy: {train_acc:.4f}")
    print("🔵 Train Classification Report:")
    print(classification_report(y_train, y_train_pred))

    y_pred = model.predict(X_valid)
    valid_acc = accuracy_score(y_valid, y_pred)
    print(f"🧪 Valid Accuracy: {valid_acc:.4f}")
    print("📋 Valid Classification Report:")
    print(classification_report(y_valid, y_pred))

    X_all = np.concatenate([X_train, X_valid])
    y_all = np.concatenate([y_train, y_valid])

    final_model = SVC(C=10)
    final_model.fit(X_all, y_all)

    final_y_pred = final_model.predict(X_all)
    final_acc = accuracy_score(y_all, final_y_pred)
    print(f"🧪 Valid Accuracy: {final_acc:.4f}")
    print("📋 Valid Classification Report:")
    print(classification_report(y_all, final_y_pred))

    joblib.dump(model, model_output)


if __name__ == "__main__":
    train_support_vector_machine(
        train_path="../../../dataset/processed/train/dataset.npz",
        valid_path="../../../dataset/processed/valid/dataset.npz"
    )
