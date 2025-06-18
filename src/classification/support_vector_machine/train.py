from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.utils import load_data


def train_support_vector_machine(X, y, model_output="../../../models/svm_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🧪 Accuracy: {acc:.4f}")
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Lưu mô hình
    joblib.dump(model, model_output)
    print(f"✅ Mô hình SVM đã được lưu tại: {model_output}")


if __name__ == "__main__":
    X, y = load_data(path="../../../dataset/processed/dataset.npz")
    train_support_vector_machine(X, y)
