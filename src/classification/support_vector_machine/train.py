from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.utils import load_data


def train_support_vector_machine(
    train_path, valid_path, model_output="../../../models/svm_model.pkl"
):
    # Load dữ liệu huấn luyện và validation
    X_train, y_train = load_data(train_path)
    X_valid, y_valid = load_data(valid_path)

    # Khởi tạo và huấn luyện mô hình SVM
    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train, y_train)

    # Đánh giá trên tập validation
    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    print(f"🧪 Accuracy on Validation Set: {acc:.4f}")
    print("📋 Classification Report:")
    print(classification_report(y_valid, y_pred))

    # Lưu mô hình
    joblib.dump(model, model_output)
    print(f"✅ Mô hình SVM đã được lưu tại: {model_output}")


if __name__ == "__main__":
    train_support_vector_machine(
        train_path="../../../dataset/processed/train/dataset.npz",
        valid_path="../../../dataset/processed/valid/dataset.npz"
    )
