from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
from src.classification.utils import load_data


def train_knn(
        X_train, y_train,
        model_output="../../../models/knn_model.pkl"
):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    print("✅ Accuracy:", accuracy_score(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    joblib.dump(knn, model_output)


if __name__ == "__main__":
    X_train, y_train = load_data(
        "../../../dataset/processed/train/dataset.npz"
    )
    X_valid, y_valid = load_data(
        "../../../dataset/processed/valid/dataset.npz"
    )
    train_knn(
        np.concatenate((X_train, X_valid)),
        np.concatenate((y_train, y_valid)),
    )
