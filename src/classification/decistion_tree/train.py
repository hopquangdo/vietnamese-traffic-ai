from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
from src.classification.utils import load_data


def train_decision_tree(
        X_train, y_train,
        model_output="../../../models/decision_tree_model.pkl",
):
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42,
    )
    dt.fit(X_train, y_train)
    y_train_pred = dt.predict(X_train)
    print("✅ Accuracy:", accuracy_score(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    joblib.dump(dt, model_output)


if __name__ == "__main__":
    X_train, y_train = load_data(
        "../../../dataset/processed/train/dataset.npz"
    )
    X_valid, y_valid = load_data(
        "../../../dataset/processed/valid/dataset.npz"
    )
    train_decision_tree(
        np.concatenate((X_train, X_valid)),
        np.concatenate((y_train, y_valid)),
    )