import time
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.classification.utils import load_data
import os

def evaluate_model(model_path, data_path, label=""):
    """
    Evaluate a scikit-learn model on a dataset, with runtime measurement.

    Args:
        model_path (str): Path to the saved model (.pkl file).
        data_path (str): Path to the dataset file (.npz).
        label (str): Optional name/label for the model (for display).
    """
    print(f"\nEvaluating model: {label or model_path}")

    # Load model and data
    model = joblib.load(model_path)
    X, y = load_data(data_path)

    # Start timing
    start_time = time.time()

    # Predict
    y_pred = model.predict(X)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Evaluate
    acc = accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="macro", zero_division=0)
    recall = recall_score(y, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print("⏱️ Inference time: {:.4f} seconds".format(elapsed_time))
    print("📋 Classification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    csv_path = "./report/metrics_summary.csv"
    os.makedirs("./report", exist_ok=True)

    # Kiểm tra file đã tồn tại chưa
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a") as f:
        if write_header:
            f.write("Model,Accuracy,Precision,Recall,F1,InferenceTime\n")
        f.write(f"{label},{acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{elapsed_time:.4f}\n")
