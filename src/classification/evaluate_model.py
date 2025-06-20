from src.classification.utils import evaluate_model
import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # data_test_path = "../../dataset/processed/test/dataset.npz"
    # models = [
    #     ("Decision Tree", "../../models/decision_tree_model.pkl"),
    #     ("K-Nearest Neighbors", "../../models/knn_model.pkl"),
    #     ("Logistic Regression", "../../models/logistic_model.pkl"),
    #     ("Support Vector Machine", "../../models/svm_model.pkl")
    # ]
    #
    # for label, model_path in models:
    #     evaluate_model(model_path=model_path, data_path=data_test_path, label=label)

    df = pd.read_csv("./report/metrics_summary.csv")

    plt.figure(figsize=(10, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    for metric in metrics:
        plt.plot(df["Model"], df[metric], marker='o', label=metric)

    os.makedirs("./report/images", exist_ok=True)

    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("So sánh hiệu suất các mô hình ML")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./report/images/model_performance.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(df["Model"], df["InferenceTime"], color='orange')
    plt.xlabel("Model")
    plt.ylabel("Thời gian suy luận (giây)")
    plt.title("So sánh tốc độ suy luận của các mô hình")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./report/images/inference_time.png")
    plt.close()