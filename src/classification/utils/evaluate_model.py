from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n📊 Evaluation on {dataset_name}")
    print(f"🎯 Accuracy: {acc:.4f}")
    print("📋 Classification Report:")
    print(classification_report(y, y_pred))
