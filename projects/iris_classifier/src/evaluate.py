import json
import os

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

if __name__ == "__main__":
    print("=" * 60)
    print("Starting evaluation...")
    print("=" * 60)

    # Load model
    model_channel = os.environ.get("SM_CHANNEL_MODEL", "./model")
    model_path = os.path.join(model_channel, "model.joblib")
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Load test data
    X, y = load_iris(return_X_y=True)
    print(f"Loaded {len(X)} samples for evaluation")

    # Predict
    y_pred = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Create evaluation report
    report_dict = {
        "metrics": {
            "accuracy": {"value": accuracy},
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1_score": {"value": f1}
        }
    }

    # Save report
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./eval")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation.json")

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Evaluation report saved to: {report_path}")
    print("=" * 60)
