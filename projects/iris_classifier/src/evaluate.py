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

    # For SageMaker processing jobs, the model is downloaded to /opt/ml/processing/model
    # The pipeline sets this via ProcessingInput destination
    model_channel = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/processing/model")
    
    print(f"Model channel: {model_channel}")
    print(f"Model channel exists: {os.path.exists(model_channel)}")
    
    # Debug: List what's actually in the directory
    if os.path.exists(model_channel):
        print(f"Contents of {model_channel}:")
        for item in os.listdir(model_channel):
            item_path = os.path.join(model_channel, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  - {item} ({size} bytes)")
            else:
                print(f"  - {item}/ (directory)")
    else:
        print(f"❌ Model channel does not exist!")
        # List parent directory
        parent = os.path.dirname(model_channel)
        if os.path.exists(parent):
            print(f"Contents of parent {parent}:")
            for item in os.listdir(parent):
                print(f"  - {item}")
    
    # Load model
    model_path = os.path.join(model_channel, "model.joblib")
    print(f"Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print("✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("Checking if model.tar.gz exists...")
        tar_path = os.path.join(model_channel, "model.tar.gz")
        if os.path.exists(tar_path):
            print("Found model.tar.gz - it wasn't extracted!")
            import tarfile
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=model_channel)
            print("✓ Extracted model.tar.gz")
            model = joblib.load(model_path)
        else:
            raise

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
            "accuracy": {"value": float(accuracy)},
            "precision": {"value": float(precision)},
            "recall": {"value": float(recall)},
            "f1_score": {"value": float(f1)}
        }
    }

    # Save report
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation.json")

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"✓ Evaluation report saved to: {report_path}")
    print("=" * 60)