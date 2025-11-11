import os

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Load data
    X, y = load_iris(return_X_y=True)
    print(f"Loaded {len(X)} samples")

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    print("Model training complete")

    # Save the model
    model_dir = os.environ.get("SM_MODEL_DIR", "./model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    print("=" * 60)
