import json
import os
import sys

import joblib
import numpy as np


def model_fn(model_dir):
    """Load the model from the model directory.

    Args:
        model_dir: Path to the directory containing the model artifacts.
                  SageMaker passes this as /opt/ml/model

    Returns:
        Loaded sklearn model
    """
    print(f"Loading model from: {model_dir}")
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"Looking for model at: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded successfully: {type(model)}")
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        input_data = request_body.strip().split("\n")
        parsed_data = []
        for line in input_data:
            parsed_data.append([float(x) for x in line.split(",")])
        return np.array(parsed_data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions


def output_fn(predictions, accept):
    if accept == "text/csv":
        output = "\n".join(str(int(p)) for p in predictions)
        return output, accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
