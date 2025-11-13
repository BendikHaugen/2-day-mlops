import json
import os
import sys

import joblib
import numpy as np


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)


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
