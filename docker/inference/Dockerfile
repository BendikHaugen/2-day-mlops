FROM python:3.9-slim

WORKDIR /opt/ml/code

COPY docker/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY projects/iris_classifier/src/inference.py .

ENV SAGEMAKER_PROGRAM inference.py

ENTRYPOINT ["python", "-m", "sagemaker_inference.default_inference_handler"]
