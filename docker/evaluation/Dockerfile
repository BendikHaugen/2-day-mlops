FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /opt/ml/processing

COPY docker/evaluation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY projects/iris_classifier/src/evaluate.py /opt/ml/processing/input/code/

ENTRYPOINT ["python", "/opt/ml/processing/input/code/evaluate.py"]