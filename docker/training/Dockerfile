FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /opt/ml/code

COPY docker/training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY projects/iris_classifier/src/train.py .

ENTRYPOINT ["python", "train.py"]