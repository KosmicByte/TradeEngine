# Dockerfile for Ultra-Low Latency Trading Engine
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir     torch     onnxruntime-gpu     redis     numpy     pandas     shap     matplotlib

# Expose ports: 5000 app, 8000 metrics
EXPOSE 8000 5000

CMD ["python", "integration_sample.py"]
