FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app.py .
COPY templates ./templates
COPY model ./model
EXPOSE 8000
CMD ["python", "app.py"]
