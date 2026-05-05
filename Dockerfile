FROM python:3.12-slim
WORKDIR /app
COPY der-tisch-backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY der-tisch-backend/ .
CMD uvicorn api_server:app --host 0.0.0.0 --port $PORT
