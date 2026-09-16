# XPERTENTiSCH — Produktionsabbild.
# Baukontext ist dieses Verzeichnis: docker build -t xpertentisch xpertentisch/

# --- Stufe 1: Oberfläche bauen -------------------------------------------
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Der Build prüft anschließend selbst, dass keine Zugangsdaten im Bundle stehen.
RUN npm run build

# --- Stufe 2: Laufzeit ----------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XT_ENV=production \
    XT_DB_PATH=/data/xpertentisch.sqlite3

WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY --from=ui /ui/dist ./frontend/dist

RUN mkdir -p /data && useradd -m -u 10001 tisch && chown -R tisch /data /app
USER tisch

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
