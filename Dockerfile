# Multi-stage build: build React frontend, then serve via FastAPI

FROM node:18-alpine AS ui
WORKDIR /app/frontend
ENV NODE_OPTIONS=--max-old-space-size=3072
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ .
RUN npm run build

FROM python:3.10-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy built frontend into place
COPY --from=ui /app/frontend/dist ./frontend/dist

# Environment
ENV PORT=8000 \
    DATECT_ENV=production

# Expose port
EXPOSE 8000

# Healthcheck (FastAPI /health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD sh -c 'curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1'

# Run FastAPI (serves API and static frontend) honoring Cloud Run's $PORT
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

