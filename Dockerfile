# DATect Domoic Acid Forecasting System Docker Container
# =====================================================
# 
# This Dockerfile creates a reproducible environment for the DATect
# forecasting system, ensuring consistent results across different
# computing environments for research and production deployment.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATECT_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for organized data structure
RUN mkdir -p /app/data/raw /app/data/intermediate /app/data/processed \
             /app/outputs/logs /app/outputs/reports /app/outputs/plots \
             /app/analysis /app/tools

# Expose ports for dashboards
EXPOSE 8065 8071

# Create non-root user for security
RUN groupadd -r datect && useradd -r -g datect datect
RUN chown -R datect:datect /app
USER datect

# Health check - test core forecasting components
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import forecasting.core.forecast_engine; import forecasting.core.model_factory; print('OK')" || exit 1

# Default command - run main forecasting pipeline
CMD ["python", "modular-forecast.py"]