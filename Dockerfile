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

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs /app/scientific_evidence

# Expose ports for dashboards
EXPOSE 8065 8071

# Create non-root user for security
RUN groupadd -r datect && useradd -r -g datect datect
RUN chown -R datect:datect /app
USER datect

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import forecasting.core.forecast_engine; print('OK')" || exit 1

# Default command - run scientific validation
CMD ["python", "run_scientific_validation.py", "--tests", "all", "--verbose"]