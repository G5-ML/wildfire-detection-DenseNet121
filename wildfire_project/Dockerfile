FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WILDFIRE_CONFIG=/app/params.yaml

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[serve]"
COPY params.yaml ./params.yaml

EXPOSE 8000
CMD ["uvicorn", "wildfire_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]

