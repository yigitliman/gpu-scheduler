FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN useradd -m -u 1000 appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 8000

# Shell form so ${PORT} expands: PaaS hosts inject PORT, local and compose do not.
CMD uvicorn src.serve:app --host 0.0.0.0 --port ${PORT:-8000}
