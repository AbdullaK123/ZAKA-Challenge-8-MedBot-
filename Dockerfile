# --- Build stage: install deps ---
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user
RUN useradd --create-home --no-log-init --shell /usr/sbin/nologin appuser
RUN mkdir -p logs && chown appuser:appuser logs

USER appuser

# Copy app code
COPY --chown=appuser:appuser . .

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]