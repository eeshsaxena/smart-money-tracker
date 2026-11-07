FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8050

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8050
# Refresh the data on start, then serve. If a source is down the refresh logs a
# warning and the app serves whatever data it already has.
CMD python -m smt refresh --months 6 || true; \
    exec gunicorn "smt.dashboard.app:server" --bind 0.0.0.0:$PORT --workers 2 --timeout 120
