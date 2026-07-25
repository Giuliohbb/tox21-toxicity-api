FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Torch is the largest and least-frequently-changed dependency: install it in
# its own layer, from the CPU wheel index, before the rest of requirements.txt
# so it stays cached across changes to other dependencies.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies before copying application code, so
# code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY artifacts/ ./artifacts/

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
