FROM python:3.11-slim

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Persist DB + shared files outside the image
ENV AGENTS_DB=/app/data/agents.db
RUN mkdir -p /app/data /app/backend/shared

EXPOSE 8000

# --proxy-headers so OAuth/HTTPS links are correct behind a reverse proxy
CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
