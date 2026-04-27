# Cogram dashboard: FastAPI app + force-graph viz at http://localhost:7801/
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dashboard deps (FastAPI + Neo4j + Redis + asyncpg + httpx for MCP probes)
RUN pip install --upgrade pip && pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    "neo4j>=5.26" \
    "redis>=5.1" \
    "asyncpg>=0.30" \
    "httpx>=0.27" \
    "jinja2>=3.1" \
    "websockets>=13"

COPY containers/cogram-dashboard/ ./

EXPOSE 7801

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7801"]
