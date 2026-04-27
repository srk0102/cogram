# Cogram MCP server: serves cogram tools over stdio AND HTTP/SSE on :7800/mcp
#
# Layout: the repo root IS the cogram package. pyproject.toml + __init__.py
# + cogram.py + py.typed all sit at the same level. Build context is the repo
# root. Setuptools' package_dir = {"cogram": "."} maps the package "cogram"
# to that directory.
FROM python:3.12-slim

WORKDIR /app

# System packages (build deps for asyncpg, uvloop, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire repo (which IS the cogram package) into /app/cogram/
COPY . /app/cogram/
WORKDIR /app/cogram

# Single editable install — package "cogram" maps to /app/cogram per pyproject
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 7800

# stdio mode is selected via CLI flag if needed
CMD ["python", "-m", "cogram.server.mcp", "--http", "--host", "0.0.0.0", "--port", "7800"]
