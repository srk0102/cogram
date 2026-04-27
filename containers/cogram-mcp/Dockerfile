# Cogram MCP server: serves cogram tools over stdio AND HTTP/SSE on :7800/mcp
#
# Layout: the repo IS the cogram package (pyproject.toml + __init__.py + main.py
# + py.typed all sit at the same level). We copy the entire repo to /app/cogram/
# and use PYTHONPATH=/app so `import cogram` resolves to /app/cogram/__init__.py.
# Skipping `pip install -e .` because setuptools' editable install fights with
# package_dir = {"cogram": "."} when the repo root contains both __init__.py
# and other Python files at the same level.
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

# Install dependencies from requirements.txt (no editable package install needed)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/cogram/requirements.txt

# Make /app discoverable so `import cogram` finds /app/cogram/__init__.py
ENV PYTHONPATH=/app

EXPOSE 7800

# stdio mode is selected via CLI flag if needed
CMD ["python", "-m", "cogram.server.mcp", "--http", "--host", "0.0.0.0", "--port", "7800"]
