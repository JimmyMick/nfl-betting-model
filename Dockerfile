# NFL betting model + graph ingestion image.
FROM python:3.12-slim

# Bring in the uv binary (fast, reproducible installs from uv.lock).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (cached unless the lockfile changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the project and finish the install.
COPY . .
RUN uv sync --frozen --no-dev

# Default to graph ingestion; override for the model, e.g.
#   docker compose run --rm app uv run main.py --train 2010-2022 --test 2023
CMD ["uv", "run", "ingest_graph.py", "--seasons", "2022-2023"]
