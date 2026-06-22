# ── Stage 1: build ──────────────────────────────────────────────────────────
# Install all Python deps into /root/.local so the final stage can copy them
# without build tooling (gcc needed to compile asyncpg C extensions).
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user pytest-cov


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

# Non-root user for principle of least privilege
RUN groupadd --gid 1001 eia \
    && useradd --uid 1001 --gid eia --no-create-home eia

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/eia/.local

ENV PYTHONPATH=/app \
    PATH=/home/eia/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source
COPY src/ ./src/
COPY stream1_log_anomaly/ ./stream1_log_anomaly/
COPY examples/ ./examples/
COPY evaluation/ ./evaluation/
COPY tests/ ./tests/
COPY mypy.ini pytest.ini .coveragerc ./

RUN chown -R eia:eia /app
USER eia

# Default: run the EIA RAG demo (no AWS/DB required)
CMD ["python", "examples/demo_queries.py"]
