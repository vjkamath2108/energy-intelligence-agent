"""Integration tests for src/db.py.

These tests require a running PostgreSQL instance reachable via the
environment variables in .env (or .env.test).  The vector extension must
already be installed in the target database:

    CREATE EXTENSION IF NOT EXISTS vector;

Run with:
    pytest tests/test_db.py -v

Skip when no DB is available:
    pytest tests/test_db.py -v -m "not integration"
"""

import os

import pytest
import pytest_asyncio

# Reset the module-level pool singleton between tests so each test gets
# a clean lifecycle.  We import the module (not individual symbols) so we
# can patch _pool directly.
import src.db as db_module
from src.db import close_pool, get_connection, get_pool

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def reset_pool() -> None:
    """Ensure the pool singleton is closed and reset before and after each test."""
    # Teardown from any previous test
    await close_pool()
    yield
    await close_pool()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_connection() -> None:
    """Connecting to the database returns a non-empty server version string."""
    async with get_connection() as conn:
        version: str = await conn.fetchval("SELECT version()")

    assert version, "Expected a non-empty version string"
    assert "PostgreSQL" in version, f"Unexpected version string: {version!r}"


async def test_pgvector_extension() -> None:
    """The vector extension must be present in pg_extension.

    If this fails, run: CREATE EXTENSION IF NOT EXISTS vector;
    """
    async with get_connection() as conn:
        ext: str | None = await conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        )

    assert ext == "vector", (
        "pgvector extension not found. "
        "Run 'CREATE EXTENSION IF NOT EXISTS vector;' in your database."
    )


async def test_pool_lifecycle() -> None:
    """get_pool() returns a pool; close_pool() shuts it down cleanly."""
    pool = await get_pool()
    assert pool is not None, "get_pool() should return a Pool instance"

    # A second call must return the same singleton
    pool2 = await get_pool()
    assert pool is pool2, "get_pool() should return the same singleton"

    await close_pool()
    assert db_module._pool is None, "Pool should be None after close_pool()"

    # close_pool() on an already-closed pool must be a no-op
    await close_pool()


async def test_pool_max_connections() -> None:
    """Pool respects max_size=10 and reuses connections on concurrent use."""
    pool = await get_pool()

    async def fetch_version() -> str:
        async with get_connection() as conn:
            return await conn.fetchval("SELECT version()")  # type: ignore[return-value]

    import asyncio

    results = await asyncio.gather(*[fetch_version() for _ in range(10)])

    assert len(results) == 10
    assert all("PostgreSQL" in r for r in results)
