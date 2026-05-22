"""Async PostgreSQL connection pool with pgvector support.

Builds a singleton asyncpg pool from environment variables and exposes
a context manager for acquiring connections.
"""

import asyncio
import logging
import os
import ssl
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Connection, Pool
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

load_dotenv()

logger = logging.getLogger(__name__)

_pool: Pool | None = None

_DSN_TEMPLATE = "postgresql://{user}:{password}@{host}:{port}/{dbname}"


def _build_dsn() -> str:
    """Build a PostgreSQL DSN from environment variables.

    Returns:
        Connection string in postgresql://user:password@host:port/dbname format.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    required = {"DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"}
    missing = required - set(os.environ)
    if missing:
        raise ValueError(f"Missing required environment variables: {sorted(missing)}")

    return _DSN_TEMPLATE.format(
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ["DB_NAME"],
    )


def _build_ssl_context() -> ssl.SSLContext | bool:
    """Return an SSL context based on DB_SSLMODE.

    sslmode=require  -> ssl.SSLContext with no cert verification (dev tunnel)
    sslmode=disable  -> False (plain TCP)
    anything else    -> ssl.SSLContext with no cert verification

    Returns:
        SSL context or False when SSL is disabled.
    """
    sslmode = os.environ.get("DB_SSLMODE", "require").lower()
    if sslmode == "disable":
        return False
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _on_connect(conn: Connection) -> None:
    """Register pgvector codec on every new connection in the pool."""
    await register_vector(conn)
    logger.debug("pgvector registered on connection %s", conn.get_server_pid())


async def get_pool() -> Pool:
    """Return the singleton asyncpg connection pool, creating it if needed.

    The pool is initialised with:
    - min_size=2, max_size=10
    - pgvector registered on every connection via the init callback
    - SSL from DB_SSLMODE (default: require, no cert verification)

    Returns:
        The active asyncpg Pool instance.
    """
    global _pool  # noqa: PLW0603
    if _pool is not None:
        return _pool

    dsn = _build_dsn()
    ssl_ctx = _build_ssl_context()

    logger.info(
        "Creating asyncpg pool -- host=%s db=%s min=2 max=10",
        os.environ.get("DB_HOST"),
        os.environ.get("DB_NAME"),
    )

    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=10,
        ssl=ssl_ctx,
        init=_on_connect,
    )

    logger.info("Connection pool ready")
    return _pool


async def close_pool() -> None:
    """Gracefully close the singleton connection pool.

    Safe to call when the pool has not been initialised (no-op).
    """
    global _pool  # noqa: PLW0603
    if _pool is None:
        return
    logger.info("Closing connection pool")
    await _pool.close()
    _pool = None
    logger.info("Connection pool closed")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection, None]:
    """Async context manager that yields a connection from the pool.

    Acquires a connection on entry and releases it back to the pool on exit,
    even if an exception is raised.

    Yields:
        An asyncpg Connection with pgvector already registered.

    Example::

        async with get_connection() as conn:
            version = await conn.fetchval("SELECT version()")
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


# -- Smoke test ---------------------------------------------------------------

async def _smoke_test() -> None:
    """Connect, query server version and pgvector extension, then disconnect."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    try:
        async with get_connection() as conn:
            version: str = await conn.fetchval("SELECT version()")
            logger.info("Server version: %s", version)

            ext: str | None = await conn.fetchval(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            )
            if ext:
                logger.info("pgvector extension: PRESENT")
            else:
                logger.warning(
                    "pgvector extension: NOT FOUND -- "
                    "run 'CREATE EXTENSION IF NOT EXISTS vector;' in your database"
                )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_smoke_test())
