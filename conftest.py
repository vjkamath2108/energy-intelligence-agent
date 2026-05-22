import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Set asyncio_mode=auto so every async test runs without extra markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
