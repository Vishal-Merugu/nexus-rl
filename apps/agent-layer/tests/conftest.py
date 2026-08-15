from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only. Trio is not a dependency here."""
    return "asyncio"
