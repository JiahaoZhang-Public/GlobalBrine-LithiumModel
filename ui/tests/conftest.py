import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    os.environ["UI_TEST_MODE"] = "1"
    from ui.backend.app import app

    # Ensure FastAPI startup events run (initializes app.state.wrapper).
    with TestClient(app) as test_client:
        yield test_client
