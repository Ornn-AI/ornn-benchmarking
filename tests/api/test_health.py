"""Tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Health endpoint contract tests."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health returns HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient) -> None:
        """Response body contains ``status: ok``."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client: TestClient) -> None:
        """Response body includes the application version."""
        data = client.get("/health").json()
        assert "version" in data
        assert data["version"] == "0.2.0"

    def test_health_includes_service_name(self, client: TestClient) -> None:
        """Response body includes the service name."""
        data = client.get("/health").json()
        assert "service" in data
        assert data["service"] == "Ornn Benchmarking API"

    def test_health_response_is_json(self, client: TestClient) -> None:
        """Content-Type is application/json."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"

    def test_health_allows_get_only(self, client: TestClient) -> None:
        """POST to /health is not allowed (405)."""
        response = client.post("/health")
        assert response.status_code == 405
