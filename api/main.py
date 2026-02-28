"""FastAPI application for the Ornn Benchmarking API.

Entrypoint for the Cloud Run service.  Wire routers and middleware here;
keep endpoint logic in ``api.routers.*`` modules.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.config import get_settings
from api.routers.health import router as health_router


def create_app() -> FastAPI:
    """Application factory - construct and configure the FastAPI instance.

    Using a factory function enables test code to build fresh instances
    with overridden settings / dependencies.
    """
    settings = get_settings()

    application = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version,
    )

    # --- routers --------------------------------------------------------
    application.include_router(health_router)

    return application


# Module-level instance used by ``uvicorn api.main:app``
app: FastAPI = create_app()
