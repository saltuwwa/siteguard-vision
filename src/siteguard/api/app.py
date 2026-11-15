import os
import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from siteguard.events.schema import ViolationEvent
from siteguard.storage.database import EventStore


class EventPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ViolationEvent]


class Camera(BaseModel):
    cameraid: str
    streamsessionid: str
    lastseen: str
    metrics: dict[str, Any]


def create_app(store: EventStore | None = None, token: str | None = None) -> FastAPI:
    store = store or EventStore(os.getenv("SITEGUARD_DATABASE_URL", "sqlite:///runs/siteguard.sqlite"))
    api_token = token if token is not None else os.getenv("SITEGUARD_API_TOKEN")
    app = FastAPI(title="Siteguard headless API", docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if not api_token:
            raise HTTPException(503, "API token not configured")
        if not authorization or not secrets.compare_digest(authorization, f"Bearer {api_token}"):
            raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.status_code, "message": exc.detail}},
                            status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": {"code": 422, "message": "Invalid request parameters"}},
                            status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": {"code": 500, "message": "Internal service error"}},
                            status_code=500)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authorize)])
    def ready() -> dict[str, Any]:
        try:
            store.list_events(1)
            return {"ready": True, "scope": "API/database", "worker_readiness": "see /cameras"}
        except Exception:
            raise HTTPException(503, "Database unavailable") from None

    @app.get("/cameras", response_model=list[Camera], dependencies=[Depends(authorize)])
    def cameras() -> list[dict[str, Any]]:
        return store.cameras()

    @app.get("/events", response_model=EventPage, dependencies=[Depends(authorize)])
    def events(limit: Annotated[int, Query(ge=1, le=200)] = 50,
               offset: Annotated[int, Query(ge=0)] = 0, camera_id: str | None = None) -> EventPage:
        total, items = store.list_events(limit, offset, camera_id)
        return EventPage(total=total, limit=limit, offset=offset, items=items)

    @app.get("/events/{event_id}", response_model=ViolationEvent, dependencies=[Depends(authorize)])
    def event(event_id: str) -> ViolationEvent:
        found = store.get(event_id)
        if not found:
            raise HTTPException(404, "Event not found")
        return found

    @app.get("/metrics", dependencies=[Depends(authorize)])
    def metrics() -> dict[str, Any]:
        return {"outbox": store.publication_metrics(), "cameras": store.cameras()}

    return app
