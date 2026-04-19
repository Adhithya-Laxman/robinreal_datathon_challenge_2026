from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from app.api.routes.events import router as events_router
from app.api.routes.listings import router as listings_router
from app.config import get_settings
from app.harness.bootstrap import bootstrap_database
from app.harness.events import ensure_events_table
from app.harness.session_middleware import SessionMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    bootstrap_database(db_path=settings.db_path, raw_data_dir=settings.raw_data_dir)
    ensure_events_table(db_path=settings.db_path)
    yield


app = FastAPI(
    title="Datathon 2026 Listings Harness",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(SessionMiddleware)

app.include_router(listings_router)
app.include_router(events_router)

_sred_images_dir = get_settings().raw_data_dir / "sred_images"
if _sred_images_dir.exists():
    app.mount(
        "/raw-data-images",
        StaticFiles(directory=str(_sred_images_dir)),
        name="raw-data-images",
    )
