"""FastAPI application factory for FarmLink."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.config import APP_TITLE

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
