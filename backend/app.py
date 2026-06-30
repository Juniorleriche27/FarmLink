"""Backward-compatible ASGI entrypoint for FarmLink.

The backend is split into api/core/domain/services/schemas modules. This file stays
small so existing launch commands such as `uvicorn app:app` keep working.
"""

from api.app import create_app

app = create_app()
