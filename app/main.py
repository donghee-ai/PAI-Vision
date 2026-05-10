"""Compatibility entrypoint for the local perception API adapter.

The long-term hub/bridge process should live outside this repository. Keep
`app.main:app` importable so existing uvicorn commands continue to work while
new code can point at `app.local_api:app` directly.
"""

from app.adapters.local_api import app

__all__ = ["app"]
