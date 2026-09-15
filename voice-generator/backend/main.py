"""
FastAPI entrypoint. Chatterbox is loaded once at startup (lifespan) and kept
resident for the life of the process -- see chatterbox_service.py.

Run from the voice-generator/ project root with the venv active:
    uvicorn backend.main:app --host 127.0.0.1 --port 8100
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router as api_router
from .chatterbox_service import get_chatterbox_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up Chatterbox (download/load can take a while on first run)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_chatterbox_service().load_model)
    logger.info("Model ready. Serving requests.")
    yield


app = FastAPI(title="Voice Generator", lifespan=lifespan)

# API routes are registered before the static-file catch-all mount below, so
# they take priority over it.
app.include_router(api_router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
