"""
FastAPI entrypoint. Models are loaded once at startup (lifespan) and kept
resident for the life of the process -- see chatterbox_service/stt_service.

Run from the ai-french-teacher/ project root with the venv active:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import base64
import binascii
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from . import config
from .api.conversation import router as conversation_router
from .api.tts import router as tts_router
from .services.chatterbox_service import get_chatterbox_service
from .services.conversation_service import cleanup_expired_sessions
from .services.stt_service import get_stt_service
from .services.video_library import warm_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def _cleanup_loop():
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(config.SESSION_CLEANUP_INTERVAL_SECONDS)
        await loop.run_in_executor(None, cleanup_expired_sessions)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up models (Chatterbox download/load can take a while on first run)...")
    # Chatterbox (GPU), Whisper (CPU), and the video-matching embedding model
    # (CPU) are independent -- load them on separate threads concurrently
    # instead of back-to-back, since each is I/O/compute bound on its own
    # device/model and releases the GIL during the heavy lifting.
    # warm_embeddings() also backfills any catalog video missing a cached
    # embedding, so that one-time cost is paid here, not mid-conversation.
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(None, get_chatterbox_service().load_model),
        loop.run_in_executor(None, get_stt_service().load_model),
        loop.run_in_executor(None, warm_embeddings),
    )
    logger.info("Models ready. Serving requests.")

    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(title="AI French Teacher", lifespan=lifespan)


@app.middleware("http")
async def basic_auth_gate(request, call_next):
    # Shared team credential for public hosting -- see config.APP_USERNAME/
    # APP_PASSWORD. Inactive for local dev (both unset), so this is a no-op
    # until you deliberately configure it. Every request needs a valid
    # Authorization header, checked with compare_digest to avoid a timing
    # side-channel on the password.
    if not config.APP_PASSWORD:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, _, password = decoded.partition(":")
        except (binascii.Error, UnicodeDecodeError):
            username, password = "", ""
        if secrets.compare_digest(username, config.APP_USERNAME) and secrets.compare_digest(password, config.APP_PASSWORD):
            return await call_next(request)

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="AI French Teacher"'},
    )


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    # Without this, browsers can reuse a cached index.html/app.js/styles.css
    # across reloads without even asking the server -- so an edited frontend
    # file silently keeps running the old code until a hard refresh. Forcing
    # revalidation means the browser always asks; unchanged files still come
    # back as a fast 304 (StaticFiles sets ETag/Last-Modified already).
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# API routes are registered before the static-file catch-all mount below, so
# they take priority over it.
app.include_router(conversation_router)
app.include_router(tts_router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
