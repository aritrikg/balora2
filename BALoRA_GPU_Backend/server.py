"""HTTPS-proxied inference endpoint for the BALoRA Streamlit dashboard."""
import hmac
import logging
import os
from threading import Lock
from typing import Literal

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from predict import load_model, segment
from scans import png_mask

MAX_SIDE = 2048
logger = logging.getLogger(__name__)
api = FastAPI(title="BALoRA inference", docs_url=None, redoc_url=None, openapi_url=None)
inference_lock = Lock()
cached_model = None
cached_method = None


@api.get("/healthz")
def health():
    return {"status": "ok"}


def generate_mask(method, gray, point):
    global cached_model, cached_method
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(429, "GPU busy", headers={"Retry-After": "3"})
    try:
        if cached_method != method or cached_model is None:
            cached_method = None
            cached_model = None
            cached_model = load_model(method)
            cached_method = method
        return png_mask(segment(cached_model, method, gray, point))
    except Exception as exc:
        logger.error("Inference failed (%s): %s", method, type(exc).__name__)
        raise HTTPException(503, "GPU inference unavailable") from exc
    finally:
        inference_lock.release()


@api.post("/segment")
async def segment_slice(
    request: Request,
    method: Literal["balora", "deterministic", "frozen"],
    width: int = Query(ge=2, le=MAX_SIDE),
    height: int = Query(ge=2, le=MAX_SIDE),
    x: int = Query(ge=0),
    y: int = Query(ge=0),
    authorization: str = Header(default=""),
):
    token = os.environ.get("BALORA_API_TOKEN", "")
    if len(token) < 32:
        raise HTTPException(503, "Authentication has not been configured")
    received = authorization.encode("utf-8")
    expected = f"Bearer {token}".encode("utf-8")
    if not hmac.compare_digest(received, expected):
        raise HTTPException(401, "Invalid credentials")
    if x >= width or y >= height:
        raise HTTPException(422, "The point must lie inside the image")
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/octet-stream":
        raise HTTPException(415, "Expected raw float32 pixels")
    expected_bytes = width * height * 4
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > expected_bytes:
            raise HTTPException(413, "Slice body exceeds the declared dimensions")
        raw.extend(chunk)
    if len(raw) != expected_bytes:
        raise HTTPException(422, "Slice body does not match the declared dimensions")

    gray = np.frombuffer(raw, dtype="<f4").reshape(height, width)
    if not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
        raise HTTPException(422, "Pixel values must be finite and in [0,1]")
    mask = await run_in_threadpool(generate_mask, method, gray, (x, y))
    return Response(mask, media_type="image/png", headers={"Cache-Control": "no-store"})
