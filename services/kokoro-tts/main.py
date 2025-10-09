# Copyright 2025 Fluently AI, Inc. DBA Gabber. All rights reserved.
# SPDX-License-Identifier: SUL-1.0

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Response
from kokoro_onnx import SAMPLE_RATE, Kokoro
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import uvicorn

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", "kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", "voices-v1.0.bin")
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "af_sarah")
DEFAULT_LANG = os.environ.get("KOKORO_DEFAULT_LANG", "en-us")

kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

app = FastAPI()


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(DEFAULT_VOICE, description="Voice identifier defined by Kokoro")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed multiplier")
    lang: str = Field(DEFAULT_LANG, description="Language code supported by Kokoro")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/voices")
async def voices() -> dict[str, Any]:
    voices = kokoro.get_voices()
    return {
        "voices": voices,
        "default": DEFAULT_VOICE,
        "languages": kokoro.get_languages(),
        "default_language": DEFAULT_LANG,
    }


async def _synthesize(request: TTSRequest) -> Response:
    async def generate_audio() -> tuple[np.ndarray, int]:
        return kokoro.create(
            request.text,
            voice=request.voice,
            speed=request.speed,
            lang=request.lang,
        )

    try:
        audio, sample_rate = await run_in_threadpool(generate_audio)
    except ValueError as exc:
        logging.exception("Invalid Kokoro request")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Unexpected Kokoro failure")
        raise HTTPException(status_code=500, detail="Kokoro synthesis failed") from exc

    if sample_rate != SAMPLE_RATE:
        logging.warning(
            "Kokoro returned unexpected sample rate %s (expected %s)",
            sample_rate,
            SAMPLE_RATE,
        )

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    if len(audio_bytes) % 2 != 0:
        audio_bytes += b"\x00"

    media_type = f"audio/l16;rate={sample_rate}"
    return Response(content=audio_bytes, media_type=media_type)


@app.post("/kokoro-tts")
async def kokoro_tts(request: TTSRequest = Body(...)) -> Response:
    """Primary synthesis endpoint used by the Gabber Kokoro node."""
    return await _synthesize(request)


@app.post("/tts")
async def tts(request: TTSRequest = Body(...)) -> Response:
    """Backward compatible endpoint matching the kitten-tts route."""
    return await _synthesize(request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7004)
