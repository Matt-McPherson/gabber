#!/usr/bin/env python3
"""
Quick benchmark for an OpenAI-compatible local LLM server (e.g., Qwen3-Omni)
to evaluate viability as an STT alternative.

Defaults target your host: http://192.168.1.130:7002/v1 and model
"Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit".

Supports two API shapes:
- Chat multimodal with input_audio via /v1/chat/completions
- Whisper-like /v1/audio/transcriptions

Usage examples:
  python scripts/bench_qwen_stt.py sample.wav --ref "your reference text"
  python scripts/bench_qwen_stt.py samples/*.wav --mode chat
  python scripts/bench_qwen_stt.py sample.wav --base-url http://127.0.0.1:7002/v1 --model Qwen3-Omni

Outputs one JSON line per file and a final summary.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import requests


def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _normalize_text(s: str) -> str:
    return " ".join("".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).split())


def _levenshtein(a: List[str], b: List[str]) -> int:
    # Word-level Levenshtein distance
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def word_error_rate(ref: str, hyp: str) -> float:
    ref_norm = _normalize_text(ref)
    hyp_norm = _normalize_text(hyp)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein(ref_words, hyp_words)
    return dist / max(1, len(ref_words))


@dataclass
class BenchConfig:
    base_url: str
    model: str
    api_key: str
    mode: str  # "auto", "chat", or "transcribe"
    timeout: int


def transcribe_chat(cfg: BenchConfig, wav_path: str) -> Tuple[str, float]:
    audio_b64 = _b64_file(wav_path)
    payload = {
        "model": cfg.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Transcribe the following audio exactly."},
                    {"type": "input_audio", "audio": {"data": [audio_b64], "format": "wav"}},
                ],
            }
        ],
        "temperature": 0,
    }
    t0 = time.time()
    r = requests.post(
        f"{cfg.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=cfg.timeout,
    )
    t = time.time() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    return text, t


def transcribe_whisper(cfg: BenchConfig, wav_path: str) -> Tuple[str, float]:
    files = {"file": (os.path.basename(wav_path), open(wav_path, "rb"), "audio/wav")}
    data = {"model": cfg.model}
    t0 = time.time()
    r = requests.post(
        f"{cfg.base_url}/audio/transcriptions",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        files=files,
        data=data,
        timeout=cfg.timeout,
    )
    t = time.time() - t0
    r.raise_for_status()
    js = r.json()
    # OpenAI-compatible usually returns { text: "..." }
    text = js.get("text") or js.get("text_output") or json.dumps(js)
    return str(text).strip(), t


def run_one(cfg: BenchConfig, wav_path: str, reference: Optional[str]) -> dict:
    err = None
    hyp = ""
    latency = None

    try:
        if cfg.mode in ("auto", "chat"):
            try:
                hyp, latency = transcribe_chat(cfg, wav_path)
            except requests.HTTPError as e:
                # Fallback to whisper endpoint on 404/400 etc.
                if cfg.mode == "auto":
                    hyp, latency = transcribe_whisper(cfg, wav_path)
                else:
                    raise e
        else:
            hyp, latency = transcribe_whisper(cfg, wav_path)
    except Exception as e:
        err = str(e)

    wer_val = None
    if reference is not None and hyp:
        try:
            wer_val = round(word_error_rate(reference, hyp), 3)
        except Exception as e:
            err = f"WER failed: {e}"

    return {
        "file": wav_path,
        "latency_s": None if latency is None else round(latency, 3),
        "hypothesis": hyp,
        "reference": reference,
        "wer": wer_val,
        "error": err,
    }


def parse_kv_list(items: Iterable[str]) -> dict:
    out = {}
    for it in items:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k] = v
    return out


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Benchmark local LLM as STT")
    p.add_argument("inputs", nargs="+", help="One or more WAV paths (supports glob)")
    p.add_argument("--ref", help="Reference transcript for single-file run")
    p.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:7002/v1"),
    )
    p.add_argument(
        "--model",
        default=os.environ.get(
            "QWEN_MODEL", "Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit"
        ),
    )
    p.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", "sk-local"))
    p.add_argument(
        "--mode",
        choices=["auto", "chat", "transcribe"],
        default=os.environ.get("QWEN_MODE", "auto"),
        help="Which API to use; auto tries chat then falls back to transcriptions",
    )
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument(
        "--refs",
        nargs="*",
        help="Optional mapping ref per file like file.wav=reference ...",
    )

    args = p.parse_args(argv)

    paths: List[str] = []
    for inp in args.inputs:
        expanded = glob.glob(inp)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(inp)

    if not paths:
        print("No input files found", file=sys.stderr)
        return 2

    cfg = BenchConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        mode=args.mode,
        timeout=args.timeout,
    )

    # Build file->reference map
    ref_map = parse_kv_list(args.refs or [])
    results = []
    t_start = time.time()
    for pth in paths:
        ref = None
        if args.ref and len(paths) == 1:
            ref = args.ref
        elif pth in ref_map:
            ref = ref_map[pth]

        res = run_one(cfg, pth, ref)
        print(json.dumps(res, ensure_ascii=False))
        results.append(res)

    total_t = time.time() - t_start

    # Summary
    ok = [r for r in results if not r.get("error")]
    latencies = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    wers = [r["wer"] for r in ok if r.get("wer") is not None]
    summary = {
        "count": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "avg_wer": round(sum(wers) / len(wers), 3) if wers else None,
        "wall_time_s": round(total_t, 3),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
