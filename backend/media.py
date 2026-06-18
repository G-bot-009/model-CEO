"""External media generation (BYO key) — real images / video / voice.

These are a DIFFERENT provider stack from Claude: Google Gemini for images,
MiniMax (Hailuo) for video, and a TTS provider for voice. Each needs the user's
own API key, stored in settings (never hardcoded). All calls use stdlib urllib.

Honest note: Claude cannot make raster photos or video — that's why these are
separate. If a key is missing the caller falls back to SVG / text only. Live
calls can't be exercised in the sandbox (outbound blocked); the request shapes
follow each provider's documented API.
"""

from __future__ import annotations

import base64
import json
import urllib.request

_UA = "GOffice-media/1.0"


def _post_json(url: str, obj: dict, headers: dict | None = None, timeout: int = 60) -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def gemini_image(prompt: str, api_key: str, model: str = "gemini-2.5-flash-image") -> dict:
    """Generate a real image via Google Gemini. Returns {mime, b64}."""
    if not api_key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์ Gemini")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    d = _post_json(url, body)
    for part in d.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return {"mime": inline.get("mimeType") or inline.get("mime_type") or "image/png",
                    "b64": inline["data"]}
    raise RuntimeError("Gemini ไม่ส่งภาพกลับมา")


def minimax_video(prompt: str, api_key: str, model: str = "MiniMax-Hailuo-02") -> dict:
    """Kick off a MiniMax (Hailuo) text/image-to-video job. Returns {task_id}."""
    if not api_key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์ MiniMax")
    url = "https://api.minimax.io/v1/video_generation"
    d = _post_json(url, {"model": model, "prompt": prompt},
                   headers={"Authorization": f"Bearer {api_key}"})
    tid = d.get("task_id") or d.get("data", {}).get("task_id")
    if not tid:
        raise RuntimeError(f"MiniMax ไม่คืน task_id: {str(d)[:120]}")
    return {"task_id": tid}


def minimax_video_status(task_id: str, api_key: str) -> dict:
    url = f"https://api.minimax.io/v1/query/video_generation?task_id={task_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return {"status": d.get("status"), "file_id": d.get("file_id")}


def gemini_tts(text: str, api_key: str, model: str = "gemini-2.5-flash-preview-tts",
               voice: str = "Kore") -> dict:
    """Generate speech (Thai-capable) via Gemini TTS. Returns {mime, b64}."""
    if not api_key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์ TTS")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}},
    }
    d = _post_json(url, body)
    for part in d.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return {"mime": inline.get("mimeType") or "audio/wav", "b64": inline["data"]}
    raise RuntimeError("TTS ไม่ส่งเสียงกลับมา")
