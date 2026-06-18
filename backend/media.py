"""External media generation (BYO key) — real images / video / voice.

A DIFFERENT provider stack from Claude. The user picks a provider + model per
media type and supplies their own key (stored in settings, never hardcoded).
All calls use stdlib urllib.

Providers wired with real request shapes:
  image : Gemini, OpenAI (ChatGPT), Stability AI
  video : MiniMax (Hailuo), Luma Dream Machine
  voice : Gemini TTS, OpenAI TTS, ElevenLabs

Honest note: Claude can't make raster photos/video — that's why these exist.
Live calls can't run in the sandbox (outbound blocked); shapes follow each
provider's documented API. Missing key → caller falls back to SVG / text.
"""

from __future__ import annotations

import base64
import json
import secrets
import urllib.request

_UA = "GOffice-media/1.0"

# Catalog surfaced to the UI dropdowns (single source of truth)
PROVIDERS = {
    "image": [
        {"id": "gemini",    "name": "Gemini (Google)",     "models": ["gemini-2.5-flash-image", "gemini-3-pro-image"], "get": "https://aistudio.google.com/apikey"},
        {"id": "openai",    "name": "OpenAI (ChatGPT)",    "models": ["gpt-image-1", "dall-e-3"],                      "get": "https://platform.openai.com/api-keys"},
        {"id": "stability", "name": "Stability AI",        "models": ["sd3.5-large", "core"],                          "get": "https://platform.stability.ai/account/keys"},
    ],
    "video": [
        {"id": "minimax",   "name": "MiniMax (Hailuo)",    "models": ["MiniMax-Hailuo-02"],                            "get": "https://www.minimax.io/"},
        {"id": "luma",      "name": "Luma Dream Machine",  "models": ["ray-2"],                                        "get": "https://lumalabs.ai/dream-machine/api"},
    ],
    "voice": [
        {"id": "gemini",     "name": "Gemini TTS",         "models": ["gemini-2.5-flash-preview-tts"],                 "get": "https://aistudio.google.com/apikey"},
        {"id": "openai",     "name": "OpenAI TTS",         "models": ["gpt-4o-mini-tts", "tts-1"],                     "get": "https://platform.openai.com/api-keys"},
        {"id": "elevenlabs", "name": "ElevenLabs",         "models": ["eleven_multilingual_v2"],                       "get": "https://elevenlabs.io/app/settings/api-keys"},
    ],
    "music": [
        {"id": "elevenlabs", "name": "ElevenLabs (Sound/Music)", "models": ["eleven_text_to_sound_v2"],                "get": "https://elevenlabs.io/app/settings/api-keys"},
        {"id": "stability",  "name": "Stability Stable Audio",   "models": ["stable-audio-2"],                          "get": "https://platform.stability.ai/account/keys"},
        {"id": "replicate",  "name": "Replicate (MusicGen)",     "models": ["meta/musicgen"],                           "get": "https://replicate.com/account/api-tokens"},
    ],
}


def _post_json(url, obj, headers=None, timeout=90) -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post_raw(url, obj, headers=None, timeout=90) -> bytes:
    h = {"Content-Type": "application/json", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _post_multipart(url, fields, headers=None, timeout=120) -> bytes:
    boundary = "----GOffice" + secrets.token_hex(8)
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    body += (f"--{boundary}--\r\n").encode()
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ----------------------------------------------------------------- images ----
def generate_image(provider: str, model: str, key: str, prompt: str) -> dict:
    """Return {mime, b64}. Raises with a Thai message on missing key/failure."""
    if not key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์รูปภาพ")
    p = (provider or "gemini").lower()
    if p == "openai":
        m = model or "gpt-image-1"
        body = {"model": m, "prompt": prompt, "size": "1024x1024", "n": 1}
        if m.startswith("dall-e"):          # gpt-image-1 ไม่รับ response_format (คืน b64 อยู่แล้ว)
            body["response_format"] = "b64_json"
        d = _post_json("https://api.openai.com/v1/images/generations", body,
                       headers={"Authorization": f"Bearer {key}"})
        item = (d.get("data") or [{}])[0]
        b64 = item.get("b64_json") or ""
        if not b64:
            raise RuntimeError("OpenAI ไม่ส่งภาพกลับมา: " + str(d)[:140])
        return {"mime": "image/png", "b64": b64}
    if p == "stability":
        m = (model or "core")
        url = "https://api.stability.ai/v2beta/stable-image/generate/" + ("sd3" if m.startswith("sd3") else "core")
        raw = _post_multipart(url, {"prompt": prompt, "output_format": "png", **({"model": m} if m.startswith("sd3") else {})},
                              headers={"Authorization": f"Bearer {key}", "Accept": "image/*"})
        return {"mime": "image/png", "b64": base64.b64encode(raw).decode()}
    # default: Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-2.5-flash-image'}:generateContent?key={key}"
    d = _post_json(url, {"contents": [{"parts": [{"text": prompt}]}]})
    for part in d.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return {"mime": inline.get("mimeType") or "image/png", "b64": inline["data"]}
    raise RuntimeError("Gemini ไม่ส่งภาพกลับมา")


# ------------------------------------------------------------------ video ----
def generate_video(provider: str, model: str, key: str, prompt: str) -> dict:
    """Kick off a text-to-video job. Returns {task_id} (poll separately)."""
    if not key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์วิดีโอ")
    p = (provider or "minimax").lower()
    if p == "luma":
        d = _post_json("https://api.lumalabs.ai/dream-machine/v1/generations",
                       {"prompt": prompt, "model": model or "ray-2"},
                       headers={"Authorization": f"Bearer {key}"})
        tid = d.get("id")
        if not tid:
            raise RuntimeError(f"Luma ไม่คืน id: {str(d)[:120]}")
        return {"task_id": tid}
    d = _post_json("https://api.minimax.io/v1/video_generation",
                   {"model": model or "MiniMax-Hailuo-02", "prompt": prompt},
                   headers={"Authorization": f"Bearer {key}"})
    tid = d.get("task_id") or d.get("data", {}).get("task_id")
    if not tid:
        raise RuntimeError(f"MiniMax ไม่คืน task_id: {str(d)[:120]}")
    return {"task_id": tid}


# ------------------------------------------------------------------ voice ----
def generate_voice(provider: str, model: str, key: str, text: str, voice: str = "") -> dict:
    """Return {mime, b64}."""
    if not key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์เสียง")
    p = (provider or "gemini").lower()
    if p == "openai":
        raw = _post_raw("https://api.openai.com/v1/audio/speech",
                        {"model": model or "gpt-4o-mini-tts", "voice": voice or "alloy",
                         "input": text, "response_format": "mp3"},
                        headers={"Authorization": f"Bearer {key}"})
        return {"mime": "audio/mpeg", "b64": base64.b64encode(raw).decode()}
    if p == "elevenlabs":
        vid = voice or "21m00Tcm4TlvDq8ikWAM"
        raw = _post_raw(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
                        {"text": text, "model_id": model or "eleven_multilingual_v2"},
                        headers={"xi-api-key": key, "Accept": "audio/mpeg"})
        return {"mime": "audio/mpeg", "b64": base64.b64encode(raw).decode()}
    # default: Gemini TTS
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-2.5-flash-preview-tts'}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice or "Kore"}}}}}
    d = _post_json(url, body)
    for part in d.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return {"mime": inline.get("mimeType") or "audio/wav", "b64": inline["data"]}
    raise RuntimeError("TTS ไม่ส่งเสียงกลับมา")


def _get_json(url, headers=None, timeout=30) -> dict:
    h = {"Accept": "application/json", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_bytes(url, timeout=60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def generate_music(provider: str, model: str, key: str, prompt: str, duration: int = 20) -> dict:
    """Generate music/sound from a text prompt. Return {mime, b64}. Thai error on failure."""
    if not key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์ Music AI")
    p = (provider or "elevenlabs").lower()
    dur = max(5, min(int(duration or 20), 120))
    if p == "elevenlabs":
        raw = _post_raw("https://api.elevenlabs.io/v1/sound-generation",
                        {"text": prompt, "duration_seconds": min(dur, 22)},
                        headers={"xi-api-key": key, "Accept": "audio/mpeg"})
        return {"mime": "audio/mpeg", "b64": base64.b64encode(raw).decode()}
    if p == "stability":
        raw = _post_multipart("https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio",
                              {"prompt": prompt, "duration": str(dur), "output_format": "mp3"},
                              headers={"Authorization": f"Bearer {key}", "Accept": "audio/*"})
        return {"mime": "audio/mpeg", "b64": base64.b64encode(raw).decode()}
    if p == "replicate":
        import time as _t
        d = _post_json("https://api.replicate.com/v1/models/meta/musicgen/predictions",
                       {"input": {"prompt": prompt, "duration": dur}},
                       headers={"Authorization": f"Token {key}", "Prefer": "wait"})
        status = d.get("status")
        get_url = (d.get("urls") or {}).get("get")
        for _ in range(30):
            if status in ("succeeded", "failed", "canceled") or not get_url:
                break
            _t.sleep(2)
            d = _get_json(get_url, headers={"Authorization": f"Token {key}"})
            status = d.get("status")
        if status != "succeeded":
            raise RuntimeError("Replicate สร้างเพลงไม่สำเร็จ: " + str(d.get("error") or status))
        out = d.get("output")
        audio_url = out[0] if isinstance(out, list) and out else out
        if not audio_url:
            raise RuntimeError("Replicate ไม่คืนไฟล์เสียง")
        return {"mime": "audio/wav", "b64": base64.b64encode(_get_bytes(audio_url)).decode()}
    raise RuntimeError("ไม่รู้จัก provider เพลงนี้")
