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
import urllib.error
import urllib.request

_UA = "GOffice-media/1.0"


def _read(req, timeout):
    """urlopen + read; on HTTP error, surface the response body (real provider message)."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")[:400]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {body or e.reason}")

# Catalog surfaced to the UI dropdowns (single source of truth)
PROVIDERS = {
    "image": [
        {"id": "gemini",    "name": "Gemini (Google)",     "models": ["gemini-2.5-flash-image", "gemini-3-pro-image"], "get": "https://aistudio.google.com/apikey"},
        {"id": "openai",    "name": "OpenAI (ChatGPT)",    "models": ["gpt-image-1", "dall-e-3"],                      "get": "https://platform.openai.com/api-keys"},
        {"id": "stability", "name": "Stability AI",        "models": ["sd3.5-large", "core"],                          "get": "https://platform.stability.ai/account/keys"},
    ],
    "video": [
        {"id": "veo",       "name": "Google Veo",          "models": ["veo-3.0-fast-generate-001", "veo-3.0-generate-001"], "get": "https://aistudio.google.com/apikey"},
        {"id": "luma",      "name": "Luma Dream Machine",  "models": ["ray-2"],                                        "get": "https://lumalabs.ai/dream-machine/api"},
        {"id": "minimax",   "name": "MiniMax (Hailuo)",    "models": ["MiniMax-Hailuo-02"],                            "get": "https://www.minimax.io/"},
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
    return json.loads(_read(req, timeout).decode())


def _post_raw(url, obj, headers=None, timeout=90) -> bytes:
    h = {"Content-Type": "application/json", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers=h, method="POST")
    return _read(req, timeout)


def _post_multipart(url, fields, headers=None, timeout=120) -> bytes:
    """Text fields as str values; file parts as (filename, bytes, content_type) tuples."""
    boundary = "----GOffice" + secrets.token_hex(8)
    body = b""
    for k, v in fields.items():
        if isinstance(v, tuple):   # file part
            fname, data, ctype = v
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                     f"filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
        else:
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    body += (f"--{boundary}--\r\n").encode()
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": _UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    return _read(req, timeout)


# ----------------------------------------------------------------- images ----
def _ar_parts(aspect: str):
    """'16:9' -> (16.0, 9.0). Returns None for blank/auto/invalid."""
    try:
        w, h = (aspect or "").split(":")
        w, h = float(w), float(h)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return None


# aspect ratios each provider's API accepts natively
_STABILITY_AR = {"1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"}
_GEMINI_AR = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}


def _openai_size(aspect: str) -> str:
    """gpt-image-1 supports square / landscape / portrait — pick by orientation."""
    ar = _ar_parts(aspect)
    if not ar:
        return "1024x1024"
    w, h = ar
    if abs(w - h) < 0.01:
        return "1024x1024"
    return "1536x1024" if w > h else "1024x1536"


def generate_image(provider: str, model: str, key: str, prompt: str, image: str = "",
                   aspect: str = "", resolution: str = "1K") -> dict:
    """Return {mime, b64}. When `image` (data URI) is given, edit it instead of
    generating from scratch (image-to-image). `aspect` (e.g. '16:9') and
    `resolution` ('1K'/'2K') shape the output where the provider supports it.
    Raises a Thai message on failure."""
    if not key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์รูปภาพ")
    p = (provider or "gemini").lower()
    has_img = bool(image)
    aspect = (aspect or "").strip()
    if p == "openai":
        m = model or "gpt-image-1"
        size = _openai_size(aspect)
        if has_img:                          # edit the supplied image via /images/edits
            mime, b64in = _split_data_uri(image)
            ext = "png" if "png" in mime else ("jpg" if "jp" in mime else "png")
            raw = _post_multipart("https://api.openai.com/v1/images/edits",
                                  {"model": "gpt-image-1", "prompt": prompt, "size": size,
                                   "image": (f"base.{ext}", base64.b64decode(b64in), mime)},
                                  headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
            d = json.loads(raw.decode())
        else:
            body = {"model": m, "prompt": prompt, "size": size, "n": 1}
            if m.startswith("dall-e"):        # gpt-image-1 ไม่รับ response_format (คืน b64 อยู่แล้ว)
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
        ar_field = {"aspect_ratio": aspect} if aspect in _STABILITY_AR else {}
        if has_img:                          # image-to-image via the sd3 endpoint
            mime, b64in = _split_data_uri(image)
            ext = "png" if "png" in mime else ("jpg" if "jp" in mime else "png")
            raw = _post_multipart("https://api.stability.ai/v2beta/stable-image/generate/sd3",
                                  {"prompt": prompt, "mode": "image-to-image", "strength": "0.65",
                                   "output_format": "png",
                                   "model": (m if m.startswith("sd3") else "sd3.5-large"),
                                   "image": (f"base.{ext}", base64.b64decode(b64in), mime)},
                                  headers={"Authorization": f"Bearer {key}", "Accept": "image/*"})
        else:
            url = "https://api.stability.ai/v2beta/stable-image/generate/" + ("sd3" if m.startswith("sd3") else "core")
            raw = _post_multipart(url, {"prompt": prompt, "output_format": "png", **ar_field,
                                        **({"model": m} if m.startswith("sd3") else {})},
                                  headers={"Authorization": f"Bearer {key}", "Accept": "image/*"})
        return {"mime": "image/png", "b64": base64.b64encode(raw).decode()}
    # default: Gemini — editing = include the input image as an inline part
    parts = [{"text": prompt}]
    if has_img:
        mime, b64in = _split_data_uri(image)
        parts.append({"inlineData": {"mimeType": mime, "data": b64in}})
    body = {"contents": [{"parts": parts}]}
    img_cfg = {}
    if aspect in _GEMINI_AR:
        img_cfg["aspectRatio"] = aspect
    if (resolution or "").upper() in ("1K", "2K", "4K"):
        img_cfg["imageSize"] = (resolution or "1K").upper()
    if img_cfg:
        body["generationConfig"] = {"imageConfig": img_cfg}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-2.5-flash-image'}:generateContent?key={key}"
    d = _post_json(url, body)
    for part in d.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return {"mime": inline.get("mimeType") or "image/png", "b64": inline["data"]}
    raise RuntimeError("Gemini ไม่ส่งภาพกลับมา")


# ------------------------------------------------------------------ video ----
def _split_data_uri(uri: str):
    """data:image/png;base64,xxxx → (mime, b64). Returns ('image/png', uri) if not a data URI."""
    if uri.startswith("data:"):
        try:
            head, b64 = uri.split(",", 1)
            return head.split(":", 1)[1].split(";", 1)[0] or "image/png", b64
        except Exception:
            return "image/png", ""
    return "image/png", uri


def generate_video(provider: str, model: str, key: str, prompt: str, endpoint: str = "",
                   aspect_ratio: str = "16:9", resolution: str = "1080p",
                   mode: str = "t2v", image: str = "", opts: dict = None) -> dict:
    """Kick off a text/image-to-video job. Returns {task_id} (poll separately)."""
    if not key:
        raise RuntimeError("ยังไม่ได้ใส่คีย์วิดีโอ")
    opts = opts or {}
    p = (provider or "veo").lower()
    ar = aspect_ratio if aspect_ratio in ("16:9", "9:16", "1:1") else "16:9"
    is_i2v = (mode == "i2v" and bool(image))
    if p == "veo":
        m = model or "veo-3.0-fast-generate-001"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:predictLongRunning?key={key}"
        inst = {"prompt": prompt}
        if is_i2v:
            mime, b64 = _split_data_uri(image)
            inst["image"] = {"bytesBase64Encoded": b64, "mimeType": mime}
        params = {"aspectRatio": ar}
        if not opts.get("generate_audio", True):
            params["generateAudio"] = False
        d = _post_json(url, {"instances": [inst], "parameters": params}, timeout=90)
        name = d.get("name")
        if not name:
            raise RuntimeError(f"Veo ไม่คืน operation: {str(d)[:160]}")
        return {"task_id": name}
    if p == "fal":
        ep = endpoint or "fal-ai/minimax/video-01"
        body = {"prompt": prompt, "aspect_ratio": ar, "resolution": resolution}
        if opts.get("fixed_camera"):
            body["camera_fixed"] = True
        if opts.get("fps"):
            body["fps"] = int(opts["fps"])
        if "generate_audio" in opts:
            body["generate_audio"] = bool(opts["generate_audio"])
        if opts.get("audio"):
            # Talking Avatar: image + speech audio → lip-synced talking video
            body = {"image_url": image, "audio_url": opts["audio"]}
        elif is_i2v:
            body["image_url"] = image   # fal accepts a data URI here
        elif mode == "v2v" and opts.get("video"):
            body["video_url"] = opts["video"]   # fal accepts a data URI here
        d = _post_json(f"https://queue.fal.run/{ep}", body,
                       headers={"Authorization": f"Key {key}"}, timeout=90)
        status_url = d.get("status_url"); resp_url = d.get("response_url", "")
        if not status_url:
            raise RuntimeError(f"fal.ai ไม่คืน status_url: {str(d)[:160]}")
        return {"task_id": status_url + "|" + resp_url}
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


def poll_video(provider: str, model: str, key: str, task_id: str) -> dict:
    """Poll a video job. Returns {status:'pending'|'done'|'error', mime?, b64?, error?}."""
    p = (provider or "veo").lower()
    if p == "fal":
        status_url, _, resp_url = task_id.partition("|")
        d = _get_json(status_url, headers={"Authorization": f"Key {key}"}, timeout=30)
        st = (d.get("status") or "").upper()
        if st in ("IN_QUEUE", "IN_PROGRESS", ""):
            return {"status": "pending"}
        if st != "COMPLETED":
            return {"status": "error", "error": f"fal.ai: {st}"}
        r = _get_json(resp_url or status_url, headers={"Authorization": f"Key {key}"}, timeout=30)
        vid = r.get("video") or {}
        url = vid.get("url") if isinstance(vid, dict) else None
        if not url and isinstance(r.get("videos"), list) and r["videos"]:
            url = (r["videos"][0] or {}).get("url")
        if not url:
            return {"status": "error", "error": "fal.ai ไม่คืนลิงก์วิดีโอ"}
        raw = _get_bytes(url, timeout=120)
        return {"status": "done", "mime": "video/mp4", "b64": base64.b64encode(raw).decode()}
    if p == "veo":
        d = _get_json(f"https://generativelanguage.googleapis.com/v1beta/{task_id}?key={key}", timeout=30)
        if not d.get("done"):
            return {"status": "pending"}
        if d.get("error"):
            return {"status": "error", "error": str(d["error"])[:180]}
        resp = d.get("response", {}) or {}
        samples = ((resp.get("generateVideoResponse") or {}).get("generatedSamples")
                   or resp.get("generatedSamples") or [])
        uri = (samples[0].get("video", {}) or {}).get("uri") if samples else None
        if not uri:
            return {"status": "error", "error": "Veo ไม่คืนไฟล์วิดีโอ"}
        sep = "&" if "?" in uri else "?"
        raw = _get_bytes(uri + f"{sep}key={key}", timeout=120)
        return {"status": "done", "mime": "video/mp4", "b64": base64.b64encode(raw).decode()}
    if p == "luma":
        d = _get_json(f"https://api.lumalabs.ai/dream-machine/v1/generations/{task_id}",
                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        state = d.get("state")
        if state == "failed":
            return {"status": "error", "error": d.get("failure_reason") or "Luma ล้มเหลว"}
        url = (d.get("assets") or {}).get("video")
        if state != "completed" or not url:
            return {"status": "pending"}
        raw = _get_bytes(url, timeout=120)
        return {"status": "done", "mime": "video/mp4", "b64": base64.b64encode(raw).decode()}
    # MiniMax: query status → retrieve file url
    d = _get_json(f"https://api.minimax.io/v1/query/video_generation?task_id={task_id}",
                  headers={"Authorization": f"Bearer {key}"}, timeout=30)
    status = d.get("status") or (d.get("data") or {}).get("status") or ""
    if status.lower() in ("queueing", "preparing", "processing", "queuing"):
        return {"status": "pending"}
    if status.lower() == "fail":
        return {"status": "error", "error": "MiniMax สร้างไม่สำเร็จ"}
    file_id = d.get("file_id") or (d.get("data") or {}).get("file_id")
    if not file_id:
        return {"status": "pending"}
    f = _get_json(f"https://api.minimax.io/v1/files/retrieve?file_id={file_id}",
                  headers={"Authorization": f"Bearer {key}"}, timeout=30)
    url = ((f.get("file") or {}).get("download_url"))
    if not url:
        return {"status": "error", "error": "MiniMax ไม่คืนลิงก์ไฟล์"}
    raw = _get_bytes(url, timeout=120)
    return {"status": "done", "mime": "video/mp4", "b64": base64.b64encode(raw).decode()}


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
