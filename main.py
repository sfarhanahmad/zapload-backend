import os
import re
import logging
import subprocess
import httpx
from datetime import datetime
from urllib.parse import urlparse

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, validator
import yt_dlp

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("zapload")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ZapLoad API", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS (update ALLOWED_ORIGIN in .env for production) ──────────────────────
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Security headers middleware ───────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Powered-By-Hidden"] = ""
    return response

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_SERVER_BYTES   = 500 * 1024 * 1024   # 500 MB  → stream through server
MAX_ALLOWED_BYTES  = 2  * 1024 * 1024 * 1024  # 2 GB hard cap
SIZE_THRESHOLD     = 500 * 1024 * 1024   # same as MAX_SERVER_BYTES

ALLOWED_SCHEMES    = {"http", "https"}
BLOCKED_HOSTS      = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_url(url: str) -> str:
    """Strict URL validation — raises HTTPException on failure."""
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL.")

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise HTTPException(status_code=400, detail="Invalid URL.")
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL.")
    if parsed.hostname in BLOCKED_HOSTS:
        raise HTTPException(status_code=400, detail="Invalid URL.")
    # Block private IP ranges
    if re.match(r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)", parsed.hostname or ""):
        raise HTTPException(status_code=400, detail="Invalid URL.")
    return url


def format_bytes(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


# ── Models ────────────────────────────────────────────────────────────────────
class InfoRequest(BaseModel):
    url: str

    @validator("url")
    def url_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("URL is required.")
        return v.strip()


class DownloadRequest(BaseModel):
    url: str
    format_id: str = "best"

    @validator("url")
    def url_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("URL is required.")
        return v.strip()

    @validator("format_id")
    def safe_format(cls, v):
        # Only allow safe characters to prevent injection
        if not re.match(r'^[\w\+\-\.\[\]\(\)\/\,\s]+$', v):
            raise ValueError("Invalid format.")
        return v.strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ZapLoad API is running ⚡"}


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/info")
@limiter.limit("10/minute")
async def get_info(request: Request, body: InfoRequest):
    """
    Returns video/file info: title, thumbnail, available formats, file size.
    Also decides whether the file should be served via server or direct link.
    """
    url = validate_url(body.url)

    # ── Try yt-dlp first (handles YT, TikTok, Insta, etc.) ──────────────────
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "cookiefile": None,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        formats = []
        seen = set()
        for f in (info.get("formats") or []):
            height   = f.get("height")
            ext      = f.get("ext", "mp4")
            fid      = f.get("format_id", "")
            abr      = f.get("abr")
            vcodec   = f.get("vcodec", "none")
            acodec   = f.get("acodec", "none")

            if vcodec != "none" and height:
                label = f"{height}p ({ext.upper()})"
                key   = f"{height}p"
            elif vcodec == "none" and acodec != "none":
                label = f"Audio only ({int(abr or 0)}kbps {ext.upper()})"
                key   = f"audio_{abr}"
            else:
                continue

            if key in seen:
                continue
            seen.add(key)

            filesize = f.get("filesize") or f.get("filesize_approx") or 0
            formats.append({
                "format_id": fid,
                "label":     label,
                "ext":       ext,
                "filesize":  filesize,
                "filesize_human": format_bytes(filesize) if filesize else "Unknown",
            })

        # Sort: highest quality first, audio last
        formats.sort(key=lambda x: (
            0 if "Audio" not in x["label"] else 1,
            -(int(x["label"].split("p")[0]) if x["label"][0].isdigit() else 0)
        ))

        # Add a best-quality default
        formats.insert(0, {
            "format_id": "bestvideo+bestaudio/best",
            "label":     "Best Quality (auto)",
            "ext":       "mp4",
            "filesize":  0,
            "filesize_human": "Unknown",
        })

        total_size = info.get("filesize") or info.get("filesize_approx") or 0
        large = total_size > SIZE_THRESHOLD

        return {
            "type":        "media",
            "title":       info.get("title", "Unknown"),
            "thumbnail":   info.get("thumbnail"),
            "duration":    info.get("duration"),
            "uploader":    info.get("uploader"),
            "formats":     formats,
            "filesize":    total_size,
            "filesize_human": format_bytes(total_size) if total_size else "Unknown",
            "large_file":  large,
            "large_warning": f"Large file ({format_bytes(total_size)}) — your browser will download this directly" if large else None,
            "direct_url":  url if large else None,
        }

    except yt_dlp.utils.DownloadError:
        pass  # Not a media URL — fall through to direct file handling

    # ── Direct file link (ZIP, RAR, EXE, ISO …) ─────────────────────────────
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            head = await client.head(url)
            content_length = int(head.headers.get("content-length", 0))
            content_type   = head.headers.get("content-type", "application/octet-stream")
            final_url      = str(head.url)

        if content_length > MAX_ALLOWED_BYTES:
            raise HTTPException(status_code=400, detail="File exceeds 2 GB limit.")

        large = content_length > SIZE_THRESHOLD
        filename = final_url.split("/")[-1].split("?")[0] or "download"

        return {
            "type":        "direct",
            "title":       filename,
            "thumbnail":   None,
            "formats":     [{"format_id": "direct", "label": "Direct Download", "ext": filename.split(".")[-1], "filesize": content_length, "filesize_human": format_bytes(content_length) if content_length else "Unknown"}],
            "filesize":    content_length,
            "filesize_human": format_bytes(content_length) if content_length else "Unknown",
            "large_file":  large,
            "large_warning": f"Large file ({format_bytes(content_length)}) — your browser will download this directly" if large else None,
            "direct_url":  final_url,
            "content_type": content_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Info error | url={url} | {e}")
        raise HTTPException(status_code=400, detail="Could not fetch file info. Check the URL and try again.")


@app.post("/download")
@limiter.limit("10/minute")
async def download(request: Request, body: DownloadRequest):
    """
    For small files (< 500MB): streams file through our server.
    For large files: returns the direct URL for browser to download.
    """
    url       = validate_url(body.url)
    format_id = body.format_id

    # ── Large / direct file: just redirect ──────────────────────────────────
    if format_id == "direct":
        return JSONResponse({"redirect": url})

    # ── yt-dlp media download (small files only) ─────────────────────────────
    ydl_opts = {
        "quiet":       True,
        "no_warnings": True,
        "format":      format_id,
        # Use aria2c for parallel chunk downloading (IDM-style speed)
        "external_downloader":      "aria2c",
        "external_downloader_args": ["-x", "16", "-s", "16", "-k", "1M"],
        "outtmpl":     "/tmp/zapload_%(id)s.%(ext)s",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info     = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)

        if not os.path.exists(filepath):
            # yt-dlp sometimes changes extension
            base = filepath.rsplit(".", 1)[0]
            for ext in ["mp4", "mkv", "webm", "mp3", "m4a"]:
                candidate = f"{base}.{ext}"
                if os.path.exists(candidate):
                    filepath = candidate
                    break

        filesize = os.path.getsize(filepath)
        if filesize > MAX_SERVER_BYTES:
            os.remove(filepath)
            return JSONResponse({
                "redirect": url,
                "warning":  "File too large to stream — downloading directly from source."
            })

        filename = os.path.basename(filepath)

        def iterfile():
            with open(filepath, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
            try:
                os.remove(filepath)
            except Exception:
                pass

        return StreamingResponse(
            iterfile(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except Exception as e:
        logger.error(f"Download error | url={url} | format={format_id} | {e}")
        raise HTTPException(status_code=500, detail="Download failed. Please try again.")
