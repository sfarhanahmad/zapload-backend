import os
import re
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("zapload")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ZapLoad API", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

SIZE_THRESHOLD  = 500 * 1024 * 1024
MAX_ALLOWED     = 2 * 1024 * 1024 * 1024
ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_HOSTS   = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

YDL_BASE = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
    "extractor_args": {
        "youtube": {
            "player_client": ["web", "android"],
        }
    },
    "socket_timeout": 30,
}

def validate_url(url: str) -> str:
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
    if re.match(r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)", parsed.hostname or ""):
        raise HTTPException(status_code=400, detail="Invalid URL.")
    return url

def format_bytes(b):
    if not b:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"

def build_formats(info):
    formats = []
    seen_labels = set()
    raw_formats = info.get("formats") or []

    heights = set()
    for f in raw_formats:
        h = f.get("height")
        vcodec = f.get("vcodec", "none")
        if h and vcodec != "none" and h >= 144:
            heights.add(h)

    quality_labels = {
        2160: "4K (2160p)",
        1440: "1440p (2K)",
        1080: "1080p Full HD",
        720:  "720p HD",
        480:  "480p",
        360:  "360p",
        240:  "240p",
        144:  "144p",
    }

    for h in sorted(heights, reverse=True):
        label = quality_labels.get(h, f"{h}p")
        if label in seen_labels:
            continue
        seen_labels.add(label)

        best = None
        best_tbr = 0
        for f in raw_formats:
            if f.get("height") == h and f.get("vcodec", "none") != "none":
                tbr = f.get("tbr") or 0
                if tbr > best_tbr:
                    best_tbr = tbr
                    best = f

        if best:
            fid = best.get("format_id", "best")
            filesize = best.get("filesize") or best.get("filesize_approx") or 0
            formats.append({
                "format_id": f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={h}]+bestaudio/best[height<={h}]",
                "label": f"🎬 {label} — MP4",
                "ext": "mp4",
                "filesize": filesize,
                "filesize_human": format_bytes(filesize),
                "height": h,
            })

    audio_formats = []
    for f in raw_formats:
        vcodec = f.get("vcodec", "none")
        acodec = f.get("acodec", "none")
        if vcodec == "none" and acodec != "none":
            abr = int(f.get("abr") or 0)
            if abr > 0:
                audio_formats.append((abr, f.get("format_id", "bestaudio")))

    seen_abr = set()
    for abr, fid in sorted(audio_formats, reverse=True):
        if abr in seen_abr:
            continue
        seen_abr.add(abr)
        formats.append({
            "format_id": "bestaudio/best",
            "label": f"🎵 Audio Only — {abr}kbps MP3",
            "ext": "mp3",
            "filesize": 0,
            "filesize_human": "Unknown",
            "height": 0,
        })

    formats.insert(0, {
        "format_id": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "label": "⭐ Best Quality (auto)",
        "ext": "mp4",
        "filesize": info.get("filesize") or info.get("filesize_approx") or 0,
        "filesize_human": format_bytes(info.get("filesize") or info.get("filesize_approx") or 0),
        "height": 9999,
    })

    return formats


class InfoRequest(BaseModel):
    url: str

    @validator("url")
    def url_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("URL is required.")
        return v.strip()


class DownloadRequest(BaseModel):
    url: str
    format_id: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    @validator("url")
    def url_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("URL is required.")
        return v.strip()

    @validator("format_id")
    def safe_format(cls, v):
        if len(v) > 300:
            raise ValueError("Invalid format.")
        return v.strip()


@app.get("/")
async def root():
    return {"status": "ZapLoad API is running ⚡"}

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/info")
@limiter.limit("15/minute")
async def get_info(request: Request, body: InfoRequest):
    url = validate_url(body.url)

    try:
        opts = {**YDL_BASE}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        formats = build_formats(info)
        total_size = info.get("filesize") or info.get("filesize_approx") or 0
        large = total_size > SIZE_THRESHOLD

        return {
            "type": "media",
            "title": info.get("title", "Unknown"),
            "thumbnail": info.get("thumbnail"),
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
            "formats": formats,
            "filesize": total_size,
            "filesize_human": format_bytes(total_size),
            "large_file": large,
            "large_warning": f"Large file ({format_bytes(total_size)}) — your browser will download this directly" if large else None,
        }

    except yt_dlp.utils.DownloadError as e:
        err = str(e).lower()
        if any(x in err for x in ["sign in", "login", "private", "removed", "unavailable", "blocked"]):
            raise HTTPException(status_code=400, detail="This video is private, removed, or requires login.")
        pass
    except Exception as e:
        logger.error(f"yt-dlp error: {e}")
        pass

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            head = await client.head(url)
            content_length = int(head.headers.get("content-length", 0))
            content_type   = head.headers.get("content-type", "application/octet-stream")
            final_url      = str(head.url)

        if content_length > MAX_ALLOWED:
            raise HTTPException(status_code=400, detail="File exceeds 2 GB limit.")

        large = content_length > SIZE_THRESHOLD
        filename = final_url.split("/")[-1].split("?")[0] or "download"
        ext = filename.split(".")[-1] if "." in filename else "file"

        if ext in ["mp4", "mkv", "avi", "mov", "webm"]:
            icon = "🎬"; label = f"Video File ({ext.upper()})"
        elif ext in ["mp3", "m4a", "wav", "flac", "aac"]:
            icon = "🎵"; label = f"Audio File ({ext.upper()})"
        elif ext in ["zip", "rar", "7z", "tar", "gz"]:
            icon = "📦"; label = f"Archive ({ext.upper()})"
        else:
            icon = "📁"; label = f"File ({ext.upper()})"

        return {
            "type": "direct",
            "title": filename,
            "thumbnail": None,
            "formats": [{
                "format_id": "direct",
                "label": f"⬇️ {label} — {format_bytes(content_length)}",
                "ext": ext,
                "filesize": content_length,
                "filesize_human": format_bytes(content_length),
            }],
            "filesize": content_length,
            "filesize_human": format_bytes(content_length),
            "large_file": large,
            "large_warning": f"Large file ({format_bytes(content_length)}) — your browser will download this directly" if large else None,
            "direct_url": final_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Direct file error | {e}")
        raise HTTPException(status_code=400, detail="Could not fetch info. Please check the link and try again.")


@app.post("/download")
@limiter.limit("10/minute")
async def download(request: Request, body: DownloadRequest):
    url       = validate_url(body.url)
    format_id = body.format_id

    if format_id == "direct":
        return JSONResponse({"redirect": url})

    tmp_dir = "/tmp"
    ydl_opts = {
        **YDL_BASE,
        "skip_download": False,
        "format": format_id,
        "outtmpl": f"{tmp_dir}/zapload_%(id)s.%(ext)s",
        "merge_output_format": "mp4",
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    try:
        import subprocess
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
        ydl_opts["external_downloader"] = "aria2c"
        ydl_opts["external_downloader_args"] = ["-x", "16", "-s", "16", "-k", "1M"]
    except Exception:
        pass

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info     = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            if not os.path.exists(filepath):
                base = filepath.rsplit(".", 1)[0]
                for ext in ["mp4", "mkv", "webm", "mp3", "m4a"]:
                    candidate = f"{base}.{ext}"
                    if os.path.exists(candidate):
                        filepath = candidate
                        break

        if not os.path.exists(filepath):
            raise HTTPException(status_code=500, detail="Download failed.")

        filesize = os.path.getsize(filepath)
        if filesize > SIZE_THRESHOLD:
            os.remove(filepath)
            return JSONResponse({"redirect": url})

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error | {e}")
        raise HTTPException(status_code=500, detail="Download failed. Please try again.")