import io, os, tempfile, threading, logging, re
from gradio_client.exceptions import AppError
from starlette.requests import ClientDisconnect
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from gradio_client import Client, handle_file
from starlette.concurrency import run_in_threadpool

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
ROOT = Path(__file__).resolve().parent
SPACE = "ZhengPeng7/BiRefNet_demo"
MAX_BYTES = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
slots = threading.BoundedSemaphore(1)
log = logging.getLogger("case-cult")

@app.get("/")
def home():
    return FileResponse(ROOT / "index.html", headers={"Cache-Control":"no-store"})

@app.get("/api/health")
def health():
    return {"ok": True, "model": SPACE, "upstream_verified": False, "revision": "cutout-v2"}

def output_files(value):
    if isinstance(value, (str, os.PathLike)):
        yield Path(value)
    elif isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield Path(value["path"])
        else:
            for item in value.values():
                yield from output_files(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from output_files(item)

def cutout(raw):
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format not in ("PNG","JPEG","WEBP") or opened.width * opened.height > 20_000_000:
                raise ValueError("Unsupported image")
            original = ImageOps.exif_transpose(opened).convert("RGBA")
            original.thumbnail((1600,1600))
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Ungültiges oder zu großes Bild.")
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "KI beschäftigt. Bitte erneut versuchen.")
    stage = "connect"
    try:
        with tempfile.TemporaryDirectory(prefix="casecult-") as temp:
            directory = Path(temp)
            inp = directory / "input.png"
            original.convert("RGB").save(inp)
            results_dir = directory / "results"
            results_dir.mkdir()
            client = Client(SPACE, token=os.getenv("HF_TOKEN") or None, verbose=False,
                            download_files=str(results_dir), analytics_enabled=False,
                            max_workers=2, httpx_kwargs={"timeout": 60.0})
            try:
                stage = "inference"
                job = client.submit(handle_file(str(inp)), "1024x1024", "General", api_name="/image")
                result = job.result(timeout=180)
                stage = "decode"
                log.info("BiRefNet result type: %s", type(result).__name__)
                for candidate in reversed(list(output_files(result))):
                    output_path = candidate.resolve()
                    if not output_path.is_relative_to(results_dir.resolve()) or not output_path.is_file():
                        continue
                    try:
                        with Image.open(output_path) as processed:
                            if "A" not in processed.getbands() and "transparency" not in processed.info:
                                continue
                            output = processed.convert("RGBA")
                            if output.size != original.size:
                                output = output.resize(original.size, Image.Resampling.LANCZOS)
                            lo, hi = output.getchannel("A").getextrema()
                            if lo >= 240 or hi <= 128:
                                continue
                            buf = io.BytesIO()
                            output.save(buf, format="PNG")
                            return buf.getvalue()
                    except (OSError, ValueError):
                        continue
                raise ValueError("No valid transparent image returned")
            finally:
                try:
                    client.close()
                except Exception:
                    log.warning("Gradio client cleanup failed")
    except HTTPException:
        raise
    except Exception as e:
        # Record diagnostic category without exposing image paths, URLs or credentials.
        msg = str(e).lower()
        reason = "quota" if any(k in msg for k in ("quota", "rate limit", "gpu time", "429")) else "auth" if any(k in msg for k in ("unauthorized", "authentication", "401", "403", "login")) else "timeout" if isinstance(e, TimeoutError) or "timeout" in msg else "upstream" if isinstance(e, AppError) else "invalid-output" if stage == "decode" else "connection"
        log.warning("BiRefNet failure stage=%s type=%s reason=%s", stage, type(e).__name__, reason)
        details = {"quota":"Hugging Face GPU-Kontingent erschöpft.", "auth":"Hugging Face verlangt eine Anmeldung oder API-Zugriffsberechtigung.", "timeout":"Zeitlimit der KI-Verarbeitung erreicht.", "upstream":"Hugging Face meldet einen Fehler bei der KI-Verarbeitung.", "invalid-output":"Die KI lieferte kein gültiges transparentes PNG.", "connection":"Verbindung zur KI fehlgeschlagen."}
        raise HTTPException(502, details[reason] + " [" + stage + "/" + type(e).__name__ + "]") from e
    finally:
        slots.release()

@app.post("/api/remove-background")
async def remove_background(request: Request):
    if request.headers.get("content-type","").split(";")[0] != "image/png":
        raise HTTPException(415, "PNG erwartet.")
    raw = bytearray()
    try:
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BYTES:
                raise HTTPException(413, "Bild zu groß (max. 12 MB).")
    except ClientDisconnect:
        raise HTTPException(400, "Upload unterbrochen.")
    output = await run_in_threadpool(cutout, bytes(raw))
    return Response(output, media_type="image/png", headers={"Cache-Control":"no-store","X-Content-Type-Options":"nosniff"})
