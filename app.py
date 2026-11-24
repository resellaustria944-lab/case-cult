import io, os, tempfile, threading, logging
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
    return {"ok": True, "model": SPACE, "upstream_verified": False}

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
    try:
        with tempfile.TemporaryDirectory(prefix="casecult-") as temp:
            directory = Path(temp)
            inp = directory / "input.png"
            original.save(inp)
            results_dir = directory / "results"
            results_dir.mkdir()
            client = Client(SPACE, token=os.getenv("HF_TOKEN") or None, verbose=False,
                            download_files=str(results_dir), analytics_enabled=False,
                            max_workers=2, httpx_kwargs={"timeout": 60.0})
            try:
                job = client.submit(handle_file(str(inp)), "1024x1024", "General", api_name="/image")
                result = job.result(timeout=180)
                if not isinstance(result, (tuple,list)) or len(result) != 2:
                    raise ValueError("Unexpected ImageSlider response")
                candidate = result[1]
                if not isinstance(candidate, (str,os.PathLike)):
                    raise ValueError("No output file")
                output_path = Path(candidate).resolve()
                if not output_path.is_relative_to(directory.resolve()):
                    raise ValueError("Unexpected file path")
                with Image.open(output_path) as processed:
                    if "A" not in processed.getbands() and "transparency" not in processed.info:
                        raise ValueError("Missing alpha channel")
                    output = processed.convert("RGBA")
                    if output.size != original.size:
                        output = output.resize(original.size, Image.Resampling.LANCZOS)
                    lo, hi = output.getchannel("A").getextrema()
                    if lo >= 240 or hi <= 128:
                        raise ValueError("Invalid transparency")
                    buf = io.BytesIO()
                    output.save(buf, format="PNG")
                    return buf.getvalue()
            finally:
                client.close()
    except HTTPException:
        raise
    except Exception as e:
        log.warning("BiRefNet request failed: %s", type(e).__name__)
        raise HTTPException(502, "BiRefNet nicht erreichbar, ausgelastet oder ungültige Antwort. Bitte erneut versuchen.") from e
    finally:
        slots.release()

@app.post("/api/remove-background")
async def remove_background(request: Request):
    if request.headers.get("content-type","").split(";")[0] != "image/png":
        raise HTTPException(415, "PNG erwartet.")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "Bild zu groß (max. 12 MB).")
    output = await run_in_threadpool(cutout, bytes(raw))
    return Response(output, media_type="image/png", headers={"Cache-Control":"no-store","X-Content-Type-Options":"nosniff"})
