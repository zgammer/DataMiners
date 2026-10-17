import os, shutil, tempfile, uuid, secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector import analyze

API_KEY = os.getenv("API_KEY", "")
MAX_BYTES = 300 * 1024 * 1024
UPLOADS = Path(tempfile.gettempdir()) / "cashier_uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="Cashier discipline AI")
pool = ThreadPoolExecutor(max_workers=1)   # одна тяжёлая задача за раз
JOBS: dict[str, dict] = {}


def auth(x_api_key: str = Header(default="")):
    if not API_KEY or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Invalid API key")


def _run(job_id: str, path: Path):
    try:
        JOBS[job_id] = analyze(path, job_id)
    except Exception as e:
        JOBS[job_id] = dict(job_id=job_id, status="failed", error=str(e))
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/v1/analyze", dependencies=[Depends(auth)])
async def analyze_video(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in {".mov", ".mp4"}:
        raise HTTPException(400, "Нужен видеофайл .mov")
    job_id = uuid.uuid4().hex[:8]
    path = UPLOADS / f"{job_id}.mov"
    size = 0
    with open(path, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                out.close(); path.unlink(missing_ok=True)
                raise HTTPException(413, "Файл слишком большой")
            out.write(chunk)
    JOBS[job_id] = dict(job_id=job_id, status="processing")
    pool.submit(_run, job_id, path)
    return dict(job_id=job_id)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


@app.get("/health")
def health():
    return {"ok": True}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
