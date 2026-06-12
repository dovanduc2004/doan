import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from source.api.router import api_router

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_router, prefix="/api")


STATIC_DIR = Path(__file__).parent / "static"


def asset_url(rel_path: str) -> str:
    """Trả về URL static kèm query param `?v=<mtime>` — browser tự reload khi
    file thay đổi trên disk, không cần sửa tay version mỗi lần edit."""
    path = STATIC_DIR / rel_path
    try:
        version = int(path.stat().st_mtime)
    except FileNotFoundError:
        version = 0
    return f"/static/{rel_path}?v={version}"


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "asset_url": asset_url},
    )
