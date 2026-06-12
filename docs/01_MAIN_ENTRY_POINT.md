# MAIN.PY - Entry Point của ứng dụng

## File: `main.py`

## 1. Chức năng

File `main.py` là điểm khởi đầu (entry point) của toàn bộ ứng dụng. Nó khởi tạo FastAPI application, cấu hình CORS middleware, mount static files, đăng ký API router, và định nghĩa route trang chủ. Ngoài ra còn có một tiện ích **cache-busting** cho static asset.

## 2. Chi tiết hoạt động

### 2.1 Load Environment Variables
```python
load_dotenv()
```
- Dùng `python-dotenv` load biến môi trường từ `.env` (API keys, Qdrant URL, model paths, AWS config...).

### 2.2 Khởi tạo FastAPI App
```python
app = FastAPI()
```

### 2.3 CORS Middleware
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- Cho phép tất cả origins/methods/headers. **Lưu ý**: `allow_origins=["*"]` chỉ phù hợp development; production nên giới hạn cụ thể.

### 2.4 Template Engine & Static Files
```python
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
```
- **Jinja2Templates**: render `index.html`.
- **StaticFiles**: serve CSS/JS từ `static/` tại URL `/static`.

### 2.5 API Router
```python
app.include_router(api_router, prefix="/api")
```
- Mọi endpoint có dạng `/api/chat/...` hoặc `/api/session/...`.

### 2.6 Tiện ích cache-busting `asset_url`
```python
STATIC_DIR = Path(__file__).parent / "static"

def asset_url(rel_path: str) -> str:
    path = STATIC_DIR / rel_path
    try:
        version = int(path.stat().st_mtime)
    except FileNotFoundError:
        version = 0
    return f"/static/{rel_path}?v={version}"
```
- Trả về URL static kèm `?v=<mtime>` (thời điểm sửa file). Khi file thay đổi trên disk, query param đổi theo → browser tự reload, không cần sửa version thủ công.

### 2.7 Route trang chủ
```python
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "asset_url": asset_url},
    )
```
- GET `/` render `index.html`, truyền hàm `asset_url` vào template để dùng cho `<script>`/`<link>` (cache-busting).

## 3. Luồng khởi động

```
1. load_dotenv()           → Load .env
2. FastAPI()               → Tạo app instance
3. CORSMiddleware          → Cấu hình CORS
4. Jinja2Templates         → Setup template engine
5. StaticFiles             → Mount /static
6. include_router          → Đăng ký /api routes
   (import chat.py kích hoạt việc load TẤT CẢ AI model + khởi tạo Orchestrator)
7. uvicorn chạy app (run_mac.sh: port 8080; mặc định khác có thể 8000)
```

> Việc import `source.api.endpoints.chat` ở bước 6 sẽ load toàn bộ model (embedding, BERT QA, Cross-Encoder) và khởi tạo các agent — đây là lý do thời gian khởi động lâu (vài phút). Xem doc 02.

## 4. Dependencies

- `fastapi`, `uvicorn`: Web framework + ASGI server
- `fastapi.middleware.cors`: CORS
- `fastapi.templating`, `fastapi.staticfiles`: Template + static
- `python-dotenv`: Load env
- `source.api.router`: API router module
