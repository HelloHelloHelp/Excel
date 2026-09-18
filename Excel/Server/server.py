from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import pathlib
import base64
import json
import os

# Attempt relative import first (for local dev) then absolute for uvicorn run-from-root
try:
    from .camera import process_image
except Exception:
    from camera import process_image

app = FastAPI()

# Allow CORS for the web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = pathlib.Path(__file__).resolve().parent


@app.get("/")
async def index():
    html_path = ROOT_DIR / "index.html"
    return FileResponse(html_path)


@app.post('/process')
async def process(request: Request, file: UploadFile = File(...)):
    try:
        data = await file.read()
        res = process_image(data)
        return JSONResponse(status_code=200, content=res)
    except Exception as e:
        logging.exception('processing failed')
        return JSONResponse(status_code=500, content={'success': False, 'message': str(e)})


@app.post('/scan_json')
async def scan_json(request: Request):
    """Accept JSON {"image":"data:image/...;base64,...."} and return process_image output directly."""
    try:
        body = await request.body()
        payload = json.loads(body.decode('utf-8'))
        img_b64 = payload.get('image')
        if not img_b64:
            return JSONResponse(status_code=400, content={'success': False, 'message': 'no image field'})
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]
        img_bytes = base64.b64decode(img_b64)

        res = process_image(img_bytes)
        # Return whatever process_image returns, including debug info
        return JSONResponse(status_code=200, content=res)
    except Exception as e:
        logging.exception('scan_json failed')
        return JSONResponse(status_code=500, content={'success': False, 'message': str(e)})


@app.post('/upload_template')
async def upload_template(file: UploadFile = File(...)):
    """Upload a small template image (e.g., hp logo) to ./templates/hp_logo.png"""
    try:
        content = await file.read()
        tpl_dir = ROOT_DIR / 'templates'
        tpl_dir.mkdir(parents=True, exist_ok=True)
        tpl_path = tpl_dir / 'hp_logo.png'
        with open(tpl_path, 'wb') as f:
            f.write(content)
        return JSONResponse(status_code=200, content={'success': True, 'message': 'template uploaded', 'path': str(tpl_path)})
    except Exception as e:
        logging.exception('upload_template failed')
        return JSONResponse(status_code=500, content={'success': False, 'message': str(e)})