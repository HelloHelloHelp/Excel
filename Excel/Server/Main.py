from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import pathlib
import cv2
import numpy as np
from ultralytics import YOLO

# Added imports
import os
import logging
import requests
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None

# Import our connection/search helpers: prefer relative import when running as a package
try:
    from . import connect
except Exception:
    import connect

app = FastAPI()

# Enable simple CORS for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic logging
logging.basicConfig(level=logging.INFO)

# Load YOLO once when the server starts
model = YOLO("yolo11n.pt")


@app.get("/", response_class=FileResponse)
async def home():
    # Resolve index.html next to this file in ./Server/index.html
    index_path = pathlib.Path(__file__).resolve().parent / "Server" / "index.html"
    return FileResponse(index_path, media_type="text/html")


# Simple web search: Bing if API key provided, fallback to Wikipedia
BING_API_KEY = os.environ.get("BING_API_KEY")


def web_search(query, top_k=3):
    docs = []
    if not query:
        return docs

    if BING_API_KEY:
        try:
            headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
            params = {"q": query, "count": top_k}
            r = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            for item in data.get("webPages", {}).get("value", [])[:top_k]:
                docs.append({"name": item.get("name"), "snippet": item.get("snippet"), "url": item.get("url")})
            return docs
        except Exception:
            # fall through to Wikipedia fallback
            pass

    # Wikipedia fallback
    try:
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": top_k}
        r = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for s in data.get("query", {}).get("search", [])[:top_k]:
            title = s.get("title")
            snippet = s.get("snippet")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            docs.append({"name": title, "snippet": snippet, "url": url})
    except Exception:
        pass

    return docs


@app.post("/scan")
async def scan(request: Request, file: UploadFile = File(...)):
    if pytesseract is None:
        return {"success": False, "message": "pytesseract not available: install pytesseract and Tesseract engine"}

    # Log incoming request basics for debugging connectivity issues
    try:
        content_type = request.headers.get("content-type")
        content_length = request.headers.get("content-length")
        logging.info("/scan invoked from %s content-type=%s content-length=%s", request.client.host if request.client else "-", content_type, content_length)
    except Exception:
        logging.exception("Failed to log request headers for /scan")

    try:
        # Read uploaded photo
        image_data = await file.read()

        # Convert photo to NumPy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        # Decode image with OpenCV
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode the photo.")

        # Run YOLO detection
        results = model(frame)

        output = {"detections": []}

        img_h, img_w = frame.shape[:2]

        for result in results:
            # result.boxes contains the detected boxes for this image
            for box in result.boxes:
                # Extract bounding box coordinates robustly
                xy = None
                try:
                    # ultralytics v8+: box.xyxy is a tensor-like with shape (1,4)
                    xy = box.xyxy[0].tolist()
                except Exception:
                    try:
                        xy = box.xyxy.cpu().numpy().tolist()[0]
                    except Exception:
                        # as a last resort try box.xyxy itself
                        try:
                            xy = list(box.xyxy)
                        except Exception:
                            xy = None

                if not xy:
                    # skip if we can't get coords
                    continue

                x1, y1, x2, y2 = map(int, [max(0, xy[0]), max(0, xy[1]), min(img_w - 1, xy[2]), min(img_h - 1, xy[3])])

                # Crop the detected region
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]

                # OCR on crop (BGR -> RGB)
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    # Use a conservative config to get single-line or block text
                    ocr_text = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
                except Exception:
                    ocr_text = ""

                # Class and confidence
                try:
                    class_id = int(box.cls[0])
                except Exception:
                    # fallback: if cls not accessible
                    class_id = None
                try:
                    confidence = float(box.conf[0])
                except Exception:
                    confidence = None

                class_name = result.names.get(class_id, str(class_id)) if class_id is not None else ""

                # Choose query: prefer OCR text when it looks meaningful
                query = ""
                if ocr_text and len(ocr_text) >= 3:
                    query = ocr_text
                elif class_name:
                    query = class_name

                web_results = web_search(query) if query else []

                det = {
                    "class": class_name,
                    "confidence": round(confidence, 3) if confidence is not None else None,
                    "ocr": ocr_text,
                    "query_used": query,
                    "web_results": web_results
                }
                output["detections"].append(det)

        return {"success": True, "data": output}

    except Exception as error:
        logging.exception("Scan error")
        return JSONResponse(status_code=500, content={"success": False, "message": str(error)})


class ImagePayload(BaseModel):
    image: str


@app.post("/scan_json")
async def scan_json(request: Request, payload: ImagePayload):
    """Accept JSON with a base64-encoded image string in payload.image
    Example: {"image": "data:image/jpeg;base64,/9j/4AAQ..."}
    Enhanced search strategy:
      - Try full-image visual search first
      - If no good visual results, run YOLO detections and try multiple augmented crops with visual search
      - If still no visual matches, build richer text queries (OCR + class name) and run web search
    """
    if pytesseract is None:
        return JSONResponse(status_code=500, content={"success": False, "message": "pytesseract not available"})

    try:
        logging.info("/scan_json invoked from %s", request.client.host if request.client else "-")

        img_b64 = payload.image
        # Strip data URL prefix if present
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        image_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the photo.")

        img_h, img_w = frame.shape[:2]

        # 1) Try full-image visual search (best for product-level matches)
        try:
            s_full = connect.search(query=None, image_bytes=img_bytes, top_k=5)
            logging.info("full-image visual search result: %s", s_full.get("source") if isinstance(s_full, dict) else str(type(s_full)))
            if s_full.get("success") and s_full.get("results"):
                return JSONResponse(status_code=200, content={"success": True, "data": {"source": "full_image_visual", "results": s_full.get("results")}})
        except Exception:
            logging.exception("full-image visual search failed")

        # 2) Run YOLO detections and attempt visual/text search per detection with augmentations
        results = model(frame)
        output = {"detections": []}

        def augment_crops(crop):
            """Yield augmented versions of crop: original, scaled, rotated"""
            crops = [crop]
            h, w = crop.shape[:2]
            # scales
            for scale in (0.9, 1.1):
                try:
                    nh = max(10, int(h * scale))
                    nw = max(10, int(w * scale))
                    scaled = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    crops.append(scaled)
                except Exception:
                    pass
            # rotations
            for ang in (-15, 15):
                try:
                    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
                    rotated = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    crops.append(rotated)
                except Exception:
                    pass
            # center crop
            try:
                cy, cx = h//2, w//2
                ch, cw = max(10, h//2), max(10, w//2)
                y1 = max(0, cy - ch//2); x1 = max(0, cx - cw//2)
                center = crop[y1:y1+ch, x1:x1+cw]
                if center.size:
                    crops.append(center)
            except Exception:
                pass
            # dedupe by shape
            unique = []
            shapes = set()
            for c in crops:
                shapes.add(c.shape)
            for c in crops:
                if c.shape in shapes:
                    unique.append(c)
                    shapes.remove(c.shape)
            return unique

        for result in results:
            for box in result.boxes:
                # robustly extract coords
                xy = None
                try:
                    xy = box.xyxy[0].tolist()
                except Exception:
                    try:
                        xy = box.xyxy.cpu().numpy().tolist()[0]
                    except Exception:
                        try:
                            xy = list(box.xyxy)
                        except Exception:
                            xy = None
                if not xy:
                    continue
                x1, y1, x2, y2 = map(int, [max(0, xy[0]), max(0, xy[1]), min(img_w - 1, xy[2]), min(img_h - 1, xy[3])])
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]

                # OCR on crop
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    ocr_text = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
                except Exception:
                    ocr_text = ""

                try:
                    class_id = int(box.cls[0])
                except Exception:
                    class_id = None
                try:
                    confidence = float(box.conf[0])
                except Exception:
                    confidence = None

                class_name = result.names.get(class_id, str(class_id)) if class_id is not None else ""

                web_results = []
                used_query = ""

                # Try augmented visual searches for this crop
                aug_crops = augment_crops(crop)
                for ac in aug_crops:
                    try:
                        _, ac_jpg = cv2.imencode('.jpg', ac)
                        ac_bytes = ac_jpg.tobytes()
                    except Exception:
                        ac_bytes = None
                    if not ac_bytes:
                        continue
                    try:
                        s = connect.search(query=None, image_bytes=ac_bytes, top_k=5)
                        if s.get('success') and s.get('results'):
                            web_results = s.get('results')
                            used_query = 'visual_crop'
                            logging.info('visual crop search succeeded for class %s', class_name)
                            break
                    except Exception:
                        logging.exception('visual crop search failed')
                # If no visual results, try richer text queries
                if not web_results:
                    queries = []
                    if ocr_text:
                        queries.append(ocr_text)
                        queries.append(f"{ocr_text} {class_name}" if class_name else ocr_text)
                    if class_name:
                        queries.append(class_name)
                        queries.append(f"{class_name} product")
                        queries.append(f"{class_name} brand")
                    # try queries (unique)
                    seenq = set()
                    for q in queries:
                        if not q:
                            continue
                        if q in seenq:
                            continue
                        seenq.add(q)
                        try:
                            s = connect.search(query=q, image_bytes=None, top_k=5)
                            if s.get('success') and s.get('results'):
                                web_results = s.get('results')
                                used_query = q
                                logging.info('text search succeeded for query "%s"', q)
                                break
                        except Exception:
                            logging.exception('text search failed for query %s', q)

                det = {
                    'class': class_name,
                    'confidence': round(confidence, 3) if confidence is not None else None,
                    'ocr': ocr_text,
                    'query_used': used_query,
                    'web_results': web_results,
                }
                output['detections'].append(det)

        # If no detections or no web results found at all, try a final fallback: class-level full image text search
        any_results = any(d.get('web_results') for d in output.get('detections', []))
        if not any_results:
            # try class-level or generic query from model names
            fallback_queries = []
            # use top class names detected
            top_names = [d['class'] for d in output.get('detections', []) if d.get('class')]
            if top_names:
                fallback_queries.extend(top_names[:3])
            # also try generic queries
            fallback_queries.append('product')
            fallback_results = []
            for q in fallback_queries:
                try:
                    s = connect.search(query=q, image_bytes=None, top_k=5)
                    if s.get('success') and s.get('results'):
                        fallback_results = s.get('results')
                        logging.info('fallback text search succeeded for %s', q)
                        break
                except Exception:
                    logging.exception('fallback search failed')
            if fallback_results:
                return JSONResponse(status_code=200, content={"success": True, "data": {"source": "fallback_text", "results": fallback_results, "detections": output}})

        return JSONResponse(status_code=200, content={"success": True, "data": {"source": "per_detection", "detections": output}})

    except Exception as error:
        logging.exception("scan_json error")
        return JSONResponse(status_code=500, content={"success": False, "message": str(error)})


# Debug echo endpoint to log headers and a small preview of the body
@app.post('/debug_echo')
async def debug_echo(request: Request):
    try:
        headers = dict(request.headers)
        body = await request.body()
        preview = body[:1024].decode('utf-8', errors='replace')
        logging.info("debug_echo headers=%s", headers)
        logging.info("debug_echo body_preview=%s", preview)
        return JSONResponse(status_code=200, content={"ok": True, "headers": headers, "body_preview": preview})
    except Exception as e:
        logging.exception('debug_echo failed')
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/health")
async def health():
    """Health endpoint: checks internet, Bing key, OCR and model status."""
    try:
        inet = connect.check_internet()
    except Exception as e:
        inet = {"ok": False, "error": str(e)}

    status = {
        "internet": inet,
        "bing_api_key_present": bool(connect.BING_API_KEY),
        "pytesseract_available": pytesseract is not None,
        "model_loaded": model is not None,
    }

    return JSONResponse(status_code=200, content={"status": "ok", "details": status})


@app.get("/ping")
async def ping():
    """Simple reachability check for clients."""
    return JSONResponse(status_code=200, content={"ok": True, "service": "scananddiscover"})