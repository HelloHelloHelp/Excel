import os
# Limit native threads to reduce memory footprint on small hosts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Reduce default image sizes aggressively for low-memory hosts
DEFAULT_MAX_DIM = int(os.environ.get("MAX_DIM", "112"))
DEFAULT_IMG_SIZE = int(os.environ.get("IMG_SIZE", "112"))

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import pathlib
import cv2
import numpy as np

# defer ultralytics import until needed

# Added imports
import logging
import requests
import gc
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None

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

# Import our connection/search helpers (use relative import inside package)
from . import connect

# Lazy-load YOLO model to avoid allocating heavy memory at import time
_model = None

def get_model():
    """Load and return the YOLO model only when USE_LOCAL_MODEL=1.

    This function will not import ultralytics unless explicitly enabled via
    the USE_LOCAL_MODEL environment variable to avoid importing Torch on
    low-memory hosts.
    """
    global _model
    if os.environ.get("USE_LOCAL_MODEL", "0") != "1":
        raise RuntimeError("Local model use is disabled (USE_LOCAL_MODEL!=1)")
    if _model is None:
        from ultralytics import YOLO as _YOLO
        model_name = os.environ.get("YOLO_MODEL", "yolo8n.pt")
        logging.info("Loading YOLO model (lazy) name=%s...", model_name)
        _model = _YOLO(model_name)
        logging.info("YOLO model loaded: %s", model_name)
    return _model


@app.get("/", response_class=FileResponse)
async def home():
    # Resolve index.html next to this file in ./static/index.html
    index_path = pathlib.Path(__file__).resolve().parent / "index.html"
    return FileResponse(index_path, media_type="text/html")


# Simple web search: Bing if API key provided, fallback to Wikipedia
BING_API_KEY = os.environ.get("BING_API_KEY")


def bing_visual_search(image_bytes, top_k=5):
    """Call Bing Visual Search with image bytes. Return list of dicts with name/snippet/url."""
    if not BING_API_KEY or not image_bytes:
        return []
    try:
        url = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        files = {
            'image': ('image.jpg', image_bytes, 'application/octet-stream')
        }
        r = requests.post(url, headers=headers, files=files, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = []
        # Parse tags -> actions -> data -> value
        for tag in data.get('tags', []):
            for action in tag.get('actions', []):
                data_nodes = action.get('data', {})
                values = data_nodes.get('value', []) if isinstance(data_nodes, dict) else []
                for v in values:
                    name = v.get('name') or v.get('hostPageDisplayUrl') or v.get('accentColor')
                    snippet = v.get('snippet') if isinstance(v.get('snippet'), str) else ''
                    urlv = v.get('hostPageDisplayUrl') or v.get('contentUrl') or v.get('webSearchUrl')
                    results.append({"name": name, "snippet": snippet, "url": urlv})
                    if len(results) >= top_k:
                        return results
        return results
    except Exception:
        return []


def web_search(query, top_k=3, image_bytes=None):
    """If image_bytes provided and Bing key available, try visual search first.
    Otherwise prefer Bing Web Search (if key) then fallback to Wikipedia."""
    docs = []
    if image_bytes and BING_API_KEY:
        vs = bing_visual_search(image_bytes, top_k=top_k)
        if vs:
            return vs

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
        return JSONResponse(status_code=500, content={"success": False, "message": "pytesseract not available: install pytesseract and Tesseract engine"})

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

        # Resize to reduce memory (use very small max dim)
        MAX_DIM = DEFAULT_MAX_DIM
        h, w = frame.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Decide whether to use local YOLO model or remote visual search
        # KEEP_MODEL controls whether a loaded model should persist between requests
        KEEP_MODEL = os.environ.get('KEEP_MODEL_IN_MEMORY', '0') == '1'
        USE_LOCAL = os.environ.get("USE_LOCAL_MODEL", "0") == "1"
        # small inference size (used when running local model)
        IMG_SIZE = int(os.environ.get("IMG_SIZE", "160"))

        # Keep original bytes for remote visual search attempts
        img_bytes = image_data

        if not USE_LOCAL:
            # 1) Try full-image visual search (best for product-level matches)
            try:
                s_full = connect.search(query=None, image_bytes=img_bytes, top_k=5)
                if isinstance(s_full, dict) and s_full.get("success") and s_full.get("results"):
                    top = s_full.get("results")[0]
                    message = f"Possible match: {top.get('name') or top.get('url')}"
                    # free memory and return
                    del frame, image_array, image_data
                    gc.collect()
                    return {"success": True, "message": message, "source": "full_image_visual", "results": s_full.get("results")}
            except Exception:
                logging.exception("full-image visual search failed in /scan")

            # 2) Try a few center crops with visual search
            def center_crop(img, size):
                h, w = img.shape[:2]
                if h < size or w < size:
                    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                cy, cx = h // 2, w // 2
                y1 = max(0, cy - size // 2)
                x1 = max(0, cx - size // 2)
                return img[y1:y1+size, x1:x1+size]

            try:
                for s in (IMG_SIZE, max(128, IMG_SIZE//2)):
                    try:
                        c = center_crop(frame, s)
                        _, cj = cv2.imencode('.jpg', c)
                        sres = connect.search(query=None, image_bytes=cj.tobytes(), top_k=5)
                        if isinstance(sres, dict) and sres.get('success') and sres.get('results'):
                            top = sres.get('results')[0]
                            message = f"Possible match: {top.get('name') or top.get('url')}"
                            del frame, image_array, image_data
                            gc.collect()
                            return {"success": True, "message": message, "source": "crop_visual", "results": sres.get('results')}
                    except Exception:
                        logging.exception('crop visual search failed')
            except Exception:
                logging.exception('center crop loop failed')

            # 3) Try OCR on whole image and text search
            try:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ocr_text = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
                if ocr_text and len(ocr_text) > 2:
                    stext = connect.search(query=ocr_text, image_bytes=None, top_k=5)
                    if isinstance(stext, dict) and stext.get('success') and stext.get('results'):
                        top = stext.get('results')[0]
                        message = f"Possible match (from text): {top.get('name') or top.get('url')}"
                        del frame, image_array, image_data
                        gc.collect()
                        return {"success": True, "message": message, "source": "ocr_text", "results": stext.get('results')}
            except Exception:
                logging.exception('OCR/text search failed in /scan')

            # If nothing matched
            message = "Unknown"
        else:
            # Use local YOLO detection path
            if KEEP_MODEL:
                model = get_model()
            else:
                from ultralytics import YOLO as _YOLO_local
                model_name_local = os.environ.get("YOLO_MODEL", "yolo8n.pt")
                model = _YOLO_local(model_name_local)

            # Run local inference (CPU, small img size to reduce memory)
            results = model(frame, imgsz=IMG_SIZE, device='cpu')

            detected = []
            for result in results:
                for box in result.boxes:
                    try:
                        class_id = int(box.cls[0])
                    except Exception:
                        class_id = None
                    try:
                        confidence = float(box.conf[0])
                    except Exception:
                        confidence = None
                    name = result.names[class_id] if class_id is not None and class_id in result.names else str(class_id)
                    if confidence is not None:
                        detected.append(f"{name} ({confidence * 100:.0f}%)")
                    else:
                        detected.append(name)

            if not detected:
                message = "I couldn't identify any objects."
            else:
                message = "I found: " + ", ".join(detected)

        # free memory and optionally unload model
        del frame, image_array, image_data
        if not KEEP_MODEL:
            try:
                del model
            except Exception:
                pass
        gc.collect()

        return {
            "success": True,
            "message": message
        }

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

        # Respect USE_LOCAL_MODEL: if a local model is requested, skip the
        # initial full-image visual search and prefer local detections.
        USE_LOCAL = os.environ.get("USE_LOCAL_MODEL", "0") == "1"

        # 1) Try full-image visual search (best for product-level matches)
        # Only attempt when not using a local model.
        if not USE_LOCAL:
            try:
                s_full = connect.search(query=None, image_bytes=img_bytes, top_k=5)
                logging.info("full-image visual search result: %s", s_full.get("source") if isinstance(s_full, dict) else str(type(s_full)))
                if s_full.get("success") and s_full.get("results"):
                    return JSONResponse(status_code=200, content={"success": True, "data": {"source": "full_image_visual", "results": s_full.get("results")}})
            except Exception:
                logging.exception("full-image visual search failed")
        else:
            logging.info("USE_LOCAL_MODEL enabled - skipping full-image visual search")

        # 2) Run YOLO detections and attempt visual/text search per detection with augmentations
        KEEP_MODEL = os.environ.get('KEEP_MODEL_IN_MEMORY', '0') == '1'
        if KEEP_MODEL:
            model_obj = get_model()
        else:
            from ultralytics import YOLO as _YOLO_local
            model_name_local = os.environ.get("YOLO_MODEL", "yolo8n.pt")
            model_obj = _YOLO_local(model_name_local)

        IMG_SIZE = int(os.environ.get("IMG_SIZE", "160"))
        results = model_obj(frame, imgsz=IMG_SIZE, device='cpu')
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

        if results is None:
            # No local model: attempt visual search on a few crops and OCR-based text search
            output = {"detections": []}
            found_any = False
            # try a couple of crops (center and quarter)
            def make_crops(img):
                h, w = img.shape[:2]
                crops = []
                for size in (DEFAULT_IMG_SIZE, max(64, DEFAULT_IMG_SIZE//2)):
                    if h > 0 and w > 0:
                        cy, cx = h // 2, w // 2
                        y1 = max(0, cy - size // 2); x1 = max(0, cx - size // 2)
                        c = img[y1:y1+size, x1:x1+size]
                        if c.size:
                            crops.append(c)
                # quarter-top-left
                qh, qw = max(10, h//4), max(10, w//4)
                crops.append(img[0:qh, 0:qw])
                return crops

            crops = make_crops(frame)
            for c in crops:
                try:
                    _, cj = cv2.imencode('.jpg', c)
                    sres = connect.search(query=None, image_bytes=cj.tobytes(), top_k=5)
                    if sres.get('success') and sres.get('results'):
                        output['detections'].append({
                            'class': '', 'confidence': None, 'ocr': '', 'query_used': 'visual_crop', 'web_results': sres.get('results')
                        })
                        found_any = True
                except Exception:
                    logging.exception('crop visual search failed in scan_json')

            # OCR on whole image and try text search
            try:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ocr_text = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
                if ocr_text:
                    stext = connect.search(query=ocr_text, image_bytes=None, top_k=5)
                    if stext.get('success') and stext.get('results'):
                        output['detections'].append({'class': '', 'confidence': None, 'ocr': ocr_text, 'query_used': 'ocr_text', 'web_results': stext.get('results')})
                        found_any = True
            except Exception:
                logging.exception('OCR/text search failed in scan_json fallback')

            if found_any:
                del frame, image_array, img_bytes
                gc.collect()
                return JSONResponse(status_code=200, content={"success": True, "data": {"source": "visual_only", "detections": output}})
            # else fall through to try fallback text searches below
        else:
            # local model results handled by detection loop below
            pass

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
                # free memory
                del frame, image_array, img_bytes
                gc.collect()
                return JSONResponse(status_code=200, content={"success": True, "data": {"source": "fallback_text", "results": fallback_results, "detections": output}})

        # free memory and optionally unload model
        del frame, image_array, img_bytes
        if not KEEP_MODEL:
            try:
                del model_obj
            except Exception:
                pass
        gc.collect()
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
