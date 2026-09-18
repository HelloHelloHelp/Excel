from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pathlib
import cv2
import numpy as np
from ultralytics import YOLO

# Added imports
import os
import requests
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None

app = FastAPI()

# Load YOLO once when the server starts
model = YOLO("yolo11n.pt")


@app.get("/", response_class=FileResponse)
async def home():
    # Resolve index.html next to this file in ./static/index.html
    index_path = pathlib.Path(__file__).resolve().parent / "static" / "index.html"
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
async def scan(file: UploadFile = File(...)
    if pytesseract is None:
        return {"success": False, "message": "pytesseract not available: install pytesseract and Tesseract engine"}

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

                # Encode crop to bytes for visual search if needed
                try:
                    _, crop_jpg = cv2.imencode('.jpg', crop)
                    crop_bytes = crop_jpg.tobytes()
                except Exception:
                    crop_bytes = None

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

                web_results = []
                # If OCR gave a query use that to search web; otherwise try visual search with the crop
                if query:
                    web_results = web_search(query, top_k=3, image_bytes=None)
                    # if web results empty and we have image, try visual search
                    if not web_results and crop_bytes:
                        web_results = web_search(None, top_k=3, image_bytes=crop_bytes)
                else:
                    # No OCR text: try visual search (Bing) first, fallback to class name text search
                    if crop_bytes and BING_API_KEY:
                        web_results = web_search(None, top_k=3, image_bytes=crop_bytes)
                    if not web_results and class_name:
                        web_results = web_search(class_name, top_k=3, image_bytes=None)

                det = {
                    "class": class_name,
                    "confidence": round(confidence, 3) if confidence is not None else None,
                    "ocr": ocr_text,
                    "query_used": query if query else ("visual_search" if (crop_bytes and BING_API_KEY) else class_name),
                    "web_results": web_results
                }
                output["detections"].append(det)

        return {"success": True, "data": output}

    except Exception as error:
        print("Scan error:", error)
        return {"success": False, "message": str(error)}
