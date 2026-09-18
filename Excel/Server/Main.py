from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
async def scan(file: UploadFile = File(...)):
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
