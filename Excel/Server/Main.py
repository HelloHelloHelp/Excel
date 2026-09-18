from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pathlib
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load YOLO once when the server starts
model = YOLO("yolo11n.pt")


@app.get("/", response_class=FileResponse)
async def home():
    # Resolve index.html next to this file in ./static/index.html
    index_path = pathlib.Path(__file__).resolve().parent / "Server" / "index.html"
    return FileResponse(index_path, media_type="text/html")


@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    try:
        # Read uploaded photo
        image_data = await file.read()

        # Convert photo to NumPy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        # Decode image with OpenCV
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode the photo.")

        print("Photo received:", frame.shape)

        # YOLO ANALYSIS
        results = model(frame)

        detected = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                name = result.names[class_id]
                detected.append(f"{name} ({confidence * 100:.0f}%)")

        # RESULT
        if not detected:
            message = "I couldn't identify any objects."
        else:
            message = "I found: " + ", ".join(detected)

        return {"success": True, "message": message}

    except Exception as error:
        print("Scan error:", error)
        return {"success": False, "message": str(error)}
