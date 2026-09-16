import cv2
import numpy as np

from PIL import Image
from ultralytics import YOLO
import easyocr


# ============================================================
# AI MODELS
# ============================================================

model = YOLO("yolo11n.pt")

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)


# ============================================================
# OCR
# ============================================================

def read_text(image):

    try:

        results = ocr_reader.readtext(image)

        text_parts = []

        for detection in results:

            text = detection[1]
            confidence = detection[2]

            if confidence >= 0.40:
                text_parts.append(text)

        return " ".join(text_parts).strip()

    except Exception as error:

        print("OCR error:", error)

        return ""


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image_data):

    # Convert uploaded bytes to NumPy
    image_array = np.frombuffer(
        image_data,
        dtype=np.uint8
    )

    # Decode image
    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError(
            "Could not decode image."
        )

    print(
        "Image received successfully."
    )

    print(
        "Image size:",
        frame.shape
    )

    # ========================================================
    # YOLO
    # ========================================================

    results = model(frame)

    yolo_objects = []

    for result in results:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < 0.40:
                continue

            class_id = int(
                box.cls[0]
            )

            name = result.names[class_id]

            yolo_objects.append(name)

    # Remove duplicates
    yolo_objects = list(
        dict.fromkeys(yolo_objects)
    )

    print(
        "YOLO:",
        yolo_objects
    )

    # ========================================================
    # OCR
    # ========================================================

    ocr_text = read_text(frame)

    print(
        "OCR:",
        ocr_text
    )

    # ========================================================
    # RESULT
    # ========================================================

    lines = []

    if yolo_objects:

        lines.append(
            "🤖 Objects detected:"
        )

        for obj in yolo_objects:

            lines.append(
                f"• {obj}"
            )

        lines.append("")

    if ocr_text:

        lines.append(
            "🔤 Text found:"
        )

        lines.append(
            ocr_text
        )

    if not lines:

        lines.append(
            "ℹ️ I couldn't identify anything."
        )

    return "\n".join(lines)