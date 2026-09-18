import cv2
import numpy as np

from ultralytics import YOLO

import requests


# ============================================================
# YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded.")


# ============================================================
# OCR
# ============================================================

# Do NOT load EasyOCR when the server starts.
# This saves RAM during Render startup.

ocr_reader = None

try:
    import easyocr
    EASY_OCR_AVAILABLE = True
except Exception:
    easyocr = None
    EASY_OCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    Image = None
    PYTESSERACT_AVAILABLE = False


def get_ocr_reader():

    global ocr_reader

    if ocr_reader is None:

        print("Loading EasyOCR...")

        import easyocr

        ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )

        print("EasyOCR loaded.")

    return ocr_reader


def recognize_text_from_image(image):
    """Attempt OCR using EasyOCR first, fallback to pytesseract.

    Args:
        image: OpenCV BGR image (numpy array)
    Returns:
        text: Recognized text (string)
    """
    # Try EasyOCR
    if EASY_OCR_AVAILABLE:
        try:
            # easyocr expects RGB
            img_rgb = image[:, :, ::-1]
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(img_rgb)
            # results is list of (bbox, text, confidence)
            texts = [r[1] for r in results if r and len(r) > 1]
            return "\n".join(texts).strip()
        except Exception:
            pass

    # Fallback to pytesseract
    if PYTESSERACT_AVAILABLE and Image is not None:
        try:
            img_rgb = image[:, :, ::-1]
            pil = Image.fromarray(img_rgb)
            text = pytesseract.image_to_string(pil, config='--psm 6')
            return text.strip()
        except Exception:
            pass

    return ""


# ============================================================
# IDENTIFICATION
# ============================================================

def identify_object(
    ocr_text,
    yolo_objects
):

    search_text = ""

    # ========================================================
    # OCR TEXT
    # ========================================================

    if ocr_text:

        lines = []

        for line in ocr_text.splitlines():

            line = line.strip()

            if len(line) >= 3:

                lines.append(line)

        if lines:

            search_text = " ".join(
                lines[:5]
            )

    # ========================================================
    # YOLO FALLBACK
    # ========================================================

    if (
        not search_text
        and yolo_objects
    ):

        search_text = " ".join(
            yolo_objects[:3]
        )

    if not search_text:

        return None

    print(
        "Searching Wikipedia for:",
        search_text
    )

    try:

        title = search_wikipedia(
            search_text
        )

        if not title:

            return None

        page = get_wikipedia_page(
            title
        )

        if not page:

            return None

        wikidata = (
            get_wikidata_information(
                page.get("wikidata_id")
            )
        )

        return {

            "name": page.get(
                "title"
            ),

            "history": page.get(
                "description"
            ),

            "year": wikidata.get(
                "year"
            ),

            "creator": wikidata.get(
                "creator"
            )
        }

    except Exception as error:

        print(
            "Information lookup error:",
            error
        )

        return None


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image_data):

    # ========================================================
    # READ IMAGE
    # ========================================================

    image_array = np.frombuffer(
        image_data,
        dtype=np.uint8
    )

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
        "Original image size:",
        frame.shape
    )

    # ========================================================
    # REDUCE LARGE PHONE IMAGES
    # ========================================================

    MAX_SIZE = 1280

    original_max_size = max(
        frame.shape[:2]
    )

    if original_max_size > MAX_SIZE:

        scale = (
            MAX_SIZE /
            original_max_size
        )

        new_width = int(
            frame.shape[1] * scale
        )

        new_height = int(
            frame.shape[0] * scale
        )

        frame = cv2.resize(
            frame,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

    print(
        "Image size after resizing:",
        frame.shape
    )

    # ========================================================
    # YOLO
    # ========================================================

    print("Running YOLO...")

    results = model(
        frame,
        verbose=False
    )

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

            name = result.names[
                class_id
            ]

            yolo_objects.append(
                name
            )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    yolo_objects = list(
        dict.fromkeys(
            yolo_objects
        )
    )

    print(
        "YOLO:",
        yolo_objects
    )

    # ========================================================
    # OCR
    # ========================================================

    print("Running OCR...")

    ocr_text = read_text(
        frame
    )

    print(
        "OCR:",
        ocr_text
    )

    # ========================================================
    # INFORMATION LOOKUP
    # ========================================================

    information = identify_object(
        ocr_text,
        yolo_objects
    )

    # ========================================================
    # BUILD RESULT
    # ========================================================

    lines = []

    # ========================================================
    # OBJECTS
    # ========================================================

    if yolo_objects:

        lines.append(
            "🤖 Objects detected:"
        )

        for obj in yolo_objects:

            lines.append(
                f"• {obj}"
            )

        lines.append("")

    # ========================================================
    # OCR
    # ========================================================

    if ocr_text:

        lines.append(
            "🔤 Text found:"
        )

        lines.append(
            ocr_text
        )

        lines.append("")

    # ========================================================
    # INFORMATION
    # ========================================================

    if information:

        lines.append(
            "📚 Information:"
        )

        lines.append(
            f"Name: {information['name']}"
        )

        if information.get("year"):

            lines.append(
                f"Year: {information['year']}"
            )

        if information.get("creator"):

            lines.append(
                "Creator / inventor: "
                + information["creator"]
            )

        lines.append("")

        if information.get("history"):

            lines.append(
                "History:"
            )

            lines.append(
                information["history"]
            )

    else:

        lines.append(
            "ℹ️ I could not find reliable "
            "historical information for "
            "this object."
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    message = "\n".join(
        lines
    )

    return {

        "message": message,

        "objects": yolo_objects,

        "ocr": ocr_text,

        "information": information
    }