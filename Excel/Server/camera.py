import cv2
import numpy as np
from ultralytics import YOLO

import requests
import easyocr


# ============================================================
# AI MODELS
# ============================================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Loading OCR...")

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("AI models loaded.")


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
# WIKIPEDIA SEARCH
# ============================================================

def search_wikipedia(search_text):

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_text,
        "format": "json",
        "utf8": 1,
        "srlimit": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=10,
        headers={
            "User-Agent": "ScanAndDiscover/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    results = (
        data.get("query", {})
        .get("search", [])
    )

    if not results:
        return None

    return results[0]["title"]


# ============================================================
# WIKIPEDIA PAGE
# ============================================================

def get_wikipedia_page(title):

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "prop": "extracts|pageprops",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
        "utf8": 1
    }

    response = requests.get(
        url,
        params=params,
        timeout=10,
        headers={
            "User-Agent": "ScanAndDiscover/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    pages = (
        data.get("query", {})
        .get("pages", {})
    )

    for page in pages.values():

        return {
            "title": page.get(
                "title",
                title
            ),

            "description": page.get(
                "extract",
                ""
            ),

            "wikidata_id": page.get(
                "pageprops",
                {}
            ).get(
                "wikibase_item"
            )
        }

    return None


# ============================================================
# WIKIDATA
# ============================================================

def get_wikidata_information(wikidata_id):

    if not wikidata_id:
        return {}

    url = (
        "https://www.wikidata.org/wiki/Special:EntityData/"
        + wikidata_id
        + ".json"
    )

    response = requests.get(
        url,
        timeout=10,
        headers={
            "User-Agent": "ScanAndDiscover/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    entity = (
        data.get("entities", {})
        .get(wikidata_id)
    )

    if not entity:
        return {}

    claims = entity.get(
        "claims",
        {}
    )

    # ========================================================
    # DATE
    # P571 = inception
    # ========================================================

    year = None

    if "P571" in claims:

        try:

            value = claims["P571"][0][
                "mainsnak"
            ]["datavalue"]["value"]["time"]

            year = value[1:5]

        except Exception:
            pass

    # ========================================================
    # CREATOR
    # P170 = creator
    # P61 = discoverer/inventor
    # ========================================================

    person_id = None

    for property_id in ["P170", "P61"]:

        if property_id in claims:

            try:

                person_id = claims[
                    property_id
                ][0]["mainsnak"][
                    "datavalue"
                ]["value"]["id"]

                break

            except Exception:
                pass

    person_name = None

    if person_id:

        person_url = (
            "https://www.wikidata.org/wiki/Special:EntityData/"
            + person_id
            + ".json"
        )

        try:

            person_response = requests.get(
                person_url,
                timeout=10,
                headers={
                    "User-Agent": "ScanAndDiscover/1.0"
                }
            )

            person_response.raise_for_status()

            person_data = person_response.json()

            person_entity = person_data[
                "entities"
            ][person_id]

            labels = person_entity.get(
                "labels",
                {}
            )

            person_name = (
                labels.get("en", {})
                .get("value")
            )

        except Exception:
            pass

    return {
        "year": year,
        "creator": person_name
    }


# ============================================================
# IDENTIFICATION
# ============================================================

def identify_object(ocr_text, yolo_objects):

    search_text = ""

    # ========================================================
    # OCR
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

    if not search_text and yolo_objects:

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

        wikidata = get_wikidata_information(
            page.get("wikidata_id")
        )

        return {
            "name": page.get("title"),

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

    if max(frame.shape[:2]) > MAX_SIZE:

        scale = (
            MAX_SIZE /
            max(frame.shape[:2])
        )

        new_width = int(
            frame.shape[1] * scale
        )

        new_height = int(
            frame.shape[0] * scale
        )

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    print(
        "Image size after resizing:",
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

    message = "\n".join(
        lines
    )

    return {
        "message": message,
        "objects": yolo_objects,
        "ocr": ocr_text,
        "information": information
    }