from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import io
import requests
import pytesseract


app = FastAPI(title="Scan & Discover")


# ============================================================
# YOLO
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# WEB PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return """
    <!DOCTYPE html>

    <html lang="en">

    <head>

        <meta charset="UTF-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <title>Scan & Discover</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 30px 20px;
            }

            h1 {
                font-size: 32px;
                margin-bottom: 10px;
            }

            p {
                color: #bbb;
            }

            #photoButton {
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 18px 30px;
                border-radius: 12px;
                font-size: 20px;
                cursor: pointer;
                margin-top: 20px;
            }

            #photo {
                display: none;
            }

            #preview {
                display: none;
                width: 100%;
                max-width: 500px;
                margin: 25px auto;
                border-radius: 12px;
            }

            #loading {
                display: none;
                color: #3498db;
                font-size: 18px;
                margin: 20px;
            }

            #result {
                background: #222;
                padding: 20px;
                margin: 25px auto;
                max-width: 600px;
                border-radius: 12px;
                text-align: left;
                line-height: 1.6;
                white-space: pre-line;
            }

        </style>

    </head>


    <body>

        <h1>🔎 Scan & Discover</h1>

        <p>
            Take a photo of an object and discover its history.
        </p>


        <label id="photoButton" for="photo">
            📷 Take Photo
        </label>


        <input
            id="photo"
            type="file"
            accept="image/*"
            capture="environment"
        >


        <img
            id="preview"
            alt="Photo preview"
        >


        <div id="loading">
            🔍 Analysing photo...
        </div>


        <div id="result">
            Take a photo to begin.
        </div>


        <script>

            const photoInput =
                document.getElementById("photo");

            const preview =
                document.getElementById("preview");

            const result =
                document.getElementById("result");

            const loading =
                document.getElementById("loading");


            photoInput.addEventListener(
                "change",
                async function () {

                    const file = this.files[0];

                    if (!file) {
                        return;
                    }


                    // Show photo

                    preview.src =
                        URL.createObjectURL(file);

                    preview.style.display =
                        "block";


                    result.innerText = "";

                    loading.style.display =
                        "block";


                    // Upload photo

                    const formData =
                        new FormData();

                    formData.append(
                        "file",
                        file,
                        "photo.jpg"
                    );


                    try {

                        const response =
                            await fetch(
                                "/scan",
                                {
                                    method: "POST",
                                    body: formData
                                }
                            );


                        const data =
                            await response.json();


                        if (data.success) {

                            result.innerText =
                                data.message;

                        } else {

                            result.innerText =
                                "❌ " +
                                data.message;

                        }


                    } catch (error) {

                        console.error(error);

                        result.innerText =
                            "❌ Could not connect to the server.";

                    }


                    loading.style.display =
                        "none";

                }
            );

        </script>

    </body>

    </html>
    """


# ============================================================
# OCR
# ============================================================

def read_text(image):

    """
    Reads text from the photograph.
    """

    try:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Improve text visibility
        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2
        )

        gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        text = pytesseract.image_to_string(
            gray
        )

        return text.strip()

    except Exception as error:

        print("OCR error:", error)

        return ""


# ============================================================
# WIKIPEDIA SEARCH
# ============================================================

def search_wikipedia(search_text):

    """
    Search Wikipedia for the identified object.
    """

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

    results = data.get(
        "query",
        {}
    ).get(
        "search",
        []
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

    pages = data.get(
        "query",
        {}
    ).get(
        "pages",
        {}
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

    """
    Get creator/inventor and date information
    from Wikidata.
    """

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

    entity = data.get(
        "entities",
        {}
    ).get(
        wikidata_id
    )

    if not entity:
        return {}

    claims = entity.get(
        "claims",
        {}
    )


    # --------------------------------------------------------
    # DATE
    # P571 = inception
    # --------------------------------------------------------

    year = None

    if "P571" in claims:

        try:

            value = claims["P571"][0][
                "mainsnak"
            ]["datavalue"]["value"]["time"]

            year = value[1:5]

        except Exception:
            pass


    # --------------------------------------------------------
    # CREATOR
    # P170 = creator
    # P61  = discoverer/inventor
    # --------------------------------------------------------

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

    """
    Try to identify the actual object.

    OCR gets priority because text such as a brand,
    product name or model number is much more specific
    than a generic YOLO class.
    """

    search_text = ""


    # OCR text

    if ocr_text:

        # Keep only useful lines
        lines = []

        for line in ocr_text.splitlines():

            line = line.strip()

            if len(line) >= 3:

                lines.append(line)


        if lines:

            search_text = " ".join(
                lines[:5]
            )


    # If OCR didn't find anything,
    # use YOLO object names.

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
# SCAN
# ============================================================

@app.post("/scan")
async def scan(
    file: UploadFile = File(...)
):

    try:

        # ====================================================
        # READ PHOTO
        # ====================================================

        image_data = await file.read()

        if not image_data:

            raise ValueError(
                "The photo is empty."
            )


        # ====================================================
        # PILLOW
        # ====================================================

        pil_image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")


        print(
            "Photo:",
            pil_image.size
        )


        # ====================================================
        # NUMPY
        # ====================================================

        image_array = np.array(
            pil_image
        )


        # ====================================================
        # OPENCV
        # ====================================================

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )


        # ====================================================
        # YOLO
        # ====================================================

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


        # Remove duplicates

        yolo_objects = list(
            dict.fromkeys(
                yolo_objects
            )
        )


        print(
            "YOLO:",
            yolo_objects
        )


        # ====================================================
        # OCR
        # ====================================================

        ocr_text = read_text(
            frame
        )


        print(
            "OCR:",
            ocr_text
        )


        # ====================================================
        # INFORMATION LOOKUP
        # ====================================================

        information = identify_object(
            ocr_text,
            yolo_objects
        )


        # ====================================================
        # BUILD RESULT
        # ====================================================

        lines = []


        # Objects

        if yolo_objects:

            lines.append(
                "🤖 Objects detected:"
            )

            for obj in yolo_objects:

                lines.append(
                    f"• {obj}"
                )

            lines.append("")


        # OCR

        if ocr_text:

            lines.append(
                "🔤 Text found:"
            )

            lines.append(
                ocr_text
            )

            lines.append("")


        # Information

        if information:

            lines.append(
                "📚 Information:"
            )

            lines.append(
                f"Name: {information['name']}"
            )


            if information.get(
                "year"
            ):

                lines.append(
                    f"Year: {information['year']}"
                )


            if information.get(
                "creator"
            ):

                lines.append(
                    "Creator / inventor: "
                    + information["creator"]
                )


            lines.append("")


            if information.get(
                "history"
            ):

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
            "success": True,
            "message": message,
            "objects": yolo_objects,
            "ocr": ocr_text,
            "information": information
        }


    except Exception as error:

        print(
            "SCAN ERROR:",
            error
        )


        return {
            "success": False,
            "message": str(error)
        }
