from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from camera import process_image


app = FastAPI(title="Scan & Discover")


BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    return FileResponse(
        BASE_DIR / "index.html"
    )


# ============================================================
# STYLE
# ============================================================

@app.get("/style.css")
async def style():

    return FileResponse(
        BASE_DIR / "style.css"
    )


# ============================================================
# SCAN
# ============================================================

@app.post("/scan")
async def scan(
    file: UploadFile = File(...)
):

    try:

        image_data = await file.read()

        if not image_data:

            raise ValueError(
                "The photo is empty."
            )

        print(
            "Photo received successfully."
        )

        result = process_image(
            image_data
        )

        return JSONResponse({

            "success": True,

            "message": result["message"],

            "objects": result["objects"],

            "ocr": result["ocr"],

            "information": result["information"]

        })


    except Exception as error:

        print(
            "SCAN ERROR:",
            error
        )

        return JSONResponse({

            "success": False,

            "message": str(error)

        })