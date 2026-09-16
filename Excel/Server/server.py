from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from camera import process_image

app = FastAPI(title="Scan & Discover")


@app.get("/")
async def home():
    return FileResponse("index.html")


@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    try:
        image_data = await file.read()

        if not image_data:
            raise ValueError("The photo is empty.")

        message = process_image(image_data)

        return {
            "success": True,
            "message": message
        }

    except Exception as error:

        print("SCAN ERROR:", error)

        return {
            "success": False,
            "message": str(error)
        }