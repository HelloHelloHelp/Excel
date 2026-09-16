from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from camera import process_image


app = FastAPI(title="Scan & Discover")


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():
    return FileResponse("index.html")


# ============================================================
# SCAN
# ============================================================

@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    try:
        image_data = await file.read()

        if not image_data:
            raise ValueError("The photo is empty.")

        result = process_image(image_data)

        return JSONResponse({
            "success": True,
            "message": result["message"],
            "objects": result["objects"],
            "ocr": result["ocr"],
            "information": result["information"]
        })

    except Exception as error:

        print("SCAN ERROR:", error)

        return JSONResponse({
            "success": False,
            "message": str(error)
        })


# ============================================================
# LOCAL STARTUP
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000
    )