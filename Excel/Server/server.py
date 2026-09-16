from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="Scan & Discover")

# Load YOLO once
model = YOLO("yolo11n.pt")


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
            }

            p {
                color: #bbb;
            }

            #takePhoto {
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

            #result {
                background: #222;
                padding: 20px;
                margin: 25px auto;
                max-width: 500px;
                border-radius: 12px;
            }

            #loading {
                display: none;
                color: #3498db;
                font-size: 18px;
            }

        </style>
    </head>

    <body>

        <h1>Scan & Discover</h1>

        <p>
            Take a photo and I'll analyse it.
        </p>

        <!-- PHOTO BUTTON -->

        <label id="takePhoto" for="photo">
            📷 Take Photo
        </label>

        <input
            id="photo"
            type="file"
            accept="image/*"
            capture="environment"
        >

        <!-- PHOTO PREVIEW -->

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


                    // Show the captured photo

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
                                "Error: " +
                                data.message;

                        }

                    } catch (error) {

                        console.error(error);

                        result.innerText =
                            "Could not connect to the server.";

                    }


                    loading.style.display =
                        "none";

                }
            );

        </script>

    </body>

    </html>
    """


@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    try:

        # Read uploaded photo
        image_data = await file.read()

        # Convert to NumPy
        image_array = np.frombuffer(
            image_data,
            dtype=np.uint8
        )

        # Decode with OpenCV
        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise ValueError(
                "Could not decode photo."
            )

        print(
            "Photo received:",
            frame.shape
        )

        # YOLO
        results = model(frame)

        detected = []

        for result_item in results:

            for box in result_item.boxes:

                class_id = int(box.cls[0])

                confidence = float(
                    box.conf[0]
                )

                name = result_item.names[
                    class_id
                ]

                detected.append(
                    f"{name} "
                    f"({confidence * 100:.0f}%)"
                )


        if detected:

            message = (
                "I found: "
                + ", ".join(detected)
            )

        else:

            message = (
                "I couldn't identify "
                "any objects."
            )


        return {
            "success": True,
            "message": message
        }


    except Exception as error:

        print(
            "Scan error:",
            error
        )

        return {
            "success": False,
            "message": str(error)
        }
