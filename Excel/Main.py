from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="Scan & Discover")

# Load YOLO once when the server starts
model = YOLO("yolo11n.pt")


@app.get("/", response_class=HTMLResponse)
async def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

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
                margin-bottom: 10px;
            }

            p {
                color: #bbb;
            }

            #photoButton {
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 16px 30px;
                border-radius: 10px;
                font-size: 18px;
                cursor: pointer;
                margin-top: 20px;
            }

            #result {
                background: #222;
                border-radius: 10px;
                padding: 20px;
                margin: 25px auto;
                max-width: 600px;
            }

            #preview {
                max-width: 100%;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
            }

            #loading {
                display: none;
                color: #3498db;
            }

        </style>

    </head>


    <body>

        <h1>Scan & Discover</h1>

        <p>
            Take a photo of something and I'll analyse it.
        </p>


        <!-- PHONE CAMERA -->

        <label id="photoButton" for="photo">
            📷 Take Photo
        </label>

        <input
            id="photo"
            type="file"
            accept="image/*"
            capture="environment"
            style="display:none"
        >


        <!-- PHOTO PREVIEW -->

        <img id="preview">


        <!-- STATUS -->

        <div id="loading">
            Analysing photo...
        </div>


        <!-- RESULT -->

        <div id="result">
            No photo taken yet.
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
                async function() {

                    const file = this.files[0];

                    if (!file) {
                        return;
                    }


                    // Show preview

                    preview.src =
                        URL.createObjectURL(file);

                    preview.style.display = "block";


                    result.innerText = "";

                    loading.style.display = "block";


                    // Prepare upload

                    const formData =
                        new FormData();

                    formData.append(
                        "file",
                        file
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


                    loading.style.display = "none";

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


        # Convert photo to NumPy array
        image_array = np.frombuffer(
            image_data,
            dtype=np.uint8
        )


        # Decode image with OpenCV
        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            raise ValueError(
                "Could not decode the photo."
            )


        print(
            "Photo received:",
            frame.shape
        )


        # -----------------------------
        # YOLO ANALYSIS
        # -----------------------------

        results = model(frame)


        detected = []


        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(
                    box.conf[0]
                )

                name = result.names[class_id]


                detected.append(
                    f"{name} "
                    f"({confidence * 100:.0f}%)"
                )


        # -----------------------------
        # RESULT
        # -----------------------------

        if not detected:

            message = "I couldn't identify any objects."

        else:

            message = (
                "I found: "
                + ", ".join(detected)
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
