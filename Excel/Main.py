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
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Scan & Discover</title>

        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
                padding: 20px;
            }

            video {
                width: 100%;
                max-width: 600px;
                border-radius: 12px;
                background: black;
            }

            button {
                padding: 15px 25px;
                margin: 10px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
            }

            #start {
                background: #2ecc71;
                color: white;
            }

            #scan {
                background: #3498db;
                color: white;
            }

            #stop {
                background: #e74c3c;
                color: white;
            }

            #result {
                margin: 20px auto;
                padding: 15px;
                max-width: 600px;
                background: #222;
                border-radius: 8px;
            }
        </style>
    </head>

    <body>

        <h1>Scan & Discover</h1>

        <button id="start" onclick="startCamera()">
            Start Camera
        </button>

        <button id="scan" onclick="scan()" disabled>
            Scan
        </button>

        <button id="stop" onclick="stopCamera()" disabled>
            Stop Camera
        </button>

        <br><br>

        <video id="video" autoplay playsinline></video>

        <canvas id="canvas" style="display:none;"></canvas>

        <div id="result">
            Start the camera.
        </div>

        <script>

            let stream = null;

            async function startCamera() {

                try {

                    stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: "environment"
                        },
                        audio: false
                    });

                    const video = document.getElementById("video");

                    video.srcObject = stream;

                    document.getElementById("scan").disabled = false;
                    document.getElementById("stop").disabled = false;
                    document.getElementById("start").disabled = true;

                    document.getElementById("result").innerText =
                        "Camera ready. Point at an object and press Scan.";

                } catch (error) {

                    console.error(error);

                    document.getElementById("result").innerText =
                        "Camera error: " + error.message;
                }
            }


            function stopCamera() {

                if (stream) {

                    stream.getTracks().forEach(
                        track => track.stop()
                    );

                    stream = null;
                }

                document.getElementById("video").srcObject = null;

                document.getElementById("scan").disabled = true;
                document.getElementById("stop").disabled = true;
                document.getElementById("start").disabled = false;

                document.getElementById("result").innerText =
                    "Camera stopped.";
            }


            async function scan() {

                if (!stream) {
                    return;
                }

                const video = document.getElementById("video");
                const canvas = document.getElementById("canvas");

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                const context = canvas.getContext("2d");

                context.drawImage(
                    video,
                    0,
                    0,
                    canvas.width,
                    canvas.height
                );

                document.getElementById("result").innerText =
                    "Scanning...";

                canvas.toBlob(async function(blob) {

                    const formData = new FormData();

                    formData.append(
                        "file",
                        blob,
                        "camera.jpg"
                    );

                    try {

                        const response = await fetch("/scan", {
                            method: "POST",
                            body: formData
                        });

                        const data = await response.json();

                        document.getElementById("result").innerText =
                            data.message;

                    } catch (error) {

                        console.error(error);

                        document.getElementById("result").innerText =
                            "Server error.";
                    }

                }, "image/jpeg", 0.85);
            }

        </script>

    </body>
    </html>
    """


@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    try:

        image_data = await file.read()

        # Convert uploaded image to NumPy
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
            return {
                "success": False,
                "message": "Could not decode image."
            }

        print("Image received:", frame.shape)

        # YOLO detection
        results = model(frame)

        detected = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                name = result.names[class_id]

                detected.append(
                    f"{name} ({confidence * 100:.0f}%)"
                )

        if not detected:

            message = "No objects detected."

        else:

            message = "Detected: " + ", ".join(detected)

        return {
            "success": True,
            "message": message
        }

    except Exception as error:

        print("ERROR:", error)

        return {
            "success": False,
            "message": "Processing error: " + str(error)
        }
