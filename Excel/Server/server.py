from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from camera import process_image

app = FastAPI(title="Scan & Discover")


@app.get("/", response_class=HTMLResponse)
def home():
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
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 20px;
                background: #111;
                color: white;
            }

            h1 {
                margin-bottom: 20px;
            }

            #camera {
                width: 100%;
                max-width: 600px;
                border-radius: 12px;
                background: black;
            }

            button {
                margin: 10px;
                padding: 14px 24px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }

            #startButton {
                background: #2ecc71;
                color: white;
            }

            #stopButton {
                background: #e74c3c;
                color: white;
            }

            #scanButton {
                background: #3498db;
                color: white;
            }

            button:disabled {
                background: #555 !important;
                color: #aaa;
                cursor: not-allowed;
            }

            #result {
                margin: 20px auto;
                max-width: 600px;
                padding: 15px;
                border-radius: 8px;
                background: #222;
            }

            canvas {
                display: none;
            }

        </style>

    </head>


    <body>

        <h1>Scan & Discover</h1>


        <!-- CAMERA BUTTONS -->

        <button
            id="startButton"
            onclick="startCamera()"
        >
            Start Camera
        </button>


        <button
            id="stopButton"
            onclick="stopCamera()"
            disabled
        >
            Stop Camera
        </button>


        <br>


        <!-- CAMERA -->

        <video
            id="camera"
            autoplay
            playsinline
        ></video>


        <br>


        <!-- SCAN -->

        <button
            id="scanButton"
            onclick="scanImage()"
            disabled
        >
            Scan
        </button>


        <canvas id="canvas"></canvas>


        <div id="result">
            Press "Start Camera" to begin.
        </div>


        <script>

            let stream = null;


            // ==========================================
            // START CAMERA
            // ==========================================

            async function startCamera() {

                try {

                    stream = await navigator.mediaDevices.getUserMedia({

                        video: {
                            facingMode: {
                                ideal: "environment"
                            }
                        },

                        audio: false

                    });


                    const video =
                        document.getElementById("camera");


                    video.srcObject = stream;


                    // Enable buttons

                    document.getElementById(
                        "scanButton"
                    ).disabled = false;


                    document.getElementById(
                        "stopButton"
                    ).disabled = false;


                    document.getElementById(
                        "startButton"
                    ).disabled = true;


                    document.getElementById(
                        "result"
                    ).innerText =
                        "Camera started. Point it at something and press Scan.";


                } catch (error) {

                    console.error(error);


                    document.getElementById(
                        "result"
                    ).innerText =
                        "Could not access the camera. Please allow camera permission.";

                }

            }


            // ==========================================
            // STOP CAMERA
            // ==========================================

            function stopCamera() {

                if (stream) {

                    stream
                        .getTracks()
                        .forEach(function(track) {

                            track.stop();

                        });


                    stream = null;

                }


                // Remove camera stream from video

                const video =
                    document.getElementById("camera");

                video.srcObject = null;


                // Disable Scan and Stop

                document.getElementById(
                    "scanButton"
                ).disabled = true;


                document.getElementById(
                    "stopButton"
                ).disabled = true;


                // Enable Start

                document.getElementById(
                    "startButton"
                ).disabled = false;


                document.getElementById(
                    "result"
                ).innerText =
                    "Camera stopped.";

            }


            // ==========================================
            // SCAN IMAGE
            // ==========================================

            async function scanImage() {

                if (!stream) {

                    document.getElementById(
                        "result"
                    ).innerText =
                        "Please start the camera first.";

                    return;

                }


                const video =
                    document.getElementById("camera");


                const canvas =
                    document.getElementById("canvas");


                if (
                    !video.videoWidth ||
                    !video.videoHeight
                ) {

                    document.getElementById(
                        "result"
                    ).innerText =
                        "Camera is not ready yet.";

                    return;

                }


                // Set canvas size to camera image

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;


                const context =
                    canvas.getContext("2d");


                // Capture current camera frame

                context.drawImage(
                    video,
                    0,
                    0,
                    canvas.width,
                    canvas.height
                );


                document.getElementById(
                    "result"
                ).innerText =
                    "Scanning...";


                // Convert image to JPEG

                canvas.toBlob(
                    async function(blob) {

                        if (!blob) {

                            document.getElementById(
                                "result"
                            ).innerText =
                                "Could not capture image.";

                            return;

                        }


                        const formData =
                            new FormData();


                        formData.append(
                            "file",
                            blob,
                            "camera.jpg"
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

                                document.getElementById(
                                    "result"
                                ).innerText =
                                    data.message;

                            } else {

                                document.getElementById(
                                    "result"
                                ).innerText =
                                    "Scan failed: " +
                                    data.message;

                            }


                        } catch (error) {

                            console.error(error);


                            document.getElementById(
                                "result"
                            ).innerText =
                                "Could not connect to the server.";

                        }

                    },
                    "image/jpeg",
                    0.90
                );

            }

        </script>

    </body>

    </html>
    """


@app.post("/scan")
async def scan(file: UploadFile = File(...)):

    try:

        image_data = await file.read()

        result = process_image(image_data)

        return JSONResponse({
            "success": True,
            "message": result
        })

    except Exception as error:

        print("Scan error:", error)

        return JSONResponse(
            {
                "success": False,
                "message": "Could not process the image."
            },
            status_code=500
        )