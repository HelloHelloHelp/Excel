import cv2
import numpy as np
from ultralytics import YOLO


# Load the model once when the server starts.
# This avoids loading the model every time someone presses Scan.
model = YOLO("yolo11n.pt")


def process_image(image_data):
    """
    Process an uploaded camera image using OpenCV and YOLO.
    """

    # Convert uploaded bytes into a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    # Decode JPEG/PNG using OpenCV
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image.")

    print("Image received successfully.")
    print("Image size:", frame.shape)

    # OpenCV processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("Image converted to grayscale.")

    # YOLO object detection
    results = model(frame)

    detected_objects = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            detected_objects.append({
                "object": class_name,
                "confidence": round(confidence, 2)
            })

    print("Detected objects:", detected_objects)

    if not detected_objects:
        return "No objects detected."

    # Create readable result
    descriptions = []

    for item in detected_objects:

        descriptions.append(
            f"{item['object']} "
            f"({item['confidence'] * 100:.0f}%)"
        )

    return "Detected: " + ", ".join(descriptions)


def start_camera():
    """
    Desktop-only camera function.

    The phone camera is handled by the browser.
    """

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Could not open camera")

        return

    print("Camera started. Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read camera")

            break

        cv2.imshow(
            "Scan & Discover - Camera",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()
