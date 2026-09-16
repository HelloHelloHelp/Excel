import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def process_image(image_data):

    image_array = np.frombuffer(
        image_data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError("Could not decode image.")

    print("Image received successfully.")
    print("Image size:", frame.shape)

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
        return "No objects detected."

    return "Detected: " + ", ".join(detected)
