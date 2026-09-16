import cv2
from ultralytics import YOLO


def scan_image(image_array):
    """Scans the given image using OpenCV."""

    # Decode image using OpenCV
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image.")

    print("Image received successfully.")
    print("Image size:", frame.shape)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Image converted to grayscale.")

    return frame


def recognize_objects(image_path):
    """
    Recognizes objects in the given image using YOLO.

    Args:
        image_path (str): Path to the image file.

    Returns:
        YOLO detection results.
    """

    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Perform object detection
    results = model(image_path)

    return results
