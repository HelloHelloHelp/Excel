import cv2
import numpy as np


def process_image(image_data):

    # Convert uploaded bytes into a NumPy array
    image_array = np.frombuffer(image_data, np.uint8)

    # Decode image using OpenCV
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image.")

    print("Image received successfully.")
    print("Image size:", frame.shape)

    # ------------------------------------------------
    # YOUR OPENCV SCANNING CODE GOES HERE
    # ------------------------------------------------

    # Example:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("Image converted to grayscale.")

    # For now, simply confirm that the image was received.
    return "Scan successful! Image received and processed."


def start_camera():
    """
    Desktop-only camera function.

    This is kept for testing the project directly on a computer.
    Render/phone does NOT use this function.
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