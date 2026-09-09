import cv2


def start_camera():

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

        cv2.imshow("Scan & Discover - Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
