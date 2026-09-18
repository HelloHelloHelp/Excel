import cv2
import numpy as np
import sys
import os

def image_search(scene_path, template_path, threshold=0.8):
    """
    Search for a template image inside a scene image.

    :param scene_path: Path to the larger scene image
    :param template_path: Path to the template image
    :param threshold: Matching threshold (0-1), higher means stricter match
    :return: List of match coordinates [(x, y, w, h), ...]
    """
    # Validate file paths
    if not os.path.exists(scene_path) or not os.path.exists(template_path):
        raise FileNotFoundError("Scene or template image file not found.")

    # Read images
    scene_img = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if scene_img is None or template_img is None:
        raise ValueError("Could not read one of the images. Check file format.")

    # Convert to grayscale for matching
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # Get template dimensions
    w, h = template_gray.shape[::-1]

    # Perform template matching
    result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find locations above threshold
    locations = np.where(result >= threshold)
    matches = []

    for pt in zip(*locations[::-1]):  # Switch x and y
        matches.append((pt[0], pt[1], w, h))
        # Draw rectangle on the scene image
        cv2.rectangle(scene_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Matches", scene_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return matches

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python image_search.py <scene_image> <template_image>")
        sys.exit(1)

    scene_file = sys.argv[1]
    template_file = sys.argv[2]

    try:
        found_matches = image_search(scene_file, template_file, threshold=0.85)
        if found_matches:
            print(f"Found {len(found_matches)} matches:")
            for match in found_matches:
                print(f"Location: x={match[0]}, y={match[1]}, width={match[2]}, height={match[3]}")
        else:
            print("No matches found.")
    except Exception as e:
        print(f"Error: {e}")
