import cv2
import numpy as np
import math
import time


# ============================================================
# SETTINGS
# ============================================================

MIN_SHAPE_AREA = 1200
MIN_HAND_AREA = 5000

# HSV skin-color range
SKIN_LOWER = np.array([0, 20, 70], dtype=np.uint8)
SKIN_UPPER = np.array([20, 255, 255], dtype=np.uint8)


# ============================================================
# SHAPE DETECTION
# ============================================================

def classify_shape(contour):
    """
    Classify a contour into a basic geometric shape.
    """

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Unknown", 0

    approximation = cv2.approxPolyDP(
        contour,
        0.03 * perimeter,
        True
    )

    vertices = len(approximation)

    # Triangle
    if vertices == 3:
        shape_name = "Triangle"

    # 4 sides
    elif vertices == 4:

        x, y, w, h = cv2.boundingRect(approximation)

        if h == 0:
            return "Rectangle", vertices

        aspect_ratio = w / float(h)

        if 0.90 <= aspect_ratio <= 1.10:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"

    # Pentagon
    elif vertices == 5:
        shape_name = "Pentagon"

    # Hexagon
    elif vertices == 6:
        shape_name = "Hexagon"

    # More than 6 sides
    elif vertices > 6:

        area = cv2.contourArea(contour)

        if area == 0:
            shape_name = "Polygon"
        else:
            circularity = (
                4 * math.pi * area /
                (perimeter * perimeter)
            )

            if circularity > 0.75:
                shape_name = "Circle"
            else:
                shape_name = "Polygon"

    else:
        shape_name = "Unknown"

    return shape_name, vertices


def detect_shapes(frame):
    """
    Detect geometric shapes using Canny edge detection.
    """

    output = frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Edge detection
    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected_shapes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_SHAPE_AREA:
            continue

        shape_name, vertices = classify_shape(contour)

        if shape_name == "Unknown":
            continue

        # Draw contour
        cv2.drawContours(
            output,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # Center
        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            cx = int(
                moments["m10"] /
                moments["m00"]
            )

            cy = int(
                moments["m01"] /
                moments["m00"]
            )

            cv2.circle(
                output,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                output,
                shape_name,
                (cx - 60, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        detected_shapes.append(shape_name)

    return output, detected_shapes


# ============================================================
# FINGER DETECTION
# ============================================================

def calculate_angle(point_a, point_b, point_c):
    """
    Calculate angle ABC.
    """

    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    c = np.array(point_c, dtype=np.float64)

    ba = a - b
    bc = c - b

    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 180.0

    cosine_value = np.dot(ba, bc) / (
        magnitude_ba * magnitude_bc
    )

    # Prevent floating-point errors
    cosine_value = np.clip(
        cosine_value,
        -1.0,
        1.0
    )

    angle = math.degrees(
        math.acos(cosine_value)
    )

    return angle


def detect_fingers(frame):
    """
    Detect hand and estimate number of fingers.

    This function is deliberately defensive about the
    convexityDefects() array format.
    """

    output = frame.copy()

    # --------------------------------------------------------
    # Convert to HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Skin mask
    # --------------------------------------------------------

    mask = cv2.inRange(
        hsv,
        SKIN_LOWER,
        SKIN_UPPER
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.GaussianBlur(
        mask,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Find hand contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0, output

    # Largest contour
    hand_contour = max(
        contours,
        key=cv2.contourArea
    )

    hand_area = cv2.contourArea(
        hand_contour
    )

    if hand_area < MIN_HAND_AREA:
        return 0, output

    # --------------------------------------------------------
    # Draw hand contour
    # --------------------------------------------------------

    cv2.drawContours(
        output,
        [hand_contour],
        -1,
        (255, 0, 0),
        2
    )

    # --------------------------------------------------------
    # Convex hull
    # --------------------------------------------------------

    hull_points = cv2.convexHull(
        hand_contour
    )

    cv2.polylines(
        output,
        [hull_points],
        True,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Convexity defects
    # --------------------------------------------------------

    hull_indices = cv2.convexHull(
        hand_contour,
        returnPoints=False
    )

    # Need at least 3 hull points
    if hull_indices is None:
        return 0, output

    hull_indices = np.asarray(
        hull_indices
    )

    if hull_indices.size < 3:
        return 0, output

    hull_indices = hull_indices.reshape(-1, 1)

    try:
        defects = cv2.convexityDefects(
            hand_contour,
            hull_indices
        )
    except cv2.error:
        return 0, output

    if defects is None:
        return 0, output

    # --------------------------------------------------------
    # SAFE DEFECT PROCESSING
    # --------------------------------------------------------

    defect_array = np.asarray(defects)

    if defect_array.size == 0:
        return 0, output

    # OpenCV normally returns:
    # (N, 1, 4)
    #
    # But depending on version/build it may be
    # represented differently.
    #
    # We flatten it into N rows of 4 values.

    try:
        defect_array = defect_array.reshape(-1, 4)
    except ValueError:
        return 0, output

    finger_gaps = 0

    for defect in defect_array:

        # Make sure we really have 4 values
        if len(defect) != 4:
            continue

        start_index = int(defect[0])
        end_index = int(defect[1])
        far_index = int(defect[2])
        depth = float(defect[3])

        # Validate contour indices
        if (
            start_index < 0
            or start_index >= len(hand_contour)
        ):
            continue

        if (
            end_index < 0
            or end_index >= len(hand_contour)
        ):
            continue

        if (
            far_index < 0
            or far_index >= len(hand_contour)
        ):
            continue

        # ----------------------------------------------------
        # Get points
        # ----------------------------------------------------

        start_point = tuple(
            hand_contour[start_index][0]
        )

        end_point = tuple(
            hand_contour[end_index][0]
        )

        far_point = tuple(
            hand_contour[far_index][0]
        )

        # ----------------------------------------------------
        # Calculate angle
        # ----------------------------------------------------

        angle = calculate_angle(
            start_point,
            far_point,
            end_point
        )

        # ----------------------------------------------------
        # Defect depth
        #
        # OpenCV stores depth in fixed-point format.
        # Divide by 256.
        # ----------------------------------------------------

        real_depth = depth / 256.0

        # ----------------------------------------------------
        # Finger-gap condition
        # ----------------------------------------------------

        if (
            angle < 90
            and real_depth > 10
        ):

            finger_gaps += 1

            cv2.circle(
                output,
                far_point,
                8,
                (0, 0, 255),
                -1
            )

            cv2.line(
                output,
                start_point,
                far_point,
                (255, 0, 255),
                2
            )

            cv2.line(
                output,
                end_point,
                far_point,
                (255, 0, 255),
                2
            )

    # --------------------------------------------------------
    # Estimate finger count
    # --------------------------------------------------------

    if finger_gaps > 0:
        finger_count = finger_gaps + 1
    else:
        finger_count = 0

    # Keep result in normal range
    finger_count = max(
        0,
        min(5, finger_count)
    )

    return finger_count, output


# ============================================================
# FPS
# ============================================================

def calculate_fps(previous_time):
    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed <= 0:
        return 0, current_time

    fps = 1.0 / elapsed

    return fps, current_time


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("CAMERA SHAPE + FINGER DETECTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        print()
        print("Try changing:")
        print("    cv2.VideoCapture(0)")
        print("to:")
        print("    cv2.VideoCapture(1)")

        return

    # --------------------------------------------------------
    # Camera resolution
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    shape_detection = True
    finger_detection = True

    previous_time = time.time()

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    print()
    print("Controls:")
    print("  Q = Quit")
    print("  S = Toggle Shape Detection")
    print("  F = Toggle Finger Detection")
    print("  R = Reset FPS")
    print()

    # --------------------------------------------------------
    # Main camera loop
    # --------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read frame.")

            break

        # Mirror image
        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Shape detection
        # ----------------------------------------------------

        detected_shapes = []

        if shape_detection:

            frame, detected_shapes = detect_shapes(
                frame
            )

        # ----------------------------------------------------
        # Finger detection
        # ----------------------------------------------------

        finger_count = 0

        if finger_detection:

            finger_count, frame = detect_fingers(
                frame
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        fps, previous_time = calculate_fps(
            previous_time
        )

        # ----------------------------------------------------
        # Information panel
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (400, 150),
            (0, 0, 0),
            -1
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Finger count
        cv2.putText(
            frame,
            f"Fingers: {finger_count}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Shape mode
        shape_status = (
            "ON"
            if shape_detection
            else "OFF"
        )

        cv2.putText(
            frame,
            f"Shapes: {shape_status}",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Finger mode
        finger_status = (
            "ON"
            if finger_detection
            else "OFF"
        )

        cv2.putText(
            frame,
            f"Finger: {finger_status}",
            (25, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 150, 0),
            2
        )

        # ----------------------------------------------------
        # Shape information
        # ----------------------------------------------------

        if detected_shapes:

            shape_text = ", ".join(
                detected_shapes[:5]
            )

            cv2.putText(
                frame,
                shape_text,
                (20, 690),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "Shape + Finger Detector",
            frame
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Q = Quit
        if key == ord("q"):
            break

        # S = Toggle shapes
        elif key == ord("s"):

            shape_detection = not shape_detection

            print(
                "Shape Detection:",
                "ON" if shape_detection else "OFF"
            )

        # F = Toggle fingers 
        elif key == ord("f"):

            finger_detection = not finger_detection

            print(
                "Finger Detection:",
                "ON" if finger_detection else "OFF"
            )

        # R = Reset FPS
        elif key == ord("r"):

            previous_time = time.time()

            print("FPS timer reset.")


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("Application closed.")


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()