
"""
STUDENT NAME 1: AYMEN BENDOUMA 
STUDENT ID: 1193312

STUDENT NAME 2: BENITA THOMAS 
STUDENT ID: 1178480
SignCoach - Hand tracking and gesture recording prototype.

Uses OpenCV and MediaPipe to track hands, count extended fingers,
and record hand movements for future gesture comparison.


"""

import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp


# Project settings
CAMERA_INDEX = 0
MAX_HANDS = 2
TRAIL_LENGTH = 40
SAVE_DIR = "recordings"
MIN_FRAMES_TO_SAVE = 5

# Hand landmark indices
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17

FINGER_JOINTS = [
    (8, 6),
    (12, 10),
    (16, 14),
    (20, 18)
]

# MediaPipe hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def landmarks_to_array(hand_landmarks):
    """Convert hand landmarks into a NumPy array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )


def normalize_landmarks(points):
    """Center landmarks at the wrist and normalize by hand size.

    This reduces position and scale differences between recordings.
    It does not fully correct for rotation or camera perspective.
    """
    shifted = points - points[WRIST]

    hand_size = np.linalg.norm(shifted[MIDDLE_MCP])

    if hand_size < 1e-6:
        hand_size = 1e-6

    return shifted / hand_size


def count_raised_fingers(points):
    """Estimate the number of extended fingers."""
    wrist = points[WRIST]
    count = 0

    # Compare fingertip and joint distances from the wrist
    for tip, pip in FINGER_JOINTS:
        tip_distance = np.linalg.norm(points[tip] - wrist)
        joint_distance = np.linalg.norm(points[pip] - wrist)

        if tip_distance > joint_distance:
            count += 1

    # The thumb extends sideways, so use the pinky base
    pinky_base = points[PINKY_MCP]

    thumb_tip_distance = np.linalg.norm(
        points[THUMB_TIP] - pinky_base
    )

    thumb_joint_distance = np.linalg.norm(
        points[THUMB_IP] - pinky_base
    )

    if thumb_tip_distance > thumb_joint_distance:
        count += 1

    return count


def draw_trail(frame, trail):
    """Draw the recent path of an index fingertip."""
    n = len(trail)

    for i in range(1, n):
        age = i / n

        color = (
            0,
            int(255 * age),
            int(255 * (1 - age) + 100 * age)
        )

        thickness = max(1, int(6 * age))

        cv2.line(
            frame,
            trail[i - 1],
            trail[i],
            color,
            thickness
        )


def draw_text(frame, text, position,
              scale=0.6, color=(255, 255, 255)):
    """Display readable text over the camera image."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA
    )


def save_recording(frames):
    """Save recorded landmarks as a NumPy file."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Include microseconds to avoid duplicate filenames
    filename = datetime.now().strftime(
        "sign_%Y%m%d_%H%M%S_%f.npy"
    )

    file_path = os.path.join(SAVE_DIR, filename)

    data = np.stack(frames).astype(np.float32)
    np.save(file_path, data)

    return file_path


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}"
        )

    # Maintain a separate fingertip trail for each hand
    trails = {
        "Left": deque(maxlen=TRAIL_LENGTH),
        "Right": deque(maxlen=TRAIL_LENGTH)
    }

    recording = False
    recording_side = None
    rec_frames = []

    status_msg = ""
    status_until = 0.0

    fps = 0.0
    prev_time = time.time()

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                success, frame = cap.read()

                if not success:
                    print("Could not read camera frame.")
                    break

                # Mirror the image for a natural webcam view
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                # MediaPipe processes RGB images
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb_frame)

                seen_sides = set()

                if (results.multi_hand_landmarks
                        and results.multi_handedness):

                    for hand_lms, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        side = handedness.classification[0].label

                        # Avoid mixing two detections into one trail
                        if side not in trails or side in seen_sides:
                            continue

                        seen_sides.add(side)

                        # Draw hand landmarks and connections
                        mp_draw.draw_landmarks(
                            frame,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                        points = landmarks_to_array(hand_lms)

                        # Track the position of the index fingertip
                        tip = points[INDEX_TIP]

                        tip_x = int(tip[0] * width)
                        tip_y = int(tip[1] * height)

                        trails[side].append((tip_x, tip_y))

                        # Estimate how many fingers are extended
                        fingers = count_raised_fingers(points)

                        wrist_x = int(points[WRIST][0] * width)
                        wrist_y = int(points[WRIST][1] * height)

                        text_x = max(wrist_x - 70, 5)
                        text_y = min(wrist_y + 30, height - 10)

                        draw_text(
                            frame,
                            f"{side} hand | {fingers} finger(s)",
                            (text_x, text_y),
                            color=(0, 255, 255)
                        )

                        # Record one hand throughout the gesture
                        if recording:

                            if recording_side is None:
                                recording_side = side

                            if side == recording_side:
                                normalized = normalize_landmarks(points)
                                rec_frames.append(normalized)

                # Remove trails when their hands are no longer visible
                for side, trail in trails.items():

                    if side not in seen_sides:
                        trail.clear()

                    draw_trail(frame, trail)

                # Calculate a smoothed frame rate
                current_time = time.time()

                elapsed = max(
                    current_time - prev_time,
                    1e-6
                )

                current_fps = 1.0 / elapsed

                fps = 0.9 * fps + 0.1 * current_fps
                prev_time = current_time

                # Display frame rate and keyboard controls
                draw_text(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 25),
                    scale=0.7,
                    color=(0, 255, 0)
                )

                draw_text(
                    frame,
                    "r: record on/off   c: clear trails   q: quit",
                    (10, height - 12),
                    scale=0.55
                )

                # Recording indicator
                if recording:

                    cv2.circle(
                        frame,
                        (width - 30, 28),
                        10,
                        (0, 0, 255),
                        -1
                    )

                    draw_text(
                        frame,
                        f"REC {len(rec_frames)} frames",
                        (width - 190, 33),
                        scale=0.6,
                        color=(0, 0, 255)
                    )

                # Display the latest recording status
                if time.time() < status_until:

                    draw_text(
                        frame,
                        status_msg,
                        (10, 55),
                        scale=0.55,
                        color=(255, 200, 0)
                    )

                cv2.imshow("SignCoach hand test", frame)

                # Read keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):

                    for trail in trails.values():
                        trail.clear()

                elif key == ord("r"):

                    if not recording:
                        # Start a new recording
                        recording = True
                        recording_side = None
                        rec_frames = []

                        status_msg = "Recording started"
                        status_until = time.time() + 2.0

                        print(status_msg)

                    else:
                        # Stop recording and save collected frames
                        recording = False

                        if len(rec_frames) >= MIN_FRAMES_TO_SAVE:

                            path = save_recording(rec_frames)

                            status_msg = (
                                f"Saved {len(rec_frames)} frames -> {path}"
                            )

                        else:
                            status_msg = (
                                "Too few frames with a hand; not saved"
                            )

                        recording_side = None
                        status_until = time.time() + 4.0

                        print(status_msg)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
