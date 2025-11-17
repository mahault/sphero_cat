import sys
# Force MTA COM apartment for Bleak on Windows – must be FIRST
sys.coinit_flags = 0

import time
import numpy as np
import cv2
import types
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ---- Stub out heavy plotting/dataframe libs that break under NumPy 2 ----
# We create minimal fake modules so pymdp.utils can import them
# without pulling in the real compiled wheels (pandas/seaborn/matplotlib).

fake_pandas = types.ModuleType("pandas")
class _DummyDataFrame:
    pass
fake_pandas.DataFrame = _DummyDataFrame
sys.modules["pandas"] = fake_pandas

fake_seaborn = types.ModuleType("seaborn")
sys.modules["seaborn"] = fake_seaborn

fake_matplotlib = types.ModuleType("matplotlib")
sys.modules["matplotlib"] = fake_matplotlib

# ---- Now import Sphero + pymdp + YOLO ----
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

from pymdp.agent import Agent   # still using pymdp!

from ultralytics import YOLO


# ====== LOCAL VERSIONS OF pymdp.utils HELPERS ======

def obj_array(num_factors):
    """
    Minimal replacement for pymdp.utils.obj_array:
    returns a 1D numpy object array of given length.
    """
    return np.empty(num_factors, dtype=object)


def obj_array_uniform(shape_list):
    """
    Minimal replacement for pymdp.utils.obj_array_uniform:
    for each dimension n in shape_list, create a uniform 1D array of length n.
    Returns an object array of these vectors.
    """
    arr = obj_array(len(shape_list))
    for i, n in enumerate(shape_list):
        vec = np.ones(n, dtype=float)
        vec /= vec.sum()
        arr[i] = vec
    return arr


# ====== CONFIGURATION ======

CAM_INDEX = 0  # webcam index

# Sphero movement parameters
SPHERO_SPEED = 100          # base speed
SEARCH_SPEED = 60          # slower speed when searching or very close
HEADING_DELTA = 10         # degrees to turn per step (left/right)
LOOP_DELAY = 0.05          # seconds between action updates

# POMDP discrete categories
# States / observations: 0 = LEFT, 1 = CENTER, 2 = RIGHT
N_STATES = 3
N_OBS = 3
# Actions: 0 = TURN_LEFT, 1 = STRAIGHT, 2 = TURN_RIGHT
N_ACTIONS = 3

# Threshold to consider the person "close enough" (ratio of bbox height to frame height)
NEAR_THRESHOLD = 0.7  # adjust based on your camera & distance

# YOLO class ID for "person" in COCO
PERSON_CLASS_ID = 0


# ====== YOLO PERSON DETECTOR ======

# This will automatically download yolov8n.pt the first time
yolo_model = YOLO("yolov8n.pt")


# ====== POMDP SETUP ======

def build_pomdp_agent():
    """
    POMDP:
        - Observations: LEFT, CENTER, RIGHT
        - Hidden states: LEFT, CENTER, RIGHT
        - Actions: TURN_LEFT, STRAIGHT, TURN_RIGHT

    A: identity (full observability)
    B: approximate effect of turning on person position in view
    C: prefers CENTER observations
    D: uniform prior
    """

    num_obs = [N_OBS]
    num_states = [N_STATES]
    num_controls = [N_ACTIONS]

    # --- A: p(o|s) – full observability
    A = obj_array(1)
    A[0] = np.eye(N_OBS)

    # --- B: p(s'|s,u)
    B = obj_array(1)
    B[0] = np.zeros((N_STATES, N_STATES, N_ACTIONS))

    # Helper
    def set_transition(s_from, a, s_to):
        B[0][s_to, s_from, a] = 1.0

    # States: 0=LEFT, 1=CENTER, 2=RIGHT
    # Actions: 0=TURN_LEFT, 1=STRAIGHT, 2=TURN_RIGHT

    # From LEFT (0)
    set_transition(0, 1, 0)  # STRAIGHT -> LEFT
    set_transition(0, 0, 1)  # TURN_LEFT -> CENTER
    set_transition(0, 2, 0)  # TURN_RIGHT -> LEFT

    # From CENTER (1)
    set_transition(1, 1, 1)  # STRAIGHT -> CENTER
    set_transition(1, 0, 2)  # TURN_LEFT -> RIGHT
    set_transition(1, 2, 0)  # TURN_RIGHT -> LEFT

    # From RIGHT (2)
    set_transition(2, 1, 2)  # STRAIGHT -> RIGHT
    set_transition(2, 2, 1)  # TURN_RIGHT -> CENTER
    set_transition(2, 0, 2)  # TURN_LEFT -> RIGHT

    # --- C: preferences over observations – CENTER is preferred
    C = obj_array(1)
    C[0] = np.array([0.0, 2.0, 0.0])

    # --- D: prior over hidden states – uniform
    D = obj_array(1)
    D[0] = np.ones(N_STATES, dtype=float) / N_STATES

    agent = Agent(A=A, B=B, C=C, D=D)
    return agent


# ====== VISION / OBSERVATION MAPPING (YOLO) ======

def detect_person_region(frame):
    """
    Use YOLOv8 to detect a person and return:

        obs_index:
            0 -> LEFT
            1 -> CENTER
            2 -> RIGHT

        size_ratio:
            ratio of bounding box height to frame height (0..1)
            (0 if no person detected)

        has_person:
            True if a person was detected, else False

        vis_frame:
            visualization frame with bounding box & labels
    """

    obs_index = 1  # default = CENTER
    size_ratio = 0.0
    has_person = False

    # Resize for speed & consistency
    frame_resized = cv2.resize(frame, (640, 480))
    h, w = frame_resized.shape[:2]
    third = w // 3

    # Draw partition lines
    cv2.line(frame_resized, (third, 0), (third, h), (200, 200, 200), 1)
    cv2.line(frame_resized, (2 * third, 0), (2 * third, h), (200, 200, 200), 1)

    # Run YOLOv8 inference (restricted to "person" class)
    results = yolo_model.predict(
        source=frame_resized,
        imgsz=640,
        conf=0.5,
        verbose=False,
        classes=[PERSON_CLASS_ID]
    )

    if results and len(results) > 0:
        res = results[0]
        boxes = res.boxes

        if boxes is not None and len(boxes) > 0:
            # Filter just in case (class 0 = person)
            person_boxes = [b for b in boxes if int(b.cls[0]) == PERSON_CLASS_ID]

            if person_boxes:
                # Pick the person with highest confidence
                best_box = max(person_boxes, key=lambda b: float(b.conf[0]))

                # xyxy coordinates
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Clamp to frame
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)

                if bw > 0 and bh > 0:
                    has_person = True
                    size_ratio = bh / float(h)

                    cx = x1 + bw // 2
                    cy = y1 + bh // 2

                    # Draw bounding box and center
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame_resized, (cx, cy), 5, (0, 0, 255), -1)

                    # Determine LEFT / CENTER / RIGHT
                    if cx < third:
                        obs_index = 0
                        region_label = "LEFT"
                    elif cx < 2 * third:
                        obs_index = 1
                        region_label = "CENTER"
                    else:
                        obs_index = 2
                        region_label = "RIGHT"

                    cv2.putText(frame_resized, region_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    has_person = False
                    obs_index = 1
                    size_ratio = 0.0
            else:
                has_person = False
                obs_index = 1
                size_ratio = 0.0
        else:
            has_person = False
            obs_index = 1
            size_ratio = 0.0
    else:
        has_person = False
        obs_index = 1
        size_ratio = 0.0

    return obs_index, size_ratio, has_person, frame_resized


# ====== CONTROL LOOP ======

def pomdp_sphero_webcam_loop(droid: SpheroEduAPI):
    """
    Main loop:
        - Reads frames from webcam
        - Uses YOLOv8 to locate the person (L/C/R)
        - POMDP chooses turning action
        - Speed is constant (no modulation by distance or detection)
        - When facing the person, drive straight forward in fast bursts
    """

    agent = build_pomdp_agent()

    # Constant forward speed
    FORWARD_SPEED = 120     # constant, no matter what
    TURN_SPEED = 120        # same speed while turning
    COMMIT_DURATION = 0.3   # seconds of committed forward drive

    # Initialize heading
    current_heading = 0
    droid.reset_aim()
    droid.set_main_led(Color(0, 255, 0))  # green

    # Open webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Starting webcam + POMDP person-following (YOLO). Press 'q' to quit.")

    # Initial observation (assume CENTER)
    observation = [1]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Detect person (LEFT/CENTER/RIGHT)
            obs_index, size_ratio, has_person, vis_frame = detect_person_region(frame)
            observation = [obs_index]

            # POMDP step
            qs = agent.infer_states(observation)
            agent.infer_policies()
            action = agent.sample_action()

            # action is usually array-like
            if isinstance(action, (list, tuple, np.ndarray)):
                a = int(action[0])
            else:
                a = int(action)

            # OVERRIDE: if person is CENTER, force STRAIGHT
            if obs_index == 1:
                a = 1

            # Map action to heading change
            # 0 -> TURN_LEFT, 1 -> STRAIGHT, 2 -> TURN_RIGHT
            if a == 0:
                current_heading = (current_heading - HEADING_DELTA) % 360
                action_label = "TURN_LEFT"
            elif a == 2:
                current_heading = (current_heading + HEADING_DELTA) % 360
                action_label = "TURN_RIGHT"
            else:
                action_label = "STRAIGHT"

            # CONSTANT SPEED – no modulation from distance
            speed = FORWARD_SPEED

            # Always green LED (unless you want differently)
            droid.set_main_led(Color(0, 255, 0))

            # Apply heading + constant speed
            droid.set_heading(int(current_heading))
            droid.set_speed(speed)

            # Visual overlay
            status_text = (
                f"Obs: {obs_index}  Act: {action_label}  Speed: {speed}"
            )
            cv2.putText(
                vis_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            cv2.imshow("Webcam (tracking - YOLO)", vis_frame)

            # Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit requested.")
                break

            # Commit: make the Sphero move forward for total COMMIT_DURATION
            time.sleep(COMMIT_DURATION)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt – stopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        droid.set_speed(0)
        droid.set_main_led(Color(255, 255, 255))
        time.sleep(0.2)


# ====== MAIN ENTRY POINT ======

def main():
    print("Searching for Sphero toy...")
    toy = scanner.find_toy()
    if not toy:
        print("No Sphero toy found. Make sure it is awake and nearby.")
        return

    print(f"Found {toy.name}. Connecting...")
    with SpheroEduAPI(toy) as droid:
        print("Connected to Sphero.")
        droid.set_main_led(Color(0, 0, 255))  # blue for connected
        time.sleep(0.5)

        pomdp_sphero_webcam_loop(droid)

    print("Disconnected from Sphero.")


if __name__ == "__main__":
    main()
