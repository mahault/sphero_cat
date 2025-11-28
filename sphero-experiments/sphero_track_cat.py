import sys
# Force MTA COM apartment for Bleak on Windows – must be FIRST
sys.coinit_flags = 0

import time
import numpy as np
import cv2
import types

# ---- Stub out heavy plotting/dataframe libs if they cause trouble ----
fake_pandas = types.ModuleType("pandas")
class _DummyDataFrame:  # noqa: E701
    pass
fake_pandas.DataFrame = _DummyDataFrame
sys.modules["pandas"] = fake_pandas

fake_seaborn = types.ModuleType("seaborn")
sys.modules["seaborn"] = fake_seaborn

fake_matplotlib = types.ModuleType("matplotlib")
fake_pyplot = types.ModuleType("matplotlib.pyplot")
def _nop(*args, **kwargs):  # noqa: E701
    pass
fake_pyplot.plot = _nop
fake_pyplot.imshow = _nop
fake_pyplot.figure = _nop
fake_pyplot.show = _nop
fake_pyplot.title = _nop
fake_pyplot.xlabel = _nop
fake_pyplot.ylabel = _nop
sys.modules["matplotlib"] = fake_matplotlib
sys.modules["matplotlib.pyplot"] = fake_pyplot

# ---- Sphero + pymdp + YOLO ----
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

from pymdp.agent import Agent

from ultralytics import YOLO


# ====== LOCAL VERSIONS OF pymdp.utils HELPERS ======

def obj_array(num_factors: int):
    return np.empty(num_factors, dtype=object)


def obj_array_uniform(shape_list):
    arr = obj_array(len(shape_list))
    for i, n in enumerate(shape_list):
        vec = np.ones(n, dtype=float)
        vec /= vec.sum()
        arr[i] = vec
    return arr


# ====== CONFIGURATION ======

CAM_INDEX = 0

# Sphero movement parameters
SPHERO_SPEED = 120         # speed when tracking cat
TURN_SPEED = 120           # speed when turning/diagonal
CIRCLE_SPEED = 60          # slower speed for idle circles
HEADING_DELTA = 10         # base turning increment (degrees)
LOOP_DELAY = 0.05
COMMIT_DURATION = 0.3      # commit to tracking move

# 3x3 grid: 9 cells
N_STATES = 9
N_OBS = 9

# Actions: 0 = STAY, 1 = UP, 2 = DOWN, 3 = LEFT, 4 = RIGHT,
#          5 = UP_LEFT, 6 = UP_RIGHT, 7 = DOWN_LEFT, 8 = DOWN_RIGHT
N_ACTIONS = 9

# YOLO COCO class id for "cat" (0=person, 15=cat in standard COCO)
CAT_CLASS_ID = 15


# ====== YOLO CAT DETECTOR ======

# This will automatically download yolov8n.pt the first time
det_model = YOLO("yolov8m.pt")


# ====== POMDP SETUP ======

def build_pomdp_agent():
    """
    POMDP over a 3x3 grid (0..8) with 9 directional actions.
    Center cell (index 4) is preferred.
    """

    num_obs = [N_OBS]
    num_states = [N_STATES]
    num_controls = [N_ACTIONS]

    # A: identity mapping p(o|s)
    A = obj_array(1)
    A[0] = np.eye(N_OBS)

    # B: grid dynamics p(s'|s,u)
    B = obj_array(1)
    B[0] = np.zeros((N_STATES, N_STATES, N_ACTIONS))

    def idx_to_rc(idx: int):
        return idx // 3, idx % 3

    def rc_to_idx(r: int, c: int):
        return r * 3 + c

    def move(r: int, c: int, a: int):
        # 0 = STAY
        # 1 = UP
        # 2 = DOWN
        # 3 = LEFT
        # 4 = RIGHT
        # 5 = UP_LEFT
        # 6 = UP_RIGHT
        # 7 = DOWN_LEFT
        # 8 = DOWN_RIGHT
        dr = dc = 0
        if a == 1:
            dr, dc = -1, 0
        elif a == 2:
            dr, dc = 1, 0
        elif a == 3:
            dr, dc = 0, -1
        elif a == 4:
            dr, dc = 0, 1
        elif a == 5:
            dr, dc = -1, -1
        elif a == 6:
            dr, dc = -1, 1
        elif a == 7:
            dr, dc = 1, -1
        elif a == 8:
            dr, dc = 1, 1
        # 0 = STAY -> (0,0)

        r_new = max(0, min(2, r + dr))
        c_new = max(0, min(2, c + dc))
        return r_new, c_new

    for s in range(N_STATES):
        r, c = idx_to_rc(s)
        for a in range(N_ACTIONS):
            r_new, c_new = move(r, c, a)
            s_new = rc_to_idx(r_new, c_new)
            B[0][s_new, s, a] = 1.0

    # C: prefer center (index 4)
    C = obj_array(1)
    prefs = np.zeros(N_OBS)
    prefs[4] = 2.0
    C[0] = prefs

    # D: uniform prior over states
    D = obj_array(1)
    D[0] = np.ones(N_STATES, dtype=float) / N_STATES

    return Agent(A=A, B=B, C=C, D=D)


# ====== VISION / OBSERVATION MAPPING (CAT IN 3x3 GRID) ======

def detect_cat_region(frame):
    """
    Use YOLOv8 object detection to detect a cat and return:

        obs_index: 0..8 (3x3 grid cell index, row*3 + col)
        size_ratio: bounding-box height / frame height (unused but kept)
        has_cat: True if a cat was detected
        vis_frame: frame with grid, bbox & cat dot
    """

    obs_index = 4   # default center
    size_ratio = 0.0
    has_cat = False

    frame_resized = cv2.resize(frame, (640, 480))
    h, w = frame_resized.shape[:2]

    # Draw 3x3 grid
    third_w = w // 3
    third_h = h // 3
    for i in range(1, 3):
        cv2.line(frame_resized, (i * third_w, 0), (i * third_w, h), (200, 200, 200), 1)
        cv2.line(frame_resized, (0, i * third_h), (w, i * third_h), (200, 200, 200), 1)

    # Run YOLO detection (filter to cat class)
    results = det_model.predict(
        source=frame_resized,
        imgsz=640,
        conf=0.5,
        verbose=False,
        classes=[CAT_CLASS_ID]
    )

    if not results or len(results) == 0:
        return obs_index, size_ratio, has_cat, frame_resized

    res = results[0]
    boxes = res.boxes

    if boxes is None or len(boxes) == 0:
        return obs_index, size_ratio, has_cat, frame_resized

    # Filter cat boxes (should already be filtered by classes, but double-check)
    cat_boxes = [b for b in boxes if int(b.cls[0]) == CAT_CLASS_ID]
    if not cat_boxes:
        return obs_index, size_ratio, has_cat, frame_resized

    # Pick the most confident cat
    best_box = max(cat_boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Clamp
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)

    if bw <= 0 or bh <= 0:
        return obs_index, size_ratio, has_cat, frame_resized

    has_cat = True
    size_ratio = bh / float(h)

    # Center of cat box
    cx = x1 + bw // 2
    cy = y1 + bh // 2

    # Map to 3x3 grid cell
    col = int(3 * cx / w)
    row = int(3 * cy / h)
    col = max(0, min(2, col))
    row = max(0, min(2, row))
    obs_index = row * 3 + col

    # Draw cat box and center
    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame_resized, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(
        frame_resized,
        f"Cat cell: {row},{col}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return obs_index, size_ratio, has_cat, frame_resized


# ====== CONTROL LOOP ======

def pomdp_sphero_webcam_loop(droid: SpheroEduAPI):
    """
    Main loop:
        - Reads frames from webcam
        - Uses YOLOv8 to locate a cat in a 3x3 grid
        - If cat is present:
            * POMDP chooses directional action
            * Sphero moves toward that grid cell
        - If no cat:
            * Sphero moves in a small circle (idle/search mode)
    """

    agent = build_pomdp_agent()

    current_heading = 0
    droid.reset_aim()
    droid.set_main_led(Color(0, 255, 0))

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Starting webcam + POMDP cat-tracking (3x3, YOLO). Press 'q' to quit.")

    observation = [4]  # start at center

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            obs_index, size_ratio, has_cat, vis_frame = detect_cat_region(frame)
            observation = [obs_index]

            if has_cat:
                # === TRACKING MODE: use POMDP ===
                qs = agent.infer_states(observation)
                agent.infer_policies()
                action = agent.sample_action()
                if isinstance(action, (list, tuple, np.ndarray)):
                    a = int(action[0])
                else:
                    a = int(action)

                action_label = "STAY"
                speed = 0

                if a == 0:
                    speed = 0
                    action_label = "STAY"

                elif a == 1:  # UP
                    speed = SPHERO_SPEED
                    action_label = "UP"

                elif a == 2:  # DOWN
                    current_heading = (current_heading + 180) % 360
                    speed = SPHERO_SPEED
                    action_label = "DOWN"

                elif a == 3:  # LEFT
                    current_heading = (current_heading - HEADING_DELTA) % 360
                    speed = TURN_SPEED
                    action_label = "LEFT"

                elif a == 4:  # RIGHT
                    current_heading = (current_heading + HEADING_DELTA) % 360
                    speed = TURN_SPEED
                    action_label = "RIGHT"

                elif a == 5:  # UP_LEFT
                    current_heading = (current_heading - HEADING_DELTA // 2) % 360
                    speed = SPHERO_SPEED
                    action_label = "UP_LEFT"

                elif a == 6:  # UP_RIGHT
                    current_heading = (current_heading + HEADING_DELTA // 2) % 360
                    speed = SPHERO_SPEED
                    action_label = "UP_RIGHT"

                elif a == 7:  # DOWN_LEFT
                    current_heading = (current_heading + 180 - HEADING_DELTA // 2) % 360
                    speed = SPHERO_SPEED
                    action_label = "DOWN_LEFT"

                elif a == 8:  # DOWN_RIGHT
                    current_heading = (current_heading + 180 + HEADING_DELTA // 2) % 360
                    speed = SPHERO_SPEED
                    action_label = "DOWN_RIGHT"

                droid.set_heading(int(current_heading))
                droid.set_speed(speed)
                droid.set_main_led(Color(0, 255, 0))  # green for tracking

                status_text = f"[CAT] Obs: {obs_index}  Act: {action_label}  Speed: {speed}"
                cv2.putText(
                    vis_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

                if speed > 0:
                    sleep_time = COMMIT_DURATION
                else:
                    sleep_time = LOOP_DELAY

            else:
                # === IDLE MODE: small circle about ~1 foot wide ===
                action_label = "CIRCLE"

                # PARAMETERS FOR A ~1-FOOT-WIDE CIRCLE
                CIRCLE_TURN = 30         # degrees per update
                CIRCLE_SPEED = 60       # moderate forward speed
                CIRCLE_DELAY = 0.05     # timing between updates (20 Hz)

                # increment heading
                current_heading = (current_heading + CIRCLE_TURN) % 360

                # apply heading + speed
                droid.set_heading(int(current_heading))
                droid.set_speed(CIRCLE_SPEED)

                # LED indicates idle
                droid.set_main_led(Color(0, 0, 255))

                # overlay text
                status_text = f"[NO CAT] Circle  Speed:{CIRCLE_SPEED}  Turn:{CIRCLE_TURN}°"
                cv2.putText(
                    vis_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

                sleep_time = CIRCLE_DELAY


            cv2.imshow("Webcam (cat tracking - 3x3 YOLO)", vis_frame)

            # Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit requested.")
                break

            time.sleep(sleep_time)

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
