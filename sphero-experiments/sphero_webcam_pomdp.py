import sys
# Force MTA COM apartment for Bleak on Windows – must be FIRST
sys.coinit_flags = 0

import time
import numpy as np
import cv2

from spherov2 import scanner
from spherov2.scanner import ToyNotFoundError  # Add this import
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

from pymdp import utils
from pymdp.agent import Agent


# ====== CONFIGURATION ======

# ====== PERSON DETECTOR (HOG) ======

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Webcam index (usually 0 is the built-in/default camera)
CAM_INDEX = 0

# Sphero movement parameters
SPHERO_SPEED = 40          # 0–255
HEADING_DELTA = 20         # degrees to turn per step (left/right)
LOOP_DELAY = 0.05          # seconds between action updates

# POMDP discrete categories
N_STATES = 3   # LEFT, CENTER, RIGHT
N_OBS = 3      # same
N_ACTIONS = 3  # TURN_LEFT, GO_STRAIGHT, TURN_RIGHT

# HSV color range for the target (blue-ish).
# You will likely need to tweak these based on your lighting.
BLUE_LOWER = np.array([90, 80, 50])   # H, S, V
BLUE_UPPER = np.array([130, 255, 255])


# ====== POMDP SETUP ======

def build_pomdp_agent():
    """
    Build a minimal POMDP with:
        - 1 observation modality (3 categories: left, center, right)
        - 1 hidden-state factor (3 states: left, center, right)
        - 1 control factor (3 actions: left, straight, right)

    A: identity (full observability)
    B: next_state = action_index  (deterministic)
    C: uniform (no strong prefs)
    D: uniform prior
    """

    num_obs = [N_OBS]         # one modality, 3 observations
    num_states = [N_STATES]   # one hidden-state factor, 3 states
    num_controls = [N_ACTIONS]

    # A: p(o|s) – full observability
    A = utils.obj_array(1)
    A[0] = np.eye(N_OBS)

    # B: p(s'|s,u) – next_state = action_index, independent of s
    B = utils.obj_array(1)
    B[0] = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            B[0][a, s, a] = 1.0

    # C: preferences over observations – uniform log-probs
    C = utils.obj_array_uniform(num_obs)

    # D: prior over hidden states – uniform
    D = utils.obj_array(1)
    D[0] = np.ones(N_STATES) / N_STATES

    agent = Agent(A=A, B=B, C=C, D=D)
    return agent


# ====== VISION / OBSERVATION MAPPING ======

def detect_person_region(frame):
    """
    Detect the most prominent person in the frame and return an observation index:

        0 -> LEFT
        1 -> CENTER
        2 -> RIGHT

    If no person is detected, returns 1 (CENTER) as a neutral default.

    Also returns a visualization frame with the detection drawn on it.
    """

    obs_index = 1  # default = CENTER

    # Optionally resize for speed & more robust detection
    frame_resized = cv2.resize(frame, (640, 480))
    h, w = frame_resized.shape[:2]
    third = w // 3

    # Draw vertical partitions for LEFT / CENTER / RIGHT
    cv2.line(frame_resized, (third, 0), (third, h), (200, 200, 200), 1)
    cv2.line(frame_resized, (2 * third, 0), (2 * third, h), (200, 200, 200), 1)

    # HOG person detection
    # returns rects (x, y, w, h) and weights
    rects, weights = hog.detectMultiScale(
        frame_resized,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    if len(rects) > 0:
        # Pick the largest detection by area
        areas = [w_i * h_i for (x_i, y_i, w_i, h_i) in rects]
        idx = int(np.argmax(areas))
        (x, y, bw, bh) = rects[idx]

        # Compute center of the bounding box
        cx = x + bw // 2
        cy = y + bh // 2

        # Draw the detection
        cv2.rectangle(frame_resized, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
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

        cv2.putText(frame_resized, region_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # No person detected -> CENTER
        obs_index = 1

    return obs_index, frame_resized


# ====== CONTROL LOOP ======

def pomdp_sphero_webcam_loop(droid: SpheroEduAPI):
    """
    Main loop:
        - Reads frames from webcam
        - Detects target location (L/C/R)
        - Feeds observation to POMDP agent
        - Samples action (LEFT/STRAIGHT/RIGHT)
        - Translates action into heading changes and sends Sphero commands
    """

    agent = build_pomdp_agent()

    # Initialize heading
    current_heading = 0
    droid.reset_aim()
    droid.set_main_led(Color(0, 255, 0))  # green

    # Open webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Starting webcam + POMDP control. Press 'q' to quit.")

    # Initial observation (assume CENTER)
    observation = [1]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Resize for speed (optional)
            frame = cv2.resize(frame, (640, 480))

            # Detect target region (LEFT/CENTER/RIGHT)
            obs_index, vis_frame = detect_person_region(frame)
            observation = [obs_index]

            # POMDP inference
            qs = agent.infer_states(observation)
            agent.infer_policies()
            action = agent.sample_action()

            # action is usually an array/list
            if isinstance(action, (list, tuple, np.ndarray)):
                a = int(action[0])
            else:
                a = int(action)

            # Map action to heading change
            # 0 -> TURN_LEFT, 1 -> STRAIGHT, 2 -> TURN_RIGHT
            if a == 0:
                current_heading = (current_heading - HEADING_DELTA) % 360
                action_label = "TURN_LEFT"
            elif a == 2:
                current_heading = (current_heading + HEADING_DELTA) % 360
                action_label = "TURN_RIGHT"
            else:
                # 1 -> STRAIGHT
                action_label = "STRAIGHT"

            # Send command to Sphero
            droid.set_heading(int(current_heading))
            droid.set_speed(SPHERO_SPEED)

            # Visual feedback on frame
            cv2.putText(vis_frame, f"Obs: {obs_index}  Act: {action_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

            cv2.imshow("Webcam (target tracking)", vis_frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit requested.")
                break

            time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt – stopping.")
    finally:
        # Cleanup
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
