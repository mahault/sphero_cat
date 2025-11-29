import sys
sys.coinit_flags = 0  # Fix COM init on Windows for Bluetooth

import time
import math
import csv
import datetime
import numpy as np
import cv2
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from pathlib import Path  # NEW

# --- PATHS (relative to repository root) ---
# This file lives in:  sphero/scripts/sphero_track_test_v2.py
# So BASE_DIR is:      sphero/
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# --- STUBBING PLOTTING LIBS (ultralytics sometimes imports these) ---
fake_pandas = types.ModuleType("pandas")
sys.modules["pandas"] = fake_pandas
sys.modules["seaborn"] = types.ModuleType("seaborn")
sys.modules["matplotlib"] = types.ModuleType("matplotlib")
sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ================= CONFIGURATION =================

CAM_INDEX = 1
LOOP_DELAY = 0.05            # Control loop ~20 Hz
VISION_LATENCY_SEC = 0.18    # Display/assumed latency (no longer used in EST)

# --- NAVIGATION & SAFETY ---
MAX_SPEED = 30               # Slightly conservative
MIN_SPEED = 5
TARGET_REACHED_PIX = 35
BORDER_MARGIN = 100          # Bigger safe margin in small field
PREDICT_HORIZON_SEC = 0.4    # Prediction horizon for exit check
CENTER_RETURN_SPEED = 30

# --- BALL DETECTION (robust: Hough + YOLO + bright blobs) ---
BRIGHTNESS_THRESHOLD = 190
MIN_RADIUS_PX = 8
MAX_RADIUS_PX = 60
MIN_CONFIDENCE = 0.25
SEARCH_WINDOW = 100
PRIOR_WEIGHT = 0.8
RESET_THRESHOLD = 160
YOLO_BALL_CLASS = 32  # COCO sports ball

# --- YOLO TARGETS (cat/dog/person) ---
MIN_TARGET_CONF = 0.35
PERSON_CLASS = 0
CAT_CLASS = 15
DOG_CLASS = 16
PRIMARY_TARGET_CLASSES = [CAT_CLASS, DOG_CLASS]
SECONDARY_TARGET_CLASSES = [PERSON_CLASS]

# --- CALIBRATION (heading + scale) ---
CALIB_MIN_MOVE_DIST = 5
CALIB_MAX_ERR_TO_UPDATE = 70  # for runtime refinement
DEFAULT_PX_PER_SPEED_PER_SEC = 3.0  # fallback

# --- STUCK DETECTION ---
STUCK_DIST_THRESHOLD = 2.0     # px
STUCK_FRAMES_THRESHOLD = 10    # frames of "we're not moving" while commanded

# ================= MATH HELPERS =================

def normalize_angle(a: float) -> float:
    return a % 360.0

def angle_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0

def get_screen_angle(p1, p2) -> float:
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]  # invert Y
    deg = math.degrees(math.atan2(dx, dy))
    return normalize_angle(deg)

def is_near_boundary(x, y, w, h, margin=BORDER_MARGIN):
    return (
        x < margin or x > w - margin or
        y < margin or y > h - margin
    )

def will_exit_bounds(x, y, vx, vy, w, h,
                     horizon_sec=PREDICT_HORIZON_SEC,
                     margin=BORDER_MARGIN):
    """
    Predict if, under current velocity, we will cross the boundary band
    within 'horizon_sec' seconds.
    """
    future_x = x + vx * horizon_sec
    future_y = y + vy * horizon_sec
    future_x = float(np.clip(future_x, 0, w-1))
    future_y = float(np.clip(future_y, 0, h-1))

    near_now = is_near_boundary(x, y, w, h, margin)
    near_future = is_near_boundary(future_x, future_y, w, h, margin)

    return near_future, (future_x, future_y), near_now

# ================= TRACKER (KALMAN FOR BALL) =================

class Tracker:
    """
    Simple 2D position + velocity Kalman filter for the ball:
      state = [x, y, vx, vy]^T
    """
    def __init__(self, process_noise=0.01, measurement_noise=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.prediction = np.zeros((4, 1), np.float32)
        self.frames_lost = 100
        self.radius = 20

    def predict(self):
        self.prediction = self.kf.predict()
        self.frames_lost += 1
        return int(self.prediction[0, 0]), int(self.prediction[1, 0])

    def update(self, x, y, r=0):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.prediction = self.kf.correct(meas)
        self.radius = r
        self.frames_lost = 0

    def reset(self, x, y):
        self.kf.statePost = np.array(
            [[np.float32(x)], [np.float32(y)], [0], [0]], np.float32
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        self.prediction = self.kf.statePost
        self.frames_lost = 0

    def get_state(self, w=640, h=480):
        x = int(np.clip(self.prediction[0, 0], 0, w-1))
        y = int(np.clip(self.prediction[1, 0], 0, h-1))
        return x, y

    def get_velocity(self):
        return float(self.prediction[2, 0]), float(self.prediction[3, 0])

# ================= ROBUST BALL DETECTION =================

def find_candidates(frame, model, bright_mask):
    """
    Ball candidate detection using:
      - Hough circles
      - YOLO ball detection
      - Bright blob circularity
    Fused into scored candidates.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    EDGE_MARGIN = 30
    OVERLAP_DIST = 40

    hough_detections = []
    yolo_detections = []
    blob_detections = []

    # 1. Hough circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=60,
        param1=150, param2=30,
        minRadius=MIN_RADIUS_PX, maxRadius=MAX_RADIUS_PX
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = int(i[0]), int(i[1]), int(i[2])
            if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
                cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
                continue
            mask_roi = np.zeros_like(gray)
            cv2.circle(mask_roi, (cx, cy), r, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask_roi)[0]
            if mean_brightness < BRIGHTNESS_THRESHOLD:
                continue
            hough_detections.append(
                {'x': cx, 'y': cy, 'r': r, 'brightness': mean_brightness}
            )

    # 2. YOLO ball detections (class 32)
    results = model(frame, verbose=False, conf=MIN_CONFIDENCE)
    if results[0].boxes:
        for box in results[0].boxes:
            if int(box.cls[0]) == YOLO_BALL_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r = max((x2 - x1) // 2, (y2 - y1) // 2)
                conf = float(box.conf[0])
                if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
                    cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
                    continue
                yolo_detections.append(
                    {'x': cx, 'y': cy, 'r': r, 'conf': conf}
                )

    # 3. Bright blobs (brightness-based detection)
    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity > 0.6:
            (x, y), r = cv2.minEnclosingCircle(c)
            cx, cy = int(x), int(y)
            if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
                cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
                continue
            if MIN_RADIUS_PX < r < MAX_RADIUS_PX:
                blob_detections.append(
                    {'x': cx, 'y': cy, 'r': int(r), 'circ': circularity}
                )

    candidates = []

    def distance(p1, p2):
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

    # Fuse Hough + YOLO + Blob
    for h_det in hough_detections:
        score = 100
        src = 'Hough'
        score += (h_det['brightness'] - BRIGHTNESS_THRESHOLD) / 2

        yolo_match = None
        for y_det in yolo_detections:
            if distance(h_det, y_det) < OVERLAP_DIST:
                yolo_match = y_det
                break

        blob_match = None
        for b_det in blob_detections:
            if distance(h_det, b_det) < OVERLAP_DIST:
                blob_match = b_det
                break

        if yolo_match and blob_match:
            score += 200
            src = 'Hough+YOLO+Blob'
        elif yolo_match:
            score += 150
            src = 'Hough+YOLO'
        elif blob_match:
            score += 50
            src = 'Hough+Blob'

        candidates.append({
            'x': h_det['x'],
            'y': h_det['y'],
            'r': h_det['r'],
            'score': score,
            'src': src
        })

    # YOLO-only (unmatched)
    for y_det in yolo_detections:
        if any(distance(h_det, y_det) < OVERLAP_DIST for h_det in hough_detections):
            continue
        score = 120
        src = 'YOLO'
        blob_match = None
        for b_det in blob_detections:
            if distance(y_det, b_det) < OVERLAP_DIST:
                blob_match = b_det
                break
        if blob_match:
            score += 60
            src = 'YOLO+Blob'
        candidates.append({
            'x': y_det['x'],
            'y': y_det['y'],
            'r': y_det['r'],
            'score': score,
            'src': src
        })

    return candidates

def select_best_ball(candidates, tracker: Tracker):
    if not candidates:
        return None
    pred_x, pred_y = tracker.get_state()
    best_candidate = None
    highest_adjusted_score = -1e9
    for c in candidates:
        dist = math.sqrt((c['x'] - pred_x)**2 + (c['y'] - pred_y)**2)
        if tracker.frames_lost < 30:
            if dist < SEARCH_WINDOW:
                spatial_bonus = (SEARCH_WINDOW - dist) * 0.2
                c['final_score'] = c['score'] + spatial_bonus
            else:
                c['final_score'] = c['score'] - (dist * PRIOR_WEIGHT)
                if c['score'] > RESET_THRESHOLD:
                    c['final_score'] = c['score']
        else:
            c['final_score'] = c['score']

        if c['final_score'] > highest_adjusted_score:
            highest_adjusted_score = c['final_score']
            best_candidate = c
    return best_candidate

def process_vision(frame, model, ball_tracker: Tracker):
    """
    One-step robust vision processing:
      - detect ball candidates (Hough + YOLO + bright blobs)
      - select best via Tracker prior
      - update ball_tracker
      - return best_ball and candidates
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(
        gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    bright_mask = cv2.dilate(bright_mask, None, iterations=2)

    candidates = find_candidates(frame, model, bright_mask)
    best_ball = select_best_ball(candidates, ball_tracker)

    # Kalman predict and correct
    bx_pred, by_pred = ball_tracker.predict()
    if best_ball:
        dist = math.sqrt((best_ball['x'] - bx_pred)**2 +
                         (best_ball['y'] - by_pred)**2)
        if dist > SEARCH_WINDOW and best_ball['score'] > RESET_THRESHOLD:
            ball_tracker.reset(best_ball['x'], best_ball['y'])
        else:
            ball_tracker.update(best_ball['x'], best_ball['y'], best_ball['r'])

    return best_ball, candidates

# ================= TARGET DETECTION =================

def detect_target(frame, model):
    h, w = frame.shape[:2]
    results = model(frame, verbose=False, conf=MIN_TARGET_CONF)
    res = results[0]

    target_mask_primary = np.zeros((h, w), dtype=np.uint8)
    target_mask_secondary = np.zeros((h, w), dtype=np.uint8)
    label = None
    center = None

    if res.masks is not None and res.boxes is not None:
        masks = res.masks.data.cpu().numpy()
        boxes = res.boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            m = cv2.resize(masks[i], (w, h))
            binary = (m > 0.5).astype(np.uint8)

            if cls_id in PRIMARY_TARGET_CLASSES:
                target_mask_primary = cv2.bitwise_or(target_mask_primary, binary)
            elif cls_id in SECONDARY_TARGET_CLASSES:
                target_mask_secondary = cv2.bitwise_or(target_mask_secondary, binary)

    if target_mask_primary.any():
        mask = target_mask_primary
        label = "CAT/DOG"
    elif target_mask_secondary.any():
        mask = target_mask_secondary
        label = "PERSON"
    else:
        return None, None, np.zeros((h, w), dtype=np.uint8)

    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)

    return center, label, mask.astype(np.uint8) * 255

# ================= CALIBRATION =================

def calibrate_orientation_and_scale(cap, droid, model, ball_tracker):
    """
    After manual alignment, do a short auto-calibration:
      - Move Sphero in 4 cardinal directions
      - Use robust ball detection via process_vision()
      - Infer:
          calibration_offset  (difference between Sphero 0° and camera frame)
          px_per_speed_per_sec (pixels per unit Sphero speed per second)
    """
    print("\n=== AUTO CALIBRATION (heading + scale) ===")
    calib_window_open = False

    # Make sure ball is visible
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        best_ball, _ = process_vision(frame, model, ball_tracker)
        bx, by = ball_tracker.get_state(frame.shape[1], frame.shape[0])
        ball_found = ball_tracker.frames_lost < 10
        if ball_found:
            print(f"[CALIB] Ball found at ({bx},{by})")
            break
        cv2.putText(frame, "Place Sphero in view for calibration",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        calib_window_open = True
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if calib_window_open:
        cv2.destroyWindow("Calibration")

    test_dirs = [0, 90, 180, 270]
    CALIB_SPEED = 40
    MOVE_TIME = 0.6
    SETTLE_TIME = 0.4

    offsets = []
    scales = []

    for d in test_dirs:
        print(f"[CALIB] Testing command heading {d}°")
        # let detection settle
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                continue
            process_vision(frame, model, ball_tracker)
            time.sleep(LOOP_DELAY)

        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        process_vision(frame, model, ball_tracker)
        sx, sy = ball_tracker.get_state(w, h)
        ball_found = ball_tracker.frames_lost < 10
        if not ball_found:
            print("[CALIB] Ball lost before move, skipping sample.")
            continue
        start = (sx, sy)

        # Move Sphero
        droid.set_heading(int(d))
        droid.set_speed(CALIB_SPEED)
        t0 = time.time()
        while time.time() - t0 < MOVE_TIME:
            ret, frame = cap.read()
            if not ret:
                continue
            process_vision(frame, model, ball_tracker)
            time.sleep(LOOP_DELAY)

        droid.set_speed(0)
        time.sleep(SETTLE_TIME)

        # Measure end position
        for _ in range(3):
            ret, frame = cap.read()
            if not ret:
                continue
            process_vision(frame, model, ball_tracker)
            time.sleep(LOOP_DELAY)

        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        process_vision(frame, model, ball_tracker)
        ex, ey = ball_tracker.get_state(w, h)
        ball_found = ball_tracker.frames_lost < 10
        if not ball_found:
            print("[CALIB] Ball lost after move, skipping sample.")
            continue
        end = (ex, ey)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < CALIB_MIN_MOVE_DIST:
            print(f"[CALIB] Movement too small ({dist:.1f}px), skipping.")
            continue

        actual_angle = get_screen_angle(start, end)
        off = angle_diff(actual_angle, d)
        offsets.append(off)

        scale = dist / (CALIB_SPEED * MOVE_TIME)
        scales.append(scale)

        print(f"[CALIB] cmd:{d}°, actual:{actual_angle:.1f}°,"
              f" offset:{off:.1f}°, dist:{dist:.1f}px, scale sample:{scale:.2f}")

    if not offsets:
        print("[CALIB] No valid samples. Using offset=0°, default scale.")
        calibration_offset = 0.0
        px_per_speed_per_sec = DEFAULT_PX_PER_SPEED_PER_SEC
    else:
        rad = np.deg2rad(offsets)
        mean_sin = np.mean(np.sin(rad))
        mean_cos = np.mean(np.cos(rad))
        calibration_offset = normalize_angle(np.rad2deg(math.atan2(mean_sin, mean_cos)))
        px_per_speed_per_sec = float(np.mean(scales))
        print(f"[CALIB] Final orientation offset: {calibration_offset:.1f}°")
        print(f"[CALIB] Final scale: {px_per_speed_per_sec:.2f} px/(speed·s)")

    print("=== AUTO CALIBRATION DONE ===\n")
    return calibration_offset, px_per_speed_per_sec

# ================= MAIN =================

def main():
    print("Scanning for Sphero...")
    toy = None
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(scanner.find_toy)
        try:
            toy = fut.result(timeout=10)
        except Exception:
            toy = None

    if not toy:
        print("No Sphero found.")
        return

    print("Loading YOLO model...")
    model_path = MODELS_DIR / "yolov8s-seg.pt"   # NEW
    if model_path.exists():
        print(f"[YOLO] Using weights at {model_path}")
        model = YOLO(str(model_path))
    else:
        # Fallback so it still works if someone drops the weights in CWD
        print(f"[YOLO] WARNING: {model_path} not found, falling back to 'yolov8s-seg.pt' in CWD")
        model = YOLO("yolov8s-seg.pt")

    # Ball tracker for robust detection
    ball_tracker = Tracker(process_noise=0.01, measurement_noise=1.0)

    with SpheroEduAPI(toy) as droid:
        print("[SPHERO] Setting up robot...")
        droid.set_speed(0)
        droid.set_main_led(Color(255, 255, 255))
        time.sleep(0.3)

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            print(f"[CAMERA] Could not open camera index {CAM_INDEX}")
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[CAMERA] Cannot read first frame.")
            return

        h, w = frame.shape[:2]
        print(f"[CAMERA] Resolution: {w}x{h}")

        print("[SPHERO] Resetting heading reference (0°)...")
        droid.reset_aim()
        time.sleep(0.8)

        # Manual alignment screen (unchanged)
        print("\n" + "="*60)
        print("[ALIGNMENT] Manual alignment step.")
        print("[ALIGNMENT] Imagine Sphero's 'forward' (0° heading) pointing straight up in the camera view.")
        print("[ALIGNMENT] Place/rotate Sphero so its physical forward matches 'up' in the video.")
        print("[ALIGNMENT] Press SPACE or ENTER in the video window when ready.")
        print("="*60 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            h, w = disp.shape[:2]
            arrow_tip = (w // 2, 40)
            arrow_base = (w // 2, 130)
            arrow_width = 40
            cv2.line(disp, arrow_base, arrow_tip, (0, 255, 0), 8)
            pts = np.array([
                [arrow_tip[0], arrow_tip[1]],
                [arrow_tip[0] - arrow_width, arrow_tip[1] + arrow_width],
                [arrow_tip[0] + arrow_width, arrow_tip[1] + arrow_width]
            ], np.int32)
            cv2.fillPoly(disp, [pts], (0, 255, 0))

            cv2.putText(disp, "ALIGN SPHERO 0° WITH GREEN ARROW (UP)",
                        (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(disp, "Press SPACE/ENTER when done.",
                        (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Alignment", disp)
            key = cv2.waitKey(30) & 0xFF
            if key in (32, 13):
                break

        cv2.destroyWindow("Alignment")
        print("[ALIGNMENT] Alignment confirmed.\n")

        # ---- AUTO CALIBRATION (heading + scale) ----
        calibration_offset, px_per_speed_per_sec = calibrate_orientation_and_scale(
            cap, droid, model, ball_tracker
        )

        # ---- SETUP LOGGING ----
        LOGS_DIR.mkdir(parents=True, exist_ok=True)  # NEW: ensure logs/ exists
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = LOGS_DIR / f"sphero_log_{ts}.csv"  # NEW: logs folder
        log_file = open(log_filename, mode="w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "timestamp", "loop_idx",
            "est_x", "est_y",
            "meas_x", "meas_y",
            "goal_x", "goal_y", "goal_label",
            "calibration_offset_deg",
            "px_per_speed_per_sec",
            "last_command_heading_deg",
            "last_command_speed",
            "dist_to_goal",
            "dist_to_center",
            "prediction_error_px",
            "ball_found",
            "target_found",
            "safety_active",
            "stuck_frames",
            # latency-related/debug
            "meas_vx", "meas_vy",
            "est_vx", "est_vy"
        ])
        print(f"[LOG] Logging to {log_filename}")

        # STATE
        est_x = w // 2
        est_y = h // 2
        est_vx = 0.0
        est_vy = 0.0

        last_meas_x = None
        last_meas_y = None
        last_meas_time = None

        last_est_for_calib = None
        last_command_heading = None
        last_command_speed = 0.0
        last_est_for_stuck = None
        stuck_frames = 0
        last_loop_time = time.time()

        pred_errors = []
        loop_idx = 0

        print("--- VISION-BASED SERVO WITH ROBUST BALL DETECTION, SAFETY & LOGGING ---")

        try:
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                now = time.time()
                loop_idx += 1

                # --- Robust ball detection ---
                best_ball, candidates = process_vision(frame, model, ball_tracker)
                bx, by = ball_tracker.get_state(w, h)
                br = ball_tracker.radius
                ball_found = ball_tracker.frames_lost < 10

                prediction_error = float("nan")
                meas_x_for_log = float("nan")
                meas_y_for_log = float("nan")
                meas_vx_for_log = float("nan")
                meas_vy_for_log = float("nan")

                if ball_found:
                    # measurement velocities
                    if last_meas_time is not None:
                        dtm = now - last_meas_time
                        if dtm > 1e-3:
                            meas_vx = (bx - last_meas_x) / dtm
                            meas_vy = (by - last_meas_y) / dtm
                        else:
                            meas_vx, meas_vy = est_vx, est_vy
                    else:
                        meas_vx, meas_vy = 0.0, 0.0

                    last_meas_x, last_meas_y = bx, by
                    last_meas_time = now
                    meas_x_for_log = bx
                    meas_y_for_log = by
                    meas_vx_for_log = meas_vx
                    meas_vy_for_log = meas_vy

                    # --- NEW: no latency extrapolation, just smoothing ---
                    ALPHA = 0.6  # trust measurement fairly strongly
                    if loop_idx == 1 or last_meas_time is None:
                        est_x, est_y = bx, by
                    else:
                        est_x = (1-ALPHA) * est_x + ALPHA * bx
                        est_y = (1-ALPHA) * est_y + ALPHA * by

                    est_vx, est_vy = meas_vx, meas_vy

                    # prediction "error" is now just smoothing error
                    err = math.sqrt((est_x - bx)**2 + (est_y - by)**2)
                    pred_errors.append(err)
                    if len(pred_errors) > 200:
                        pred_errors.pop(0)
                    prediction_error = err
                else:
                    # Dead-reckon using command-based model if no measurement
                    dt = now - last_loop_time
                    v_cmd = px_per_speed_per_sec * last_command_speed
                    heading_world = normalize_angle(
                        (last_command_heading or 0.0) + calibration_offset
                    )
                    rad = math.radians(heading_world)
                    est_x += v_cmd * dt * math.sin(rad)
                    est_y += v_cmd * dt * -math.cos(rad)

                est_x = float(np.clip(est_x, 0, w-1))
                est_y = float(np.clip(est_y, 0, h-1))

                # --- Target detection (cat/dog/person) ---
                target_center, target_label, target_mask = detect_target(frame, model)
                target_found = target_center is not None
                if target_center is not None:
                    goal_x, goal_y = target_center
                    goal_label = target_label
                else:
                    goal_x, goal_y = w // 2, h // 2
                    goal_label = "CENTER"

                # --- Runtime calibration refinement from movement ---
                if last_est_for_calib is not None and last_command_heading is not None:
                    dxm = est_x - last_est_for_calib[0]
                    dym = est_y - last_est_for_calib[1]
                    distm = math.sqrt(dxm*dxm + dym*dym)
                    if distm > CALIB_MIN_MOVE_DIST:
                        actual_ang = get_screen_angle(last_est_for_calib, (est_x, est_y))
                        expected_ang = normalize_angle(last_command_heading + calibration_offset)
                        err_ang = angle_diff(actual_ang, expected_ang)
                        if abs(err_ang) < CALIB_MAX_ERR_TO_UPDATE:
                            calibration_offset = normalize_angle(
                                calibration_offset + 0.07 * err_ang
                            )
                last_est_for_calib = (est_x, est_y)

                # --- Stuck detection ---
                if last_est_for_stuck is None:
                    last_est_for_stuck = (est_x, est_y)

                dist_since_last = math.sqrt(
                    (est_x - last_est_for_stuck[0])**2 +
                    (est_y - last_est_for_stuck[1])**2
                )

                # --- Control logic ---
                dx = goal_x - est_x
                dy = goal_y - est_y
                dist_to_goal = math.sqrt(dx*dx + dy*dy)

                center_x, center_y = w // 2, h // 2
                dist_to_center = math.sqrt(
                    (est_x - center_x)**2 + (est_y - center_y)**2
                )

                status = ""
                should_move = False
                heading_cmd = 0.0
                speed_cmd = 0
                safety_active = False

                # Use commanded model for exit prediction
                v_model = px_per_speed_per_sec * last_command_speed
                heading_world = normalize_angle(
                    (last_command_heading or 0.0) + calibration_offset
                )
                rad = math.radians(heading_world)
                vx_model = v_model * math.sin(rad)
                vy_model = v_model * -math.cos(rad)

                will_exit, (fx, fy), near_now = will_exit_bounds(
                    est_x, est_y, vx_model, vy_model, w, h
                )

                # Safety priority
                if will_exit or is_near_boundary(est_x, est_y, w, h):
                    safety_active = True
                    angle_to_center = get_screen_angle((est_x, est_y), (center_x, center_y))
                    heading_cmd = normalize_angle(angle_to_center - calibration_offset)
                    dist_c = dist_to_center
                    speed_cmd = min(CENTER_RETURN_SPEED,
                                    max(MIN_SPEED, int(dist_c / 4)))
                    should_move = dist_c > 30
                    status = "SAFETY: retreat to center"

                elif dist_to_goal < TARGET_REACHED_PIX:
                    speed_cmd = 0
                    should_move = False
                    status = f"ARRIVED ({goal_label})"

                else:
                    # Desired direction to goal
                    desired_screen_angle = get_screen_angle((est_x, est_y), (goal_x, goal_y))
                    heading_cmd = normalize_angle(desired_screen_angle - calibration_offset)

                    # --- NEW: Distance-first speed profile ---
                    if dist_to_goal > 4 * TARGET_REACHED_PIX:
                        base_speed = MAX_SPEED
                    elif dist_to_goal > 2 * TARGET_REACHED_PIX:
                        # ramp down from MAX to MIN as we go from 4R to 2R
                        frac = (dist_to_goal - 2 * TARGET_REACHED_PIX) / (2 * TARGET_REACHED_PIX)
                        frac = max(0.0, min(1.0, frac))
                        base_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * frac
                    else:
                        base_speed = MIN_SPEED  # very close → creep

                    # --- Mild velocity-aware adjustment ---
                    if dist_to_goal > 1e-3:
                        ux = dx / dist_to_goal
                        uy = dy / dist_to_goal
                    else:
                        ux, uy = 0.0, 0.0

                    # Radial velocity toward goal (px/s)
                    v_r = est_vx * ux + est_vy * uy

                    T_BRAKE = 0.3
                    d_stop = max(0.0, v_r) * T_BRAKE  # px

                    if d_stop > 0.5 * dist_to_goal:
                        # approaching too fast, cut base speed in half
                        speed_cmd = max(MIN_SPEED, int(base_speed * 0.5))
                    else:
                        speed_cmd = int(base_speed)

                    speed_cmd = max(MIN_SPEED, min(MAX_SPEED, speed_cmd))

                    should_move = True
                    status = f"CHASING {goal_label}"

                # --- Apply command & update stuck ---
                if should_move:
                    droid.set_heading(int(heading_cmd))
                    droid.set_speed(int(speed_cmd))
                    last_command_heading = heading_cmd
                    last_command_speed = float(speed_cmd)

                    if dist_since_last < STUCK_DIST_THRESHOLD:
                        stuck_frames += 1
                    else:
                        stuck_frames = 0
                else:
                    droid.set_speed(0)
                    last_command_speed = 0.0
                    stuck_frames = 0

                # Stuck escape
                if stuck_frames >= STUCK_FRAMES_THRESHOLD:
                    safety_active = True
                    angle_to_center = get_screen_angle((est_x, est_y), (center_x, center_y))
                    heading_cmd = normalize_angle(angle_to_center - calibration_offset)
                    speed_cmd = MIN_SPEED
                    droid.set_heading(int(heading_cmd))
                    droid.set_speed(int(speed_cmd))
                    status = "STUCK: escaping to center"
                    stuck_frames = max(0, stuck_frames - 2)

                last_est_for_stuck = (est_x, est_y)

                # --- LOGGING ---
                ts_loop = datetime.datetime.now().isoformat()
                log_writer.writerow([
                    ts_loop, loop_idx,
                    est_x, est_y,
                    meas_x_for_log, meas_y_for_log,
                    goal_x, goal_y, goal_label,
                    calibration_offset,
                    px_per_speed_per_sec,
                    last_command_heading if last_command_heading is not None else float("nan"),
                    last_command_speed,
                    dist_to_goal,
                    dist_to_center,
                    prediction_error,
                    int(ball_found),
                    int(target_found),
                    int(safety_active),
                    stuck_frames,
                    meas_vx_for_log, meas_vy_for_log,
                    est_vx, est_vy
                ])
                if loop_idx % 20 == 0:
                    log_file.flush()

                # --- Visualisation ---
                vis = frame.copy()

                # overlay target mask (cat/dog/person)
                if target_mask is not None and target_mask.any():
                    overlay = np.zeros_like(vis)
                    overlay[target_mask == 255] = (0, 0, 255)
                    vis = cv2.addWeighted(vis, 1.0, overlay, 0.3, 0)

                # draw ball candidates (debug)
                for cand in candidates:
                    if 'final_score' in cand:
                        cv2.circle(vis, (cand['x'], cand['y']), cand['r'],
                                   (100, 100, 100), 1)
                        cv2.putText(vis, f"{int(cand['final_score'])}",
                                    (cand['x']+5, cand['y']-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

                if best_ball:
                    cv2.circle(vis, (best_ball['x'], best_ball['y']),
                               best_ball['r'], (0, 255, 0), 2)
                    cv2.putText(vis, best_ball['src'],
                                (best_ball['x']+best_ball['r']+5, best_ball['y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # predicted / estimated pos
                cv2.drawMarker(vis, (int(est_x), int(est_y)), (255, 255, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
                cv2.putText(vis, "EST", (int(est_x)+10, int(est_y)+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

                # goal
                cv2.circle(vis, (int(goal_x), int(goal_y)), 10, (255, 255, 255), 2)
                cv2.putText(vis, goal_label, (int(goal_x)+12, int(goal_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                # border band
                cv2.rectangle(vis,
                              (BORDER_MARGIN, BORDER_MARGIN),
                              (w-BORDER_MARGIN, h-BORDER_MARGIN),
                              (0, 0, 255), 1)

                # arrows: desired + commanded
                desired_screen_angle = get_screen_angle((est_x, est_y), (goal_x, goal_y))
                rad_des = math.radians(desired_screen_angle)
                des_end_x = int(est_x + 50 * math.sin(rad_des))
                des_end_y = int(est_y - 50 * math.cos(rad_des))
                cv2.arrowedLine(vis, (int(est_x), int(est_y)),
                                (des_end_x, des_end_y), (0, 255, 0), 2)

                if should_move or status.startswith("STUCK"):
                    cmd_global = normalize_angle(
                        (heading_cmd if last_command_heading is not None else 0.0)
                        + calibration_offset
                    )
                    rad_cmd = math.radians(cmd_global)
                    cmd_end_x = int(est_x + 50 * math.sin(rad_cmd))
                    cmd_end_y = int(est_y - 50 * math.cos(rad_cmd))
                    cv2.arrowedLine(vis, (int(est_x), int(est_y)),
                                    (cmd_end_x, cmd_end_y), (255, 0, 255), 2)

                cv2.drawMarker(vis, (w//2, h//2), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                avg_err = np.mean(pred_errors) if pred_errors else 0.0

                cv2.putText(vis, status, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                cv2.putText(vis, f"Calib offset: {int(calibration_offset)} deg",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(vis,
                            f"Scale: {px_per_speed_per_sec:.2f} px/(speed·s)  PredErr(avg): {avg_err:.1f}px",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.putText(vis, f"Est: ({int(est_x)}, {int(est_y)})  Goal: {goal_label}",
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(vis, f"Dist goal: {int(dist_to_goal)} px  Latency: {int(VISION_LATENCY_SEC*1000)} ms",
                            (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.putText(vis, f"Stuck frames: {stuck_frames}",
                            (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

                cv2.imshow("Sphero Vision Servo", vis)

                last_loop_time = now
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                elapsed = time.time() - loop_start
                if elapsed < LOOP_DELAY:
                    time.sleep(LOOP_DELAY - elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            droid.set_speed(0)
            cap.release()
            cv2.destroyAllWindows()
            log_file.flush()
            log_file.close()
            print("Shutting down.")
            print(f"[LOG] Closed {log_filename}")

if __name__ == "__main__":
    main()


# import sys
# sys.coinit_flags = 0  # Fix COM init on Windows for Bluetooth

# import time
# import math
# import csv
# import datetime
# import numpy as np
# import cv2
# import types
# from collections import deque
# from concurrent.futures import ThreadPoolExecutor
# from ultralytics import YOLO

# # --- STUBBING PLOTTING LIBS (ultralytics sometimes imports these) ---
# fake_pandas = types.ModuleType("pandas")
# sys.modules["pandas"] = fake_pandas
# sys.modules["seaborn"] = types.ModuleType("seaborn")
# sys.modules["matplotlib"] = types.ModuleType("matplotlib")
# sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")

# from spherov2 import scanner
# from spherov2.sphero_edu import SpheroEduAPI
# from spherov2.types import Color

# # ================= CONFIGURATION =================

# CAM_INDEX = 1
# LOOP_DELAY = 0.05            # Control loop ~20 Hz
# VISION_LATENCY_SEC = 0.18    # Display/assumed latency (not used for EST now)

# # --- NAVIGATION & SAFETY ---
# MAX_SPEED = 30               # Slightly conservative
# MIN_SPEED = 5
# TARGET_REACHED_PIX = 35
# BORDER_MARGIN = 100          # Bigger safe margin in small field
# PREDICT_HORIZON_SEC = 0.4    # Prediction horizon for exit check
# CENTER_RETURN_SPEED = 30

# # --- BALL DETECTION (robust: Hough + YOLO + bright blobs) ---
# BRIGHTNESS_THRESHOLD = 190
# MIN_RADIUS_PX = 8
# MAX_RADIUS_PX = 60
# MIN_CONFIDENCE = 0.25
# SEARCH_WINDOW = 100
# PRIOR_WEIGHT = 0.8
# RESET_THRESHOLD = 160
# YOLO_BALL_CLASS = 32  # COCO sports ball

# # --- YOLO TARGETS (cat/dog/person) ---
# MIN_TARGET_CONF = 0.35
# PERSON_CLASS = 0
# CAT_CLASS = 15
# DOG_CLASS = 16
# PRIMARY_TARGET_CLASSES = [CAT_CLASS, DOG_CLASS]
# SECONDARY_TARGET_CLASSES = [PERSON_CLASS]

# # --- CALIBRATION (heading + scale) ---
# CALIB_MIN_MOVE_DIST = 5
# CALIB_MAX_ERR_TO_UPDATE = 70  # for runtime refinement
# DEFAULT_PX_PER_SPEED_PER_SEC = 3.0  # fallback

# # --- STUCK DETECTION ---
# STUCK_DIST_THRESHOLD = 2.0     # px
# STUCK_FRAMES_THRESHOLD = 10    # frames of "we're not moving" while commanded

# # ================= MATH HELPERS =================

# def normalize_angle(a: float) -> float:
#     return a % 360.0

# def angle_diff(a: float, b: float) -> float:
#     return (a - b + 180.0) % 360.0 - 180.0

# def get_screen_angle(p1, p2) -> float:
#     dx = p2[0] - p1[0]
#     dy = p1[1] - p2[1]  # invert Y
#     deg = math.degrees(math.atan2(dx, dy))
#     return normalize_angle(deg)

# def is_near_boundary(x, y, w, h, margin=BORDER_MARGIN):
#     return (
#         x < margin or x > w - margin or
#         y < margin or y > h - margin
#     )

# def will_exit_bounds(x, y, vx, vy, w, h,
#                      horizon_sec=PREDICT_HORIZON_SEC,
#                      margin=BORDER_MARGIN):
#     """
#     Predict if, under current velocity, we will cross the boundary band
#     within 'horizon_sec' seconds.
#     """
#     future_x = x + vx * horizon_sec
#     future_y = y + vy * horizon_sec
#     future_x = float(np.clip(future_x, 0, w-1))
#     future_y = float(np.clip(future_y, 0, h-1))

#     near_now = is_near_boundary(x, y, w, h, margin)
#     near_future = is_near_boundary(future_x, future_y, w, h, margin)

#     return near_future, (future_x, future_y), near_now

# # ================= TRACKER (KALMAN FOR BALL) =================

# class Tracker:
#     """
#     Simple 2D position + velocity Kalman filter for the ball:
#       state = [x, y, vx, vy]^T
#     """
#     def __init__(self, process_noise=0.01, measurement_noise=1.0):
#         self.kf = cv2.KalmanFilter(4, 2)
#         self.kf.measurementMatrix = np.array(
#             [[1, 0, 0, 0],
#              [0, 1, 0, 0]], np.float32
#         )
#         self.kf.transitionMatrix = np.array(
#             [[1, 0, 1, 0],
#              [0, 1, 0, 1],
#              [0, 0, 1, 0],
#              [0, 0, 0, 1]], np.float32
#         )
#         self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
#         self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
#         self.prediction = np.zeros((4, 1), np.float32)
#         self.frames_lost = 100
#         self.radius = 20

#     def predict(self):
#         self.prediction = self.kf.predict()
#         self.frames_lost += 1
#         return int(self.prediction[0, 0]), int(self.prediction[1, 0])

#     def update(self, x, y, r=0):
#         meas = np.array([[np.float32(x)], [np.float32(y)]])
#         self.prediction = self.kf.correct(meas)
#         self.radius = r
#         self.frames_lost = 0

#     def reset(self, x, y):
#         self.kf.statePost = np.array(
#             [[np.float32(x)], [np.float32(y)], [0], [0]], np.float32
#         )
#         self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
#         self.prediction = self.kf.statePost
#         self.frames_lost = 0

#     def get_state(self, w=640, h=480):
#         x = int(np.clip(self.prediction[0, 0], 0, w-1))
#         y = int(np.clip(self.prediction[1, 0], 0, h-1))
#         return x, y

#     def get_velocity(self):
#         return float(self.prediction[2, 0]), float(self.prediction[3, 0])

# # ================= ROBUST BALL DETECTION =================

# def find_candidates(frame, model, bright_mask):
#     """
#     Ball candidate detection using:
#       - Hough circles
#       - YOLO ball detection
#       - Bright blob circularity
#     Fused into scored candidates.
#     """
#     h, w = frame.shape[:2]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     EDGE_MARGIN = 30
#     OVERLAP_DIST = 40

#     hough_detections = []
#     yolo_detections = []
#     blob_detections = []

#     # 1. Hough circles
#     circles = cv2.HoughCircles(
#         gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=60,
#         param1=150, param2=30,
#         minRadius=MIN_RADIUS_PX, maxRadius=MAX_RADIUS_PX
#     )
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             cx, cy, r = int(i[0]), int(i[1]), int(i[2])
#             if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
#                 cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
#                 continue
#             mask_roi = np.zeros_like(gray)
#             cv2.circle(mask_roi, (cx, cy), r, 255, -1)
#             mean_brightness = cv2.mean(gray, mask=mask_roi)[0]
#             if mean_brightness < BRIGHTNESS_THRESHOLD:
#                 continue
#             hough_detections.append(
#                 {'x': cx, 'y': cy, 'r': r, 'brightness': mean_brightness}
#             )

#     # 2. YOLO ball detections (class 32)
#     results = model(frame, verbose=False, conf=MIN_CONFIDENCE)
#     if results[0].boxes:
#         for box in results[0].boxes:
#             if int(box.cls[0]) == YOLO_BALL_CLASS:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 r = max((x2 - x1) // 2, (y2 - y1) // 2)
#                 conf = float(box.conf[0])
#                 if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
#                     cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
#                     continue
#                 yolo_detections.append(
#                     {'x': cx, 'y': cy, 'r': r, 'conf': conf}
#                 )

#     # 3. Bright blobs (brightness-based detection)
#     contours, _ = cv2.findContours(
#         bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < 50:
#             continue
#         perimeter = cv2.arcLength(c, True)
#         if perimeter == 0:
#             continue
#         circularity = 4 * math.pi * area / (perimeter ** 2)
#         if circularity > 0.6:
#             (x, y), r = cv2.minEnclosingCircle(c)
#             cx, cy = int(x), int(y)
#             if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
#                 cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
#                 continue
#             if MIN_RADIUS_PX < r < MAX_RADIUS_PX:
#                 blob_detections.append(
#                     {'x': cx, 'y': cy, 'r': int(r), 'circ': circularity}
#                 )

#     candidates = []

#     def distance(p1, p2):
#         return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

#     # Fuse Hough + YOLO + Blob
#     for h_det in hough_detections:
#         score = 100
#         src = 'Hough'
#         score += (h_det['brightness'] - BRIGHTNESS_THRESHOLD) / 2

#         yolo_match = None
#         for y_det in yolo_detections:
#             if distance(h_det, y_det) < OVERLAP_DIST:
#                 yolo_match = y_det
#                 break

#         blob_match = None
#         for b_det in blob_detections:
#             if distance(h_det, b_det) < OVERLAP_DIST:
#                 blob_match = b_det
#                 break

#         if yolo_match and blob_match:
#             score += 200
#             src = 'Hough+YOLO+Blob'
#         elif yolo_match:
#             score += 150
#             src = 'Hough+YOLO'
#         elif blob_match:
#             score += 50
#             src = 'Hough+Blob'

#         candidates.append({
#             'x': h_det['x'],
#             'y': h_det['y'],
#             'r': h_det['r'],
#             'score': score,
#             'src': src
#         })

#     # YOLO-only (unmatched)
#     for y_det in yolo_detections:
#         if any(distance(h_det, y_det) < OVERLAP_DIST for h_det in hough_detections):
#             continue
#         score = 120
#         src = 'YOLO'
#         blob_match = None
#         for b_det in blob_detections:
#             if distance(y_det, b_det) < OVERLAP_DIST:
#                 blob_match = b_det
#                 break
#         if blob_match:
#             score += 60
#             src = 'YOLO+Blob'
#         candidates.append({
#             'x': y_det['x'],
#             'y': y_det['y'],
#             'r': y_det['r'],
#             'score': score,
#             'src': src
#         })

#     return candidates

# def select_best_ball(candidates, tracker: Tracker):
#     if not candidates:
#         return None
#     pred_x, pred_y = tracker.get_state()
#     best_candidate = None
#     highest_adjusted_score = -1e9
#     for c in candidates:
#         dist = math.sqrt((c['x'] - pred_x)**2 + (c['y'] - pred_y)**2)
#         if tracker.frames_lost < 30:
#             if dist < SEARCH_WINDOW:
#                 spatial_bonus = (SEARCH_WINDOW - dist) * 0.2
#                 c['final_score'] = c['score'] + spatial_bonus
#             else:
#                 c['final_score'] = c['score'] - (dist * PRIOR_WEIGHT)
#                 if c['score'] > RESET_THRESHOLD:
#                     c['final_score'] = c['score']
#         else:
#             c['final_score'] = c['score']

#         if c['final_score'] > highest_adjusted_score:
#             highest_adjusted_score = c['final_score']
#             best_candidate = c
#     return best_candidate

# def process_vision(frame, model, ball_tracker: Tracker):
#     """
#     One-step robust vision processing:
#       - detect ball candidates (Hough + YOLO + bright blobs)
#       - select best via Tracker prior
#       - update ball_tracker
#       - return best_ball and candidates
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, bright_mask = cv2.threshold(
#         gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
#     )
#     bright_mask = cv2.dilate(bright_mask, None, iterations=2)

#     candidates = find_candidates(frame, model, bright_mask)
#     best_ball = select_best_ball(candidates, ball_tracker)

#     # Kalman predict and correct
#     bx_pred, by_pred = ball_tracker.predict()
#     if best_ball:
#         dist = math.sqrt((best_ball['x'] - bx_pred)**2 +
#                          (best_ball['y'] - by_pred)**2)
#         if dist > SEARCH_WINDOW and best_ball['score'] > RESET_THRESHOLD:
#             ball_tracker.reset(best_ball['x'], best_ball['y'])
#         else:
#             ball_tracker.update(best_ball['x'], best_ball['y'], best_ball['r'])

#     return best_ball, candidates

# # ================= TARGET DETECTION =================

# def detect_target(frame, model):
#     h, w = frame.shape[:2]
#     results = model(frame, verbose=False, conf=MIN_TARGET_CONF)
#     res = results[0]

#     target_mask_primary = np.zeros((h, w), dtype=np.uint8)
#     target_mask_secondary = np.zeros((h, w), dtype=np.uint8)
#     label = None
#     center = None

#     if res.masks is not None and res.boxes is not None:
#         masks = res.masks.data.cpu().numpy()
#         boxes = res.boxes
#         for i, box in enumerate(boxes):
#             cls_id = int(box.cls[0])
#             m = cv2.resize(masks[i], (w, h))
#             binary = (m > 0.5).astype(np.uint8)

#             if cls_id in PRIMARY_TARGET_CLASSES:
#                 target_mask_primary = cv2.bitwise_or(target_mask_primary, binary)
#             elif cls_id in SECONDARY_TARGET_CLASSES:
#                 target_mask_secondary = cv2.bitwise_or(target_mask_secondary, binary)

#     if target_mask_primary.any():
#         mask = target_mask_primary
#         label = "CAT/DOG"
#     elif target_mask_secondary.any():
#         mask = target_mask_secondary
#         label = "PERSON"
#     else:
#         return None, None, np.zeros((h, w), dtype=np.uint8)

#     M = cv2.moments(mask.astype(np.uint8))
#     if M["m00"] > 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         center = (cx, cy)

#     return center, label, mask.astype(np.uint8) * 255

# # ================= CALIBRATION =================

# def calibrate_orientation_and_scale(cap, droid, model, ball_tracker):
#     """
#     After manual alignment, do a short auto-calibration:
#       - Move Sphero in 4 cardinal directions
#       - Use robust ball detection via process_vision()
#       - Infer:
#           calibration_offset  (difference between Sphero 0° and camera frame)
#           px_per_speed_per_sec (pixels per unit Sphero speed per second)
#     """
#     print("\n=== AUTO CALIBRATION (heading + scale) ===")
#     calib_window_open = False

#     # Make sure ball is visible
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         best_ball, _ = process_vision(frame, model, ball_tracker)
#         bx, by = ball_tracker.get_state(frame.shape[1], frame.shape[0])
#         ball_found = ball_tracker.frames_lost < 10
#         if ball_found:
#             print(f"[CALIB] Ball found at ({bx},{by})")
#             break
#         cv2.putText(frame, "Place Sphero in view for calibration",
#                     (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         calib_window_open = True
#         cv2.imshow("Calibration", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     if calib_window_open:
#         cv2.destroyWindow("Calibration")

#     test_dirs = [0, 90, 180, 270]
#     CALIB_SPEED = 40
#     MOVE_TIME = 0.6
#     SETTLE_TIME = 0.4

#     offsets = []
#     scales = []

#     for d in test_dirs:
#         print(f"[CALIB] Testing command heading {d}°")
#         # let detection settle
#         for _ in range(5):
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             process_vision(frame, model, ball_tracker)
#             time.sleep(LOOP_DELAY)

#         ret, frame = cap.read()
#         if not ret:
#             continue
#         h, w = frame.shape[:2]
#         process_vision(frame, model, ball_tracker)
#         sx, sy = ball_tracker.get_state(w, h)
#         ball_found = ball_tracker.frames_lost < 10
#         if not ball_found:
#             print("[CALIB] Ball lost before move, skipping sample.")
#             continue
#         start = (sx, sy)

#         # Move Sphero
#         droid.set_heading(int(d))
#         droid.set_speed(CALIB_SPEED)
#         t0 = time.time()
#         while time.time() - t0 < MOVE_TIME:
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             process_vision(frame, model, ball_tracker)
#             time.sleep(LOOP_DELAY)

#         droid.set_speed(0)
#         time.sleep(SETTLE_TIME)

#         # Measure end position
#         for _ in range(3):
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             process_vision(frame, model, ball_tracker)
#             time.sleep(LOOP_DELAY)

#         ret, frame = cap.read()
#         if not ret:
#             continue
#         h, w = frame.shape[:2]
#         process_vision(frame, model, ball_tracker)
#         ex, ey = ball_tracker.get_state(w, h)
#         ball_found = ball_tracker.frames_lost < 10
#         if not ball_found:
#             print("[CALIB] Ball lost after move, skipping sample.")
#             continue
#         end = (ex, ey)

#         dx = end[0] - start[0]
#         dy = end[1] - start[1]
#         dist = math.sqrt(dx*dx + dy*dy)
#         if dist < CALIB_MIN_MOVE_DIST:
#             print(f"[CALIB] Movement too small ({dist:.1f}px), skipping.")
#             continue

#         actual_angle = get_screen_angle(start, end)
#         off = angle_diff(actual_angle, d)
#         offsets.append(off)

#         scale = dist / (CALIB_SPEED * MOVE_TIME)
#         scales.append(scale)

#         print(f"[CALIB] cmd:{d}°, actual:{actual_angle:.1f}°,"
#               f" offset:{off:.1f}°, dist:{dist:.1f}px, scale sample:{scale:.2f}")

#     if not offsets:
#         print("[CALIB] No valid samples. Using offset=0°, default scale.")
#         calibration_offset = 0.0
#         px_per_speed_per_sec = DEFAULT_PX_PER_SPEED_PER_SEC
#     else:
#         rad = np.deg2rad(offsets)
#         mean_sin = np.mean(np.sin(rad))
#         mean_cos = np.mean(np.cos(rad))
#         calibration_offset = normalize_angle(np.rad2deg(math.atan2(mean_sin, mean_cos)))
#         px_per_speed_per_sec = float(np.mean(scales))
#         print(f"[CALIB] Final orientation offset: {calibration_offset:.1f}°")
#         print(f"[CALIB] Final scale: {px_per_speed_per_sec:.2f} px/(speed·s)")

#     print("=== AUTO CALIBRATION DONE ===\n")
#     return calibration_offset, px_per_speed_per_sec

# # ================= MAIN =================

# def main():
#     print("Scanning for Sphero...")
#     toy = None
#     with ThreadPoolExecutor(max_workers=1) as ex:
#         fut = ex.submit(scanner.find_toy)
#         try:
#             toy = fut.result(timeout=10)
#         except Exception:
#             toy = None

#     if not toy:
#         print("No Sphero found.")
#         return

#     print("Loading YOLO model...")
#     model = YOLO("yolov8s-seg.pt")

#     # Ball tracker for robust detection
#     ball_tracker = Tracker(process_noise=0.01, measurement_noise=1.0)

#     with SpheroEduAPI(toy) as droid:
#         print("[SPHERO] Setting up robot...")
#         droid.set_speed(0)
#         droid.set_main_led(Color(255, 255, 255))
#         time.sleep(0.3)

#         cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
#         if not cap.isOpened():
#             cap = cv2.VideoCapture(CAM_INDEX)
#         if not cap.isOpened():
#             print(f"[CAMERA] Could not open camera index {CAM_INDEX}")
#             return

#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("[CAMERA] Cannot read first frame.")
#             return

#         h, w = frame.shape[:2]
#         print(f"[CAMERA] Resolution: {w}x{h}")

#         print("[SPHERO] Resetting heading reference (0°)...")
#         droid.reset_aim()
#         time.sleep(0.8)

#         # Manual alignment screen (unchanged)
#         print("\n" + "="*60)
#         print("[ALIGNMENT] Manual alignment step.")
#         print("[ALIGNMENT] Imagine Sphero's 'forward' (0° heading) pointing straight up in the camera view.")
#         print("[ALIGNMENT] Place/rotate Sphero so its physical forward matches 'up' in the video.")
#         print("[ALIGNMENT] Press SPACE or ENTER in the video window when ready.")
#         print("="*60 + "\n")

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             disp = frame.copy()
#             h, w = disp.shape[:2]
#             arrow_tip = (w // 2, 40)
#             arrow_base = (w // 2, 130)
#             arrow_width = 40
#             cv2.line(disp, arrow_base, arrow_tip, (0, 255, 0), 8)
#             pts = np.array([
#                 [arrow_tip[0], arrow_tip[1]],
#                 [arrow_tip[0] - arrow_width, arrow_tip[1] + arrow_width],
#                 [arrow_tip[0] + arrow_width, arrow_tip[1] + arrow_width]
#             ], np.int32)
#             cv2.fillPoly(disp, [pts], (0, 255, 0))

#             cv2.putText(disp, "ALIGN SPHERO 0° WITH GREEN ARROW (UP)",
#                         (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#             cv2.putText(disp, "Press SPACE/ENTER when done.",
#                         (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

#             cv2.imshow("Alignment", disp)
#             key = cv2.waitKey(30) & 0xFF
#             if key in (32, 13):
#                 break

#         cv2.destroyWindow("Alignment")
#         print("[ALIGNMENT] Alignment confirmed.\n")

#         # ---- AUTO CALIBRATION (heading + scale) ----
#         calibration_offset, px_per_speed_per_sec = calibrate_orientation_and_scale(
#             cap, droid, model, ball_tracker
#         )

#         # ---- SETUP LOGGING ----
#         ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         log_filename = f"sphero_log_{ts}.csv"
#         log_file = open(log_filename, mode="w", newline="")
#         log_writer = csv.writer(log_file)
#         log_writer.writerow([
#             "timestamp", "loop_idx",
#             "est_x", "est_y",
#             "meas_x", "meas_y",
#             "goal_x", "goal_y", "goal_label",
#             "calibration_offset_deg",
#             "px_per_speed_per_sec",
#             "last_command_heading_deg",
#             "last_command_speed",
#             "dist_to_goal",
#             "dist_to_center",
#             "prediction_error_px",
#             "ball_found",
#             "target_found",
#             "safety_active",
#             "stuck_frames",
#             # latency-related/debug
#             "meas_vx", "meas_vy",
#             "est_vx", "est_vy"
#         ])
#         print(f"[LOG] Logging to {log_filename}")

#         # STATE
#         est_x = w // 2
#         est_y = h // 2
#         est_vx = 0.0
#         est_vy = 0.0

#         last_meas_x = None
#         last_meas_y = None
#         last_meas_time = None

#         last_est_for_calib = None
#         last_command_heading = None
#         last_command_speed = 0.0
#         last_est_for_stuck = None
#         stuck_frames = 0
#         last_loop_time = time.time()

#         pred_errors = []
#         loop_idx = 0

#         print("--- VISION-BASED SERVO WITH ROBUST BALL DETECTION, SAFETY & LOGGING ---")

#         try:
#             while True:
#                 loop_start = time.time()
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 h, w = frame.shape[:2]
#                 now = time.time()
#                 loop_idx += 1

#                 # --- Robust ball detection ---
#                 best_ball, candidates = process_vision(frame, model, ball_tracker)
#                 bx, by = ball_tracker.get_state(w, h)
#                 br = ball_tracker.radius
#                 ball_found = ball_tracker.frames_lost < 10

#                 prediction_error = float("nan")
#                 meas_x_for_log = float("nan")
#                 meas_y_for_log = float("nan")
#                 meas_vx_for_log = float("nan")
#                 meas_vy_for_log = float("nan")

#                 if ball_found:
#                     # measurement velocities
#                     if last_meas_time is not None:
#                         dtm = now - last_meas_time
#                         if dtm > 1e-3:
#                             meas_vx = (bx - last_meas_x) / dtm
#                             meas_vy = (by - last_meas_y) / dtm
#                         else:
#                             meas_vx, meas_vy = est_vx, est_vy
#                     else:
#                         meas_vx, meas_vy = 0.0, 0.0

#                     last_meas_x, last_meas_y = bx, by
#                     last_meas_time = now
#                     meas_x_for_log = bx
#                     meas_y_for_log = by
#                     meas_vx_for_log = meas_vx
#                     meas_vy_for_log = meas_vy

#                     # no latency extrapolation, just smoothing
#                     ALPHA = 0.6  # trust measurement fairly strongly
#                     if loop_idx == 1 or last_meas_time is None:
#                         est_x, est_y = bx, by
#                     else:
#                         est_x = (1-ALPHA) * est_x + ALPHA * bx
#                         est_y = (1-ALPHA) * est_y + ALPHA * by

#                     est_vx, est_vy = meas_vx, meas_vy

#                     # smoothing error
#                     err = math.sqrt((est_x - bx)**2 + (est_y - by)**2)
#                     pred_errors.append(err)
#                     if len(pred_errors) > 200:
#                         pred_errors.pop(0)
#                     prediction_error = err
#                 else:
#                     # Dead-reckon using command-based model if no measurement
#                     dt = now - last_loop_time
#                     v_cmd = px_per_speed_per_sec * last_command_speed
#                     heading_world = normalize_angle(
#                         (last_command_heading or 0.0) + calibration_offset
#                     )
#                     rad = math.radians(heading_world)
#                     est_x += v_cmd * dt * math.sin(rad)
#                     est_y += v_cmd * dt * -math.cos(rad)

#                 est_x = float(np.clip(est_x, 0, w-1))
#                 est_y = float(np.clip(est_y, 0, h-1))

#                 # --- Target detection (cat/dog/person) ---
#                 target_center, target_label, target_mask = detect_target(frame, model)
#                 target_found = target_center is not None
#                 if target_center is not None:
#                     goal_x, goal_y = target_center
#                     goal_label = target_label
#                 else:
#                     goal_x, goal_y = w // 2, h // 2
#                     goal_label = "CENTER"

#                 # --- Runtime calibration refinement from movement ---
#                 if last_est_for_calib is not None and last_command_heading is not None:
#                     dxm = est_x - last_est_for_calib[0]
#                     dym = est_y - last_est_for_calib[1]
#                     distm = math.sqrt(dxm*dxm + dym*dym)
#                     if distm > CALIB_MIN_MOVE_DIST:
#                         actual_ang = get_screen_angle(last_est_for_calib, (est_x, est_y))
#                         expected_ang = normalize_angle(last_command_heading + calibration_offset)
#                         err_ang = angle_diff(actual_ang, expected_ang)
#                         if abs(err_ang) < CALIB_MAX_ERR_TO_UPDATE:
#                             calibration_offset = normalize_angle(
#                                 calibration_offset + 0.07 * err_ang
#                             )
#                 last_est_for_calib = (est_x, est_y)

#                 # --- Stuck detection ---
#                 if last_est_for_stuck is None:
#                     last_est_for_stuck = (est_x, est_y)

#                 dist_since_last = math.sqrt(
#                     (est_x - last_est_for_stuck[0])**2 +
#                     (est_y - last_est_for_stuck[1])**2
#                 )

#                 # --- Control logic ---
#                 dx = goal_x - est_x
#                 dy = goal_y - est_y
#                 dist_to_goal = math.sqrt(dx*dx + dy*dy)

#                 center_x, center_y = w // 2, h // 2
#                 dist_to_center = math.sqrt(
#                     (est_x - center_x)**2 + (est_y - center_y)**2
#                 )

#                 status = ""
#                 should_move = False
#                 heading_cmd = 0.0
#                 speed_cmd = 0
#                 safety_active = False

#                 # Use commanded model for exit prediction
#                 v_model = px_per_speed_per_sec * last_command_speed
#                 heading_world = normalize_angle(
#                     (last_command_heading or 0.0) + calibration_offset
#                 )
#                 rad = math.radians(heading_world)
#                 vx_model = v_model * math.sin(rad)
#                 vy_model = v_model * -math.cos(rad)

#                 will_exit, (fx, fy), near_now = will_exit_bounds(
#                     est_x, est_y, vx_model, vy_model, w, h
#                 )

#                 # --- SAFETY ---
#                 if will_exit or is_near_boundary(est_x, est_y, w, h):
#                     safety_active = True
#                     angle_to_center = get_screen_angle((est_x, est_y), (center_x, center_y))
#                     heading_cmd = normalize_angle(angle_to_center - calibration_offset)
#                     dist_c = dist_to_center
#                     speed_cmd = min(CENTER_RETURN_SPEED,
#                                     max(MIN_SPEED, int(dist_c / 4)))
#                     should_move = dist_c > 10
#                     status = "SAFETY: retreat to center"

#                 else:
#                     # --- PD CONTROL TOWARD GOAL ---
#                     # Desired direction to goal
#                     desired_screen_angle = get_screen_angle((est_x, est_y), (goal_x, goal_y))
#                     heading_cmd = normalize_angle(desired_screen_angle - calibration_offset)

#                     # Compute radial velocity toward goal
#                     if dist_to_goal > 1e-3:
#                         ux = dx / dist_to_goal
#                         uy = dy / dist_to_goal
#                     else:
#                         ux, uy = 0.0, 0.0

#                     v_r = est_vx * ux + est_vy * uy  # px/s toward goal

#                     # PD gains (tune as needed)
#                     # We want MAX_SPEED around ~4*R distance
#                     R = TARGET_REACHED_PIX
#                     k_p = MAX_SPEED / (4.0 * R)   # proportional gain
#                     k_d = 0.15                    # derivative gain (braking)

#                     # Raw PD output in "speed units"
#                     pd_speed = k_p * dist_to_goal - k_d * v_r

#                     # Clamp to [0, MAX_SPEED], then apply MIN_SPEED if >0
#                     if pd_speed <= 0:
#                         speed_cmd = 0
#                     else:
#                         speed_cmd = min(MAX_SPEED, pd_speed)
#                         if speed_cmd < MIN_SPEED:
#                             # Only enforce MIN_SPEED if we are not essentially at the goal
#                             if dist_to_goal > 5:
#                                 speed_cmd = MIN_SPEED
#                             else:
#                                 speed_cmd = 0

#                     if speed_cmd > 0:
#                         should_move = True
#                         status = f"CHASING {goal_label}"
#                     else:
#                         should_move = False
#                         status = f"ARRIVED ({goal_label})" if dist_to_goal <= 5 else "BRAKING"

#                 # --- Apply command & update stuck ---
#                 if should_move:
#                     droid.set_heading(int(heading_cmd))
#                     droid.set_speed(int(speed_cmd))
#                     last_command_heading = heading_cmd
#                     last_command_speed = float(speed_cmd)

#                     if dist_since_last < STUCK_DIST_THRESHOLD:
#                         stuck_frames += 1
#                     else:
#                         stuck_frames = 0
#                 else:
#                     droid.set_speed(0)
#                     last_command_speed = 0.0
#                     stuck_frames = 0

#                 # Stuck escape
#                 if stuck_frames >= STUCK_FRAMES_THRESHOLD:
#                     safety_active = True
#                     angle_to_center = get_screen_angle((est_x, est_y), (center_x, center_y))
#                     heading_cmd = normalize_angle(angle_to_center - calibration_offset)
#                     speed_cmd = MIN_SPEED
#                     droid.set_heading(int(heading_cmd))
#                     droid.set_speed(int(speed_cmd))
#                     status = "STUCK: escaping to center"
#                     stuck_frames = max(0, stuck_frames - 2)

#                 last_est_for_stuck = (est_x, est_y)

#                 # --- LOGGING ---
#                 ts_loop = datetime.datetime.now().isoformat()
#                 log_writer.writerow([
#                     ts_loop, loop_idx,
#                     est_x, est_y,
#                     meas_x_for_log, meas_y_for_log,
#                     goal_x, goal_y, goal_label,
#                     calibration_offset,
#                     px_per_speed_per_sec,
#                     last_command_heading if last_command_heading is not None else float("nan"),
#                     last_command_speed,
#                     dist_to_goal,
#                     dist_to_center,
#                     prediction_error,
#                     int(ball_found),
#                     int(target_found),
#                     int(safety_active),
#                     stuck_frames,
#                     meas_vx_for_log, meas_vy_for_log,
#                     est_vx, est_vy
#                 ])
#                 if loop_idx % 20 == 0:
#                     log_file.flush()

#                 # --- Visualisation ---
#                 vis = frame.copy()

#                 # overlay target mask (cat/dog/person)
#                 if target_mask is not None and target_mask.any():
#                     overlay = np.zeros_like(vis)
#                     overlay[target_mask == 255] = (0, 0, 255)
#                     vis = cv2.addWeighted(vis, 1.0, overlay, 0.3, 0)

#                 # draw ball candidates (debug)
#                 for cand in candidates:
#                     if 'final_score' in cand:
#                         cv2.circle(vis, (cand['x'], cand['y']), cand['r'],
#                                    (100, 100, 100), 1)
#                         cv2.putText(vis, f"{int(cand['final_score'])}",
#                                     (cand['x']+5, cand['y']-5),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

#                 if best_ball:
#                     cv2.circle(vis, (best_ball['x'], best_ball['y']),
#                                best_ball['r'], (0, 255, 0), 2)
#                     cv2.putText(vis, best_ball['src'],
#                                 (best_ball['x']+best_ball['r']+5, best_ball['y']),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#                 # predicted / estimated pos
#                 cv2.drawMarker(vis, (int(est_x), int(est_y)), (255, 255, 0),
#                                markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
#                 cv2.putText(vis, "EST", (int(est_x)+10, int(est_y)+10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

#                 # goal
#                 cv2.circle(vis, (int(goal_x), int(goal_y)), 10, (255, 255, 255), 2)
#                 cv2.putText(vis, goal_label, (int(goal_x)+12, int(goal_y)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

#                 # border band
#                 cv2.rectangle(vis,
#                               (BORDER_MARGIN, BORDER_MARGIN),
#                               (w-BORDER_MARGIN, h-BORDER_MARGIN),
#                               (0, 0, 255), 1)

#                 # arrows: desired + commanded
#                 desired_screen_angle = get_screen_angle((est_x, est_y), (goal_x, goal_y))
#                 rad_des = math.radians(desired_screen_angle)
#                 des_end_x = int(est_x + 50 * math.sin(rad_des))
#                 des_end_y = int(est_y - 50 * math.cos(rad_des))
#                 cv2.arrowedLine(vis, (int(est_x), int(est_y)),
#                                 (des_end_x, des_end_y), (0, 255, 0), 2)

#                 if should_move or status.startswith("STUCK"):
#                     cmd_global = normalize_angle(
#                         (heading_cmd if last_command_heading is not None else 0.0)
#                         + calibration_offset
#                     )
#                     rad_cmd = math.radians(cmd_global)
#                     cmd_end_x = int(est_x + 50 * math.sin(rad_cmd))
#                     cmd_end_y = int(est_y - 50 * math.cos(rad_cmd))
#                     cv2.arrowedLine(vis, (int(est_x), int(est_y)),
#                                     (cmd_end_x, cmd_end_y), (255, 0, 255), 2)

#                 cv2.drawMarker(vis, (w//2, h//2), (0, 0, 255),
#                                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

#                 avg_err = np.mean(pred_errors) if pred_errors else 0.0

#                 cv2.putText(vis, status, (20, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
#                 cv2.putText(vis, f"Calib offset: {int(calibration_offset)} deg",
#                             (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
#                 cv2.putText(vis,
#                             f"Scale: {px_per_speed_per_sec:.2f} px/(speed·s)  PredErr(avg): {avg_err:.1f}px",
#                             (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
#                 cv2.putText(vis, f"Est: ({int(est_x)}, {int(est_y)})  Goal: {goal_label}",
#                             (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
#                 cv2.putText(vis, f"Dist goal: {int(dist_to_goal)} px  Latency: {int(VISION_LATENCY_SEC*1000)} ms",
#                             (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
#                 cv2.putText(vis, f"Stuck frames: {stuck_frames}",
#                             (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

#                 cv2.imshow("Sphero Vision Servo", vis)

#                 last_loop_time = now
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#                 elapsed = time.time() - loop_start
#                 if elapsed < LOOP_DELAY:
#                     time.sleep(LOOP_DELAY - elapsed)

#         except KeyboardInterrupt:
#             pass
#         finally:
#             droid.set_speed(0)
#             cap.release()
#             cv2.destroyAllWindows()
#             log_file.flush()
#             log_file.close()
#             print("Shutting down.")
#             print(f"[LOG] Closed {log_filename}")

# if __name__ == "__main__":
#     main()

