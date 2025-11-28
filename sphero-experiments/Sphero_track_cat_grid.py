import sys
sys.coinit_flags = 0  # required on Windows for COM (camera + BT)

import time
import math
import types
from collections import deque

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ---- Stub out pandas/matplotlib if missing (ultralytics sometimes assumes them) ----
fake_pandas = types.ModuleType("pandas")
sys.modules["pandas"] = fake_pandas
sys.modules["seaborn"] = types.ModuleType("seaborn")
sys.modules["matplotlib"] = types.ModuleType("matplotlib")
sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")

# ================= CONFIGURATION =================

CAM_INDEX = 1
LOOP_DELAY = 0.05          # ~20 FPS

# --- LATENCY ESTIMATION ---
LATENCY_INIT = 0.18              # Initial guess (seconds)
LATENCY_MAX_FOR_PRED = 0.10      # Cap prediction horizon at 100 ms
LATENCY_SPEED_EVENT_THRESHOLD = 10  # speed above which we consider "moving"
LATENCY_MOVE_THRESH_PX = 5.0     # movement to detect motion onset
LATENCY_BETA = 0.3               # smoothing for latency estimate

# --- NAVIGATION & SAFETY ---
MAX_SPEED = 30             # Max pursuit speed
MIN_SPEED = 6              # Min approach speed
TARGET_REACHED_PIX = 35    # px, target arrival threshold
BRAKE_TIME = 0.3           # seconds, braking time constant for velocity-aware stopping
BORDER_MARGIN = 100        # px from edge
PREDICT_HORIZON_SEC = 0.4  # Prediction horizon for exit check
STUCK_DIST_THRESHOLD = 2   # px/frame threshold for "no movement"
STUCK_FRAMES_THRESHOLD = 10  # frames of no movement → stuck
RETURN_TO_CENTER_AFTER_IDLE = 120  # frames with no target (~6s)
CENTER_RETURN_SPEED = 30   # Speed when returning to center

# --- CALIBRATION ---
CALIB_MIN_MOVE_DIST = 5
CALIB_MAX_ERR_TO_UPDATE = 70  # for runtime refinement
DEFAULT_PX_PER_SPEED_PER_SEC = 3.0  # fallback

# --- COLORS ---
COL_WINNER = (0, 255, 0)
COL_TEXT = (255, 255, 255)
COL_SAFETY = (0, 0, 255)
COL_PRED_HEAD = (255, 0, 0)
COL_CMD_HEAD = (255, 0, 255)

# --- DETECTION ---
MIN_CONFIDENCE = 0.25
BRIGHTNESS_THRESHOLD = 190
MIN_RADIUS = 8
MAX_RADIUS = 60
SEARCH_WINDOW = 100
PRIOR_WEIGHT = 0.8
RESET_THRESHOLD = 160

# --- TARGET PRIORITY ---
PERSON_CLASS = 0
CAT_CLASS = 15
DOG_CLASS = 16
PRIMARY_TARGET_CLASSES = [CAT_CLASS, DOG_CLASS]
SECONDARY_TARGET_CLASSES = [PERSON_CLASS]

# ================= MATH HELPERS =================

def normalize_angle(a: float) -> float:
    return a % 360

def angle_diff(a: float, b: float) -> float:
    """Minimal signed difference between angles a and b in (-180, 180]."""
    return (a - b + 180) % 360 - 180

def get_screen_angle(p1, p2) -> float:
    """
    Angle from p1 to p2 in screen coordinates.
    0° = up, 90° = right (accounting for inverted screen Y).
    """
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]  # invert Y (screen y grows downwards)
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

# ================= KALMAN TRACKER =================

class Tracker:
    """Simple constant-velocity Kalman filter in image space."""
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
        self.debug_frame_count = 0

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
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.prediction = self.kf.statePost
        self.frames_lost = 0

    def get_state(self, w=640, h=480):
        x = int(np.clip(self.prediction[0, 0], 0, w))
        y = int(np.clip(self.prediction[1, 0], 0, h))
        return x, y

    def get_velocity(self):
        vx = float(self.prediction[2, 0])
        vy = float(self.prediction[3, 0])
        return vx, vy

    def is_off_screen(self, w=640, h=480, margin=50):
        x = self.prediction[0, 0]
        y = self.prediction[1, 0]
        return (x < -margin or x > w + margin or
                y < -margin or y > h + margin)

# ================= BALL (SPHERO) DETECTION =================

def find_candidates(frame, model, bright_mask):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    EDGE_MARGIN = 30
    OVERLAP_DIST = 40

    hough_detections = []
    yolo_detections = []
    blob_detections = []

    # Hough circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=60,
        param1=150, param2=30,
        minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
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

    # YOLO ball detections (class 32)
    results = model(frame, verbose=False, conf=MIN_CONFIDENCE)
    if results[0].boxes:
        for box in results[0].boxes:
            if int(box.cls[0]) == 32:
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

    # Bright blobs
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
            if MIN_RADIUS < r < MAX_RADIUS:
                blob_detections.append(
                    {'x': cx, 'y': cy, 'r': int(r), 'circ': circularity}
                )

    candidates = []

    def distance(p1, p2):
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

    # Fuse Hough + YOLO + blob
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

# ================= VISION (BALL + TARGET) =================

def process_vision(frame, model, ball_tracker: Tracker, target_tracker: Tracker):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(
        gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    bright_mask = cv2.dilate(bright_mask, None, iterations=2)

    # --- ball (Sphero LED) ---
    candidates = find_candidates(frame, model, bright_mask)
    best_ball = select_best_ball(candidates, ball_tracker)
    bx_pred, by_pred = ball_tracker.predict()

    if best_ball:
        dist = math.sqrt((best_ball['x'] - bx_pred)**2 + (best_ball['y'] - by_pred)**2)
        if dist > SEARCH_WINDOW and best_ball['score'] > RESET_THRESHOLD:
            ball_tracker.reset(best_ball['x'], best_ball['y'])
        else:
            ball_tracker.update(best_ball['x'], best_ball['y'], best_ball['r'])

    # --- target detection (cat/dog/person via segmentation) ---
    results = model(frame, verbose=False, conf=MIN_CONFIDENCE)
    primary_mask = np.zeros((h, w), dtype=np.uint8)
    secondary_mask = np.zeros((h, w), dtype=np.uint8)

    debug_detections = []
    primary_found_count = 0
    secondary_found_count = 0
    chosen_target = None

    if results[0].boxes:
        has_masks = results[0].masks is not None
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            debug_detections.append((cls_id, conf))
            if not has_masks or i >= len(results[0].masks.data):
                continue
            m = cv2.resize(results[0].masks.data[i].cpu().numpy(), (w, h))
            binary_m = (m > 0.5).astype(np.uint8) * 255
            if cls_id in PRIMARY_TARGET_CLASSES:
                primary_found_count += 1
                primary_mask = cv2.bitwise_or(primary_mask, binary_m)
            elif cls_id in SECONDARY_TARGET_CLASSES:
                secondary_found_count += 1
                secondary_mask = cv2.bitwise_or(secondary_mask, binary_m)

    if primary_found_count > 0:
        target_mask = primary_mask
        chosen_target = "CAT/DOG"
        target_color = (0, 0, 255)
    elif secondary_found_count > 0:
        target_mask = secondary_mask
        chosen_target = "PERSON"
        target_color = (255, 165, 0)
    else:
        target_mask = np.zeros((h, w), dtype=np.uint8)
        chosen_target = None
        target_color = (128, 128, 128)

    # Only track targets if target_tracker is provided
    if target_tracker is not None:
        target_tracker.debug_frame_count += 1
        if target_tracker.debug_frame_count % 60 == 0:
            if debug_detections:
                print(f"[TARGET DEBUG] first 5: {debug_detections[:5]}")
                print(f"[TARGET DEBUG] Cat/Dog: {primary_found_count}, Person: {secondary_found_count}")
                print(f"[TARGET DEBUG] chosen: {chosen_target}, has_masks: {results[0].masks is not None}")
            else:
                print("[TARGET DEBUG] no detections")

        M = cv2.moments(target_mask)
        if M["m00"] > 0:
            tx = int(M["m10"] / M["m00"])
            ty = int(M["m01"] / M["m00"])
            target_tracker.update(tx, ty, 0)
            if target_tracker.debug_frame_count % 60 == 0:
                print(f"[TARGET DEBUG] center=({tx},{ty}), area={int(M['m00'])}")
        else:
            target_tracker.predict()
            if target_tracker.debug_frame_count % 60 == 0 and chosen_target:
                print("[TARGET DEBUG] mask detected but no moments?")

    return best_ball, target_mask, candidates, chosen_target, target_color

# ================= CALIBRATION =================

def calibrate_orientation_and_scale(cap, droid, model, ball_tracker):
    """
    After manual alignment, do a short auto-calibration:
      - Move Sphero in 4 cardinal directions (0, 90, 180, 270)
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
        best_ball, target_mask, candidates, chosen_target, target_color = process_vision(
            frame, model, ball_tracker, None
        )
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
            process_vision(frame, model, ball_tracker, None)
            time.sleep(LOOP_DELAY)

        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        process_vision(frame, model, ball_tracker, None)
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
            process_vision(frame, model, ball_tracker, None)
            time.sleep(LOOP_DELAY)

        droid.set_speed(0)
        time.sleep(SETTLE_TIME)

        # Measure end position
        for _ in range(3):
            ret, frame = cap.read()
            if not ret:
                continue
            process_vision(frame, model, ball_tracker, None)
            time.sleep(LOOP_DELAY)

        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        process_vision(frame, model, ball_tracker, None)
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
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(scanner.find_toy)
        try:
            toy = future.result(timeout=10)
        except Exception as e:
            print(f"[ERROR] Scanner exception: {type(e).__name__}: {e}")
            toy = None

    if not toy:
        print("No Sphero found.")
        return

    print("Loading YOLO model...")
    model = YOLO("yolov8s-seg.pt")

    ball_tracker = Tracker(process_noise=0.01, measurement_noise=1.0)
    target_tracker = Tracker(process_noise=0.15, measurement_noise=2.0)

    with SpheroEduAPI(toy) as droid:
        print("[SPHERO] Initializing...")
        droid.set_speed(0)
        time.sleep(0.2)
        droid.set_main_led(Color(255, 255, 255))
        time.sleep(0.2)

        # Camera
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[CAMERA] DSHOW failed, trying default...")
            cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            print("[CAMERA] could not open")
            return

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("[CAMERA] cannot read first frame")
            return

        h0, w0 = test_frame.shape[:2]
        print(f"[CAMERA] opened {CAM_INDEX}, resolution: {w0}x{h0}")

        # Reset heading
        print("[SPHERO] reset heading (0°)...")
        droid.reset_aim()
        time.sleep(1.0)

        # --- Manual alignment with LED arrow and overlay ---
        print("\n" + "="*60)
        print("[ALIGNMENT] Sphero will display UP ARROW on LED matrix.")
        print("[ALIGNMENT] Rotate Sphero so arrow points toward GREEN ARROW on screen (up).")
        print("[ALIGNMENT] Press SPACE or ENTER when aligned.")
        print("="*60 + "\n")

        arrow_pattern = [
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
        ]

        try:
            try:
                droid.set_matrix_character("^")
            except Exception:
                try:
                    for y in range(8):
                        for x in range(8):
                            if arrow_pattern[y][x] == 1:
                                droid.set_matrix_pixel(x, y, Color(0, 0, 255))
                            else:
                                droid.set_matrix_pixel(x, y, Color(0, 0, 0))
                except Exception:
                    droid.set_matrix(arrow_pattern)
        except Exception:
            print("[ALIGNMENT] no LED matrix, falling back to blue LED")
            droid.set_main_led(Color(0, 0, 255))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            disp = frame.copy()

            arrow_tip = (w // 2, 30)
            arrow_base = (w // 2, 100)
            arrow_width = 40
            cv2.line(disp, arrow_base, arrow_tip, (0, 255, 0), 8)
            pts = np.array([
                [arrow_tip[0], arrow_tip[1]],
                [arrow_tip[0] - arrow_width, arrow_tip[1] + arrow_width],
                [arrow_tip[0] + arrow_width, arrow_tip[1] + arrow_width]
            ], np.int32)
            cv2.fillPoly(disp, [pts], (0, 255, 0))
            cv2.putText(disp, "ALIGN SPHERO ARROW WITH GREEN ARROW",
                        (w//2 - 280, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
            cv2.putText(disp, "Press SPACE or ENTER when ready",
                        (w//2 - 220, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imshow("Sphero Alignment", disp)
            key = cv2.waitKey(30)
            if key in (32, 13):  # space or enter
                break

        cv2.destroyWindow("Sphero Alignment")
        # Clear matrix, set back to white
        try:
            clear_matrix = [[0]*8 for _ in range(8)]
            droid.set_matrix(clear_matrix)
        except Exception:
            pass
        droid.set_main_led(Color(255, 255, 255))
        time.sleep(0.5)

        print("[ALIGNMENT] done.")

        # ---- AUTO CALIBRATION (square pattern: 0, 90, 180, 270) ----
        calibration_offset, px_per_speed_per_sec = calibrate_orientation_and_scale(
            cap, droid, model, ball_tracker
        )

        print("[SPHERO] Ready for tracking.")

        # Control state
        est_x = w // 2
        est_y = h // 2
        est_vx = 0.0
        est_vy = 0.0

        last_pos = None
        last_meas_time = None
        last_command_heading = 0
        last_speed = 0
        stuck_frames = 0
        idle_frames = 0

        # Latency estimation
        latency_est = LATENCY_INIT
        pending_latency_event = None  # dict with keys: time, x, y
        last_speed_for_latency = 0.0

        print("--- LATENCY-COMPENSATED VISUAL TRACKER ---")
        print("Target priority: CAT/DOG > PERSON. Default: center.")

        try:
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]

                best_ball, target_mask, candidates, chosen_target, target_color = process_vision(
                    frame, model, ball_tracker, target_tracker
                )
                bx, by = ball_tracker.get_state(w, h)
                vx, vy = ball_tracker.get_velocity()

                tx, ty = target_tracker.get_state(w, h)
                tvx, tvy = target_tracker.get_velocity()

                ball_found = ball_tracker.frames_lost < 20
                target_found = target_tracker.frames_lost < 30 and chosen_target is not None

                # --- Update state estimation with velocity tracking ---
                now = time.time()
                if ball_found:
                    # measurement velocities
                    if last_meas_time is not None and last_pos is not None:
                        dt = now - last_meas_time
                        if dt > 1e-3:
                            meas_vx = (bx - last_pos[0]) / dt
                            meas_vy = (by - last_pos[1]) / dt
                        else:
                            meas_vx, meas_vy = est_vx, est_vy
                    else:
                        meas_vx, meas_vy = 0.0, 0.0

                    last_pos = (bx, by)
                    last_meas_time = now

                    # SMALL prediction horizon based on latency_est (clamped)
                    horizon = min(latency_est, LATENCY_MAX_FOR_PRED)
                    pred_x = bx + meas_vx * horizon
                    pred_y = by + meas_vy * horizon

                    # smooth measurement + tiny prediction
                    ALPHA = 0.5
                    est_x = (1-ALPHA) * est_x + ALPHA * pred_x
                    est_y = (1-ALPHA) * est_y + ALPHA * pred_y

                    est_vx, est_vy = meas_vx, meas_vy
                else:
                    # Dead-reckon using command-based model if no measurement
                    if last_meas_time is not None:
                        dt = now - last_meas_time
                        v_cmd = px_per_speed_per_sec * last_speed
                        heading_world = normalize_angle(
                            (last_command_heading or 0.0) + calibration_offset
                        )
                        rad = math.radians(heading_world)
                        est_x += v_cmd * dt * math.sin(rad)
                        est_y += v_cmd * dt * -math.cos(rad)
                        last_meas_time = now

                est_x = float(np.clip(est_x, 0, w-1))
                est_y = float(np.clip(est_y, 0, h-1))

                # --- Latency estimation from command → movement ---
                if pending_latency_event is not None and ball_found:
                    ex, ey = pending_latency_event["x"], pending_latency_event["y"]
                    dmove = math.sqrt((bx - ex)**2 + (by - ey)**2)
                    if dmove > LATENCY_MOVE_THRESH_PX:
                        sample = now - pending_latency_event["time"]
                        latency_est = (1 - LATENCY_BETA) * latency_est + LATENCY_BETA * sample
                        pending_latency_event = None

                # --- Target prediction for chasing ---
                tx_pred = int(tx + tvx * min(latency_est, LATENCY_MAX_FOR_PRED))
                ty_pred = int(ty + tvy * min(latency_est, LATENCY_MAX_FOR_PRED))
                tx_pred = int(np.clip(tx_pred, 0, w-1))
                ty_pred = int(np.clip(ty_pred, 0, h-1))

                vis = frame.copy()
                # overlay target mask
                overlay = np.zeros_like(vis)
                overlay[target_mask == 255] = target_color
                vis = cv2.addWeighted(vis, 1.0, overlay, 0.4, 0)

                # draw candidates
                for cand in candidates:
                    if 'final_score' in cand:
                        cv2.circle(vis, (cand['x'], cand['y']), cand['r'], (100, 100, 100), 1)
                        cv2.putText(vis, f"{int(cand['final_score'])}",
                                    (cand['x']+5, cand['y']-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

                if best_ball:
                    cv2.circle(vis, (best_ball['x'], best_ball['y']), best_ball['r'], COL_WINNER, 2)
                    cv2.putText(vis, best_ball['src'],
                                (best_ball['x']+best_ball['r']+5, best_ball['y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WINNER, 1)

                # draw ball measurement and state estimate
                cv2.drawMarker(vis, (bx, by), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.circle(vis, (int(est_x), int(est_y)), 6, (0, 255, 255), 2)
                cv2.putText(vis, "EST", (int(est_x)+8, int(est_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # center cross (default waypoint)
                cx0, cy0 = w//2, h//2
                cv2.circle(vis, (cx0, cy0), 30, (0, 0, 255), 2)
                cv2.drawMarker(vis, (cx0, cy0), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

                status = ""
                heading_cmd = None
                speed_cmd = 0
                should_move = False

                # --- Control logic: dist to goal/center ---
                dist_to_goal = 0.0
                center_x, center_y = w // 2, h // 2
                dist_to_center = math.sqrt(
                    (est_x - center_x)**2 + (est_y - center_y)**2
                )

                # --- safety: boundaries, predicted exit using commanded model ---
                v_model = px_per_speed_per_sec * last_speed
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
                    cv2.rectangle(vis, (0,0), (w,h), COL_SAFETY, 3)

                elif not ball_found:
                    status = "BALL LOST"
                    should_move = False
                    speed_cmd = 0
                    idle_frames += 1

                else:
                    # --- normal control ---
                    # choose target: target center if found, else center point
                    if target_found:
                        idle_frames = 0
                        goal_x, goal_y = tx_pred, ty_pred
                        goal_label = chosen_target
                    else:
                        idle_frames += 1
                        goal_x, goal_y = center_x, center_y
                        goal_label = "CENTER"

                    # compute heading from state estimate to target
                    ang_screen = get_screen_angle((est_x, est_y), (goal_x, goal_y))
                    heading_cmd = normalize_angle(ang_screen - calibration_offset)

                    dx = goal_x - est_x
                    dy = goal_y - est_y
                    dist_to_goal = math.sqrt(dx*dx + dy*dy)

                    # Check if arrived
                    if dist_to_goal < TARGET_REACHED_PIX:
                        speed_cmd = 0
                        should_move = False
                        status = f"ARRIVED ({goal_label})"

                    else:
                        # --- BASE SPEED FROM DISTANCE ---
                        if dist_to_goal > 3 * TARGET_REACHED_PIX:
                            base_speed = MAX_SPEED
                        else:
                            frac = dist_to_goal / (3 * TARGET_REACHED_PIX)
                            base_speed = int(MIN_SPEED + (MAX_SPEED - MIN_SPEED) * frac)
                            base_speed = max(MIN_SPEED, min(MAX_SPEED, base_speed))

                        # --- VELOCITY-AWARE BRAKING ---
                        if dist_to_goal > 1e-3:
                            ux = dx / dist_to_goal  # unit vector toward target
                            uy = dy / dist_to_goal
                        else:
                            ux, uy = 0.0, 0.0

                        # Radial velocity toward target (px/s)
                        v_r = est_vx * ux + est_vy * uy

                        # Stopping distance estimate
                        d_stop = max(0.0, v_r) * BRAKE_TIME  # px

                        if d_stop >= dist_to_goal:
                            # Too fast to stop in time: creep in
                            speed_cmd = MIN_SPEED
                        else:
                            margin = dist_to_goal - d_stop
                            margin_ratio = margin / max(dist_to_goal, 1e-6)
                            # Don't drop below 30% of base_speed
                            speed_cmd = int(base_speed * max(0.3, margin_ratio))

                        speed_cmd = max(MIN_SPEED, min(MAX_SPEED, speed_cmd))
                        should_move = True
                        status = f"CHASING {goal_label}"

                    if target_found:
                        cv2.circle(vis, (tx, ty), 20, (255, 255, 255), 2)
                        cv2.putText(vis, goal_label, (tx+20, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # --- Runtime calibration refinement from movement ---
                last_est_for_calib = (est_x, est_y)  # save for next iteration
                if ball_found and heading_cmd is not None and last_speed > MIN_SPEED:
                    # compare actual movement direction with expected heading
                    if last_pos is not None:
                        dx_moved = est_x - last_pos[0]
                        dy_moved = est_y - last_pos[1]
                        distm = math.sqrt(dx_moved*dx_moved + dy_moved*dy_moved)
                        if distm > CALIB_MIN_MOVE_DIST:
                            actual_ang = get_screen_angle(last_pos, (est_x, est_y))
                            expected_ang = normalize_angle(last_command_heading + calibration_offset)
                            err_ang = angle_diff(actual_ang, expected_ang)
                            if abs(err_ang) < CALIB_MAX_ERR_TO_UPDATE:
                                calibration_offset = normalize_angle(
                                    calibration_offset + 0.07 * err_ang
                                )

                # --- Apply command & update latency-events ---
                prev_speed_for_latency = last_speed_for_latency
                last_speed_for_latency = speed_cmd

                if should_move and heading_cmd is not None:
                    droid.set_heading(int(heading_cmd))
                    droid.set_speed(int(speed_cmd))
                    last_command_heading = heading_cmd
                    last_speed = speed_cmd

                    # Start latency event if we just accelerated above threshold
                    if (prev_speed_for_latency <= LATENCY_SPEED_EVENT_THRESHOLD and
                        speed_cmd >= LATENCY_SPEED_EVENT_THRESHOLD and
                        pending_latency_event is None):
                        pending_latency_event = {
                            "time": now,
                            "x": est_x,
                            "y": est_y
                        }
                else:
                    droid.set_speed(0)
                    last_speed = 0

                # draw heading arrows from state estimate
                if ball_found and heading_cmd is not None:
                    # robot's "forward" (0° + offset)
                    pred_ang = normalize_angle(0 + calibration_offset)
                    rad_pred = math.radians(pred_ang)
                    fx = int(est_x + 40 * math.sin(rad_pred))
                    fy = int(est_y - 40 * math.cos(rad_pred))
                    cv2.arrowedLine(vis, (int(est_x), int(est_y)), (fx, fy), COL_PRED_HEAD, 2)
                    # command arrow
                    cmd_rad = math.radians(heading_cmd)
                    cx1 = int(est_x + 60 * math.sin(cmd_rad))
                    cy1 = int(est_y - 60 * math.cos(cmd_rad))
                    cv2.arrowedLine(vis, (int(est_x), int(est_y)), (cx1, cy1), COL_CMD_HEAD, 2)

                # HUD
                cv2.putText(vis, status, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)
                cv2.putText(vis, f"Calib offset: {int(calibration_offset)}°  "
                            f"Latency: {int(latency_est*1000)}ms  "
                            f"Scale: {px_per_speed_per_sec:.1f}px/s",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(vis, f"Dist goal: {int(dist_to_goal)}px",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow("Latency-Compensated Tracker", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                elapsed = time.time() - loop_start
                if elapsed < LOOP_DELAY:
                    time.sleep(max(0, LOOP_DELAY - elapsed))

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            droid.set_speed(0)
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
