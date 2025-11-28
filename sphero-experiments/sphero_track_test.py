import sys
sys.coinit_flags = 0 

import time
import math
import numpy as np
import cv2
import random
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ================= CONFIGURATION =================

CAM_INDEX = 1
LOOP_DELAY = 0.05

# --- TEST SETTINGS ---
CALIB_SPEED = 60
MOVE_DURATION = 0.8
STOP_DURATION = 1.0
ACCURACY_TOLERANCE = 20 # Degrees allowed error

# --- COLORS ---
COL_INTENT = (255, 0, 0)    # Blue (Where we want to go)
COL_ACTUAL = (0, 0, 255)    # Red (Where we actually went)
COL_BALL   = (0, 255, 0)    # Green
COL_TEXT   = (255, 255, 255)

# --- DETECTION ---
BRIGHTNESS_THRESHOLD = 180
MIN_RADIUS = 10
MAX_RADIUS = 60

# ================= MATH =================

def normalize_angle(a):
    return a % 360

def angle_diff(a, b):
    return (a - b + 180) % 360 - 180

def get_visual_angle(p1, p2):
    """ 
    Calculates angle from p1 to p2 on screen coordinates.
    Standard geometric angle: 0=Right, 90=Up (Negative Y), 180=Left, 270=Down
    """
    dx = p2[0] - p1[0]
    # Invert Y because screen Y increases downwards, but we want 'Up' to be positive angle
    dy = -(p2[1] - p1[1]) 
    deg = math.degrees(math.atan2(dy, dx))
    return normalize_angle(deg)

# ================= TRACKER =================

class Tracker:
    def __init__(self):
        # [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        
        self.prediction = np.zeros((4, 1), np.float32)
        self.frames_lost = 100
        self.velocity = 0

    def predict(self):
        self.prediction = self.kf.predict()
        self.frames_lost += 1
        return int(self.prediction[0, 0]), int(self.prediction[1, 0])

    def update(self, x, y):
        # Calculate velocity before update
        old_x, old_y = self.prediction[0, 0], self.prediction[1, 0]
        
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.prediction = self.kf.correct(meas)
        self.frames_lost = 0
        
        # Estimate speed
        dx = self.prediction[0, 0] - old_x
        dy = self.prediction[1, 0] - old_y
        self.velocity = math.sqrt(dx**2 + dy**2)

    def get_state(self):
        return int(self.prediction[0, 0]), int(self.prediction[1, 0])
    
    def is_stable(self):
        # Returns true if ball is found and barely moving
        return self.frames_lost < 5 and self.velocity < 2.0

# ================= VISION =================

def find_ball(frame, bright_mask):
    # Simple Blob + Hough (Fast and robust for test)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    max_score = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20: continue
        ((x, y), r) = cv2.minEnclosingCircle(c)
        
        if MIN_RADIUS < r < MAX_RADIUS:
            # Circularity check
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * math.pi * area / (perimeter**2)
            
            if circularity > 0.6:
                score = area
                if score > max_score:
                    max_score = score
                    best = (int(x), int(y), int(r))
    return best

# ================= MAIN =================

def main():
    print("Scanning for Sphero...")
    toy = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(scanner.find_toy)
        try: toy = future.result(timeout=10)
        except: pass
    if not toy: print("No Sphero found."); return

    tracker = Tracker()

    with SpheroEduAPI(toy) as droid:
        droid.set_main_led(Color(255, 255, 255))
        droid.set_speed(0)

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(CAM_INDEX)
        
        # --- STATE MACHINE ---
        STATE_SETTLE = 0
        STATE_CALIBRATE_MOVE = 1
        STATE_CALCULATE_OFFSET = 2
        STATE_TEST_PREDICT = 3
        STATE_TEST_VERIFY = 4
        
        current_state = STATE_SETTLE
        state_start_time = time.time()
        start_pos = None
        
        calibration_offset = 0.0
        is_calibrated = False
        
        # Visual Debug Vars
        test_target_angle = 0
        actual_angle = 0
        last_error = 0
        
        print("--- PHYSICS TEST MODE ---")
        
        try:
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret: break
                h, w = frame.shape[:2]
                
                # 1. VISION
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
                mask = cv2.dilate(mask, None, iterations=2)
                
                bx, by = tracker.predict()
                ball = find_ball(frame, mask)
                
                if ball:
                    # Check if valid update (no huge jumps)
                    dist = math.sqrt((ball[0]-bx)**2 + (ball[1]-by)**2)
                    if dist < 100 or tracker.frames_lost > 10:
                        tracker.update(ball[0], ball[1])
                    bx, by = ball[0], ball[1]

                # 2. STATE LOGIC
                status = "UNKNOWN"
                
                if tracker.frames_lost < 10:
                    
                    # --- STATE: SETTLE (Stop moving) ---
                    if current_state == STATE_SETTLE:
                        status = "STABILIZING..."
                        droid.set_speed(0)
                        if tracker.is_stable() and (time.time() - state_start_time > 1.0):
                            # If we aren't calibrated, go calibrate. Else test.
                            if not is_calibrated:
                                current_state = STATE_CALIBRATE_MOVE
                            else:
                                current_state = STATE_TEST_PREDICT
                            state_start_time = time.time()
                            start_pos = (bx, by)

                    # --- STATE: CALIBRATE (The Kick) ---
                    elif current_state == STATE_CALIBRATE_MOVE:
                        status = "CALIBRATING (KICK 0 deg)..."
                        # Command 0 degrees (Forward relative to robot)
                        droid.set_heading(0)
                        droid.set_speed(CALIB_SPEED)
                        
                        if time.time() - state_start_time > MOVE_DURATION:
                            droid.set_speed(0)
                            current_state = STATE_CALCULATE_OFFSET
                    
                    # --- STATE: MATH ---
                    elif current_state == STATE_CALCULATE_OFFSET:
                        status = "CALCULATING OFFSET..."
                        
                        # Wait a moment for ball to stop rolling
                        time.sleep(0.5) 
                        
                        dx = bx - start_pos[0]
                        # Invert Y because screen Y is down
                        dy = -(by - start_pos[1]) 
                        
                        dist_moved = math.sqrt(dx**2 + dy**2)
                        
                        if dist_moved > 20:
                            # Visual Angle
                            visual_angle = math.degrees(math.atan2(dy, dx))
                            visual_angle = normalize_angle(visual_angle)
                            
                            # Offset = Visual - Command(0)
                            calibration_offset = visual_angle
                            is_calibrated = True
                            print(f"CALIBRATED! Offset: {int(calibration_offset)}")
                            droid.set_main_led(Color(0, 255, 0)) # Green Flash
                            time.sleep(0.2)
                            droid.set_main_led(Color(255, 255, 255))
                            
                            current_state = STATE_SETTLE
                        else:
                            print("Movement too small, retrying...")
                            current_state = STATE_SETTLE
                        
                        state_start_time = time.time()

                    # --- STATE: TEST PREDICTION ---
                    elif current_state == STATE_TEST_PREDICT:
                        status = "GENERATING PREDICTION..."
                        # Pick random command angle
                        test_target_angle = random.randint(0, 359)
                        
                        # Calculate where we EXPECT to see it go visually
                        # Visual = Command + Offset
                        expected_visual = normalize_angle(test_target_angle + calibration_offset)
                        
                        current_state = STATE_TEST_VERIFY
                        state_start_time = time.time()
                        start_pos = (bx, by)
                        
                        # Execute Move
                        droid.set_heading(test_target_angle)
                        droid.set_speed(CALIB_SPEED)

                    # --- STATE: VERIFY ---
                    elif current_state == STATE_TEST_VERIFY:
                        status = "VERIFYING..."
                        
                        if time.time() - state_start_time > MOVE_DURATION:
                            droid.set_speed(0)
                            
                            # What happened?
                            actual_heading = get_visual_angle(start_pos, (bx, by))
                            expected_visual = normalize_angle(test_target_angle + calibration_offset)
                            
                            error = abs(angle_diff(actual_heading, expected_visual))
                            last_error = error
                            actual_angle = actual_heading # For display
                            
                            print(f"Cmd: {test_target_angle} | Exp Vis: {int(expected_visual)} | Act Vis: {int(actual_heading)} | Err: {int(error)}")
                            
                            if error < ACCURACY_TOLERANCE:
                                print(">>> PREDICTION ACCURATE <<<")
                                current_state = STATE_SETTLE
                            else:
                                print(">>> PREDICTION FAILED - RECALIBRATING <<<")
                                is_calibrated = False
                                droid.set_main_led(Color(255, 0, 0)) # Red Flash
                                time.sleep(0.2)
                                droid.set_main_led(Color(255, 255, 255))
                                current_state = STATE_SETTLE
                            
                            state_start_time = time.time()

                else:
                    status = "LOST BALL"
                    droid.set_speed(0)

                # 3. VISUALIZATION
                vis = frame.copy()
                
                # Draw Ball
                cv2.circle(vis, (bx, by), 15, COL_BALL, 2)
                
                # Draw History Line (Start -> Current)
                if start_pos:
                    cv2.line(vis, start_pos, (bx, by), COL_ACTUAL, 2)
                
                # Draw Predicted Vector (Only during verify)
                if current_state == STATE_TEST_VERIFY and start_pos:
                    expected_vis_ang = normalize_angle(test_target_angle + calibration_offset)
                    rad = math.radians(expected_vis_ang)
                    # Convert polar to cartesian (Screen Y is flipped)
                    # dx = cos(ang), dy = sin(ang) because standard trig 0=Right
                    # wait, get_visual_angle uses standard trig.
                    
                    ex = int(start_pos[0] + 100 * math.cos(rad))
                    ey = int(start_pos[1] - 100 * math.sin(rad)) # Y is down
                    
                    cv2.arrowedLine(vis, start_pos, (ex, ey), COL_INTENT, 2)
                    cv2.putText(vis, "PREDICTION", (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_INTENT, 1)

                # HUD
                cv2.putText(vis, f"State: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)
                cv2.putText(vis, f"Offset: {int(calibration_offset)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 1)
                
                if is_calibrated:
                    col = (0, 255, 0) if last_error < ACCURACY_TOLERANCE else (0, 0, 255)
                    cv2.putText(vis, f"Last Error: {int(last_error)} deg", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                cv2.imshow("Physics Test", vis)
                # cv2.imshow("Mask", mask)

                if cv2.waitKey(1) & 0xFF == ord('q'): break
                elapsed = time.time() - loop_start
                if elapsed < LOOP_DELAY: time.sleep(LOOP_DELAY - elapsed)

        except KeyboardInterrupt: pass
        finally:
            droid.set_speed(0)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()