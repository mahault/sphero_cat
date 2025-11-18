import sys
sys.coinit_flags = 0

import time
import math
import numpy as np
import cv2
import types
import copy

# --- STUBBING ---
fake_pandas = types.ModuleType("pandas")
sys.modules["pandas"] = fake_pandas
sys.modules["seaborn"] = types.ModuleType("seaborn")
sys.modules["matplotlib"] = types.ModuleType("matplotlib")
sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
from pymdp.agent import Agent
from ultralytics import YOLO

# ================= CONFIGURATION =================

CAM_INDEX = 1

# --- VISION SETTINGS ---
MIN_BRIGHTNESS = 150   
MIN_CIRCULARITY = 0.5  
MIN_ASPECT_RATIO = 0.75 
MIN_RADIUS_PX = 8
MAX_RADIUS_PX = 80

CAT_CONFIDENCE = 0.65
PIXELS_PER_CM = 2.5
SLIP_FACTOR = 0.5
LOOP_DELAY = 0.05

# --- GRID ---
N_STATES = 9
N_OBS = 9

# ================= MATH HELPERS =================

def angle_diff(a, b):
    """ Calculates smallest difference between two angles (-180 to 180) """
    diff = (a - b + 180) % 360 - 180
    return diff

def normalize_angle(a):
    return a % 360

# ================= KALMAN FILTER =================

class SpheroKalmanFilter:
    def __init__(self, start_x, start_y, dt):
        self.dt = dt
        self.x = np.array([start_x, start_y, 0, 0], dtype=float)
        self.P = np.eye(4) * 50.0 
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.1 
        self.R = np.eye(2) * 1.0 

    def predict(self, control_vx, control_vy):
        # Note: We rely less on control inputs now because orientation might be wrong
        # We trust the "Process Noise" (Q) to allow the filter to follow visual data
        self.x = self.F @ self.x
        # Fusion (Weakly pull towards control to help smoothness)
        self.x[2] = self.x[2]*0.9 + control_vx*0.1
        self.x[3] = self.x[3]*0.9 + control_vy*0.1
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas_x, meas_y):
        z = np.array([meas_x, meas_y])
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - (K @ self.H)) @ self.P

    def force_state(self, x, y):
        self.x[0] = x; self.x[1] = y
        self.x[2] = 0; self.x[3] = 0
        self.P = np.eye(4) * 10.0

    def get_state(self):
        return int(self.x[0]), int(self.x[1])

    def get_uncertainty(self):
        return int(np.sqrt((self.P[0,0] + self.P[1,1])/2) * 3)

# ================= POMDP AGENT =================

def obj_array(n): return np.empty(n, dtype=object)

def build_cat_tracker_agent():
    A = obj_array(1); A[0] = np.eye(N_OBS)
    B = obj_array(1); B[0] = np.eye(N_STATES).reshape(N_STATES, N_STATES, 1)
    C = obj_array(1); C[0] = np.zeros(N_OBS)
    D = obj_array(1); D[0] = np.ones(N_STATES) / N_STATES
    return Agent(A=A, B=B, C=C, D=D)

# ================= VISION =================

def find_sphero_strict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_circle = None
    max_score = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_RADIUS_PX**2: continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY: continue
        
        # Aspect Ratio
        if len(c) < 5: continue
        (center, (axis1, axis2), angle) = cv2.fitEllipse(c)
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)
        if major_axis == 0: continue
        if (minor_axis / major_axis) < MIN_ASPECT_RATIO: continue

        if area > max_score:
            max_score = area
            best_circle = (int(center[0]), int(center[1]), int(major_axis/2))

    if best_circle: return best_circle, True, thresh 
    return (0,0,0), False, thresh

def get_grid_center(cell_idx, w, h):
    row = cell_idx // 3; col = cell_idx % 3
    return (col * (w//3)) + (w // 6), (row * (h//3)) + (h // 6)

# ================= MAIN LOOP =================

def main():
    print("Scanning for Sphero...")
    toy = scanner.find_toy()
    if not toy: print("No Sphero found."); return

    print("Loading YOLO...")
    cat_model = YOLO("yolov8n.pt") 
    agent = build_cat_tracker_agent()

    with SpheroEduAPI(toy) as droid:
        droid.set_main_led(Color(0, 255, 0)) 
        try: droid.set_matrix_character("O", Color(0, 255, 0)) 
        except: pass 

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(CAM_INDEX)
        
        # --- SELF-CALIBRATION VARIABLES ---
        calibration_offset = 0.0 # The inferred rotation of the robot
        last_pos = None
        last_time = time.time()
        last_command_heading = None
        is_moving = False

        kf = SpheroKalmanFilter(640//2, 480//2, LOOP_DELAY)
        qs = copy.deepcopy(agent.D) 

        print("--- STARTED (AUTO-CALIBRATION MODE) ---")
        
        try:
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret: break
                h, w = frame.shape[:2]
                
                (sx, sy, sr), sphero_found, debug_mask = find_sphero_strict(frame)
                
                # --- UPDATE KALMAN ---
                # We ignore motor telemetry for prediction initially because we don't know direction
                kf.predict(0, 0) 
                
                if sphero_found:
                    if kf.get_uncertainty() > 100: kf.force_state(sx, sy)
                    else: kf.update(sx, sy)
                    cv2.circle(frame, (sx, sy), sr, (0, 255, 0), 2)

                est_sx, est_sy = kf.get_state()
                uncert = kf.get_uncertainty()

                # --- AUTO CALIBRATION LOGIC ---
                curr_time = time.time()
                dt = curr_time - last_time
                
                if sphero_found and last_pos is not None and is_moving and last_command_heading is not None:
                    # 1. Calculate Visual Vector
                    dx = est_sx - last_pos[0]
                    dy = last_pos[1] - est_sy # Flip Y (Screen Up is Positive)
                    
                    dist_moved = math.sqrt(dx**2 + dy**2)
                    
                    # Only calibrate if moved significantly (reject jitter)
                    if dist_moved > 5:
                        # 2. Calculate Visual Angle (0=Up, 90=Right)
                        vis_angle = math.degrees(math.atan2(dx, dy))
                        
                        # 3. Calculate Difference (Visual - Command)
                        # e.g. Vis=90 (Right), Cmd=0 (Up) -> Diff = +90
                        instant_offset = angle_diff(vis_angle, last_command_heading)
                        
                        # 4. Update Global Offset (Running Average for smoothness)
                        # Learn fast at first (0.2), then stabilize
                        alpha = 0.1 
                        calibration_offset += alpha * angle_diff(instant_offset, calibration_offset)
                        calibration_offset = normalize_angle(calibration_offset)

                last_pos = (est_sx, est_sy)
                last_time = curr_time

                # --- CAT DETECTION ---
                results = cat_model(frame, stream=True, verbose=False, conf=CAT_CONFIDENCE)
                observed_cat_cell = None
                cat_box = None
                for r in results:
                    for box in r.boxes:
                        if "cat" in cat_model.names[int(box.cls[0])].lower():
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx, cy = (x1+x2)//2, (y1+y2)//2
                            col = min(2, max(0, int(cx / (w/3))))
                            row = min(2, max(0, int(cy / (h/3))))
                            observed_cat_cell = row * 3 + col
                            cat_box = (x1, y1, x2, y2)
                            break 
                    if observed_cat_cell is not None: break

                if observed_cat_cell is not None:
                    qs = agent.infer_states([observed_cat_cell])
                
                likely_cat_cell = np.argmax(qs[0])
                confidence_cat = qs[0][likely_cat_cell]

                # --- NAVIGATION ---
                should_move = False
                target_angle = 0
                
                if confidence_cat > 0.4 and uncert < 150:
                    tx, ty = get_grid_center(likely_cat_cell, w, h)
                    dx = tx - est_sx
                    dy = est_sy - ty # Flip Y
                    
                    if math.sqrt(dx*dx + dy*dy) > 60:
                        # 1. Calculate Desired Visual Angle
                        desired_vis_angle = math.degrees(math.atan2(dx, dy))
                        
                        # 2. Subtract Offset to get Command Angle
                        # Cmd = Vis - Offset
                        target_angle = normalize_angle(desired_vis_angle - calibration_offset)
                        
                        should_move = True
                        cv2.line(frame, (est_sx, est_sy), (tx, ty), (255, 255, 0), 2)

                if should_move:
                    droid.set_heading(int(target_angle))
                    droid.set_speed(80) # Slower speed helps calibration
                    last_command_heading = target_angle
                    is_moving = True
                    status = f"CHASING (Offset: {int(calibration_offset)})"
                else:
                    droid.set_speed(0)
                    is_moving = False
                    status = "WAITING / LOST"

                # --- DRAWING ---
                cv2.drawMarker(frame, (est_sx, est_sy), (0,0,255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(frame, (est_sx, est_sy), uncert, (255,0,0), 1)
                if cat_box: cv2.rectangle(frame, (cat_box[0], cat_box[1]), (cat_box[2], cat_box[3]), (0,0,255), 2)
                
                # Grid
                for i in range(1,3):
                    cv2.line(frame, (i*(w//3),0), (i*(w//3),h), (50,50,50), 1)
                    cv2.line(frame, (0,i*(h//3)), (w, i*(h//3)), (50,50,50), 1)

                cv2.putText(frame, f"{status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                found_col = (0,255,0) if sphero_found else (0,0,255)
                cv2.putText(frame, f"Sphero: {sphero_found} | Calib: {int(calibration_offset)}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, found_col, 2)

                cv2.imshow("Auto-Calib Tracker", frame)
                cv2.imshow("Debug Mask", debug_mask)

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