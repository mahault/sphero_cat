# import sys
# sys.coinit_flags = 0

# import time
# import math
# import numpy as np
# import cv2
# import types
# import copy

# # --- STUBBING ---
# fake_pandas = types.ModuleType("pandas")
# sys.modules["pandas"] = fake_pandas
# sys.modules["seaborn"] = types.ModuleType("seaborn")
# sys.modules["matplotlib"] = types.ModuleType("matplotlib")
# sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")

# from spherov2 import scanner
# from spherov2.sphero_edu import SpheroEduAPI
# from spherov2.types import Color
# from pymdp.agent import Agent
# from ultralytics import YOLO

# # ================= CONFIGURATION =================

# CAM_INDEX = 1

# # --- VISION SETTINGS ---
# MIN_BRIGHTNESS = 150   
# MIN_CIRCULARITY = 0.5  
# MIN_ASPECT_RATIO = 0.75 
# MIN_RADIUS_PX = 8
# MAX_RADIUS_PX = 80

# CAT_CONFIDENCE = 0.65
# PIXELS_PER_CM = 2.5
# SLIP_FACTOR = 0.5
# LOOP_DELAY = 0.05

# # --- GRID ---
# N_STATES = 9
# N_OBS = 9

# # ================= MATH HELPERS =================

# def angle_diff(a, b):
#     """ Calculates smallest difference between two angles (-180 to 180) """
#     diff = (a - b + 180) % 360 - 180
#     return diff

# def normalize_angle(a):
#     return a % 360

# # ================= KALMAN FILTER =================

# class SpheroKalmanFilter:
#     def __init__(self, start_x, start_y, dt):
#         self.dt = dt
#         self.x = np.array([start_x, start_y, 0, 0], dtype=float)
#         self.P = np.eye(4) * 50.0 
#         self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
#         self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
#         self.Q = np.eye(4) * 0.1 
#         self.R = np.eye(2) * 1.0 

#     def predict(self, control_vx, control_vy):
#         # Note: We rely less on control inputs now because orientation might be wrong
#         # We trust the "Process Noise" (Q) to allow the filter to follow visual data
#         self.x = self.F @ self.x
#         # Fusion (Weakly pull towards control to help smoothness)
#         self.x[2] = self.x[2]*0.9 + control_vx*0.1
#         self.x[3] = self.x[3]*0.9 + control_vy*0.1
#         self.P = self.F @ self.P @ self.F.T + self.Q

#     def update(self, meas_x, meas_y):
#         z = np.array([meas_x, meas_y])
#         y = z - (self.H @ self.x)
#         S = self.H @ self.P @ self.H.T + self.R
#         K = self.P @ self.H.T @ np.linalg.inv(S)
#         self.x = self.x + (K @ y)
#         self.P = (np.eye(4) - (K @ self.H)) @ self.P

#     def force_state(self, x, y):
#         self.x[0] = x; self.x[1] = y
#         self.x[2] = 0; self.x[3] = 0
#         self.P = np.eye(4) * 10.0

#     def get_state(self):
#         return int(self.x[0]), int(self.x[1])

#     def get_uncertainty(self):
#         return int(np.sqrt((self.P[0,0] + self.P[1,1])/2) * 3)

# # ================= POMDP AGENT =================

# def obj_array(n): return np.empty(n, dtype=object)

# def build_cat_tracker_agent():
#     A = obj_array(1); A[0] = np.eye(N_OBS)
#     B = obj_array(1); B[0] = np.eye(N_STATES).reshape(N_STATES, N_STATES, 1)
#     C = obj_array(1); C[0] = np.zeros(N_OBS)
#     D = obj_array(1); D[0] = np.ones(N_STATES) / N_STATES
#     return Agent(A=A, B=B, C=C, D=D)

# # ================= VISION =================

# def find_sphero_strict(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((5,5), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     best_circle = None
#     max_score = 0
    
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < MIN_RADIUS_PX**2: continue
#         perimeter = cv2.arcLength(c, True)
#         if perimeter == 0: continue
#         circularity = (4 * math.pi * area) / (perimeter ** 2)
#         if circularity < MIN_CIRCULARITY: continue
        
#         # Aspect Ratio
#         if len(c) < 5: continue
#         (center, (axis1, axis2), angle) = cv2.fitEllipse(c)
#         major_axis = max(axis1, axis2)
#         minor_axis = min(axis1, axis2)
#         if major_axis == 0: continue
#         if (minor_axis / major_axis) < MIN_ASPECT_RATIO: continue

#         if area > max_score:
#             max_score = area
#             best_circle = (int(center[0]), int(center[1]), int(major_axis/2))

#     if best_circle: return best_circle, True, thresh 
#     return (0,0,0), False, thresh

# def get_grid_center(cell_idx, w, h):
#     row = cell_idx // 3; col = cell_idx % 3
#     return (col * (w//3)) + (w // 6), (row * (h//3)) + (h // 6)


# def find_sphero_robust(frame):
#     """
#     Combines Color (HSV), Brightness, and Blob Shape to find the Sphero.
#     Assumes Sphero is set to RED Color.
#     """
#     # 1. Pre-processing
#     blurred = cv2.GaussianBlur(frame, (5, 5), 0)
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

#     # 2. COLOR MASK (Looking for RED)
#     # Red in HSV wraps around 0/180. We need two ranges.
#     # Range 1: 0-10 (Red)
#     lower_red1 = np.array([0, 120, 100]) 
#     upper_red1 = np.array([10, 255, 255])
#     # Range 2: 170-180 (Red)
#     lower_red2 = np.array([170, 120, 100]) 
#     upper_red2 = np.array([180, 255, 255])
    
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     color_mask = cv2.bitwise_or(mask1, mask2)

#     # 3. BRIGHTNESS MASK (Relaxed threshold)
#     # We lower the brightness req because the Color requirement is strict
#     _, bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

#     # 4. COMBINE MASKS (Intersection)
#     # Must be RED AND BRIGHT(ish)
#     combined_mask = cv2.bitwise_and(color_mask, bright_mask)

#     # Clean up noise (Dilate to merge the broken LED parts into one blob)
#     kernel = np.ones((5, 5), np.uint8)
#     combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
#     combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

#     # 5. Find Contours
#     contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     best_circle = None
#     max_score = 0

#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < MIN_RADIUS_PX**2: continue # Too small
        
#         # Geometry calculations
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
        
#         # --- SCORING SYSTEM ---
        
#         # 1. Circularity Score (How round is it?)
#         perimeter = cv2.arcLength(c, True)
#         if perimeter == 0: continue
#         circularity = (4 * math.pi * area) / (perimeter ** 2)
        
#         # 2. Size Score (Is it a reasonable size?)
#         # Penalize if it's massive (glare on wall) or tiny
#         size_score = 1.0
#         if radius > MAX_RADIUS_PX: size_score = 0.2
        
#         # 3. Color Density Score
#         # Check the original color mask to ensure this blob is actually red
#         # (prevents white lights from passing if they happen to overlap slightly)
#         mask_roi = np.zeros_like(gray)
#         cv2.drawContours(mask_roi, [c], -1, 255, -1)
#         # Calculate mean brightness of this contour in the Color Mask
#         mean_val = cv2.mean(color_mask, mask=mask_roi)[0]
#         color_confidence = mean_val / 255.0

#         # FINAL SCORE
#         # We weigh Color Confidence heavily.
#         total_score = (circularity * 0.3) + (color_confidence * 0.7) * size_score

#         if total_score > max_score and total_score > 0.4:
#             max_score = total_score
#             best_circle = (int(x), int(y), int(radius))

#     found = best_circle is not None
    
#     # Return a tuple consistent with your loop
#     if not found:
#         return (0, 0, 0), False, combined_mask
    
#     return best_circle, True, combined_mask

# # ================= MAIN LOOP =================

# def main():
#     print("Scanning for Sphero...")
#     toy = scanner.find_toy()
#     if not toy: print("No Sphero found."); return

#     print("Loading YOLO...")
#     cat_model = YOLO("yolov8n.pt") 
#     agent = build_cat_tracker_agent()

#     with SpheroEduAPI(toy) as droid:
#         droid.set_main_led(Color(255, 0, 0)) 
#         try: droid.set_matrix_character("O", Color(255, 0, 0)) 
#         except: pass 

#         cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
#         if not cap.isOpened(): cap = cv2.VideoCapture(CAM_INDEX)
        
#         # --- SELF-CALIBRATION VARIABLES ---
#         calibration_offset = 0.0 # The inferred rotation of the robot
#         last_pos = None
#         last_time = time.time()
#         last_command_heading = None
#         is_moving = False

#         kf = SpheroKalmanFilter(640//2, 480//2, LOOP_DELAY)
#         qs = copy.deepcopy(agent.D) 

#         print("--- STARTED (AUTO-CALIBRATION MODE) ---")
        
#         try:
#             while True:
#                 loop_start = time.time()
#                 ret, frame = cap.read()
#                 if not ret: break
#                 h, w = frame.shape[:2]
                
#                 (sx, sy, sr), sphero_found, debug_mask = find_sphero_robust(frame)
                
#                 # --- UPDATE KALMAN ---
#                 # We ignore motor telemetry for prediction initially because we don't know direction
#                 kf.predict(0, 0) 
                
#                 if sphero_found:
#                     if kf.get_uncertainty() > 100: kf.force_state(sx, sy)
#                     else: kf.update(sx, sy)
#                     cv2.circle(frame, (sx, sy), sr, (0, 255, 0), 2)

#                 est_sx, est_sy = kf.get_state()
#                 uncert = kf.get_uncertainty()

#                 # --- AUTO CALIBRATION LOGIC ---
#                 curr_time = time.time()
#                 dt = curr_time - last_time
                
#                 if sphero_found and last_pos is not None and is_moving and last_command_heading is not None:
#                     # 1. Calculate Visual Vector
#                     dx = est_sx - last_pos[0]
#                     dy = last_pos[1] - est_sy # Flip Y (Screen Up is Positive)
                    
#                     dist_moved = math.sqrt(dx**2 + dy**2)
                    
#                     # Only calibrate if moved significantly (reject jitter)
#                     if dist_moved > 5:
#                         # 2. Calculate Visual Angle (0=Up, 90=Right)
#                         vis_angle = math.degrees(math.atan2(dx, dy))
                        
#                         # 3. Calculate Difference (Visual - Command)
#                         # e.g. Vis=90 (Right), Cmd=0 (Up) -> Diff = +90
#                         instant_offset = angle_diff(vis_angle, last_command_heading)
                        
#                         # 4. Update Global Offset (Running Average for smoothness)
#                         # Learn fast at first (0.2), then stabilize
#                         alpha = 0.1 
#                         calibration_offset += alpha * angle_diff(instant_offset, calibration_offset)
#                         calibration_offset = normalize_angle(calibration_offset)

#                 last_pos = (est_sx, est_sy)
#                 last_time = curr_time

#                 # --- CAT DETECTION ---
#                 results = cat_model(frame, stream=True, verbose=False, conf=CAT_CONFIDENCE)
#                 observed_cat_cell = None
#                 cat_box = None
#                 for r in results:
#                     for box in r.boxes:
#                         if "cat" in cat_model.names[int(box.cls[0])].lower():
#                             x1, y1, x2, y2 = map(int, box.xyxy[0])
#                             cx, cy = (x1+x2)//2, (y1+y2)//2
#                             col = min(2, max(0, int(cx / (w/3))))
#                             row = min(2, max(0, int(cy / (h/3))))
#                             observed_cat_cell = row * 3 + col
#                             cat_box = (x1, y1, x2, y2)
#                             break 
#                     if observed_cat_cell is not None: break

#                 if observed_cat_cell is not None:
#                     qs = agent.infer_states([observed_cat_cell])
                
#                 likely_cat_cell = np.argmax(qs[0])
#                 confidence_cat = qs[0][likely_cat_cell]

#                 # --- NAVIGATION ---
#                 should_move = False
#                 target_angle = 0
                
#                 if confidence_cat > 0.4 and uncert < 150:
#                     tx, ty = get_grid_center(likely_cat_cell, w, h)
#                     dx = tx - est_sx
#                     dy = est_sy - ty # Flip Y
                    
#                     if math.sqrt(dx*dx + dy*dy) > 60:
#                         # 1. Calculate Desired Visual Angle
#                         desired_vis_angle = math.degrees(math.atan2(dx, dy))
                        
#                         # 2. Subtract Offset to get Command Angle
#                         # Cmd = Vis - Offset
#                         target_angle = normalize_angle(desired_vis_angle - calibration_offset)
                        
#                         should_move = True
#                         cv2.line(frame, (est_sx, est_sy), (tx, ty), (255, 255, 0), 2)

#                 if should_move:
#                     droid.set_heading(int(target_angle))
#                     droid.set_speed(80) # Slower speed helps calibration
#                     last_command_heading = target_angle
#                     is_moving = True
#                     status = f"CHASING (Offset: {int(calibration_offset)})"
#                 else:
#                     droid.set_speed(0)
#                     is_moving = False
#                     status = "WAITING / LOST"

#                 # --- DRAWING ---
#                 cv2.drawMarker(frame, (est_sx, est_sy), (0,0,255), cv2.MARKER_CROSS, 20, 2)
#                 cv2.circle(frame, (est_sx, est_sy), uncert, (255,0,0), 1)
#                 if cat_box: cv2.rectangle(frame, (cat_box[0], cat_box[1]), (cat_box[2], cat_box[3]), (0,0,255), 2)
                
#                 # Grid
#                 for i in range(1,3):
#                     cv2.line(frame, (i*(w//3),0), (i*(w//3),h), (50,50,50), 1)
#                     cv2.line(frame, (0,i*(h//3)), (w, i*(h//3)), (50,50,50), 1)

#                 cv2.putText(frame, f"{status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
#                 found_col = (0,255,0) if sphero_found else (0,0,255)
#                 cv2.putText(frame, f"Sphero: {sphero_found} | Calib: {int(calibration_offset)}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, found_col, 2)

#                 cv2.imshow("Auto-Calib Tracker", frame)
#                 cv2.imshow("Debug Mask", debug_mask)

#                 if cv2.waitKey(1) & 0xFF == ord('q'): break
#                 elapsed = time.time() - loop_start
#                 if elapsed < LOOP_DELAY: time.sleep(LOOP_DELAY - elapsed)

#         except KeyboardInterrupt: pass
#         finally:
#             droid.set_speed(0)
#             cap.release()
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# import sys
# sys.coinit_flags = 0 # Force MTA for Bluetooth

# import time
# import math
# import numpy as np
# import cv2
# from concurrent.futures import ThreadPoolExecutor
# import types
# from ultralytics import YOLO

# # --- STUBBING ---
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
# LOOP_DELAY = 0.05

# # --- THRESHOLDS ---
# CAT_CONFIDENCE = 0.50
# MIN_BALL_BRIGHTNESS = 180  # 0-255: How "glowing" must the ball be?
# MIN_BALL_RADIUS = 10
# MAX_BALL_RADIUS = 100

# # --- COLORS (BGR) ---
# COL_BALL = (0, 255, 0)     # Green
# COL_CAT = (0, 0, 255)      # Red
# COL_OBSTACLE = (255, 0, 0) # Blue
# COL_FLOOR = (100, 100, 100)# Gray

# # --- YOLO CLASS IDs ---
# CLS_PERSON = 0
# CLS_CAT = 15
# CLS_DOG = 16
# CLS_CHAIR = 56
# CLS_COUCH = 57
# CLS_BED = 59
# OBSTACLE_CLASSES = [CLS_PERSON, CLS_CHAIR, CLS_COUCH, CLS_BED]

# # ================= VISION PIPELINE =================

# def get_floor_mask(frame, obstacle_mask):
#     """
#     Identifies the floor using a Flood Fill algorithm.
#     1. Starts at the bottom-center of the screen (assumed to be floor).
#     2. Expands outward finding similarly colored pixels.
#     3. Stops at walls (dark/different color) or Obstacles (YOLO mask).
#     """
#     h, w = frame.shape[:2]
    
#     # 1. Downsample for speed
#     small = cv2.resize(frame, (w//4, h//4))
#     h_s, w_s = small.shape[:2]
    
#     # 2. Create a mask for floodFill (needs to be +2 pixels larger)
#     # We pre-fill it with the YOLO obstacles so floodFill stops there.
#     fill_mask = np.zeros((h_s+2, w_s+2), np.uint8)
    
#     # Resize obstacle mask to match downsampled size
#     obs_small = cv2.resize(obstacle_mask, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
#     fill_mask[1:-1, 1:-1] = obs_small # Embed obstacles
    
#     # 3. Flood Fill Seed Point (Bottom Center)
#     seed_pt = (w_s//2, h_s - 5)
    
#     # Check if seed is valid (not already an obstacle)
#     if fill_mask[seed_pt[1]+1, seed_pt[0]+1] == 0:
#         # Flood fill similar colors (tolerance of 30)
#         cv2.floodFill(small, fill_mask, seed_pt, (255, 255, 255), (30,)*3, (30,)*3, flags=8 | (255 << 8))
    
#     # 4. Extract the "Filled" area (which is marked with 255 in the mask)
#     # The mask is +2 larger, crop it.
#     floor_mask_small = fill_mask[1:-1, 1:-1]
    
#     # 5. Resize back up
#     floor_mask = cv2.resize(floor_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    
#     # Ensure it's binary
#     _, floor_mask = cv2.threshold(floor_mask, 128, 255, cv2.THRESH_BINARY)
    
#     return floor_mask

# def detect_glowing_circle(frame):
#     """
#     Finds the Sphero by looking for a Geometric Circle that is also Bright.
#     This ignores complex shapes (hands) and focuses on the glowing LED.
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # 1. Hough Circle Transform (Geometry)
#     circles = cv2.HoughCircles(
#         gray, 
#         cv2.HOUGH_GRADIENT, 
#         dp=1.5,           # Inverse ratio of resolution
#         minDist=50,       # Min dist between circles
#         param1=150,       # Canny edge threshold (High to avoid noise)
#         param2=30,        # Center detection threshold (Lower = more sensitive)
#         minRadius=MIN_BALL_RADIUS, 
#         maxRadius=MAX_BALL_RADIUS
#     )
    
#     best_ball = None
    
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             cx, cy, r = i[0], i[1], i[2]
            
#             # 2. Brightness Check (Luminosity)
#             # Create a mask for this specific circle
#             mask = np.zeros_like(gray)
#             cv2.circle(mask, (cx, cy), r, 255, -1)
            
#             # Calculate average brightness inside this circle
#             mean_val = cv2.mean(gray, mask=mask)[0]
            
#             if mean_val > MIN_BALL_BRIGHTNESS:
#                 # Found it!
#                 best_ball = (cx, cy, r)
#                 break # Return the first/strongest one
                
#     return best_ball

# def process_vision(frame, model):
#     h, w = frame.shape[:2]
    
#     # --- 1. YOLO INFERENCE (Obstacles & Cat) ---
#     results = model(frame, verbose=False, conf=0.25)
#     result = results[0]
    
#     # Masks
#     obstacle_mask = np.zeros((h, w), dtype=np.uint8)
#     cat_mask = np.zeros((h, w), dtype=np.uint8)
#     cat_found = False
#     cat_center = None
    
#     if result.masks is not None:
#         masks = result.masks.data.cpu().numpy()
#         boxes = result.boxes
        
#         for i, box in enumerate(boxes):
#             cls_id = int(box.cls[0])
            
#             # Resize YOLO mask to frame size
#             m = cv2.resize(masks[i], (w, h))
#             binary_m = (m > 0.5).astype(np.uint8)
            
#             if cls_id in OBSTACLE_CLASSES:
#                 obstacle_mask = cv2.bitwise_or(obstacle_mask, binary_m)
#             elif cls_id == CLS_CAT:
#                 if box.conf[0] > CAT_CONFIDENCE:
#                     cat_mask = cv2.bitwise_or(cat_mask, binary_m)
#                     cat_found = True
#                     # Calculate center
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cat_center = ((x1+x2)//2, (y1+y2)//2)

#     # --- 2. FLOOR DETECTION ---
#     # Use flood fill, avoiding the known obstacles
#     floor_mask = get_floor_mask(frame, obstacle_mask)
    
#     # --- 3. BALL DETECTION (Geometric + Brightness) ---
#     ball_data = detect_glowing_circle(frame)
    
#     # --- 4. BUILD VISUALIZATION ---
#     # Create a colored overlay
#     overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Paint Floor (Gray)
#     overlay[floor_mask == 255] = COL_FLOOR
#     # Paint Obstacles (Blue) - Overwrite floor
#     overlay[obstacle_mask == 1] = COL_OBSTACLE
#     # Paint Cat (Red)
#     overlay[cat_mask == 1] = COL_CAT
    
#     # Paint Ball (Green) - Manually draw the circle on the overlay
#     if ball_data:
#         cx, cy, r = ball_data
#         cv2.circle(overlay, (cx, cy), r, COL_BALL, -1) # Filled green circle
        
#     return overlay, floor_mask, cat_found, cat_center, ball_data

# # ================= MAIN LOOP =================

# def main():
#     print("Scanning for Sphero...")
    
#     # Threaded scan
#     toy = None
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(scanner.find_toy)
#         try: toy = future.result(timeout=10)
#         except: pass

#     if not toy: print("No Sphero found."); return

#     print("Loading YOLOv8 Segmentation...")
#     model = YOLO("yolov8s-seg.pt") 
    
#     with SpheroEduAPI(toy) as droid:
#         # Set LED to Bright White to help the "Glowing" detector
#         droid.set_main_led(Color(255, 255, 255)) 
#         droid.set_speed(0)

#         cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
#         if not cap.isOpened(): cap = cv2.VideoCapture(CAM_INDEX)
        
#         # State
#         calibration_offset = 0.0
#         last_pos = None
#         last_heading = 0
        
#         print("--- SYSTEM READY ---")
        
#         try:
#             while True:
#                 loop_start = time.time()
#                 ret, frame = cap.read()
#                 if not ret: break
                
#                 # 1. VISION PROCESSING
#                 overlay, floor_mask, cat_found, cat_center, ball_data = process_vision(frame, model)
                
#                 # 2. BLEND IMAGES (0.6 Original + 0.4 Overlay)
#                 vis_frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                
#                 # 3. LOGIC
#                 status = "IDLE"
#                 should_move = False
#                 target_angle = 0
                
#                 if ball_data:
#                     bx, by, _ = ball_data
                    
#                     # AUTO-CALIBRATION (Only when moving)
#                     if last_pos:
#                         dx, dy = bx - last_pos[0], last_pos[1] - by # Flip Y
#                         if math.sqrt(dx**2 + dy**2) > 5:
#                             vis_angle = math.degrees(math.atan2(dx, dy))
#                             diff = (vis_angle - last_heading + 180) % 360 - 180
#                             calibration_offset += 0.1 * diff # Soft update
                            
#                     last_pos = (bx, by)
                    
#                     # NAVIGATION LOGIC
#                     if cat_found and cat_center:
#                         cx, cy = cat_center
                        
#                         # Check if Ball is on the Floor
#                         # (Access floor mask at ball coordinates)
#                         # Safe bounds check
#                         if 0 <= by < floor_mask.shape[0] and 0 <= bx < floor_mask.shape[1]:
#                             is_on_floor = floor_mask[by, bx] == 255
#                         else:
#                             is_on_floor = False
                        
#                         if is_on_floor:
#                             # Calculate Vector to Cat
#                             vec_x = cx - bx
#                             vec_y = by - cy # Screen Y is inverted
                            
#                             dist = math.sqrt(vec_x**2 + vec_y**2)
                            
#                             if dist > 80: # Don't ram the cat
#                                 desired_angle = math.degrees(math.atan2(vec_x, vec_y))
#                                 target_angle = (desired_angle - calibration_offset) % 360
#                                 should_move = True
#                                 status = "CHASING CAT"
#                                 cv2.line(vis_frame, (bx, by), (cx, cy), (0, 255, 0), 2)
#                         else:
#                             status = "BALL NOT ON FLOOR"
#                     else:
#                         status = "NO CAT DETECTED"
#                 else:
#                     status = "BALL LOST"

#                 # 4. EXECUTE
#                 if should_move:
#                     droid.set_heading(int(target_angle))
#                     droid.set_speed(60)
#                     last_heading = target_angle
#                 else:
#                     droid.set_speed(0)

#                 # 5. DRAW HUD
#                 cv2.putText(vis_frame, f"STATUS: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 if ball_data:
#                     cv2.circle(vis_frame, (ball_data[0], ball_data[1]), 5, (255,255,255), -1)

#                 cv2.imshow("Semantic Eye", vis_frame)
                
#                 if cv2.waitKey(1) & 0xFF == ord('q'): break
#                 elapsed = time.time() - loop_start
#                 if elapsed < LOOP_DELAY: time.sleep(LOOP_DELAY - elapsed)

#         except KeyboardInterrupt: pass
#         finally:
#             droid.set_speed(0)
#             cap.release()
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import sys
sys.coinit_flags = 0 

import time
import math
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ================= CONFIGURATION =================

CAM_INDEX = 1
LOOP_DELAY = 0.05

# --- NAVIGATION & SAFETY ---
MAX_SPEED = 45              # Reduced max speed for better control
MIN_SPEED = 15              # Lower minimum for gentle approach
TOUCH_DISTANCE = 80         # Increased stopping distance (accounts for momentum)
BRAKE_DISTANCE = 150        # Start aggressive braking
BORDER_MARGIN = 50          # Pixels from edge to trigger safety stop

# --- CALIBRATION ---
MIN_MOVE_DIST = 5           # Minimum pixels moved to trigger calibration update
LEARNING_RATE = 0.1         # How fast we correct the angle (0.1 = smooth, 0.5 = twitchy)

# --- COLORS ---
COL_CAT = (0, 0, 255)       
COL_WINNER = (0, 255, 0)    
COL_PREDICTION = (255, 0, 0)
COL_TEXT = (255, 255, 255)
COL_SAFETY = (0, 0, 255)

# --- DETECTION ---
MIN_CONFIDENCE = 0.25
BRIGHTNESS_THRESHOLD = 190
HOUGH_PARAM2 = 25
MIN_RADIUS = 8
MAX_RADIUS = 60
SEARCH_WINDOW = 100
PRIOR_WEIGHT = 0.8
RESET_THRESHOLD = 160

# --- TESTING MODE ---
# ┌─────────────────────────────────────────────────────────────┐
# │ SWITCH BETWEEN TESTING AND PRODUCTION MODE:                │
# │ TESTING_MODE = True  → Hunts PERSON (orange overlay)       │
# │ TESTING_MODE = False → Hunts CAT (red overlay)             │
# └─────────────────────────────────────────────────────────────┘
TESTING_MODE = True  # ← CHANGE THIS TO SWITCH MODES
TARGET_CLASSES = [0] if TESTING_MODE else [15, 16]  # 0=person, 15=cat, 16=dog
TARGET_NAME = "PERSON" if TESTING_MODE else "CAT"

# ================= MATH HELPERS =================

def normalize_angle(a):
    return a % 360

def angle_diff(a, b):
    """ Returns minimal difference between two angles (-180 to 180) """
    return (a - b + 180) % 360 - 180

def get_screen_angle(p1, p2):
    """ 
    Calculates angle from p1 to p2 on screen.
    0=Up, 90=Right
    """
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1] # Invert Y (Screen Y is down)
    deg = math.degrees(math.atan2(dx, dy))
    return normalize_angle(deg)

# ================= STABILIZATION =================

class Tracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.prediction = np.zeros((4, 1), np.float32)
        self.frames_lost = 100
        self.radius = 20

    def predict(self):
        self.prediction = self.kf.predict()
        self.frames_lost += 1
        return int(self.prediction[0, 0]), int(self.prediction[1, 0])

    def update(self, x, y, r):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.prediction = self.kf.correct(meas)
        self.radius = r
        self.frames_lost = 0

    def reset(self, x, y):
        self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.prediction = self.kf.statePost
        self.frames_lost = 0
        
    def get_state(self):
        return int(self.prediction[0, 0]), int(self.prediction[1, 0])

# ================= VISION SYSTEM =================

def find_candidates(frame, model, bright_mask):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge margin to reject wall corners
    EDGE_MARGIN = 30
    OVERLAP_DIST = 40  # Distance to consider two detections as "same object"

    hough_detections = []
    yolo_detections = []
    blob_detections = []

    # 1. HOUGH CIRCLES (Geometric detection)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=60,
                               param1=150, param2=30,
                               minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = int(i[0]), int(i[1]), int(i[2])

            # Skip if near edges (likely wall corners)
            if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
                cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
                continue

            # Check brightness
            mask_roi = np.zeros_like(gray)
            cv2.circle(mask_roi, (cx, cy), r, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask_roi)[0]

            # Must be bright
            if mean_brightness < BRIGHTNESS_THRESHOLD:
                continue

            hough_detections.append({'x': cx, 'y': cy, 'r': r, 'brightness': mean_brightness})

    # 2. YOLO BALL DETECTION (Semantic detection)
    results = model(frame, verbose=False, conf=MIN_CONFIDENCE)
    if results[0].boxes:
        for box in results[0].boxes:
            if int(box.cls[0]) == 32:  # Ball class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                r = max((x2-x1)//2, (y2-y1)//2)
                conf = float(box.conf[0])

                # Skip if near edges
                if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
                    cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
                    continue

                yolo_detections.append({'x': cx, 'y': cy, 'r': r, 'conf': conf})

    # 3. BRIGHT BLOBS (Brightness-based detection)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50: continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * math.pi * area / (perimeter**2)

        if circularity > 0.6:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cx, cy = int(x), int(y)

            # Skip if near edges
            if (cx < EDGE_MARGIN or cx > w - EDGE_MARGIN or
                cy < EDGE_MARGIN or cy > h - EDGE_MARGIN):
                continue

            if MIN_RADIUS < r < MAX_RADIUS:
                blob_detections.append({'x': cx, 'y': cy, 'r': int(r), 'circ': circularity})

    # 4. CROSS-REFERENCE AND SCORE
    # The best candidate is one confirmed by multiple methods
    candidates = []

    def distance(p1, p2):
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

    # Check each Hough detection
    for h in hough_detections:
        score = 100  # Base score for Hough
        src = 'Hough'

        # Brightness bonus
        score += (h['brightness'] - BRIGHTNESS_THRESHOLD) / 2

        # Check if YOLO confirms this
        yolo_match = None
        for y in yolo_detections:
            if distance(h, y) < OVERLAP_DIST:
                yolo_match = y
                break

        # Check if Blob confirms this
        blob_match = None
        for b in blob_detections:
            if distance(h, b) < OVERLAP_DIST:
                blob_match = b
                break

        # Multi-method confirmation bonuses
        if yolo_match and blob_match:
            score += 200  # Triple confirmation!
            src = 'Hough+YOLO+Blob'
        elif yolo_match:
            score += 150  # Hough + YOLO
            src = 'Hough+YOLO'
        elif blob_match:
            score += 50   # Hough + Blob
            src = 'Hough+Blob'

        candidates.append({'x': h['x'], 'y': h['y'], 'r': h['r'], 'score': score, 'src': src})

    # Also check YOLO detections that weren't matched to Hough
    for y in yolo_detections:
        # Skip if already matched
        if any(distance(h, y) < OVERLAP_DIST for h in hough_detections):
            continue

        score = 120  # YOLO alone is decent
        src = 'YOLO'

        # Check blob match
        blob_match = None
        for b in blob_detections:
            if distance(y, b) < OVERLAP_DIST:
                blob_match = b
                break

        if blob_match:
            score += 60
            src = 'YOLO+Blob'

        candidates.append({'x': y['x'], 'y': y['y'], 'r': y['r'], 'score': score, 'src': src})

    return candidates

def select_best_ball(candidates, tracker):
    if not candidates: return None
    pred_x, pred_y = tracker.get_state()
    best_candidate = None
    highest_adjusted_score = -9999
    
    for c in candidates:
        dist = math.sqrt((c['x'] - pred_x)**2 + (c['y'] - pred_y)**2)
        if tracker.frames_lost < 30:
            if dist < SEARCH_WINDOW:
                spatial_bonus = (SEARCH_WINDOW - dist) * 0.2
                c['final_score'] = c['score'] + spatial_bonus
            else:
                c['final_score'] = c['score'] - (dist * PRIOR_WEIGHT)
                if c['score'] > RESET_THRESHOLD: c['final_score'] = c['score'] 
        else:
            c['final_score'] = c['score']
        if c['final_score'] > highest_adjusted_score:
            highest_adjusted_score = c['final_score']
            best_candidate = c
    return best_candidate

def process_vision(frame, model, ball_tracker, cat_tracker):
    h, w = frame.shape[:2]
    bright_mask = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    bright_mask = cv2.dilate(bright_mask, None, iterations=2)

    # Ball
    candidates = find_candidates(frame, model, bright_mask)
    best_ball = select_best_ball(candidates, ball_tracker)
    bx, by = ball_tracker.predict()

    if best_ball:
        dist = math.sqrt((best_ball['x'] - bx)**2 + (best_ball['y'] - by)**2)
        if dist > SEARCH_WINDOW and best_ball['score'] > RESET_THRESHOLD:
            ball_tracker.reset(best_ball['x'], best_ball['y'])
        else:
            ball_tracker.update(best_ball['x'], best_ball['y'], best_ball['r'])

    # Target Detection (Person for testing, Cat for production)
    # TESTING_MODE controls which target to track
    results = model(frame, verbose=False, conf=MIN_CONFIDENCE)
    cat_mask = np.zeros((h, w), dtype=np.uint8)  # Still named cat_mask for compatibility
    if results[0].masks:
        for i, box in enumerate(results[0].boxes):
            # Use TARGET_CLASSES from config
            if int(box.cls[0]) in TARGET_CLASSES:
                m = cv2.resize(results[0].masks.data[i].cpu().numpy(), (w, h))
                cat_mask = cv2.bitwise_or(cat_mask, (m > 0.5).astype(np.uint8) * 255)

    M = cv2.moments(cat_mask)
    if M["m00"] > 0:
        cat_tracker.update(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), 0)
    else:
        cat_tracker.predict()

    return best_ball, cat_mask, candidates

# ================= MAIN LOOP =================

def main():
    print("Scanning for Sphero...")
    toy = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(scanner.find_toy)
        try: toy = future.result(timeout=10)
        except: pass
    if not toy: print("No Sphero found."); return

    print("Loading AI...")
    model = YOLO("yolov8s-seg.pt") 
    ball_tracker = Tracker()
    cat_tracker = Tracker()

    with SpheroEduAPI(toy) as droid:
        droid.set_main_led(Color(255, 255, 255))
        droid.set_speed(0)

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(CAM_INDEX)
        
        # STATE
        calibration_offset = 0.0
        last_pos = None
        last_command_heading = 0
        is_moving = False

        # CALIBRATION STATE
        calib_state = "INITIAL"  # INITIAL, TEST_MOVE, OPERATIONAL, RECOVERY
        calib_test_phase = 0  # Which test we're on
        calib_start_pos = None
        calib_wait_frames = 0
        calib_samples = []

        # RECOVERY STATE
        ball_lost_frames = 0
        last_command_before_lost = None
        recovery_heading = None

        print("--- CONTINUOUS ADAPTIVE TRACKER ---")
        print(f"MODE: {'TESTING (Hunting PERSON)' if TESTING_MODE else 'PRODUCTION (Hunting CAT)'}")
        print("Starting INITIAL CALIBRATION phase...")
        
        try:
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret: break
                h, w = frame.shape[:2]
                
                # 1. VISION
                best_ball, cat_mask, candidates = process_vision(frame, model, ball_tracker, cat_tracker)
                bx, by = ball_tracker.get_state()
                cx, cy = cat_tracker.get_state()
                ball_found = ball_tracker.frames_lost < 20
                cat_found = cat_tracker.frames_lost < 30

                # 2. VISUALIZATION
                vis = frame.copy()
                overlay = np.zeros_like(vis)
                # Different color for testing vs production
                target_color = (255, 165, 0) if TESTING_MODE else COL_CAT  # Orange for person, Red for cat
                overlay[cat_mask == 255] = target_color
                vis = cv2.addWeighted(vis, 1.0, overlay, 0.5, 0)

                # Draw all candidates (gray circles with scores)
                for cand in candidates:
                    color = (100, 100, 100)  # Gray for rejected
                    if cand.get('final_score'):
                        cv2.circle(vis, (cand['x'], cand['y']), cand['r'], color, 1)
                        cv2.putText(vis, f"{int(cand['final_score'])}", (cand['x']+5, cand['y']-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw selected ball (bright green)
                if best_ball:
                    cv2.circle(vis, (best_ball['x'], best_ball['y']), best_ball['r'], COL_WINNER, 3)
                    cv2.putText(vis, f"{best_ball['src']}", (best_ball['x']+best_ball['r']+5, best_ball['y']),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WINNER, 2)

                # Draw tracker prediction (blue cross)
                cv2.drawMarker(vis, (bx, by), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)

                # Draw recovery arrow if in recovery mode
                if ball_lost_frames > 10 and recovery_heading is not None:
                    # Draw a large orange arrow showing recovery direction
                    rec_rad = math.radians(recovery_heading)
                    rec_x = int(w//2 + 80 * math.sin(rec_rad))
                    rec_y = int(h//2 - 80 * math.cos(rec_rad))
                    cv2.arrowedLine(vis, (w//2, h//2), (rec_x, rec_y), (0, 165, 255), 4)

                # 3. LOGIC
                status = "WAITING..."
                speed = 0
                heading = 0
                should_move = False
                calib_updated = False

                # ========== CALIBRATION STATE MACHINE ==========
                if calib_state == "INITIAL":
                    # Wait for ball detection
                    if ball_found:
                        calib_start_pos = (bx, by)
                        calib_state = "TEST_MOVE"
                        calib_test_phase = 0
                        calib_wait_frames = 0
                        status = "CALIBRATING: Starting test"
                        print(f"[CALIB] Ball found at ({bx}, {by}). Starting test movements...")
                    else:
                        status = "CALIBRATING: Waiting for ball..."

                elif calib_state == "TEST_MOVE":
                    # Perform circular calibration movement
                    TEST_SPEED = 5  # Ultra slow speed for tiny circle
                    CIRCLE_FRAMES = 40  # Fast rotation at slow speed = very tight circle
                    SAMPLE_INTERVAL = 5  # Sample every 5 frames

                    calib_wait_frames += 1

                    # Move in a circle by constantly changing heading
                    if calib_wait_frames <= CIRCLE_FRAMES:
                        # Calculate heading for circular motion (360° over CIRCLE_FRAMES)
                        heading = int((calib_wait_frames * 360.0 / CIRCLE_FRAMES) % 360)
                        speed = TEST_SPEED
                        should_move = True
                        status = f"CALIBRATING: Tiny circle {int(calib_wait_frames/CIRCLE_FRAMES*100)}%"

                        # Sample at intervals while moving
                        if calib_wait_frames % SAMPLE_INTERVAL == 0 and last_pos and ball_found:
                            dist_moved = math.sqrt((bx - last_pos[0])**2 + (by - last_pos[1])**2)

                            if dist_moved > 1:  # Ultra low threshold for tiny movements
                                actual_angle = get_screen_angle(last_pos, (bx, by))
                                commanded_angle = last_command_heading
                                measured_offset = angle_diff(actual_angle, commanded_angle)
                                calib_samples.append(measured_offset)
                                print(f"[CALIB] Sample {len(calib_samples)}: Cmd={int(commanded_angle)}°, Actual={int(actual_angle)}°, Offset={int(measured_offset)}°, Dist={int(dist_moved)}px")

                    else:
                        # Circle complete, stop and calculate
                        should_move = False
                        speed = 0

                        if len(calib_samples) >= 3:  # Need at least 3 samples
                            calibration_offset = sum(calib_samples) / len(calib_samples)
                            calibration_offset = normalize_angle(calibration_offset)
                            print(f"[CALIB] COMPLETE! Final offset: {int(calibration_offset)}° (from {len(calib_samples)} samples)")
                            calib_state = "OPERATIONAL"
                            status = "CALIBRATION COMPLETE"
                        else:
                            print(f"[CALIB] Not enough samples ({len(calib_samples)}), retrying...")
                            calib_wait_frames = 0
                            calib_samples = []

                elif calib_state == "OPERATIONAL":
                    # Normal operation with prediction-based calibration

                    # Prediction-based calibration refinement
                    if not is_moving:
                        pass  # Not moving, no refinement needed
                    elif not last_pos:
                        print(f"[REFINE BLOCKED] is_moving=True but no last_pos")
                    elif not ball_found:
                        print(f"[REFINE BLOCKED] is_moving=True but ball not found")
                    else:
                        # We have all conditions met
                        dist_moved = math.sqrt((bx - last_pos[0])**2 + (by - last_pos[1])**2)

                        if dist_moved > MIN_MOVE_DIST:
                            # Where did we actually go?
                            actual_heading = get_screen_angle(last_pos, (bx, by))

                            # Where did we EXPECT to go? (Command + current offset)
                            expected_heading = normalize_angle(last_command_heading + calibration_offset)

                            # Calculate expected position (for visualization)
                            expected_rad = math.radians(expected_heading)
                            expected_x = int(last_pos[0] + dist_moved * math.sin(expected_rad))
                            expected_y = int(last_pos[1] - dist_moved * math.cos(expected_rad))

                            # Prediction error
                            prediction_error = angle_diff(actual_heading, expected_heading)
                            position_error = math.sqrt((bx - expected_x)**2 + (by - expected_y)**2)

                            # Always print to see what's happening
                            print(f"[REFINE] Cmd:{int(last_command_heading)}° + Offset:{int(calibration_offset)}° = Expected:{int(expected_heading)}° | Actual:{int(actual_heading)}° | AngleErr:{int(prediction_error)}° PosErr:{int(position_error)}px")

                            # VERY aggressive calibration - even tiny errors trigger correction
                            if abs(prediction_error) > 0.5:  # 0.5° error triggers correction (was 1°)
                                old_offset = calibration_offset
                                # More aggressive learning rate for faster convergence
                                calibration_offset += 0.4 * prediction_error  # Increased from 0.25 to 0.4 (40% correction)
                                calibration_offset = normalize_angle(calibration_offset)
                                calib_updated = True
                                print(f"[REFINE APPLY] Offset: {int(old_offset)}° → {int(calibration_offset)}° (Δ{int(0.4 * prediction_error)}°)")

                            # Draw calibration vectors
                            # Draw expected position as a circle
                            cv2.circle(vis, (expected_x, expected_y), 12, (255, 255, 0), 2)  # Yellow circle
                            cv2.circle(vis, (expected_x, expected_y), 3, (255, 255, 0), -1)  # Yellow dot

                            # Draw line from expected to actual (error visualization)
                            cv2.line(vis, (expected_x, expected_y), (bx, by), (255, 0, 255), 2)  # Magenta error line

                            # Expected direction arrow from last pos
                            exp_rad = math.radians(expected_heading)
                            exp_x = int(last_pos[0] + 60 * math.sin(exp_rad))
                            exp_y = int(last_pos[1] - 60 * math.cos(exp_rad))
                            cv2.arrowedLine(vis, last_pos, (exp_x, exp_y), (255, 255, 0), 2)  # Yellow

                            # Actual direction
                            cv2.arrowedLine(vis, last_pos, (bx, by), (0, 255, 255), 2)  # Cyan

                            # Draw position error text
                            mid_x = (expected_x + bx) // 2
                            mid_y = (expected_y + by) // 2
                            cv2.putText(vis, f"{int(position_error)}px", (mid_x + 5, mid_y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        else:
                            print(f"[REFINE SKIP] Movement too small: {int(dist_moved)}px < {MIN_MOVE_DIST}px")

                    if ball_found:
                        # --- A. SAFETY CHECK (BOUNDS) ---
                        if (bx < BORDER_MARGIN or bx > w - BORDER_MARGIN or
                            by < BORDER_MARGIN or by > h - BORDER_MARGIN):
                            status = "SAFETY STOP (BOUNDS)"
                            should_move = False
                            cv2.rectangle(vis, (0,0), (w,h), COL_SAFETY, 5)

                        # --- C. CHASE LOGIC ---
                        elif cat_found:
                            vec_ang = get_screen_angle((bx, by), (cx, cy))
                            dist_px = math.sqrt((bx-cx)**2 + (by-cy)**2)

                            # Visuals
                            # Draw distance zones around ball
                            cv2.circle(vis, (bx, by), int(TOUCH_DISTANCE), (255, 0, 0), 1)  # Blue = stop zone
                            cv2.circle(vis, (bx, by), int(BRAKE_DISTANCE), (0, 255, 255), 1)  # Cyan = brake zone

                            # White line = desired direction (to cat)
                            cv2.line(vis, (bx, by), (cx, cy), (255, 255, 255), 2)

                            # Predicted "Forward" Arrow (what robot thinks is forward)
                            pred_ang = normalize_angle(0 + calibration_offset)
                            rad = math.radians(pred_ang)
                            end_x = int(bx + 40 * math.sin(rad))
                            end_y = int(by - 40 * math.cos(rad))
                            cv2.arrowedLine(vis, (bx, by), (end_x, end_y), COL_PREDICTION, 2)

                            # Draw desired angle arrow (green - where we want to go)
                            des_rad = math.radians(vec_ang)
                            des_x = int(bx + 50 * math.sin(des_rad))
                            des_y = int(by - 50 * math.cos(des_rad))
                            cv2.arrowedLine(vis, (bx, by), (des_x, des_y), (0, 255, 0), 2)

                            # Calculate safe stopping distance
                            touch_gap = dist_px - (ball_tracker.radius + 40)

                            # ALWAYS calculate heading toward target (continuous tracking)
                            heading = normalize_angle(vec_ang - calibration_offset)

                            # Draw command arrow (magenta - what we're actually sending)
                            cmd_rad = math.radians(heading)
                            cmd_x = int(bx + 60 * math.sin(cmd_rad))
                            cmd_y = int(by - 60 * math.cos(cmd_rad))
                            cv2.arrowedLine(vis, (bx, by), (cmd_x, cmd_y), (255, 0, 255), 2)

                            # Aggressive deceleration curve - ALWAYS move toward target
                            if touch_gap > BRAKE_DISTANCE:
                                # Far away: full speed
                                speed = MAX_SPEED
                                speed_zone = "FULL"
                            elif touch_gap > BRAKE_DISTANCE * 0.6:
                                # Medium distance: quadratic deceleration
                                factor = (touch_gap - BRAKE_DISTANCE * 0.6) / (BRAKE_DISTANCE * 0.4)
                                speed = int(MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (factor ** 2))
                                speed_zone = "DECEL"
                            elif touch_gap > TOUCH_DISTANCE:
                                # Close: very slow, proportional to distance
                                factor = max(0.2, touch_gap / (BRAKE_DISTANCE * 0.6))
                                speed = int(MIN_SPEED * factor)
                                speed_zone = "APPROACH"
                            else:
                                # Very close but still track! Use minimal speed to stay with target
                                speed = max(8, int(MIN_SPEED * 0.3))  # Minimum 8 speed to keep tracking
                                speed_zone = "TOUCH"
                                status = "TOUCHING (tracking)"

                            should_move = True
                            if speed_zone != "TOUCH":
                                status = f"CHASING {speed_zone} | Speed:{speed} | Dist:{int(touch_gap)}px"
                            print(f"[CHASE] Zone:{speed_zone} Dist:{int(touch_gap)}px Speed:{speed} | Heading:{int(heading)}°")
                        else:
                            status = f"LOOKING FOR {TARGET_NAME}"
                    else:
                        status = "BALL LOST"

                # ========== BALL LOSS RECOVERY ==========
                if calib_state == "OPERATIONAL":
                    if not ball_found:
                        ball_lost_frames += 1

                        # If just lost, remember what we were doing
                        if ball_lost_frames == 1:
                            last_command_before_lost = last_command_heading
                            # Reverse direction to bring ball back
                            recovery_heading = normalize_angle(last_command_heading + 180)
                            print(f"[RECOVERY] Ball lost! Last cmd:{int(last_command_heading)}°, reversing to {int(recovery_heading)}°")

                        # Execute recovery if ball lost for a few frames
                        if ball_lost_frames > 10:
                            heading = recovery_heading
                            speed = 30  # Slow recovery speed
                            should_move = True
                            status = f"RECOVERY: Reversing {ball_lost_frames}f"
                            print(f"[RECOVERY] Executing reverse heading: {int(recovery_heading)}°")

                        # If lost for too long, restart calibration
                        if ball_lost_frames > 100:
                            print(f"[RECOVERY] Ball lost for {ball_lost_frames} frames. Restarting calibration...")
                            calib_state = "INITIAL"
                            calib_samples = []
                            calibration_offset = 0.0
                            status = "RECALIBRATING"

                    else:
                        # Ball found again after being lost
                        if ball_lost_frames > 10:
                            # Use recovery to calibrate
                            # We sent "recovery_heading" and ball appeared at current position
                            # This tells us the actual direction
                            print(f"[RECOVERY] Ball recovered after {ball_lost_frames} frames!")

                            # If we have a good position history, use it to refine calibration
                            if last_pos:
                                # Ball reappeared - check if recovery worked
                                actual_angle = get_screen_angle(last_pos, (bx, by))
                                expected_angle = normalize_angle(recovery_heading + calibration_offset)
                                error = angle_diff(actual_angle, expected_angle)

                                print(f"[RECOVERY CALIB] Recovery cmd:{int(recovery_heading)}° → Actual:{int(actual_angle)}° | Error:{int(error)}°")

                                # Any error during recovery means offset needs adjustment
                                if abs(error) > 3:  # Lower threshold from 10° to 3°
                                    old_offset = calibration_offset
                                    calibration_offset += 0.5 * error  # Very aggressive correction during recovery
                                    calibration_offset = normalize_angle(calibration_offset)
                                    print(f"[RECOVERY CALIB] Adjusted offset: {int(old_offset)}° → {int(calibration_offset)}°")

                        # Reset recovery state
                        ball_lost_frames = 0
                        last_command_before_lost = None
                        recovery_heading = None

                # 4. ACT
                if should_move:
                    droid.set_heading(int(heading))
                    droid.set_speed(speed)
                    last_command_heading = heading
                    is_moving = True
                else:
                    droid.set_speed(0)
                    is_moving = False

                # 5. DRAW NEXT PREDICTED POSITION (if we just sent a command)
                if should_move and ball_found and heading is not None:
                    # Predict where ball will be next frame based on current command
                    predicted_heading = normalize_angle(heading + calibration_offset)
                    # Estimate distance moved per frame (rough approximation based on speed)
                    est_dist = speed * 0.4  # Rough pixels per frame at given speed
                    pred_rad = math.radians(predicted_heading)
                    next_pred_x = int(bx + est_dist * math.sin(pred_rad))
                    next_pred_y = int(by - est_dist * math.cos(pred_rad))

                    # Draw prediction for next frame
                    cv2.circle(vis, (next_pred_x, next_pred_y), 8, (0, 255, 255), 2)  # Cyan circle
                    cv2.circle(vis, (next_pred_x, next_pred_y), 2, (0, 255, 255), -1)
                    cv2.line(vis, (bx, by), (next_pred_x, next_pred_y), (0, 255, 255), 1)  # Dashed effect
                    cv2.putText(vis, "Next", (next_pred_x + 10, next_pred_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # 6. UPDATE POSITION HISTORY (AFTER sending command)
                # This way, next frame we can compare the movement that resulted from THIS command
                if ball_found:
                    last_pos = (bx, by)

                # HUD
                # Show testing mode indicator
                mode_color = (255, 165, 0) if TESTING_MODE else (0, 255, 0)
                mode_text = f"[TEST MODE: Hunting {TARGET_NAME}]" if TESTING_MODE else f"[CAT MODE]"
                cv2.putText(vis, mode_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

                cv2.putText(vis, status, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_TEXT, 2)

                # Calibration state and offset
                state_color = (255, 165, 0) if calib_state != "OPERATIONAL" else (0, 255, 0) if calib_updated else (200, 200, 200)
                cv2.putText(vis, f"State: {calib_state}", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
                cv2.putText(vis, f"Calib Offset: {int(calibration_offset)}°", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

                if calib_state == "TEST_MOVE":
                    progress = int(calib_wait_frames / 40 * 100) if calib_wait_frames <= 40 else 100
                    cv2.putText(vis, f"Samples: {len(calib_samples)} | Progress: {progress}%", (20, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                    # Draw circle path visualization (very small circle)
                    cv2.circle(vis, (bx, by), 8, (255, 165, 0), 1)

                cv2.putText(vis, f"Ball: {ball_tracker.frames_lost}f lost | {TARGET_NAME}: {cat_tracker.frames_lost}f lost",
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.putText(vis, f"Candidates: {len(candidates)}", (20, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # Show offset and speed info during chase
                if calib_state == "OPERATIONAL" and should_move and 'speed_zone' in locals():
                    zone_colors = {'FULL': (0, 255, 0), 'DECEL': (0, 255, 255), 'APPROACH': (0, 165, 255)}
                    zone_color = zone_colors.get(speed_zone, (255, 255, 255))
                    cv2.putText(vis, f"Zone: {speed_zone} | Speed: {speed}", (20, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                    cv2.putText(vis, f"Offset: {int(calibration_offset)}°", (20, 235),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Show recovery status
                if ball_lost_frames > 0:
                    recovery_color = (0, 165, 255) if ball_lost_frames > 10 else (255, 255, 0)
                    cv2.putText(vis, f"Ball Lost: {ball_lost_frames}f", (20, 265),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, recovery_color, 2)
                    if recovery_heading is not None:
                        cv2.putText(vis, f"Recovery Heading: {int(recovery_heading)}°", (20, 290),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, recovery_color, 1)

                # Legend (only show in OPERATIONAL mode)
                if calib_state == "OPERATIONAL":
                    legend_y = h - 240
                    cv2.putText(vis, "ZONES & PREDICTION:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(vis, "Blue circle = Stop zone (80px)", (20, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    cv2.putText(vis, "Cyan circle (big) = Brake zone", (20, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.putText(vis, "Cyan circle (small) = Next predicted", (20, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.putText(vis, "Yellow circle = Last expected pos", (20, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(vis, "Magenta line = Prediction error", (20, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    cv2.putText(vis, "Green arrow = Desired direction", (20, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(vis, "Magenta arrow = Sent command", (20, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    cv2.putText(vis, "Red arrow = Robot's 'forward'", (20, legend_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_PREDICTION, 1)
                    cv2.putText(vis, "Yellow arrow = Expected heading", (20, legend_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(vis, "Cyan arrow = Actual heading", (20, legend_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.putText(vis, "Orange arrow = Recovery", (20, legend_y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                cv2.imshow("Continuous Tracker", vis)

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