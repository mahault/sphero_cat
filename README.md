# Sphero Cat Toy

An autonomous cat toy using a Sphero robot ball that tracks and chases cats, dogs, or people using computer vision, YOLOv8, and small **active inference / POMDP** modules.

---

## 📁 Project Layout

From the repo root (e.g. `.../sphero`):

```text
sphero/
  models/                 # YOLO weights (e.g. yolov8s-seg.pt)
  logs/                   # CSV logs (auto-created)
  scripts/
    sphero_track_test_v2.py   # Main control loop (vision + control + POMDP hooks)
    run_sphero_calib.py       # Entry script to run the toy
    calibration_pomdp.py      # Calibration-confidence POMDP (Active Inference)
    visibility_pomdp.py       # Visibility / occlusion POMDP (Active Inference)
  sphero-experiments/     # Older prototypes, test scripts, archives

``` 

## 🎯 Current Working System
**Entry Script**
scripts/run_sphero_calib.py

Tiny wrapper that imports and runs main() from:

- scripts/sphero_track_test_v2.py

Running this is the normal way to start the toy.

## 🧠 Core Script: sphero_track_test_v2.py

This is the main, production-style script. It combines:

- robust ball detection and tracking,

- YOLOv8 segmentation for targets,

- automatic calibration and runtime refinement,

- safety-aware navigation,

- two small POMDPs implemented with pymdp:

- Calibration Confidence POMDP (calibration_pomdp.py)

- Visibility / Occlusion POMDP (visibility_pomdp.py)

### What It Does

The script creates an autonomous cat toy that:

1. **Detects the Sphero ball** using a multi-method fusion approach:
   - Hough circle detection for geometric precision
   - YOLO object detection (sports ball class) for robust recognition
   - Brightness-based blob detection for LED tracking
   - Kalman filtering for smooth position tracking and velocity estimation

2. **Tracks targets** using YOLOv8 segmentation:
   - Primary targets: Cats and dogs
   - Secondary targets: People (when no pets are detected)
   - If no target is visible, behaviour is governed by the Visibility POMDP:
         - either WAIT in place (if it believes the target is briefly occluded),
         - or TRACK (move towards the arena center) if it believes the target is truly gone.

3. **Navigates intelligently**:
   - Full automatic calibration:
      - Moves Sphero in four cardinal directions,
      - Estimates a heading offset between Sphero’s “0°” and camera frame,
      - Estimates movement scale in px/(speed·s).
   - Runtime calibration refinement:
      - During operation, compares expected vs actual motion,
      - Gradually adjusts the heading offset to reduce systematic drift.
   - Predictive boundary avoidance:
      - Uses current heading and command velocity to predict future position,
      - Engages a safety mode if it predicts leaving the safe border margin.
   - Stuck detection and escape:
      - Monitors commanded motion vs actual displacement,
      - If stuck, executes an escape manoeuvre towards the arena center.
   - Distance-first speed profile:
      - High speed when far away,
      - Ramps down as it approaches the target,
      - Uses velocity-aware braking to avoid overshooting.

4. **Safety features**:
   - Border margin that defines an inner “safe box” (keeps Sphero in camera view),
   - Predictive “will exit soon?” check based on current motion,
   - Automatic return-to-center behaviour near boundaries,
   - Stuck frames counter with explicit “escape to center” behaviour,
   - Calibration POMDP can reduce the maximum speed (CAUTION mode) when it infers calibration drift or misalignment.

5. **Comprehensive logging**:
   - Writes timestamped CSV logs under logs/:
      - Positions (estimated & measured),
      - Commands (headings, speeds),
      - Distances and prediction errors,
      - Stuck frames, safety activations,
      - POMDP beliefs and actions:
         - Calibration-belief over OK/DRIFT/BAD,
         - Calibration observation bins & actions (NORMAL vs CAUTION),
         - Visibility-belief over VISIBLE/OCCLUDED/GONE,
         - Visibility observation bins & actions (TRACK vs WAIT).

## 🧩 Active Inference / POMDP Modules
1. ** Calibration Confidence POMDP (scripts/calibration_pomdp.py) **

This module uses pymdp to maintain a small generative model over how well-calibrated the system is.

- Hidden states:
   - OK – calibration is likely good,
   - DRIFT – mild drift or rising prediction errors,
   - BAD – highly likely to be miscalibrated.

- Observations (derived from prediction_error_px, stuck_frames, safety activity):
   - ERR_LOW – small error,
   - ERR_MED – moderate error,
   - ERR_HIGH – large or persistent error.

- Actions:
   - NORMAL – full MAX_SPEED and standard border behavior,
   - CAUTION – reduced effective_max_speed (currently ~70% of MAX_SPEED).

The agent:
- infers belief over OK/DRIFT/BAD each frame,
- chooses a meta-action (NORMAL vs CAUTION),
- the main script uses this to scale the effective max speed for that frame.

In the logs you’ll see lines like:

[POMDP] loop=100 obs=ERR_MED action=CAUTION belief=[OK:0.71, DRIFT:0.26, BAD:0.03] eff_max_speed=21


indicating that the POMDP has inferred (mostly) DRIFT and is temporarily playing it safe.


2. ** Visibility / Occlusion POMDP (scripts/visibility_pomdp.py) **

This module reasons about whether the target is truly gone or just briefly occluded (e.g. behind furniture, out of YOLO’s view).

- Hidden states:
   - VISIBLE – target clearly in view,
   - OCCLUDED – probably nearby but temporarily missing,
   - GONE – likely left the scene.

- Observations (from target_found + how long it's been lost):
   - SEE_TARGET – YOLO segmentation sees a target,
   - RECENTLY_LOST – target not seen, but only for a short period (a few dozen frames),
   - LONG_LOST – target not seen for a longer duration.

- Actions:
   - TRACK – behave as before: use center as goal when no target is visible,
   - WAIT – stay in place (don’t move) while waiting for the target to re-appear.

The main loop uses this as:
- If target is visible → chase as usual.
- If target is not visible:
   - If visibility action = WAIT → stop and wait.
   - If visibility action = TRACK → move toward the arena center.

Console logs look like:

[VIS-POMDP] loop=50  obs=RECENTLY_LOST action=WAIT  belief=[VIS:0.20, OCC:0.70, GONE:0.10]
[VIS-POMDP] loop=100 obs=LONG_LOST     action=TRACK belief=[VIS:0.00, OCC:0.02, GONE:0.98]


So the agent:

- pauses when it thinks the target is locally occluded,
- resumes “go to center” when it believes the target is truly gone.

### How to Run

#### Prerequisites

```bash
pip install spherov2 opencv-python numpy ultralytics pymdp
```

You'll also need:
- A Sphero robot (BOLT, SPRK+, or compatible model)
- A webcam with a clear overhead view of the play area
- YOLOv8 model weights (`yolov8s-seg.pt` - will auto-download on first run)
   - If not present, the script will attempt to download yolov8s-seg.pt into the current directory.

#### Setup

1. **Position your camera**: Mount it overhead with a clear view of the play area
2. **Set camera index**: Adjust CAM_INDEX in scripts/sphero_track_test_v2.py if your camera is not at index 1.
3. **Power on your Sphero**: Make sure it's charged and in range

#### Running the Script

```bash
cd scripts
python run_sphero_calib.py
```

The script will:

1. Scan for and connect to your Sphero,
2. Open the camera feed and report resolution,
3. Reset Sphero’s 0° heading,
4. Show an alignment screen with a green arrow pointing “up”:
   - Rotate the Sphero so its physical forward direction (0°) matches “up” in the camera,
   - Press SPACE or ENTER to confirm.
5. Run automatic calibration:
   - Sphero moves in 4 cardinal directions,
   - The system estimates orientation offset and movement scale.
6. Start the main loop with:
   - robust ball tracking,
   - cat/dog/person detection,
   - safety checks,
   - ** Calibration POMDP (NORMAL / CAUTION), **
   - ** Visibility POMDP (WAIT / TRACK). **

### Controls

- Press Q in the OpenCV window to quit gracefully,
- Press Ctrl+C in the terminal for an emergency stop if needed.

#### Configuration

Key parameters you can adjust at the top of the script:

```python
CAM_INDEX = 1                # Your camera device index
MAX_SPEED = 30               # Maximum Sphero speed (0-255)
MIN_SPEED = 5                # Minimum movement speed
TARGET_REACHED_PIX = 35      # How close to get to target (pixels)
BORDER_MARGIN = 100          # Safety margin from edges (pixels)
PREDICT_HORIZON_SEC = 0.4    # How far ahead to predict boundary exits
```

In the POMDP modules:
- calibration_pomdp.py:
   - Error bin thresholds,
   - Transition model B (how fast OK → DRIFT / DRIFT → BAD),
   - Preferences C over error bins.
- visibility_pomdp.py:
   - Frame thresholds for “recently lost” vs “long lost”,
   - Transition model B (how quickly OCCLUDED resolves to VISIBLE vs GONE),
   - Preferences C over observations.

### Output

The script creates timestamped CSV log files:
```
logs/
   sphero_log_2025-11-28_14-30-45.csv
```

Each log contains:
- Timestamp, loop index,
- Estimated & measured positions,
- Goal position & goal label (CENTER, PERSON, CAT/DOG),
- Calibration offset and movement scale,
- Command heading & speed,
- Distances to goal and center,
- Prediction error, stuck frames, safety flags,
- Measurement & estimated velocities,
- Calibration POMDP beliefs + obs + action,
- Visibility POMDP beliefs + obs + action.


## 🐱 Best Practices
- Use even lighting; avoid very dark or saturated scenes.
- A darker floor makes the bright Sphero LED easier to detect.
- Keep the camera stable and high enough (e.g. 1.5–2 m) to see the whole play area.
- Give Sphero a clear 2x2 m area initially.
- Let the calibration routine complete without obstruction.
- If behaviour feels too aggressive, lower MAX_SPEED. If too timid, raise it slightly.

## 📁 Project Evolution

### Early Versions

**`run_sphero.py`**
- Basic Sphero control
- Simple circular motion patterns
- Obstacle avoidance placeholder (no vision)
- Foundation for motor control

**`run_sphero_grid.py`** & **`run_sphero_track.py`**
- Early experiments with waypoint navigation
- Grid-based movement patterns
- No computer vision integration

### POMDP and Bayesian Approaches

**`run_sphero_bayesian.py`**
- Attempted Bayesian position estimation
- Probabilistic tracking experiments
- Explored uncertainty handling

**`sphero_webcam_pomdp.py`**
- POMDP (Partially Observable Markov Decision Process) framework
- Used pymdp for decision-making under uncertainty
- Complex state estimation
- *Abandoned due to complexity and performance issues*

### Cat Tracking Iterations

**`sphero_track_cat.py`**
- First version with actual cat tracking using YOLO
- Basic POMDP integration
- Combined vision with decision theory
- Foundation for autonomous behavior

**`Sphero_track_cat_grid.py`**
- Added grid-based navigation to cat tracking
- Quantized movement space
- Hybrid approach between POMDP and direct control
- Current git HEAD shows this "works but needs tweaking"

### Test Versions (Leading to v2)

**`spher_tratck_test.py`** (Selected in editor)
- Robust multi-method ball detection (Hough + YOLO + brightness)
- Kalman filtering for ball tracking
- Introduced calibration system
- Early safety features
- Uses vision latency compensation

**`sphero_track_test.py`**
- Similar to above but intermediate iteration
- Testing different parameter sets

**`sphero_track_test_v2.py`** ✅ **WORKING VERSION**
- Removed vision latency extrapolation (caused instability)
- Simplified to measurement smoothing with Kalman filtering
- Improved calibration (both auto and runtime refinement)
- Better safety system with predictive boundary checking
- Distance-first speed profile with velocity-aware braking
- Stuck detection and escape behavior
- Comprehensive logging
- **Most stable and reliable version**

## 🔧 Key Technical Improvements in v2

1. **Simplified State Estimation**: Removed latency compensation extrapolation that caused oscillations; now uses simple exponential smoothing with Kalman position estimates

2. **Robust Ball Detection**: Fuses three detection methods (Hough circles, YOLO sports ball, brightness blobs) with confidence scoring

3. **Automatic Calibration**: Moves in cardinal directions to self-calibrate heading offset and movement scale

4. **Predictive Safety**: Uses commanded velocity model to predict boundary exits before they happen

5. **Intelligent Speed Control**: Distance-based speed ramping with velocity-aware braking prevents overshooting

6. **Runtime Adaptation**: Continuously refines calibration offset during operation based on observed movement

## 🐱 Tips for Best Results

1. **Lighting**: Ensure good, even lighting so the white Sphero LED is visible
2. **Contrast**: Use a dark floor for better ball detection
3. **Camera Height**: Mount camera 1.5-2 meters high for best overhead view
4. **Play Area**: Clear, flat space about 2x2 meters minimum
5. **Calibration**: Do the initial alignment carefully for best tracking
6. **Speed Tuning**: If Sphero is too aggressive, reduce `MAX_SPEED`; if too timid, increase it

## 📊 Troubleshooting

**Sphero not found:**
- Check Bluetooth is enabled
- Make sure Sphero is charged and awake
- Try moving it or putting it in the charger briefly to wake it

**Ball not detected:**
- Increase brightness in the room
- Adjust `BRIGHTNESS_THRESHOLD` (lower = more sensitive)
- Check that Sphero LED is set to white
- Ensure camera has clear view with no obstructions

**Poor tracking:**
- Re-run calibration with better alignment
- Adjust `BORDER_MARGIN` if Sphero escapes the view
- Check camera is stable and not moving
- Reduce `MAX_SPEED` for more controlled movement

**Sphero goes off-screen:**
- Increase `BORDER_MARGIN`
- Decrease `PREDICT_HORIZON_SEC` for earlier boundary detection
- Ensure camera covers the full play area

## 📝 License

This is a personal project for cat entertainment and robotics learning.

## 🙏 Acknowledgments

- Uses [spherov2](https://github.com/artificial-intelligence-class/spherov2.py) for Sphero control
- Uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- OpenCV for computer vision processing
